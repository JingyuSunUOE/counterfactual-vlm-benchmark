"""
Evaluation pipeline for the VLM Bias benchmark.
"""

import argparse
import base64
import json
import os
import re
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

EVAL_CODE_DIR = Path(__file__).resolve().parents[1]
if str(EVAL_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(EVAL_CODE_DIR))

from server_vram_planner import add_server_preflight_arguments, plan_pair_from_args, plan_server_vram, preflight_or_warn
from provider_cache import (
    add_anthropic_cache_control,
    add_provider_cache_arguments,
    build_provider_cache_key,
    extract_cache_usage,
    openai_cache_kwargs,
    provider_cache_state,
    summarize_usage_dict,
)
from structured_outputs import (
    JUDGE_LABEL_SCHEMA_ID,
    MC_ANSWER_SCHEMA_ID,
    OPEN_ANSWER_SCHEMA_ID,
    YES_NO_ANSWER_SCHEMA_ID,
    add_structured_output_arguments,
    format_structured_fallback_reason,
    gemini_rest_generation_config,
    looks_like_structured_output_unsupported,
    normalize_structured_fallback,
    openai_chat_response_format,
    parse_structured_judge_response,
    parse_structured_mc_answer,
    parse_structured_open_answer,
    parse_structured_yes_no_answer,
    strip_thinking_blocks,
    structured_record_fields,
)
from mime_utils import detect_mime_type

from eval_questions import (
    BOTH_RAW_PAIR_DIFFERENCE_PREFIX,
    DISPLAY_NAME_MAP,
    EVALUATION_TARGETS,
    EVAL_FASHION,
    EVAL_INDUSTRY,
    QUESTION_DESIGN_VERSION,
    both_raw_policy_fields,
    build_question_payload,
    get_primed_question,
    get_questions,
)


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]

FASHION_CF_DIR = REPO_ROOT / "cf_dataset" / "fashion_cf"
INDUSTRY_CF_DIR = REPO_ROOT / "cf_dataset" / "industry_cf"
FASHION_DATASET_DIR = REPO_ROOT / "dataset" / "fashion_dataset"
INDUSTRY_DATASET_DIR = REPO_ROOT / "dataset" / "industry_dataset"
RESULTS_ROOT = REPO_ROOT / "eval_results" / "fashion_industry"
RAW_RESULTS_DIR = RESULTS_ROOT / "raw_runs"
METADATA_DIR = RESULTS_ROOT / "metadata"

DASHSCOPE_BASE_URL = "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"
DEFAULT_SERVER_URL = "http://localhost:8000"
QUESTION_TYPES = ("Q1", "Q2", "Q3", "Q_MC")
TOOL_CONDITIONS = ("raw", "bbox", "crop", "zoom_panel")
EVIDENCE_QC_FILTERS = ("pass", "pass_review", "all")
MISSING_EVIDENCE_POLICIES = ("skip", "error", "raw_fallback")
EVIDENCE_SCHEMA_VERSION = "fashion_industry_sam3_object_evidence_v1"
TOOL_VIEW_PREFIX = (
    "Additional images, if provided, show alternative visual views of the same scene. "
    "Use the provided visual information to answer the question."
)
TOOL_VIEW_BOTH_PREFIX = (
    "If two image groups are provided, answer about the second image group. "
    "Additional images in each group show alternative visual views of the same scene."
)
SUCCESS_LABELS = {"correct", "biased", "other"}
ERROR_KINDS = {
    "quota_exceeded",
    "rate_limit",
    "auth_error",
    "server_error",
    "network_error",
    "timeout",
    "judge_error",
    "unknown_error",
}
RETRYABLE_ERROR_KINDS = {"rate_limit", "server_error", "network_error", "timeout"}
RESUME_MATCH_IGNORED_KEYS = {
    "max_retries",
    "retry_base_seconds",
    "retry_max_seconds",
    "stop_on_quota",
    "max_consecutive_errors",
    "resume",
    "output_root",
    "server_vram_plan",
    "judge_server_vram_plan",
}

MODEL_SETS = {
    "closed": ["gemini-3.1-pro-preview", "claude-sonnet-4", "gpt-5"],
    "open": [
        "qwen3-vl-8b",
        "qwen3-vl-32b",
        "internvl3.5-8b",
        "internvl3.5-38b-hf",
        "gemma-4-e4b-it",
        "gemma-4-31b-it",
        "qwen2.5-vl-72b",
    ],
}

JUDGE_MODELS = {"closed": "gpt-4o-mini-2024-07-18", "open": "gpt-4o-mini-2024-07-18"}

MODEL_IDS = {
    "gemini-2.5-pro": "gemini-2.5-pro",
    "gemini-3.1-pro-preview": "gemini-3.1-pro-preview",
    "claude-sonnet-3.7": "claude-3-7-sonnet-20250219",
    "claude-sonnet-4": "claude-sonnet-4-20250514",
    "gpt-4.1": "gpt-4.1-2025-04-14",
    "gpt-5": "gpt-5",
    "qwen3-vl-32b": "Qwen/Qwen3-VL-32B-Instruct",
    "qwen3-vl-8b": "Qwen/Qwen3-VL-8B-Instruct",
    "qwen2.5-vl-32b": "Qwen/Qwen2.5-VL-32B-Instruct",
    "qwen2.5-vl-72b": "Qwen/Qwen2.5-VL-72B-Instruct",
    "internvl3.5-8b": "OpenGVLab/InternVL3_5-8B",
    "internvl3_5-8b": "OpenGVLab/InternVL3_5-8B",
    "internvl3.5-38b-hf": "OpenGVLab/InternVL3_5-38B-HF",
    "internvl3_5-38b-hf": "OpenGVLab/InternVL3_5-38B-HF",
    "gemma-4-e4b-it": "google/gemma-4-E4B-it",
    "gemma-4-31b-it": "google/gemma-4-31B-it",
    "gpt-4o-mini": "gpt-4o-mini-2024-07-18",
    "qwen3-8b": "qwen3-8b",
}

MODEL_GROUPS = {
    **{name: "closed" for name in MODEL_SETS["closed"]},
    **{name: "open" for name in MODEL_SETS["open"]},
}


load_dotenv(REPO_ROOT / ".env", override=True)

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "") or os.environ.get("GOOGLE_API_KEY", "")
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
QWEN_API_KEY = os.environ.get("QWEN_API_KEY", "")

LAST_PROVIDER_CACHE_INFO = {
    "provider_cache_enabled": False,
    "provider_cache_key": None,
    "provider_cache_reason": None,
    "provider_cache_usage": {},
}
LAST_STRUCTURED_OUTPUT_INFO = {
    "structured_output_used": False,
    "structured_output_schema": None,
    "structured_output_fallback_reason": None,
}


def set_last_provider_cache_info(*, enabled=False, cache_key=None, reason=None, usage=None):
    LAST_PROVIDER_CACHE_INFO.update(
        {
            "provider_cache_enabled": bool(enabled),
            "provider_cache_key": cache_key if enabled else None,
            "provider_cache_reason": reason,
            "provider_cache_usage": usage or {},
        }
    )


def get_last_provider_cache_info():
    return dict(LAST_PROVIDER_CACHE_INFO)


def set_last_structured_output_info(*, used=False, schema=None, fallback_reason=None):
    LAST_STRUCTURED_OUTPUT_INFO.update(
        {
            "structured_output_used": bool(used),
            "structured_output_schema": schema,
            "structured_output_fallback_reason": fallback_reason,
        }
    )


def get_last_structured_output_info():
    return dict(LAST_STRUCTURED_OUTPUT_INFO)


def classify_error(error, status_code=None, default="unknown_error"):
    response = getattr(error, "response", None)
    if status_code is None:
        status_code = getattr(error, "status_code", None) or getattr(error, "status", None)
    if status_code is None and response is not None:
        status_code = getattr(response, "status_code", None)
    text = str(error or "")
    if response is not None:
        try:
            text += f" {getattr(response, 'text', '')}"
        except Exception:
            pass
    text_lower = text.lower()
    if status_code in {401, 403} or any(
        token in text_lower
        for token in (
            "invalid_api_key",
            "invalid api key",
            "authentication",
            "unauthorized",
            "permission_denied",
            "permission denied",
            "forbidden",
        )
    ):
        return "auth_error"
    if any(
        token in text_lower
        for token in (
            "insufficient_quota",
            "exceeded your current quota",
            "quota exceeded",
            "resource_exhausted",
            "billing",
            "balance",
            "credit",
        )
    ):
        return "quota_exceeded"
    if status_code == 429 or any(token in text_lower for token in ("rate limit", "rate_limit", "too many requests")):
        return "rate_limit"
    if status_code is not None and 500 <= int(status_code) <= 599:
        return "server_error"
    if any(token in text_lower for token in ("timeout", "timed out", "read timed out")):
        return "timeout"
    if any(
        token in text_lower
        for token in (
            "connection",
            "network",
            "chunkedencodingerror",
            "connectionerror",
            "connection reset",
            "name resolution",
        )
    ):
        return "network_error"
    return default if default in ERROR_KINDS else "unknown_error"


def is_fatal_error_kind(error_kind, stop_on_quota=True):
    if error_kind == "quota_exceeded":
        return stop_on_quota
    return error_kind == "auth_error"


def retry_delay_seconds(attempt_index, base_seconds, max_seconds):
    return min(max_seconds, max(0.0, base_seconds) * (2 ** max(0, attempt_index - 1)))


def query_with_retries(callable_fn, *, max_retries, retry_base_seconds, retry_max_seconds, stop_on_quota):
    max_attempts = max(1, int(max_retries))
    retry_count = 0
    last_response, last_error = None, None
    last_kind = None
    last_fatal = False
    for attempt in range(1, max_attempts + 1):
        try:
            response, error = callable_fn()
        except Exception as exc:
            response, error = None, str(exc)
        if not error:
            return response, None, retry_count, None, False
        last_response, last_error = response, error
        last_kind = classify_error(error)
        last_fatal = is_fatal_error_kind(last_kind, stop_on_quota=stop_on_quota)
        if last_fatal or last_kind not in RETRYABLE_ERROR_KINDS or attempt == max_attempts:
            return last_response, last_error, retry_count, last_kind, last_fatal
        retry_count += 1
        time.sleep(
            retry_delay_seconds(
                attempt_index=attempt,
                base_seconds=retry_base_seconds,
                max_seconds=retry_max_seconds,
            )
        )
    return last_response, last_error, retry_count, last_kind or "unknown_error", last_fatal


def task_signature(model_name, input_mode, task):
    return (
        model_name,
        input_mode,
        task.get("tool_condition", "raw"),
        task.get("key"),
        task.get("mod_id"),
        task.get("question_type"),
        task.get("basename"),
    )


def record_signature(record):
    basename = record.get("basename")
    if not basename:
        image_paths = record.get("image_paths") or []
        basename = Path(image_paths[-1]).name if image_paths else None
    return (
        record.get("model"),
        record.get("input_mode"),
        record.get("tool_condition", "raw"),
        record.get("key"),
        record.get("mod_id"),
        record.get("question_type"),
        basename,
    )


def record_is_success(record):
    if record.get("error"):
        return False
    return record.get("judge_label") in SUCCESS_LABELS


def record_is_error(record):
    return bool(record.get("error")) or record.get("judge_label") in {"query_error", "judge_error"}


def write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def load_jsonl(path):
    rows = []
    if not path.exists():
        return rows
    with open(path) as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def load_checkpoint(run_dir):
    path = run_dir / "checkpoint.json"
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except Exception:
        return {}


def completed_success_count(records):
    return len({record_signature(record) for record in records if record_is_success(record)})


def write_checkpoint(
    *,
    run_dir,
    run_id,
    run_status,
    total_tasks,
    records,
    skipped_resume_count,
    pending_count,
    last_task_signature=None,
    last_error_kind=None,
    last_error_message=None,
    evidence_filter_stats=None,
):
    payload = {
        "run_id": run_id,
        "run_status": run_status,
        "total_tasks": total_tasks,
        "completed_success_count": completed_success_count(records),
        "error_count": sum(1 for record in records if record_is_error(record)),
        "skipped_resume_count": skipped_resume_count,
        "pending_count": max(0, pending_count),
        "last_task_signature": list(last_task_signature) if isinstance(last_task_signature, tuple) else last_task_signature,
        "last_error_kind": last_error_kind,
        "last_error_message": last_error_message,
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    if evidence_filter_stats is not None:
        payload["evidence_filter"] = evidence_filter_stats
    write_json(run_dir / "checkpoint.json", payload)


def run_config_matches(config, expected):
    for key, expected_value in expected.items():
        if key in RESUME_MATCH_IGNORED_KEYS:
            continue
        actual_value = config.get(key)
        if isinstance(expected_value, list):
            if list(actual_value or []) != expected_value:
                return False
        elif expected_value is None:
            if actual_value not in (None, ""):
                return False
        elif key == "tool_condition" and actual_value in (None, ""):
            if expected_value != "raw":
                return False
        elif actual_value != expected_value:
            return False
    return True


def discover_fashion_run_dirs(root):
    if not root.exists():
        return []
    if root.is_file():
        return [root.parent]
    if (root / "records.jsonl").exists() or any(root.glob("*.jsonl")):
        return [root]
    return sorted([path for path in root.iterdir() if path.is_dir()])


def resolve_resume_run_dir(output_root, resume, expected_config):
    if resume == "off" or not output_root.exists():
        return None
    candidates = []
    for run_dir in discover_fashion_run_dirs(output_root):
        if run_dir == output_root:
            continue
        if (run_dir / "run_config.json").exists() or (run_dir / "checkpoint.json").exists() or any(run_dir.glob("*.jsonl")):
            candidates.append(run_dir)
    if not candidates:
        return None
    candidates = sorted(candidates, key=lambda path: (path.stat().st_mtime, path.name), reverse=True)
    if resume == "latest":
        return candidates[0]
    for run_dir in candidates:
        config_path = run_dir / "run_config.json"
        if not config_path.exists():
            continue
        try:
            config = json.loads(config_path.read_text())
        except Exception:
            continue
        if not run_config_matches(config, expected_config):
            continue
        checkpoint = load_checkpoint(run_dir)
        if checkpoint.get("run_status") == "completed":
            continue
        return run_dir
    return None


def encode_image(path: str) -> str:
    with open(path, "rb") as handle:
        return base64.b64encode(handle.read()).decode()


def guess_mime_type(path: str) -> str:
    return detect_mime_type(path)


def make_data_url(path: str) -> str:
    return f"data:{guess_mime_type(path)};base64,{encode_image(path)}"


def _content_to_text(content):
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        texts = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                texts.append(item.get("text", ""))
        return "\n".join(texts).strip()
    return str(content)


def _openai_client(api_key, base_url=None):
    from openai import OpenAI

    kwargs = {"api_key": api_key}
    if base_url:
        kwargs["base_url"] = base_url
    return OpenAI(**kwargs)


def _normalize_openai_compatible_url(url):
    normalized_url = (url or DEFAULT_SERVER_URL).rstrip("/")
    if not normalized_url.endswith("/v1"):
        normalized_url = f"{normalized_url}/v1"
    return normalized_url


def _openai_chat_token_limit_kwargs(*, model: str, max_tokens: int, base_url=None) -> dict:
    if base_url is None and str(model).lower().startswith("gpt-5"):
        return {"max_completion_tokens": max_tokens}
    return {"max_tokens": max_tokens}


def _openai_chat_temperature_kwargs(*, model: str, temperature: float, base_url=None) -> dict:
    if base_url is None and str(model).lower().startswith("gpt-5"):
        return {}
    return {"temperature": temperature}


def _query_openai_compatible_vision(
    image_paths,
    question,
    model,
    api_key,
    base_url=None,
    max_tokens=1024,
    provider_cache="off",
    prompt_cache_retention="in_memory",
    cache_key=None,
    cache_usage_log=True,
    structured_output_mode="off",
    structured_output_schema=None,
    structured_output_fallback="retry_plain",
):
    set_last_structured_output_info(used=False, schema=structured_output_schema, fallback_reason=None)
    client = _openai_client(api_key=api_key, base_url=base_url)
    content = []
    for image_path in image_paths:
        content.append({"type": "image_url", "image_url": {"url": make_data_url(image_path)}})
    content.append({"type": "text", "text": question})
    if base_url:
        cache_state = {"enabled": False, "reason": "unsupported_openai_compatible_endpoint"}
    else:
        cache_state = provider_cache_state(
            provider="openai",
            provider_cache=provider_cache if image_paths else "off",
            model_id=model,
            prompt_cache_retention=prompt_cache_retention,
        )
    set_last_provider_cache_info(enabled=cache_state.get("enabled"), cache_key=cache_key, reason=cache_state.get("reason"))
    request_kwargs = {
        "model": model,
        "messages": [{"role": "user", "content": content}],
    }
    request_kwargs.update(_openai_chat_token_limit_kwargs(model=model, max_tokens=max_tokens, base_url=base_url))
    request_kwargs.update(_openai_chat_temperature_kwargs(model=model, temperature=0, base_url=base_url))
    request_kwargs.update(
        openai_cache_kwargs(
            cache_state=cache_state,
            cache_key=cache_key,
            model_id=model,
            prompt_cache_retention=prompt_cache_retention,
        )
    )
    if structured_output_mode != "off" and structured_output_schema:
        request_kwargs["response_format"] = openai_chat_response_format(structured_output_schema)
    response = _chat_create_with_structured_fallback(
        client=client,
        request_kwargs=request_kwargs,
        mode=structured_output_mode,
        schema_id=structured_output_schema,
        fallback=structured_output_fallback,
    )
    if cache_usage_log:
        set_last_provider_cache_info(
            enabled=cache_state.get("enabled"),
            cache_key=cache_key,
            reason=cache_state.get("reason"),
            usage=extract_cache_usage("openai", response),
        )
    return _content_to_text(response.choices[0].message.content)


def _query_openai_compatible_text(
    system_prompt,
    user_prompt,
    model,
    api_key,
    base_url=None,
    max_tokens=512,
    structured_output_mode="off",
    structured_output_schema=None,
    structured_output_fallback="retry_plain",
):
    set_last_structured_output_info(used=False, schema=structured_output_schema, fallback_reason=None)
    client = _openai_client(api_key=api_key, base_url=base_url)
    request_kwargs = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    }
    request_kwargs.update(_openai_chat_token_limit_kwargs(model=model, max_tokens=max_tokens, base_url=base_url))
    request_kwargs.update(_openai_chat_temperature_kwargs(model=model, temperature=0, base_url=base_url))
    if structured_output_mode != "off" and structured_output_schema:
        request_kwargs["response_format"] = openai_chat_response_format(structured_output_schema)
    response = _chat_create_with_structured_fallback(
        client=client,
        request_kwargs=request_kwargs,
        mode=structured_output_mode,
        schema_id=structured_output_schema,
        fallback=structured_output_fallback,
    )
    return _content_to_text(response.choices[0].message.content)


def _chat_create_with_structured_fallback(client, request_kwargs, mode, schema_id, fallback):
    if mode == "off" or not schema_id:
        return client.chat.completions.create(**_strip_chat_structured_output(request_kwargs))
    try:
        response = client.chat.completions.create(**request_kwargs)
        set_last_structured_output_info(used=True, schema=schema_id, fallback_reason=None)
        return response
    except Exception as exc:
        fallback_mode = normalize_structured_fallback(mode, fallback)
        if fallback_mode == "retry_plain" and looks_like_structured_output_unsupported(exc):
            reason = format_structured_fallback_reason(exc)
            set_last_structured_output_info(used=False, schema=schema_id, fallback_reason=reason)
            return client.chat.completions.create(**_strip_chat_structured_output(request_kwargs))
        raise


def _strip_chat_structured_output(request_kwargs):
    updated = dict(request_kwargs)
    updated.pop("response_format", None)
    return updated


def query_gemini(
    image_paths: list[str],
    question: str,
    model: str = "gemini-2.5-pro",
    provider_cache: str = "off",
    prompt_cache_retention: str = "in_memory",
    cache_key: str | None = None,
    cache_usage_log: bool = True,
    max_tokens: int = 1024,
    reasoning_effort: str = "off",
    structured_output_mode: str = "off",
    structured_output_schema: str | None = None,
    structured_output_fallback: str = "retry_plain",
):
    import requests

    cache_state = provider_cache_state(
        provider="google",
        provider_cache=provider_cache if image_paths else "off",
        model_id=model,
        prompt_cache_retention=prompt_cache_retention,
    )
    set_last_provider_cache_info(enabled=cache_state.get("enabled"), cache_key=cache_key, reason=cache_state.get("reason"))
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={GEMINI_API_KEY}"
    parts = []
    for image_path in image_paths:
        parts.append(
            {
                "inlineData": {
                    "mimeType": guess_mime_type(image_path),
                    "data": encode_image(image_path),
                }
            }
        )
    parts.append({"text": question})
    generation_config = {"temperature": 0, "maxOutputTokens": max_tokens}
    if model.startswith("gemini-2.5") and reasoning_effort != "provider_default":
        generation_config["thinkingConfig"] = {"thinkingBudget": 0}
    if structured_output_mode != "off" and structured_output_schema:
        generation_config.update(gemini_rest_generation_config(structured_output_schema))
    plain_generation_config = dict(generation_config)
    plain_generation_config.pop("responseMimeType", None)
    plain_generation_config.pop("responseJsonSchema", None)
    payload = {
        "contents": [{"parts": parts}],
        "generationConfig": generation_config,
    }
    plain_payload = {"contents": [{"parts": parts}], "generationConfig": plain_generation_config}
    for attempt in range(3):
        try:
            try:
                response = requests.post(url, json=payload, timeout=90)
                if structured_output_mode != "off" and structured_output_schema and response.status_code == 200:
                    set_last_structured_output_info(used=True, schema=structured_output_schema, fallback_reason=None)
                elif structured_output_mode == "off":
                    set_last_structured_output_info(used=False, schema=structured_output_schema, fallback_reason=None)
            except Exception:
                raise
            break
        except (
            requests.exceptions.Timeout,
            requests.exceptions.ConnectionError,
            requests.exceptions.ChunkedEncodingError,
        ) as exc:
            if attempt == 2:
                return None, f"net_error after 3 retries: {type(exc).__name__}: {str(exc)[:120]}"
            time.sleep(2**attempt)
        except Exception as exc:
            return None, f"unexpected_error: {type(exc).__name__}: {str(exc)[:120]}"
    if (
        response.status_code != 200
        and structured_output_mode != "off"
        and structured_output_schema
        and normalize_structured_fallback(structured_output_mode, structured_output_fallback) == "retry_plain"
        and looks_like_structured_output_unsupported(Exception(response.text[:500]))
    ):
        set_last_structured_output_info(
            used=False,
            schema=structured_output_schema,
            fallback_reason=format_structured_fallback_reason(Exception(response.text[:500])),
        )
        response = requests.post(url, json=plain_payload, timeout=90)
    if response.status_code != 200:
        return None, f"HTTP {response.status_code}: {response.text[:160]}"
    try:
        body = response.json()
        if cache_usage_log:
            set_last_provider_cache_info(
                enabled=cache_state.get("enabled"),
                cache_key=cache_key,
                reason=cache_state.get("reason"),
                usage=summarize_usage_dict(provider="google", usage=body.get("usageMetadata") or {}),
            )
        candidates = body.get("candidates") or []
        cand = candidates[0] if candidates else {}
        content = cand.get("content") or {}
        content_parts = content.get("parts")
        if not isinstance(content_parts, list):
            return None, f"gemini_empty_parts finishReason={cand.get('finishReason')}"
        for part in content_parts:
            if isinstance(part, dict) and isinstance(part.get("text"), str) and part["text"]:
                return part["text"], None
        return None, f"gemini_empty_parts finishReason={cand.get('finishReason')}"
    except (KeyError, IndexError, TypeError) as exc:
        return None, str(exc)


def query_openai(
    image_paths: list[str],
    question: str,
    model: str = "gpt-4.1-2025-04-14",
    provider_cache: str = "off",
    prompt_cache_retention: str = "in_memory",
    cache_key: str | None = None,
    cache_usage_log: bool = True,
    max_tokens: int = 1024,
    structured_output_mode: str = "off",
    structured_output_schema: str | None = None,
    structured_output_fallback: str = "retry_plain",
):
    try:
        return _query_openai_compatible_vision(
            image_paths=image_paths,
            question=question,
            model=model,
            api_key=OPENAI_API_KEY,
            provider_cache=provider_cache,
            prompt_cache_retention=prompt_cache_retention,
            cache_key=cache_key,
            cache_usage_log=cache_usage_log,
            max_tokens=max_tokens,
            structured_output_mode=structured_output_mode,
            structured_output_schema=structured_output_schema,
            structured_output_fallback=structured_output_fallback,
        ), None
    except Exception as exc:
        return None, str(exc)[:200]


def query_qwen_vl(
    image_paths: list[str],
    question: str,
    model: str = "qwen3-vl-32b-instruct",
    max_tokens: int = 1024,
    structured_output_mode: str = "off",
    structured_output_schema: str | None = None,
    structured_output_fallback: str = "retry_plain",
):
    set_last_provider_cache_info(enabled=False, reason="unsupported_dashscope")
    try:
        return _query_openai_compatible_vision(
            image_paths=image_paths,
            question=question,
            model=model,
            api_key=QWEN_API_KEY,
            base_url=DASHSCOPE_BASE_URL,
            max_tokens=max_tokens,
            structured_output_mode=structured_output_mode,
            structured_output_schema=structured_output_schema,
            structured_output_fallback=structured_output_fallback,
        ), None
    except Exception as exc:
        return None, str(exc)[:200]


def query_claude(
    image_paths: list[str],
    question: str,
    model: str = "claude-3-7-sonnet-20250219",
    provider_cache: str = "off",
    prompt_cache_retention: str = "in_memory",
    cache_key: str | None = None,
    cache_usage_log: bool = True,
    max_tokens: int = 1024,
    structured_output_mode: str = "off",
    structured_output_schema: str | None = None,
    structured_output_fallback: str = "retry_plain",
):
    import requests

    cache_state = provider_cache_state(
        provider="anthropic",
        provider_cache=provider_cache if image_paths else "off",
        model_id=model,
        prompt_cache_retention=prompt_cache_retention,
    )
    set_last_provider_cache_info(enabled=cache_state.get("enabled"), cache_key=cache_key, reason=cache_state.get("reason"))
    content = []
    for image_path in image_paths:
        content.append(
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": guess_mime_type(image_path),
                    "data": encode_image(image_path),
                },
            }
        )
    add_anthropic_cache_control(content, enabled=bool(cache_state.get("enabled")))
    content.append({"type": "text", "text": question})
    payload = {
        "model": model,
        "max_tokens": max_tokens,
        "temperature": 0,
        "messages": [{"role": "user", "content": content}],
    }
    if structured_output_mode != "off" and structured_output_schema:
        payload["tools"] = [
            {
                "name": "emit_structured_output",
                "description": "Return the benchmark answer in the required JSON schema.",
                "input_schema": openai_chat_response_format(structured_output_schema)["json_schema"]["schema"],
            }
        ]
        payload["tool_choice"] = {"type": "tool", "name": "emit_structured_output"}
    plain_payload = dict(payload)
    plain_payload.pop("tools", None)
    plain_payload.pop("tool_choice", None)

    for attempt in range(3):
        try:
            response = requests.post(
                "https://api.anthropic.com/v1/messages",
                headers={
                    "x-api-key": ANTHROPIC_API_KEY,
                    "anthropic-version": "2023-06-01",
                    "content-type": "application/json",
                },
                json=payload,
                timeout=90,
            )
            if structured_output_mode != "off" and structured_output_schema and response.status_code == 200:
                set_last_structured_output_info(used=True, schema=structured_output_schema, fallback_reason=None)
            elif structured_output_mode == "off":
                set_last_structured_output_info(used=False, schema=structured_output_schema, fallback_reason=None)
            break
        except (
            requests.exceptions.Timeout,
            requests.exceptions.ConnectionError,
            requests.exceptions.ChunkedEncodingError,
        ) as exc:
            if attempt == 2:
                return None, f"net_error after 3 retries: {type(exc).__name__}: {str(exc)[:120]}"
            time.sleep(2**attempt)
        except Exception as exc:
            return None, f"unexpected_error: {type(exc).__name__}: {str(exc)[:120]}"
    if (
        response.status_code != 200
        and structured_output_mode != "off"
        and structured_output_schema
        and normalize_structured_fallback(structured_output_mode, structured_output_fallback) == "retry_plain"
        and looks_like_structured_output_unsupported(Exception(response.text[:500]))
    ):
        set_last_structured_output_info(
            used=False,
            schema=structured_output_schema,
            fallback_reason=format_structured_fallback_reason(Exception(response.text[:500])),
        )
        response = requests.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "x-api-key": ANTHROPIC_API_KEY,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
            json=plain_payload,
            timeout=90,
        )
    if response.status_code != 200:
        return None, f"HTTP {response.status_code}: {response.text[:160]}"
    try:
        body = response.json()
        if cache_usage_log:
            set_last_provider_cache_info(
                enabled=cache_state.get("enabled"),
                cache_key=cache_key,
                reason=cache_state.get("reason"),
                usage=summarize_usage_dict(provider="anthropic", usage=body.get("usage") or {}),
            )
        first_content = body["content"][0]
        if first_content.get("type") == "tool_use" and isinstance(first_content.get("input"), dict):
            text = json.dumps(first_content["input"], ensure_ascii=False)
        else:
            text = first_content["text"]
        return text, None
    except (KeyError, IndexError, TypeError) as exc:
        return None, str(exc)


def query_openrouter(image_paths: list[str], question: str, model: str):
    import requests

    set_last_provider_cache_info(enabled=False, reason="unsupported_openrouter")
    content = []
    for image_path in image_paths:
        content.append({"type": "image_url", "image_url": {"url": make_data_url(image_path)}})
    content.append({"type": "text", "text": question})

    for attempt in range(3):
        try:
            response = requests.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                    "Content-Type": "application/json",
                    "HTTP-Referer": "https://github.com/YanLin-Quinne/vlm_biased",
                    "X-Title": "vlm-biased-eval",
                },
                json={
                    "model": model,
                    "messages": [{"role": "user", "content": content}],
                    "max_tokens": 1024,
                    "temperature": 0,
                },
                timeout=90,
            )
            break
        except (
            requests.exceptions.Timeout,
            requests.exceptions.ConnectionError,
            requests.exceptions.ChunkedEncodingError,
        ) as exc:
            if attempt == 2:
                return None, f"net_error after 3 retries: {type(exc).__name__}: {str(exc)[:120]}"
            time.sleep(2**attempt)
        except Exception as exc:
            return None, f"unexpected_error: {type(exc).__name__}: {str(exc)[:120]}"
    if response.status_code != 200:
        return None, f"HTTP {response.status_code}: {response.text[:160]}"
    try:
        text = response.json()["choices"][0]["message"]["content"]
        return _content_to_text(text), None
    except (KeyError, IndexError, TypeError) as exc:
        return None, str(exc)


def query_local(
    image_paths: list[str],
    question: str,
    model_name: str,
    local_url: str = "http://localhost:8000",
    max_tokens: int = 1024,
    structured_output_mode: str = "off",
    structured_output_schema: str | None = None,
    structured_output_fallback: str = "retry_plain",
):
    set_last_provider_cache_info(enabled=False, reason="unsupported_server")
    normalized_url = _normalize_openai_compatible_url(local_url)
    try:
        return _query_openai_compatible_vision(
            image_paths=image_paths,
            question=question,
            model=model_name,
            api_key="EMPTY",
            base_url=normalized_url,
            max_tokens=max_tokens,
            structured_output_mode=structured_output_mode,
            structured_output_schema=structured_output_schema,
            structured_output_fallback=structured_output_fallback,
        ), None
    except Exception as exc:
        return None, str(exc)[:200]


def query_vlm(
    image_paths: list[str],
    question: str,
    model_name: str,
    local_url: str | None = None,
    provider: str = "native",
    open_backend: str = "server",
    server_model_id: str | None = None,
    provider_cache: str = "off",
    prompt_cache_retention: str = "in_memory",
    cache_key: str | None = None,
    cache_usage_log: bool = True,
    max_tokens: int = 1024,
    reasoning_effort: str = "off",
    structured_output_mode: str = "off",
    structured_output_schema: str | None = None,
    structured_output_fallback: str = "retry_plain",
):
    """
    Unified VLM query dispatcher.
    """
    openrouter_models = {
        "gemini-2.5-pro": "google/gemini-2.5-pro",
        "gemini-3.1-pro-preview": "google/gemini-3.1-pro-preview",
        "claude-sonnet-3.7": "anthropic/claude-3.7-sonnet",
        "claude-sonnet-4": "anthropic/claude-sonnet-4",
        "gpt-4.1": "openai/gpt-4.1",
        "gpt-5": "openai/gpt-5",
    }
    gemini_models = {"gemini-2.5-pro", "gemini-3.1-pro-preview"}
    claude_models = {"claude-sonnet-3.7", "claude-sonnet-4"}
    openai_models = {"gpt-4.1", "gpt-5"}
    if model_name in gemini_models:
        if provider == "openrouter":
            return query_openrouter(image_paths, question, model=openrouter_models[model_name])
        return query_gemini(
            image_paths,
            question,
            model=MODEL_IDS[model_name],
            provider_cache=provider_cache,
            prompt_cache_retention=prompt_cache_retention,
            cache_key=cache_key,
            cache_usage_log=cache_usage_log,
            max_tokens=max_tokens,
            reasoning_effort=reasoning_effort,
            structured_output_mode=structured_output_mode,
            structured_output_schema=structured_output_schema,
            structured_output_fallback=structured_output_fallback,
        )
    if model_name in claude_models:
        if provider == "openrouter":
            return query_openrouter(image_paths, question, model=openrouter_models[model_name])
        return query_claude(
            image_paths,
            question,
            model=MODEL_IDS[model_name],
            provider_cache=provider_cache,
            prompt_cache_retention=prompt_cache_retention,
            cache_key=cache_key,
            cache_usage_log=cache_usage_log,
            max_tokens=max_tokens,
            structured_output_mode=structured_output_mode,
            structured_output_schema=structured_output_schema,
            structured_output_fallback=structured_output_fallback,
        )
    if model_name in openai_models:
        if provider == "openrouter":
            return query_openrouter(image_paths, question, model=openrouter_models[model_name])
        return query_openai(
            image_paths,
            question,
            model=MODEL_IDS[model_name],
            provider_cache=provider_cache,
            prompt_cache_retention=prompt_cache_retention,
            cache_key=cache_key,
            cache_usage_log=cache_usage_log,
            max_tokens=max_tokens,
            structured_output_mode=structured_output_mode,
            structured_output_schema=structured_output_schema,
            structured_output_fallback=structured_output_fallback,
        )
    if model_name in MODEL_SETS["open"]:
        if open_backend == "dashscope":
            if not MODEL_IDS[model_name].startswith("Qwen/"):
                raise SystemExit(
                    f"DashScope open backend is only supported for Qwen model aliases, got {model_name}. "
                    "Use --open-backend server for InternVL, Gemma, or other OpenAI-compatible local servers."
                )
            dashscope_model = MODEL_IDS[model_name].split("/")[-1].lower()
            return query_qwen_vl(
                image_paths,
                question,
                model=dashscope_model,
                max_tokens=max_tokens,
                structured_output_mode=structured_output_mode,
                structured_output_schema=structured_output_schema,
                structured_output_fallback=structured_output_fallback,
            )
        server_target = server_model_id or MODEL_IDS[model_name]
        return query_local(
            image_paths,
            question,
            model_name=server_target,
            local_url=local_url or DEFAULT_SERVER_URL,
            max_tokens=max_tokens,
            structured_output_mode=structured_output_mode,
            structured_output_schema=structured_output_schema,
            structured_output_fallback=structured_output_fallback,
        )
    local_target = server_model_id or MODEL_IDS.get(model_name, model_name)
    return query_local(
        image_paths,
        question,
        model_name=local_target,
        local_url=local_url or DEFAULT_SERVER_URL,
        max_tokens=max_tokens,
        structured_output_mode=structured_output_mode,
        structured_output_schema=structured_output_schema,
        structured_output_fallback=structured_output_fallback,
    )


def extract_answer(response):
    """
    Extract the last {...} span when present, otherwise return stripped text.
    """
    response = strip_thinking_blocks(response or "")
    if not response:
        return None
    structured_open = parse_structured_open_answer(response)
    if structured_open is not None:
        return structured_open
    structured_yes_no = parse_structured_yes_no_answer(response)
    if structured_yes_no is not None:
        return structured_yes_no
    matches = re.findall(r"\{([^}]+)\}", response)
    if matches:
        return matches[-1].strip()
    return response.strip()


def extract_mc_letter(response):
    response = strip_thinking_blocks(response or "")
    structured = parse_structured_mc_answer(response or "")
    if structured is not None:
        return structured
    text = (response or "").strip()
    if not text:
        return None
    patterns = (
        r"^\s*(?:option|choice|answer)?\s*([A-D])\s*(?:[\).:\-]|$|\n)",
        r"^\s*(?:the\s+)?(?:answer|option|choice)\s*(?:is|:)\s*([A-D])\b",
        r"\b(?:final\s+answer|answer|option|choice)\s*(?:is|:)\s*([A-D])\b",
    )
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            return match.group(1).upper()
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if lines:
        match = re.fullmatch(r"(?:final\s+answer\s*[:\-]?\s*)?([A-D])\s*[\).]?", lines[-1], flags=re.IGNORECASE)
        if match:
            return match.group(1).upper()
    match = re.search(r"(?:^|[\s\n])([A-D])\s*$", text, flags=re.IGNORECASE)
    if match:
        return match.group(1).upper()
    return None


def _first_sorted_value(mapping):
    if not mapping:
        return ""
    first_key = sorted(mapping)[0]
    value = mapping[first_key]
    if isinstance(value, list):
        return ", ".join(str(item) for item in value)
    return str(value)


def load_metadata() -> dict[str, dict]:
    """
    Load CF metadata and normalize it into a filename-indexed dict.
    """
    index = {}
    sources = [
        ("fashion", METADATA_DIR / "fashion_cf_metadata.json"),
        ("industry", METADATA_DIR / "industry_cf_metadata.json"),
    ]
    for domain, path in sources:
        records = json.loads(path.read_text())
        for item in records:
            if item.get("success") is False:
                continue
            key = item.get("brand_key") or item.get("sub_category")
            filename = item["filename"]
            original_image = item["original_image"]
            source = "real" if "_real_" in original_image else "ai"
            cf_gt = _first_sorted_value(item.get("ground_truth") or {})
            cf_bias = _first_sorted_value(item.get("bias_answer") or {})
            index[filename] = {
                "key": key,
                "mod_id": item["mod_id"],
                "mod_type": item["mod_type"],
                "gt": cf_gt,
                "bias": cf_bias,
                "answers": {
                    "cf_only": {"correct": cf_gt, "biased": cf_bias},
                    "both": {"correct": cf_gt, "biased": cf_bias},
                    "orig_only": {"correct": cf_bias, "biased": None},
                },
                "model_gen": item.get("model", ""),
                "domain": domain,
                "original_image": original_image,
                "source": source,
            }
    return index


def _resolve_cf_path(domain, key, filename):
    root = FASHION_CF_DIR if domain == "fashion" else INDUSTRY_CF_DIR
    return root / key / filename


def _resolve_original_path(domain, key, filename):
    root = FASHION_DATASET_DIR if domain == "fashion" else INDUSTRY_DATASET_DIR
    return root / key / filename


def resolve_manifest_path(value, *, base_dir=None):
    raw = Path(str(value)).expanduser()
    if raw.is_absolute() and raw.exists():
        return raw.resolve()
    value_text = str(value)
    for marker in (
        "dataset/fashion_dataset/",
        "dataset/industry_dataset/",
        "cf_dataset/fashion_cf/",
        "cf_dataset/industry_cf/",
        "vision_dataset/fashion/",
        "vision_dataset/industry/",
        "eval_results/fashion_industry/",
    ):
        marker_index = value_text.find(marker)
        if marker_index != -1:
            return (REPO_ROOT / value_text[marker_index:]).resolve(strict=False)
    if raw.is_absolute():
        return raw.resolve(strict=False)
    root = base_dir or REPO_ROOT
    return (root / raw).resolve(strict=False)


def load_evidence_manifest(path):
    if path is None:
        return {}
    manifest_path = Path(path).expanduser()
    manifest_path = manifest_path if manifest_path.is_absolute() else (REPO_ROOT / manifest_path)
    manifest_path = manifest_path.resolve(strict=False)
    if not manifest_path.exists():
        raise SystemExit(f"Evidence manifest not found: {manifest_path}")
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    records = payload.get("records", payload) if isinstance(payload, dict) else payload
    if not isinstance(records, list):
        raise SystemExit(f"Evidence manifest must be a list or contain records: {manifest_path}")
    index = {}
    problems = []
    for idx, record in enumerate(records, start=1):
        if not isinstance(record, dict):
            problems.append(f"record {idx}: expected object")
            continue
        if record.get("schema_version") != EVIDENCE_SCHEMA_VERSION:
            problems.append(f"record {idx}: unsupported schema_version={record.get('schema_version')!r}")
            continue
        key = evidence_manifest_key_from_record(record)
        if not all(key):
            problems.append(f"record {idx}: missing domain/key/cf_filename")
            continue
        if key in index:
            problems.append(f"record {idx}: duplicate evidence key {key}")
            continue
        for role in ("original", "cf"):
            payload = record.get(role)
            if not isinstance(payload, dict):
                problems.append(f"record {idx}: missing {role} evidence block")
                continue
            for field in ("raw", "bbox_overlay", "crop", "zoom_panel"):
                value = payload.get(field)
                if not value:
                    continue
                resolved = resolve_manifest_path(value, base_dir=manifest_path.parent)
                if not resolved.exists():
                    problems.append(f"record {idx}: missing referenced file {resolved}")
        index[key] = record
    if problems:
        preview = "\n- ".join(problems[:20])
        suffix = "" if len(problems) <= 20 else f"\n- ... {len(problems) - 20} additional problems"
        raise SystemExit(f"Evidence manifest validation failed:\n- {preview}{suffix}")
    return index


def evidence_manifest_key_from_record(record):
    return (
        str(record.get("domain") or ""),
        str(record.get("key") or record.get("subcategory") or ""),
        str(record.get("cf_filename") or ""),
    )


def evidence_manifest_key_from_task(task):
    return (
        str(task.get("domain") or ""),
        str(task.get("key") or ""),
        str(task.get("cf_filename") or ""),
    )


def validate_tool_condition_args(tool_condition, evidence_manifest, evidence_qc_filter, missing_evidence_policy):
    if tool_condition not in TOOL_CONDITIONS:
        raise SystemExit(f"Unsupported tool condition: {tool_condition}")
    if evidence_qc_filter not in EVIDENCE_QC_FILTERS:
        raise SystemExit(f"Unsupported evidence QC filter: {evidence_qc_filter}")
    if missing_evidence_policy not in MISSING_EVIDENCE_POLICIES:
        raise SystemExit(f"Unsupported missing evidence policy: {missing_evidence_policy}")
    if tool_condition != "raw" and evidence_manifest is None:
        raise SystemExit("--evidence-manifest is required when --tool-condition is not raw.")


def required_roles_for_input_mode(input_mode):
    if input_mode == "cf_only":
        return ("cf",)
    if input_mode == "both":
        return ("original", "cf")
    if input_mode == "orig_only":
        return ("original",)
    raise ValueError(f"Unsupported input mode: {input_mode}")


def role_name_for_record(role):
    return "target" if role == "cf" else "original"


def tool_condition_field(tool_condition):
    if tool_condition == "raw":
        return None
    return {
        "bbox": "bbox_overlay",
        "crop": "crop",
        "zoom_panel": "zoom_panel",
    }[tool_condition]


def qc_status_allowed(qc_status, evidence_qc_filter):
    if evidence_qc_filter == "all":
        return True
    if evidence_qc_filter == "pass_review":
        return qc_status in {"pass", "review"}
    return qc_status == "pass"


def role_has_tool_evidence(role_payload, tool_condition, *, base_dir=None):
    field = tool_condition_field(tool_condition)
    if not field:
        return True, None
    value = role_payload.get(field)
    if not value:
        return False, field
    if not resolve_manifest_path(value, base_dir=base_dir).exists():
        return False, field
    return True, None


def check_task_tool_eligibility(
    task,
    *,
    input_mode,
    tool_condition,
    manifest_record,
    evidence_qc_filter,
    missing_evidence_policy,
    evidence_manifest_path=None,
):
    if manifest_record is None:
        if tool_condition == "raw" and evidence_manifest_path is None:
            return True, "no_manifest_required", None, None, False
        return False, "missing_manifest_record", None, None, False
    base_dir = Path(evidence_manifest_path).expanduser().resolve().parent if evidence_manifest_path else None
    raw_fallback = False
    for role in required_roles_for_input_mode(input_mode):
        role_payload = manifest_record.get(role)
        if not isinstance(role_payload, dict):
            return False, "missing_role_payload", None, role, False
        role_qc = str(role_payload.get("qc_status") or manifest_record.get("qc_status") or "")
        if not qc_status_allowed(role_qc, evidence_qc_filter):
            return False, "qc_status_filtered", role_qc, role, False
        if tool_condition == "raw":
            continue
        ok, missing_field = role_has_tool_evidence(role_payload, tool_condition, base_dir=base_dir)
        if ok:
            continue
        if missing_evidence_policy == "raw_fallback":
            raw_fallback = True
            continue
        return False, "missing_tool_condition", role_qc, f"{role}.{missing_field}", False
    return True, "raw_fallback" if raw_fallback else "eligible", None, None, raw_fallback


def filter_tasks_for_evidence(
    tasks,
    *,
    input_mode,
    tool_condition,
    evidence_manifest,
    evidence_manifest_path,
    evidence_qc_filter,
    missing_evidence_policy,
):
    eligible = []
    skipped_by_reason = Counter()
    skipped_by_qc_status = Counter()
    skipped_by_missing_tool_condition = Counter()
    raw_fallback_count = 0
    for task in tasks:
        manifest_record = evidence_manifest.get(evidence_manifest_key_from_task(task)) if evidence_manifest else None
        ok, reason, qc_status, missing_tool_condition, raw_fallback = check_task_tool_eligibility(
            task,
            input_mode=input_mode,
            tool_condition=tool_condition,
            manifest_record=manifest_record,
            evidence_qc_filter=evidence_qc_filter,
            missing_evidence_policy=missing_evidence_policy,
            evidence_manifest_path=evidence_manifest_path,
        )
        if ok:
            task = dict(task)
            task["_evidence_manifest_record"] = manifest_record
            task["_evidence_raw_fallback"] = raw_fallback
            task["_evidence_eligibility_reason"] = reason
            eligible.append(task)
            if raw_fallback:
                raw_fallback_count += 1
            continue
        skipped_by_reason[reason] += 1
        if qc_status:
            skipped_by_qc_status[qc_status] += 1
        if missing_tool_condition:
            skipped_by_missing_tool_condition[missing_tool_condition] += 1
    stats = {
        "manifest_records_total": len(evidence_manifest or {}),
        "input_task_count": len(tasks),
        "eligible_task_count": len(eligible),
        "skipped_task_count_due_to_evidence": len(tasks) - len(eligible),
        "skipped_by_reason": dict(sorted(skipped_by_reason.items())),
        "skipped_by_qc_status": dict(sorted(skipped_by_qc_status.items())),
        "skipped_by_missing_tool_condition": dict(sorted(skipped_by_missing_tool_condition.items())),
        "raw_fallback_task_count": raw_fallback_count,
        "evidence_qc_filter": evidence_qc_filter,
        "missing_evidence_policy": missing_evidence_policy,
    }
    if missing_evidence_policy == "error" and stats["skipped_task_count_due_to_evidence"] > 0:
        raise SystemExit("Evidence eligibility failed: " + json.dumps(stats, sort_keys=True))
    return eligible, stats


def print_evidence_filter_summary(stats):
    print(
        "Evidence filter: "
        f"manifest_records={stats.get('manifest_records_total', 0)} "
        f"input_tasks={stats.get('input_task_count', 0)} "
        f"eligible_tasks={stats.get('eligible_task_count', 0)} "
        f"skipped_tasks={stats.get('skipped_task_count_due_to_evidence', 0)} "
        f"qc_filter={stats.get('evidence_qc_filter')} "
        f"missing_policy={stats.get('missing_evidence_policy')}"
    )
    if stats.get("skipped_by_reason"):
        print("Evidence skipped_by_reason=" + json.dumps(stats["skipped_by_reason"], sort_keys=True))
    if stats.get("skipped_by_qc_status"):
        print("Evidence skipped_by_qc_status=" + json.dumps(stats["skipped_by_qc_status"], sort_keys=True))
    if stats.get("skipped_by_missing_tool_condition"):
        print(
            "Evidence skipped_by_missing_tool_condition="
            + json.dumps(stats["skipped_by_missing_tool_condition"], sort_keys=True)
        )


def build_image_groups(task, *, input_mode, tool_condition, evidence_manifest_path=None, missing_evidence_policy="error"):
    manifest_record = task.get("_evidence_manifest_record")
    if tool_condition == "raw":
        if input_mode == "cf_only":
            return [{"role": "target", "raw": task["cf_path"], "evidence": []}], manifest_record
        if input_mode == "orig_only":
            return [{"role": "original", "raw": task["original_path"], "evidence": []}], manifest_record
        if input_mode == "both":
            return [
                {"role": "original", "raw": task["original_path"], "evidence": []},
                {"role": "target", "raw": task["cf_path"], "evidence": []},
            ], manifest_record
        raise ValueError(f"Unsupported input mode: {input_mode}")
    if manifest_record is None:
        raise SystemExit(f"Missing evidence manifest record for {evidence_manifest_key_from_task(task)}")
    base_dir = Path(evidence_manifest_path).expanduser().resolve().parent if evidence_manifest_path else None
    groups = []
    for role in required_roles_for_input_mode(input_mode):
        role_payload = manifest_record[role]
        raw = resolve_manifest_path(role_payload["raw"], base_dir=base_dir)
        evidence = []
        field = tool_condition_field(tool_condition)
        value = role_payload.get(field) if field else None
        if value:
            evidence_path = resolve_manifest_path(value, base_dir=base_dir)
            if evidence_path.exists():
                evidence.append(str(evidence_path))
        if field and not evidence and missing_evidence_policy != "raw_fallback":
            raise ValueError(f"Missing evidence for {role}.{field}")
        groups.append({"role": role_name_for_record(role), "raw": str(raw), "evidence": evidence})
    return groups, manifest_record


def flatten_image_groups(groups):
    image_paths = []
    for group in groups:
        image_paths.append(str(group["raw"]))
        image_paths.extend(str(path) for path in group.get("evidence", []))
    return image_paths


def serialize_image_groups(groups):
    return [
        {
            "role": group.get("role"),
            "raw": str(group.get("raw")),
            "evidence": [str(path) for path in group.get("evidence", [])],
        }
        for group in groups
    ]


def apply_tool_prompt_prefix(question, *, input_mode, tool_condition, image_group_labels="off"):
    if tool_condition == "raw":
        if input_mode == "both":
            return f"{BOTH_RAW_PAIR_DIFFERENCE_PREFIX}\n\n{question.strip()}"
        return question
    if input_mode == "both":
        prefix = TOOL_VIEW_BOTH_PREFIX
    else:
        prefix = TOOL_VIEW_PREFIX
    if image_group_labels == "on":
        prefix = (
            f"{prefix} "
            "The image order is grouped: each group starts with the base view, followed by its alternative view."
        )
    return f"{prefix}\n\n{question.strip()}"


def tool_record_fields(
    *,
    task,
    tool_condition,
    evidence_manifest_path,
    image_groups,
    evidence_record,
    evidence_qc_filter,
    missing_evidence_policy,
):
    used_evidence_paths = [path for group in image_groups for path in group.get("evidence", [])]
    role_instance_ids = {}
    role_instances_merged = {}
    role_prompt_texts = {}
    if isinstance(evidence_record, dict):
        for role in required_roles_for_input_mode(task["input_mode"]):
            payload = evidence_record.get(role) or {}
            record_role = role_name_for_record(role)
            role_instance_ids[record_role] = payload.get("selected_instance_id")
            role_instances_merged[record_role] = payload.get("instances_merged")
            role_prompt_texts[record_role] = payload.get("selected_prompt_text")
    generator = evidence_record.get("generator", {}) if isinstance(evidence_record, dict) else {}
    return {
        "tool_condition": tool_condition,
        "evidence_manifest": str(evidence_manifest_path) if evidence_manifest_path else None,
        "evidence_schema_version": evidence_record.get("schema_version") if isinstance(evidence_record, dict) else None,
        "evidence_qc_status": evidence_record.get("qc_status") if isinstance(evidence_record, dict) else None,
        "evidence_qc_filter": evidence_qc_filter,
        "missing_evidence_policy": missing_evidence_policy,
        "used_evidence_paths": [str(path) for path in used_evidence_paths],
        "used_image_groups": serialize_image_groups(image_groups),
        "evidence_selected_instance_id": role_instance_ids,
        "instances_merged": role_instances_merged,
        "evidence_selected_prompt_text": role_prompt_texts,
        "evidence_generator": generator.get("model_id"),
        "evidence_prompt_profile": generator.get("prompt_profile"),
        "evidence_raw_fallback": bool(task.get("_evidence_raw_fallback")),
    }


def _parse_question_types(question_types):
    if question_types == "all":
        return list(QUESTION_TYPES)
    parsed = [item.strip().upper() for item in question_types.split(",") if item.strip()]
    invalid = [item for item in parsed if item not in QUESTION_TYPES]
    if invalid:
        raise ValueError(f"Invalid question types: {', '.join(invalid)}")
    return parsed


def _evaluation_target(key, mod_id):
    target = EVALUATION_TARGETS.get((key, mod_id))
    if target:
        return target
    return get_questions(key, mod_id=mod_id)["Q1"]


def _build_bias_rationale(key, canonical_answer, actual_answer):
    display_name = DISPLAY_NAME_MAP[key]
    return (
        f"A {display_name} is strongly associated with {canonical_answer}. "
        f"A model relying on prior knowledge may answer '{canonical_answer}' "
        f"even when the image clearly shows '{actual_answer}'."
    )


def _select_cf_records(records, n_per_cat):
    sorted_records = sorted(records, key=lambda record: record["filename"])
    if n_per_cat is None or n_per_cat >= len(sorted_records):
        return sorted_records
    if n_per_cat <= 1:
        return sorted_records[:n_per_cat]

    selected = []
    used = set()
    for model_gen in ("gemini", "gpt"):
        for idx, record in enumerate(sorted_records):
            if idx in used or record["model_gen"] != model_gen:
                continue
            selected.append(record)
            used.add(idx)
            if len(selected) == n_per_cat:
                return sorted(selected, key=lambda record: record["filename"])
            break

    for idx, record in enumerate(sorted_records):
        if len(selected) == n_per_cat:
            break
        if idx in used:
            continue
        selected.append(record)
        used.add(idx)
    return sorted(selected, key=lambda record: record["filename"])


def collect_images(
    phase="cf",
    input_mode="cf_only",
    question_types=None,
    n_per_cat=None,
    prime=True,
    domain=None,
    key=None,
    tool_condition="raw",
):
    """
    Build evaluation tasks.
    """
    question_types = question_types or list(QUESTION_TYPES)
    tasks = []

    if phase == "sanity":
        if input_mode != "orig_only":
            raise ValueError("sanity phase only supports --input-mode orig_only")
        if "Q_MC" in question_types:
            raise ValueError("Q_MC is only supported for phase=cf")
        for resolved_domain, db, dataset_root in (
            ("industry", EVAL_INDUSTRY, INDUSTRY_DATASET_DIR),
            ("fashion", EVAL_FASHION, FASHION_DATASET_DIR),
        ):
            if domain and resolved_domain != domain:
                continue
            for resolved_key in sorted(db):
                if key and resolved_key != key:
                    continue
                image_dir = dataset_root / resolved_key
                if not image_dir.exists():
                    continue
                for image_path in sorted(path for path in image_dir.iterdir() if path.is_file() and not path.name.startswith(".")):
                    source = "real" if "_real_" in image_path.name else "ai"
                    for question_type in question_types:
                        tasks.append(
                            {
                                "phase": "sanity",
                                "domain": resolved_domain,
                                "key": resolved_key,
                                "mod_id": None,
                                "mod_type": "sanity",
                                "question_type": question_type,
                                "question": get_primed_question(resolved_key, question_type, mod_id=None, prime=prime),
                                "correct": None,
                                "biased": None,
                                "evaluation_target": None,
                                "bias_rationale": "",
                                "mc_options": None,
                                "correct_letter": None,
                                "biased_letter": None,
                                "gt": None,
                                "bias": None,
                                "image_paths": [str(image_path)],
                                "original_path": str(image_path),
                                "cf_path": None,
                                "gen_model": None,
                                "source": source,
                                "original_image": image_path.name,
                                "cf_filename": None,
                                "basename": image_path.name,
                                "prior_statement": None,
                                "question_design_version": QUESTION_DESIGN_VERSION,
                            }
                        )
        return tasks

    if phase != "cf":
        raise ValueError(f"Unsupported phase: {phase}")

    metadata_index = load_metadata()
    grouped = {}
    for filename, meta in metadata_index.items():
        if domain and meta["domain"] != domain:
            continue
        if key and meta["key"] != key:
            continue
        grouped.setdefault((meta["domain"], meta["key"], meta["mod_id"]), []).append(
            {
                "filename": filename,
                **meta,
            }
        )

    for group_key in sorted(grouped):
        records = _select_cf_records(grouped[group_key], n_per_cat=n_per_cat)
        for record in records:
            cf_path = _resolve_cf_path(record["domain"], record["key"], record["filename"])
            orig_path = _resolve_original_path(record["domain"], record["key"], record["original_image"])
            if input_mode == "cf_only":
                image_paths = [str(cf_path)]
            elif input_mode == "orig_only":
                image_paths = [str(orig_path)]
            elif input_mode == "both":
                image_paths = [str(orig_path), str(cf_path)]
            else:
                raise ValueError(f"Unsupported input mode: {input_mode}")

            for question_type in question_types:
                payload = build_question_payload(
                    key=record["key"],
                    mod_id=record["mod_id"],
                    question_type=question_type,
                    input_mode=input_mode,
                    prime=prime,
                    tool_condition=tool_condition,
                )
                tasks.append(
                    {
                        "phase": "cf",
                        "domain": record["domain"],
                        "key": record["key"],
                        "mod_id": record["mod_id"],
                        "mod_type": record["mod_type"],
                        "question_type": question_type,
                        "question": payload["prompt"],
                        "correct": payload["correct"],
                        "biased": payload["biased"],
                        "evaluation_target": payload["evaluation_target"],
                        "bias_rationale": payload["bias_rationale"],
                        "mc_options": payload["mc_options"],
                        "correct_letter": payload["correct_letter"],
                        "biased_letter": payload["biased_letter"],
                        "gt": payload["correct"],
                        "bias": payload["biased"],
                        "prior_statement": payload["prior_statement"],
                        "question_design_version": payload["question_design_version"],
                        "both_raw_prompt_policy": payload.get("both_raw_prompt_policy"),
                        "both_raw_scoring_policy": payload.get("both_raw_scoring_policy"),
                        "image_paths": image_paths,
                        "original_path": str(orig_path),
                        "cf_path": str(cf_path),
                        "gen_model": record["model_gen"],
                        "source": record["source"],
                        "original_image": record["original_image"],
                        "cf_filename": record["filename"],
                        "basename": record["filename"],
                    }
                )
    return tasks


def _judge_prompt(
    question,
    answer,
    correct,
    biased,
    evaluation_target,
    bias_rationale,
    input_mode="cf_only",
    tool_condition="raw",
    both_raw_prompt_policy=None,
):
    is_pair_difference = input_mode == "both" and tool_condition == "raw"
    lines = [
        f"QUESTION-SPECIFIC VISIBLE TARGET: {evaluation_target}",
        f"QUESTION-SPECIFIC CORRECT ANSWER: {correct}",
    ]
    has_distinct_biased = biased is not None and biased != correct
    if has_distinct_biased:
        lines.append(f"QUESTION-SPECIFIC PRIOR-ALIGNED ANSWER: {biased}")
    if bias_rationale:
        lines.append(f"BIAS RATIONALE: {bias_rationale}")
    lines.extend(
        [
            "",
            "---",
            f"QUESTION: {question}",
            f"MODEL ANSWER: {answer}",
            "---",
            "",
            "Rules:",
            "1. Grade ONLY the QUESTION-SPECIFIC VISIBLE TARGET. Ignore unrelated observations.",
            "2. If MODEL ANSWER is empty, whitespace-only, or contains no answer content, output other.",
        ]
    )
    if is_pair_difference:
        lines.extend(
            [
                "3. correct = the answer identifies a relevant visible difference in the QUESTION-SPECIFIC VISIBLE TARGET across the two related images, or correctly answers no to a same/different yes-no comparison.",
                "4. Accept compact pair-difference expressions such as red to black, black vs red, red and black, flipped, reversed, swapped, or different order when they refer to the target.",
                "5. biased = the answer ignores the pair difference and gives a canonical/prior-aligned answer, or claims the two images show the same target attribute when they differ.",
                "6. Do NOT mark an explicit target pair-difference expression as biased merely because it is short.",
                "7. other = refusal, hedging, off-topic, only unrelated differences, or only one image described without comparing the visible target.",
                "8. Do not require the answer to identify image order or image identity.",
                "",
                'Output ONLY JSON: {"label": "correct|biased|other", "reason": "<15 words"}',
            ]
        )
        return "\n".join(lines)
    lines.append(
        "3. correct = the answer semantically matches the QUESTION-SPECIFIC CORRECT ANSWER for the visible target (allow synonyms, color-family variants, minor phrasing)."
    )
    if has_distinct_biased:
        lines.append(
            "4. biased = the answer matches the QUESTION-SPECIFIC PRIOR-ALIGNED ANSWER or describes the default stereotype instead of what's actually shown. If the answer lists both, pick biased."
        )
        lines.append("5. other = refusal, hedging, off-topic, or neither correct nor biased.")
    else:
        lines.append("4. other = anything that does not semantically match the QUESTION-SPECIFIC CORRECT ANSWER.")
    lines.extend(
        [
            "6. Numbers: exact match only; off-by-one -> other.",
            "",
            'Output ONLY JSON: {"label": "correct|biased|other", "reason": "<15 words"}',
        ]
    )
    if not has_distinct_biased:
        lines.append('(When biased is None, never output "biased".)')
    return "\n".join(lines)


def _parse_judge_response(raw_response):
    if not raw_response:
        return None
    candidate = raw_response.strip()
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", candidate, re.DOTALL)
        if not match:
            return None
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            return None


_PAIR_DIFFERENCE_MARKERS = (
    " to ",
    "->",
    "→",
    " vs ",
    " versus ",
    "different",
    "differs",
    "difference",
    "changed",
    "changes",
    "change",
    "flipped",
    "reversed",
    "swapped",
    "different order",
    "opposite",
    "rotated",
    "inverted",
)
_PAIR_SAME_MARKERS = (
    "no difference",
    "unchanged",
    "both ",
    " in both",
    "normal",
    "standard",
    "canonical",
)
_PAIR_VISUAL_VALUE_TOKENS = {
    "red",
    "black",
    "blue",
    "green",
    "yellow",
    "orange",
    "purple",
    "white",
    "brown",
    "pink",
    "left",
    "right",
    "upright",
    "rotated",
    "flipped",
    "reversed",
    "swapped",
    "overlapping",
    "separated",
    "interlocking",
    "missing",
    "present",
    "none",
    "zero",
    "one",
    "two",
    "three",
    "four",
    "five",
    "six",
    "front",
    "back",
    "face",
    "facing",
    "horse",
    "donkey",
    "rider",
    "mallet",
    "flag",
    "lance",
    "letter",
    "letters",
    "keys",
}


def _normalize_pair_difference_answer(answer: str) -> str:
    text = str(answer or "").strip().lower()
    text = re.sub(r"^[\s{[(\"']+|[\s}\])\"'.!,;:]+$", "", text)
    text = text.replace("_", " ")
    return re.sub(r"\s+", " ", text)


def _contains_explicit_pair_difference(answer: str) -> bool:
    text = _normalize_pair_difference_answer(answer)
    if not text:
        return False
    if any(marker in text for marker in _PAIR_SAME_MARKERS):
        return False
    if text == "same" or text.startswith("same "):
        return False
    if any(marker in text for marker in _PAIR_DIFFERENCE_MARKERS):
        return True
    if any(separator in text for separator in (";", ",")) or " and " in text:
        tokens = set(re.findall(r"[a-z]+", text))
        if len(tokens & _PAIR_VISUAL_VALUE_TOKENS) >= 2:
            return True
    return False


def maybe_override_both_raw_open_judge_label(
    *,
    label: str | None,
    answer: str,
    input_mode: str,
    tool_condition: str,
    question_type: str,
) -> tuple[str | None, bool, str | None]:
    if (
        label == "biased"
        and input_mode == "both"
        and tool_condition == "raw"
        and question_type in {"Q1", "Q2"}
        and _contains_explicit_pair_difference(answer)
    ):
        return "correct", True, "explicit compact pair-difference expression"
    return label, False, None


def llm_judge(
    question: str,
    answer: str,
    correct: str,
    biased: str,
    evaluation_target: str,
    bias_rationale: str,
    judge_model: str,
    judge_provider: str = "openai",
    judge_server_url: str | None = None,
    max_retries: int = 3,
    retry_base_seconds: float = 2.0,
    retry_max_seconds: float = 60.0,
    stop_on_quota: bool = True,
    judge_max_output_tokens: int = 512,
    judge_structured_output: str = "auto",
    structured_output_fallback: str = "retry_plain",
    input_mode: str = "cf_only",
    tool_condition: str = "raw",
    both_raw_prompt_policy: str | None = None,
):
    """
    Grade a model answer without looking at the image.
    """
    system_prompt = "You are a strict grader. You will NOT see the image."
    user_prompt = _judge_prompt(
        question=question,
        answer=answer or "",
        correct=correct,
        biased=biased,
        evaluation_target=evaluation_target,
        bias_rationale=bias_rationale,
        input_mode=input_mode,
        tool_condition=tool_condition,
        both_raw_prompt_policy=both_raw_prompt_policy,
    )
    actual_model = MODEL_IDS.get(judge_model, judge_model)
    last_raw = ""
    last_retry_count = 0
    last_error_kind = None
    last_fatal_error = False

    for _ in range(3):
        if judge_provider == "openai":
            last_raw, error, retry_count, error_kind, fatal_error = query_with_retries(
                lambda: (
                    _query_openai_compatible_text(
                        system_prompt=system_prompt,
                        user_prompt=user_prompt,
                        model=actual_model,
                        api_key=OPENAI_API_KEY,
                        max_tokens=judge_max_output_tokens,
                        structured_output_mode=judge_structured_output,
                        structured_output_schema=JUDGE_LABEL_SCHEMA_ID if judge_structured_output != "off" else None,
                        structured_output_fallback=normalize_structured_fallback(
                            judge_structured_output,
                            structured_output_fallback,
                        ),
                    ),
                    None,
                ),
                max_retries=max_retries,
                retry_base_seconds=retry_base_seconds,
                retry_max_seconds=retry_max_seconds,
                stop_on_quota=stop_on_quota,
            )
        elif judge_provider == "server":
            last_raw, error, retry_count, error_kind, fatal_error = query_with_retries(
                lambda: (
                    _query_openai_compatible_text(
                        system_prompt=system_prompt,
                        user_prompt=user_prompt,
                        model=actual_model,
                        api_key="EMPTY",
                        base_url=_normalize_openai_compatible_url(judge_server_url or DEFAULT_SERVER_URL),
                        max_tokens=judge_max_output_tokens,
                        structured_output_mode=judge_structured_output,
                        structured_output_schema=JUDGE_LABEL_SCHEMA_ID if judge_structured_output != "off" else None,
                        structured_output_fallback=normalize_structured_fallback(
                            judge_structured_output,
                            structured_output_fallback,
                        ),
                    ),
                    None,
                ),
                max_retries=max_retries,
                retry_base_seconds=retry_base_seconds,
                retry_max_seconds=retry_max_seconds,
                stop_on_quota=stop_on_quota,
            )
        elif judge_provider == "dashscope":
            last_raw, error, retry_count, error_kind, fatal_error = query_with_retries(
                lambda: (
                    _query_openai_compatible_text(
                        system_prompt=system_prompt,
                        user_prompt=user_prompt,
                        model=actual_model,
                        api_key=QWEN_API_KEY,
                        base_url=DASHSCOPE_BASE_URL,
                        max_tokens=judge_max_output_tokens,
                        structured_output_mode=judge_structured_output,
                        structured_output_schema=JUDGE_LABEL_SCHEMA_ID if judge_structured_output != "off" else None,
                        structured_output_fallback=normalize_structured_fallback(
                            judge_structured_output,
                            structured_output_fallback,
                        ),
                    ),
                    None,
                ),
                max_retries=max_retries,
                retry_base_seconds=retry_base_seconds,
                retry_max_seconds=retry_max_seconds,
                stop_on_quota=stop_on_quota,
            )
        else:
            return "judge_error", f"unsupported judge provider: {judge_provider}"[:100], 0, "judge_error", False

        last_retry_count = retry_count
        last_error_kind = error_kind
        last_fatal_error = fatal_error
        if error:
            return "judge_error", str(error)[:100], retry_count, error_kind or "judge_error", fatal_error

        try:
            parsed = parse_structured_judge_response(last_raw)
        except Exception:
            parsed = _parse_judge_response(last_raw)
        if parsed:
            label = str(parsed.get("label", "")).strip().lower()
            reason = str(parsed.get("reason", "")).strip()
            allowed_labels = {"correct", "other"} if biased is None else {"correct", "biased", "other"}
            if label in allowed_labels:
                return label, reason[:100], retry_count, None, False
        time.sleep(1)

    return "judge_error", str(last_raw)[:100], last_retry_count, last_error_kind or "judge_error", last_fatal_error


def _task_signature(model_name, input_mode, task):
    return (
        model_name,
        input_mode,
        task.get("tool_condition", "raw"),
        task["key"],
        task["mod_id"],
        task["question_type"],
        task["basename"],
    )


def _resolve_output_root(output_root=None):
    path = Path(output_root) if output_root is not None else RAW_RESULTS_DIR
    path = path.expanduser()
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path.resolve(strict=False)


def _load_done_set(model_name, input_mode, phase, question_types, tool_condition="raw", output_root=None):
    done = set()
    output_root = _resolve_output_root(output_root)
    if not output_root.exists():
        return done
    allowed_question_types = set(question_types)
    for path in _result_jsonl_paths(output_root):
        with open(path) as handle:
            for line in handle:
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if record.get("model") != model_name:
                    continue
                if record.get("phase") != phase or record.get("input_mode") != input_mode:
                    continue
                if record.get("tool_condition", "raw") != tool_condition:
                    continue
                if record.get("question_type") not in allowed_question_types:
                    continue
                if record_is_success(record):
                    done.add(record_signature(record))
    return done


def _result_jsonl_paths(output_root=None):
    output_root = _resolve_output_root(output_root)
    paths = []
    if output_root.exists():
        if output_root.is_file():
            paths.append(output_root)
        else:
            records_path = output_root / "records.jsonl"
            if records_path.exists():
                paths.append(records_path)
            paths.extend(sorted(output_root.glob("*.jsonl")))
            paths.extend(sorted(output_root.glob("*/records.jsonl")))
    return sorted(set(paths))


def _resolve_models(model=None, model_set="all"):
    if model:
        return [model]
    if model_set == "all":
        return MODEL_SETS["closed"] + MODEL_SETS["open"]
    return MODEL_SETS[model_set]


def _resolve_judge_model(model_name, judge):
    if judge != "auto":
        return MODEL_IDS.get(judge, judge)
    model_group = MODEL_GROUPS.get(model_name, "open")
    return JUDGE_MODELS[model_group]


def _jsonl_name(phase, input_mode, model_name, question_types, output_root=None):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    qt_label = "-".join(question_types)
    return _resolve_output_root(output_root) / f"eval_{phase}_{input_mode}_{model_name}_{qt_label}_{timestamp}.jsonl"


def _run_dir_name(phase, input_mode, model_name, question_types, output_root=None):
    return _jsonl_name(phase, input_mode, model_name, question_types, output_root=output_root).with_suffix("")


def resolve_server_preflight_plans(
    *,
    server_preflight,
    server_framework,
    uses_server_backbone,
    backbone_model_id,
    judge_model_id,
    judge_provider,
    server_url,
    judge_server_url,
    server_gpu_ids,
    server_dtype,
    server_tensor_parallel_size,
    server_gpu_memory_utilization,
    server_reserve_gb,
    server_load_in_4bit,
    server_hf_model_id,
    judge_server_gpu_ids,
    judge_server_dtype,
    judge_server_tensor_parallel_size,
    judge_server_gpu_memory_utilization,
    judge_server_reserve_gb,
    judge_server_load_in_4bit,
    judge_server_hf_model_id,
):
    if server_preflight == "off" or (not uses_server_backbone and judge_provider != "server"):
        return None, None
    args = argparse.Namespace(
        server_preflight=server_preflight,
        server_framework=server_framework,
        server_gpu_ids=server_gpu_ids,
        server_dtype=server_dtype,
        server_tensor_parallel_size=server_tensor_parallel_size,
        server_gpu_memory_utilization=server_gpu_memory_utilization,
        server_reserve_gb=server_reserve_gb,
        server_load_in_4bit=server_load_in_4bit,
        server_hf_model_id=server_hf_model_id,
        judge_server_gpu_ids=judge_server_gpu_ids,
        judge_server_dtype=judge_server_dtype,
        judge_server_tensor_parallel_size=judge_server_tensor_parallel_size,
        judge_server_gpu_memory_utilization=judge_server_gpu_memory_utilization,
        judge_server_reserve_gb=judge_server_reserve_gb,
        judge_server_load_in_4bit=judge_server_load_in_4bit,
        judge_server_hf_model_id=judge_server_hf_model_id,
    )
    same_judge_endpoint = judge_provider == "server" and _normalize_openai_compatible_url(server_url) == _normalize_openai_compatible_url(
        judge_server_url
    )
    backbone_plan = None
    judge_plan = None
    if uses_server_backbone:
        backbone_plan, judge_plan = plan_pair_from_args(
            args=args,
            backbone_model_id=backbone_model_id,
            judge_model_id=judge_model_id,
            judge_is_server=judge_provider == "server",
            same_judge_endpoint=same_judge_endpoint,
        )
    elif judge_provider == "server":
        judge_plan = plan_server_vram(
            model_id=judge_server_hf_model_id or judge_model_id,
            gpu_ids=judge_server_gpu_ids,
            tensor_parallel_size=judge_server_tensor_parallel_size or server_tensor_parallel_size,
            dtype=judge_server_dtype or server_dtype,
            load_in_4bit=judge_server_load_in_4bit,
            gpu_memory_utilization=judge_server_gpu_memory_utilization or server_gpu_memory_utilization,
            reserve_gb=judge_server_reserve_gb or server_reserve_gb,
        )
    plans = []
    if backbone_plan is not None:
        plans.append(("backbone_server", backbone_plan))
    if judge_plan is not None:
        plans.append(("judge_server", judge_plan))
    preflight_or_warn(server_preflight, plans)
    return (
        backbone_plan.to_json() if backbone_plan is not None else None,
        judge_plan.to_json() if judge_plan is not None else None,
    )


def run_eval(
    phase,
    model_name,
    input_mode="cf_only",
    question_types=None,
    n_per_cat=None,
    prime=True,
    judge="auto",
    local_url=None,
    provider="native",
    open_backend="server",
    server_url=None,
    server_model_id=None,
    judge_provider="openai",
    judge_server_url=None,
    limit=None,
    dry_run=False,
    domain=None,
    key=None,
    resume="off",
    max_retries=3,
    retry_base_seconds=2.0,
    retry_max_seconds=60.0,
    stop_on_quota=True,
    max_consecutive_errors=5,
    provider_cache="auto",
    prompt_cache_retention="in_memory",
    cache_key_mode="image_pair",
    cache_usage_log=True,
    server_preflight="off",
    server_framework="generic",
    server_gpu_ids=None,
    server_dtype="bfloat16",
    server_tensor_parallel_size="auto",
    server_gpu_memory_utilization=0.90,
    server_reserve_gb=4.0,
    server_load_in_4bit=False,
    server_hf_model_id=None,
    judge_server_gpu_ids=None,
    judge_server_dtype=None,
    judge_server_tensor_parallel_size=None,
    judge_server_gpu_memory_utilization=None,
    judge_server_reserve_gb=None,
    judge_server_load_in_4bit=False,
    judge_server_hf_model_id=None,
    judge_structured_output="auto",
    closed_form_structured_output="auto",
    structured_output_fallback="retry_plain",
    max_output_tokens=2048,
    judge_max_output_tokens=512,
    empty_response_policy="retry_once",
    empty_retry_max_output_tokens=2048,
    reasoning_effort="off",
    tool_condition="raw",
    evidence_manifest=None,
    evidence_qc_filter="pass",
    missing_evidence_policy="skip",
    image_group_labels="off",
    output_root=None,
):
    """
    Run evaluation for one model.
    """
    question_types = question_types or list(QUESTION_TYPES)
    tasks = collect_images(
        phase=phase,
        input_mode=input_mode,
        question_types=question_types,
        n_per_cat=n_per_cat,
        prime=prime,
        domain=domain,
        key=key,
        tool_condition=tool_condition,
    )
    validate_tool_condition_args(tool_condition, evidence_manifest, evidence_qc_filter, missing_evidence_policy)
    evidence_manifest_path = None
    loaded_evidence_manifest = {}
    if evidence_manifest is not None:
        evidence_manifest_path = Path(evidence_manifest).expanduser()
        evidence_manifest_path = evidence_manifest_path if evidence_manifest_path.is_absolute() else (REPO_ROOT / evidence_manifest_path)
        evidence_manifest_path = evidence_manifest_path.resolve(strict=False)
        loaded_evidence_manifest = load_evidence_manifest(evidence_manifest_path)
    prepared_tasks = []
    for task in tasks:
        task = dict(task)
        task["input_mode"] = input_mode
        task["tool_condition"] = tool_condition
        prepared_tasks.append(task)
    tasks, evidence_filter_stats = filter_tasks_for_evidence(
        prepared_tasks,
        input_mode=input_mode,
        tool_condition=tool_condition,
        evidence_manifest=loaded_evidence_manifest,
        evidence_manifest_path=evidence_manifest_path,
        evidence_qc_filter=evidence_qc_filter,
        missing_evidence_policy=missing_evidence_policy,
    )
    for task in tasks:
        image_groups, evidence_record = build_image_groups(
            task,
            input_mode=input_mode,
            tool_condition=tool_condition,
            evidence_manifest_path=evidence_manifest_path,
            missing_evidence_policy=missing_evidence_policy,
        )
        task["image_groups"] = serialize_image_groups(image_groups)
        task["image_paths"] = flatten_image_groups(image_groups)
        task["_tool_evidence_record"] = evidence_record
        task["question"] = apply_tool_prompt_prefix(
            task["question"],
            input_mode=input_mode,
            tool_condition=tool_condition,
            image_group_labels=image_group_labels,
        )

    output_root_path = _resolve_output_root(output_root)
    pending_tasks = list(tasks)
    if limit is not None:
        pending_tasks = pending_tasks[:limit]

    judge_model = _resolve_judge_model(model_name, judge)
    resolved_server_url = server_url or local_url or DEFAULT_SERVER_URL
    resolved_judge_server_url = judge_server_url or resolved_server_url
    resolved_server_model_id = server_model_id or MODEL_IDS.get(model_name, model_name)
    uses_server_backbone = model_name in MODEL_SETS["open"] or model_name not in MODEL_IDS
    record_open_backend = open_backend if model_name in MODEL_SETS["open"] else ("server" if uses_server_backbone else None)
    record_server_url = resolved_server_url if record_open_backend == "server" else None
    record_server_model_id = resolved_server_model_id if uses_server_backbone else None
    server_vram_plan, judge_server_vram_plan = resolve_server_preflight_plans(
        server_preflight=server_preflight,
        server_framework=server_framework,
        uses_server_backbone=record_open_backend == "server",
        backbone_model_id=resolved_server_model_id,
        judge_model_id=judge_model,
        judge_provider=judge_provider,
        server_url=resolved_server_url,
        judge_server_url=resolved_judge_server_url,
        server_gpu_ids=server_gpu_ids,
        server_dtype=server_dtype,
        server_tensor_parallel_size=server_tensor_parallel_size,
        server_gpu_memory_utilization=server_gpu_memory_utilization,
        server_reserve_gb=server_reserve_gb,
        server_load_in_4bit=server_load_in_4bit,
        server_hf_model_id=server_hf_model_id,
        judge_server_gpu_ids=judge_server_gpu_ids,
        judge_server_dtype=judge_server_dtype,
        judge_server_tensor_parallel_size=judge_server_tensor_parallel_size,
        judge_server_gpu_memory_utilization=judge_server_gpu_memory_utilization,
        judge_server_reserve_gb=judge_server_reserve_gb,
        judge_server_load_in_4bit=judge_server_load_in_4bit,
        judge_server_hf_model_id=judge_server_hf_model_id,
    )

    print("=" * 72)
    print(
        f"model={model_name} phase={phase} input_mode={input_mode} "
        f"tool_condition={tool_condition} "
        f"question_types={','.join(question_types)} total={len(tasks)} "
        f"skipped={len(tasks) - len(pending_tasks)} pending={len(pending_tasks)}"
    )
    print(
        f"output_root={output_root_path} "
        f"proposed_run_dir={_run_dir_name(phase, input_mode, model_name, question_types, output_root=output_root_path)} "
        f"resume={resume} question_design_version={QUESTION_DESIGN_VERSION}"
    )
    print(
        f"judge_model={judge_model} judge_provider={judge_provider} "
        f"prime={'on' if prime else 'off'}"
    )
    print(
        f"open_backend={open_backend} server_url={resolved_server_url} "
        f"server_model_id={resolved_server_model_id}"
    )
    sample_cache_key = (
        build_provider_cache_key(
            benchmark_name="fashion_industry",
            input_mode=input_mode,
            image_paths=pending_tasks[0]["image_paths"],
            cache_key_mode=cache_key_mode,
        )
        if pending_tasks
        else None
    )
    print(
        f"provider_cache={provider_cache} retention={prompt_cache_retention} "
        f"cache_key_mode={cache_key_mode} sample_cache_key={sample_cache_key}"
    )
    print(
        "structured_output="
        f"judge:{judge_structured_output} closed_form:{closed_form_structured_output} "
        f"fallback:{structured_output_fallback}"
    )
    if evidence_manifest_path is not None or tool_condition != "raw":
        print_evidence_filter_summary(evidence_filter_stats)
    print("=" * 72)

    if dry_run:
        for idx, task in enumerate(pending_tasks, start=1):
            preview = {
                "idx": idx,
                "key": task["key"],
                "mod_id": task["mod_id"],
                "question_type": task["question_type"],
                "question": task["question"],
                "correct": task["correct"],
                "biased": task["biased"],
                "evaluation_target": task["evaluation_target"],
                "prior_statement": task.get("prior_statement"),
                "question_design_version": task.get("question_design_version", QUESTION_DESIGN_VERSION),
                "image_paths": task["image_paths"],
                "used_image_groups": task.get("image_groups"),
                "tool_condition": task.get("tool_condition", "raw"),
                "evidence_qc_status": (task.get("_tool_evidence_record") or {}).get("qc_status"),
                "evidence_qc_filter": evidence_qc_filter,
                "missing_evidence_policy": missing_evidence_policy,
                "gen_model": task["gen_model"],
                "source": task["source"],
                "judge_structured_output": judge_structured_output,
                "closed_form_structured_output": closed_form_structured_output,
                "structured_output_fallback": structured_output_fallback,
                "max_output_tokens": max_output_tokens,
                "judge_max_output_tokens": judge_max_output_tokens,
                "empty_response_policy": empty_response_policy,
                "empty_retry_max_output_tokens": empty_retry_max_output_tokens,
                "reasoning_effort": reasoning_effort,
            }
            print(json.dumps(preview, ensure_ascii=False))
        return pending_tasks

    output_root_path.mkdir(parents=True, exist_ok=True)
    run_config_base = {
        "script_type": "fashion_industry",
        "phase": phase,
        "model": model_name,
        "input_mode": input_mode,
        "output_root": str(output_root_path),
        "question_types": list(question_types),
        "question_design_version": QUESTION_DESIGN_VERSION,
        "prime": "on" if prime else "off",
        "judge_model": judge_model,
        "open_backend": open_backend,
        "server_url": record_server_url,
        "server_model_id": record_server_model_id,
        "judge_provider": judge_provider,
        "judge_server_url": resolved_judge_server_url,
        "server_preflight": server_preflight,
        "server_framework": server_framework,
        "server_vram_settings": {
            "hf_model_id": server_hf_model_id,
            "gpu_ids": server_gpu_ids,
            "dtype": server_dtype,
            "tensor_parallel_size": server_tensor_parallel_size,
            "gpu_memory_utilization": server_gpu_memory_utilization,
            "reserve_gb": server_reserve_gb,
            "load_in_4bit": server_load_in_4bit,
        },
        "judge_server_vram_settings": {
            "hf_model_id": judge_server_hf_model_id,
            "gpu_ids": judge_server_gpu_ids,
            "dtype": judge_server_dtype or server_dtype,
            "tensor_parallel_size": judge_server_tensor_parallel_size or server_tensor_parallel_size,
            "gpu_memory_utilization": judge_server_gpu_memory_utilization or server_gpu_memory_utilization,
            "reserve_gb": judge_server_reserve_gb or server_reserve_gb,
            "load_in_4bit": judge_server_load_in_4bit,
        }
        if judge_provider == "server"
        else None,
        "server_vram_plan": server_vram_plan,
        "judge_server_vram_plan": judge_server_vram_plan,
        "domain": domain,
        "key": key,
        "limit": limit,
        "n_per_cat": n_per_cat,
        "total_tasks": len(tasks),
        "tool_condition": tool_condition,
        **both_raw_policy_fields(input_mode, tool_condition),
        "evidence_manifest": str(evidence_manifest_path) if evidence_manifest_path else None,
        "evidence_qc_filter": evidence_qc_filter,
        "missing_evidence_policy": missing_evidence_policy,
        "image_group_labels": image_group_labels,
        "evidence_filter": evidence_filter_stats,
        "manifest_records_total": evidence_filter_stats.get("manifest_records_total", 0),
        "eligible_task_count": evidence_filter_stats.get("eligible_task_count", len(tasks)),
        "skipped_task_count_due_to_evidence": evidence_filter_stats.get("skipped_task_count_due_to_evidence", 0),
        "max_retries": max_retries,
        "retry_base_seconds": retry_base_seconds,
        "retry_max_seconds": retry_max_seconds,
        "stop_on_quota": stop_on_quota,
        "max_consecutive_errors": max_consecutive_errors,
        "provider_cache": provider_cache,
        "prompt_cache_retention": prompt_cache_retention,
        "cache_key_mode": cache_key_mode,
        "cache_usage_log": cache_usage_log,
        "judge_structured_output": judge_structured_output,
        "closed_form_structured_output": closed_form_structured_output,
        "structured_output_fallback": structured_output_fallback,
        "max_output_tokens": max_output_tokens,
        "judge_max_output_tokens": judge_max_output_tokens,
        "empty_response_policy": empty_response_policy,
        "empty_retry_max_output_tokens": empty_retry_max_output_tokens,
        "reasoning_effort": reasoning_effort,
    }
    resume_run_dir = resolve_resume_run_dir(output_root_path, resume, run_config_base)
    if resume_run_dir is not None:
        run_dir = resume_run_dir
        run_id = run_dir.name
        existing_records = load_jsonl(run_dir / "records.jsonl")
        done_set = {record_signature(record) for record in existing_records if record_is_success(record)}
        pending_tasks = [task for task in tasks if _task_signature(model_name, input_mode, task) not in done_set]
        if limit is not None:
            pending_tasks = pending_tasks[:limit]
        print(f"Resuming run: {run_dir}")
    else:
        run_dir = _run_dir_name(
            phase=phase,
            input_mode=input_mode,
            model_name=model_name,
            question_types=question_types,
            output_root=output_root_path,
        )
        run_id = run_dir.name
        existing_records = []
    run_dir.mkdir(parents=True, exist_ok=True)
    out_file = run_dir / "records.jsonl"
    write_json(run_dir / "run_config.json", {"run_id": run_id, **run_config_base, "resume": resume})
    results = list(existing_records)
    skipped_resume_count = len(tasks) - len(pending_tasks)
    run_status = "completed"
    last_task_sig = None
    last_error_kind = None
    last_error_message = None
    consecutive_errors = 0

    for idx, task in enumerate(pending_tasks, start=1):
        try:
            last_task_sig = _task_signature(model_name, input_mode, task)
            provider_cache_key = build_provider_cache_key(
                benchmark_name="fashion_industry",
                input_mode=input_mode,
                image_paths=task["image_paths"],
                cache_key_mode=cache_key_mode,
            )
            print(
                f"[{idx}/{len(pending_tasks)}] {task['key']} mod{task['mod_id']} "
                f"{task['question_type']} {Path(task['basename']).name}",
                flush=True,
            )
            closed_form_schema = None
            if task["question_type"] == "Q_MC" and closed_form_structured_output != "off":
                closed_form_schema = MC_ANSWER_SCHEMA_ID
            if task["question_type"] == "Q3" and closed_form_structured_output != "off":
                closed_form_schema = YES_NO_ANSWER_SCHEMA_ID
            if task["question_type"] in {"Q1", "Q2"} and closed_form_structured_output != "off":
                closed_form_schema = OPEN_ANSWER_SCHEMA_ID
            set_last_structured_output_info(used=False, schema=closed_form_schema, fallback_reason=None)
            response, error, retry_count, error_kind, fatal_error = query_with_retries(
                lambda: query_vlm(
                    task["image_paths"],
                    task["question"],
                    model_name,
                    local_url=resolved_server_url,
                    provider=provider,
                    open_backend=open_backend,
                    server_model_id=server_model_id,
                    provider_cache=provider_cache,
                    prompt_cache_retention=prompt_cache_retention,
                    cache_key=provider_cache_key,
                    cache_usage_log=cache_usage_log,
                    max_tokens=max_output_tokens,
                    reasoning_effort=reasoning_effort,
                    structured_output_mode=closed_form_structured_output if closed_form_schema else "off",
                    structured_output_schema=closed_form_schema,
                    structured_output_fallback=normalize_structured_fallback(
                        closed_form_structured_output,
                        structured_output_fallback,
                    ),
                ),
                max_retries=max_retries,
                retry_base_seconds=retry_base_seconds,
                retry_max_seconds=retry_max_seconds,
                stop_on_quota=stop_on_quota,
            )
            empty_response = False
            empty_retry_used = False
            empty_retry_success = False
            empty_retry_reason = None
            if not error and not strip_thinking_blocks(response or ""):
                empty_response = True
                if empty_response_policy == "retry_once":
                    empty_retry_used = True
                    empty_retry_reason = "empty_backbone_response"
                    response, error, retry_count_retry, error_kind, fatal_error = query_with_retries(
                        lambda: query_vlm(
                            task["image_paths"],
                            task["question"],
                            model_name,
                            local_url=resolved_server_url,
                            provider=provider,
                            open_backend=open_backend,
                            server_model_id=server_model_id,
                            provider_cache=provider_cache,
                            prompt_cache_retention=prompt_cache_retention,
                            cache_key=provider_cache_key,
                            cache_usage_log=cache_usage_log,
                            max_tokens=empty_retry_max_output_tokens,
                            reasoning_effort=reasoning_effort,
                            structured_output_mode=closed_form_structured_output if closed_form_schema else "off",
                            structured_output_schema=closed_form_schema,
                            structured_output_fallback=normalize_structured_fallback(
                                closed_form_structured_output,
                                structured_output_fallback,
                            ),
                        ),
                        max_retries=max_retries,
                        retry_base_seconds=retry_base_seconds,
                        retry_max_seconds=retry_max_seconds,
                        stop_on_quota=stop_on_quota,
                    )
                    retry_count += retry_count_retry
                    if not error and strip_thinking_blocks(response or ""):
                        empty_response = False
                        empty_retry_success = True
                elif empty_response_policy == "error":
                    error = "Backbone generation returned an empty response."
                    error_kind = "unknown_error"
                    fatal_error = False
            backbone_structured_info = get_last_structured_output_info()
            judge_structured_info = {
                "structured_output_used": False,
                "structured_output_schema": None,
                "structured_output_fallback_reason": None,
            }
            judge_override_applied = False
            judge_override_reason = None
            extracted = extract_mc_letter(response) if task["question_type"] == "Q_MC" else extract_answer(response)

            if error and not response:
                judge_label = "query_error"
                judge_reason = error[:100]
                record_judge_model = "letter_match" if task["question_type"] == "Q_MC" else judge_model
            elif task["question_type"] != "Q_MC" and empty_response:
                error_kind = None
                fatal_error = False
                judge_label = "other"
                judge_reason = "empty response"
                record_judge_model = "empty_response_guard"
            elif task["question_type"] == "Q_MC":
                error_kind = None
                fatal_error = False
                if extracted == task["correct_letter"]:
                    judge_label = "correct"
                elif task["biased_letter"] is not None and extracted == task["biased_letter"]:
                    judge_label = "biased"
                else:
                    judge_label = "other"
                judge_reason = "mc direct match"
                record_judge_model = "letter_match"
            elif task["correct"] is not None:
                judge_label, judge_reason, judge_retry_count, judge_error_kind, judge_fatal_error = llm_judge(
                    question=task["question"],
                    answer=response or "",
                    correct=task["correct"],
                    biased=task["biased"],
                    evaluation_target=task["evaluation_target"],
                    bias_rationale=task["bias_rationale"],
                    judge_model=judge_model,
                    judge_provider=judge_provider,
                    judge_server_url=resolved_judge_server_url,
                    max_retries=max_retries,
                    retry_base_seconds=retry_base_seconds,
                    retry_max_seconds=retry_max_seconds,
                    stop_on_quota=stop_on_quota,
                    judge_max_output_tokens=judge_max_output_tokens,
                    judge_structured_output=judge_structured_output,
                    structured_output_fallback=structured_output_fallback,
                    input_mode=input_mode,
                    tool_condition=tool_condition,
                    both_raw_prompt_policy=task.get("both_raw_prompt_policy"),
                )
                judge_structured_info = get_last_structured_output_info()
                retry_count += judge_retry_count
                if judge_label == "judge_error":
                    error_kind = judge_error_kind or "judge_error"
                    fatal_error = judge_fatal_error
                else:
                    error_kind = None
                    fatal_error = False
                    judge_label, judge_override_applied, judge_override_reason = maybe_override_both_raw_open_judge_label(
                        label=judge_label,
                        answer=response or "",
                        input_mode=input_mode,
                        tool_condition=tool_condition,
                        question_type=task["question_type"],
                    )
                    if judge_override_applied:
                        judge_reason = judge_override_reason
                record_judge_model = judge_model
            else:
                judge_label, judge_reason = None, None
                record_judge_model = judge_model
                error_kind = None
                fatal_error = False
                judge_structured_info = {"structured_output_used": False, "structured_output_schema": None, "structured_output_fallback_reason": None}
            provider_cache_info = get_last_provider_cache_info()
            if task["question_type"] == "Q_MC":
                judge_structured_info = {"structured_output_used": False, "structured_output_schema": None, "structured_output_fallback_reason": None}
            structured_schema = (
                JUDGE_LABEL_SCHEMA_ID
                if task["question_type"] != "Q_MC" and judge_structured_output != "off"
                else closed_form_schema
            )
            structured_target = "judge" if task["question_type"] != "Q_MC" and judge_structured_output != "off" else ("closed_form" if closed_form_schema else "none")
            structured_fallbacks = [
                item.get("structured_output_fallback_reason")
                for item in (backbone_structured_info, judge_structured_info)
                if item and item.get("structured_output_fallback_reason")
            ]
            if reasoning_effort == "provider_default":
                reasoning_control_applied = "provider_default"
                reasoning_fallback_reason = None
            elif model_name.startswith("gemini-"):
                reasoning_control_applied = "thinking_budget=0"
                reasoning_fallback_reason = None
            else:
                reasoning_control_applied = "not_applicable_chat_or_server"
                reasoning_fallback_reason = (
                    None
                    if reasoning_effort == "off"
                    else "native_fashion_runner_does_not_enable_extended_thinking"
                )

            record = {
                "run_id": run_id,
                "model": model_name,
                "input_mode": input_mode,
                "question_types": ",".join(question_types),
                "phase": task["phase"],
                "domain": task["domain"],
                "key": task["key"],
                "mod_id": task["mod_id"],
                "mod_type": task["mod_type"],
                "question_type": task["question_type"],
                "question": task["question"],
                "correct": task["correct"],
                "biased": task["biased"],
                "evaluation_target": task["evaluation_target"],
                "bias_rationale": task["bias_rationale"],
                "prior_statement": task.get("prior_statement"),
                "question_design_version": task.get("question_design_version", QUESTION_DESIGN_VERSION),
                "both_raw_prompt_policy": task.get("both_raw_prompt_policy"),
                "both_raw_scoring_policy": task.get("both_raw_scoring_policy"),
                "mc_options": task["mc_options"],
                "correct_letter": task["correct_letter"],
                "biased_letter": task["biased_letter"],
                "gt": task["gt"],
                "bias": task["bias"],
                "image_paths": task["image_paths"],
                **tool_record_fields(
                    task=task,
                    tool_condition=tool_condition,
                    evidence_manifest_path=evidence_manifest_path,
                    image_groups=task.get("image_groups") or [],
                    evidence_record=task.get("_tool_evidence_record"),
                    evidence_qc_filter=evidence_qc_filter,
                    missing_evidence_policy=missing_evidence_policy,
                ),
                "response": response,
                "error": error,
                "error_kind": error_kind,
                "fatal_error": fatal_error,
                "retry_count": retry_count,
                "empty_response": empty_response,
                "empty_retry_used": empty_retry_used,
                "empty_retry_success": empty_retry_success,
                "empty_retry_reason": empty_retry_reason,
                "backbone_max_output_tokens": max_output_tokens,
                "judge_max_output_tokens": judge_max_output_tokens,
                "reasoning_effort_requested": reasoning_effort,
                "reasoning_control_applied": reasoning_control_applied,
                "reasoning_fallback_reason": reasoning_fallback_reason,
                "resume_skipped": False,
                "judge_label": judge_label,
                "judge_reason": judge_reason,
                "judge_override_applied": judge_override_applied,
                "judge_override_reason": judge_override_reason,
                "judge_model": record_judge_model,
                "open_backend": record_open_backend,
                "server_url": record_server_url,
                "server_model_id": record_server_model_id,
                "judge_provider": judge_provider,
                "judge_server_url": resolved_judge_server_url,
                **provider_cache_info,
                **structured_record_fields(
                    judge_mode=judge_structured_output,
                    closed_form_mode=closed_form_structured_output,
                    used=bool(backbone_structured_info.get("structured_output_used"))
                    or bool(judge_structured_info.get("structured_output_used")),
                    target=structured_target,
                    schema_id=structured_schema,
                    fallback_reason="; ".join(structured_fallbacks) if structured_fallbacks else None,
                ),
                "timestamp": datetime.now().isoformat(),
                "gen_model": task["gen_model"],
                "source": task["source"],
                "original_image": task["original_image"],
                "cf_filename": task["cf_filename"],
                "basename": task["basename"],
                "extracted_answer": extracted,
                "task_signature": list(last_task_sig),
            }
            results.append(record)

            with open(out_file, "a") as handle:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")

            if record_is_error(record):
                consecutive_errors += 1
                last_error_kind = record.get("error_kind")
                last_error_message = record.get("error")
            else:
                consecutive_errors = 0
            write_checkpoint(
                run_dir=run_dir,
                run_id=run_id,
                run_status="running",
                total_tasks=len(tasks),
                records=results,
                skipped_resume_count=skipped_resume_count,
                pending_count=max(0, len(pending_tasks) - idx),
                last_task_signature=last_task_sig,
                last_error_kind=last_error_kind,
                last_error_message=last_error_message,
                evidence_filter_stats=evidence_filter_stats,
            )
            if fatal_error:
                run_status = f"interrupted_{error_kind or 'fatal_error'}"
                break
            if max_consecutive_errors > 0 and consecutive_errors >= max_consecutive_errors:
                run_status = "interrupted_error_guard"
                break
            time.sleep(0.5)
        except KeyboardInterrupt:
            run_status = "interrupted_user"
            break
        except Exception as exc:
            error_kind = classify_error(exc)
            fatal_error = is_fatal_error_kind(error_kind, stop_on_quota=stop_on_quota)
            set_last_provider_cache_info(
                enabled=False,
                cache_key=locals().get("provider_cache_key"),
                reason="task_exception",
                usage={},
            )
            error_row = {
                "run_id": run_id,
                "model": model_name,
                "input_mode": input_mode,
                "question_types": ",".join(question_types),
                "phase": task["phase"],
                "domain": task["domain"],
                "key": task["key"],
                "mod_id": task["mod_id"],
                "mod_type": task["mod_type"],
                "question_type": task["question_type"],
                "question": task["question"],
                "correct": task["correct"],
                "biased": task["biased"],
                "evaluation_target": task["evaluation_target"],
                "bias_rationale": task["bias_rationale"],
                "prior_statement": task.get("prior_statement"),
                "question_design_version": task.get("question_design_version", QUESTION_DESIGN_VERSION),
                "mc_options": task["mc_options"],
                "correct_letter": task["correct_letter"],
                "biased_letter": task["biased_letter"],
                "gt": task["gt"],
                "bias": task["bias"],
                "image_paths": task["image_paths"],
                **tool_record_fields(
                    task=task,
                    tool_condition=tool_condition,
                    evidence_manifest_path=evidence_manifest_path,
                    image_groups=task.get("image_groups") or [],
                    evidence_record=task.get("_tool_evidence_record"),
                    evidence_qc_filter=evidence_qc_filter,
                    missing_evidence_policy=missing_evidence_policy,
                ),
                "response": None,
                "error": f"task_exception: {type(exc).__name__}: {str(exc)[:160]}",
                "error_kind": error_kind,
                "fatal_error": fatal_error,
                "retry_count": 0,
                "resume_skipped": False,
                "judge_label": "query_error",
                "judge_reason": "task exception",
                "judge_model": "letter_match" if task["question_type"] == "Q_MC" else judge_model,
                "open_backend": record_open_backend,
                "server_url": record_server_url,
                "server_model_id": record_server_model_id,
                "judge_provider": judge_provider,
                "judge_server_url": resolved_judge_server_url,
                **get_last_provider_cache_info(),
                **structured_record_fields(
                    judge_mode=judge_structured_output,
                    closed_form_mode=closed_form_structured_output,
                    used=False,
                    target=(
                        "closed_form"
                        if task["question_type"] == "Q_MC" and closed_form_structured_output != "off"
                        else ("judge" if task["question_type"] != "Q_MC" and judge_structured_output != "off" else "none")
                    ),
                    schema_id=(
                        MC_ANSWER_SCHEMA_ID
                        if task["question_type"] == "Q_MC" and closed_form_structured_output != "off"
                        else (JUDGE_LABEL_SCHEMA_ID if task["question_type"] != "Q_MC" and judge_structured_output != "off" else None)
                    ),
                    fallback_reason=get_last_structured_output_info().get("structured_output_fallback_reason"),
                ),
                "timestamp": datetime.now().isoformat(),
                "gen_model": task["gen_model"],
                "source": task["source"],
                "original_image": task["original_image"],
                "cf_filename": task["cf_filename"],
                "basename": task["basename"],
                "extracted_answer": None,
                "task_signature": list(last_task_sig) if isinstance(last_task_sig, tuple) else last_task_sig,
            }
            results.append(error_row)
            with open(out_file, "a") as handle:
                handle.write(json.dumps(error_row, ensure_ascii=False) + "\n")
            print(f"  [ERROR] task {idx}: {type(exc).__name__}: {str(exc)[:100]}")
            consecutive_errors += 1
            last_error_kind = error_kind
            last_error_message = error_row["error"]
            write_checkpoint(
                run_dir=run_dir,
                run_id=run_id,
                run_status="running",
                total_tasks=len(tasks),
                records=results,
                skipped_resume_count=skipped_resume_count,
                pending_count=max(0, len(pending_tasks) - idx),
                last_task_signature=last_task_sig,
                last_error_kind=last_error_kind,
                last_error_message=last_error_message,
                evidence_filter_stats=evidence_filter_stats,
            )
            if fatal_error:
                run_status = f"interrupted_{error_kind}"
                break
            if max_consecutive_errors > 0 and consecutive_errors >= max_consecutive_errors:
                run_status = "interrupted_error_guard"
                break
            continue

    print(f"eval_results={out_file}")
    summary = _summarize_run(
        results,
        run_id=run_id,
        run_status=run_status,
        total_tasks=len(tasks),
        evidence_filter_stats=evidence_filter_stats,
    )
    cache_summary = summary.get("provider_cache") or {}
    print(
        "provider_cache_summary="
        f"enabled={int(cache_summary.get('enabled_count', 0))}/"
        f"{int(cache_summary.get('request_count', 0))} "
        f"usage_rows={int(cache_summary.get('requests_with_usage', 0))}"
    )
    write_json(run_dir / "summary.json", summary)
    write_checkpoint(
        run_dir=run_dir,
        run_id=run_id,
        run_status=run_status,
        total_tasks=len(tasks),
        records=results,
        skipped_resume_count=skipped_resume_count,
        pending_count=max(0, len(tasks) - completed_success_count(results)),
        last_task_signature=last_task_sig,
        last_error_kind=last_error_kind,
        last_error_message=last_error_message,
        evidence_filter_stats=evidence_filter_stats,
    )
    print_readable_summary(summary, run_dir)
    return results


def generate_report(phase=None, input_mode=None, output_root=None):
    """
    Aggregate JSONL eval_results using judge_label.
    """
    output_root = _resolve_output_root(output_root)
    if not output_root.exists():
        print(f"No eval_results directory found: {output_root}")
        return

    records = []
    for path in _result_jsonl_paths(output_root):
        with open(path) as handle:
            for line in handle:
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if phase and record.get("phase") != phase:
                    continue
                if input_mode and record.get("input_mode") != input_mode:
                    continue
                records.append(record)

    if not records:
        print("=" * 72)
        print("Report rows=0")
        print("=" * 72)
        print("\nBy input_mode and judge_label")
        print("\nBy domain")
        print("No matching eval_results found.")
        return

    print("=" * 72)
    print(f"Report rows={len(records)}")
    print(f"output_root={output_root}")
    partial_runs = []
    for run_dir in sorted(path for path in output_root.glob("*") if path.is_dir()):
        checkpoint = load_checkpoint(run_dir)
        if checkpoint and checkpoint.get("run_status") != "completed":
            partial_runs.append((run_dir.name, checkpoint.get("run_status"), checkpoint.get("last_error_kind")))
    if partial_runs:
        print(f"Partial runs={len(partial_runs)}")
        for run_id, run_status, last_error_kind in partial_runs[:10]:
            print(f"- {run_id}: {run_status} last_error_kind={last_error_kind}")
    print("=" * 72)

    models = sorted({record["model"] for record in records})
    for model_name in models:
        model_records = [record for record in records if record["model"] == model_name and record.get("judge_label")]
        if not model_records:
            continue
        labels = {label: 0 for label in ["correct", "biased", "other", "judge_error", "query_error"]}
        for record in model_records:
            label = record.get("judge_label")
            labels[label] = labels.get(label, 0) + 1
        total = len(model_records)
        print(
            f"{model_name:20s} n={total:4d} "
            f"correct={labels.get('correct', 0):4d} "
            f"biased={labels.get('biased', 0):4d} "
            f"other={labels.get('other', 0):4d} "
            f"errors={labels.get('judge_error', 0) + labels.get('query_error', 0):4d}"
        )

    print("\nBy input_mode and judge_label")
    for resolved_input_mode in sorted({record.get("input_mode") for record in records}):
        subset = [record for record in records if record.get("input_mode") == resolved_input_mode and record.get("judge_label")]
        counts = {}
        for record in subset:
            counts[record["judge_label"]] = counts.get(record["judge_label"], 0) + 1
        print(f"{resolved_input_mode}: {counts} (n={len(subset)})")

    print("\nBy Tool Condition")
    for tool_condition in sorted({record.get("tool_condition", "raw") for record in records}):
        subset = [record for record in records if record.get("tool_condition", "raw") == tool_condition and record.get("judge_label")]
        counts = {}
        for record in subset:
            counts[record["judge_label"]] = counts.get(record["judge_label"], 0) + 1
        print(f"{tool_condition}: {counts} (n={len(subset)})")

    print("\nBy domain")
    for domain_name in ("industry", "fashion"):
        subset = [record for record in records if record.get("domain") == domain_name and record.get("judge_label")]
        counts = {}
        for record in subset:
            counts[record["judge_label"]] = counts.get(record["judge_label"], 0) + 1
        print(f"{domain_name}: {counts} (n={len(subset)})")


def _summarize_run(records, *, run_id, run_status, total_tasks, evidence_filter_stats=None):
    labels = {label: 0 for label in ["correct", "biased", "other", "judge_error", "query_error"]}
    for record in records:
        label = record.get("judge_label")
        if label:
            labels[label] = labels.get(label, 0) + 1
    success_count = labels.get("correct", 0) + labels.get("biased", 0) + labels.get("other", 0)
    first = records[0] if records else {}
    overall = summarize_bucket(records)
    return {
        "run_id": run_id,
        "script_type": "fashion_industry",
        "run_status": run_status,
        "is_partial": run_status != "completed" or completed_success_count(records) < total_tasks,
        "phase": first.get("phase", "unknown"),
        "eval_mode": first.get("input_mode", "unknown"),
        "input_mode": first.get("input_mode", "unknown"),
        "model": first.get("model", "unknown"),
        "backbone_model_alias": first.get("model", "unknown"),
        "backbone_model_id": MODEL_IDS.get(first.get("model"), first.get("model", "unknown")),
        "judge_model_alias": first.get("judge_model", "unknown"),
        "judge_model_id": MODEL_IDS.get(first.get("judge_model"), first.get("judge_model", "unknown")),
        "tool_condition": first.get("tool_condition", "raw"),
        "both_raw_prompt_policy": first.get("both_raw_prompt_policy"),
        "both_raw_scoring_policy": first.get("both_raw_scoring_policy"),
        "total_tasks": total_tasks,
        "record_count": len(records),
        "completed_success_count": completed_success_count(records),
        "response_count": success_count,
        "error_count": sum(1 for record in records if record_is_error(record)),
        "labels": labels,
        "accuracy": labels.get("correct", 0) / success_count if success_count else 0.0,
        "bias_rate": labels.get("biased", 0) / success_count if success_count else 0.0,
        "other_rate": labels.get("other", 0) / success_count if success_count else 0.0,
        "overall": overall,
        "by_input_mode": summarize_by_field(records, "input_mode", default="unknown"),
        "by_tool_condition": summarize_by_field(records, "tool_condition", default="raw"),
        "by_question_type": summarize_by_field(records, "question_type", default="unknown"),
        "by_domain": summarize_by_field(records, "domain", default="unknown"),
        "by_source_variant": summarize_by_field(records, "source", default="unknown"),
        "provider_cache": summarize_provider_cache(records),
        "evidence_filter": evidence_filter_stats or {},
    }


def summarize_by_field(records, field, default=None):
    output = {}
    for record in records:
        key = record.get(field, default)
        if key in (None, ""):
            key = default
        output.setdefault(str(key), []).append(record)
    return {name: summarize_bucket(bucket) for name, bucket in sorted(output.items())}


def summarize_bucket(records):
    labels = {label: 0 for label in ["correct", "biased", "other", "judge_error", "query_error"]}
    for record in records:
        label = record.get("judge_label")
        if label:
            labels[label] = labels.get(label, 0) + 1
    response_count = labels.get("correct", 0) + labels.get("biased", 0) + labels.get("other", 0)
    error_count = labels.get("judge_error", 0) + labels.get("query_error", 0) + sum(
        1 for record in records if record.get("error") and record.get("judge_label") not in {"judge_error", "query_error"}
    )
    return {
        "record_count": len(records),
        "response_count": response_count,
        "error_count": error_count,
        "empty_response_count": sum(1 for record in records if record.get("empty_response")),
        "empty_retry_success_count": sum(1 for record in records if record.get("empty_retry_success")),
        "structured_output_fallback_count": sum(
            1 for record in records if record.get("structured_output_fallback_reason")
        ),
        "reasoning_fallback_count": sum(1 for record in records if record.get("reasoning_fallback_reason")),
        "correct_count": labels.get("correct", 0),
        "biased_count": labels.get("biased", 0),
        "other_count": labels.get("other", 0),
        "accuracy": labels.get("correct", 0) / response_count if response_count else 0.0,
        "bias_rate": labels.get("biased", 0) / response_count if response_count else 0.0,
        "other_rate": labels.get("other", 0) / response_count if response_count else 0.0,
        "labels": labels,
    }


def print_readable_summary(summary, run_dir=None):
    if run_dir is not None:
        print()
        print("Run complete" if not summary.get("is_partial") else "Run partial")
    print(f"Run ID: {summary.get('run_id', 'unknown')}")
    print(f"Script Type: {summary.get('script_type', 'unknown')}")
    print(f"Run Status: {summary.get('run_status', 'unknown')} partial={summary.get('is_partial', False)}")
    print(f"Eval Mode: {summary.get('input_mode', summary.get('eval_mode', 'unknown'))}")
    print(
        "Backbone: "
        f"{summary.get('backbone_model_alias', 'unknown')} "
        f"({summary.get('backbone_model_id', 'unknown')})"
    )
    print(
        "Judge: "
        f"{summary.get('judge_model_alias', 'unknown')} "
        f"({summary.get('judge_model_id', 'unknown')})"
    )
    if run_dir is not None:
        print(f"Outputs: {run_dir}")
    cache_summary = summary.get("provider_cache") or {}
    if cache_summary:
        print(
            "Provider Cache: "
            f"enabled={int(cache_summary.get('enabled_count', 0))}/"
            f"{int(cache_summary.get('request_count', 0))} "
            f"usage_rows={int(cache_summary.get('requests_with_usage', 0))}"
        )
    print_bucket_table("Overall", {"overall": summary.get("overall", {})}, hide_header=True)
    print_bucket_table("By Input Mode", summary.get("by_input_mode", {}))
    print_bucket_table("By Tool Condition", summary.get("by_tool_condition", {}))
    print_bucket_table("By Question Type", summary.get("by_question_type", {}))
    print_bucket_table("By Domain", summary.get("by_domain", {}))
    print_bucket_table("By Source Variant", summary.get("by_source_variant", {}))


def print_bucket_table(title, buckets, *, hide_header=False):
    if not buckets:
        return
    print()
    print(title)
    if not hide_header:
        print("Name               Responses  Errors  Accuracy   Bias Rate  Other Rate")
    for name, bucket in buckets.items():
        print_bucket_line(name, bucket)


def print_bucket_line(name, bucket):
    print(
        f"{name:<18} {int(bucket.get('response_count', 0)):>9} "
        f"{int(bucket.get('error_count', 0)):>7} "
        f"{format_percent(float(bucket.get('accuracy', 0.0))):>9} "
        f"{format_percent(float(bucket.get('bias_rate', 0.0))):>10} "
        f"{format_percent(float(bucket.get('other_rate', 0.0))):>10}"
    )


def format_percent(value):
    return f"{value * 100:.2f}%"
    return output


def summarize_provider_cache(records):
    numeric_totals = {}
    usage_count = 0
    enabled_count = 0
    for record in records:
        if record.get("provider_cache_enabled"):
            enabled_count += 1
        usage = record.get("provider_cache_usage")
        if not isinstance(usage, dict) or len(usage) <= 1:
            continue
        usage_count += 1
        for key, value in usage.items():
            if key == "provider" or not isinstance(value, (int, float)):
                continue
            numeric_totals[key] = numeric_totals.get(key, 0.0) + float(value)
    return {
        "request_count": len(records),
        "enabled_count": enabled_count,
        "requests_with_usage": usage_count,
        "numeric_totals": numeric_totals,
    }


def main():
    parser = argparse.ArgumentParser(description="VLM Bias evaluation pipeline")
    parser.add_argument("--phase", choices=["sanity", "cf"], default="cf")
    parser.add_argument("--model", default=None, help="Run a single model name")
    parser.add_argument("--model-set", choices=["closed", "open", "all"], default="all")
    parser.add_argument("--input-mode", choices=["cf_only", "orig_only", "both"], default="cf_only")
    parser.add_argument("--question-types", default="all", help="Comma-separated list like Q1 or Q1,Q3")
    parser.add_argument("--n-per-cat", type=int, default=None, help="Sample N CF images per (key, mod_id)")
    parser.add_argument("--prime", choices=["on", "off"], default="on")
    parser.add_argument("--judge", default="auto")
    parser.add_argument("--provider", choices=["native", "openrouter"], default="native")
    parser.add_argument("--open-backend", choices=["server", "dashscope"], default="server")
    parser.add_argument("--server-url", default=DEFAULT_SERVER_URL)
    parser.add_argument("--server-model-id", default=None)
    parser.add_argument("--judge-provider", choices=["openai", "server", "dashscope"], default="openai")
    parser.add_argument("--judge-server-url", default=None)
    parser.add_argument("--local-url", default=None, help="Deprecated alias for --server-url.")
    add_provider_cache_arguments(parser)
    add_structured_output_arguments(parser)
    add_server_preflight_arguments(parser)
    parser.add_argument("--max-output-tokens", type=int, default=2048, help="Max generated tokens per backbone response.")
    parser.add_argument("--judge-max-output-tokens", type=int, default=512, help="Max generated tokens per judge response.")
    parser.add_argument(
        "--empty-response-policy",
        choices=("retry_once", "other", "error"),
        default="retry_once",
        help="How to handle blank backbone outputs before scoring.",
    )
    parser.add_argument(
        "--empty-retry-max-output-tokens",
        type=int,
        default=2048,
        help="Backbone token cap used for a one-time retry after an empty response.",
    )
    parser.add_argument(
        "--reasoning-effort",
        choices=("off", "minimal", "low", "medium", "high", "provider_default"),
        default="off",
        help="Reasoning/thinking control metadata. Native fashion runner does not enable extended thinking.",
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--report", action="store_true")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=RAW_RESULTS_DIR,
        help="Run family directory. Each run writes a timestamped subdirectory under this root.",
    )
    parser.add_argument("--domain", choices=["industry", "fashion"], default=None)
    parser.add_argument("--key", default=None)
    parser.add_argument("--tool-condition", choices=TOOL_CONDITIONS, default="raw")
    parser.add_argument("--evidence-manifest", type=Path, default=None)
    parser.add_argument("--evidence-qc-filter", choices=EVIDENCE_QC_FILTERS, default="pass")
    parser.add_argument("--missing-evidence-policy", choices=MISSING_EVIDENCE_POLICIES, default="skip")
    parser.add_argument("--image-group-labels", choices=["off", "on"], default="off")
    parser.add_argument("--resume", choices=["off", "latest", "auto"], default="off")
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--retry-base-seconds", type=float, default=2.0)
    parser.add_argument("--retry-max-seconds", type=float, default=60.0)
    parser.add_argument("--stop-on-quota", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max-consecutive-errors", type=int, default=5)
    args = parser.parse_args()

    if args.report:
        generate_report(phase=args.phase, input_mode=args.input_mode, output_root=args.output_root)
        return

    if args.phase == "sanity" and args.question_types == "all":
        question_types = ["Q1", "Q2", "Q3"]
    else:
        question_types = _parse_question_types(args.question_types)
    models = _resolve_models(model=args.model, model_set=args.model_set)
    prime = args.prime == "on"

    for model_name in models:
        run_eval(
            phase=args.phase,
            model_name=model_name,
            input_mode=args.input_mode,
            question_types=question_types,
            n_per_cat=args.n_per_cat,
            prime=prime,
            judge=args.judge,
            local_url=args.local_url,
            provider=args.provider,
            open_backend=args.open_backend,
            server_url=args.server_url,
            server_model_id=args.server_model_id,
            judge_provider=args.judge_provider,
            judge_server_url=args.judge_server_url,
            limit=args.limit,
            dry_run=args.dry_run,
            domain=args.domain,
            key=args.key,
            resume=args.resume,
            max_retries=args.max_retries,
            retry_base_seconds=args.retry_base_seconds,
            retry_max_seconds=args.retry_max_seconds,
            stop_on_quota=args.stop_on_quota,
            max_consecutive_errors=args.max_consecutive_errors,
            provider_cache=args.provider_cache,
            prompt_cache_retention=args.prompt_cache_retention,
            cache_key_mode=args.cache_key_mode,
            cache_usage_log=args.cache_usage_log,
            server_preflight=args.server_preflight,
            server_framework=args.server_framework,
            server_gpu_ids=args.server_gpu_ids,
            server_dtype=args.server_dtype,
            server_tensor_parallel_size=args.server_tensor_parallel_size,
            server_gpu_memory_utilization=args.server_gpu_memory_utilization,
            server_reserve_gb=args.server_reserve_gb,
            server_load_in_4bit=args.server_load_in_4bit,
            server_hf_model_id=args.server_hf_model_id,
            judge_server_gpu_ids=args.judge_server_gpu_ids,
            judge_server_dtype=args.judge_server_dtype,
            judge_server_tensor_parallel_size=args.judge_server_tensor_parallel_size,
            judge_server_gpu_memory_utilization=args.judge_server_gpu_memory_utilization,
            judge_server_reserve_gb=args.judge_server_reserve_gb,
            judge_server_load_in_4bit=args.judge_server_load_in_4bit,
            judge_server_hf_model_id=args.judge_server_hf_model_id,
            judge_structured_output=args.judge_structured_output,
            closed_form_structured_output=args.closed_form_structured_output,
            structured_output_fallback=args.structured_output_fallback,
            max_output_tokens=args.max_output_tokens,
            judge_max_output_tokens=args.judge_max_output_tokens,
            empty_response_policy=args.empty_response_policy,
            empty_retry_max_output_tokens=args.empty_retry_max_output_tokens,
            reasoning_effort=args.reasoning_effort,
            tool_condition=args.tool_condition,
            evidence_manifest=args.evidence_manifest,
            evidence_qc_filter=args.evidence_qc_filter,
            missing_evidence_policy=args.missing_evidence_policy,
            image_group_labels=args.image_group_labels,
            output_root=args.output_root,
        )


if __name__ == "__main__":
    main()
