#!/usr/bin/env python3
from __future__ import annotations

import argparse
import base64
import mimetypes
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

from dotenv import load_dotenv

EVAL_CODE_DIR = Path(__file__).resolve().parents[1]
if str(EVAL_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(EVAL_CODE_DIR))

from provider_cache import (
    add_anthropic_cache_control,
    add_provider_cache_arguments,
    build_provider_cache_key,
    extract_cache_usage,
    openai_cache_kwargs,
    provider_cache_state,
)
from structured_outputs import (
    JUDGE_LABEL_SCHEMA_ID,
    MC_ANSWER_SCHEMA_ID,
    OPEN_ANSWER_SCHEMA_ID,
    YES_NO_ANSWER_SCHEMA_ID,
    anthropic_tool,
    anthropic_tool_choice,
    extract_anthropic_tool_json,
    format_structured_fallback_reason,
    gemini_config_kwargs,
    looks_like_structured_output_unsupported,
    normalize_structured_fallback,
    openai_responses_text_format,
    structured_record_fields,
    strip_thinking_blocks,
)

from eval_common import (
    DEFAULT_QUESTIONS_PATH,
    DEFAULT_RAW_RUNS_ROOT,
    ImagePair,
    NullProgress,
    ProgressBar,
    QuestionSpec,
    PRIOR_PRIME_PREFIX,
    add_common_eval_arguments,
    append_jsonl,
    answers_for_mode,
    build_eval_tasks,
    build_group_specs,
    build_image_groups,
    build_image_pairs,
    build_judge_prompt,
    build_model_prompt,
    build_questions_by_key,
    build_summary,
    build_tool_record_fields,
    build_yes_no_judge_prompt,
    both_raw_policy_fields,
    call_with_retries,
    completed_success_count,
    count_total_evaluations,
    elapsed_ms,
    error_metadata,
    fashion_compatible_record_fields,
    filter_image_pairs_for_evidence,
    filter_groups,
    flatten_image_groups,
    flatten_labeled_image_groups,
    LabeledImage,
    load_benchmark,
    load_evidence_manifest,
    load_jsonl,
    make_run_id,
    parse_judge_response,
    parse_question_types,
    parse_response,
    print_dry_run_preview,
    print_dry_run_task_preview,
    print_evidence_filter_summary,
    print_existing_run_summaries,
    print_readable_summary,
    record_is_error,
    resolve_resume_run_dir,
    resolve_selected_groups,
    score_closed_form,
    scoring_targets,
    seed_everything,
    successful_task_signatures,
    task_signature,
    validate_tool_condition_args,
    write_checkpoint,
    write_json,
)

try:
    from anthropic import Anthropic
except Exception:  # pragma: no cover - import availability depends on env
    Anthropic = None

try:
    from google import genai
    from google.genai import types as genai_types
except Exception:  # pragma: no cover - import availability depends on env
    genai = None
    genai_types = None

try:
    from openai import OpenAI
except Exception:  # pragma: no cover - import availability depends on env
    OpenAI = None


CLOSED_MODEL_ALIASES = {
    "gpt-4.1": "openai:gpt-4.1",
    "gpt-4o-mini": "openai:gpt-4o-mini",
    "gpt-5": "openai:gpt-5",
    "gpt-5.1": "openai:gpt-5.1",
    "sonnet-3.7": "anthropic:claude-3-7-sonnet-20250219",
    "sonnet-4": "anthropic:claude-sonnet-4-20250514",
    "claude-sonnet-4": "anthropic:claude-sonnet-4-20250514",
    "opus-4.1": "anthropic:claude-opus-4-1-20250805",
    "claude-opus-4.1": "anthropic:claude-opus-4-1-20250805",
    "gemini-2.5-pro": "google:gemini-2.5-pro",
    "gemini-3.1-pro-preview": "google:gemini-3.1-pro-preview",
}

PROVIDER_ENV_VARS = {
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "google": "GOOGLE_API_KEY",
}


def openai_responses_supports_temperature(model_id: str) -> bool:
    return not model_id.lower().startswith("gpt-5")


def openai_responses_reasoning_kwargs(model_id: str, reasoning_effort: str = "off") -> tuple[Dict[str, Any], str, Optional[str]]:
    normalized = model_id.lower()
    if reasoning_effort == "provider_default":
        return {}, "provider_default", None
    if not normalized.startswith("gpt-5"):
        return {}, "unsupported", "model_does_not_support_reasoning_control"
    if normalized.startswith("gpt-5.1"):
        effort = "none" if reasoning_effort == "off" else reasoning_effort
        return {"reasoning": {"effort": effort}}, effort, None
    if reasoning_effort == "off":
        return {"reasoning": {"effort": "minimal"}}, "minimal", "gpt-5_requires_minimal_reasoning"
    return {"reasoning": {"effort": reasoning_effort}}, reasoning_effort, None


@dataclass(frozen=True)
class ClosedModelRef:
    alias: str
    provider: str
    model_id: str

    @property
    def canonical(self) -> str:
        return f"{self.provider}:{self.model_id}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate if_exist VLMs on the counterfactual benchmark.")
    parser.add_argument(
        "--questions",
        type=Path,
        default=DEFAULT_QUESTIONS_PATH,
        help="Path to the benchmark question JSON.",
    )
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument(
        "--input-mode",
        dest="eval_mode",
        choices=("cf_only", "both", "orig_only"),
        help="Image input setting aligned with fashion_industry.",
    )
    mode_group.add_argument(
        "--eval-mode",
        dest="eval_mode",
        choices=("cf_only", "both", "orig_only"),
        help="Alias for --input-mode.",
    )
    parser.add_argument(
        "--subset-path",
        type=Path,
        default=None,
        help="Optional dataset subset path: a category dir or leaf dir under dataset/ or cf_dataset/.",
    )
    parser.add_argument("--backbone-model", required=True, help="Closed model alias or provider-prefixed model id.")
    parser.add_argument(
        "--judge-model",
        required=True,
        help="Closed model alias or provider-prefixed model id used to judge yes_no and open responses.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_RAW_RUNS_ROOT,
        help="Root directory for run outputs.",
    )
    parser.add_argument(
        "--dtype",
        choices=("auto", "bfloat16", "float16", "float32"),
        default="auto",
        help="Accepted for CLI parity with the open runner; ignored for API models.",
    )
    parser.add_argument(
        "--load-in-4bit",
        action="store_true",
        help="Accepted for CLI parity with the open runner; ignored for API models.",
    )
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature.")
    parser.add_argument("--max-output-tokens", type=int, default=1024, help="Max generated tokens per backbone response.")
    parser.add_argument("--judge-max-output-tokens", type=int, default=512, help="Max generated tokens per judge response.")
    parser.add_argument(
        "--gemini-thinking-budget",
        type=int,
        default=0,
        help=(
            "Thinking budget for Gemini 2.5 models. Formal runs default to 0 to avoid explicit thinking; "
            "raise --max-output-tokens enough to leave room for final text if this is increased."
        ),
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Optional cap on the number of image pairs evaluated before question expansion.",
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed for deterministic ordering.")
    add_common_eval_arguments(parser)
    add_provider_cache_arguments(parser)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    load_dotenv(override=True)
    seed_everything(args.seed)

    if args.report:
        print_existing_run_summaries([args.output_root])
        return 0

    questions_path = args.questions.resolve()
    if not questions_path.exists():
        raise SystemExit(f"Question file not found: {questions_path}")

    benchmark = load_benchmark(questions_path)
    groups = build_group_specs(benchmark)
    selected_groups = resolve_selected_groups(groups, args.subset_path)
    question_types = parse_question_types(args.question_types)
    selected_groups = filter_groups(
        selected_groups,
        category=args.category,
        subcategory=args.subcategory,
        source_variant=args.source_variant,
        question_types=question_types,
    )
    image_pairs = build_image_pairs(selected_groups)
    if args.max_samples is not None:
        image_pairs = image_pairs[: args.max_samples]
    if not image_pairs:
        raise SystemExit("No image pairs matched the selected dataset slice.")

    backbone_ref = resolve_closed_model_ref(args.backbone_model)
    judge_ref = resolve_closed_model_ref(args.judge_model)
    validate_google_generation_settings(
        refs=(backbone_ref, judge_ref),
        max_output_tokens=args.max_output_tokens,
        gemini_thinking_budget=args.gemini_thinking_budget,
    )
    validate_tool_condition_args(
        tool_condition=args.tool_condition,
        evidence_manifest_path=args.evidence_manifest,
        evidence_qc_filter=args.evidence_qc_filter,
        missing_evidence_policy=args.missing_evidence_policy,
    )
    evidence_manifest = load_evidence_manifest(args.evidence_manifest) if args.evidence_manifest else {}
    unfiltered_pair_count = len(image_pairs)
    image_pairs, evidence_filter_stats = filter_image_pairs_for_evidence(
        image_pairs,
        eval_mode=args.eval_mode,
        tool_condition=args.tool_condition,
        evidence_manifest=evidence_manifest,
        evidence_manifest_path=args.evidence_manifest,
        evidence_qc_filter=args.evidence_qc_filter,
        missing_evidence_policy=args.missing_evidence_policy,
    )
    if not image_pairs:
        raise SystemExit("No image pairs remain after evidence filtering.")
    questions_by_key = build_questions_by_key(selected_groups)
    total_evaluations = count_total_evaluations(image_pairs, questions_by_key)
    all_tasks = build_eval_tasks(image_pairs, questions_by_key)

    if args.dry_run:
        if all_tasks:
            sample_groups, _sample_evidence = build_image_groups(
                pair=all_tasks[0][0],
                eval_mode=args.eval_mode,
                tool_condition=args.tool_condition,
                evidence_manifest=evidence_manifest,
                evidence_manifest_path=args.evidence_manifest,
                missing_evidence_policy=args.missing_evidence_policy,
            )
            sample_paths = flatten_image_groups(sample_groups)
        else:
            sample_paths = []
        sample_cache_key = build_provider_cache_key(
            benchmark_name=benchmark.get("benchmark_name", questions_path.stem),
            input_mode=args.eval_mode,
            image_paths=sample_paths,
            cache_key_mode=args.cache_key_mode,
        )
        print(
            "provider_cache="
            f"{args.provider_cache} retention={args.prompt_cache_retention} "
            f"cache_key_mode={args.cache_key_mode} sample_cache_key={sample_cache_key}"
        )
        print_evidence_filter_summary(
            evidence_filter_stats,
            questions_per_pair=(len(all_tasks) // len(image_pairs)) if image_pairs else None,
        )
        if args.resume != "off":
            resume_run_dir = resolve_resume_run_dir(
                output_root=args.output_root,
                resume=args.resume,
                expected_config={},
            )
            completed = successful_task_signatures(load_jsonl(resume_run_dir / "records.jsonl")) if resume_run_dir else set()
            pending_tasks = [
                (pair, question)
                for pair, question in all_tasks
                if task_signature(
                    pair=pair,
                    question=question,
                    eval_mode=args.eval_mode,
                    tool_condition=args.tool_condition,
                )
                not in completed
            ]
            print_dry_run_task_preview(
                script_type="closed",
                eval_mode=args.eval_mode,
                backbone_alias=backbone_ref.alias,
                judge_alias=judge_ref.alias,
                tasks=pending_tasks,
                total_evaluations=total_evaluations,
                prime=args.prime,
                tool_condition=args.tool_condition,
                evidence_manifest=evidence_manifest,
                evidence_manifest_path=args.evidence_manifest,
                image_group_labels=args.image_group_labels,
                evidence_qc_filter=args.evidence_qc_filter,
                missing_evidence_policy=args.missing_evidence_policy,
                judge_structured_output=args.judge_structured_output,
                closed_form_structured_output=args.closed_form_structured_output,
                structured_output_fallback=args.structured_output_fallback,
            )
            return 0
        print_dry_run_preview(
            script_type="closed",
            eval_mode=args.eval_mode,
            backbone_alias=backbone_ref.alias,
            judge_alias=judge_ref.alias,
            image_pairs=image_pairs,
            questions_by_key=questions_by_key,
            total_evaluations=total_evaluations,
            prime=args.prime,
            tool_condition=args.tool_condition,
            evidence_manifest=evidence_manifest,
            evidence_manifest_path=args.evidence_manifest,
            image_group_labels=args.image_group_labels,
            evidence_qc_filter=args.evidence_qc_filter,
            missing_evidence_policy=args.missing_evidence_policy,
            judge_structured_output=args.judge_structured_output,
            closed_form_structured_output=args.closed_form_structured_output,
            structured_output_fallback=args.structured_output_fallback,
        )
        return 0

    validate_provider_envs({backbone_ref.provider, judge_ref.provider})

    provider_checks = {provider: bool(os.getenv(env_var)) for provider, env_var in PROVIDER_ENV_VARS.items()}

    adapters: Dict[str, ClosedProviderAdapter] = {}

    run_config_base = {
        "script_type": "closed",
        "questions_path": str(questions_path),
        "eval_mode": args.eval_mode,
        "input_mode": args.eval_mode,
        "tool_condition": args.tool_condition,
        **both_raw_policy_fields(args.eval_mode, args.tool_condition),
        "evidence_manifest": str(args.evidence_manifest.resolve()) if args.evidence_manifest else None,
        "evidence_qc_filter": args.evidence_qc_filter,
        "missing_evidence_policy": args.missing_evidence_policy,
        "manifest_pairs_total": evidence_filter_stats["manifest_pairs_total"],
        "unfiltered_pair_count": unfiltered_pair_count,
        "eligible_pair_count": evidence_filter_stats["eligible_pair_count"],
        "skipped_pair_count_due_to_evidence": evidence_filter_stats["skipped_pair_count_due_to_evidence"],
        "skipped_by_qc_status": evidence_filter_stats["skipped_by_qc_status"],
        "skipped_by_missing_tool_condition": evidence_filter_stats["skipped_by_missing_tool_condition"],
        "raw_fallback_pair_count": evidence_filter_stats["raw_fallback_pair_count"],
        "image_group_labels": args.image_group_labels,
        "subset_path": str(args.subset_path.resolve()) if args.subset_path else None,
        "backbone_model_alias": backbone_ref.alias,
        "backbone_model_id": backbone_ref.model_id,
        "judge_model_alias": judge_ref.alias,
        "judge_model_id": judge_ref.model_id,
        "providers": {
            "backbone": backbone_ref.provider,
            "judge": judge_ref.provider,
        },
        "provider_env_checks": provider_checks,
        "dtype": args.dtype,
        "load_in_4bit": bool(args.load_in_4bit),
        "temperature": args.temperature,
        "max_output_tokens": args.max_output_tokens,
        "judge_max_output_tokens": args.judge_max_output_tokens,
        "empty_response_policy": args.empty_response_policy,
        "empty_retry_max_output_tokens": args.empty_retry_max_output_tokens,
        "reasoning_effort": args.reasoning_effort,
        "gemini_thinking_budget": args.gemini_thinking_budget,
        "max_samples": args.max_samples,
        "seed": args.seed,
        "prime": args.prime,
        "prompt_prefix": PRIOR_PRIME_PREFIX if args.prime == "on" else None,
        "category": args.category,
        "subcategory": args.subcategory,
        "source_variant": args.source_variant,
        "question_types": question_types,
        "pair_count": len(image_pairs),
        "response_target_count": total_evaluations,
        "max_retries": args.max_retries,
        "retry_base_seconds": args.retry_base_seconds,
        "retry_max_seconds": args.retry_max_seconds,
        "stop_on_quota": args.stop_on_quota,
        "max_consecutive_errors": args.max_consecutive_errors,
        "provider_cache": args.provider_cache,
        "prompt_cache_retention": args.prompt_cache_retention,
        "cache_key_mode": args.cache_key_mode,
        "cache_usage_log": args.cache_usage_log,
        "judge_structured_output": args.judge_structured_output,
        "closed_form_structured_output": args.closed_form_structured_output,
        "structured_output_fallback": normalize_structured_fallback(
            args.judge_structured_output,
            args.structured_output_fallback,
        ),
    }
    resume_run_dir = resolve_resume_run_dir(
        output_root=args.output_root,
        resume=args.resume,
        expected_config=run_config_base,
    )
    if resume_run_dir is not None:
        run_dir = resume_run_dir
        run_id = run_dir.name
        existing_records = load_jsonl(run_dir / "records.jsonl")
        print(f"Resuming run: {run_dir}")
    else:
        run_id = make_run_id()
        run_dir = (args.output_root.resolve() / run_id)
        run_dir.mkdir(parents=True, exist_ok=True)
        existing_records = []

    run_config = {"run_id": run_id, **run_config_base, "resume": args.resume}

    records_path = run_dir / "records.jsonl"
    records: List[Dict[str, Any]] = list(existing_records)
    completed_signatures = successful_task_signatures(existing_records)
    pending_tasks = [
        (pair, question)
        for pair, question in all_tasks
        if task_signature(
            pair=pair,
            question=question,
            eval_mode=args.eval_mode,
            tool_condition=args.tool_condition,
        )
        not in completed_signatures
    ]
    skipped_resume_count = len(all_tasks) - len(pending_tasks)

    write_json(run_dir / "run_config.json", run_config)

    print(
        f"Starting closed-model evaluation: {len(pending_tasks)} pending responses "
        f"({skipped_resume_count} skipped) across {len(image_pairs)} image pairs."
    )
    progress = NullProgress() if args.no_progress else ProgressBar(total=len(pending_tasks), description="Evaluating")
    run_status = "completed"
    last_task_sig = None
    last_error_kind = None
    last_error_message = None
    consecutive_errors = 0
    try:
        for pair, question in pending_tasks:
            last_task_sig = task_signature(
                pair=pair,
                question=question,
                eval_mode=args.eval_mode,
                tool_condition=args.tool_condition,
            )
            record = evaluate_question(
                pair=pair,
                question=question,
                eval_mode=args.eval_mode,
                backbone_ref=backbone_ref,
                judge_ref=judge_ref,
                adapters=adapters,
                temperature=args.temperature,
                max_output_tokens=args.max_output_tokens,
                judge_max_output_tokens=args.judge_max_output_tokens,
                gemini_thinking_budget=args.gemini_thinking_budget,
                reasoning_effort=args.reasoning_effort,
                empty_response_policy=args.empty_response_policy,
                empty_retry_max_output_tokens=args.empty_retry_max_output_tokens,
                run_id=run_id,
                prime=args.prime,
                max_retries=args.max_retries,
                retry_base_seconds=args.retry_base_seconds,
                retry_max_seconds=args.retry_max_seconds,
                stop_on_quota=args.stop_on_quota,
                provider_cache=args.provider_cache,
                prompt_cache_retention=args.prompt_cache_retention,
                cache_key_mode=args.cache_key_mode,
                cache_usage_log=args.cache_usage_log,
                judge_structured_output=args.judge_structured_output,
                closed_form_structured_output=args.closed_form_structured_output,
                structured_output_fallback=args.structured_output_fallback,
                benchmark_name=benchmark.get("benchmark_name", questions_path.stem),
                tool_condition=args.tool_condition,
                evidence_manifest=evidence_manifest,
                evidence_manifest_path=args.evidence_manifest,
                image_group_labels=args.image_group_labels,
                evidence_qc_filter=args.evidence_qc_filter,
                missing_evidence_policy=args.missing_evidence_policy,
            )
            record["task_signature"] = last_task_sig
            record["resume_skipped"] = False
            records.append(record)
            append_jsonl(records_path, record)
            if record_is_error(record):
                consecutive_errors += 1
                last_error_kind = record.get("error_kind")
                last_error_message = record.get("error_message")
            else:
                consecutive_errors = 0
            remaining = max(0, len(pending_tasks) - (len(records) - len(existing_records)))
            write_checkpoint(
                run_dir=run_dir,
                run_id=run_id,
                run_status="running",
                total_tasks=total_evaluations,
                records=records,
                skipped_resume_count=skipped_resume_count,
                pending_count=remaining,
                last_task_signature=last_task_sig,
                last_error_kind=last_error_kind,
                last_error_message=last_error_message,
                evidence_filter_stats=evidence_filter_stats,
            )
            progress.update(suffix=f"{pair.subcategory} | {question.question_type}")
            if record.get("fatal_error"):
                run_status = f"interrupted_{record.get('error_kind') or 'fatal_error'}"
                break
            if args.max_consecutive_errors > 0 and consecutive_errors >= args.max_consecutive_errors:
                run_status = "interrupted_error_guard"
                break
    except KeyboardInterrupt:
        run_status = "interrupted_user"
    finally:
        progress.close()

    summary = build_summary(
        records=records,
        run_id=run_id,
        script_type="closed",
        eval_mode=args.eval_mode,
        backbone_ref=backbone_ref,
        judge_ref=judge_ref,
        run_status=run_status,
        is_partial=run_status != "completed" or completed_success_count(records) < total_evaluations,
    )
    summary["last_error_kind"] = last_error_kind
    summary["evidence_filter"] = evidence_filter_stats
    write_json(run_dir / "summary.json", summary)
    write_checkpoint(
        run_dir=run_dir,
        run_id=run_id,
        run_status=run_status,
        total_tasks=total_evaluations,
        records=records,
        skipped_resume_count=skipped_resume_count,
        pending_count=max(0, total_evaluations - completed_success_count(records)),
        last_task_signature=last_task_sig,
        last_error_kind=last_error_kind,
        last_error_message=last_error_message,
        evidence_filter_stats=evidence_filter_stats,
    )
    print_readable_summary(summary, run_dir)
    return 0 if run_status == "completed" else 1


def resolve_closed_model_ref(raw_value: str) -> ClosedModelRef:
    canonical = CLOSED_MODEL_ALIASES.get(raw_value, raw_value)
    if ":" not in canonical:
        raise SystemExit(
            f"Closed model '{raw_value}' must be one of {sorted(CLOSED_MODEL_ALIASES)} "
            "or a provider-prefixed id like openai:gpt-4.1"
        )
    provider, model_id = canonical.split(":", 1)
    if provider not in PROVIDER_ENV_VARS:
        raise SystemExit(f"Unsupported closed-model provider '{provider}'.")
    return ClosedModelRef(alias=raw_value, provider=provider, model_id=model_id)


def validate_google_generation_settings(
    *,
    refs: Sequence[ClosedModelRef],
    max_output_tokens: int,
    gemini_thinking_budget: int,
) -> None:
    google_25_models = [ref for ref in refs if ref.provider == "google" and ref.model_id.startswith("gemini-2.5")]
    if not google_25_models:
        return
    minimum_cap = 512 if gemini_thinking_budget < 0 else gemini_thinking_budget + 256
    if max_output_tokens < minimum_cap:
        model_list = ", ".join(ref.model_id for ref in google_25_models)
        raise SystemExit(
            "Gemini 2.5 models use thinking tokens before final text. "
            f"Current --max-output-tokens={max_output_tokens} is too low for {model_list} "
            f"with --gemini-thinking-budget={gemini_thinking_budget}. "
            f"Use --max-output-tokens {minimum_cap} or higher."
        )


def validate_provider_envs(providers: Iterable[str]) -> None:
    missing = [PROVIDER_ENV_VARS[provider] for provider in providers if not os.getenv(PROVIDER_ENV_VARS[provider])]
    if missing:
        raise SystemExit("Missing required API variables:\n- " + "\n- ".join(sorted(missing)))


def maybe_add_openai_responses_structured_output(
    request_kwargs: Dict[str, Any],
    *,
    mode: str,
    schema_id: Optional[str],
) -> Dict[str, Any]:
    if mode == "off" or not schema_id:
        return request_kwargs
    updated = dict(request_kwargs)
    updated["text"] = openai_responses_text_format(schema_id)
    return updated


def maybe_add_anthropic_structured_output(
    request_kwargs: Dict[str, Any],
    *,
    mode: str,
    schema_id: Optional[str],
) -> Dict[str, Any]:
    if mode == "off" or not schema_id:
        return request_kwargs
    updated = dict(request_kwargs)
    updated["tools"] = [anthropic_tool(schema_id)]
    updated["tool_choice"] = anthropic_tool_choice()
    return updated


def strip_structured_request(request: Any) -> Any:
    if not isinstance(request, dict):
        return request
    updated = dict(request)
    for key in ("text", "tools", "tool_choice", "response_format"):
        updated.pop(key, None)
    return updated


class ClosedProviderAdapter:
    def __init__(self, model_ref: ClosedModelRef) -> None:
        self.model_ref = model_ref
        self.provider = model_ref.provider
        self.model_id = model_ref.model_id
        self.client = self._build_client()
        self.last_retry_count = 0
        self.last_provider_cache_enabled = False
        self.last_provider_cache_key = None
        self.last_provider_cache_usage: Dict[str, Any] = {}
        self.last_provider_cache_reason = None
        self.last_structured_output_used = False
        self.last_structured_output_schema = None
        self.last_structured_output_fallback_reason = None
        self.last_reasoning_control_applied = "none"
        self.last_reasoning_fallback_reason = None

    def _build_client(self) -> Any:
        if self.provider == "openai":
            if OpenAI is None:
                raise RuntimeError("openai package is not installed.")
            return OpenAI(api_key=os.getenv(PROVIDER_ENV_VARS["openai"]))
        if self.provider == "anthropic":
            if Anthropic is None:
                raise RuntimeError("anthropic package is not installed.")
            return Anthropic(api_key=os.getenv(PROVIDER_ENV_VARS["anthropic"]))
        if self.provider == "google":
            if genai is None:
                raise RuntimeError("google-genai package is not installed.")
            return genai.Client(api_key=os.getenv(PROVIDER_ENV_VARS["google"]))
        raise RuntimeError(f"Unsupported provider: {self.provider}")

    def generate(
        self,
        prompt: str,
        image_paths: Sequence[Path | LabeledImage],
        *,
        temperature: float,
        max_output_tokens: int,
        gemini_thinking_budget: int,
        max_retries: int,
        retry_base_seconds: float,
        retry_max_seconds: float,
        stop_on_quota: bool,
        provider_cache: str = "off",
        prompt_cache_retention: str = "in_memory",
        cache_key: Optional[str] = None,
        cache_usage_log: bool = True,
        structured_output_mode: str = "off",
        structured_output_schema: Optional[str] = None,
        structured_output_fallback: str = "retry_plain",
        reasoning_effort: str = "off",
    ) -> str:
        self.last_provider_cache_enabled = False
        self.last_provider_cache_key = cache_key
        self.last_provider_cache_usage = {}
        self.last_provider_cache_reason = None
        self.last_structured_output_used = False
        self.last_structured_output_schema = structured_output_schema
        self.last_structured_output_fallback_reason = None
        self.last_reasoning_control_applied = "none"
        self.last_reasoning_fallback_reason = None
        if self.provider == "openai":
            return self._generate_openai(
                prompt,
                image_paths,
                temperature=temperature,
                max_output_tokens=max_output_tokens,
                max_retries=max_retries,
                retry_base_seconds=retry_base_seconds,
                retry_max_seconds=retry_max_seconds,
                stop_on_quota=stop_on_quota,
                provider_cache=provider_cache,
                prompt_cache_retention=prompt_cache_retention,
                cache_key=cache_key,
                cache_usage_log=cache_usage_log,
                structured_output_mode=structured_output_mode,
                structured_output_schema=structured_output_schema,
                structured_output_fallback=structured_output_fallback,
                reasoning_effort=reasoning_effort,
            )
        if self.provider == "anthropic":
            return self._generate_anthropic(
                prompt,
                image_paths,
                temperature=temperature,
                max_output_tokens=max_output_tokens,
                max_retries=max_retries,
                retry_base_seconds=retry_base_seconds,
                retry_max_seconds=retry_max_seconds,
                stop_on_quota=stop_on_quota,
                provider_cache=provider_cache,
                prompt_cache_retention=prompt_cache_retention,
                cache_key=cache_key,
                cache_usage_log=cache_usage_log,
                structured_output_mode=structured_output_mode,
                structured_output_schema=structured_output_schema,
                structured_output_fallback=structured_output_fallback,
                reasoning_effort=reasoning_effort,
            )
        if self.provider == "google":
            return self._generate_google(
                prompt,
                image_paths,
                temperature=temperature,
                max_output_tokens=max_output_tokens,
                gemini_thinking_budget=gemini_thinking_budget,
                max_retries=max_retries,
                retry_base_seconds=retry_base_seconds,
                retry_max_seconds=retry_max_seconds,
                stop_on_quota=stop_on_quota,
                provider_cache=provider_cache,
                prompt_cache_retention=prompt_cache_retention,
                cache_key=cache_key,
                cache_usage_log=cache_usage_log,
                structured_output_mode=structured_output_mode,
                structured_output_schema=structured_output_schema,
                structured_output_fallback=structured_output_fallback,
                reasoning_effort=reasoning_effort,
            )
        raise RuntimeError(f"Unsupported provider: {self.provider}")

    def _generate_openai(
        self,
        prompt: str,
        image_paths: Sequence[Path | LabeledImage],
        *,
        temperature: float,
        max_output_tokens: int,
        max_retries: int,
        retry_base_seconds: float,
        retry_max_seconds: float,
        stop_on_quota: bool,
        provider_cache: str,
        prompt_cache_retention: str,
        cache_key: Optional[str],
        cache_usage_log: bool,
        structured_output_mode: str,
        structured_output_schema: Optional[str],
        structured_output_fallback: str,
        reasoning_effort: str,
    ) -> str:
        cache_state = provider_cache_state(
            provider=self.provider,
            provider_cache=provider_cache if image_paths else "off",
            model_id=self.model_id,
            prompt_cache_retention=prompt_cache_retention,
        )
        self.last_provider_cache_enabled = bool(cache_state.get("enabled"))
        self.last_provider_cache_reason = cache_state.get("reason")
        content: List[Dict[str, Any]] = []
        for item in image_paths:
            label = image_item_label(item)
            if label:
                content.append({"type": "input_text", "text": label})
            path = image_item_path(item)
            content.append({"type": "input_image", "image_url": path_to_data_url(path)})
        content.append({"type": "input_text", "text": prompt})
        request_kwargs = {
            "model": self.model_id,
            "input": [{"role": "user", "content": content}],
            "max_output_tokens": max_output_tokens,
        }
        if openai_responses_supports_temperature(self.model_id):
            request_kwargs["temperature"] = temperature
        reasoning_kwargs, reasoning_applied, reasoning_fallback = openai_responses_reasoning_kwargs(
            self.model_id,
            reasoning_effort=reasoning_effort,
        )
        self.last_reasoning_control_applied = reasoning_applied
        self.last_reasoning_fallback_reason = reasoning_fallback
        request_kwargs.update(reasoning_kwargs)
        request_kwargs.update(
            openai_cache_kwargs(
                cache_state=cache_state,
                cache_key=cache_key,
                model_id=self.model_id,
                prompt_cache_retention=prompt_cache_retention,
            )
        )
        request_kwargs = maybe_add_openai_responses_structured_output(
            request_kwargs,
            mode=structured_output_mode,
            schema_id=structured_output_schema,
        )
        response, self.last_retry_count = call_with_retries(
            lambda: self._call_with_structured_fallback(
                lambda kwargs: self.client.responses.create(**kwargs),
                request_kwargs,
                structured_output_mode,
                structured_output_schema,
                structured_output_fallback,
            ),
            max_retries=max_retries,
            retry_base_seconds=retry_base_seconds,
            retry_max_seconds=retry_max_seconds,
            stop_on_quota=stop_on_quota,
        )
        if cache_usage_log:
            self.last_provider_cache_usage = extract_cache_usage(self.provider, response)
        return extract_openai_text(response)

    def _generate_anthropic(
        self,
        prompt: str,
        image_paths: Sequence[Path | LabeledImage],
        *,
        temperature: float,
        max_output_tokens: int,
        max_retries: int,
        retry_base_seconds: float,
        retry_max_seconds: float,
        stop_on_quota: bool,
        provider_cache: str,
        prompt_cache_retention: str,
        cache_key: Optional[str],
        cache_usage_log: bool,
        structured_output_mode: str,
        structured_output_schema: Optional[str],
        structured_output_fallback: str,
        reasoning_effort: str,
    ) -> str:
        if reasoning_effort != "provider_default":
            self.last_reasoning_control_applied = "off"
        cache_state = provider_cache_state(
            provider=self.provider,
            provider_cache=provider_cache if image_paths else "off",
            model_id=self.model_id,
            prompt_cache_retention=prompt_cache_retention,
        )
        self.last_provider_cache_enabled = bool(cache_state.get("enabled"))
        self.last_provider_cache_reason = cache_state.get("reason")
        content: List[Dict[str, Any]] = []
        for item in image_paths:
            label = image_item_label(item)
            if label:
                content.append({"type": "text", "text": label})
            path = image_item_path(item)
            content.append(
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": detect_mime_type(path),
                        "data": base64.b64encode(path.read_bytes()).decode("utf-8"),
                    },
                }
            )
        add_anthropic_cache_control(content, enabled=bool(cache_state.get("enabled")))
        content.append({"type": "text", "text": prompt})
        request_kwargs = {
            "model": self.model_id,
            "max_tokens": max_output_tokens,
            "temperature": temperature,
            "messages": [{"role": "user", "content": content}],
        }
        request_kwargs = maybe_add_anthropic_structured_output(
            request_kwargs,
            mode=structured_output_mode,
            schema_id=structured_output_schema,
        )
        response, self.last_retry_count = call_with_retries(
            lambda: self._call_with_structured_fallback(
                lambda kwargs: self.client.messages.create(**kwargs),
                request_kwargs,
                structured_output_mode,
                structured_output_schema,
                structured_output_fallback,
            ),
            max_retries=max_retries,
            retry_base_seconds=retry_base_seconds,
            retry_max_seconds=retry_max_seconds,
            stop_on_quota=stop_on_quota,
        )
        if cache_usage_log:
            self.last_provider_cache_usage = extract_cache_usage(self.provider, response)
        structured_json = extract_anthropic_tool_json(response)
        if structured_json is not None:
            return structured_json
        parts = [block.text for block in response.content if getattr(block, "type", None) == "text"]
        return "\n".join(parts).strip()

    def _generate_google(
        self,
        prompt: str,
        image_paths: Sequence[Path | LabeledImage],
        *,
        temperature: float,
        max_output_tokens: int,
        gemini_thinking_budget: int,
        max_retries: int,
        retry_base_seconds: float,
        retry_max_seconds: float,
        stop_on_quota: bool,
        provider_cache: str,
        prompt_cache_retention: str,
        cache_key: Optional[str],
        cache_usage_log: bool,
        structured_output_mode: str,
        structured_output_schema: Optional[str],
        structured_output_fallback: str,
        reasoning_effort: str,
    ) -> str:
        if genai_types is None:
            raise RuntimeError("google-genai types are not available.")
        cache_state = provider_cache_state(
            provider=self.provider,
            provider_cache=provider_cache if image_paths else "off",
            model_id=self.model_id,
            prompt_cache_retention=prompt_cache_retention,
        )
        self.last_provider_cache_enabled = bool(cache_state.get("enabled"))
        self.last_provider_cache_reason = cache_state.get("reason")
        parts: List[Any] = []
        for item in image_paths:
            label = image_item_label(item)
            if label:
                parts.append(genai_types.Part.from_text(text=label))
            path = image_item_path(item)
            parts.append(
                genai_types.Part.from_bytes(
                    data=path.read_bytes(),
                    mime_type=detect_mime_type(path),
                )
            )
        parts.append(genai_types.Part.from_text(text=prompt))
        config_kwargs: Dict[str, Any] = {
            "temperature": temperature,
            "max_output_tokens": max_output_tokens,
        }
        if self.model_id.startswith("gemini-2.5") and hasattr(genai_types, "ThinkingConfig"):
            if reasoning_effort == "provider_default":
                self.last_reasoning_control_applied = "provider_default"
            else:
                self.last_reasoning_control_applied = f"thinking_budget={gemini_thinking_budget}"
            config_kwargs["thinking_config"] = genai_types.ThinkingConfig(
                thinking_budget=gemini_thinking_budget
            )
        elif reasoning_effort != "provider_default":
            self.last_reasoning_control_applied = "unsupported"
            self.last_reasoning_fallback_reason = "model_does_not_support_thinking_config"
        if structured_output_mode != "off" and structured_output_schema:
            config_kwargs.update(gemini_config_kwargs(structured_output_schema))
        config = genai_types.GenerateContentConfig(**config_kwargs)
        plain_config_kwargs = dict(config_kwargs)
        plain_config_kwargs.pop("response_mime_type", None)
        plain_config_kwargs.pop("response_json_schema", None)
        plain_config = genai_types.GenerateContentConfig(**plain_config_kwargs)
        response, self.last_retry_count = call_with_retries(
            lambda: self._call_with_structured_fallback(
                lambda request_config: self.client.models.generate_content(
                    model=self.model_id,
                    contents=parts,
                    config=request_config,
                ),
                config,
                structured_output_mode,
                structured_output_schema,
                structured_output_fallback,
                plain_request=plain_config,
            ),
            max_retries=max_retries,
            retry_base_seconds=retry_base_seconds,
            retry_max_seconds=retry_max_seconds,
            stop_on_quota=stop_on_quota,
        )
        if cache_usage_log:
            self.last_provider_cache_usage = extract_cache_usage(self.provider, response)
        text = getattr(response, "text", None)
        if text:
            return text.strip()
        candidates = getattr(response, "candidates", None) or []
        for candidate in candidates:
            content = getattr(candidate, "content", None)
            if not content:
                continue
            response_parts = getattr(content, "parts", None) or []
            texts = [
                getattr(part, "text", None)
                for part in response_parts
                if getattr(part, "text", None) and not getattr(part, "thought", False)
            ]
            if texts:
                return "\n".join(texts).strip()
        raise RuntimeError(f"Google response did not contain text output. {google_response_diagnostics(response)}")

    def _call_with_structured_fallback(
        self,
        caller: Any,
        structured_request: Any,
        mode: str,
        schema_id: Optional[str],
        fallback: str,
        *,
        plain_request: Optional[Any] = None,
    ) -> Any:
        if mode == "off" or not schema_id:
            return caller(plain_request if plain_request is not None else structured_request)
        try:
            response = caller(structured_request)
            self.last_structured_output_used = True
            return response
        except Exception as exc:
            fallback_mode = normalize_structured_fallback(mode, fallback)
            if fallback_mode == "retry_plain" and looks_like_structured_output_unsupported(exc):
                self.last_structured_output_used = False
                self.last_structured_output_fallback_reason = format_structured_fallback_reason(exc)
                return caller(plain_request if plain_request is not None else strip_structured_request(structured_request))
            raise


def get_adapter(cache: Dict[str, ClosedProviderAdapter], model_ref: ClosedModelRef) -> ClosedProviderAdapter:
    adapter = cache.get(model_ref.canonical)
    if adapter is None:
        adapter = ClosedProviderAdapter(model_ref)
        cache[model_ref.canonical] = adapter
    return adapter


def image_item_path(item: Path | LabeledImage) -> Path:
    return item.path if isinstance(item, LabeledImage) else item


def image_item_label(item: Path | LabeledImage) -> Optional[str]:
    return item.label if isinstance(item, LabeledImage) else None


def evaluate_question(
    *,
    pair: ImagePair,
    question: QuestionSpec,
    eval_mode: str,
    backbone_ref: ClosedModelRef,
    judge_ref: ClosedModelRef,
    adapters: Dict[str, ClosedProviderAdapter],
    temperature: float,
    max_output_tokens: int,
    judge_max_output_tokens: int,
    gemini_thinking_budget: int,
    reasoning_effort: str,
    empty_response_policy: str,
    empty_retry_max_output_tokens: int,
    run_id: str,
    prime: str,
    max_retries: int,
    retry_base_seconds: float,
    retry_max_seconds: float,
    stop_on_quota: bool,
    provider_cache: str,
    prompt_cache_retention: str,
    cache_key_mode: str,
    cache_usage_log: bool,
    judge_structured_output: str,
    closed_form_structured_output: str,
    structured_output_fallback: str,
    benchmark_name: str,
    tool_condition: str,
    evidence_manifest: Dict[tuple[str, str, str], Dict[str, Any]],
    evidence_manifest_path: Optional[Path],
    image_group_labels: str,
    evidence_qc_filter: str,
    missing_evidence_policy: str,
) -> Dict[str, Any]:
    prompt = build_model_prompt(
        question=question,
        prime=prime,
        eval_mode=eval_mode,
        tool_condition=tool_condition,
    )
    image_groups, evidence_record = build_image_groups(
        pair=pair,
        eval_mode=eval_mode,
        tool_condition=tool_condition,
        evidence_manifest=evidence_manifest,
        evidence_manifest_path=evidence_manifest_path,
        missing_evidence_policy=missing_evidence_policy,
    )
    used_image_paths = flatten_image_groups(image_groups)
    if eval_mode == "both" and tool_condition == "raw":
        image_group_labels = "off"
    labeled_images = flatten_labeled_image_groups(image_groups, image_group_labels=image_group_labels)
    tool_record_fields = build_tool_record_fields(
        tool_condition=tool_condition,
        evidence_manifest_path=evidence_manifest_path,
        image_groups=image_groups,
        evidence_record=evidence_record,
        evidence_qc_filter=evidence_qc_filter,
        missing_evidence_policy=missing_evidence_policy,
    )
    provider_cache_key = build_provider_cache_key(
        benchmark_name=benchmark_name,
        input_mode=eval_mode,
        image_paths=used_image_paths,
        cache_key_mode=cache_key_mode,
    )
    answers = answers_for_mode(question, eval_mode, tool_condition)
    ground_truth_target, biased_target = scoring_targets(question.question_type, answers)
    backbone_adapter = get_adapter(adapters, backbone_ref)
    closed_form_schema = None
    if question.question_type == "multiple_choice" and closed_form_structured_output != "off":
        closed_form_schema = MC_ANSWER_SCHEMA_ID
    elif question.question_type == "yes_no" and closed_form_structured_output != "off":
        closed_form_schema = YES_NO_ANSWER_SCHEMA_ID
    elif question.question_type == "open" and closed_form_structured_output != "off":
        closed_form_schema = OPEN_ANSWER_SCHEMA_ID

    started = time.perf_counter()
    empty_response = False
    empty_retry_used = False
    empty_retry_success = False
    empty_retry_reason = None
    try:
        raw_response = backbone_adapter.generate(
            prompt,
            labeled_images,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            gemini_thinking_budget=gemini_thinking_budget,
            reasoning_effort=reasoning_effort,
            max_retries=max_retries,
            retry_base_seconds=retry_base_seconds,
            retry_max_seconds=retry_max_seconds,
            stop_on_quota=stop_on_quota,
            provider_cache=provider_cache,
            prompt_cache_retention=prompt_cache_retention,
            cache_key=provider_cache_key,
            cache_usage_log=cache_usage_log,
            structured_output_mode=closed_form_structured_output if closed_form_schema else "off",
            structured_output_schema=closed_form_schema,
            structured_output_fallback=normalize_structured_fallback(
                closed_form_structured_output,
                structured_output_fallback,
            ),
        )
        if not strip_thinking_blocks(raw_response):
            empty_response = True
            if empty_response_policy == "retry_once":
                empty_retry_used = True
                empty_retry_reason = "empty_backbone_response"
                retry_response = backbone_adapter.generate(
                    prompt,
                    labeled_images,
                    temperature=temperature,
                    max_output_tokens=empty_retry_max_output_tokens,
                    gemini_thinking_budget=gemini_thinking_budget,
                    reasoning_effort=reasoning_effort,
                    max_retries=max_retries,
                    retry_base_seconds=retry_base_seconds,
                    retry_max_seconds=retry_max_seconds,
                    stop_on_quota=stop_on_quota,
                    provider_cache=provider_cache,
                    prompt_cache_retention=prompt_cache_retention,
                    cache_key=provider_cache_key,
                    cache_usage_log=cache_usage_log,
                    structured_output_mode=closed_form_structured_output if closed_form_schema else "off",
                    structured_output_schema=closed_form_schema,
                    structured_output_fallback=normalize_structured_fallback(
                        closed_form_structured_output,
                        structured_output_fallback,
                    ),
                )
                raw_response = retry_response
                if strip_thinking_blocks(retry_response):
                    empty_response = False
                    empty_retry_success = True
            elif empty_response_policy == "error":
                raise RuntimeError("Backbone generation returned an empty response.")
        backbone_retry_count = getattr(backbone_adapter, "last_retry_count", 0)
        provider_cache_enabled = bool(getattr(backbone_adapter, "last_provider_cache_enabled", False))
        provider_cache_reason = getattr(backbone_adapter, "last_provider_cache_reason", None)
        provider_cache_usage = getattr(backbone_adapter, "last_provider_cache_usage", {}) if cache_usage_log else {}
        backbone_structured_used = bool(getattr(backbone_adapter, "last_structured_output_used", False))
        backbone_structured_fallback_reason = getattr(backbone_adapter, "last_structured_output_fallback_reason", None)
        reasoning_control_applied = getattr(backbone_adapter, "last_reasoning_control_applied", "none")
        reasoning_fallback_reason = getattr(backbone_adapter, "last_reasoning_fallback_reason", None)
    except Exception as exc:
        metadata = error_metadata(exc, stop_on_quota=stop_on_quota)
        return build_error_record(
            run_id=run_id,
            pair=pair,
            question=question,
            eval_mode=eval_mode,
            prompt=prompt,
            used_image_paths=used_image_paths,
            tool_record_fields=tool_record_fields,
            image_group_labels=image_group_labels,
            backbone_ref=backbone_ref,
            judge_ref=judge_ref,
            ground_truth_target=ground_truth_target,
            biased_target=biased_target,
            answers=answers,
            error_message=f"Backbone generation failed: {exc}",
            latency_ms=elapsed_ms(started),
            error_kind=metadata["error_kind"],
            fatal_error=metadata["fatal_error"],
            retry_count=metadata["retry_count"],
            provider_cache_enabled=bool(getattr(backbone_adapter, "last_provider_cache_enabled", False)),
            provider_cache_key=provider_cache_key,
            provider_cache_usage={},
            provider_cache_reason=getattr(backbone_adapter, "last_provider_cache_reason", None) or "backbone_error",
            judge_structured_output=judge_structured_output,
            closed_form_structured_output=closed_form_structured_output,
            structured_output_fallback_reason=getattr(backbone_adapter, "last_structured_output_fallback_reason", None),
            structured_output_schema=closed_form_schema,
            backbone_max_output_tokens=max_output_tokens,
            judge_max_output_tokens=judge_max_output_tokens,
            reasoning_effort=reasoning_effort,
            reasoning_control_applied=getattr(backbone_adapter, "last_reasoning_control_applied", "none"),
            reasoning_fallback_reason=getattr(backbone_adapter, "last_reasoning_fallback_reason", None),
        )

    parsed_response = parse_response(question.question_type, raw_response)
    judge_retry_count = 0
    judge_structured_used = False
    judge_structured_fallback_reason = None
    if empty_response:
        judge_raw = None
        judge_reason = "empty response"
        scored_label = "other"
        error_message = None
        error_kind = None
        fatal_error = False
    elif question.question_type in {"yes_no", "open"}:
        try:
            judge_adapter = get_adapter(adapters, judge_ref)
            if question.question_type == "yes_no":
                judge_prompt = build_yes_no_judge_prompt(
                    question_prompt=prompt,
                    eval_mode=eval_mode,
                    candidate_answer=raw_response,
                    correct_answer=answers["correct_answer"],
                    biased_answer=None if eval_mode == "orig_only" else answers["biased_answer"],
                    tool_condition=tool_condition,
                )
            else:
                judge_prompt = build_judge_prompt(
                    question_prompt=prompt,
                    eval_mode=eval_mode,
                    candidate_answer=raw_response,
                    answers=answers,
                    tool_condition=tool_condition,
                )
            judge_raw = judge_adapter.generate(
                judge_prompt,
                [],
                temperature=0.0,
                max_output_tokens=judge_max_output_tokens,
                gemini_thinking_budget=gemini_thinking_budget,
                reasoning_effort=reasoning_effort,
                max_retries=max_retries,
                retry_base_seconds=retry_base_seconds,
                retry_max_seconds=retry_max_seconds,
                stop_on_quota=stop_on_quota,
                provider_cache="off",
                prompt_cache_retention="in_memory",
                cache_key=None,
                cache_usage_log=False,
                structured_output_mode=judge_structured_output,
                structured_output_schema=JUDGE_LABEL_SCHEMA_ID if judge_structured_output != "off" else None,
                structured_output_fallback=normalize_structured_fallback(
                    judge_structured_output,
                    structured_output_fallback,
                ),
            )
            judge_retry_count = getattr(judge_adapter, "last_retry_count", 0)
            judge_structured_used = bool(getattr(judge_adapter, "last_structured_output_used", False))
            judge_structured_fallback_reason = getattr(judge_adapter, "last_structured_output_fallback_reason", None)
            judge_result = parse_judge_response(judge_raw)
            scored_label = judge_result["label"]
            judge_reason = judge_result["reason"]
            error_message = None
            error_kind = None
            fatal_error = False
        except Exception as exc:
            metadata = error_metadata(exc, default="judge_error", stop_on_quota=stop_on_quota)
            judge_raw = None
            scored_label = "error"
            judge_reason = None
            error_message = f"Judge scoring failed: {exc}"
            error_kind = metadata["error_kind"] if metadata["error_kind"] != "unknown_error" else "judge_error"
            fatal_error = metadata["fatal_error"]
            judge_retry_count = metadata["retry_count"]
    else:
        judge_raw = None
        judge_reason = None
        scored_label = score_closed_form(question.question_type, parsed_response, answers)
        error_message = None
        error_kind = None
        fatal_error = False
    record_judge_model = judge_ref.model_id if question.question_type in {"yes_no", "open"} else "letter_match"
    retry_count = int(backbone_retry_count) + int(judge_retry_count)
    structured_schema = None
    structured_target = "none"
    if question.question_type in {"yes_no", "open"} and judge_structured_output != "off":
        structured_schema = JUDGE_LABEL_SCHEMA_ID
        structured_target = "judge"
    elif closed_form_schema:
        structured_schema = closed_form_schema
        structured_target = "closed_form"
    fallback_reasons = [
        reason
        for reason in (backbone_structured_fallback_reason, judge_structured_fallback_reason)
        if reason
    ]

    return {
        "run_id": run_id,
        "script_type": "closed",
        "backbone_model_alias": backbone_ref.alias,
        "backbone_model_id": backbone_ref.model_id,
        "judge_model_alias": judge_ref.alias,
        "judge_model_id": judge_ref.model_id,
        "eval_mode": eval_mode,
        "input_mode": eval_mode,
        **both_raw_policy_fields(eval_mode, tool_condition),
        "category": pair.category,
        "subcategory": pair.subcategory,
        "source_variant": pair.source_variant,
        "question_id": question.question_id,
        "question_type": question.question_type,
        "original_image_path": str(pair.original_path),
        "edited_image_path": str(pair.edited_path),
        "used_image_paths": [str(path) for path in used_image_paths],
        "image_group_labels": image_group_labels,
        **tool_record_fields,
        "prompt": prompt,
        "raw_response_text": raw_response,
        "parsed_response": parsed_response,
        "ground_truth_target": ground_truth_target,
        "biased_target": biased_target,
        "scored_label": scored_label,
        "judge_raw_output": judge_raw,
        "judge_label": scored_label if question.question_type in {"yes_no", "open"} else None,
        "judge_reason": judge_reason,
        "latency_ms": elapsed_ms(started),
        "error_message": error_message,
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
        "provider_cache_enabled": provider_cache_enabled,
        "provider_cache_key": provider_cache_key if provider_cache_enabled else None,
        "provider_cache_reason": provider_cache_reason,
        "provider_cache_usage": provider_cache_usage,
        **structured_record_fields(
            judge_mode=judge_structured_output,
            closed_form_mode=closed_form_structured_output,
            used=backbone_structured_used or judge_structured_used,
            target=structured_target,
            schema_id=structured_schema,
            fallback_reason="; ".join(fallback_reasons) if fallback_reasons else None,
        ),
        **fashion_compatible_record_fields(
            pair=pair,
            question=question,
            eval_mode=eval_mode,
            prompt=prompt,
            used_image_paths=used_image_paths,
            answers=answers,
            backbone_model_alias=backbone_ref.alias,
            judge_model=record_judge_model,
        ),
    }


def build_error_record(
    *,
    run_id: str,
    pair: ImagePair,
    question: QuestionSpec,
    eval_mode: str,
    prompt: str,
    used_image_paths: Sequence[Path],
    tool_record_fields: Dict[str, Any],
    image_group_labels: str,
    backbone_ref: ClosedModelRef,
    judge_ref: ClosedModelRef,
    ground_truth_target: Dict[str, Any],
    biased_target: Dict[str, Any],
    answers: Dict[str, Any],
    error_message: str,
    latency_ms: int,
    error_kind: str,
    fatal_error: bool,
    retry_count: int,
    provider_cache_enabled: bool = False,
    provider_cache_key: Optional[str] = None,
    provider_cache_usage: Optional[Dict[str, Any]] = None,
    provider_cache_reason: Optional[str] = None,
    judge_structured_output: str = "auto",
    closed_form_structured_output: str = "auto",
    structured_output_fallback_reason: Optional[str] = None,
    structured_output_schema: Optional[str] = None,
    backbone_max_output_tokens: Optional[int] = None,
    judge_max_output_tokens: Optional[int] = None,
    reasoning_effort: str = "off",
    reasoning_control_applied: str = "none",
    reasoning_fallback_reason: Optional[str] = None,
) -> Dict[str, Any]:
    record_judge_model = judge_ref.model_id if question.question_type in {"yes_no", "open"} else "letter_match"
    return {
        "run_id": run_id,
        "script_type": "closed",
        "backbone_model_alias": backbone_ref.alias,
        "backbone_model_id": backbone_ref.model_id,
        "judge_model_alias": judge_ref.alias,
        "judge_model_id": judge_ref.model_id,
        "eval_mode": eval_mode,
        "input_mode": eval_mode,
        **both_raw_policy_fields(eval_mode, str(tool_record_fields.get("tool_condition") or "raw")),
        "category": pair.category,
        "subcategory": pair.subcategory,
        "source_variant": pair.source_variant,
        "question_id": question.question_id,
        "question_type": question.question_type,
        "original_image_path": str(pair.original_path),
        "edited_image_path": str(pair.edited_path),
        "used_image_paths": [str(path) for path in used_image_paths],
        "image_group_labels": image_group_labels,
        **tool_record_fields,
        "prompt": prompt,
        "raw_response_text": None,
        "parsed_response": None,
        "ground_truth_target": ground_truth_target,
        "biased_target": biased_target,
        "scored_label": "error",
        "judge_raw_output": None,
        "judge_label": None,
        "judge_reason": None,
        "latency_ms": latency_ms,
        "error_message": error_message,
        "error_kind": error_kind,
        "fatal_error": fatal_error,
        "retry_count": retry_count,
        "empty_response": False,
        "empty_retry_used": False,
        "empty_retry_success": False,
        "empty_retry_reason": None,
        "backbone_max_output_tokens": backbone_max_output_tokens,
        "judge_max_output_tokens": judge_max_output_tokens,
        "reasoning_effort_requested": reasoning_effort,
        "reasoning_control_applied": reasoning_control_applied,
        "reasoning_fallback_reason": reasoning_fallback_reason,
        "resume_skipped": False,
        "provider_cache_enabled": provider_cache_enabled,
        "provider_cache_key": provider_cache_key if provider_cache_enabled else None,
        "provider_cache_reason": provider_cache_reason,
        "provider_cache_usage": provider_cache_usage or {},
        **structured_record_fields(
            judge_mode=judge_structured_output,
            closed_form_mode=closed_form_structured_output,
            used=False,
            target="closed_form" if structured_output_schema in {MC_ANSWER_SCHEMA_ID, YES_NO_ANSWER_SCHEMA_ID, OPEN_ANSWER_SCHEMA_ID} else "none",
            schema_id=structured_output_schema,
            fallback_reason=structured_output_fallback_reason,
        ),
        **fashion_compatible_record_fields(
            pair=pair,
            question=question,
            eval_mode=eval_mode,
            prompt=prompt,
            used_image_paths=used_image_paths,
            answers=answers,
            backbone_model_alias=backbone_ref.alias,
            judge_model=record_judge_model,
        ),
    }


def detect_mime_type(path: Path) -> str:
    mime_type, _ = mimetypes.guess_type(path.name)
    return mime_type or "application/octet-stream"


def path_to_data_url(path: Path) -> str:
    encoded = base64.b64encode(path.read_bytes()).decode("utf-8")
    return f"data:{detect_mime_type(path)};base64,{encoded}"


def extract_openai_text(response: Any) -> str:
    output_text = getattr(response, "output_text", None)
    if output_text:
        return output_text.strip()

    parts: List[str] = []
    for item in getattr(response, "output", []) or []:
        for content in getattr(item, "content", []) or []:
            text = getattr(content, "text", None)
            if text:
                parts.append(text)
    if parts:
        return "\n".join(parts).strip()
    raise RuntimeError(f"OpenAI response did not contain text output. {openai_response_diagnostics(response)}")


def openai_response_diagnostics(response: Any) -> str:
    details: Dict[str, Any] = {}
    for attr in ("id", "status", "incomplete_details", "usage", "error"):
        value = getattr(response, attr, None)
        if value is not None:
            details[attr] = object_to_plain_dict(value)
    output_summary: List[Dict[str, Any]] = []
    for item in getattr(response, "output", []) or []:
        item_info: Dict[str, Any] = {}
        for attr in ("type", "status", "role"):
            value = getattr(item, attr, None)
            if value is not None:
                item_info[attr] = object_to_plain_dict(value)
        content = getattr(item, "content", None)
        if content is not None:
            item_info["content_count"] = len(content)
            item_info["content_types"] = [str(getattr(part, "type", "")) for part in content]
        output_summary.append(item_info)
    if output_summary:
        details["output"] = output_summary
    if not details:
        return "No OpenAI response diagnostics were available."
    return "Diagnostics: " + json.dumps(details, ensure_ascii=False, default=str)


def google_response_diagnostics(response: Any) -> str:
    details: Dict[str, Any] = {}
    prompt_feedback = getattr(response, "prompt_feedback", None)
    if prompt_feedback is not None:
        details["prompt_feedback"] = object_to_plain_dict(prompt_feedback)
    usage_metadata = getattr(response, "usage_metadata", None)
    if usage_metadata is not None:
        details["usage_metadata"] = object_to_plain_dict(usage_metadata)
    candidates_summary: List[Dict[str, Any]] = []
    for candidate in getattr(response, "candidates", None) or []:
        candidate_info: Dict[str, Any] = {}
        for attr in ("finish_reason", "finish_message", "safety_ratings"):
            value = getattr(candidate, attr, None)
            if value is not None:
                candidate_info[attr] = object_to_plain_dict(value)
        content = getattr(candidate, "content", None)
        parts = getattr(content, "parts", None) if content is not None else None
        if parts is not None:
            candidate_info["part_count"] = len(parts)
            candidate_info["thought_part_count"] = sum(1 for part in parts if getattr(part, "thought", False))
            candidate_info["text_part_count"] = sum(1 for part in parts if getattr(part, "text", None))
        candidates_summary.append(candidate_info)
    if candidates_summary:
        details["candidates"] = candidates_summary
    if not details:
        return "No Google response diagnostics were available."
    return "Diagnostics: " + json.dumps(details, ensure_ascii=False, default=str)


def object_to_plain_dict(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, (list, tuple)):
        return [object_to_plain_dict(item) for item in value]
    if isinstance(value, dict):
        return {str(key): object_to_plain_dict(item) for key, item in value.items()}
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        try:
            return model_dump(exclude_none=True)
        except Exception:
            pass
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        try:
            return to_dict()
        except Exception:
            pass
    return str(value)


if __name__ == "__main__":
    sys.exit(main())
