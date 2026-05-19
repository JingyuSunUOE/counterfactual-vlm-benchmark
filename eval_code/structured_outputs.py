#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from copy import deepcopy
from typing import Any, Dict, Optional


STRUCTURED_OUTPUT_MODES = ("off", "auto", "on")
STRUCTURED_OUTPUT_FALLBACKS = ("retry_plain", "fail")

JUDGE_LABEL_SCHEMA_ID = "judge_label_v1"
MC_ANSWER_SCHEMA_ID = "mc_answer_v1"
YES_NO_ANSWER_SCHEMA_ID = "yes_no_answer_v1"
OPEN_ANSWER_SCHEMA_ID = "open_answer_v1"


_SCHEMAS: Dict[str, Dict[str, Any]] = {
    JUDGE_LABEL_SCHEMA_ID: {
        "type": "object",
        "properties": {
            "label": {"type": "string", "enum": ["correct", "biased", "other"]},
            "reason": {"type": "string"},
        },
        "required": ["label", "reason"],
        "additionalProperties": False,
    },
    MC_ANSWER_SCHEMA_ID: {
        "type": "object",
        "properties": {
            "answer": {"type": "string", "enum": ["A", "B", "C", "D"]},
        },
        "required": ["answer"],
        "additionalProperties": False,
    },
    YES_NO_ANSWER_SCHEMA_ID: {
        "type": "object",
        "properties": {
            "answer": {"type": "string", "enum": ["yes", "no"]},
        },
        "required": ["answer"],
        "additionalProperties": False,
    },
    OPEN_ANSWER_SCHEMA_ID: {
        "type": "object",
        "properties": {
            "answer": {"type": "string"},
        },
        "required": ["answer"],
        "additionalProperties": False,
    },
}


def add_structured_output_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--judge-structured-output",
        choices=STRUCTURED_OUTPUT_MODES,
        default="auto",
        help="Use provider structured output for judge calls when supported.",
    )
    parser.add_argument(
        "--closed-form-structured-output",
        choices=STRUCTURED_OUTPUT_MODES,
        default="auto",
        help="Request structured backbone output for yes/no, multiple-choice, and open questions when supported.",
    )
    parser.add_argument(
        "--structured-output-fallback",
        choices=STRUCTURED_OUTPUT_FALLBACKS,
        default="retry_plain",
        help="Fallback policy when structured output is rejected in auto mode.",
    )


def normalize_structured_fallback(mode: str, fallback: str) -> str:
    return "fail" if mode == "on" else fallback


def schema_for(schema_id: str) -> Dict[str, Any]:
    if schema_id not in _SCHEMAS:
        raise ValueError(f"Unknown structured output schema: {schema_id}")
    return deepcopy(_SCHEMAS[schema_id])


def openai_chat_response_format(schema_id: str) -> Dict[str, Any]:
    return {
        "type": "json_schema",
        "json_schema": {
            "name": schema_id,
            "schema": schema_for(schema_id),
            "strict": True,
        },
    }


def openai_responses_text_format(schema_id: str) -> Dict[str, Any]:
    return {
        "format": {
            "type": "json_schema",
            "name": schema_id,
            "schema": schema_for(schema_id),
            "strict": True,
        }
    }


def anthropic_tool(schema_id: str) -> Dict[str, Any]:
    return {
        "name": "emit_structured_output",
        "description": "Return the benchmark answer in the required JSON schema.",
        "input_schema": schema_for(schema_id),
    }


def anthropic_tool_choice() -> Dict[str, str]:
    return {"type": "tool", "name": "emit_structured_output"}


def gemini_config_kwargs(schema_id: str) -> Dict[str, Any]:
    return {
        "response_mime_type": "application/json",
        "response_json_schema": schema_for(schema_id),
    }


def gemini_rest_generation_config(schema_id: str) -> Dict[str, Any]:
    return {
        "responseMimeType": "application/json",
        "responseJsonSchema": schema_for(schema_id),
    }


def extract_anthropic_tool_json(response: Any) -> Optional[str]:
    for block in getattr(response, "content", []) or []:
        if getattr(block, "type", None) == "tool_use":
            value = getattr(block, "input", None)
            if isinstance(value, dict):
                return json.dumps(value, ensure_ascii=False)
    return None


def extract_json_object(raw_response: str) -> Dict[str, Any]:
    text = (raw_response or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.DOTALL).strip()
    try:
        payload = json.loads(text)
        if isinstance(payload, dict):
            return payload
    except Exception:
        pass
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("Structured output did not contain a JSON object.")
    payload = json.loads(text[start : end + 1])
    if not isinstance(payload, dict):
        raise ValueError("Structured output JSON was not an object.")
    return payload


def parse_structured_judge_response(raw_response: str) -> Dict[str, str]:
    payload = extract_json_object(raw_response)
    label = str(payload.get("label", "")).strip().lower()
    reason = str(payload.get("reason", "")).strip()
    if label not in {"correct", "biased", "other"}:
        raise ValueError(f"Structured judge output used unsupported label: {label!r}")
    return {"label": label, "reason": reason}


def parse_structured_mc_answer(raw_response: str) -> Optional[str]:
    try:
        payload = extract_json_object(raw_response)
    except Exception:
        return None
    answer = str(payload.get("answer", "")).strip().upper()
    return answer if answer in {"A", "B", "C", "D"} else None


def parse_structured_yes_no_answer(raw_response: str) -> Optional[str]:
    try:
        payload = extract_json_object(raw_response)
    except Exception:
        return None
    answer = str(payload.get("answer", "")).strip().lower()
    return answer if answer in {"yes", "no"} else None


def parse_structured_open_answer(raw_response: str) -> Optional[str]:
    try:
        payload = extract_json_object(raw_response)
    except Exception:
        return None
    answer = str(payload.get("answer", "")).strip()
    return answer or None


def strip_thinking_blocks(raw_response: str) -> str:
    text = raw_response or ""
    text = re.sub(r"<think\b[^>]*>.*?</think>", "", text, flags=re.IGNORECASE | re.DOTALL)
    text = re.sub(r"<thought\b[^>]*>.*?</thought>", "", text, flags=re.IGNORECASE | re.DOTALL)
    return text.strip()


def format_structured_fallback_reason(exc: Exception) -> str:
    message = str(exc).strip() or exc.__class__.__name__
    message = re.sub(r"\s+", " ", message)
    return message[:500]


def looks_like_structured_output_unsupported(exc: Exception) -> bool:
    text = f"{exc.__class__.__name__}: {exc}".lower()
    unsupported_markers = (
        "response_format",
        "json_schema",
        "response_json_schema",
        "responsejsonschema",
        "response_mime_type",
        "responsemimetype",
        "tool_choice",
        "tools",
        "structured output",
        "unsupported",
        "unrecognized",
        "unknown parameter",
        "unexpected keyword",
        "invalid request",
        "schema",
    )
    fatal_or_retry_markers = (
        "quota",
        "rate limit",
        "429",
        "401",
        "403",
        "permission",
        "unauthorized",
        "timeout",
        "connection",
        "server error",
        "503",
        "502",
        "500",
    )
    return any(marker in text for marker in unsupported_markers) and not any(
        marker in text for marker in fatal_or_retry_markers
    )


def structured_record_fields(
    *,
    judge_mode: str,
    closed_form_mode: str,
    used: bool,
    target: str,
    schema_id: Optional[str],
    fallback_reason: Optional[str],
) -> Dict[str, Any]:
    return {
        "judge_structured_output": judge_mode,
        "closed_form_structured_output": closed_form_mode,
        "structured_output_used": bool(used),
        "structured_output_target": target if used or target in {"judge", "closed_form"} else "none",
        "structured_output_schema": schema_id,
        "structured_output_fallback_reason": fallback_reason,
    }
