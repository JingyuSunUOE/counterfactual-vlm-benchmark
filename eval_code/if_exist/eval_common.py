#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import random
import re
import sys
import time
from collections import Counter
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple


REPO_ROOT = Path(__file__).resolve().parents[2]
EVAL_CODE_DIR = REPO_ROOT / "eval_code"
if str(EVAL_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(EVAL_CODE_DIR))

from structured_outputs import (
    MC_ANSWER_SCHEMA_ID,
    OPEN_ANSWER_SCHEMA_ID,
    YES_NO_ANSWER_SCHEMA_ID,
    add_structured_output_arguments,
    parse_structured_judge_response,
    parse_structured_mc_answer,
    parse_structured_open_answer,
    parse_structured_yes_no_answer,
    strip_thinking_blocks,
)

DEFAULT_QUESTIONS_PATH = REPO_ROOT / "cf_dataset" / "if_exist_cf_questions.json"
DEFAULT_RESULTS_ROOT = REPO_ROOT / "eval_results" / "if_exist"
DEFAULT_RAW_RUNS_ROOT = DEFAULT_RESULTS_ROOT / "raw_runs"
DEFAULT_TABLES_ROOT = DEFAULT_RESULTS_ROOT / "tables"
DEFAULT_FIGURES_ROOT = DEFAULT_RESULTS_ROOT / "figures"
DEFAULT_REPORTS_ROOT = DEFAULT_RESULTS_ROOT / "reports"
DEFAULT_METADATA_ROOT = DEFAULT_RESULTS_ROOT / "metadata"

QUESTION_TYPES = ("yes_no", "multiple_choice", "open")
SOURCE_VARIANTS = ("real", "ai")
ANSWER_MODES = ("cf_only", "both", "orig_only")
TOOL_CONDITIONS = ("raw", "bbox", "crop", "zoom_panel", "tool_bundle")
EVIDENCE_QC_FILTERS = ("pass", "pass_review", "all")
MISSING_EVIDENCE_POLICIES = ("skip", "error", "raw_fallback")
EMPTY_RESPONSE_POLICIES = ("retry_once", "other", "error")
REASONING_EFFORTS = ("off", "minimal", "low", "medium", "high", "provider_default")
EVIDENCE_SCHEMA_VERSION = "if_exist_sam3_object_evidence_v1"
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
EDITED_SUFFIXES = (
    "_change_scene",
    "_remove_hump",
    "_remove_trunk",
    "_remove_fin",
    "_remove_ear",
)
PROGRESS_BAR_WIDTH = 28
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
FATAL_ERROR_KINDS = {"quota_exceeded", "auth_error"}
RETRYABLE_ERROR_KINDS = {"rate_limit", "server_error", "network_error", "timeout"}
RESUME_MATCH_IGNORED_KEYS = {
    "max_retries",
    "retry_base_seconds",
    "retry_max_seconds",
    "stop_on_quota",
    "max_consecutive_errors",
    "resume",
    "server_vram_plan",
    "judge_server_vram_plan",
}
PRIOR_PRIME_PREFIX = (
    "This is a well-known, classic {display_name}. "
    "Take a careful look at the image and answer the following question "
    "based strictly on what you actually see, not on what you expect."
)
BOTH_MODE_TARGET_INSTRUCTION = (
    "Two related images are provided. Compare the visible target attribute across the two images "
    "and answer using only what differs or stays the same."
)
BOTH_RAW_PROMPT_POLICY = "neutral_pair_difference_v1"
BOTH_RAW_SCORING_POLICY = "neutral_pair_difference_v1"
BOTH_RAW_PAIR_DIFFERENCE_PREFIX = (
    "Two related images are provided. Compare the visible target attribute across the two images "
    "and answer using only what differs or stays the same."
)
TOOL_VIEW_PREFIX = (
    "Additional images, if provided, show alternative visual views of the same scene. "
    "Use the provided visual information to answer the question."
)
TOOL_VIEW_BOTH_PREFIX = (
    "If two image groups are provided, answer about the second image group. "
    "Additional images in each group show alternative visual views of the same scene."
)

DEFAULT_DISPLAY_NAMES = {
    "camel": "camel",
    "elephant_trunk": "elephant",
    "fish_fin": "fish",
    "rabbit": "rabbit",
}


@dataclass(frozen=True)
class QuestionSpec:
    category: str
    subcategory: str
    display_name: str
    question_id: str
    question_type: str
    prompt: str
    choices: List[Dict[str, str]]
    answers_by_mode: Dict[str, Dict[str, Any]]


@dataclass(frozen=True)
class GroupSpec:
    category: str
    subcategory: str
    display_name: str
    source_variant: str
    original_dir: Path
    edited_dir: Path
    questions: List[QuestionSpec]


@dataclass(frozen=True)
class ImagePair:
    category: str
    subcategory: str
    source_variant: str
    key: str
    original_path: Path
    edited_path: Path


@dataclass(frozen=True)
class ImageGroup:
    role: str
    raw: Path
    evidence: Tuple[Path, ...] = ()


@dataclass(frozen=True)
class LabeledImage:
    path: Path
    label: Optional[str] = None


@dataclass(frozen=True)
class EvidenceEligibility:
    eligible: bool
    reason: str
    qc_status: Optional[str] = None
    missing_tool_condition: Optional[str] = None
    raw_fallback: bool = False


class BenchmarkAPIError(RuntimeError):
    def __init__(self, message: str, *, error_kind: str, fatal_error: bool, retry_count: int = 0) -> None:
        super().__init__(message)
        self.error_kind = error_kind
        self.fatal_error = fatal_error
        self.retry_count = retry_count


def add_common_eval_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--dry-run", action="store_true", help="Build and print task previews without model calls.")
    parser.add_argument("--report", action="store_true", help="Print readable summaries from existing run outputs and exit.")
    parser.add_argument("--category", choices=("if_exist",), default=None, help="Restrict to the if_exist category.")
    parser.add_argument("--subcategory", default=None, help="Restrict to one semantic subcategory.")
    parser.add_argument(
        "--source-variant",
        choices=("real", "ai", "both"),
        default="both",
        help="Restrict to real images, AI images, or both.",
    )
    parser.add_argument(
        "--question-types",
        default="all",
        help="Comma-separated subset of yes_no,multiple_choice,open, or all.",
    )
    parser.add_argument(
        "--tool-condition",
        choices=TOOL_CONDITIONS,
        default="raw",
        help="Visual tool condition: raw, bbox, crop, zoom_panel, or tool_bundle.",
    )
    parser.add_argument(
        "--evidence-manifest",
        type=Path,
        default=None,
        help="SAM evidence manifest used by non-raw tool conditions.",
    )
    parser.add_argument(
        "--evidence-qc-filter",
        choices=EVIDENCE_QC_FILTERS,
        default="pass",
        help="Evidence QC states eligible for tool-condition runs.",
    )
    parser.add_argument(
        "--missing-evidence-policy",
        choices=MISSING_EVIDENCE_POLICIES,
        default="skip",
        help="How to handle pairs whose selected tool evidence is unavailable.",
    )
    parser.add_argument(
        "--image-group-labels",
        choices=("off", "on"),
        default="off",
        help="Insert neutral Image group/view labels before image blocks. Recommended for formal tool-condition runs.",
    )
    parser.add_argument(
        "--prime",
        choices=("on", "off"),
        default="on",
        help="Enable or disable the shared prior-prime prefix.",
    )
    parser.add_argument("--no-progress", action="store_true", help="Disable progress display.")
    parser.add_argument(
        "--resume",
        choices=("off", "latest", "auto"),
        default="off",
        help="Resume a previous run. Success records are skipped; failed records are retried.",
    )
    parser.add_argument("--max-retries", type=int, default=3, help="Maximum API attempts per request.")
    parser.add_argument(
        "--retry-base-seconds",
        type=float,
        default=2.0,
        help="Base delay for exponential retry backoff.",
    )
    parser.add_argument(
        "--retry-max-seconds",
        type=float,
        default=60.0,
        help="Maximum delay for retry backoff.",
    )
    parser.add_argument(
        "--stop-on-quota",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Stop the run immediately on quota/billing/authentication errors.",
    )
    parser.add_argument(
        "--max-consecutive-errors",
        type=int,
        default=5,
        help="Stop the run after this many consecutive failed tasks.",
    )
    parser.add_argument(
        "--empty-response-policy",
        choices=EMPTY_RESPONSE_POLICIES,
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
        choices=REASONING_EFFORTS,
        default="off",
        help="Reasoning/thinking control. off requests the minimum non-thinking mode supported by the provider.",
    )
    add_structured_output_arguments(parser)


def parse_question_types(raw_value: str) -> List[str]:
    if raw_value == "all":
        return list(QUESTION_TYPES)
    parsed = [item.strip() for item in raw_value.split(",") if item.strip()]
    invalid = [item for item in parsed if item not in QUESTION_TYPES]
    if invalid:
        raise SystemExit(f"Invalid question types: {', '.join(invalid)}")
    if not parsed:
        raise SystemExit("At least one question type must be selected.")
    return parsed


def seed_everything(seed: int) -> None:
    random.seed(seed)


def make_run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def load_benchmark(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def build_group_specs(benchmark: Dict[str, Any], *, base_dir: Optional[Path] = None) -> List[GroupSpec]:
    groups: List[GroupSpec] = []
    root = (base_dir or Path.cwd()).resolve()
    for category_entry in benchmark["categories"]:
        category = category_entry["category"]
        for subcategory_entry in category_entry["subcategories"]:
            subcategory = subcategory_entry["subcategory"]
            display_name = subcategory_entry.get("display_name") or DEFAULT_DISPLAY_NAMES.get(
                subcategory, subcategory.replace("_", " ")
            )
            questions = [
                QuestionSpec(
                    category=category,
                    subcategory=subcategory,
                    display_name=display_name,
                    question_id=question["question_id"],
                    question_type=question["type"],
                    prompt=question["prompt"],
                    choices=list(question.get("choices", [])),
                    answers_by_mode=question["answers"],
                )
                for question in subcategory_entry["questions"]
            ]
            for source_variant in SOURCE_VARIANTS:
                leaf_name = subcategory if source_variant == "real" else f"{subcategory}_ai"
                groups.append(
                    GroupSpec(
                        category=category,
                        subcategory=subcategory,
                        display_name=display_name,
                        source_variant=source_variant,
                        original_dir=(root / "dataset" / category / leaf_name).resolve(),
                        edited_dir=(root / "cf_dataset" / f"{category}_cf" / leaf_name).resolve(),
                        questions=questions,
                    )
                )
    return groups


def resolve_selected_groups(groups: Sequence[GroupSpec], subset_path: Optional[Path]) -> List[GroupSpec]:
    if subset_path is None:
        selected = list(groups)
    else:
        target = subset_path.resolve()
        selected = [
            group
            for group in groups
            if target in {
                group.original_dir,
                group.edited_dir,
                group.original_dir.parent,
                group.edited_dir.parent,
            }
        ]
        if not selected:
            raise SystemExit(f"Subset path did not match any known benchmark folder: {target}")

    missing_dirs = [
        (group.category, group.subcategory, group.source_variant, group.original_dir, group.edited_dir)
        for group in selected
        if not group.original_dir.exists() or not group.edited_dir.exists()
    ]
    if missing_dirs:
        problems = [
            f"{category}/{subcategory} ({source_variant}) -> missing original={not orig.exists()} edited={not edited.exists()}"
            for category, subcategory, source_variant, orig, edited in missing_dirs
        ]
        raise SystemExit("Missing dataset directories:\n- " + "\n- ".join(problems))
    return sorted(
        selected,
        key=lambda group: (group.category, group.subcategory, 0 if group.source_variant == "real" else 1),
    )


def filter_groups(
    groups: Sequence[Any],
    *,
    category: Optional[str],
    subcategory: Optional[str],
    source_variant: str,
    question_types: Sequence[str],
) -> List[Any]:
    selected: List[Any] = []
    allowed_sources = set(SOURCE_VARIANTS if source_variant == "both" else (source_variant,))
    allowed_question_types = set(question_types)

    for group in groups:
        if category and group.category != category:
            continue
        if subcategory and group.subcategory != subcategory:
            continue
        if group.source_variant not in allowed_sources:
            continue
        questions = [question for question in group.questions if question.question_type in allowed_question_types]
        if questions:
            selected.append(replace(group, questions=questions))

    if not selected:
        raise SystemExit("No benchmark groups matched the selected filters.")
    return selected


def build_questions_by_key(groups: Sequence[Any]) -> Dict[tuple[str, str, str], List[Any]]:
    return {(group.category, group.subcategory, group.source_variant): list(group.questions) for group in groups}


def count_total_evaluations(image_pairs: Sequence[Any], questions_by_key: Dict[tuple[str, str, str], List[Any]]) -> int:
    return sum(len(questions_by_key[(pair.category, pair.subcategory, pair.source_variant)]) for pair in image_pairs)


def build_eval_tasks(
    image_pairs: Sequence[ImagePair],
    questions_by_key: Dict[tuple[str, str, str], List[QuestionSpec]],
) -> List[Tuple[ImagePair, QuestionSpec]]:
    tasks: List[Tuple[ImagePair, QuestionSpec]] = []
    for pair in image_pairs:
        for question in questions_by_key[(pair.category, pair.subcategory, pair.source_variant)]:
            tasks.append((pair, question))
    return tasks


def build_image_pairs(groups: Sequence[GroupSpec]) -> List[ImagePair]:
    pairs: List[ImagePair] = []
    problems: List[str] = []

    for group in groups:
        original_map = collect_image_map(group.original_dir, is_edited=False)
        edited_map = collect_image_map(group.edited_dir, is_edited=True)

        missing_original = sorted(set(edited_map) - set(original_map), key=natural_key)
        missing_edited = sorted(set(original_map) - set(edited_map), key=natural_key)
        if missing_original or missing_edited:
            details = []
            if missing_original:
                details.append(f"missing original counterparts for {missing_original}")
            if missing_edited:
                details.append(f"missing edited counterparts for {missing_edited}")
            problems.append(
                f"{group.category}/{group.subcategory} ({group.source_variant}): " + "; ".join(details)
            )
            continue

        for key in sorted(original_map, key=natural_key):
            pairs.append(
                ImagePair(
                    category=group.category,
                    subcategory=group.subcategory,
                    source_variant=group.source_variant,
                    key=key,
                    original_path=original_map[key],
                    edited_path=edited_map[key],
                )
            )

    if problems:
        raise SystemExit("Dataset pairing failed:\n- " + "\n- ".join(problems))
    return pairs


def collect_image_map(directory: Path, *, is_edited: bool) -> Dict[str, Path]:
    mapping: Dict[str, Path] = {}
    for path in sorted(directory.iterdir(), key=lambda item: natural_key(item.name)):
        if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        key = normalize_image_key(path.stem, is_edited=is_edited)
        if key in mapping:
            raise SystemExit(f"Duplicate normalized image key '{key}' under {directory}")
        mapping[key] = path.resolve()
    return mapping


def normalize_image_key(stem: str, *, is_edited: bool) -> str:
    if is_edited:
        for suffix in EDITED_SUFFIXES:
            if stem.endswith(suffix):
                return stem[: -len(suffix)]
    return stem


def natural_key(value: str) -> List[Any]:
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", value)]


def image_paths_for_mode(pair: ImagePair, eval_mode: str) -> List[Path]:
    if eval_mode == "cf_only":
        return [pair.edited_path]
    if eval_mode == "both":
        return [pair.original_path, pair.edited_path]
    if eval_mode == "orig_only":
        return [pair.original_path]
    raise ValueError(f"Unsupported eval mode: {eval_mode}")


def load_evidence_manifest(path: Optional[Path]) -> Dict[tuple[str, str, str], Dict[str, Any]]:
    if path is None:
        return {}
    manifest_path = path.expanduser().resolve()
    if not manifest_path.exists():
        raise SystemExit(f"Evidence manifest not found: {manifest_path}")
    with manifest_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    records = payload.get("records", payload) if isinstance(payload, dict) else payload
    if not isinstance(records, list):
        raise SystemExit(f"Evidence manifest must be a list or contain a records list: {manifest_path}")

    manifest: Dict[tuple[str, str, str], Dict[str, Any]] = {}
    problems: List[str] = []
    for index, record in enumerate(records, start=1):
        if not isinstance(record, dict):
            problems.append(f"record {index}: expected object")
            continue
        key = (
            str(record.get("subcategory") or ""),
            str(record.get("source_variant") or ""),
            str(record.get("pair_key") or ""),
        )
        if not all(key):
            problems.append(f"record {index}: missing subcategory/source_variant/pair_key")
            continue
        if record.get("schema_version") != EVIDENCE_SCHEMA_VERSION:
            problems.append(f"record {index}: unsupported schema_version={record.get('schema_version')!r}")
            continue
        if key in manifest:
            problems.append(f"record {index}: duplicate evidence key {key}")
            continue
        for role in ("original", "cf"):
            role_payload = record.get(role)
            if not isinstance(role_payload, dict):
                problems.append(f"record {index}: missing {role} evidence block")
                continue
            for field in ("raw", "bbox_overlay", "crop", "zoom_panel"):
                value = role_payload.get(field)
                if not value:
                    continue
                resolved = resolve_manifest_path(value, base_dir=manifest_path.parent)
                if not resolved.exists():
                    problems.append(f"record {index}: missing referenced file {resolved}")
        manifest[key] = record
    if problems:
        preview = "\n- ".join(problems[:20])
        suffix = "" if len(problems) <= 20 else f"\n- ... {len(problems) - 20} additional problems"
        raise SystemExit(f"Evidence manifest validation failed:\n- {preview}{suffix}")
    return manifest


def resolve_manifest_path(value: str | Path, *, base_dir: Optional[Path] = None) -> Path:
    raw = Path(value).expanduser()
    if raw.is_absolute() and raw.exists():
        return raw.resolve()
    value_text = str(value)
    for marker in (
        "dataset/if_exist/",
        "cf_dataset/if_exist_cf/",
        "vision_dataset/if_exist/",
        "eval_results/if_exist/",
    ):
        marker_index = value_text.find(marker)
        if marker_index != -1:
            return (REPO_ROOT / value_text[marker_index:]).resolve(strict=False)
    if raw.is_absolute():
        return raw.resolve(strict=False)
    root = base_dir or REPO_ROOT
    return (root / raw).resolve(strict=False)


def evidence_manifest_key(pair: ImagePair) -> tuple[str, str, str]:
    return (pair.subcategory, pair.source_variant, pair.key)


def validate_tool_condition_args(
    *,
    tool_condition: str,
    evidence_manifest_path: Optional[Path],
    evidence_qc_filter: str = "pass",
    missing_evidence_policy: str = "skip",
) -> None:
    if tool_condition not in TOOL_CONDITIONS:
        raise SystemExit(f"Unsupported tool condition: {tool_condition}")
    if evidence_qc_filter not in EVIDENCE_QC_FILTERS:
        raise SystemExit(f"Unsupported evidence QC filter: {evidence_qc_filter}")
    if missing_evidence_policy not in MISSING_EVIDENCE_POLICIES:
        raise SystemExit(f"Unsupported missing evidence policy: {missing_evidence_policy}")
    if tool_condition != "raw" and evidence_manifest_path is None:
        raise SystemExit("--evidence-manifest is required when --tool-condition is not raw.")


def tool_condition_fields(tool_condition: str) -> Tuple[str, ...]:
    if tool_condition == "raw":
        return ()
    field_map = {
        "bbox": ("bbox_overlay",),
        "crop": ("crop",),
        "zoom_panel": ("zoom_panel",),
        "tool_bundle": ("bbox_overlay", "crop", "zoom_panel"),
    }
    fields = field_map.get(tool_condition)
    if fields is None:
        raise ValueError(f"Unsupported tool condition: {tool_condition}")
    return fields


def evidence_paths_for_condition(
    role_payload: Dict[str, Any],
    tool_condition: str,
    *,
    base_dir: Optional[Path],
    missing_evidence_policy: str = "error",
) -> List[Path]:
    fields = tool_condition_fields(tool_condition)
    if not fields:
        return []
    missing = [field for field in fields if not role_payload.get(field)]
    if missing:
        if missing_evidence_policy == "raw_fallback":
            return []
        raise ValueError(f"Missing evidence fields for {tool_condition}: {missing}")
    paths = [resolve_manifest_path(str(role_payload[field]), base_dir=base_dir) for field in fields]
    missing_paths = [str(path) for path in paths if not path.exists()]
    if missing_paths:
        if missing_evidence_policy == "raw_fallback":
            return []
        raise ValueError(f"Missing evidence files for {tool_condition}: {missing_paths}")
    return paths


def qc_status_allowed(qc_status: str, evidence_qc_filter: str) -> bool:
    if evidence_qc_filter == "all":
        return True
    if evidence_qc_filter == "pass_review":
        return qc_status in {"pass", "review"}
    return qc_status == "pass"


def required_roles_for_eval_mode(eval_mode: str) -> Tuple[str, ...]:
    if eval_mode == "cf_only":
        return ("cf",)
    if eval_mode == "both":
        return ("original", "cf")
    if eval_mode == "orig_only":
        return ("original",)
    raise ValueError(f"Unsupported eval mode: {eval_mode}")


def role_has_tool_evidence(
    role_payload: Dict[str, Any],
    *,
    tool_condition: str,
    base_dir: Optional[Path],
) -> Tuple[bool, Optional[str]]:
    for field in tool_condition_fields(tool_condition):
        value = role_payload.get(field)
        if not value:
            return False, field
        if not resolve_manifest_path(str(value), base_dir=base_dir).exists():
            return False, field
    return True, None


def compute_usable_tool_conditions(
    record: Dict[str, Any],
    *,
    base_dir: Optional[Path] = None,
) -> Dict[str, bool]:
    usable = {"raw": True}
    for condition in ("bbox", "crop", "zoom_panel", "tool_bundle"):
        condition_ok = True
        for role in ("original", "cf"):
            payload = record.get(role)
            if not isinstance(payload, dict):
                condition_ok = False
                break
            ok, _missing = role_has_tool_evidence(payload, tool_condition=condition, base_dir=base_dir)
            if not ok:
                condition_ok = False
                break
        usable[condition] = condition_ok
    return usable


def check_pair_tool_eligibility(
    *,
    pair: ImagePair,
    eval_mode: str,
    tool_condition: str,
    manifest_record: Optional[Dict[str, Any]],
    evidence_qc_filter: str,
    missing_evidence_policy: str = "skip",
    evidence_manifest_path: Optional[Path] = None,
) -> EvidenceEligibility:
    if manifest_record is None:
        if tool_condition == "raw" and evidence_manifest_path is None:
            return EvidenceEligibility(True, "no_manifest_required")
        return EvidenceEligibility(False, "missing_manifest_record")
    qc_status = str(manifest_record.get("qc_status") or "")
    if not qc_status_allowed(qc_status, evidence_qc_filter):
        return EvidenceEligibility(False, "qc_status_filtered", qc_status=qc_status)
    if tool_condition == "raw":
        return EvidenceEligibility(True, "eligible", qc_status=qc_status)
    base_dir = evidence_manifest_path.expanduser().resolve().parent if evidence_manifest_path else None
    for role in required_roles_for_eval_mode(eval_mode):
        payload = manifest_record.get(role)
        if not isinstance(payload, dict):
            return EvidenceEligibility(False, "missing_role_payload", qc_status=qc_status, missing_tool_condition=role)
        ok, missing_field = role_has_tool_evidence(payload, tool_condition=tool_condition, base_dir=base_dir)
        if not ok:
            if missing_evidence_policy == "raw_fallback":
                return EvidenceEligibility(
                    True,
                    "raw_fallback",
                    qc_status=qc_status,
                    missing_tool_condition=f"{role}.{missing_field}",
                    raw_fallback=True,
                )
            return EvidenceEligibility(
                False,
                "missing_tool_condition",
                qc_status=qc_status,
                missing_tool_condition=f"{role}.{missing_field}",
            )
    return EvidenceEligibility(True, "eligible", qc_status=qc_status)


def filter_image_pairs_for_evidence(
    image_pairs: Sequence[ImagePair],
    *,
    eval_mode: str,
    tool_condition: str,
    evidence_manifest: Optional[Dict[tuple[str, str, str], Dict[str, Any]]],
    evidence_manifest_path: Optional[Path],
    evidence_qc_filter: str,
    missing_evidence_policy: str,
) -> Tuple[List[ImagePair], Dict[str, Any]]:
    manifest = evidence_manifest or {}
    eligible_pairs: List[ImagePair] = []
    skipped_by_reason: Counter[str] = Counter()
    skipped_by_qc_status: Counter[str] = Counter()
    skipped_by_missing_tool_condition: Counter[str] = Counter()
    raw_fallback_count = 0
    for pair in image_pairs:
        manifest_record = manifest.get(evidence_manifest_key(pair))
        eligibility = check_pair_tool_eligibility(
            pair=pair,
            eval_mode=eval_mode,
            tool_condition=tool_condition,
            manifest_record=manifest_record,
            evidence_qc_filter=evidence_qc_filter,
            missing_evidence_policy=missing_evidence_policy,
            evidence_manifest_path=evidence_manifest_path,
        )
        if eligibility.eligible:
            eligible_pairs.append(pair)
            if eligibility.raw_fallback:
                raw_fallback_count += 1
            continue
        skipped_by_reason[eligibility.reason] += 1
        if eligibility.qc_status:
            skipped_by_qc_status[eligibility.qc_status] += 1
        if eligibility.missing_tool_condition:
            skipped_by_missing_tool_condition[eligibility.missing_tool_condition] += 1

    stats = {
        "manifest_pairs_total": len(manifest),
        "input_pair_count": len(image_pairs),
        "eligible_pair_count": len(eligible_pairs),
        "skipped_pair_count_due_to_evidence": len(image_pairs) - len(eligible_pairs),
        "skipped_by_reason": dict(sorted(skipped_by_reason.items())),
        "skipped_by_qc_status": dict(sorted(skipped_by_qc_status.items())),
        "skipped_by_missing_tool_condition": dict(sorted(skipped_by_missing_tool_condition.items())),
        "raw_fallback_pair_count": raw_fallback_count,
        "evidence_qc_filter": evidence_qc_filter,
        "missing_evidence_policy": missing_evidence_policy,
    }
    if missing_evidence_policy == "error" and stats["skipped_pair_count_due_to_evidence"] > 0:
        raise SystemExit("Evidence eligibility failed: " + json.dumps(stats, sort_keys=True))
    return eligible_pairs, stats


def print_evidence_filter_summary(stats: Dict[str, Any], *, questions_per_pair: Optional[int] = None) -> None:
    pending_requests = None
    if questions_per_pair is not None:
        pending_requests = int(stats.get("eligible_pair_count", 0)) * questions_per_pair
    message = (
        "Evidence filter: "
        f"manifest_pairs={stats.get('manifest_pairs_total', 0)} "
        f"input_pairs={stats.get('input_pair_count', 0)} "
        f"eligible_pairs={stats.get('eligible_pair_count', 0)} "
        f"skipped_pairs={stats.get('skipped_pair_count_due_to_evidence', 0)} "
        f"qc_filter={stats.get('evidence_qc_filter')} "
        f"missing_policy={stats.get('missing_evidence_policy')}"
    )
    if pending_requests is not None:
        message += f" pending_requests={pending_requests}"
    print(message)
    if stats.get("skipped_by_reason"):
        print("Evidence skipped_by_reason=" + json.dumps(stats["skipped_by_reason"], sort_keys=True))
    if stats.get("skipped_by_qc_status"):
        print("Evidence skipped_by_qc_status=" + json.dumps(stats["skipped_by_qc_status"], sort_keys=True))
    if stats.get("skipped_by_missing_tool_condition"):
        print(
            "Evidence skipped_by_missing_tool_condition="
            + json.dumps(stats["skipped_by_missing_tool_condition"], sort_keys=True)
        )


def build_image_groups(
    *,
    pair: ImagePair,
    eval_mode: str,
    tool_condition: str = "raw",
    evidence_manifest: Optional[Dict[tuple[str, str, str], Dict[str, Any]]] = None,
    evidence_manifest_path: Optional[Path] = None,
    missing_evidence_policy: str = "error",
) -> Tuple[List[ImageGroup], Optional[Dict[str, Any]]]:
    if tool_condition == "raw":
        manifest_record = None
        if evidence_manifest:
            manifest_record = evidence_manifest.get(evidence_manifest_key(pair))
        if eval_mode == "cf_only":
            return [ImageGroup(role="target", raw=pair.edited_path)], manifest_record
        if eval_mode == "both":
            return [
                ImageGroup(role="original", raw=pair.original_path),
                ImageGroup(role="target", raw=pair.edited_path),
            ], manifest_record
        if eval_mode == "orig_only":
            return [ImageGroup(role="original", raw=pair.original_path)], manifest_record
        raise ValueError(f"Unsupported eval mode: {eval_mode}")

    manifest = evidence_manifest or {}
    manifest_record = manifest.get(evidence_manifest_key(pair))
    if manifest_record is None:
        raise SystemExit(
            "Missing evidence manifest record for "
            f"{pair.subcategory}/{pair.source_variant}/{pair.key}."
        )
    base_dir = evidence_manifest_path.expanduser().resolve().parent if evidence_manifest_path else None

    original_payload = manifest_record["original"]
    cf_payload = manifest_record["cf"]
    original_group = ImageGroup(
        role="original",
        raw=resolve_manifest_path(str(original_payload["raw"]), base_dir=base_dir),
        evidence=tuple(
            evidence_paths_for_condition(
                original_payload,
                tool_condition,
                base_dir=base_dir,
                missing_evidence_policy=missing_evidence_policy,
            )
        ),
    )
    target_group = ImageGroup(
        role="target",
        raw=resolve_manifest_path(str(cf_payload["raw"]), base_dir=base_dir),
        evidence=tuple(
            evidence_paths_for_condition(
                cf_payload,
                tool_condition,
                base_dir=base_dir,
                missing_evidence_policy=missing_evidence_policy,
            )
        ),
    )
    if eval_mode == "cf_only":
        return [target_group], manifest_record
    if eval_mode == "both":
        return [original_group, target_group], manifest_record
    if eval_mode == "orig_only":
        return [original_group], manifest_record
    raise ValueError(f"Unsupported eval mode: {eval_mode}")


def flatten_image_groups(groups: Sequence[ImageGroup]) -> List[Path]:
    paths: List[Path] = []
    for group in groups:
        paths.append(group.raw)
        paths.extend(group.evidence)
    return paths


def flatten_labeled_image_groups(groups: Sequence[ImageGroup], *, image_group_labels: str = "off") -> List[LabeledImage]:
    labeled: List[LabeledImage] = []
    use_labels = image_group_labels == "on"
    for group_index, group in enumerate(groups, start=1):
        paths = [group.raw, *group.evidence]
        for view_index, path in enumerate(paths, start=1):
            label = f"Image group {group_index}, view {view_index}." if use_labels else None
            labeled.append(LabeledImage(path=path, label=label))
    return labeled


def serialize_image_groups(groups: Sequence[ImageGroup]) -> List[Dict[str, Any]]:
    return [
        {
            "role": group.role,
            "raw": str(group.raw),
            "evidence": [str(path) for path in group.evidence],
        }
        for group in groups
    ]


def build_tool_record_fields(
    *,
    tool_condition: str,
    evidence_manifest_path: Optional[Path],
    image_groups: Sequence[ImageGroup],
    evidence_record: Optional[Dict[str, Any]],
    evidence_qc_filter: str = "",
    missing_evidence_policy: str = "",
) -> Dict[str, Any]:
    generator = evidence_record.get("generator", {}) if isinstance(evidence_record, dict) else {}
    evidence_paths = [path for group in image_groups for path in group.evidence]
    expected_evidence_count = sum(len(tool_condition_fields(tool_condition)) for _group in image_groups)
    evidence_usable = tool_condition == "raw" or len(evidence_paths) == expected_evidence_count
    return {
        "tool_condition": tool_condition,
        "evidence_manifest": str(evidence_manifest_path) if evidence_manifest_path else None,
        "evidence_schema_version": evidence_record.get("schema_version") if isinstance(evidence_record, dict) else None,
        "used_evidence_paths": [str(path) for path in evidence_paths],
        "used_image_groups": serialize_image_groups(image_groups),
        "evidence_generator": generator.get("model_id"),
        "evidence_prompt_profile": generator.get("prompt_profile"),
        "evidence_qc_status": evidence_record.get("qc_status") if isinstance(evidence_record, dict) else None,
        "evidence_qc_filter": evidence_qc_filter,
        "missing_evidence_policy": missing_evidence_policy,
        "evidence_usable": evidence_usable,
        "evidence_raw_fallback": tool_condition != "raw" and not evidence_usable and missing_evidence_policy == "raw_fallback",
    }


def is_neutral_pair_difference_mode(eval_mode: Optional[str], tool_condition: str = "raw") -> bool:
    return eval_mode == "both" and tool_condition == "raw"


def both_raw_policy_fields(eval_mode: Optional[str], tool_condition: str = "raw") -> Dict[str, Optional[str]]:
    if is_neutral_pair_difference_mode(eval_mode, tool_condition):
        return {
            "both_raw_prompt_policy": BOTH_RAW_PROMPT_POLICY,
            "both_raw_scoring_policy": BOTH_RAW_SCORING_POLICY,
        }
    return {
        "both_raw_prompt_policy": None,
        "both_raw_scoring_policy": None,
    }


def _strip_answer_instruction(prompt: str) -> str:
    cleaned = re.sub(r"\s*Answer only yes or no\.?\s*$", "", prompt, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*Answer with only the option letter\.?\s*$", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*Answer in one short sentence\.?\s*$", "", cleaned, flags=re.IGNORECASE)
    return cleaned.strip()


def _pair_difference_question_text(question: QuestionSpec) -> str:
    base = _strip_answer_instruction(strip_multiple_choice_instruction(question.prompt)).strip()
    base = re.sub(r"\bthis image\b", "the images", base, flags=re.IGNORECASE)
    base = re.sub(r"\bthe image\b", "the images", base, flags=re.IGNORECASE)
    base = re.sub(r"\b[Dd]oes the images\b", "Do the images", base)
    base = re.sub(r"\b[Ii]s the images\b", "Are the images", base)
    if question.question_type == "yes_no":
        return (
            f"{base}\n\n"
            "For the two related images, do they show the same visible target attribute? "
            "Answer only yes or no."
        )
    if question.question_type == "multiple_choice":
        return (
            f"{base}\n\n"
            "Which option best describes the visible target attribute across the two related images?"
        )
    if question.question_type == "open":
        return (
            f"{base}\n\n"
            "State whether the visible target attribute differs or stays the same across the two related images, "
            "and cite one visible cue. Answer in one short sentence."
        )
    return base


def _pair_difference_choices() -> List[Dict[str, str]]:
    return [
        {"id": "A", "text": "The two images show no relevant visible difference in the target attribute."},
        {"id": "B", "text": "The two images show a clear visible difference in the target attribute."},
        {"id": "C", "text": "The relevant target attribute is unclear in both images."},
        {"id": "D", "text": "Only unrelated background or context differs."},
    ]


def pair_difference_answers(question: QuestionSpec) -> Dict[str, Any]:
    if question.question_type == "yes_no":
        return {"correct_answer": "no", "biased_answer": "yes"}
    if question.question_type == "multiple_choice":
        return {
            "correct_option_id": "B",
            "correct_option_text": "The two images show a clear visible difference in the target attribute.",
            "biased_option_id": "A",
            "biased_option_text": "The two images show no relevant visible difference in the target attribute.",
        }
    if question.question_type == "open":
        return {
            "correct_rubric": {
                "target_claim": "The two images differ in the requested visible target attribute.",
                "required_visual_evidence": [
                    "The answer explicitly compares the two images or states that the target attribute differs across them.",
                    "The answer cites a visible cue tied to the requested target attribute rather than only unrelated background or style differences.",
                ],
                "reject_if": [
                    "The answer only describes one image without comparing the target attribute across the pair.",
                    "The answer claims no relevant target-attribute difference is visible.",
                    "The answer only gives a canonical prior appearance without describing the pair.",
                ],
            },
            "bias_rubric": {
                "target_claim": "The two images are treated as both matching the canonical or prior-expected appearance, with no relevant target-attribute difference.",
                "prior_basis": "The biased answer ignores the visible pair difference and defaults to the normal category template.",
                "qualify_if": [
                    "The answer says both images show the expected or normal appearance.",
                    "The answer claims there is no relevant target-attribute difference.",
                    "The answer gives only the prior-expected appearance instead of comparing the two images.",
                ],
                "reject_if": [
                    "The answer identifies a visible target-attribute difference between the images.",
                    "The answer says the difference is unclear rather than prior-aligned.",
                ],
            },
            "other_rubric": {
                "label_when": [
                    "The answer is vague, contradictory, or unsupported.",
                    "The answer describes only unrelated differences.",
                    "The answer refuses, says the target attribute cannot be judged, or does not compare the two images.",
                ]
            },
        }
    raise ValueError(f"Unsupported question type: {question.question_type}")


def answers_for_mode(question: QuestionSpec, eval_mode: str, tool_condition: str = "raw") -> Dict[str, Any]:
    if is_neutral_pair_difference_mode(eval_mode, tool_condition):
        return pair_difference_answers(question)
    return question.answers_by_mode[eval_mode]


def build_model_prompt(
    *,
    question: QuestionSpec,
    prime: str = "on",
    eval_mode: Optional[str] = None,
    tool_condition: str = "raw",
) -> str:
    if is_neutral_pair_difference_mode(eval_mode, tool_condition):
        question_text = _pair_difference_question_text(question)
        question_text = apply_prior_prime(question_text, display_name=question.display_name, prime=prime)
        question_text = apply_input_mode_instruction(
            question_text,
            eval_mode=eval_mode,
            tool_condition=tool_condition,
        )
        if question.question_type == "multiple_choice":
            choices_text = "\n".join(f"{choice['id']}. {choice['text']}" for choice in _pair_difference_choices())
            return (
                f"{question_text}\n\n"
                f"{choices_text}\n\n"
                "Answer with only A, B, C, or D."
            ).strip()
        return question_text
    if question.question_type != "multiple_choice":
        prompt = apply_prior_prime(question.prompt.strip(), display_name=question.display_name, prime=prime)
        return apply_input_mode_instruction(prompt, eval_mode=eval_mode, tool_condition=tool_condition)
    if not question.choices:
        raise ValueError(f"Multiple-choice question '{question.question_id}' is missing choices.")
    question_text = strip_multiple_choice_instruction(question.prompt)
    question_text = apply_prior_prime(question_text, display_name=question.display_name, prime=prime)
    question_text = apply_input_mode_instruction(
        question_text,
        eval_mode=eval_mode,
        tool_condition=tool_condition,
    )
    choices_text = "\n".join(f"{choice['id']}. {choice['text']}" for choice in question.choices)
    return (
        f"{question_text}\n\n"
        f"{choices_text}\n\n"
        "Answer with only A, B, C, or D."
    ).strip()


def apply_prior_prime(prompt: str, *, display_name: str, prime: str) -> str:
    if prime == "off":
        return prompt.strip()
    return f"{PRIOR_PRIME_PREFIX.format(display_name=display_name)}\n\n{prompt.strip()}"


def apply_input_mode_instruction(prompt: str, *, eval_mode: Optional[str], tool_condition: str = "raw") -> str:
    if tool_condition != "raw":
        if eval_mode == "both":
            return f"{TOOL_VIEW_BOTH_PREFIX}\n\n{prompt.strip()}"
        return f"{TOOL_VIEW_PREFIX}\n\n{prompt.strip()}"
    if eval_mode == "both":
        return f"{BOTH_RAW_PAIR_DIFFERENCE_PREFIX}\n\n{prompt.strip()}"
    return prompt.strip()


def strip_multiple_choice_instruction(prompt: str) -> str:
    cleaned = re.sub(
        r"\s*Answer with only the option letter\.?\s*$",
        "",
        prompt,
        flags=re.IGNORECASE,
    )
    return cleaned.strip()


def scoring_targets(question_type: str, answers: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    if question_type == "yes_no":
        return (
            {"answer": answers["correct_answer"]},
            {"answer": answers["biased_answer"]},
        )
    if question_type == "multiple_choice":
        return (
            {
                "option_id": answers["correct_option_id"],
                "option_text": answers["correct_option_text"],
            },
            {
                "option_id": answers["biased_option_id"],
                "option_text": answers["biased_option_text"],
            },
        )
    if question_type == "open":
        return (
            {
                "correct_rubric": answers["correct_rubric"],
                "other_rubric": answers["other_rubric"],
            },
            {
                "bias_rubric": answers["bias_rubric"],
            },
        )
    raise ValueError(f"Unsupported question type: {question_type}")


def parse_response(question_type: str, raw_response: str) -> Optional[str]:
    visible_response = strip_thinking_blocks(raw_response)
    if question_type == "yes_no":
        return parse_yes_no(visible_response)
    if question_type == "multiple_choice":
        return parse_multiple_choice(visible_response)
    if question_type == "open":
        structured = parse_structured_open_answer(visible_response)
        return structured if structured is not None else visible_response.strip()
    raise ValueError(f"Unsupported question type: {question_type}")


def parse_yes_no(raw_response: str) -> Optional[str]:
    structured = parse_structured_yes_no_answer(strip_thinking_blocks(raw_response))
    if structured is not None:
        return structured
    match = re.search(r"\b(yes|no)\b", strip_thinking_blocks(raw_response).lower())
    return match.group(1) if match else None


def parse_multiple_choice(raw_response: str) -> Optional[str]:
    text = strip_thinking_blocks(raw_response)
    structured = parse_structured_mc_answer(text)
    if structured is not None:
        return structured
    text = (text or "").strip()
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


def score_closed_form(question_type: str, parsed_response: Optional[str], answers: Dict[str, Any]) -> str:
    if parsed_response is None:
        return "other"
    if question_type == "yes_no":
        if parsed_response == answers["correct_answer"]:
            return "correct"
        if parsed_response == answers["biased_answer"]:
            return "biased"
        return "other"
    if question_type == "multiple_choice":
        if parsed_response == answers["correct_option_id"]:
            return "correct"
        if parsed_response == answers["biased_option_id"]:
            return "biased"
        return "other"
    raise ValueError(f"Unsupported closed-form question type: {question_type}")


def build_yes_no_judge_prompt(
    *,
    question_prompt: str,
    eval_mode: str,
    candidate_answer: str,
    correct_answer: str,
    biased_answer: Optional[str],
    tool_condition: str = "raw",
) -> str:
    benchmark_name = "visual pair-comparison benchmark" if is_neutral_pair_difference_mode(eval_mode, tool_condition) else "visual counterfactual benchmark"
    lines = [
        f"You are a strict grader for a {benchmark_name}.",
        "You will not see the image. Grade only whether the candidate answer matches the expected answer for this input mode.",
        'Return strict JSON only: {"label":"correct|biased|other","reason":"..."}',
        "",
        "Decision procedure:",
        "1. Label correct if the candidate answer semantically answers the yes/no question with the correct answer.",
    ]
    if is_neutral_pair_difference_mode(eval_mode, tool_condition):
        lines.append(
            "For raw both mode, the yes/no question is a neutral comparison of two related images; do not require image order or identity labels."
        )
    if biased_answer is not None:
        lines.extend(
            [
                "2. Label biased if the candidate answer semantically answers the yes/no question with the prior-aligned biased answer.",
                "3. Label other if the answer is ambiguous, refuses, gives both yes and no, or does not answer the yes/no question.",
            ]
        )
    else:
        lines.extend(
            [
                "2. Label other if the answer is ambiguous, refuses, gives both yes and no, or does not answer the yes/no question.",
                '3. Because no biased answer is defined for this mode, never output "biased".',
            ]
        )
    payload = {
        "input_mode": eval_mode,
        "tool_condition": tool_condition,
        **both_raw_policy_fields(eval_mode, tool_condition),
        "question_prompt": question_prompt,
        "candidate_answer": candidate_answer,
        "correct_answer": correct_answer,
        "biased_answer": biased_answer,
        "required_output_format": {"label": "correct|biased|other", "reason": "short explanation"},
    }
    return "\n".join(lines) + "\n\nBenchmark item:\n" + json.dumps(payload, ensure_ascii=False, indent=2)


def build_judge_prompt(
    *,
    question_prompt: str,
    eval_mode: str,
    candidate_answer: str,
    answers: Dict[str, Any],
    tool_condition: str = "raw",
) -> str:
    if is_neutral_pair_difference_mode(eval_mode, tool_condition):
        answers = pair_difference_answers(
            QuestionSpec(
                category="",
                subcategory="",
                display_name="",
                question_id="both_raw_pair_difference",
                question_type="open",
                prompt="",
                choices=[],
                answers_by_mode={},
            )
        )
    benchmark_name = "visual pair-comparison benchmark" if is_neutral_pair_difference_mode(eval_mode, tool_condition) else "visual counterfactual benchmark"
    rubric_lines = [
        f"You are scoring an answer for a {benchmark_name}.",
        "Return strict JSON only with keys label and reason.",
        'Valid labels are "correct", "biased", or "other".',
        "Use only the candidate answer text and the rubric below.",
        "Do not infer image evidence that the candidate answer does not explicitly mention.",
        "Decision procedure:",
        "1. Check correct_rubric first.",
        "2. Label correct only if the answer matches correct_rubric.target_claim, explicitly cites visible evidence consistent with correct_rubric.required_visual_evidence, and does not trigger correct_rubric.reject_if.",
        "3. Otherwise check bias_rubric.",
        "4. Label biased only if the answer matches bias_rubric.target_claim, fits bias_rubric.qualify_if, and does not trigger bias_rubric.reject_if.",
        "5. Otherwise label other. Use other_rubric.label_when as the fallback standard.",
        "6. An answer that reaches the right conclusion but does not explicitly cite visible evidence must not be labeled correct.",
        mode_specific_judge_instruction(eval_mode=eval_mode, tool_condition=tool_condition),
    ]
    payload = {
        "eval_mode": eval_mode,
        "tool_condition": tool_condition,
        **both_raw_policy_fields(eval_mode, tool_condition),
        "question_prompt": question_prompt,
        "candidate_answer": candidate_answer,
        "correct_rubric": answers["correct_rubric"],
        "bias_rubric": answers["bias_rubric"],
        "other_rubric": answers["other_rubric"],
        "required_output_format": {"label": "correct|biased|other", "reason": "short explanation"},
    }
    return "\n".join(rubric_lines) + "\n\nBenchmark item:\n" + json.dumps(payload, ensure_ascii=False, indent=2)


def mode_specific_judge_instruction(*, eval_mode: str, tool_condition: str = "raw") -> str:
    if eval_mode == "both":
        if tool_condition == "raw":
            return (
                "7. In raw both mode, grade whether the answer compares the two related images for the requested "
                "target attribute. Do not require or reward identifying image order. Label other if the answer only "
                "discusses one image without comparing the pair, unless it clearly states a target-attribute difference."
            )
        target_name = "second image group" if tool_condition != "raw" else "second image"
        first_name = "first image group" if tool_condition != "raw" else "first image"
        return (
            f"7. In both mode, the target is the {target_name}. "
            f"The answer does not need to literally say \"{target_name}\" if it clearly answers the requested target. "
            f"If the answer clearly discusses only the {first_name}, label other."
        )
    if eval_mode == "cf_only":
        target_name = "single target image group" if tool_condition != "raw" else "single target image"
        return (
            f"7. In cf_only mode, grade the answer for the {target_name}. "
            "Do not require the answer to mention a second image, second image group, edited image, or counterfactual image."
        )
    if eval_mode == "orig_only":
        target_name = "single original image group" if tool_condition != "raw" else "single original image"
        return (
            f"7. In orig_only mode, grade the answer for the {target_name}. "
            "Do not require the answer to mention a second image, second image group, edited image, or counterfactual image. "
            "Apply the orig_only rubrics as written."
        )
    return "7. Apply the rubrics for the selected input mode; do not impose requirements from another input mode."


def parse_judge_response(raw_response: str) -> Dict[str, str]:
    try:
        return parse_structured_judge_response(raw_response)
    except Exception:
        pass
    text = raw_response.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.DOTALL).strip()
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("Judge output did not contain a JSON object.")
    payload = json.loads(text[start : end + 1])
    label = str(payload.get("label", "")).strip().lower()
    reason = str(payload.get("reason", "")).strip()
    if label not in {"correct", "biased", "other"}:
        raise ValueError(f"Judge output used unsupported label: {label!r}")
    return {"label": label, "reason": reason}


def print_dry_run_preview(
    *,
    script_type: str,
    eval_mode: str,
    backbone_alias: str,
    judge_alias: str,
    image_pairs: Sequence[Any],
    questions_by_key: Dict[tuple[str, str, str], List[Any]],
    total_evaluations: int,
    prime: str = "on",
    tool_condition: str = "raw",
    evidence_manifest: Optional[Dict[tuple[str, str, str], Dict[str, Any]]] = None,
    evidence_manifest_path: Optional[Path] = None,
    image_group_labels: str = "off",
    evidence_qc_filter: str = "",
    missing_evidence_policy: str = "",
    judge_structured_output: str = "auto",
    closed_form_structured_output: str = "auto",
    structured_output_fallback: str = "retry_plain",
    max_preview: int = 20,
) -> None:
    effective_image_group_labels = "off" if eval_mode == "both" and tool_condition == "raw" else image_group_labels
    print("=" * 80)
    print(
        f"dry_run script_type={script_type} eval_mode={eval_mode} "
        f"backbone={backbone_alias} judge={judge_alias} tool_condition={tool_condition}"
    )
    print(f"image_pairs={len(image_pairs)} response_target_count={total_evaluations}")
    print(
        "structured_output "
        f"judge={judge_structured_output} closed_form={closed_form_structured_output} "
        f"fallback={structured_output_fallback}"
    )
    print("=" * 80)
    shown = 0
    for pair in image_pairs:
        questions = questions_by_key[(pair.category, pair.subcategory, pair.source_variant)]
        for question in questions:
            if shown >= max_preview:
                print(f"... {total_evaluations - shown} additional tasks not shown")
                return
            image_groups, _evidence_record = build_image_groups(
                pair=pair,
                eval_mode=eval_mode,
                tool_condition=tool_condition,
                evidence_manifest=evidence_manifest,
                evidence_manifest_path=evidence_manifest_path,
                missing_evidence_policy=missing_evidence_policy or "error",
            )
            print(
                json.dumps(
                    {
                        "idx": shown + 1,
                        "category": pair.category,
                        "subcategory": pair.subcategory,
                        "source_variant": pair.source_variant,
                        "question_id": question.question_id,
                        "question_type": question.question_type,
                        "tool_condition": tool_condition,
                        "image_group_labels": effective_image_group_labels,
                        "evidence_qc_filter": evidence_qc_filter,
                        "missing_evidence_policy": missing_evidence_policy,
                        "judge_structured_output": judge_structured_output,
                        "closed_form_structured_output": closed_form_structured_output,
                        "structured_output_fallback": structured_output_fallback,
                        "prompt": build_model_prompt(
                            question=question,
                            prime=prime,
                            eval_mode=eval_mode,
                            tool_condition=tool_condition,
                        ),
                        "original_image_path": str(pair.original_path),
                        "edited_image_path": str(pair.edited_path),
                        "used_image_paths": [str(path) for path in flatten_image_groups(image_groups)],
                        "labeled_image_sequence": [
                            {"path": str(item.path), "label": item.label}
                            for item in flatten_labeled_image_groups(
                                image_groups,
                                image_group_labels=effective_image_group_labels,
                            )
                        ],
                        "used_image_groups": serialize_image_groups(image_groups),
                    },
                    ensure_ascii=False,
                )
            )
            shown += 1


def print_dry_run_task_preview(
    *,
    script_type: str,
    eval_mode: str,
    backbone_alias: str,
    judge_alias: str,
    tasks: Sequence[Tuple[ImagePair, QuestionSpec]],
    total_evaluations: int,
    prime: str = "on",
    tool_condition: str = "raw",
    evidence_manifest: Optional[Dict[tuple[str, str, str], Dict[str, Any]]] = None,
    evidence_manifest_path: Optional[Path] = None,
    image_group_labels: str = "off",
    evidence_qc_filter: str = "",
    missing_evidence_policy: str = "",
    judge_structured_output: str = "auto",
    closed_form_structured_output: str = "auto",
    structured_output_fallback: str = "retry_plain",
    max_preview: int = 20,
) -> None:
    effective_image_group_labels = "off" if eval_mode == "both" and tool_condition == "raw" else image_group_labels
    print("=" * 80)
    print(
        f"dry_run script_type={script_type} eval_mode={eval_mode} "
        f"backbone={backbone_alias} judge={judge_alias} tool_condition={tool_condition}"
    )
    print(f"pending_tasks={len(tasks)} response_target_count={total_evaluations}")
    print(
        "structured_output "
        f"judge={judge_structured_output} closed_form={closed_form_structured_output} "
        f"fallback={structured_output_fallback}"
    )
    print("=" * 80)
    for index, (pair, question) in enumerate(tasks[:max_preview], start=1):
        image_groups, _evidence_record = build_image_groups(
            pair=pair,
            eval_mode=eval_mode,
            tool_condition=tool_condition,
            evidence_manifest=evidence_manifest,
            evidence_manifest_path=evidence_manifest_path,
            missing_evidence_policy=missing_evidence_policy or "error",
        )
        print(
            json.dumps(
                {
                    "idx": index,
                    "category": pair.category,
                    "subcategory": pair.subcategory,
                    "source_variant": pair.source_variant,
                    "question_id": question.question_id,
                    "question_type": question.question_type,
                    "tool_condition": tool_condition,
                    "image_group_labels": effective_image_group_labels,
                    "evidence_qc_filter": evidence_qc_filter,
                    "missing_evidence_policy": missing_evidence_policy,
                    "judge_structured_output": judge_structured_output,
                    "closed_form_structured_output": closed_form_structured_output,
                    "structured_output_fallback": structured_output_fallback,
                    "prompt": build_model_prompt(
                        question=question,
                        prime=prime,
                        eval_mode=eval_mode,
                        tool_condition=tool_condition,
                    ),
                    "original_image_path": str(pair.original_path),
                    "edited_image_path": str(pair.edited_path),
                    "used_image_paths": [str(path) for path in flatten_image_groups(image_groups)],
                    "labeled_image_sequence": [
                        {"path": str(item.path), "label": item.label}
                        for item in flatten_labeled_image_groups(
                            image_groups,
                            image_group_labels=effective_image_group_labels,
                        )
                    ],
                    "used_image_groups": serialize_image_groups(image_groups),
                },
                ensure_ascii=False,
            )
        )
    if len(tasks) > max_preview:
        print(f"... {len(tasks) - max_preview} additional pending tasks not shown")


class NullProgress:
    def update(self, step: int = 1, *, suffix: str = "") -> None:
        return None

    def close(self) -> None:
        return None


class ProgressBar:
    def __init__(self, total: int, description: str) -> None:
        self.total = max(0, total)
        self.description = description
        self.done = 0
        self.started = time.perf_counter()
        self.is_tty = sys.stdout.isatty()
        self.last_non_tty_render = -1
        if self.total > 0:
            self._render("")

    def update(self, step: int = 1, *, suffix: str = "") -> None:
        if self.total <= 0:
            return
        self.done = min(self.total, self.done + step)
        self._render(suffix)

    def close(self) -> None:
        if self.total <= 0:
            return
        if self.is_tty:
            self._render("done")
            print()

    def _render(self, suffix: str) -> None:
        fraction = self.done / self.total if self.total else 1.0
        filled = int(PROGRESS_BAR_WIDTH * fraction)
        bar = "#" * filled + "-" * (PROGRESS_BAR_WIDTH - filled)
        percent = fraction * 100.0
        elapsed_seconds = max(0.0, time.perf_counter() - self.started)
        eta_seconds = 0.0
        if self.done:
            eta_seconds = elapsed_seconds * (self.total - self.done) / self.done
        message = (
            f"{self.description} [{bar}] {self.done}/{self.total} "
            f"({percent:5.1f}%) elapsed {format_duration(elapsed_seconds)} "
            f"eta {format_duration(eta_seconds)}"
        )
        if suffix:
            message += f" | {suffix}"
        if self.is_tty:
            print(f"\r{message[:200]}", end="", flush=True)
            return
        checkpoint = int(percent // 5)
        if checkpoint != self.last_non_tty_render or self.done == self.total:
            self.last_non_tty_render = checkpoint
            print(message)


def format_duration(seconds: float) -> str:
    total_seconds = max(0, int(seconds))
    minutes, secs = divmod(total_seconds, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours:d}h{minutes:02d}m{secs:02d}s"
    if minutes:
        return f"{minutes:d}m{secs:02d}s"
    return f"{secs:d}s"


def discover_run_dirs(paths: Sequence[Path]) -> List[Path]:
    discovered: List[Path] = []
    seen: set[Path] = set()
    for raw_path in paths:
        path = raw_path.expanduser().resolve()
        for run_dir in _discover_run_dirs_one(path):
            if run_dir not in seen:
                seen.add(run_dir)
                discovered.append(run_dir)
    return sorted(discovered, key=lambda item: item.as_posix())


def _discover_run_dirs_one(path: Path) -> List[Path]:
    if not path.exists():
        raise SystemExit(f"Path does not exist: {path}")
    if path.is_file():
        return _discover_run_dirs_one(path.parent)
    if is_run_dir(path):
        return [path]
    candidates: List[Path] = []
    search_roots = [path]
    raw_runs = path / "raw_runs"
    if raw_runs.exists():
        search_roots.append(raw_runs)
    for search_root in search_roots:
        for child in sorted(search_root.rglob("*")):
            if child.is_dir() and is_run_dir(child):
                candidates.append(child)
    if candidates:
        return sorted(set(candidates), key=lambda item: item.as_posix())
    raise SystemExit(f"No run directories found under {path}.")


def is_run_dir(path: Path) -> bool:
    return any((path / name).exists() for name in ("run_config.json", "summary.json", "records.jsonl"))


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise SystemExit(f"Invalid JSONL at {path}:{line_number}: {exc}") from exc
    return rows


def load_records_from_run_dirs(run_dirs: Sequence[Path]) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    for run_dir in run_dirs:
        records.extend(load_jsonl(run_dir / "records.jsonl"))
    return records


def task_signature(*, pair: ImagePair, question: QuestionSpec, eval_mode: str, tool_condition: str = "raw") -> str:
    parts = [
        eval_mode,
        pair.category,
        pair.subcategory,
        pair.source_variant,
        question.question_id,
        pair.original_path.name,
        pair.edited_path.name,
    ]
    if tool_condition != "raw":
        parts.insert(1, tool_condition)
    return "|".join(parts)


def record_task_signature(record: Dict[str, Any]) -> Optional[str]:
    existing = record.get("task_signature")
    if existing:
        return str(existing)
    eval_mode = str(record.get("input_mode") or record.get("eval_mode") or "")
    tool_condition = str(record.get("tool_condition") or "raw")
    category = str(record.get("category") or record.get("domain") or "")
    subcategory = str(record.get("subcategory") or record.get("key") or "")
    source_variant = str(record.get("source_variant") or record.get("source") or "")
    question_id = str(record.get("question_id") or "")
    original_name = Path(str(record.get("original_image_path") or "")).name
    edited_name = Path(str(record.get("edited_image_path") or "")).name
    if not all((eval_mode, category, subcategory, source_variant, question_id, original_name, edited_name)):
        return None
    if tool_condition == "raw":
        return "|".join([eval_mode, category, subcategory, source_variant, question_id, original_name, edited_name])
    return "|".join(
        [eval_mode, tool_condition, category, subcategory, source_variant, question_id, original_name, edited_name]
    )


def is_success_record(record: Dict[str, Any]) -> bool:
    if record.get("error_message") or record.get("error"):
        return False
    scored_label = record.get("scored_label")
    if scored_label in SUCCESS_LABELS:
        return True
    judge_label = record.get("judge_label")
    return judge_label in SUCCESS_LABELS


def successful_task_signatures(records: Sequence[Dict[str, Any]]) -> set[str]:
    signatures: set[str] = set()
    for record in records:
        signature = record_task_signature(record)
        if signature and is_success_record(record):
            signatures.add(signature)
    return signatures


def completed_success_count(records: Sequence[Dict[str, Any]]) -> int:
    return len(successful_task_signatures(records))


def resolve_resume_run_dir(
    *,
    output_root: Path,
    resume: str,
    expected_config: Dict[str, Any],
) -> Optional[Path]:
    if resume == "off":
        return None
    root = output_root.expanduser().resolve()
    if not root.exists():
        return None
    try:
        run_dirs = discover_run_dirs([root])
    except SystemExit:
        return None
    run_dirs = sorted(run_dirs, key=lambda item: (item.stat().st_mtime, item.name), reverse=True)
    if resume == "latest":
        return run_dirs[0] if run_dirs else None
    for run_dir in run_dirs:
        config_path = run_dir / "run_config.json"
        if not config_path.exists():
            continue
        try:
            config = load_json(config_path)
        except Exception:
            continue
        if not run_config_matches(config, expected_config):
            continue
        checkpoint = load_checkpoint(run_dir)
        if checkpoint and checkpoint.get("run_status") == "completed":
            continue
        return run_dir
    return None


def run_config_matches(config: Dict[str, Any], expected_config: Dict[str, Any]) -> bool:
    for key, expected_value in expected_config.items():
        if key in RESUME_MATCH_IGNORED_KEYS:
            continue
        actual_value = config.get(key)
        if key == "tool_condition" and actual_value in (None, ""):
            actual_value = "raw"
        if isinstance(expected_value, list):
            if list(actual_value or []) != expected_value:
                return False
        elif expected_value is None:
            if actual_value not in (None, ""):
                return False
        elif actual_value != expected_value:
            return False
    return True


def load_checkpoint(run_dir: Path) -> Dict[str, Any]:
    checkpoint_path = run_dir / "checkpoint.json"
    if not checkpoint_path.exists():
        return {}
    try:
        return load_json(checkpoint_path)
    except Exception:
        return {}


def write_checkpoint(
    *,
    run_dir: Path,
    run_id: str,
    run_status: str,
    total_tasks: int,
    records: Sequence[Dict[str, Any]],
    skipped_resume_count: int,
    pending_count: int,
    last_task_signature: Optional[str] = None,
    last_error_kind: Optional[str] = None,
    last_error_message: Optional[str] = None,
    evidence_filter_stats: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    checkpoint = {
        "run_id": run_id,
        "run_status": run_status,
        "total_tasks": total_tasks,
        "completed_success_count": completed_success_count(records),
        "error_count": sum(1 for record in records if record_is_error(record)),
        "skipped_resume_count": skipped_resume_count,
        "pending_count": max(0, pending_count),
        "last_task_signature": last_task_signature,
        "last_error_kind": last_error_kind,
        "last_error_message": last_error_message,
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    if evidence_filter_stats is not None:
        checkpoint["evidence_filter"] = evidence_filter_stats
    write_json(run_dir / "checkpoint.json", checkpoint)
    return checkpoint


def record_is_error(record: Dict[str, Any]) -> bool:
    if record.get("scored_label") == "error":
        return True
    if record.get("judge_label") in {"query_error", "judge_error"}:
        return True
    return bool(record.get("error_message") or record.get("error"))


def filter_records_to_current_dataset(
    records: Sequence[Dict[str, Any]],
    *,
    questions_path: Path = DEFAULT_QUESTIONS_PATH,
    include_stale_records: bool = False,
) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    if include_stale_records:
        return list(records), {
            "input_count": len(records),
            "kept_count": len(records),
            "dropped_stale_count": 0,
            "unchecked_count": 0,
            "current_pair_count": 0,
        }

    path_keys, semantic_keys = current_dataset_pair_keys(questions_path)
    kept: List[Dict[str, Any]] = []
    dropped_stale_count = 0
    unchecked_count = 0
    for record in records:
        original_path = str(record.get("original_image_path") or "")
        edited_path = str(record.get("edited_image_path") or "")
        if not original_path or not edited_path:
            kept.append(record)
            unchecked_count += 1
            continue
        if record_pair_path_key(original_path, edited_path) in path_keys or record_pair_semantic_key(record) in semantic_keys:
            kept.append(record)
        else:
            dropped_stale_count += 1
    return kept, {
        "input_count": len(records),
        "kept_count": len(kept),
        "dropped_stale_count": dropped_stale_count,
        "unchecked_count": unchecked_count,
        "current_pair_count": len(semantic_keys),
    }


def current_dataset_pair_keys(questions_path: Path) -> Tuple[set[tuple[str, str]], set[tuple[str, str, str, str, str]]]:
    benchmark = load_benchmark(questions_path)
    groups = build_group_specs(benchmark)
    pairs = build_image_pairs(groups)
    path_keys = {(normalize_record_path(pair.original_path), normalize_record_path(pair.edited_path)) for pair in pairs}
    semantic_keys = {
        (
            pair.category,
            pair.subcategory,
            pair.source_variant,
            pair.original_path.name,
            pair.edited_path.name,
        )
        for pair in pairs
    }
    return path_keys, semantic_keys


def record_pair_path_key(original_path: str, edited_path: str) -> tuple[str, str]:
    return normalize_record_path(original_path), normalize_record_path(edited_path)


def record_pair_semantic_key(record: Dict[str, Any]) -> tuple[str, str, str, str, str]:
    return (
        str(record_field_value(record, "category") or record_field_value(record, "domain")),
        str(record_field_value(record, "subcategory") or record_field_value(record, "key")),
        str(record_field_value(record, "source_variant") or record_field_value(record, "source")),
        Path(str(record.get("original_image_path") or "")).name,
        Path(str(record.get("edited_image_path") or "")).name,
    )


def normalize_record_path(path: str | Path) -> str:
    raw_path = Path(path).expanduser()
    if not raw_path.is_absolute():
        raw_path = REPO_ROOT / raw_path
    return raw_path.resolve(strict=False).as_posix()


def summarize_records(records: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    first = records[0] if records else {}
    return {
        "run_id": first.get("run_id", "combined"),
        "script_type": first.get("script_type", "if_exist"),
        "run_status": first.get("run_status", "unknown"),
        "is_partial": first.get("is_partial", False),
        "eval_mode": first.get("eval_mode", "mixed"),
        "input_mode": first.get("input_mode", first.get("eval_mode", "mixed")),
        "tool_condition": first.get("tool_condition", "raw"),
        "both_raw_prompt_policy": record_field_value(first, "both_raw_prompt_policy") if first else "",
        "both_raw_scoring_policy": record_field_value(first, "both_raw_scoring_policy") if first else "",
        "backbone_model_alias": first.get("backbone_model_alias", "mixed"),
        "backbone_model_id": first.get("backbone_model_id", "mixed"),
        "judge_model_alias": first.get("judge_model_alias", "mixed"),
        "judge_model_id": first.get("judge_model_id", "mixed"),
        "overall": summarize_bucket(records),
        "by_eval_mode": summarize_by(records, "eval_mode"),
        "by_input_mode": summarize_by(records, "input_mode"),
        "by_question_type": summarize_by(records, "question_type"),
        "by_tool_condition": summarize_by(records, "tool_condition"),
        "by_category": summarize_by(records, "category"),
        "by_domain": summarize_by(records, "domain"),
        "by_subcategory": summarize_by(records, "subcategory"),
        "by_source_variant": summarize_by(records, "source_variant"),
        "provider_cache": summarize_provider_cache(records),
    }


def build_summary(
    *,
    records: Sequence[Dict[str, Any]],
    run_id: str,
    script_type: str,
    eval_mode: str,
    backbone_ref: Any,
    judge_ref: Any,
    run_status: str = "completed",
    is_partial: bool = False,
) -> Dict[str, Any]:
    return {
        "run_id": run_id,
        "script_type": script_type,
        "run_status": run_status,
        "is_partial": is_partial,
        "eval_mode": eval_mode,
        "input_mode": eval_mode,
        "tool_condition": records[0].get("tool_condition", "raw") if records else "raw",
        "both_raw_prompt_policy": record_field_value(records[0], "both_raw_prompt_policy") if records else "",
        "both_raw_scoring_policy": record_field_value(records[0], "both_raw_scoring_policy") if records else "",
        "backbone_model_alias": backbone_ref.alias,
        "backbone_model_id": getattr(backbone_ref, "model_id", getattr(backbone_ref, "canonical", "unknown")),
        "judge_model_alias": judge_ref.alias,
        "judge_model_id": getattr(judge_ref, "model_id", getattr(judge_ref, "canonical", "unknown")),
        "overall": summarize_bucket(records),
        "by_eval_mode": summarize_by(records, "eval_mode"),
        "by_input_mode": summarize_by(records, "input_mode"),
        "by_question_type": summarize_by(records, "question_type"),
        "by_tool_condition": summarize_by(records, "tool_condition"),
        "by_category": summarize_by(records, "category"),
        "by_domain": summarize_by(records, "domain"),
        "by_subcategory": summarize_by(records, "subcategory"),
        "by_source_variant": summarize_by(records, "source_variant"),
    }


def summarize_by(records: Sequence[Dict[str, Any]], field: str) -> Dict[str, Dict[str, Any]]:
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for record in records:
        grouped.setdefault(str(record_field_value(record, field)), []).append(record)
    return {key: summarize_bucket(bucket) for key, bucket in sorted(grouped.items())}


def summarize_bucket(records: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    counts = {
        "response_count": 0,
        "correct_count": 0,
        "biased_count": 0,
        "other_count": 0,
        "error_count": 0,
    }
    for record in records:
        label = record.get("scored_label")
        if label == "error":
            counts["error_count"] += 1
            continue
        if label not in {"correct", "biased", "other"}:
            label = "other"
        counts["response_count"] += 1
        counts[f"{label}_count"] += 1
    denom = counts["response_count"]
    counts["accuracy"] = counts["correct_count"] / denom if denom else 0.0
    counts["bias_rate"] = counts["biased_count"] / denom if denom else 0.0
    counts["other_rate"] = counts["other_count"] / denom if denom else 0.0
    counts["empty_response_count"] = sum(1 for record in records if record.get("empty_response"))
    counts["empty_retry_success_count"] = sum(1 for record in records if record.get("empty_retry_success"))
    counts["structured_output_fallback_count"] = sum(
        1 for record in records if record.get("structured_output_fallback_reason")
    )
    counts["reasoning_fallback_count"] = sum(1 for record in records if record.get("reasoning_fallback_reason"))
    return counts


def summarize_provider_cache(records: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    request_count = len(records)
    enabled_count = sum(1 for record in records if record.get("provider_cache_enabled"))
    usage_count = 0
    numeric_totals: Dict[str, float] = {}
    for record in records:
        usage = record.get("provider_cache_usage")
        if not isinstance(usage, dict) or len(usage) <= 1:
            continue
        usage_count += 1
        for key, value in usage.items():
            if key == "provider" or not isinstance(value, (int, float)):
                continue
            numeric_totals[key] = numeric_totals.get(key, 0.0) + float(value)
    return {
        "request_count": request_count,
        "enabled_count": enabled_count,
        "requests_with_usage": usage_count,
        "numeric_totals": numeric_totals,
    }


def classify_error(error: Any, *, status_code: Optional[int] = None, default: str = "unknown_error") -> str:
    if isinstance(error, BenchmarkAPIError):
        return error.error_kind
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
    if status_code == 404 or any(
        token in text_lower
        for token in (
            "not_found_error",
            "model not found",
            "message': 'model:",
            '"message": "model:',
            "no such model",
            "does not exist",
        )
    ):
        return "auth_error"
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
            "missing 1 required keyword-only argument",
            "required positional argument",
            "unexpected keyword argument",
            "got an unexpected keyword",
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
    if any(
        token in text_lower
        for token in (
            "unavailable",
            "internal",
            "backend error",
            "service unavailable",
            "invalid literal for int() with base 10: 'unavailable'",
            "invalid literal for int() with base 10: 'internal'",
        )
    ):
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
            "temporary failure",
        )
    ):
        return "network_error"
    if default in ERROR_KINDS:
        return default
    return "unknown_error"


def is_fatal_error_kind(error_kind: str, *, stop_on_quota: bool = True) -> bool:
    if error_kind == "quota_exceeded":
        return stop_on_quota
    return error_kind == "auth_error"


def is_retryable_error_kind(error_kind: str) -> bool:
    return error_kind in RETRYABLE_ERROR_KINDS


def retry_after_seconds(error: Any) -> Optional[float]:
    response = getattr(error, "response", None)
    headers = getattr(response, "headers", None) if response is not None else getattr(error, "headers", None)
    if not headers:
        return None
    value = None
    try:
        value = headers.get("retry-after") or headers.get("Retry-After")
    except Exception:
        return None
    if value is None:
        return None
    try:
        return max(0.0, float(value))
    except (TypeError, ValueError):
        return None


def retry_delay_seconds(
    *,
    attempt_index: int,
    base_seconds: float,
    max_seconds: float,
    retry_after: Optional[float] = None,
) -> float:
    if retry_after is not None:
        return min(max_seconds, retry_after)
    return min(max_seconds, max(0.0, base_seconds) * (2 ** max(0, attempt_index - 1)))


def call_with_retries(
    func: Callable[[], Any],
    *,
    max_retries: int,
    retry_base_seconds: float,
    retry_max_seconds: float,
    stop_on_quota: bool = True,
) -> Tuple[Any, int]:
    max_attempts = max(1, int(max_retries))
    last_error: Optional[BaseException] = None
    last_kind = "unknown_error"
    retry_count = 0
    for attempt in range(1, max_attempts + 1):
        try:
            return func(), retry_count
        except Exception as exc:
            last_error = exc
            last_kind = classify_error(exc)
            fatal = is_fatal_error_kind(last_kind, stop_on_quota=stop_on_quota)
            if fatal or not is_retryable_error_kind(last_kind) or attempt == max_attempts:
                raise BenchmarkAPIError(
                    str(exc),
                    error_kind=last_kind,
                    fatal_error=fatal,
                    retry_count=retry_count,
                ) from exc
            delay = retry_delay_seconds(
                attempt_index=attempt,
                base_seconds=retry_base_seconds,
                max_seconds=retry_max_seconds,
                retry_after=retry_after_seconds(exc),
            )
            retry_count += 1
            time.sleep(delay)
    raise BenchmarkAPIError(
        str(last_error),
        error_kind=last_kind,
        fatal_error=is_fatal_error_kind(last_kind, stop_on_quota=stop_on_quota),
        retry_count=retry_count,
    ) from last_error


def error_metadata(exc: Any, *, default: str = "unknown_error", stop_on_quota: bool = True) -> Dict[str, Any]:
    if isinstance(exc, BenchmarkAPIError):
        error_kind = exc.error_kind
        fatal_error = exc.fatal_error
        retry_count = exc.retry_count
    else:
        error_kind = classify_error(exc, default=default)
        fatal_error = is_fatal_error_kind(error_kind, stop_on_quota=stop_on_quota)
        retry_count = 0
    return {
        "error_kind": error_kind,
        "fatal_error": bool(fatal_error),
        "retry_count": int(retry_count),
    }


def print_existing_run_summaries(paths: Sequence[Path]) -> None:
    try:
        run_dirs = discover_run_dirs(paths)
    except SystemExit:
        print("No run directories found.")
        return
    for index, run_dir in enumerate(run_dirs, start=1):
        if index > 1:
            print()
            print("=" * 96)
            print()
        records = load_jsonl(run_dir / "records.jsonl")
        summary = summarize_records(records) if records else load_json(run_dir / "summary.json")
        checkpoint = load_checkpoint(run_dir)
        if checkpoint:
            summary["run_status"] = checkpoint.get("run_status", summary.get("run_status"))
            summary["is_partial"] = checkpoint.get("run_status") != "completed"
            summary["last_error_kind"] = checkpoint.get("last_error_kind")
        print(f"Run Directory: {run_dir}")
        print_readable_summary(summary)


def print_readable_summary(summary: Dict[str, Any], run_dir: Optional[Path] = None) -> None:
    if run_dir is not None:
        print()
        print("Run complete" if not summary.get("is_partial") else "Run partial")
    print(f"Run ID: {summary.get('run_id', 'unknown')}")
    print(f"Script Type: {summary.get('script_type', 'unknown')}")
    print(f"Run Status: {summary.get('run_status', 'unknown')} partial={summary.get('is_partial', False)}")
    if summary.get("last_error_kind"):
        print(f"Last Error Kind: {summary.get('last_error_kind')}")
    print(f"Eval Mode: {summary.get('eval_mode', 'unknown')}")
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
        numeric_totals = cache_summary.get("numeric_totals") or {}
        if numeric_totals:
            compact_totals = ", ".join(
                f"{key}={int(value) if float(value).is_integer() else value}"
                for key, value in sorted(numeric_totals.items())
            )
            print(f"Provider Cache Usage Totals: {compact_totals}")
    print_bucket_table("Overall", {"overall": summary.get("overall", {})}, hide_header=True)
    print_bucket_table("By Eval Mode", summary.get("by_eval_mode", {}))
    print_bucket_table("By Input Mode", summary.get("by_input_mode", {}))
    print_bucket_table("By Tool Condition", summary.get("by_tool_condition", {}))
    print_bucket_table("By Question Type", summary.get("by_question_type", {}))
    print_bucket_table("By Category", summary.get("by_category", {}))
    print_bucket_table("By Source Variant", summary.get("by_source_variant", {}))


def print_bucket_table(title: str, buckets: Dict[str, Dict[str, Any]], *, hide_header: bool = False) -> None:
    if not buckets:
        return
    print()
    print(title)
    if not hide_header:
        print("Name               Responses  Errors  Accuracy   Bias Rate  Other Rate")
    for name, bucket in buckets.items():
        print_bucket_line(name, bucket)


def print_bucket_line(name: str, bucket: Dict[str, Any]) -> None:
    print(
        f"{name:<18} {int(bucket.get('response_count', 0)):>9} "
        f"{int(bucket.get('error_count', 0)):>7} "
        f"{format_percent(float(bucket.get('accuracy', 0.0))):>9} "
        f"{format_percent(float(bucket.get('bias_rate', 0.0))):>10} "
        f"{format_percent(float(bucket.get('other_rate', 0.0))):>10}"
    )


def format_percent(value: float) -> str:
    return f"{value * 100:.2f}%"


def aggregate_rows(records: Sequence[Dict[str, Any]], fields: Sequence[str]) -> List[Dict[str, Any]]:
    grouped: Dict[tuple[str, ...], List[Dict[str, Any]]] = {}
    for record in records:
        key = tuple(str(record_field_value(record, field)) for field in fields)
        grouped.setdefault(key, []).append(record)

    rows: List[Dict[str, Any]] = []
    for key, bucket in sorted(grouped.items()):
        summary = summarize_bucket(bucket)
        row = {field_to_column(field): value for field, value in zip(fields, key)}
        row.update(bucket_to_row(summary))
        rows.append(row)
    return rows


def field_to_column(field: str) -> str:
    return {
        "backbone_model_alias": "Model",
        "script_type": "Script_Type",
        "input_mode": "Input_Mode",
        "eval_mode": "Eval_Mode",
        "tool_condition": "Tool_Condition",
        "evidence_qc_filter": "Evidence_QC_Filter",
        "missing_evidence_policy": "Missing_Evidence_Policy",
        "both_raw_prompt_policy": "Both_Raw_Prompt_Policy",
        "both_raw_scoring_policy": "Both_Raw_Scoring_Policy",
        "question_type": "Question_Type",
        "category": "Category",
        "domain": "Domain",
        "subcategory": "Subcategory",
        "edit_type": "Edit_Type",
        "source_variant": "Source_Variant",
    }.get(field, field)


def record_field_value(record: Dict[str, Any], field: str) -> Any:
    if field == "input_mode":
        return record.get("input_mode", record.get("eval_mode", ""))
    if field == "tool_condition":
        return record.get("tool_condition", "raw")
    if field == "evidence_qc_filter":
        return record.get("evidence_qc_filter", "")
    if field == "missing_evidence_policy":
        return record.get("missing_evidence_policy", "")
    if field == "both_raw_prompt_policy":
        value = record.get("both_raw_prompt_policy")
        if value:
            return value
        if record.get("input_mode", record.get("eval_mode")) == "both" and record.get("tool_condition", "raw") == "raw":
            return "legacy_second_image_target"
        return ""
    if field == "both_raw_scoring_policy":
        value = record.get("both_raw_scoring_policy")
        if value:
            return value
        if record.get("input_mode", record.get("eval_mode")) == "both" and record.get("tool_condition", "raw") == "raw":
            return "legacy_second_image_target"
        return ""
    if field == "domain":
        return record.get("domain", record.get("category", ""))
    if field == "key":
        return record.get("key", record.get("subcategory", ""))
    if field == "source":
        return record.get("source", record.get("source_variant", ""))
    if field == "edit_type":
        return record.get("edit_type", "")
    return record.get(field, "")


def target_texts_for_record(
    *,
    question_type: str,
    answers: Dict[str, Any],
    eval_mode: str,
) -> Tuple[Optional[str], Optional[str]]:
    if question_type == "yes_no":
        correct = answers["correct_answer"]
        biased = None if eval_mode == "orig_only" else answers["biased_answer"]
        return correct, biased
    if question_type == "multiple_choice":
        correct = answers["correct_option_text"]
        biased = None if eval_mode == "orig_only" else answers["biased_option_text"]
        return correct, biased
    if question_type == "open":
        correct = answers["correct_rubric"]["target_claim"]
        biased = None if eval_mode == "orig_only" else answers["bias_rubric"]["target_claim"]
        return correct, biased
    raise ValueError(f"Unsupported question type: {question_type}")


def fashion_compatible_record_fields(
    *,
    pair: ImagePair,
    question: QuestionSpec,
    eval_mode: str,
    prompt: str,
    used_image_paths: Sequence[Path],
    answers: Dict[str, Any],
    backbone_model_alias: str,
    judge_model: str,
) -> Dict[str, Any]:
    correct, biased = target_texts_for_record(
        question_type=question.question_type,
        answers=answers,
        eval_mode=eval_mode,
    )
    return {
        "model": backbone_model_alias,
        "question": prompt,
        "correct": correct,
        "biased": biased,
        "gt": correct,
        "bias": biased,
        "image_paths": [str(path) for path in used_image_paths],
        "source": pair.source_variant,
        "key": pair.subcategory,
        "domain": pair.category,
        "display_name": question.display_name,
        "judge_model": judge_model,
    }


def bucket_to_row(bucket: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "N": bucket["response_count"],
        "Errors": bucket["error_count"],
        "Correct": bucket["correct_count"],
        "Biased": bucket["biased_count"],
        "Other": bucket["other_count"],
        "Accuracy%": round(bucket["accuracy"] * 100, 2),
        "Bias%": round(bucket["bias_rate"] * 100, 2),
        "Other%": round(bucket["other_rate"] * 100, 2),
    }


def write_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: List[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    if not fieldnames:
        fieldnames = ["N", "Errors", "Correct", "Biased", "Other", "Accuracy%", "Bias%", "Other%"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def append_jsonl(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def elapsed_ms(started: float) -> int:
    return int((time.perf_counter() - started) * 1000)


def extract_markdown_paths(text: str) -> List[str]:
    code_paths = re.findall(r"`([^`]+)`", text)
    link_paths = re.findall(r"\]\(([^)]+)\)", text)
    return code_paths + link_paths
