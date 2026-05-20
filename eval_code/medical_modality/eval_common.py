#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import random
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple


REPO_ROOT = Path(__file__).resolve().parents[2]
IF_EXIST_COMMON_PATH = REPO_ROOT / "eval_code" / "if_exist" / "eval_common.py"

_spec = importlib.util.spec_from_file_location("_if_exist_eval_common", IF_EXIST_COMMON_PATH)
if _spec is None or _spec.loader is None:
    raise RuntimeError(f"Cannot load shared eval helpers from {IF_EXIST_COMMON_PATH}")
_base = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = _base
_spec.loader.exec_module(_base)


DEFAULT_QUESTIONS_PATH = REPO_ROOT / "medical" / "modality_swapping" / "medical_modality_questions.json"
DEFAULT_ANNOTATIONS_PATH = REPO_ROOT / "eval_results" / "medical_modality" / "metadata" / "medical_modality_metadata.json"
DEFAULT_RESULTS_ROOT = REPO_ROOT / "eval_results" / "medical_modality"
DEFAULT_RAW_RUNS_ROOT = DEFAULT_RESULTS_ROOT / "raw_runs"
DEFAULT_TABLES_ROOT = DEFAULT_RESULTS_ROOT / "tables"
DEFAULT_FIGURES_ROOT = DEFAULT_RESULTS_ROOT / "figures"
DEFAULT_REPORTS_ROOT = DEFAULT_RESULTS_ROOT / "reports"
DEFAULT_METADATA_ROOT = DEFAULT_RESULTS_ROOT / "metadata"

QUESTION_TYPES = ("yes_no", "multiple_choice", "open")
SOURCE_VARIANTS = ("clean", "overlay")
ANSWER_MODES = ("cf_only", "both", "orig_only")
TOOL_CONDITIONS = ("raw", "bbox", "contour", "crop", "zoom_panel")
EVIDENCE_QC_FILTERS = _base.EVIDENCE_QC_FILTERS
MISSING_EVIDENCE_POLICIES = _base.MISSING_EVIDENCE_POLICIES
EVIDENCE_SCHEMA_VERSION = "medical_label_evidence_v1"
PROGRESS_BAR_WIDTH = _base.PROGRESS_BAR_WIDTH
SUCCESS_LABELS = _base.SUCCESS_LABELS
ERROR_KINDS = _base.ERROR_KINDS
FATAL_ERROR_KINDS = _base.FATAL_ERROR_KINDS
RETRYABLE_ERROR_KINDS = _base.RETRYABLE_ERROR_KINDS
RESUME_MATCH_IGNORED_KEYS = _base.RESUME_MATCH_IGNORED_KEYS
PRIOR_PRIME_PREFIX = _base.PRIOR_PRIME_PREFIX
BOTH_MODE_TARGET_INSTRUCTION = _base.BOTH_MODE_TARGET_INSTRUCTION
BOTH_RAW_PROMPT_POLICY = _base.BOTH_RAW_PROMPT_POLICY
BOTH_RAW_SCORING_POLICY = _base.BOTH_RAW_SCORING_POLICY
BOTH_RAW_PAIR_DIFFERENCE_PREFIX = _base.BOTH_RAW_PAIR_DIFFERENCE_PREFIX
TOOL_VIEW_PREFIX = _base.TOOL_VIEW_PREFIX
TOOL_VIEW_BOTH_PREFIX = _base.TOOL_VIEW_BOTH_PREFIX
MEDICAL_TOOL_VIEW_PREFIX = (
    "Additional images, if provided, show alternative visual views of the same MRI slice. "
    "Use the provided visual information to answer the question."
)
MEDICAL_TOOL_VIEW_BOTH_PREFIX = (
    "If two MRI image groups are provided, answer about the second image group. "
    "Additional images in each group show alternative visual views of the same MRI slice."
)


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
class MedicalAnnotation:
    category: str
    subset: str
    subcategory: str
    source_variant: str
    key: str
    original_path: Path
    edited_path: Path
    question_group_id: str
    edit_type: str
    count_direction: str
    dataset: str
    case_id: str
    visual_version: str
    swap_direction: str
    background_modality: str
    tumor_region_modality: str
    biased_modality: str
    display_name: str
    annotation_pair_key: str


@dataclass(frozen=True)
class GroupSpec:
    category: str
    subset: str
    subcategory: str
    display_name: str
    source_variant: str
    question_group_id: str
    edit_type: str
    count_direction: str
    visual_version: str
    dataset: str
    questions: List[QuestionSpec]
    annotations: List[MedicalAnnotation]


ImagePair = MedicalAnnotation
ImageGroup = _base.ImageGroup
LabeledImage = _base.LabeledImage
EvidenceEligibility = _base.EvidenceEligibility
BenchmarkAPIError = _base.BenchmarkAPIError
NullProgress = _base.NullProgress
ProgressBar = _base.ProgressBar


def add_common_eval_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--dry-run", action="store_true", help="Build and print task previews without model calls.")
    parser.add_argument("--report", action="store_true", help="Print readable summaries from existing run outputs and exit.")
    parser.add_argument("--category", choices=("medical_modality",), default=None, help=argparse.SUPPRESS)
    parser.add_argument("--dataset", "--subset", dest="subset", choices=("all", "brats2018", "brats2023"), default="all")
    parser.add_argument("--subcategory", default=None, help="Restrict to a question group id.")
    parser.add_argument(
        "--visual-version",
        "--source-variant",
        dest="source_variant",
        choices=("clean", "overlay", "both"),
        default="clean",
        help="Use clean images for the main benchmark or overlay images for ablation.",
    )
    parser.add_argument(
        "--swap-direction",
        "--edit-type",
        dest="edit_type",
        default="all",
        help="Restrict to one swap direction, or all.",
    )
    parser.add_argument("--count-direction", choices=("all", "modality_swap"), default="all", help=argparse.SUPPRESS)
    parser.add_argument("--case-id", default=None, help="Restrict to one BraTS case id.")
    parser.add_argument("--question-types", default="all", help="Comma-separated yes_no,multiple_choice,open, or all.")
    parser.add_argument("--prime", choices=("on", "off"), default="off", help="Medical modality defaults to off.")
    parser.add_argument(
        "--tool-condition",
        choices=TOOL_CONDITIONS,
        default="raw",
        help="Optional label-derived visual tool condition. raw preserves the original benchmark behavior.",
    )
    parser.add_argument(
        "--evidence-manifest",
        type=Path,
        default=None,
        help="Medical label-derived evidence manifest JSON used by non-raw tool conditions.",
    )
    parser.add_argument(
        "--evidence-qc-filter",
        choices=EVIDENCE_QC_FILTERS,
        default="pass",
        help="Which evidence QC statuses are eligible for tool-condition evaluation.",
    )
    parser.add_argument(
        "--missing-evidence-policy",
        choices=MISSING_EVIDENCE_POLICIES,
        default="skip",
        help="How to handle pairs with missing required evidence before API calls.",
    )
    parser.add_argument(
        "--image-group-labels",
        choices=("off", "on"),
        default="off",
        help="Insert neutral image group/view separators between raw and tool images.",
    )
    parser.add_argument("--no-progress", action="store_true", help="Disable progress display.")
    parser.add_argument(
        "--sample-fraction",
        type=float,
        default=None,
        help="Randomly sample this fraction of image pairs before question expansion, e.g. 0.2 for 20%%.",
    )
    parser.add_argument(
        "--sample-strategy",
        choices=("stratified", "random"),
        default="stratified",
        help="Sampling strategy used with --sample-fraction. Stratified preserves dataset/visual-version/swap-direction balance.",
    )
    parser.add_argument("--resume", choices=("off", "latest", "auto"), default="off")
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--retry-base-seconds", type=float, default=2.0)
    parser.add_argument("--retry-max-seconds", type=float, default=60.0)
    parser.add_argument("--stop-on-quota", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max-consecutive-errors", type=int, default=5)
    parser.add_argument(
        "--empty-response-policy",
        choices=_base.EMPTY_RESPONSE_POLICIES,
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
        choices=_base.REASONING_EFFORTS,
        default="off",
        help="Reasoning/thinking control. off requests the minimum non-thinking mode supported by the provider.",
    )
    _base.add_structured_output_arguments(parser)


def load_benchmark(path: Path) -> Dict[str, Any]:
    return _base.load_benchmark(path)


def load_annotations(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        raise SystemExit(
            f"Medical modality metadata not found: {path}\n"
            "Generate it with gen_code/medical_modality/prepare_medical_modality.py after configuring BraTS NIfTI roots."
        )
    records = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(records, list) or not records:
        raise SystemExit("Medical modality metadata must contain a non-empty JSON list.")
    return records


def resolve_medical_data_path(value: str | Path) -> Path:
    """Rebase portable medical paths that may contain stale repo-root prefixes."""
    raw = Path(value).expanduser()
    if raw.is_absolute() and raw.exists():
        return raw.resolve()
    value_text = str(value)
    for marker in (
        "medical/modality_swapping/",
        "medical/BraTS2023_GLI/",
        "vision_dataset/medical_modality/",
    ):
        marker_index = value_text.find(marker)
        if marker_index != -1:
            return (REPO_ROOT / value_text[marker_index:]).resolve(strict=False)
    if raw.is_absolute():
        return raw.resolve(strict=False)
    return (REPO_ROOT / raw).resolve(strict=False)


def build_group_specs(benchmark: Dict[str, Any], *, annotations: Sequence[Dict[str, Any]]) -> List[GroupSpec]:
    question_groups = benchmark["categories"][0]["question_groups"]
    group_meta = {group["question_group_id"]: group for group in question_groups}
    questions_by_group = {
        group_id: [
            QuestionSpec(
                category="medical_modality",
                subcategory=group["subcategory"],
                display_name=group["display_name"],
                question_id=question["question_id"],
                question_type=question["type"],
                prompt=question["prompt"],
                choices=list(question.get("choices", [])),
                answers_by_mode=question["answers"],
            )
            for question in group["questions"]
        ]
        for group_id, group in group_meta.items()
    }

    annotations_by_key: Dict[tuple[str, str], List[MedicalAnnotation]] = defaultdict(list)
    for raw in annotations:
        group_id = str(raw["question_group_id"])
        if group_id not in questions_by_group:
            raise SystemExit(f"Metadata references missing question_group_id: {group_id}")
        original_path = resolve_medical_data_path(raw["original_path"])
        cf_path = resolve_medical_data_path(raw["cf_path"])
        for image_path in (original_path, cf_path):
            if not image_path.exists():
                raise SystemExit(f"Medical modality image path does not exist: {image_path}")
        annotation = MedicalAnnotation(
            category="medical_modality",
            subset=str(raw["dataset"]),
            subcategory=group_id,
            source_variant=str(raw["visual_version"]),
            key=str(raw["case_id"]),
            original_path=original_path,
            edited_path=cf_path,
            question_group_id=group_id,
            edit_type=str(raw["swap_direction"]),
            count_direction="modality_swap",
            dataset=str(raw["dataset"]),
            case_id=str(raw["case_id"]),
            visual_version=str(raw["visual_version"]),
            swap_direction=str(raw["swap_direction"]),
            background_modality=str(raw["background_modality"]),
            tumor_region_modality=str(raw["tumor_region_modality"]),
            biased_modality=str(raw["biased_modality"]),
            display_name=str(group_meta[group_id]["display_name"]),
            annotation_pair_key=str(raw["pair_key"]),
        )
        annotations_by_key[(group_id, annotation.source_variant)].append(annotation)

    groups: List[GroupSpec] = []
    for (group_id, visual_version), bucket in sorted(annotations_by_key.items()):
        first = bucket[0]
        groups.append(
            GroupSpec(
                category="medical_modality",
                subset=first.dataset,
                subcategory=group_id,
                display_name=first.display_name,
                source_variant=visual_version,
                question_group_id=group_id,
                edit_type=first.swap_direction,
                count_direction="modality_swap",
                visual_version=visual_version,
                dataset=first.dataset,
                questions=list(questions_by_group[group_id]),
                annotations=sorted(bucket, key=lambda item: _base.natural_key(item.annotation_pair_key)),
            )
        )
    return sorted_groups(groups)


def sorted_groups(groups: Sequence[GroupSpec]) -> List[GroupSpec]:
    return sorted(groups, key=lambda group: (group.dataset, group.visual_version, group.edit_type))


def resolve_selected_groups(groups: Sequence[GroupSpec], subset_path: Optional[Path]) -> List[GroupSpec]:
    if subset_path is None:
        return sorted_groups(groups)
    target = subset_path.resolve()
    selected = [
        group
        for group in groups
        if any(target in {pair.original_path, pair.edited_path, pair.original_path.parent, pair.edited_path.parent} for pair in group.annotations)
    ]
    if not selected:
        raise SystemExit(f"Subset path did not match any medical modality image or folder: {target}")
    return sorted_groups(selected)


def filter_groups(
    groups: Sequence[GroupSpec],
    *,
    category: Optional[str],
    subcategory: Optional[str],
    source_variant: str,
    question_types: Sequence[str],
    subset: str = "all",
    edit_type: str = "all",
    count_direction: str = "all",
) -> List[GroupSpec]:
    allowed_visuals = set(SOURCE_VARIANTS if source_variant == "both" else (source_variant,))
    case_id = case_id_from_argv()
    selected: List[GroupSpec] = []
    for group in groups:
        if category and group.category != category:
            continue
        if subset != "all" and group.dataset != subset:
            continue
        if subcategory and group.subcategory != subcategory:
            continue
        if edit_type != "all" and group.edit_type != edit_type:
            continue
        if count_direction != "all" and group.count_direction != count_direction:
            continue
        if group.source_variant not in allowed_visuals:
            continue
        questions = [question for question in group.questions if question.question_type in set(question_types)]
        annotations = [pair for pair in group.annotations if case_id is None or pair.case_id == case_id]
        if questions and annotations:
            selected.append(replace(group, questions=questions, annotations=annotations))
    if not selected:
        raise SystemExit("No medical modality groups matched the selected filters.")
    return selected


def case_id_from_argv() -> Optional[str]:
    for index, token in enumerate(sys.argv):
        if token == "--case-id" and index + 1 < len(sys.argv):
            return sys.argv[index + 1]
        if token.startswith("--case-id="):
            return token.split("=", 1)[1]
    return None


def build_image_pairs(groups: Sequence[GroupSpec]) -> List[MedicalAnnotation]:
    pairs: List[MedicalAnnotation] = []
    for group in groups:
        pairs.extend(group.annotations)
    return sorted(pairs, key=lambda pair: _base.natural_key(pair.annotation_pair_key))


def apply_pair_sampling(
    image_pairs: Sequence[ImagePair],
    *,
    sample_fraction: Optional[float],
    sample_strategy: str,
    seed: int,
    max_samples: Optional[int],
) -> Tuple[List[ImagePair], Dict[str, Any]]:
    if max_samples is not None and sample_fraction is not None:
        raise SystemExit("--max-samples and --sample-fraction are mutually exclusive.")
    if max_samples is not None:
        selected = list(image_pairs[:max_samples])
        return selected, sampling_metadata(
            original_count=len(image_pairs),
            selected=selected,
            sample_fraction=None,
            sample_strategy="prefix_max_samples",
            seed=seed,
            max_samples=max_samples,
        )
    if sample_fraction is None:
        selected = list(image_pairs)
        return selected, sampling_metadata(
            original_count=len(image_pairs),
            selected=selected,
            sample_fraction=None,
            sample_strategy="all",
            seed=seed,
            max_samples=None,
        )
    if not 0 < sample_fraction <= 1:
        raise SystemExit("--sample-fraction must be in the interval (0, 1].")
    if sample_fraction >= 1:
        selected = list(image_pairs)
        return selected, sampling_metadata(
            original_count=len(image_pairs),
            selected=selected,
            sample_fraction=sample_fraction,
            sample_strategy=sample_strategy,
            seed=seed,
            max_samples=None,
        )

    rng = random.Random(seed)
    original_order = {pair_key(pair): index for index, pair in enumerate(image_pairs)}
    if sample_strategy == "random":
        sample_count = max(1, round(len(image_pairs) * sample_fraction))
        selected = rng.sample(list(image_pairs), sample_count)
    else:
        buckets: Dict[Tuple[str, ...], List[ImagePair]] = defaultdict(list)
        for pair in image_pairs:
            buckets[sampling_stratum(pair)].append(pair)
        selected = []
        for key in sorted(buckets):
            bucket = sorted(buckets[key], key=lambda pair: _base.natural_key(pair_key(pair)))
            sample_count = max(1, round(len(bucket) * sample_fraction))
            selected.extend(rng.sample(bucket, sample_count))
    selected = sorted(selected, key=lambda pair: original_order[pair_key(pair)])
    return selected, sampling_metadata(
        original_count=len(image_pairs),
        selected=selected,
        sample_fraction=sample_fraction,
        sample_strategy=sample_strategy,
        seed=seed,
        max_samples=None,
    )


def sampling_stratum(pair: ImagePair) -> Tuple[str, ...]:
    return (
        str(getattr(pair, "dataset", getattr(pair, "subset", ""))),
        str(getattr(pair, "visual_version", getattr(pair, "source_variant", ""))),
        str(getattr(pair, "swap_direction", getattr(pair, "edit_type", ""))),
        str(getattr(pair, "count_direction", "")),
    )


def pair_key(pair: ImagePair) -> str:
    return str(
        getattr(
            pair,
            "annotation_pair_key",
            f"{getattr(pair, 'original_path', '')}|{getattr(pair, 'edited_path', '')}",
        )
    )


def sampling_metadata(
    *,
    original_count: int,
    selected: Sequence[ImagePair],
    sample_fraction: Optional[float],
    sample_strategy: str,
    seed: int,
    max_samples: Optional[int],
) -> Dict[str, Any]:
    keys = [pair_key(pair) for pair in selected]
    digest = hashlib.sha256("\n".join(keys).encode("utf-8")).hexdigest() if keys else None
    return {
        "original_pair_count": original_count,
        "selected_pair_count": len(selected),
        "max_samples": max_samples,
        "sample_fraction": sample_fraction,
        "sample_strategy": sample_strategy,
        "sample_seed": seed,
        "selected_pair_key_digest": digest,
    }


def build_questions_by_key(groups: Sequence[GroupSpec]) -> Dict[str, List[QuestionSpec]]:
    return {group.question_group_id: list(group.questions) for group in groups}


def count_total_evaluations(image_pairs: Sequence[MedicalAnnotation], questions_by_key: Dict[str, List[QuestionSpec]]) -> int:
    return sum(len(questions_by_key[pair.question_group_id]) for pair in image_pairs)


def build_eval_tasks(image_pairs: Sequence[MedicalAnnotation], questions_by_key: Dict[str, List[QuestionSpec]]) -> List[Tuple[MedicalAnnotation, QuestionSpec]]:
    tasks: List[Tuple[MedicalAnnotation, QuestionSpec]] = []
    for pair in image_pairs:
        for question in questions_by_key[pair.question_group_id]:
            tasks.append((pair, question))
    return tasks


def image_paths_for_mode(pair: MedicalAnnotation, eval_mode: str) -> List[Path]:
    return _base.image_paths_for_mode(pair, eval_mode)


def load_evidence_manifest(path: Optional[Path]) -> Dict[str, Dict[str, Any]]:
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

    manifest: Dict[str, Dict[str, Any]] = {}
    problems: List[str] = []
    for index, record in enumerate(records, start=1):
        if not isinstance(record, dict):
            problems.append(f"record {index}: expected object")
            continue
        if record.get("schema_version") != EVIDENCE_SCHEMA_VERSION:
            problems.append(f"record {index}: unsupported schema_version={record.get('schema_version')!r}")
            continue
        pair_key_value = str(record.get("pair_key") or "")
        if not pair_key_value:
            problems.append(f"record {index}: missing pair_key")
            continue
        if pair_key_value in manifest:
            problems.append(f"record {index}: duplicate pair_key {pair_key_value}")
            continue
        for role in ("original", "cf"):
            role_payload = record.get(role)
            if not isinstance(role_payload, dict):
                problems.append(f"record {index}: missing {role} evidence block")
                continue
            for field in ("raw", "bbox_overlay", "contour_overlay", "crop", "zoom_panel"):
                value = role_payload.get(field)
                if not value:
                    continue
                resolved = resolve_manifest_path(value, base_dir=manifest_path.parent)
                if not resolved.exists():
                    problems.append(f"record {index}: missing referenced file {resolved}")
        manifest[pair_key_value] = record
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
        "vision_dataset/medical_modality/",
        "eval_results/medical_modality/",
        "medical/modality_swapping/",
        "medical/BraTS2023_GLI/",
    ):
        marker_index = value_text.find(marker)
        if marker_index != -1:
            return (REPO_ROOT / value_text[marker_index:]).resolve(strict=False)
    if raw.is_absolute():
        return raw.resolve(strict=False)
    root = base_dir or REPO_ROOT
    return (root / raw).resolve(strict=False)


def evidence_manifest_key(pair: MedicalAnnotation) -> str:
    return pair.annotation_pair_key


def validate_tool_condition_args(
    *,
    tool_condition: str,
    evidence_manifest_path: Optional[Path],
    evidence_qc_filter: str,
    missing_evidence_policy: str,
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
        "contour": ("contour_overlay",),
        "crop": ("crop",),
        "zoom_panel": ("zoom_panel",),
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


def check_pair_tool_eligibility(
    *,
    pair: MedicalAnnotation,
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
    question_group_id = str(manifest_record.get("question_group_id") or "")
    if question_group_id and question_group_id != pair.question_group_id:
        return EvidenceEligibility(False, "question_group_mismatch")
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
    image_pairs: Sequence[MedicalAnnotation],
    *,
    eval_mode: str,
    tool_condition: str,
    evidence_manifest: Dict[str, Dict[str, Any]],
    evidence_manifest_path: Optional[Path],
    evidence_qc_filter: str,
    missing_evidence_policy: str,
) -> Tuple[List[MedicalAnnotation], Dict[str, Any]]:
    manifest = evidence_manifest or {}
    eligible_pairs: List[MedicalAnnotation] = []
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
        "tool_condition": tool_condition,
        "evidence_qc_filter": evidence_qc_filter,
        "missing_evidence_policy": missing_evidence_policy,
    }
    if missing_evidence_policy == "error" and stats["skipped_pair_count_due_to_evidence"] > 0:
        raise SystemExit("Evidence eligibility failed: " + json.dumps(stats, sort_keys=True))
    return eligible_pairs, stats


def print_evidence_filter_summary(stats: Dict[str, Any], *, questions_per_pair: Optional[int] = None) -> None:
    _base.print_evidence_filter_summary(stats, questions_per_pair=questions_per_pair)


def build_image_groups(
    *,
    pair: MedicalAnnotation,
    eval_mode: str,
    tool_condition: str = "raw",
    evidence_manifest: Optional[Dict[str, Dict[str, Any]]] = None,
    evidence_manifest_path: Optional[Path] = None,
    missing_evidence_policy: str = "error",
) -> Tuple[List[ImageGroup], Optional[Dict[str, Any]]]:
    if tool_condition == "raw":
        manifest_record = evidence_manifest.get(evidence_manifest_key(pair)) if evidence_manifest else None
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
        raise SystemExit(f"Missing evidence manifest record for pair_key={pair.annotation_pair_key}.")
    base_dir = evidence_manifest_path.expanduser().resolve().parent if evidence_manifest_path else None

    original_payload = manifest_record["original"]
    cf_payload = manifest_record["cf"]
    original_group = ImageGroup(
        role="original",
        raw=pair.original_path,
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
        raw=pair.edited_path,
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
    return _base.flatten_image_groups(groups)


def flatten_labeled_image_groups(groups: Sequence[ImageGroup], *, image_group_labels: str = "off") -> List[LabeledImage]:
    return _base.flatten_labeled_image_groups(groups, image_group_labels=image_group_labels)


def serialize_image_groups(groups: Sequence[ImageGroup]) -> List[Dict[str, Any]]:
    return _base.serialize_image_groups(groups)


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
        "evidence_generator": generator.get("source"),
        "evidence_generator_source": generator.get("source"),
        "evidence_mask_rule": generator.get("mask_rule"),
        "crop_padding": generator.get("crop_padding"),
        "evidence_qc_status": evidence_record.get("qc_status") if isinstance(evidence_record, dict) else None,
        "evidence_qc_filter": evidence_qc_filter,
        "missing_evidence_policy": missing_evidence_policy,
        "evidence_usable": evidence_usable,
        "evidence_raw_fallback": tool_condition != "raw" and not evidence_usable and missing_evidence_policy == "raw_fallback",
    }


def build_model_prompt(
    *,
    question: QuestionSpec,
    prime: str = "off",
    eval_mode: Optional[str] = None,
    tool_condition: str = "raw",
) -> str:
    prompt = _base.build_model_prompt(question=question, prime=prime, eval_mode=eval_mode, tool_condition=tool_condition)
    if tool_condition == "raw":
        return prompt
    if eval_mode == "both":
        return prompt.replace(TOOL_VIEW_BOTH_PREFIX, MEDICAL_TOOL_VIEW_BOTH_PREFIX)
    return prompt.replace(TOOL_VIEW_PREFIX, MEDICAL_TOOL_VIEW_PREFIX)


def build_judge_prompt(
    *,
    question_prompt: str,
    eval_mode: str,
    candidate_answer: str,
    answers: Dict[str, Any],
    tool_condition: str = "raw",
) -> str:
    target_rule = {
        "cf_only": "Score the answer as a judgment about the single provided medical image.",
        "orig_only": "Score the answer as a judgment about the single provided original medical image.",
        "both": (
            "Score the answer as a neutral comparison of the two related medical images. The answer should identify "
            "whether tumor-region/background modality consistency differs across the pair; do not require image order."
            if tool_condition == "raw"
            else "Score the answer as a judgment about the second image group; do not require the answer to literally say second image group unless it only discusses the first group."
        ),
    }.get(eval_mode, "Score the answer against the selected input mode.")
    policy = (
        "You are scoring an answer for a medical MRI modality-consistency benchmark, not making a clinical diagnosis.\n"
        "Return strict JSON only with keys label and reason. Valid labels are correct, biased, or other.\n"
        "Use only the candidate answer text and the rubric below. Do not infer image evidence that the candidate answer does not mention.\n"
        f"Target rule: {target_rule}\n"
        "Decision procedure:\n"
        "1. Label correct only if the answer reaches the correct modality-consistency conclusion and cites visible MRI signal evidence.\n"
        "2. Label biased if the answer follows the surrounding/background sequence prior and treats the tumor region as normal or background-like despite the target rubric.\n"
        "3. Label other if the answer is vague, contradictory, refuses, gives only a clinical diagnosis, or lacks a visible MRI signal cue.\n"
        "4. Do not reward statements about tumor type, malignancy, or diagnosis unless they also answer modality consistency with visible signal evidence.\n"
    )
    payload = {
        "eval_mode": eval_mode,
        "tool_condition": tool_condition,
        **_base.both_raw_policy_fields(eval_mode, tool_condition),
        "question_prompt": question_prompt,
        "candidate_answer": candidate_answer,
        "correct_rubric": answers["correct_rubric"],
        "bias_rubric": answers["bias_rubric"],
        "other_rubric": answers["other_rubric"],
        "required_output_format": {"label": "correct|biased|other", "reason": "short explanation"},
    }
    return policy + "\nBenchmark item:\n" + json.dumps(payload, ensure_ascii=False, indent=2)


def task_signature(
    *,
    pair: MedicalAnnotation,
    question: QuestionSpec,
    eval_mode: str,
    tool_condition: str = "raw",
) -> str:
    parts = [
        eval_mode,
        pair.category,
        pair.dataset,
        pair.visual_version,
        pair.swap_direction,
        pair.case_id,
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
    values = [
        eval_mode,
        str(record.get("category") or "medical_modality"),
        str(record.get("dataset") or record.get("subset") or ""),
        str(record.get("visual_version") or record.get("source_variant") or ""),
        str(record.get("swap_direction") or record.get("edit_type") or ""),
        str(record.get("case_id") or record.get("key") or ""),
        str(record.get("question_id") or ""),
        Path(str(record.get("original_image_path") or "")).name,
        Path(str(record.get("edited_image_path") or "")).name,
    ]
    if not all(values):
        return None
    if tool_condition == "raw":
        return "|".join(values)
    return "|".join([eval_mode, tool_condition, *values[1:]])


def successful_task_signatures(records: Sequence[Dict[str, Any]]) -> set[str]:
    return {signature for record in records if (signature := record_task_signature(record)) and is_success_record(record)}


def completed_success_count(records: Sequence[Dict[str, Any]]) -> int:
    return len(successful_task_signatures(records))


def is_success_record(record: Dict[str, Any]) -> bool:
    return _base.is_success_record(record)


def record_is_error(record: Dict[str, Any]) -> bool:
    return _base.record_is_error(record)


def fashion_compatible_record_fields(
    *,
    pair: MedicalAnnotation,
    question: QuestionSpec,
    eval_mode: str,
    prompt: str,
    used_image_paths: Sequence[Path],
    answers: Dict[str, Any],
    backbone_model_alias: str,
    judge_model: str,
    tool_condition: str = "raw",
    evidence_manifest_path: Optional[Path] = None,
    image_groups: Optional[Sequence[ImageGroup]] = None,
    evidence_record: Optional[Dict[str, Any]] = None,
    evidence_qc_filter: str = "",
    missing_evidence_policy: str = "",
) -> Dict[str, Any]:
    payload = _base.fashion_compatible_record_fields(
        pair=pair,
        question=question,
        eval_mode=eval_mode,
        prompt=prompt,
        used_image_paths=used_image_paths,
        answers=answers,
        backbone_model_alias=backbone_model_alias,
        judge_model=judge_model,
    )
    payload.update(
        {
            "dataset": pair.dataset,
            "case_id": pair.case_id,
            "visual_version": pair.visual_version,
            "swap_direction": pair.swap_direction,
            "background_modality": pair.background_modality,
            "tumor_region_modality": pair.tumor_region_modality,
            "biased_modality": pair.biased_modality,
            "question_group_id": pair.question_group_id,
            "annotation_pair_key": pair.annotation_pair_key,
            "cf_image_path": str(pair.edited_path),
        }
    )
    if image_groups is not None:
        payload.update(
            build_tool_record_fields(
                tool_condition=tool_condition,
                evidence_manifest_path=evidence_manifest_path,
                image_groups=image_groups,
                evidence_record=evidence_record,
                evidence_qc_filter=evidence_qc_filter,
                missing_evidence_policy=missing_evidence_policy,
            )
        )
    return payload


def build_summary(*, records: Sequence[Dict[str, Any]], run_id: str, script_type: str, eval_mode: str, backbone_ref: Any, judge_ref: Any, run_status: str = "completed", is_partial: bool = False) -> Dict[str, Any]:
    summary = _base.build_summary(
        records=records,
        run_id=run_id,
        script_type=script_type,
        eval_mode=eval_mode,
        backbone_ref=backbone_ref,
        judge_ref=judge_ref,
        run_status=run_status,
        is_partial=is_partial,
    )
    summary["by_dataset"] = summarize_by(records, "dataset")
    summary["by_visual_version"] = summarize_by(records, "visual_version")
    summary["by_swap_direction"] = summarize_by(records, "swap_direction")
    summary["by_tool_condition"] = summarize_by(records, "tool_condition")
    return summary


def summarize_records(records: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    summary = _base.summarize_records(records)
    summary["script_type"] = summary.get("script_type", "medical_modality")
    summary["by_dataset"] = summarize_by(records, "dataset")
    summary["by_visual_version"] = summarize_by(records, "visual_version")
    summary["by_swap_direction"] = summarize_by(records, "swap_direction")
    summary["by_tool_condition"] = summarize_by(records, "tool_condition")
    return summary


def summarize_by(records: Sequence[Dict[str, Any]], field: str) -> Dict[str, Dict[str, Any]]:
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for record in records:
        grouped.setdefault(str(record_field_value(record, field)), []).append(record)
    return {key: _base.summarize_bucket(bucket) for key, bucket in sorted(grouped.items())}


def record_field_value(record: Dict[str, Any], field: str) -> Any:
    if field == "tool_condition":
        return record.get("tool_condition", "raw")
    aliases = {
        "dataset": ("dataset", "subset"),
        "visual_version": ("visual_version", "source_variant"),
        "swap_direction": ("swap_direction", "edit_type"),
        "cf_image_path": ("cf_image_path", "edited_image_path"),
        "evidence_qc_filter": ("evidence_qc_filter",),
        "missing_evidence_policy": ("missing_evidence_policy",),
    }
    if field in aliases:
        for key in aliases[field]:
            if record.get(key) not in (None, ""):
                return record.get(key)
        return ""
    return _base.record_field_value(record, field)


def field_to_column(field: str) -> str:
    return {
        "dataset": "Dataset",
        "visual_version": "Visual_Version",
        "swap_direction": "Swap_Direction",
        "background_modality": "Background_Modality",
        "tumor_region_modality": "Tumor_Region_Modality",
        "case_id": "Case_ID",
        "cf_image_path": "CF_Image_Path",
        "tool_condition": "Tool_Condition",
        "evidence_qc_filter": "Evidence_QC_Filter",
        "missing_evidence_policy": "Missing_Evidence_Policy",
    }.get(field, _base.field_to_column(field))


def aggregate_rows(records: Sequence[Dict[str, Any]], fields: Sequence[str]) -> List[Dict[str, Any]]:
    grouped: Dict[tuple[str, ...], List[Dict[str, Any]]] = {}
    for record in records:
        key = tuple(str(record_field_value(record, field)) for field in fields)
        grouped.setdefault(key, []).append(record)
    rows: List[Dict[str, Any]] = []
    for key, bucket in sorted(grouped.items()):
        row = {field_to_column(field): value for field, value in zip(fields, key)}
        row.update(_base.bucket_to_row(_base.summarize_bucket(bucket)))
        rows.append(row)
    return rows


def filter_records_to_current_dataset(
    records: Sequence[Dict[str, Any]],
    *,
    questions_path: Path = DEFAULT_QUESTIONS_PATH,
    include_stale_records: bool = False,
) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    if include_stale_records:
        return list(records), {"input_count": len(records), "kept_count": len(records), "dropped_stale_count": 0, "unchecked_count": 0, "current_pair_count": 0}
    current = current_dataset_pair_keys()
    kept: List[Dict[str, Any]] = []
    dropped = 0
    unchecked = 0
    for record in records:
        key = (str(record.get("annotation_pair_key") or ""), str(record.get("question_group_id") or ""), str(record.get("question_id") or ""))
        if not all(key):
            kept.append(record)
            unchecked += 1
            continue
        if key in current:
            kept.append(record)
        else:
            dropped += 1
    return kept, {"input_count": len(records), "kept_count": len(kept), "dropped_stale_count": dropped, "unchecked_count": unchecked, "current_pair_count": len(current)}


def current_dataset_pair_keys() -> set[tuple[str, str, str]]:
    benchmark = load_benchmark(DEFAULT_QUESTIONS_PATH)
    annotations = load_annotations(DEFAULT_ANNOTATIONS_PATH)
    groups = build_group_specs(benchmark, annotations=annotations)
    return {
        (pair.annotation_pair_key, pair.question_group_id, question.question_id)
        for pair in build_image_pairs(groups)
        for question in build_questions_by_key(groups)[pair.question_group_id]
    }


def print_dry_run_preview(
    *,
    script_type: str,
    eval_mode: str,
    backbone_alias: str,
    judge_alias: str,
    image_pairs: Sequence[MedicalAnnotation],
    questions_by_key: Dict[str, List[QuestionSpec]],
    total_evaluations: int,
    prime: str = "off",
    judge_structured_output: str = "auto",
    closed_form_structured_output: str = "auto",
    structured_output_fallback: str = "retry_plain",
    tool_condition: str = "raw",
    evidence_manifest: Optional[Dict[str, Dict[str, Any]]] = None,
    evidence_manifest_path: Optional[Path] = None,
    evidence_qc_filter: str = "pass",
    missing_evidence_policy: str = "skip",
    image_group_labels: str = "off",
    max_preview: int = 20,
) -> None:
    effective_image_group_labels = "off" if eval_mode == "both" and tool_condition == "raw" else image_group_labels
    print("=" * 80)
    print(f"dry_run script_type={script_type} eval_mode={eval_mode} backbone={backbone_alias} judge={judge_alias}")
    print(
        f"image_pairs={len(image_pairs)} response_target_count={total_evaluations} "
        f"tool_condition={tool_condition} evidence_qc_filter={evidence_qc_filter} "
        f"missing_evidence_policy={missing_evidence_policy}"
    )
    print(
        "structured_output "
        f"judge={judge_structured_output} closed_form={closed_form_structured_output} "
        f"fallback={structured_output_fallback}"
    )
    print("=" * 80)
    shown = 0
    for pair in image_pairs:
        for question in questions_by_key[pair.question_group_id]:
            if shown >= max_preview:
                print(f"... {total_evaluations - shown} additional tasks not shown")
                return
            image_groups, evidence_record = build_image_groups(
                pair=pair,
                eval_mode=eval_mode,
                tool_condition=tool_condition,
                evidence_manifest=evidence_manifest,
                evidence_manifest_path=evidence_manifest_path,
                missing_evidence_policy=missing_evidence_policy,
            )
            used_image_paths = flatten_image_groups(image_groups)
            record = preview_record(shown + 1, pair, question, eval_mode, prime, tool_condition=tool_condition)
            record.update(
                {
                    "judge_structured_output": judge_structured_output,
                    "closed_form_structured_output": closed_form_structured_output,
                    "structured_output_fallback": structured_output_fallback,
                    "tool_condition": tool_condition,
                    "evidence_qc_status": evidence_record.get("qc_status") if isinstance(evidence_record, dict) else None,
                    "evidence_qc_filter": evidence_qc_filter,
                    "missing_evidence_policy": missing_evidence_policy,
                    "used_image_paths": [str(path) for path in used_image_paths],
                    "used_image_groups": serialize_image_groups(image_groups),
                    "labeled_image_sequence": [
                        {"path": str(item.path), "label": item.label}
                        for item in flatten_labeled_image_groups(image_groups, image_group_labels=effective_image_group_labels)
                    ],
                }
            )
            print(json.dumps(record, ensure_ascii=False))
            shown += 1


def print_dry_run_task_preview(
    *,
    script_type: str,
    eval_mode: str,
    backbone_alias: str,
    judge_alias: str,
    tasks: Sequence[Tuple[MedicalAnnotation, QuestionSpec]],
    total_evaluations: int,
    prime: str = "off",
    judge_structured_output: str = "auto",
    closed_form_structured_output: str = "auto",
    structured_output_fallback: str = "retry_plain",
    tool_condition: str = "raw",
    evidence_manifest: Optional[Dict[str, Dict[str, Any]]] = None,
    evidence_manifest_path: Optional[Path] = None,
    evidence_qc_filter: str = "pass",
    missing_evidence_policy: str = "skip",
    image_group_labels: str = "off",
    max_preview: int = 20,
) -> None:
    effective_image_group_labels = "off" if eval_mode == "both" and tool_condition == "raw" else image_group_labels
    print("=" * 80)
    print(f"dry_run script_type={script_type} eval_mode={eval_mode} backbone={backbone_alias} judge={judge_alias}")
    print(
        f"pending_tasks={len(tasks)} response_target_count={total_evaluations} "
        f"tool_condition={tool_condition} evidence_qc_filter={evidence_qc_filter} "
        f"missing_evidence_policy={missing_evidence_policy}"
    )
    print(
        "structured_output "
        f"judge={judge_structured_output} closed_form={closed_form_structured_output} "
        f"fallback={structured_output_fallback}"
    )
    print("=" * 80)
    for index, (pair, question) in enumerate(tasks[:max_preview], start=1):
        image_groups, evidence_record = build_image_groups(
            pair=pair,
            eval_mode=eval_mode,
            tool_condition=tool_condition,
            evidence_manifest=evidence_manifest,
            evidence_manifest_path=evidence_manifest_path,
            missing_evidence_policy=missing_evidence_policy,
        )
        used_image_paths = flatten_image_groups(image_groups)
        record = preview_record(index, pair, question, eval_mode, prime, tool_condition=tool_condition)
        record.update(
            {
                "judge_structured_output": judge_structured_output,
                "closed_form_structured_output": closed_form_structured_output,
                "structured_output_fallback": structured_output_fallback,
                "tool_condition": tool_condition,
                "evidence_qc_status": evidence_record.get("qc_status") if isinstance(evidence_record, dict) else None,
                "evidence_qc_filter": evidence_qc_filter,
                "missing_evidence_policy": missing_evidence_policy,
                "used_image_paths": [str(path) for path in used_image_paths],
                "used_image_groups": serialize_image_groups(image_groups),
                "labeled_image_sequence": [
                    {"path": str(item.path), "label": item.label}
                    for item in flatten_labeled_image_groups(image_groups, image_group_labels=effective_image_group_labels)
                ],
            }
        )
        print(json.dumps(record, ensure_ascii=False))
    if len(tasks) > max_preview:
        print(f"... {len(tasks) - max_preview} additional pending tasks not shown")


def preview_record(
    index: int,
    pair: MedicalAnnotation,
    question: QuestionSpec,
    eval_mode: str,
    prime: str,
    *,
    tool_condition: str = "raw",
) -> Dict[str, Any]:
    return {
        "idx": index,
        "dataset": pair.dataset,
        "visual_version": pair.visual_version,
        "case_id": pair.case_id,
        "swap_direction": pair.swap_direction,
        "background_modality": pair.background_modality,
        "tumor_region_modality": pair.tumor_region_modality,
        "question_group_id": pair.question_group_id,
        "question_id": question.question_id,
        "question_type": question.question_type,
        "prompt": build_model_prompt(question=question, prime=prime, eval_mode=eval_mode, tool_condition=tool_condition),
        "original_image_path": str(pair.original_path),
        "cf_image_path": str(pair.edited_path),
    }


parse_question_types = _base.parse_question_types
seed_everything = _base.seed_everything
make_run_id = _base.make_run_id
def answers_for_mode(question: QuestionSpec, eval_mode: str, tool_condition: str = "raw") -> Dict[str, Any]:
    if _base.is_neutral_pair_difference_mode(eval_mode, tool_condition) and question.question_type == "open":
        return {
            "correct_rubric": {
                "target_claim": "The two medical images differ in tumor-region/background modality consistency.",
                "required_visual_evidence": [
                    "The answer explicitly compares modality consistency or tumor-region contrast across the two images.",
                    "The answer cites a visible MRI signal cue such as local brightness, darkness, or contrast discontinuity.",
                ],
                "reject_if": [
                    "The answer only describes one image without comparing modality consistency across the pair.",
                    "The answer claims both images are consistently explained by the surrounding/global MRI appearance.",
                    "The answer gives only a clinical diagnosis without modality-consistency evidence.",
                ],
            },
            "bias_rubric": {
                "target_claim": "Both images are treated as following the surrounding/global MRI modality prior with no relevant tumor-region inconsistency difference.",
                "prior_basis": "The biased answer follows the global MRI appearance prior rather than comparing tumor-region contrast across the pair.",
                "qualify_if": [
                    "The answer says both tumor regions match the surrounding sequence.",
                    "The answer gives only the background/global modality as the answer.",
                    "The answer ignores the visible tumor-region contrast difference across the pair.",
                ],
                "reject_if": "Reject if the answer identifies a tumor-region/background consistency difference with visible signal evidence.",
            },
            "other_rubric": {
                "label_when": "The answer is vague, contradictory, refuses, gives only a clinical diagnosis, describes unrelated differences, or lacks a visible MRI signal cue."
            },
        }
    return _base.answers_for_mode(question, eval_mode, tool_condition)

both_raw_policy_fields = _base.both_raw_policy_fields
is_neutral_pair_difference_mode = _base.is_neutral_pair_difference_mode
scoring_targets = _base.scoring_targets
parse_response = _base.parse_response
parse_yes_no = _base.parse_yes_no
parse_multiple_choice = _base.parse_multiple_choice
score_closed_form = _base.score_closed_form
build_yes_no_judge_prompt = _base.build_yes_no_judge_prompt
parse_judge_response = _base.parse_judge_response
discover_run_dirs = _base.discover_run_dirs
load_json = _base.load_json
load_jsonl = _base.load_jsonl
load_records_from_run_dirs = _base.load_records_from_run_dirs
resolve_resume_run_dir = _base.resolve_resume_run_dir
write_checkpoint = _base.write_checkpoint
load_checkpoint = _base.load_checkpoint
classify_error = _base.classify_error
is_fatal_error_kind = _base.is_fatal_error_kind
is_retryable_error_kind = _base.is_retryable_error_kind
retry_after_seconds = _base.retry_after_seconds
retry_delay_seconds = _base.retry_delay_seconds
call_with_retries = _base.call_with_retries
error_metadata = _base.error_metadata
print_existing_run_summaries = _base.print_existing_run_summaries
print_readable_summary = _base.print_readable_summary
print_bucket_table = _base.print_bucket_table
print_bucket_line = _base.print_bucket_line
format_percent = _base.format_percent
summarize_bucket = _base.summarize_bucket
summarize_provider_cache = _base.summarize_provider_cache
target_texts_for_record = _base.target_texts_for_record
bucket_to_row = _base.bucket_to_row
write_csv = _base.write_csv
write_json = _base.write_json
append_jsonl = _base.append_jsonl
elapsed_ms = _base.elapsed_ms
extract_markdown_paths = _base.extract_markdown_paths
format_duration = _base.format_duration
