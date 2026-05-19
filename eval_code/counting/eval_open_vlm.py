#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from dotenv import load_dotenv
from PIL import Image

EVAL_CODE_DIR = Path(__file__).resolve().parents[1]
if str(EVAL_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(EVAL_CODE_DIR))

from structured_outputs import structured_record_fields, strip_thinking_blocks

from eval_common import (
    DEFAULT_ANNOTATIONS_PATH,
    DEFAULT_QUESTIONS_PATH,
    DEFAULT_RAW_RUNS_ROOT,
    ImagePair,
    NullProgress,
    ProgressBar,
    QuestionSpec,
    PRIOR_PRIME_PREFIX,
    add_common_eval_arguments,
    apply_pair_sampling,
    append_jsonl,
    answers_for_mode,
    build_eval_tasks,
    build_image_groups,
    build_group_specs,
    build_image_pairs,
    build_judge_prompt,
    build_model_prompt,
    build_questions_by_key,
    build_summary,
    build_tool_record_fields,
    build_yes_no_judge_prompt,
    both_raw_policy_fields,
    completed_success_count,
    count_total_evaluations,
    elapsed_ms,
    error_metadata,
    fashion_compatible_record_fields,
    filter_image_pairs_for_evidence,
    filter_groups,
    flatten_image_groups,
    flatten_labeled_image_groups,
    image_paths_for_mode,
    LabeledImage,
    load_benchmark,
    load_annotations,
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
    import torch
except Exception:  # pragma: no cover - import availability depends on env
    torch = None

try:
    from accelerate import init_empty_weights, infer_auto_device_map
except Exception:  # pragma: no cover - import availability depends on env
    init_empty_weights = None
    infer_auto_device_map = None

try:
    from transformers import AutoConfig, AutoProcessor, BitsAndBytesConfig
except Exception:  # pragma: no cover - import availability depends on env
    AutoConfig = None
    AutoProcessor = None
    BitsAndBytesConfig = None

try:
    from transformers import AutoModelForImageTextToText
except Exception:  # pragma: no cover - import availability depends on env
    AutoModelForImageTextToText = None

try:
    from transformers import AutoModelForMultimodalLM
except Exception:  # pragma: no cover - import availability depends on env
    AutoModelForMultimodalLM = None

try:
    from transformers import Gemma4ForConditionalGeneration
except Exception:  # pragma: no cover - import availability depends on env
    Gemma4ForConditionalGeneration = None


OPEN_MODEL_ALIASES = {
    "qwen3-vl-8b": "Qwen/Qwen3-VL-8B-Instruct",
    "qwen3-vl-32b": "Qwen/Qwen3-VL-32B-Instruct",
    "qwen2.5-vl-32b": "Qwen/Qwen2.5-VL-32B-Instruct",
    "qwen2.5-vl-72b": "Qwen/Qwen2.5-VL-72B-Instruct",
    "internvl3.5-8b": "OpenGVLab/InternVL3_5-8B",
    "internvl3_5-8b": "OpenGVLab/InternVL3_5-8B",
    "internvl3.5-38b-hf": "OpenGVLab/InternVL3_5-38B-HF",
    "internvl3_5-38b-hf": "OpenGVLab/InternVL3_5-38B-HF",
    "gemma-4-e4b-it": "google/gemma-4-E4B-it",
    "gemma-4-31b": "google/gemma-4-31B",
    "gemma-4-31b-it": "google/gemma-4-31B-it",
}

MODEL_BYTES_SAFETY_FACTOR = 1.0
USABLE_GPU_MEMORY_RATIO = 0.90
FOUR_BIT_MEMORY_FACTOR = 0.35


@dataclass(frozen=True)
class HFModelRef:
    alias: str
    model_id: str

    @property
    def canonical(self) -> str:
        return self.model_id


@dataclass
class DevicePlan:
    mode: str
    assigned_devices: List[int]
    reserved_bytes_by_device: Dict[int, int]
    device_map: Any
    max_memory: Optional[Dict[Any, int]]
    estimated_total_bytes: int
    model_family: str
    dtype_name: str
    load_in_4bit: bool


@dataclass
class LoadedHFModel:
    ref: HFModelRef
    processor: Any
    model: Any
    device_plan: DevicePlan
    model_family: str
    dtype_name: str
    load_in_4bit: bool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate open VLMs on the counting counterfactual benchmark.")
    parser.add_argument(
        "--questions",
        type=Path,
        default=DEFAULT_QUESTIONS_PATH,
        help="Path to the benchmark question JSON.",
    )
    parser.add_argument(
        "--annotations",
        type=Path,
        default=DEFAULT_ANNOTATIONS_PATH,
        help="Path to counting count annotations JSON.",
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
    parser.add_argument("--backbone-model", required=True, help="Open-model alias or Hugging Face model id.")
    parser.add_argument(
        "--judge-model",
        required=True,
        help="Open-model alias or Hugging Face model id used to judge yes_no and open responses.",
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
        help="Weight dtype used for local model loading when 4-bit is disabled.",
    )
    parser.add_argument(
        "--load-in-4bit",
        action="store_true",
        help="Enable explicit 4-bit loading. This is never auto-enabled.",
    )
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature.")
    parser.add_argument("--max-output-tokens", type=int, default=1024, help="Max generated tokens per backbone response.")
    parser.add_argument("--judge-max-output-tokens", type=int, default=512, help="Max generated tokens per judge response.")
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Optional cap on the number of image pairs evaluated before question expansion.",
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed for deterministic ordering.")
    add_common_eval_arguments(parser)
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

    annotations_path = args.annotations.resolve()
    if not annotations_path.exists():
        raise SystemExit(f"Annotation file not found: {annotations_path}")

    benchmark = load_benchmark(questions_path)
    annotations = load_annotations(annotations_path)
    groups = build_group_specs(benchmark, annotations=annotations)
    selected_groups = resolve_selected_groups(groups, args.subset_path)
    question_types = parse_question_types(args.question_types)
    selected_groups = filter_groups(
        selected_groups,
        category=args.category,
        subcategory=args.subcategory,
        source_variant=args.source_variant,
        question_types=question_types,
        subset=args.subset,
        edit_type=args.edit_type,
        count_direction=args.count_direction,
    )
    all_image_pairs = build_image_pairs(selected_groups)
    image_pairs, sampling_info = apply_pair_sampling(
        all_image_pairs,
        sample_fraction=args.sample_fraction,
        sample_strategy=args.sample_strategy,
        seed=args.seed,
        max_samples=args.max_samples,
    )
    validate_tool_condition_args(
        tool_condition=args.tool_condition,
        evidence_manifest_path=args.evidence_manifest,
        evidence_qc_filter=args.evidence_qc_filter,
        missing_evidence_policy=args.missing_evidence_policy,
    )
    evidence_manifest_path = args.evidence_manifest.resolve() if args.evidence_manifest else None
    evidence_manifest = load_evidence_manifest(evidence_manifest_path) if evidence_manifest_path else {}
    image_pairs, evidence_filter_stats = filter_image_pairs_for_evidence(
        image_pairs,
        eval_mode=args.eval_mode,
        tool_condition=args.tool_condition,
        evidence_manifest=evidence_manifest,
        evidence_manifest_path=evidence_manifest_path,
        evidence_qc_filter=args.evidence_qc_filter,
        missing_evidence_policy=args.missing_evidence_policy,
    )
    if not image_pairs:
        raise SystemExit("No image pairs matched the selected dataset slice.")

    backbone_ref = resolve_hf_model_ref(args.backbone_model)
    judge_ref = resolve_hf_model_ref(args.judge_model)
    questions_by_key = build_questions_by_key(selected_groups)
    total_evaluations = count_total_evaluations(image_pairs, questions_by_key)
    all_tasks = build_eval_tasks(image_pairs, questions_by_key)

    if args.dry_run:
        print_evidence_filter_summary(evidence_filter_stats, questions_per_pair=len(next(iter(questions_by_key.values()), [])))
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
                script_type="open",
                eval_mode=args.eval_mode,
                backbone_alias=backbone_ref.alias,
                judge_alias=judge_ref.alias,
                tasks=pending_tasks,
                total_evaluations=total_evaluations,
                prime=args.prime,
                tool_condition=args.tool_condition,
                evidence_manifest=evidence_manifest,
                evidence_manifest_path=evidence_manifest_path,
                evidence_qc_filter=args.evidence_qc_filter,
                missing_evidence_policy=args.missing_evidence_policy,
                image_group_labels=args.image_group_labels,
            )
            return 0
        print_dry_run_preview(
            script_type="open",
            eval_mode=args.eval_mode,
            backbone_alias=backbone_ref.alias,
            judge_alias=judge_ref.alias,
            image_pairs=image_pairs,
            questions_by_key=questions_by_key,
            total_evaluations=total_evaluations,
            prime=args.prime,
            tool_condition=args.tool_condition,
            evidence_manifest=evidence_manifest,
            evidence_manifest_path=evidence_manifest_path,
            evidence_qc_filter=args.evidence_qc_filter,
            missing_evidence_policy=args.missing_evidence_policy,
            image_group_labels=args.image_group_labels,
        )
        return 0

    require_open_runtime(args.load_in_4bit)
    resolved_dtype_name, resolved_dtype = resolve_torch_dtype(args.dtype)

    free_bytes = get_free_cuda_bytes()
    placement_plans, residual_free_bytes = plan_requested_models(
        backbone_ref=backbone_ref,
        judge_ref=judge_ref,
        free_bytes_by_device=free_bytes,
        dtype_name=resolved_dtype_name,
        dtype=resolved_dtype,
        load_in_4bit=args.load_in_4bit,
    )

    run_config_base = {
        "script_type": "open",
        "questions_path": str(questions_path),
        "annotations_path": str(annotations_path),
        "eval_mode": args.eval_mode,
        "input_mode": args.eval_mode,
        "subset_path": str(args.subset_path.resolve()) if args.subset_path else None,
        "backbone_model_alias": backbone_ref.alias,
        "backbone_model_id": backbone_ref.model_id,
        "judge_model_alias": judge_ref.alias,
        "judge_model_id": judge_ref.model_id,
        "dtype": resolved_dtype_name,
        "load_in_4bit": bool(args.load_in_4bit),
        "temperature": args.temperature,
        "max_output_tokens": args.max_output_tokens,
        "judge_max_output_tokens": args.judge_max_output_tokens,
        "empty_response_policy": args.empty_response_policy,
        "empty_retry_max_output_tokens": args.empty_retry_max_output_tokens,
        "reasoning_effort": args.reasoning_effort,
        "max_samples": args.max_samples,
        "sample_fraction": args.sample_fraction,
        "sample_strategy": args.sample_strategy,
        "sampling": sampling_info,
        "seed": args.seed,
        "prime": args.prime,
        "prompt_prefix": PRIOR_PRIME_PREFIX if args.prime == "on" else None,
        "tool_condition": args.tool_condition,
        **both_raw_policy_fields(args.eval_mode, args.tool_condition),
        "evidence_manifest": str(evidence_manifest_path) if evidence_manifest_path else None,
        "evidence_qc_filter": args.evidence_qc_filter,
        "missing_evidence_policy": args.missing_evidence_policy,
        "image_group_labels": args.image_group_labels,
        **evidence_filter_stats,
        "category": args.category,
        "subcategory": args.subcategory,
        "source_variant": args.source_variant,
        "subset": args.subset,
        "edit_type": args.edit_type,
        "count_direction": args.count_direction,
        "question_types": question_types,
        "pair_count": len(image_pairs),
        "cuda_devices": cuda_device_report(free_bytes),
        "residual_cuda_memory_bytes_after_planning": residual_free_bytes,
        "placement_plans": {
            model_id: device_plan_to_dict(plan) for model_id, plan in placement_plans.items()
        },
        "response_target_count": total_evaluations,
        "max_retries": args.max_retries,
        "retry_base_seconds": args.retry_base_seconds,
        "retry_max_seconds": args.retry_max_seconds,
        "stop_on_quota": args.stop_on_quota,
        "max_consecutive_errors": args.max_consecutive_errors,
        "judge_structured_output": args.judge_structured_output,
        "closed_form_structured_output": args.closed_form_structured_output,
        "structured_output_fallback": args.structured_output_fallback,
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
        run_dir = args.output_root.resolve() / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        existing_records = []
    run_config = {"run_id": run_id, **run_config_base, "resume": args.resume}

    loaded_models: Dict[str, LoadedHFModel] = {}
    records: List[Dict[str, Any]] = list(existing_records)
    records_path = run_dir / "records.jsonl"
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
        f"Starting open-model evaluation: {len(pending_tasks)} pending responses "
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
                placement_plans=placement_plans,
                loaded_models=loaded_models,
                dtype_name=resolved_dtype_name,
                dtype=resolved_dtype,
                load_in_4bit=args.load_in_4bit,
                temperature=args.temperature,
                max_output_tokens=args.max_output_tokens,
                judge_max_output_tokens=args.judge_max_output_tokens,
                empty_response_policy=args.empty_response_policy,
                empty_retry_max_output_tokens=args.empty_retry_max_output_tokens,
                reasoning_effort=args.reasoning_effort,
                judge_structured_output=args.judge_structured_output,
                closed_form_structured_output=args.closed_form_structured_output,
                run_id=run_id,
                prime=args.prime,
                tool_condition=args.tool_condition,
                evidence_manifest=evidence_manifest,
                evidence_manifest_path=evidence_manifest_path,
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
        script_type="open",
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


def require_open_runtime(load_in_4bit: bool) -> None:
    if torch is None:
        raise SystemExit("PyTorch is not installed. Install project dependencies before running eval_open_vlm.py.")
    if AutoConfig is None or AutoProcessor is None:
        raise SystemExit("transformers is not installed or is missing required processor/config classes.")
    if (
        AutoModelForImageTextToText is None
        and AutoModelForMultimodalLM is None
        and Gemma4ForConditionalGeneration is None
    ):
        raise SystemExit("transformers is not installed or is missing required multimodal classes.")
    if init_empty_weights is None or infer_auto_device_map is None:
        raise SystemExit("accelerate is not installed or is missing required device-planning helpers.")
    if not torch.cuda.is_available() or torch.cuda.device_count() < 1:
        raise SystemExit("CUDA is required for eval_open_vlm.py but no CUDA device is available.")
    if load_in_4bit and BitsAndBytesConfig is None:
        raise SystemExit("4-bit loading was requested but BitsAndBytesConfig is unavailable.")
    if load_in_4bit:
        import importlib.util

        if importlib.util.find_spec("bitsandbytes") is None:
            raise SystemExit("4-bit loading was requested but bitsandbytes is not installed.")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    if torch is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)


def resolve_hf_model_ref(raw_value: str) -> HFModelRef:
    model_id = OPEN_MODEL_ALIASES.get(raw_value, raw_value)
    return HFModelRef(alias=raw_value, model_id=model_id)


def infer_model_family(*, model_id: str, config: Any | None = None) -> str:
    lower_model_id = model_id.lower()
    model_type = str(getattr(config, "model_type", "")).lower()
    if "gemma-4" in lower_model_id or model_type == "gemma4":
        return "gemma4"
    return "image_text_to_text"


def select_model_loader(*, model_family: str) -> Any:
    if model_family == "gemma4":
        if AutoModelForMultimodalLM is not None:
            return AutoModelForMultimodalLM
        if Gemma4ForConditionalGeneration is not None:
            return Gemma4ForConditionalGeneration
        if AutoModelForImageTextToText is not None:
            return AutoModelForImageTextToText
        raise SystemExit(
            "Gemma 4 requires a transformers build with Gemma4 multimodal support. "
            "Upgrade transformers or use a supported Qwen-VL checkpoint."
        )
    if AutoModelForImageTextToText is not None:
        return AutoModelForImageTextToText
    if AutoModelForMultimodalLM is not None:
        return AutoModelForMultimodalLM
    raise SystemExit("No compatible multimodal model loader is available in the current transformers install.")


def resolve_torch_dtype(dtype_name: str) -> Tuple[str, Any]:
    if torch is None:
        raise SystemExit("PyTorch is required to resolve local model dtype.")
    if dtype_name == "auto":
        if torch.cuda.is_bf16_supported():
            return "bfloat16", torch.bfloat16
        return "float16", torch.float16
    mapping = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    return dtype_name, mapping[dtype_name]


def get_free_cuda_bytes() -> Dict[int, int]:
    assert torch is not None
    torch.cuda.empty_cache()
    free_bytes: Dict[int, int] = {}
    for device_index in range(torch.cuda.device_count()):
        free, _total = torch.cuda.mem_get_info(device_index)
        free_bytes[device_index] = int(free)
    return free_bytes


def cuda_device_report(free_bytes_by_device: Dict[int, int]) -> Dict[str, Dict[str, Any]]:
    assert torch is not None
    report: Dict[str, Dict[str, Any]] = {}
    for device_index, free_bytes in free_bytes_by_device.items():
        properties = torch.cuda.get_device_properties(device_index)
        report[str(device_index)] = {
            "name": properties.name,
            "free_bytes": free_bytes,
            "total_bytes": int(properties.total_memory),
        }
    return report


def plan_requested_models(
    *,
    backbone_ref: HFModelRef,
    judge_ref: HFModelRef,
    free_bytes_by_device: Dict[int, int],
    dtype_name: str,
    dtype: Any,
    load_in_4bit: bool,
) -> Tuple[Dict[str, DevicePlan], Dict[int, int]]:
    residual = dict(free_bytes_by_device)
    plans: Dict[str, DevicePlan] = {}

    for model_ref in (backbone_ref, judge_ref):
        if model_ref.canonical in plans:
            continue
        plan = plan_single_model(
            model_id=model_ref.model_id,
            dtype_name=dtype_name,
            dtype=dtype,
            load_in_4bit=load_in_4bit,
            free_bytes_by_device=residual,
        )
        plans[model_ref.canonical] = plan
        for device_index, reserved in plan.reserved_bytes_by_device.items():
            residual[device_index] = max(0, residual[device_index] - reserved)

    return plans, residual


def plan_single_model(
    *,
    model_id: str,
    dtype_name: str,
    dtype: Any,
    load_in_4bit: bool,
    free_bytes_by_device: Dict[int, int],
) -> DevicePlan:
    config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
    model_family = infer_model_family(model_id=model_id, config=config)
    meta_model = build_meta_model(config=config, model_family=model_family)
    estimated_total_bytes = estimate_model_bytes(model=meta_model, dtype=dtype, load_in_4bit=load_in_4bit)

    required_single_device_bytes = int(estimated_total_bytes * MODEL_BYTES_SAFETY_FACTOR)
    chosen_device = pick_single_device(required_single_device_bytes, free_bytes_by_device)
    if chosen_device is not None:
        return DevicePlan(
            mode="single",
            assigned_devices=[chosen_device],
            reserved_bytes_by_device={chosen_device: required_single_device_bytes},
            device_map={"": f"cuda:{chosen_device}"},
            max_memory=None,
            estimated_total_bytes=estimated_total_bytes,
            model_family=model_family,
            dtype_name=dtype_name,
            load_in_4bit=load_in_4bit,
        )

    sharded_plan = plan_sharded_model(
        meta_model=meta_model,
        model_family=model_family,
        dtype=dtype,
        dtype_name=dtype_name,
        load_in_4bit=load_in_4bit,
        free_bytes_by_device=free_bytes_by_device,
        estimated_total_bytes=estimated_total_bytes,
    )
    if sharded_plan is None:
        raise SystemExit(build_insufficient_vram_message(model_id, dtype_name, load_in_4bit, free_bytes_by_device, estimated_total_bytes))
    return sharded_plan


def build_meta_model(*, config: Any, model_family: str) -> Any:
    if init_empty_weights is None:
        raise SystemExit("accelerate or transformers multimodal classes are unavailable.")
    loader = select_model_loader(model_family=model_family)
    with init_empty_weights():
        return loader.from_config(config, trust_remote_code=True)


def estimate_model_bytes(*, model: Any, dtype: Any, load_in_4bit: bool) -> int:
    total_params = 0
    for tensor in list(model.parameters()) + list(model.buffers()):
        total_params += int(tensor.numel())
    if load_in_4bit:
        return int(total_params * torch.tensor([], dtype=torch.float16).element_size() * FOUR_BIT_MEMORY_FACTOR)
    return int(total_params * torch.tensor([], dtype=dtype).element_size())


def pick_single_device(required_bytes: int, free_bytes_by_device: Dict[int, int]) -> Optional[int]:
    candidates = []
    for device_index, free_bytes in free_bytes_by_device.items():
        usable_free = int(free_bytes * USABLE_GPU_MEMORY_RATIO)
        if usable_free >= required_bytes:
            candidates.append((usable_free, device_index))
    if not candidates:
        return None
    candidates.sort()
    return candidates[0][1]


def plan_sharded_model(
    *,
    meta_model: Any,
    model_family: str,
    dtype: Any,
    dtype_name: str,
    load_in_4bit: bool,
    free_bytes_by_device: Dict[int, int],
    estimated_total_bytes: int,
) -> Optional[DevicePlan]:
    if infer_auto_device_map is None:
        return None

    effective_ratio = FOUR_BIT_MEMORY_FACTOR if load_in_4bit else 1.0
    max_memory = {
        device_index: int((free_bytes * USABLE_GPU_MEMORY_RATIO) / effective_ratio)
        for device_index, free_bytes in free_bytes_by_device.items()
    }
    no_split_modules = getattr(meta_model, "_no_split_modules", None) or []
    infer_dtype = torch.float16 if load_in_4bit else dtype
    device_map = infer_auto_device_map(
        meta_model,
        max_memory=max_memory,
        no_split_module_classes=no_split_modules,
        dtype=infer_dtype,
    )
    if any(str(device).lower() in {"cpu", "disk"} for device in device_map.values()):
        return None

    direct_sizes = direct_module_sizes(meta_model, infer_dtype, load_in_4bit)
    reserved_bytes_by_device = accumulate_reserved_bytes(device_map, direct_sizes)
    assigned_devices = sorted(device for device in reserved_bytes_by_device if isinstance(device, int))
    if not assigned_devices:
        return None

    return DevicePlan(
        mode="sharded",
        assigned_devices=assigned_devices,
        reserved_bytes_by_device=reserved_bytes_by_device,
        device_map=device_map,
        max_memory=max_memory,
        estimated_total_bytes=estimated_total_bytes,
        model_family=model_family,
        dtype_name=dtype_name,
        load_in_4bit=load_in_4bit,
    )


def direct_module_sizes(model: Any, dtype: Any, load_in_4bit: bool) -> Dict[str, int]:
    factor = FOUR_BIT_MEMORY_FACTOR if load_in_4bit else 1.0
    bytes_per_value = torch.tensor([], dtype=torch.float16 if load_in_4bit else dtype).element_size()
    sizes: Dict[str, int] = {}
    for module_name, module in model.named_modules():
        size = 0
        for parameter in module.parameters(recurse=False):
            size += parameter.numel() * bytes_per_value
        for buffer in module.buffers(recurse=False):
            size += buffer.numel() * bytes_per_value
        sizes[module_name] = int(size * factor)
    return sizes


def accumulate_reserved_bytes(device_map: Dict[str, Any], direct_sizes: Dict[str, int]) -> Dict[int, int]:
    reserved: Dict[int, int] = {}
    assigned_module_names = sorted(device_map, key=lambda name: len(name))
    for module_name, size in direct_sizes.items():
        if size == 0:
            continue
        owner = find_assigned_owner(module_name, assigned_module_names, device_map)
        device_index = normalize_device_index(owner)
        if device_index is not None:
            reserved[device_index] = reserved.get(device_index, 0) + size
    return reserved


def find_assigned_owner(module_name: str, assigned_module_names: Sequence[str], device_map: Dict[str, Any]) -> Any:
    current = module_name
    while True:
        if current in device_map:
            return device_map[current]
        if not current:
            break
        if "." not in current:
            current = ""
        else:
            current = current.rsplit(".", 1)[0]
    if "" in device_map:
        return device_map[""]
    return None


def normalize_device_index(value: Any) -> Optional[int]:
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.startswith("cuda:"):
        try:
            return int(value.split(":", 1)[1])
        except ValueError:
            return None
    return None


def build_insufficient_vram_message(
    model_id: str,
    dtype_name: str,
    load_in_4bit: bool,
    free_bytes_by_device: Dict[int, int],
    estimated_total_bytes: int,
) -> str:
    lines = [
        "Insufficient VRAM for the requested model configuration.",
        f"model_id: {model_id}",
        f"dtype: {dtype_name}",
        f"load_in_4bit: {load_in_4bit}",
        f"estimated_model_bytes: {estimated_total_bytes}",
        "available_cuda_memory_bytes:",
    ]
    lines.extend([f"- cuda:{device_index}: {free_bytes}" for device_index, free_bytes in free_bytes_by_device.items()])
    return "\n".join(lines)


def device_plan_to_dict(plan: DevicePlan) -> Dict[str, Any]:
    return {
        "mode": plan.mode,
        "assigned_devices": plan.assigned_devices,
        "reserved_bytes_by_device": plan.reserved_bytes_by_device,
        "device_map": plan.device_map,
        "max_memory": plan.max_memory,
        "estimated_total_bytes": plan.estimated_total_bytes,
        "model_family": plan.model_family,
        "dtype_name": plan.dtype_name,
        "load_in_4bit": plan.load_in_4bit,
    }


def get_loaded_model(
    *,
    model_ref: HFModelRef,
    placement_plans: Dict[str, DevicePlan],
    loaded_models: Dict[str, LoadedHFModel],
    dtype_name: str,
    dtype: Any,
    load_in_4bit: bool,
) -> LoadedHFModel:
    cached = loaded_models.get(model_ref.canonical)
    if cached is not None:
        return cached

    plan = placement_plans[model_ref.canonical]
    loader = select_model_loader(model_family=plan.model_family)
    processor = AutoProcessor.from_pretrained(model_ref.model_id, trust_remote_code=True)
    model_kwargs: Dict[str, Any] = {
        "trust_remote_code": True,
        "low_cpu_mem_usage": True,
        "device_map": plan.device_map,
    }
    if plan.max_memory is not None:
        model_kwargs["max_memory"] = plan.max_memory

    if load_in_4bit:
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=dtype,
        )
    else:
        model_kwargs["dtype"] = dtype

    model = loader.from_pretrained(model_ref.model_id, **model_kwargs)
    runtime = LoadedHFModel(
        ref=model_ref,
        processor=processor,
        model=model,
        device_plan=plan,
        model_family=plan.model_family,
        dtype_name=dtype_name,
        load_in_4bit=load_in_4bit,
    )
    loaded_models[model_ref.canonical] = runtime
    return runtime


def evaluate_question(
    *,
    pair: ImagePair,
    question: QuestionSpec,
    eval_mode: str,
    backbone_ref: HFModelRef,
    judge_ref: HFModelRef,
    placement_plans: Dict[str, DevicePlan],
    loaded_models: Dict[str, LoadedHFModel],
    dtype_name: str,
    dtype: Any,
    load_in_4bit: bool,
    temperature: float,
    max_output_tokens: int,
    judge_max_output_tokens: int,
    empty_response_policy: str,
    empty_retry_max_output_tokens: int,
    reasoning_effort: str,
    judge_structured_output: str,
    closed_form_structured_output: str,
    run_id: str,
    prime: str,
    tool_condition: str,
    evidence_manifest: Dict[str, Dict[str, Any]],
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
    answers = answers_for_mode(question, eval_mode, tool_condition)
    ground_truth_target, biased_target = scoring_targets(question.question_type, answers)

    started = time.perf_counter()
    try:
        backbone_runtime = get_loaded_model(
            model_ref=backbone_ref,
            placement_plans=placement_plans,
            loaded_models=loaded_models,
            dtype_name=dtype_name,
            dtype=dtype,
            load_in_4bit=load_in_4bit,
        )
        raw_response = generate_with_hf_model(
            runtime=backbone_runtime,
            prompt=prompt,
            image_paths=labeled_images,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
        )
    except Exception as exc:
        metadata = error_metadata(exc)
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
            backbone_max_output_tokens=max_output_tokens,
            judge_max_output_tokens=judge_max_output_tokens,
            reasoning_effort=reasoning_effort,
            judge_structured_output=judge_structured_output,
            closed_form_structured_output=closed_form_structured_output,
        )

    empty_retry_used = False
    empty_retry_success = False
    empty_retry_reason = None
    empty_response = not strip_thinking_blocks(raw_response).strip()
    if empty_response:
        if empty_response_policy == "retry_once":
            empty_retry_used = True
            empty_retry_reason = "blank_backbone_response"
            raw_retry = generate_with_hf_model(
                runtime=backbone_runtime,
                prompt=prompt,
                image_paths=labeled_images,
                temperature=temperature,
                max_output_tokens=empty_retry_max_output_tokens,
            )
            if strip_thinking_blocks(raw_retry).strip():
                raw_response = raw_retry
                empty_response = False
                empty_retry_success = True
        elif empty_response_policy == "error":
            raise RuntimeError("Backbone returned an empty response.")

    parsed_response = parse_response(question.question_type, raw_response)
    if empty_response:
        judge_raw = None
        judge_reason = "empty response"
        scored_label = "other"
        error_message = None
        error_kind = None
        fatal_error = False
    elif question.question_type in {"yes_no", "open"}:
        try:
            judge_runtime = get_loaded_model(
                model_ref=judge_ref,
                placement_plans=placement_plans,
                loaded_models=loaded_models,
                dtype_name=dtype_name,
                dtype=dtype,
                load_in_4bit=load_in_4bit,
            )
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
            judge_raw = generate_with_hf_model(
                runtime=judge_runtime,
                prompt=judge_prompt,
                image_paths=[],
                temperature=0.0,
                max_output_tokens=judge_max_output_tokens,
            )
            judge_result = parse_judge_response(judge_raw)
            scored_label = judge_result["label"]
            judge_reason = judge_result["reason"]
            error_message = None
            error_kind = None
            fatal_error = False
        except Exception as exc:
            metadata = error_metadata(exc, default="judge_error")
            judge_raw = None
            scored_label = "error"
            judge_reason = None
            error_message = f"Judge scoring failed: {exc}"
            error_kind = metadata["error_kind"] if metadata["error_kind"] != "unknown_error" else "judge_error"
            fatal_error = metadata["fatal_error"]
    else:
        judge_raw = None
        judge_reason = None
        scored_label = score_closed_form(question.question_type, parsed_response, answers)
        error_message = None
        error_kind = None
        fatal_error = False
    record_judge_model = judge_ref.model_id if question.question_type in {"yes_no", "open"} else "letter_match"

    return {
        "run_id": run_id,
        "script_type": "open",
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
        "prompt": prompt,
        "raw_response_text": raw_response,
        "parsed_response": parsed_response,
        "empty_response": empty_response,
        "empty_retry_used": empty_retry_used,
        "empty_retry_success": empty_retry_success,
        "empty_retry_reason": empty_retry_reason,
        "backbone_max_output_tokens": max_output_tokens,
        "judge_max_output_tokens": judge_max_output_tokens,
        "reasoning_effort_requested": reasoning_effort,
        "reasoning_control_applied": "not_applicable_hf_direct",
        "reasoning_fallback_reason": None if reasoning_effort in {"off", "provider_default"} else "hf_direct_runner_does_not_enable_extended_thinking",
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
        "retry_count": 0,
        "resume_skipped": False,
        **structured_record_fields(
            judge_mode=judge_structured_output,
            closed_form_mode=closed_form_structured_output,
            used=False,
            target="none",
            schema_id=None,
            fallback_reason="unsupported_hf_direct_runner",
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
            tool_condition=tool_condition,
            evidence_manifest_path=evidence_manifest_path,
            image_groups=image_groups,
            evidence_record=evidence_record,
            evidence_qc_filter=evidence_qc_filter,
            missing_evidence_policy=missing_evidence_policy,
        ),
    }


def generate_with_hf_model(
    *,
    runtime: LoadedHFModel,
    prompt: str,
    image_paths: Sequence[Path | LabeledImage],
    temperature: float,
    max_output_tokens: int,
) -> str:
    pil_images = [Image.open(image_item_path(item)).convert("RGB") for item in image_paths]
    if runtime.model_family == "gemma4":
        batch = build_gemma4_inputs(runtime=runtime, prompt=prompt, image_paths=image_paths)
    else:
        content: List[Dict[str, Any]] = []
        for item in image_paths:
            label = image_item_label(item)
            if label:
                content.append({"type": "text", "text": label})
            content.append({"type": "image", "image": f"file://{image_item_path(item).resolve()}"})
        content.append({"type": "text", "text": prompt})
        messages = [
            {
                "role": "user",
                "content": content,
            }
        ]
        text = runtime.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        if pil_images:
            batch = runtime.processor(text=[text], images=pil_images, padding=True, return_tensors="pt")
        else:
            batch = runtime.processor(text=[text], padding=True, return_tensors="pt")

    input_device = runtime_input_device(runtime)
    batch = move_batch_to_device(batch, input_device)

    generation_kwargs: Dict[str, Any] = {
        "max_new_tokens": max_output_tokens,
        "do_sample": temperature > 0,
    }
    if temperature > 0:
        generation_kwargs["temperature"] = temperature

    with torch.no_grad():
        output_ids = runtime.model.generate(**batch, **generation_kwargs)

    input_ids = batch.get("input_ids")
    if input_ids is not None:
        trimmed_ids = output_ids[:, input_ids.shape[1] :]
    else:
        trimmed_ids = output_ids
    if runtime.model_family == "gemma4" and hasattr(runtime.processor, "decode"):
        text_output = runtime.processor.decode(trimmed_ids[0], skip_special_tokens=True)
    else:
        text_output = runtime.processor.batch_decode(
            trimmed_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]
    return text_output.strip()


def build_gemma4_inputs(*, runtime: LoadedHFModel, prompt: str, image_paths: Sequence[Path | LabeledImage]) -> Dict[str, Any]:
    content: List[Dict[str, Any]] = []
    for item in image_paths:
        label = image_item_label(item)
        if label:
            content.append({"type": "text", "text": label})
        content.append({"type": "image", "url": image_item_path(item).resolve().as_uri()})
    content.append({"type": "text", "text": prompt})
    messages = [{"role": "user", "content": content}]
    batch = runtime.processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )
    return dict(batch)


def image_item_path(item: Path | LabeledImage) -> Path:
    return item.path if isinstance(item, LabeledImage) else item


def image_item_label(item: Path | LabeledImage) -> Optional[str]:
    return item.label if isinstance(item, LabeledImage) else None


def runtime_input_device(runtime: LoadedHFModel) -> Any:
    if hasattr(runtime.model, "device") and str(runtime.model.device) != "meta":
        return runtime.model.device
    hf_device_map = getattr(runtime.model, "hf_device_map", None)
    if hf_device_map:
        cuda_devices = []
        for value in hf_device_map.values():
            if isinstance(value, str) and value.startswith("cuda:"):
                cuda_devices.append(int(value.split(":", 1)[1]))
            elif isinstance(value, int):
                cuda_devices.append(value)
        if cuda_devices:
            return torch.device(f"cuda:{min(cuda_devices)}")
    if runtime.device_plan.assigned_devices:
        return torch.device(f"cuda:{runtime.device_plan.assigned_devices[0]}")
    raise RuntimeError("Could not determine an input device for the local model.")


def move_batch_to_device(batch: Dict[str, Any], device: Any) -> Dict[str, Any]:
    moved: Dict[str, Any] = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            moved[key] = value.to(device)
        else:
            moved[key] = value
    return moved


def build_error_record(
    *,
    run_id: str,
    pair: ImagePair,
    question: QuestionSpec,
    eval_mode: str,
    prompt: str,
    used_image_paths: Sequence[Path],
    tool_record_fields: Optional[Dict[str, Any]],
    image_group_labels: str,
    backbone_ref: HFModelRef,
    judge_ref: HFModelRef,
    ground_truth_target: Dict[str, Any],
    biased_target: Dict[str, Any],
    answers: Dict[str, Any],
    error_message: str,
    latency_ms: int,
    error_kind: str,
    fatal_error: bool,
    retry_count: int,
    backbone_max_output_tokens: int = 1024,
    judge_max_output_tokens: int = 512,
    reasoning_effort: str = "off",
    judge_structured_output: str = "auto",
    closed_form_structured_output: str = "auto",
) -> Dict[str, Any]:
    record_judge_model = judge_ref.model_id if question.question_type in {"yes_no", "open"} else "letter_match"
    return {
        "run_id": run_id,
        "script_type": "open",
        "backbone_model_alias": backbone_ref.alias,
        "backbone_model_id": backbone_ref.model_id,
        "judge_model_alias": judge_ref.alias,
        "judge_model_id": judge_ref.model_id,
        "eval_mode": eval_mode,
        "input_mode": eval_mode,
        **both_raw_policy_fields(eval_mode, str((tool_record_fields or {}).get("tool_condition") or "raw")),
        "category": pair.category,
        "subcategory": pair.subcategory,
        "source_variant": pair.source_variant,
        "question_id": question.question_id,
        "question_type": question.question_type,
        "original_image_path": str(pair.original_path),
        "edited_image_path": str(pair.edited_path),
        "used_image_paths": [str(path) for path in used_image_paths],
        "image_group_labels": image_group_labels,
        "prompt": prompt,
        "raw_response_text": None,
        "parsed_response": None,
        "empty_response": False,
        "empty_retry_used": False,
        "empty_retry_success": False,
        "empty_retry_reason": None,
        "backbone_max_output_tokens": backbone_max_output_tokens,
        "judge_max_output_tokens": judge_max_output_tokens,
        "reasoning_effort_requested": reasoning_effort,
        "reasoning_control_applied": "not_applicable_hf_direct",
        "reasoning_fallback_reason": None if reasoning_effort in {"off", "provider_default"} else "hf_direct_runner_does_not_enable_extended_thinking",
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
        "resume_skipped": False,
        **structured_record_fields(
            judge_mode=judge_structured_output,
            closed_form_mode=closed_form_structured_output,
            used=False,
            target="none",
            schema_id=None,
            fallback_reason="unsupported_hf_direct_runner",
        ),
        **(tool_record_fields or {}),
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


if __name__ == "__main__":
    sys.exit(main())
