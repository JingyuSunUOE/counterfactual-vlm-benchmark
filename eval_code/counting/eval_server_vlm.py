#!/usr/bin/env python3
from __future__ import annotations

import argparse
import base64
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from dotenv import load_dotenv

EVAL_CODE_DIR = Path(__file__).resolve().parents[1]
if str(EVAL_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(EVAL_CODE_DIR))

from server_vram_planner import add_server_preflight_arguments, plan_pair_from_args, preflight_or_warn
from structured_outputs import (
    JUDGE_LABEL_SCHEMA_ID,
    MC_ANSWER_SCHEMA_ID,
    OPEN_ANSWER_SCHEMA_ID,
    YES_NO_ANSWER_SCHEMA_ID,
    format_structured_fallback_reason,
    looks_like_structured_output_unsupported,
    normalize_structured_fallback,
    openai_chat_response_format,
    structured_record_fields,
    strip_thinking_blocks,
)
from mime_utils import detect_mime_type as detect_content_mime_type

from eval_common import (
    DEFAULT_ANNOTATIONS_PATH,
    DEFAULT_QUESTIONS_PATH,
    DEFAULT_RAW_RUNS_ROOT,
    ImagePair,
    NullProgress,
    PRIOR_PRIME_PREFIX,
    ProgressBar,
    QuestionSpec,
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
    from openai import OpenAI
except Exception:  # pragma: no cover - import availability depends on env
    OpenAI = None


DEFAULT_SERVER_URL = "http://localhost:8000"
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
    "gemma-4-31b-it": "google/gemma-4-31B-it",
}
OPENAI_JUDGE_ALIASES = {
    "gpt-4o-mini": "gpt-4o-mini",
    "gpt-4.1": "gpt-4.1",
}


@dataclass(frozen=True)
class ServerModelRef:
    alias: str
    model_id: str

    @property
    def canonical(self) -> str:
        return self.model_id


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate VLMs through an OpenAI-compatible server on the counting benchmark."
    )
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
    parser.add_argument("--backbone-model", required=True, help="Open-model alias or server model id.")
    parser.add_argument(
        "--server-url",
        default=DEFAULT_SERVER_URL,
        help="OpenAI-compatible VLM server URL. /v1 is appended when omitted.",
    )
    parser.add_argument(
        "--server-api-key",
        default=None,
        help="API key for the OpenAI-compatible server. Defaults to OPENAI_COMPATIBLE_API_KEY or EMPTY.",
    )
    parser.add_argument(
        "--server-model-id",
        default=None,
        help="Override the concrete model id sent to the OpenAI-compatible server.",
    )
    parser.add_argument("--judge-model", default="gpt-4o-mini", help="Judge model alias or model id.")
    parser.add_argument(
        "--judge-provider",
        choices=("openai", "server"),
        default="openai",
        help="Use OpenAI or the OpenAI-compatible server for yes_no/open judging.",
    )
    parser.add_argument(
        "--judge-server-url",
        default=None,
        help="OpenAI-compatible judge server URL. Defaults to --server-url.",
    )
    parser.add_argument(
        "--judge-server-api-key",
        default=None,
        help="API key for the judge server. Defaults to --server-api-key.",
    )
    add_server_preflight_arguments(parser)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_RAW_RUNS_ROOT,
        help="Root directory for run outputs.",
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

    backbone_ref = resolve_server_model_ref(args.backbone_model, override_model_id=args.server_model_id)
    judge_ref = resolve_judge_model_ref(args.judge_model, provider=args.judge_provider)
    questions_by_key = build_questions_by_key(selected_groups)
    total_evaluations = count_total_evaluations(image_pairs, questions_by_key)
    all_tasks = build_eval_tasks(image_pairs, questions_by_key)
    server_vram_plan, judge_server_vram_plan = resolve_server_preflight_plans(args, backbone_ref, judge_ref)

    if args.dry_run:
        print_evidence_filter_summary(evidence_filter_stats, questions_per_pair=len(next(iter(questions_by_key.values()), [])))
        print(
            f"server_model_id={backbone_ref.model_id} server_url={args.server_url} "
            f"judge_provider={args.judge_provider} judge_model_id={judge_ref.model_id}"
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
                script_type="server",
                eval_mode=args.eval_mode,
                backbone_alias=backbone_ref.alias,
                judge_alias=judge_ref.alias,
                tasks=pending_tasks,
                total_evaluations=total_evaluations,
                prime=args.prime,
                judge_structured_output=args.judge_structured_output,
                closed_form_structured_output=args.closed_form_structured_output,
                structured_output_fallback=args.structured_output_fallback,
                tool_condition=args.tool_condition,
                evidence_manifest=evidence_manifest,
                evidence_manifest_path=evidence_manifest_path,
                evidence_qc_filter=args.evidence_qc_filter,
                missing_evidence_policy=args.missing_evidence_policy,
                image_group_labels=args.image_group_labels,
            )
            return 0
        print_dry_run_preview(
            script_type="server",
            eval_mode=args.eval_mode,
            backbone_alias=backbone_ref.alias,
            judge_alias=judge_ref.alias,
            image_pairs=image_pairs,
            questions_by_key=questions_by_key,
            total_evaluations=total_evaluations,
            prime=args.prime,
            judge_structured_output=args.judge_structured_output,
            closed_form_structured_output=args.closed_form_structured_output,
            structured_output_fallback=args.structured_output_fallback,
            tool_condition=args.tool_condition,
            evidence_manifest=evidence_manifest,
            evidence_manifest_path=evidence_manifest_path,
            evidence_qc_filter=args.evidence_qc_filter,
            missing_evidence_policy=args.missing_evidence_policy,
            image_group_labels=args.image_group_labels,
        )
        return 0

    require_runtime(args)

    server_api_key = resolve_server_api_key(args.server_api_key)
    judge_server_url = args.judge_server_url or args.server_url
    judge_server_api_key = args.judge_server_api_key or server_api_key

    clients = {
        "backbone": OpenAI(api_key=server_api_key, base_url=normalize_openai_compatible_url(args.server_url)),
    }
    if args.judge_provider == "server":
        clients["judge"] = OpenAI(
            api_key=judge_server_api_key,
            base_url=normalize_openai_compatible_url(judge_server_url),
        )
    else:
        clients["judge"] = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    run_config_base = {
        "script_type": "server",
        "questions_path": str(questions_path),
        "annotations_path": str(annotations_path),
        "eval_mode": args.eval_mode,
        "input_mode": args.eval_mode,
        "subset_path": str(args.subset_path.resolve()) if args.subset_path else None,
        "backbone_model_alias": backbone_ref.alias,
        "backbone_model_id": backbone_ref.model_id,
        "judge_model_alias": judge_ref.alias,
        "judge_model_id": judge_ref.model_id,
        "server_url": args.server_url,
        "server_model_id": backbone_ref.model_id,
        "judge_provider": args.judge_provider,
        "judge_server_url": judge_server_url if args.judge_provider == "server" else None,
        "server_preflight": args.server_preflight,
        "server_framework": args.server_framework,
        "server_vram_settings": server_vram_settings(args, prefix="server"),
        "judge_server_vram_settings": server_vram_settings(args, prefix="judge_server") if args.judge_provider == "server" else None,
        "server_vram_plan": server_vram_plan,
        "judge_server_vram_plan": judge_server_vram_plan,
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
    write_json(run_dir / "run_config.json", run_config)

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
    print(
        f"Starting server-model evaluation: {len(pending_tasks)} pending responses "
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
                clients=clients,
                judge_provider=args.judge_provider,
                server_url=args.server_url,
                judge_server_url=judge_server_url,
                temperature=args.temperature,
                max_output_tokens=args.max_output_tokens,
                judge_max_output_tokens=args.judge_max_output_tokens,
                empty_response_policy=args.empty_response_policy,
                empty_retry_max_output_tokens=args.empty_retry_max_output_tokens,
                reasoning_effort=args.reasoning_effort,
                run_id=run_id,
                prime=args.prime,
                max_retries=args.max_retries,
                retry_base_seconds=args.retry_base_seconds,
                retry_max_seconds=args.retry_max_seconds,
                stop_on_quota=args.stop_on_quota,
                judge_structured_output=args.judge_structured_output,
                closed_form_structured_output=args.closed_form_structured_output,
                structured_output_fallback=args.structured_output_fallback,
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
        script_type="server",
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


def resolve_server_model_ref(raw_value: str, *, override_model_id: Optional[str]) -> ServerModelRef:
    return ServerModelRef(alias=raw_value, model_id=override_model_id or OPEN_MODEL_ALIASES.get(raw_value, raw_value))


def resolve_judge_model_ref(raw_value: str, *, provider: str) -> ServerModelRef:
    if provider == "openai":
        return ServerModelRef(alias=raw_value, model_id=OPENAI_JUDGE_ALIASES.get(raw_value, raw_value))
    return ServerModelRef(alias=raw_value, model_id=OPEN_MODEL_ALIASES.get(raw_value, raw_value))


def require_runtime(args: argparse.Namespace) -> None:
    if OpenAI is None:
        raise SystemExit("openai package is required for eval_server_vlm.py.")
    if args.judge_provider == "openai" and not os.getenv("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY is required when --judge-provider openai.")


def resolve_server_preflight_plans(
    args: argparse.Namespace,
    backbone_ref: ServerModelRef,
    judge_ref: ServerModelRef,
) -> tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
    if args.server_preflight == "off":
        return None, None
    judge_server_url = args.judge_server_url or args.server_url
    same_judge_endpoint = (
        args.judge_provider == "server"
        and normalize_openai_compatible_url(judge_server_url) == normalize_openai_compatible_url(args.server_url)
    )
    backbone_plan, judge_plan = plan_pair_from_args(
        args=args,
        backbone_model_id=backbone_ref.model_id,
        judge_model_id=judge_ref.model_id,
        judge_is_server=args.judge_provider == "server",
        same_judge_endpoint=same_judge_endpoint,
    )
    plans = []
    if backbone_plan is not None:
        plans.append(("backbone_server", backbone_plan))
    if judge_plan is not None:
        plans.append(("judge_server", judge_plan))
    preflight_or_warn(args.server_preflight, plans)
    return (
        backbone_plan.to_json() if backbone_plan is not None else None,
        judge_plan.to_json() if judge_plan is not None else None,
    )


def server_vram_settings(args: argparse.Namespace, *, prefix: str) -> Dict[str, Any]:
    if prefix == "server":
        return {
            "hf_model_id": args.server_hf_model_id,
            "gpu_ids": args.server_gpu_ids,
            "dtype": args.server_dtype,
            "tensor_parallel_size": args.server_tensor_parallel_size,
            "gpu_memory_utilization": args.server_gpu_memory_utilization,
            "reserve_gb": args.server_reserve_gb,
            "load_in_4bit": args.server_load_in_4bit,
        }
    return {
        "hf_model_id": args.judge_server_hf_model_id,
        "gpu_ids": args.judge_server_gpu_ids,
        "dtype": args.judge_server_dtype or args.server_dtype,
        "tensor_parallel_size": args.judge_server_tensor_parallel_size or args.server_tensor_parallel_size,
        "gpu_memory_utilization": args.judge_server_gpu_memory_utilization or args.server_gpu_memory_utilization,
        "reserve_gb": args.judge_server_reserve_gb or args.server_reserve_gb,
        "load_in_4bit": args.judge_server_load_in_4bit,
    }


def resolve_server_api_key(raw_key: Optional[str]) -> str:
    return raw_key or os.getenv("OPENAI_COMPATIBLE_API_KEY") or "EMPTY"


def normalize_openai_compatible_url(url: str) -> str:
    normalized = url.rstrip("/")
    if not normalized.endswith("/v1"):
        normalized = f"{normalized}/v1"
    return normalized


def image_item_path(item: Path | LabeledImage) -> Path:
    return item.path if isinstance(item, LabeledImage) else item


def image_item_label(item: Path | LabeledImage) -> Optional[str]:
    return item.label if isinstance(item, LabeledImage) else None


def evaluate_question(
    *,
    pair: ImagePair,
    question: QuestionSpec,
    eval_mode: str,
    backbone_ref: ServerModelRef,
    judge_ref: ServerModelRef,
    clients: Dict[str, Any],
    judge_provider: str,
    server_url: str,
    judge_server_url: str,
    temperature: float,
    max_output_tokens: int,
    judge_max_output_tokens: int,
    empty_response_policy: str,
    empty_retry_max_output_tokens: int,
    reasoning_effort: str,
    run_id: str,
    prime: str,
    max_retries: int,
    retry_base_seconds: float,
    retry_max_seconds: float,
    stop_on_quota: bool,
    judge_structured_output: str,
    closed_form_structured_output: str,
    structured_output_fallback: str,
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
        raw_response = generate_vision(
            client=clients["backbone"],
            model=backbone_ref.model_id,
            prompt=prompt,
            image_paths=labeled_images,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            max_retries=max_retries,
            retry_base_seconds=retry_base_seconds,
            retry_max_seconds=retry_max_seconds,
            stop_on_quota=stop_on_quota,
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
                retry_response = generate_vision(
                    client=clients["backbone"],
                    model=backbone_ref.model_id,
                    prompt=prompt,
                    image_paths=labeled_images,
                    temperature=temperature,
                    max_output_tokens=empty_retry_max_output_tokens,
                    max_retries=max_retries,
                    retry_base_seconds=retry_base_seconds,
                    retry_max_seconds=retry_max_seconds,
                    stop_on_quota=stop_on_quota,
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
        backbone_retry_count = getattr(generate_vision, "last_retry_count", 0)
        backbone_structured_used = bool(getattr(generate_vision, "last_structured_output_used", False))
        backbone_structured_fallback_reason = getattr(generate_vision, "last_structured_output_fallback_reason", None)
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
            server_url=server_url,
            judge_provider=judge_provider,
            judge_server_url=judge_server_url,
            error_message=f"Backbone generation failed: {exc}",
            latency_ms=elapsed_ms(started),
            error_kind=metadata["error_kind"],
            fatal_error=metadata["fatal_error"],
            retry_count=metadata["retry_count"],
            judge_structured_output=judge_structured_output,
            closed_form_structured_output=closed_form_structured_output,
            structured_output_fallback_reason=getattr(generate_vision, "last_structured_output_fallback_reason", None),
            structured_output_schema=closed_form_schema,
            backbone_max_output_tokens=max_output_tokens,
            judge_max_output_tokens=judge_max_output_tokens,
            reasoning_effort=reasoning_effort,
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
            judge_raw = generate_text(
                client=clients["judge"],
                model=judge_ref.model_id,
                prompt=judge_prompt,
                temperature=0.0,
                max_output_tokens=judge_max_output_tokens,
                max_retries=max_retries,
                retry_base_seconds=retry_base_seconds,
                retry_max_seconds=retry_max_seconds,
                stop_on_quota=stop_on_quota,
                structured_output_mode=judge_structured_output,
                structured_output_schema=JUDGE_LABEL_SCHEMA_ID if judge_structured_output != "off" else None,
                structured_output_fallback=normalize_structured_fallback(
                    judge_structured_output,
                    structured_output_fallback,
                ),
            )
            judge_retry_count = getattr(generate_text, "last_retry_count", 0)
            judge_structured_used = bool(getattr(generate_text, "last_structured_output_used", False))
            judge_structured_fallback_reason = getattr(generate_text, "last_structured_output_fallback_reason", None)
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
        "script_type": "server",
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
        "reasoning_control_applied": "server_default" if reasoning_effort == "provider_default" else "not_applicable_server",
        "reasoning_fallback_reason": None if reasoning_effort in {"provider_default", "off"} else "openai_compatible_server_reasoning_control_not_supported",
        "resume_skipped": False,
        "server_url": server_url,
        "server_model_id": backbone_ref.model_id,
        "judge_provider": judge_provider,
        "judge_server_url": judge_server_url if judge_provider == "server" else None,
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
            tool_condition=tool_condition,
            evidence_manifest_path=evidence_manifest_path,
            image_groups=image_groups,
            evidence_record=evidence_record,
            evidence_qc_filter=evidence_qc_filter,
            missing_evidence_policy=missing_evidence_policy,
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
    tool_record_fields: Optional[Dict[str, Any]],
    image_group_labels: str,
    backbone_ref: ServerModelRef,
    judge_ref: ServerModelRef,
    ground_truth_target: Dict[str, Any],
    biased_target: Dict[str, Any],
    answers: Dict[str, Any],
    server_url: str,
    judge_provider: str,
    judge_server_url: str,
    error_message: str,
    latency_ms: int,
    error_kind: str,
    fatal_error: bool,
    retry_count: int,
    judge_structured_output: str = "auto",
    closed_form_structured_output: str = "auto",
    structured_output_fallback_reason: Optional[str] = None,
    structured_output_schema: Optional[str] = None,
    backbone_max_output_tokens: Optional[int] = None,
    judge_max_output_tokens: Optional[int] = None,
    reasoning_effort: str = "off",
) -> Dict[str, Any]:
    record_judge_model = judge_ref.model_id if question.question_type in {"yes_no", "open"} else "letter_match"
    return {
        "run_id": run_id,
        "script_type": "server",
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
        "reasoning_control_applied": "not_applicable_server",
        "reasoning_fallback_reason": None,
        "resume_skipped": False,
        "server_url": server_url,
        "server_model_id": backbone_ref.model_id,
        "judge_provider": judge_provider,
        "judge_server_url": judge_server_url if judge_provider == "server" else None,
        **structured_record_fields(
            judge_mode=judge_structured_output,
            closed_form_mode=closed_form_structured_output,
            used=False,
            target="closed_form" if structured_output_schema in {MC_ANSWER_SCHEMA_ID, YES_NO_ANSWER_SCHEMA_ID, OPEN_ANSWER_SCHEMA_ID} else "none",
            schema_id=structured_output_schema,
            fallback_reason=structured_output_fallback_reason,
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


def generate_vision(
    *,
    client: Any,
    model: str,
    prompt: str,
    image_paths: Sequence[Path | LabeledImage],
    temperature: float,
    max_output_tokens: int,
    max_retries: int,
    retry_base_seconds: float,
    retry_max_seconds: float,
    stop_on_quota: bool,
    structured_output_mode: str = "off",
    structured_output_schema: Optional[str] = None,
    structured_output_fallback: str = "retry_plain",
) -> str:
    generate_vision.last_structured_output_used = False  # type: ignore[attr-defined]
    generate_vision.last_structured_output_fallback_reason = None  # type: ignore[attr-defined]
    content: List[Dict[str, Any]] = []
    for item in image_paths:
        label = image_item_label(item)
        if label:
            content.append({"type": "text", "text": label})
        path = image_item_path(item)
        content.append({"type": "image_url", "image_url": {"url": path_to_data_url(path)}})
    content.append({"type": "text", "text": prompt})
    request_kwargs = {
        "model": model,
        "messages": [{"role": "user", "content": content}],
        "temperature": temperature,
        "max_tokens": max_output_tokens,
    }
    if structured_output_mode != "off" and structured_output_schema:
        request_kwargs["response_format"] = openai_chat_response_format(structured_output_schema)
    response, retry_count = call_with_retries(
        lambda: call_chat_with_structured_fallback(
            client=client,
            request_kwargs=request_kwargs,
            mode=structured_output_mode,
            schema_id=structured_output_schema,
            fallback=structured_output_fallback,
            state_owner=generate_vision,
        ),
        max_retries=max_retries,
        retry_base_seconds=retry_base_seconds,
        retry_max_seconds=retry_max_seconds,
        stop_on_quota=stop_on_quota,
    )
    generate_vision.last_retry_count = retry_count  # type: ignore[attr-defined]
    return extract_chat_text(response)


def generate_text(
    *,
    client: Any,
    model: str,
    prompt: str,
    temperature: float,
    max_output_tokens: int,
    max_retries: int,
    retry_base_seconds: float,
    retry_max_seconds: float,
    stop_on_quota: bool,
    structured_output_mode: str = "off",
    structured_output_schema: Optional[str] = None,
    structured_output_fallback: str = "retry_plain",
) -> str:
    generate_text.last_structured_output_used = False  # type: ignore[attr-defined]
    generate_text.last_structured_output_fallback_reason = None  # type: ignore[attr-defined]
    request_kwargs = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "max_tokens": max_output_tokens,
    }
    if structured_output_mode != "off" and structured_output_schema:
        request_kwargs["response_format"] = openai_chat_response_format(structured_output_schema)
    response, retry_count = call_with_retries(
        lambda: call_chat_with_structured_fallback(
            client=client,
            request_kwargs=request_kwargs,
            mode=structured_output_mode,
            schema_id=structured_output_schema,
            fallback=structured_output_fallback,
            state_owner=generate_text,
        ),
        max_retries=max_retries,
        retry_base_seconds=retry_base_seconds,
        retry_max_seconds=retry_max_seconds,
        stop_on_quota=stop_on_quota,
    )
    generate_text.last_retry_count = retry_count  # type: ignore[attr-defined]
    return extract_chat_text(response)


def call_chat_with_structured_fallback(
    *,
    client: Any,
    request_kwargs: Dict[str, Any],
    mode: str,
    schema_id: Optional[str],
    fallback: str,
    state_owner: Any,
) -> Any:
    if mode == "off" or not schema_id:
        return client.chat.completions.create(**strip_chat_structured_output(request_kwargs))
    try:
        response = client.chat.completions.create(**request_kwargs)
        state_owner.last_structured_output_used = True
        return response
    except Exception as exc:
        fallback_mode = normalize_structured_fallback(mode, fallback)
        if fallback_mode == "retry_plain" and looks_like_structured_output_unsupported(exc):
            state_owner.last_structured_output_used = False
            state_owner.last_structured_output_fallback_reason = format_structured_fallback_reason(exc)
            return client.chat.completions.create(**strip_chat_structured_output(request_kwargs))
        raise


def strip_chat_structured_output(request_kwargs: Dict[str, Any]) -> Dict[str, Any]:
    updated = dict(request_kwargs)
    updated.pop("response_format", None)
    return updated


def extract_chat_text(response: Any) -> str:
    content = response.choices[0].message.content
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        texts = [item.get("text", "") for item in content if isinstance(item, dict) and item.get("type") == "text"]
        return "\n".join(texts).strip()
    return str(content).strip()


def detect_mime_type(path: Path) -> str:
    return detect_content_mime_type(path)


def path_to_data_url(path: Path) -> str:
    encoded = base64.b64encode(path.read_bytes()).decode("utf-8")
    return f"data:{detect_mime_type(path)};base64,{encoded}"


if __name__ == "__main__":
    sys.exit(main())
