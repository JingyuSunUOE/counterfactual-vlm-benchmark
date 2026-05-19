#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List

from dotenv import load_dotenv

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from eval_common import (
    DEFAULT_QUESTIONS_PATH,
    append_jsonl,
    build_group_specs,
    build_judge_prompt,
    build_summary,
    elapsed_ms,
    error_metadata,
    load_annotations,
    load_benchmark,
    load_json,
    load_jsonl,
    parse_judge_response,
    print_readable_summary,
    record_is_error,
    record_task_signature,
    write_checkpoint,
    write_json,
)
from eval_closed_vlm import (
    ClosedProviderAdapter,
    resolve_closed_model_ref,
    validate_provider_envs,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rejudge saved counting open-question records without rerunning the VLM backbone."
    )
    parser.add_argument("--input-run", type=Path, required=True, help="Existing run directory with records.jsonl.")
    parser.add_argument("--output-run", type=Path, required=True, help="New run directory for rejudged outputs.")
    parser.add_argument("--questions", type=Path, default=DEFAULT_QUESTIONS_PATH, help="Counting question JSON path.")
    parser.add_argument("--judge-model", default="gpt-4o-mini", help="Closed judge alias or provider-prefixed model id.")
    parser.add_argument("--question-types", default="open", choices=("open",), help="Only open is supported in v1.")
    parser.add_argument(
        "--input-mode",
        choices=("all", "cf_only", "both", "orig_only"),
        default="all",
        help="Restrict rejudging to open records from one input mode.",
    )
    parser.add_argument("--max-output-tokens", type=int, default=512, help="Maximum judge output tokens.")
    parser.add_argument("--gemini-thinking-budget", type=int, default=0, help="Gemini judge thinking budget if used.")
    parser.add_argument("--max-retries", type=int, default=3, help="Maximum judge API attempts per record.")
    parser.add_argument("--retry-base-seconds", type=float, default=2.0, help="Base retry backoff in seconds.")
    parser.add_argument("--retry-max-seconds", type=float, default=60.0, help="Maximum retry backoff in seconds.")
    parser.add_argument(
        "--request-timeout-seconds",
        type=float,
        default=60.0,
        help="Per-request timeout for SDK clients that support client.with_options(timeout=...).",
    )
    parser.add_argument(
        "--stop-on-quota",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Stop immediately on quota/billing/authentication errors.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print planned rejudge count without API calls or writes.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite known output files if output run exists.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    load_dotenv(override=True)
    input_run = args.input_run.resolve()
    output_run = args.output_run.resolve()
    validate_input_run(input_run)
    prepare_output_run(output_run, overwrite=args.overwrite, dry_run=args.dry_run)

    input_records = load_jsonl(input_run / "records.jsonl")
    run_config = load_json(input_run / "run_config.json")
    question_answers = load_open_answers(args.questions.resolve())
    rejudge_indexes = [
        index
        for index, record in enumerate(input_records)
        if should_rejudge_record(record, input_mode=args.input_mode)
    ]
    validate_records(input_records, rejudge_indexes)
    print(
        f"Loaded {len(input_records)} records from {input_run}. "
        f"Open records to rejudge: {len(rejudge_indexes)}. input_mode={args.input_mode}.",
        flush=True,
    )
    if args.dry_run:
        print("Dry run complete. No files were written and no judge calls were made.")
        return 0

    judge_ref = resolve_closed_model_ref(args.judge_model)
    validate_provider_envs([judge_ref.provider])
    judge_adapter = ClosedProviderAdapter(judge_ref)
    apply_client_timeout(judge_adapter, args.request_timeout_seconds)
    run_id = str(input_records[0].get("run_id") or input_run.name)
    eval_mode = str(input_records[0].get("input_mode") or input_records[0].get("eval_mode") or "mixed")
    records_path = output_run / "records.jsonl"
    reset_output_files(output_run)
    write_json(
        output_run / "run_config.json",
        rejudged_run_config(run_config, input_run, judge_ref, input_mode=args.input_mode),
    )

    output_records: List[Dict[str, Any]] = []
    run_status = "completed"
    last_error_kind = None
    last_error_message = None
    last_task_signature = None
    started_all = time.perf_counter()

    for index, record in enumerate(input_records, start=1):
        new_record = copy.deepcopy(record)
        if should_rejudge_record(record, input_mode=args.input_mode):
            last_task_signature = record_task_signature(record)
            try:
                new_record = rejudge_record(
                    record=new_record,
                    question_answers=question_answers,
                    judge_adapter=judge_adapter,
                    judge_ref=judge_ref,
                    max_output_tokens=args.max_output_tokens,
                    gemini_thinking_budget=args.gemini_thinking_budget,
                    max_retries=args.max_retries,
                    retry_base_seconds=args.retry_base_seconds,
                    retry_max_seconds=args.retry_max_seconds,
                    stop_on_quota=args.stop_on_quota,
                )
            except Exception as exc:
                metadata = error_metadata(exc, default="judge_error", stop_on_quota=args.stop_on_quota)
                last_error_kind = metadata["error_kind"] if metadata["error_kind"] != "unknown_error" else "judge_error"
                last_error_message = str(exc)
                new_record["scored_label"] = "error"
                new_record["judge_label"] = None
                new_record["judge_reason"] = None
                new_record["judge_raw_output"] = None
                new_record["error_message"] = f"Rejudge failed: {exc}"
                new_record["error_kind"] = last_error_kind
                new_record["fatal_error"] = metadata["fatal_error"]
                new_record["rejudged_open"] = False
                if metadata["fatal_error"]:
                    run_status = f"interrupted_{last_error_kind}"
            output_records.append(new_record)
            append_jsonl(records_path, new_record)
            if len(output_records) % 25 == 0 or index == len(input_records):
                print(f"Rejudged/copied {len(output_records)}/{len(input_records)} records.", flush=True)
            if run_status != "completed":
                break
            continue

        output_records.append(new_record)
        append_jsonl(records_path, new_record)

    if len(output_records) != len(input_records) and run_status == "completed":
        run_status = "interrupted_incomplete_output"
    summary = build_summary(
        records=output_records,
        run_id=run_id,
        script_type=str(input_records[0].get("script_type") or "closed"),
        eval_mode=eval_mode,
        backbone_ref=SimpleNamespace(
            alias=input_records[0].get("backbone_model_alias", "unknown"),
            model_id=input_records[0].get("backbone_model_id", "unknown"),
        ),
        judge_ref=judge_ref,
        run_status=run_status,
        is_partial=run_status != "completed",
    )
    summary["rejudged_from"] = str(input_run)
    summary["rejudge_elapsed_ms"] = elapsed_ms(started_all)
    summary["last_error_kind"] = last_error_kind
    write_json(output_run / "summary.json", summary)
    write_checkpoint(
        run_dir=output_run,
        run_id=run_id,
        run_status=run_status,
        total_tasks=len(input_records),
        records=output_records,
        skipped_resume_count=0,
        pending_count=max(0, len(input_records) - len(output_records)),
        last_task_signature=last_task_signature,
        last_error_kind=last_error_kind,
        last_error_message=last_error_message,
    )
    print_readable_summary(summary, output_run)
    return 0 if run_status == "completed" else 1


def validate_input_run(input_run: Path) -> None:
    missing = [name for name in ("records.jsonl", "run_config.json") if not (input_run / name).exists()]
    if missing:
        raise SystemExit(f"Input run is missing required files {missing}: {input_run}")


def prepare_output_run(output_run: Path, *, overwrite: bool, dry_run: bool) -> None:
    if dry_run:
        return
    if output_run.exists() and any(output_run.iterdir()) and not overwrite:
        raise SystemExit(f"Output run exists. Use --overwrite to replace known output files: {output_run}")
    output_run.mkdir(parents=True, exist_ok=True)


def reset_output_files(output_run: Path) -> None:
    for name in ("records.jsonl", "run_config.json", "summary.json", "checkpoint.json"):
        path = output_run / name
        if path.exists():
            path.unlink()


def apply_client_timeout(judge_adapter: ClosedProviderAdapter, timeout_seconds: float) -> None:
    client = getattr(judge_adapter, "client", None)
    with_options = getattr(client, "with_options", None)
    if callable(with_options):
        judge_adapter.client = with_options(timeout=timeout_seconds)


def load_open_answers(questions_path: Path) -> Dict[tuple[str, str], Dict[str, Any]]:
    benchmark = load_benchmark(questions_path)
    # The annotation loader validates image paths and keeps question parsing aligned with normal runs.
    annotations = load_annotations(
        Path(__file__).resolve().parents[2] / "eval_results" / "counting" / "metadata" / "counting_count_annotations.json"
    )
    groups = build_group_specs(benchmark, annotations=annotations)
    answers: Dict[tuple[str, str], Dict[str, Any]] = {}
    for group in groups:
        for question in group.questions:
            if question.question_type != "open":
                continue
            for mode, payload in question.answers_by_mode.items():
                answers[(question.question_id, mode)] = payload
    return answers


def should_rejudge_record(record: Dict[str, Any], *, input_mode: str) -> bool:
    if record.get("question_type") != "open" or record_is_error(record):
        return False
    if input_mode == "all":
        return True
    mode = str(record.get("input_mode") or record.get("eval_mode") or "")
    return mode == input_mode


def validate_records(records: List[Dict[str, Any]], rejudge_indexes: List[int]) -> None:
    if not records:
        raise SystemExit("Input run records.jsonl is empty.")
    signatures = [record_task_signature(record) for record in records]
    concrete = [signature for signature in signatures if signature]
    if len(concrete) != len(set(concrete)):
        raise SystemExit("Input records contain duplicate task signatures.")
    if len(records) == 768 and len(rejudge_indexes) != 256:
        raise SystemExit(f"Expected 256 open records for a full counting run, found {len(rejudge_indexes)}.")


def rejudge_record(
    *,
    record: Dict[str, Any],
    question_answers: Dict[tuple[str, str], Dict[str, Any]],
    judge_adapter: ClosedProviderAdapter,
    judge_ref: Any,
    max_output_tokens: int,
    gemini_thinking_budget: int,
    max_retries: int,
    retry_base_seconds: float,
    retry_max_seconds: float,
    stop_on_quota: bool,
) -> Dict[str, Any]:
    mode = str(record.get("input_mode") or record.get("eval_mode") or "")
    key = (str(record.get("question_id") or ""), mode)
    if key not in question_answers:
        raise RuntimeError(f"No current open rubric found for question/mode: {key}")
    candidate = str(record.get("raw_response_text") or "").strip()
    if not candidate:
        raise RuntimeError(f"Open record has empty raw_response_text: {key}")
    judge_prompt = build_judge_prompt(
        question_prompt=str(record.get("prompt") or ""),
        eval_mode=mode,
        candidate_answer=candidate,
        answers=question_answers[key],
    )
    judge_raw = judge_adapter.generate(
        judge_prompt,
        [],
        temperature=0.0,
        max_output_tokens=max_output_tokens,
        gemini_thinking_budget=gemini_thinking_budget,
        max_retries=max_retries,
        retry_base_seconds=retry_base_seconds,
        retry_max_seconds=retry_max_seconds,
        stop_on_quota=stop_on_quota,
        provider_cache="off",
        prompt_cache_retention="in_memory",
        cache_key=None,
        cache_usage_log=False,
    )
    judge_result = parse_judge_response(judge_raw)
    record["judge_model_alias"] = judge_ref.alias
    record["judge_model_id"] = judge_ref.model_id
    record["judge_model"] = judge_ref.model_id
    record["judge_raw_output"] = judge_raw
    record["judge_label"] = judge_result["label"]
    record["judge_reason"] = judge_result["reason"]
    record["scored_label"] = judge_result["label"]
    record["error_message"] = None
    record["error_kind"] = None
    record["fatal_error"] = False
    record["rejudged_open"] = True
    record["rejudge_retry_count"] = getattr(judge_adapter, "last_retry_count", 0)
    record["rejudged_at"] = datetime.now(timezone.utc).isoformat()
    return record


def rejudged_run_config(run_config: Dict[str, Any], input_run: Path, judge_ref: Any, *, input_mode: str) -> Dict[str, Any]:
    payload = copy.deepcopy(run_config)
    payload["rejudged_from"] = str(input_run)
    payload["rejudge_question_types"] = ["open"]
    payload["rejudge_input_mode"] = input_mode
    payload["rejudge_model_alias"] = judge_ref.alias
    payload["rejudge_model_id"] = judge_ref.model_id
    payload["rejudge_timestamp"] = datetime.now(timezone.utc).isoformat()
    return payload


if __name__ == "__main__":
    raise SystemExit(main())
