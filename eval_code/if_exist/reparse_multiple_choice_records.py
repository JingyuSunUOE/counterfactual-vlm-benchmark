#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from eval_common import (  # noqa: E402
    append_jsonl,
    build_summary,
    elapsed_ms,
    load_json,
    load_jsonl,
    parse_multiple_choice,
    print_readable_summary,
    record_is_error,
    record_task_signature,
    score_closed_form,
    write_checkpoint,
    write_json,
)


PARSER_VERSION = "strict_multiple_choice_v3"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reparse saved if_exist multiple-choice records with the strict MC parser."
    )
    parser.add_argument("--input-run", type=Path, required=True, help="Existing run directory with records.jsonl.")
    parser.add_argument("--output-run", type=Path, required=True, help="New run directory for reparsed outputs.")
    parser.add_argument(
        "--input-mode",
        choices=("all", "cf_only", "both", "orig_only"),
        default="all",
        help="Restrict reparsing to multiple-choice records from one input mode.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print planned changes without writing files.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite known output files if output run exists.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_run = args.input_run.expanduser().resolve()
    output_run = args.output_run.expanduser().resolve()
    validate_input_run(input_run)

    input_records = load_jsonl(input_run / "records.jsonl")
    run_config = load_json(input_run / "run_config.json")
    target_indexes = [
        index
        for index, record in enumerate(input_records)
        if should_reparse_record(record, input_mode=args.input_mode)
    ]
    validate_records(input_records)
    preview = preview_reparse(input_records, target_indexes)
    print(
        f"Loaded {len(input_records)} records from {input_run}. "
        f"Multiple-choice records to reparse: {len(target_indexes)}. input_mode={args.input_mode}.",
        flush=True,
    )
    print(f"Old labels: {dict(preview['old_labels'])}", flush=True)
    print(f"New labels: {dict(preview['new_labels'])}", flush=True)
    print(f"Changed labels: {preview['changed_labels']}", flush=True)
    print(f"New unparsed/other due to strict parser: {preview['new_unparsed']}", flush=True)
    if args.dry_run:
        print("Dry run complete. No files were written.")
        return 0

    prepare_output_run(output_run, overwrite=args.overwrite)
    reset_output_files(output_run)
    started = time.perf_counter()
    run_id = str(input_records[0].get("run_id") or input_run.name)
    eval_mode = str(input_records[0].get("input_mode") or input_records[0].get("eval_mode") or "mixed")
    records_path = output_run / "records.jsonl"

    write_json(output_run / "run_config.json", reparsed_run_config(run_config, input_run, input_mode=args.input_mode))
    output_records: List[Dict[str, Any]] = []
    for index, record in enumerate(input_records, start=1):
        new_record = copy.deepcopy(record)
        if should_reparse_record(record, input_mode=args.input_mode):
            new_record = reparse_record(new_record)
        output_records.append(new_record)
        append_jsonl(records_path, new_record)
        if len(output_records) % 100 == 0 or index == len(input_records):
            print(f"Reparsed/copied {len(output_records)}/{len(input_records)} records.", flush=True)

    summary = build_summary(
        records=output_records,
        run_id=run_id,
        script_type=str(input_records[0].get("script_type") or "closed"),
        eval_mode=eval_mode,
        backbone_ref=SimpleNamespace(
            alias=input_records[0].get("backbone_model_alias", "unknown"),
            model_id=input_records[0].get("backbone_model_id", "unknown"),
        ),
        judge_ref=SimpleNamespace(
            alias=input_records[0].get("judge_model_alias", "unknown"),
            model_id=input_records[0].get("judge_model_id", "unknown"),
        ),
        run_status="completed",
        is_partial=False,
    )
    summary["reparsed_from"] = str(input_run)
    summary["reparse_question_types"] = ["multiple_choice"]
    summary["multiple_choice_parser"] = PARSER_VERSION
    summary["reparse_elapsed_ms"] = elapsed_ms(started)
    write_json(output_run / "summary.json", summary)
    write_checkpoint(
        run_dir=output_run,
        run_id=run_id,
        run_status="completed",
        total_tasks=len(input_records),
        records=output_records,
        skipped_resume_count=0,
        pending_count=0,
    )
    print_readable_summary(summary, output_run)
    return 0


def validate_input_run(input_run: Path) -> None:
    missing = [name for name in ("records.jsonl", "run_config.json") if not (input_run / name).exists()]
    if missing:
        raise SystemExit(f"Input run is missing required files {missing}: {input_run}")


def prepare_output_run(output_run: Path, *, overwrite: bool) -> None:
    if output_run.exists() and any(output_run.iterdir()) and not overwrite:
        raise SystemExit(f"Output run exists. Use --overwrite to replace known output files: {output_run}")
    output_run.mkdir(parents=True, exist_ok=True)


def reset_output_files(output_run: Path) -> None:
    for name in ("records.jsonl", "run_config.json", "summary.json", "checkpoint.json"):
        path = output_run / name
        if path.exists():
            path.unlink()


def should_reparse_record(record: Dict[str, Any], *, input_mode: str) -> bool:
    if record.get("question_type") != "multiple_choice" or record_is_error(record):
        return False
    if input_mode == "all":
        return True
    mode = str(record.get("input_mode") or record.get("eval_mode") or "")
    return mode == input_mode


def validate_records(records: List[Dict[str, Any]]) -> None:
    if not records:
        raise SystemExit("Input run records.jsonl is empty.")
    signatures = [record_task_signature(record) for record in records]
    concrete = [signature for signature in signatures if signature]
    if len(concrete) != len(set(concrete)):
        raise SystemExit("Input records contain duplicate task signatures.")


def preview_reparse(records: List[Dict[str, Any]], target_indexes: List[int]) -> Dict[str, Any]:
    old_labels: Counter[str] = Counter()
    new_labels: Counter[str] = Counter()
    changed_labels = 0
    new_unparsed = 0
    targets = set(target_indexes)
    for index, record in enumerate(records):
        if index not in targets:
            continue
        old_label = str(record.get("scored_label") or "other")
        new_record = reparse_record(copy.deepcopy(record))
        new_label = str(new_record.get("scored_label") or "other")
        old_labels[old_label] += 1
        new_labels[new_label] += 1
        if old_label != new_label:
            changed_labels += 1
        if new_record.get("parsed_response") is None:
            new_unparsed += 1
    return {
        "old_labels": old_labels,
        "new_labels": new_labels,
        "changed_labels": changed_labels,
        "new_unparsed": new_unparsed,
    }


def reparse_record(record: Dict[str, Any]) -> Dict[str, Any]:
    old_parsed = record.get("parsed_response")
    old_label = record.get("scored_label")
    parsed = parse_multiple_choice(str(record.get("raw_response_text") or ""))
    answers = answers_from_record(record)
    label = score_closed_form("multiple_choice", parsed, answers)
    record["old_parsed_response"] = old_parsed
    record["old_scored_label"] = old_label
    record["parsed_response"] = parsed
    record["scored_label"] = label
    record["judge_model"] = "letter_match"
    record["judge_label"] = None
    record["judge_reason"] = None
    record["judge_raw_output"] = None
    record["multiple_choice_parser"] = PARSER_VERSION
    record["reparsed_multiple_choice"] = True
    record["reparsed_at"] = datetime.now(timezone.utc).isoformat()
    return record


def answers_from_record(record: Dict[str, Any]) -> Dict[str, Any]:
    correct = record.get("ground_truth_target") or record.get("gt") or {}
    biased = record.get("biased_target") or record.get("bias") or {}
    correct_option_id = correct.get("option_id") or correct.get("answer")
    biased_option_id = biased.get("option_id") or biased.get("answer")
    if not correct_option_id or not biased_option_id:
        raise RuntimeError(f"Multiple-choice record is missing option targets: {record.get('question_id')}")
    return {
        "correct_option_id": str(correct_option_id).strip().upper(),
        "biased_option_id": str(biased_option_id).strip().upper(),
    }


def reparsed_run_config(run_config: Dict[str, Any], input_run: Path, *, input_mode: str) -> Dict[str, Any]:
    payload = copy.deepcopy(run_config)
    payload["reparsed_from"] = str(input_run)
    payload["reparse_question_types"] = ["multiple_choice"]
    payload["reparse_input_mode"] = input_mode
    payload["multiple_choice_parser"] = PARSER_VERSION
    payload["reparse_timestamp"] = datetime.now(timezone.utc).isoformat()
    return payload


if __name__ == "__main__":
    raise SystemExit(main())
