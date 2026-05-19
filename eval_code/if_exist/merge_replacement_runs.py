#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

from eval_common import (
    QUESTION_TYPES,
    completed_success_count,
    load_json,
    load_jsonl,
    record_is_error,
    record_task_signature,
    summarize_records,
    write_checkpoint,
    write_json,
)


CONFIG_MATCH_KEYS = (
    "script_type",
    "questions_path",
    "eval_mode",
    "input_mode",
    "backbone_model_alias",
    "backbone_model_id",
    "judge_model_alias",
    "judge_model_id",
    "source_variant",
    "prime",
    "temperature",
    "max_output_tokens",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a merged if_exist run by replacing selected question types from another run."
    )
    parser.add_argument("--base-run", type=Path, required=True, help="Run directory that provides unchanged records.")
    parser.add_argument(
        "--replacement-run",
        type=Path,
        required=True,
        help="Run directory that provides replacement records for selected question types.",
    )
    parser.add_argument(
        "--replace-question-types",
        required=True,
        help="Comma-separated question types to replace, for example multiple_choice.",
    )
    parser.add_argument("--output-run", type=Path, required=True, help="Output merged run directory.")
    parser.add_argument("--run-id", default=None, help="Run id to write into merged records. Defaults to output dir name.")
    parser.add_argument("--overwrite", action="store_true", help="Delete an existing output run before writing.")
    parser.add_argument(
        "--allow-count-mismatch",
        action="store_true",
        help="Allow missing or extra replacement records for the selected question types.",
    )
    parser.add_argument(
        "--allow-config-mismatch",
        action="store_true",
        help="Allow base and replacement run_config.json to differ on core evaluation fields.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print the merge plan without writing files.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    replace_types = parse_question_types(args.replace_question_types)
    base_run = args.base_run.expanduser().resolve()
    replacement_run = args.replacement_run.expanduser().resolve()
    output_run = args.output_run.expanduser().resolve()
    run_id = args.run_id or output_run.name

    validate_run_dir(base_run, "base")
    validate_run_dir(replacement_run, "replacement")
    if output_run.exists() and not args.overwrite and not args.dry_run:
        raise SystemExit(f"Output run already exists. Use --overwrite to replace it: {output_run}")

    base_config = load_json(base_run / "run_config.json")
    replacement_config = load_json(replacement_run / "run_config.json")
    config_mismatches = compare_run_configs(base_config, replacement_config)
    if config_mismatches and not args.allow_config_mismatch:
        formatted = "\n".join(f"- {key}: base={base!r} replacement={replacement!r}" for key, base, replacement in config_mismatches)
        raise SystemExit(f"Run configs are not compatible:\n{formatted}\nUse --allow-config-mismatch to override.")

    base_records = load_jsonl(base_run / "records.jsonl")
    replacement_records = load_jsonl(replacement_run / "records.jsonl")
    merged_records, stats = merge_records(
        base_records=base_records,
        replacement_records=replacement_records,
        replace_types=replace_types,
        output_run_id=run_id,
        base_run=base_run,
        replacement_run=replacement_run,
        allow_count_mismatch=args.allow_count_mismatch,
    )

    print_merge_plan(
        base_run=base_run,
        replacement_run=replacement_run,
        output_run=output_run,
        replace_types=replace_types,
        stats=stats,
        config_mismatches=config_mismatches,
    )
    if args.dry_run:
        print("dry_run=true; no files written")
        return 0

    if output_run.exists() and args.overwrite:
        shutil.rmtree(output_run)
    output_run.mkdir(parents=True, exist_ok=True)
    write_records(output_run / "records.jsonl", merged_records)
    write_json(output_run / "run_config.json", build_merged_run_config(base_config, replacement_config, run_id, stats))
    summary = summarize_records(merged_records)
    summary["run_status"] = "completed"
    summary["is_partial"] = False
    summary["merge_metadata"] = stats
    write_json(output_run / "summary.json", summary)
    write_checkpoint(
        run_dir=output_run,
        run_id=run_id,
        run_status="completed",
        total_tasks=len(merged_records),
        records=merged_records,
        skipped_resume_count=0,
        pending_count=0,
        last_task_signature=record_task_signature(merged_records[-1]) if merged_records else None,
    )
    print(f"Wrote merged run to {output_run}")
    return 0


def parse_question_types(raw_value: str) -> List[str]:
    values = [value.strip() for value in raw_value.split(",") if value.strip()]
    if not values:
        raise SystemExit("--replace-question-types must not be empty.")
    invalid = [value for value in values if value not in QUESTION_TYPES]
    if invalid:
        raise SystemExit(f"Unsupported question types: {invalid}. Allowed: {QUESTION_TYPES}")
    return values


def validate_run_dir(path: Path, label: str) -> None:
    missing = [name for name in ("records.jsonl", "run_config.json") if not (path / name).exists()]
    if missing:
        raise SystemExit(f"{label} run is missing required files {missing}: {path}")


def compare_run_configs(base_config: Dict[str, Any], replacement_config: Dict[str, Any]) -> List[tuple[str, Any, Any]]:
    mismatches: List[tuple[str, Any, Any]] = []
    for key in CONFIG_MATCH_KEYS:
        base_value = base_config.get(key)
        replacement_value = replacement_config.get(key)
        if base_value != replacement_value:
            if key == "question_types":
                continue
            mismatches.append((key, base_value, replacement_value))
    return mismatches


def merge_records(
    *,
    base_records: Sequence[Dict[str, Any]],
    replacement_records: Sequence[Dict[str, Any]],
    replace_types: Sequence[str],
    output_run_id: str,
    base_run: Path,
    replacement_run: Path,
    allow_count_mismatch: bool,
) -> tuple[List[Dict[str, Any]], Dict[str, Any]]:
    replace_type_set = set(replace_types)
    replacement_by_signature: Dict[str, Dict[str, Any]] = {}
    ignored_replacement_count = 0
    for record in replacement_records:
        if record.get("question_type") not in replace_type_set:
            ignored_replacement_count += 1
            continue
        signature = require_signature(record, "replacement")
        if signature in replacement_by_signature:
            raise SystemExit(f"Duplicate replacement task signature: {signature}")
        replacement_by_signature[signature] = record

    merged: List[Dict[str, Any]] = []
    used_replacement_signatures: set[str] = set()
    removed_base_count = 0
    missing_replacement_signatures: List[str] = []

    for record in base_records:
        if record.get("question_type") not in replace_type_set:
            merged.append(normalize_record(record, output_run_id, "base", base_run))
            continue
        removed_base_count += 1
        signature = require_signature(record, "base")
        replacement = replacement_by_signature.get(signature)
        if replacement is None:
            missing_replacement_signatures.append(signature)
            if allow_count_mismatch:
                continue
            continue
        merged.append(normalize_record(replacement, output_run_id, "replacement", replacement_run))
        used_replacement_signatures.add(signature)

    extra_replacement_signatures = sorted(set(replacement_by_signature) - used_replacement_signatures)
    if (missing_replacement_signatures or extra_replacement_signatures) and not allow_count_mismatch:
        raise SystemExit(
            "Replacement records do not exactly match removed base records:\n"
            f"- missing_replacements={len(missing_replacement_signatures)}\n"
            f"- extra_replacements={len(extra_replacement_signatures)}"
        )
    for signature in extra_replacement_signatures:
        merged.append(normalize_record(replacement_by_signature[signature], output_run_id, "replacement_extra", replacement_run))

    assert_no_duplicate_tasks(merged)
    stats = {
        "base_run": str(base_run),
        "replacement_run": str(replacement_run),
        "replace_question_types": list(replace_types),
        "base_record_count": len(base_records),
        "replacement_record_count": len(replacement_records),
        "removed_base_count": removed_base_count,
        "replacement_used_count": len(used_replacement_signatures),
        "replacement_extra_count": len(extra_replacement_signatures),
        "ignored_replacement_count": ignored_replacement_count,
        "merged_record_count": len(merged),
        "merged_success_count": completed_success_count(merged),
        "merged_error_count": sum(1 for record in merged if record_is_error(record)),
    }
    return merged, stats


def require_signature(record: Dict[str, Any], label: str) -> str:
    signature = record_task_signature(record)
    if not signature:
        raise SystemExit(f"{label} record is missing fields required for task signature: {record}")
    return signature


def normalize_record(record: Dict[str, Any], output_run_id: str, merge_origin: str, source_run: Path) -> Dict[str, Any]:
    updated = dict(record)
    updated["source_run_id"] = record.get("run_id")
    updated["source_run_dir"] = str(source_run)
    updated["run_id"] = output_run_id
    updated["merge_origin"] = merge_origin
    updated["run_status"] = "completed"
    updated["is_partial"] = False
    return updated


def assert_no_duplicate_tasks(records: Sequence[Dict[str, Any]]) -> None:
    seen: set[str] = set()
    duplicates: List[str] = []
    for record in records:
        signature = require_signature(record, "merged")
        if signature in seen:
            duplicates.append(signature)
        seen.add(signature)
    if duplicates:
        raise SystemExit(f"Merged records contain duplicate task signatures: {duplicates[:5]}")


def build_merged_run_config(
    base_config: Dict[str, Any],
    replacement_config: Dict[str, Any],
    run_id: str,
    stats: Dict[str, Any],
) -> Dict[str, Any]:
    merged_config = dict(base_config)
    merged_config["run_id"] = run_id
    merged_config["question_types"] = list(QUESTION_TYPES)
    merged_config["response_target_count"] = stats["merged_record_count"]
    merged_config["merge"] = {
        "strategy": "replace_question_types",
        "base_run_id": base_config.get("run_id"),
        "replacement_run_id": replacement_config.get("run_id"),
        **stats,
    }
    return merged_config


def write_records(path: Path, records: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def print_merge_plan(
    *,
    base_run: Path,
    replacement_run: Path,
    output_run: Path,
    replace_types: Sequence[str],
    stats: Dict[str, Any],
    config_mismatches: Sequence[tuple[str, Any, Any]],
) -> None:
    print(f"base_run={base_run}")
    print(f"replacement_run={replacement_run}")
    print(f"output_run={output_run}")
    print(f"replace_question_types={','.join(replace_types)}")
    if config_mismatches:
        print(f"config_mismatches={len(config_mismatches)}")
    for key in (
        "base_record_count",
        "replacement_record_count",
        "removed_base_count",
        "replacement_used_count",
        "replacement_extra_count",
        "ignored_replacement_count",
        "merged_record_count",
        "merged_success_count",
        "merged_error_count",
    ):
        print(f"{key}={stats[key]}")


if __name__ == "__main__":
    sys.exit(main())
