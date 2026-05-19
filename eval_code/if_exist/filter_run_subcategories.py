#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence

from eval_common import (
    load_json,
    load_jsonl,
    record_is_error,
    record_task_signature,
    summarize_records,
    write_checkpoint,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Filter subcategories out of an existing if_exist run.")
    parser.add_argument("--input-run", type=Path, required=True)
    parser.add_argument("--output-run", type=Path, required=True)
    parser.add_argument("--exclude-subcategories", required=True, help="Comma-separated subcategory names to remove.")
    parser.add_argument("--dataset-revision", default="if_exist_no_cars")
    parser.add_argument("--run-id", default=None, help="Defaults to the input run_id.")
    parser.add_argument("--expected-record-count", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_run = args.input_run.expanduser().resolve()
    output_run = args.output_run.expanduser().resolve()
    excluded = {item.strip() for item in args.exclude_subcategories.split(",") if item.strip()}
    if not excluded:
        raise SystemExit("--exclude-subcategories must not be empty.")
    validate_run(input_run)
    if output_run.exists() and not args.overwrite and not args.dry_run:
        raise SystemExit(f"Output run already exists. Use --overwrite: {output_run}")

    records = load_jsonl(input_run / "records.jsonl")
    kept = [
        normalize_record(record, excluded, dataset_revision=args.dataset_revision)
        for record in records
        if record.get("subcategory") not in excluded
    ]
    removed = len(records) - len(kept)
    if args.expected_record_count is not None and len(kept) != args.expected_record_count:
        raise SystemExit(f"Expected {args.expected_record_count} kept records, got {len(kept)}.")
    run_config = build_run_config(
        load_json(input_run / "run_config.json"),
        run_id=args.run_id,
        excluded=sorted(excluded),
        dataset_revision=args.dataset_revision,
        input_run=input_run,
        original_record_count=len(records),
        removed_record_count=removed,
        kept_record_count=len(kept),
    )
    run_id = run_config.get("run_id") or args.run_id or input_run.name
    kept = [dict(record, run_id=run_id) for record in kept]
    summary = summarize_records(kept)
    summary.update(
        {
            "run_id": run_id,
            "run_status": "completed",
            "is_partial": False,
            "excluded_subcategories": sorted(excluded),
            "dataset_revision": args.dataset_revision,
            "filter_metadata": {
                "source_run": str(input_run),
                "original_record_count": len(records),
                "removed_record_count": removed,
                "kept_record_count": len(kept),
            },
        }
    )
    print(
        json.dumps(
            {
                "input_run": str(input_run),
                "output_run": str(output_run),
                "excluded_subcategories": sorted(excluded),
                "original_record_count": len(records),
                "removed_record_count": removed,
                "kept_record_count": len(kept),
                "error_count": sum(1 for record in kept if record_is_error(record)),
            },
            indent=2,
        )
    )
    if args.dry_run:
        print("dry_run=true; no files written")
        return 0

    if output_run.exists() and args.overwrite:
        shutil.rmtree(output_run)
    output_run.mkdir(parents=True, exist_ok=True)
    write_records(output_run / "records.jsonl", kept)
    write_json(output_run / "run_config.json", run_config)
    write_json(output_run / "summary.json", summary)
    write_checkpoint(
        run_dir=output_run,
        run_id=run_id,
        run_status="completed",
        total_tasks=len(kept),
        records=kept,
        skipped_resume_count=0,
        pending_count=0,
        last_task_signature=record_task_signature(kept[-1]) if kept else None,
    )
    print(f"Wrote filtered run to {output_run}")
    return 0


def validate_run(path: Path) -> None:
    missing = [name for name in ("records.jsonl", "run_config.json") if not (path / name).exists()]
    if missing:
        raise SystemExit(f"Run is missing required files {missing}: {path}")


def normalize_record(record: Dict[str, Any], excluded: set[str], *, dataset_revision: str) -> Dict[str, Any]:
    updated = dict(record)
    updated["excluded_subcategories"] = sorted(excluded)
    updated["dataset_revision"] = dataset_revision
    return updated


def build_run_config(
    config: Dict[str, Any],
    *,
    run_id: str | None,
    excluded: Sequence[str],
    dataset_revision: str,
    input_run: Path,
    original_record_count: int,
    removed_record_count: int,
    kept_record_count: int,
) -> Dict[str, Any]:
    updated = dict(config)
    updated["run_id"] = run_id or config.get("run_id") or input_run.name
    updated["excluded_subcategories"] = list(excluded)
    updated["dataset_revision"] = dataset_revision
    updated["filter_metadata"] = {
        "source_run": str(input_run),
        "original_record_count": original_record_count,
        "removed_record_count": removed_record_count,
        "kept_record_count": kept_record_count,
    }
    return updated


def write_records(path: Path, records: Sequence[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    sys.exit(main())
