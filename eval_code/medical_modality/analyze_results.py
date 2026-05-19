#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from eval_common import (
    DEFAULT_QUESTIONS_PATH,
    DEFAULT_RAW_RUNS_ROOT,
    DEFAULT_TABLES_ROOT,
    aggregate_rows,
    discover_run_dirs,
    filter_records_to_current_dataset,
    load_records_from_run_dirs,
    print_readable_summary,
    summarize_records,
    write_csv,
)


TABLE_SPECS = {
    "full_matrix.csv": ("backbone_model_alias", "input_mode", "question_type", "dataset", "visual_version", "tool_condition"),
    "by_model.csv": ("backbone_model_alias", "script_type"),
    "by_input_mode.csv": ("input_mode", "backbone_model_alias"),
    "by_question_type.csv": ("question_type", "backbone_model_alias"),
    "by_dataset.csv": ("dataset", "backbone_model_alias"),
    "by_visual_version.csv": ("visual_version", "backbone_model_alias"),
    "by_swap_direction.csv": ("swap_direction", "backbone_model_alias"),
    "by_background_modality.csv": ("background_modality", "backbone_model_alias"),
    "by_tumor_region_modality.csv": ("tumor_region_modality", "backbone_model_alias"),
    "by_tool_condition.csv": ("tool_condition", "backbone_model_alias"),
    "by_tool_condition_question_type.csv": ("tool_condition", "question_type", "backbone_model_alias"),
    "by_tool_condition_input_mode.csv": ("tool_condition", "input_mode", "backbone_model_alias"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate medical modality run outputs into CSV tables.")
    parser.add_argument("--input-root", type=Path, nargs="+", default=[DEFAULT_RAW_RUNS_ROOT])
    parser.add_argument("--output-root", type=Path, default=DEFAULT_TABLES_ROOT)
    parser.add_argument("--questions", type=Path, default=DEFAULT_QUESTIONS_PATH)
    parser.add_argument("--include-stale-records", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run_dirs = discover_run_dirs(args.input_root)
    records = load_records_from_run_dirs(run_dirs)
    if not records:
        raise SystemExit("No records found in selected runs.")
    records, filter_stats = filter_records_to_current_dataset(
        records,
        questions_path=args.questions,
        include_stale_records=args.include_stale_records,
    )
    print(
        "Dataset filter: "
        f"kept={filter_stats['kept_count']} "
        f"dropped_stale={filter_stats['dropped_stale_count']} "
        f"unchecked={filter_stats['unchecked_count']} "
        f"current_pairs={filter_stats['current_pair_count']}"
    )
    if not records:
        raise SystemExit("No records remain after filtering stale image pairs.")
    print(f"Loaded {len(records)} records from {len(run_dirs)} run directory/directories.")
    print_readable_summary(summarize_records(records))

    for filename, fields in TABLE_SPECS.items():
        rows = aggregate_rows(records, fields)
        path = args.output_root / filename
        print(f"{filename}: {len(rows)} rows")
        if not args.dry_run:
            write_csv(path, rows)
            print(f"  wrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
