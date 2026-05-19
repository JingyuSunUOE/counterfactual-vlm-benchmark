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
    load_benchmark,
    load_records_from_run_dirs,
    print_readable_summary,
    summarize_records,
    write_csv,
)


TABLE_SPECS = {
    "full_matrix.csv": (
        "backbone_model_alias",
        "tool_condition",
        "both_raw_prompt_policy",
        "both_raw_scoring_policy",
        "evidence_qc_filter",
        "missing_evidence_policy",
        "input_mode",
        "eval_mode",
        "question_type",
    ),
    "by_model.csv": ("backbone_model_alias", "script_type"),
    "by_input_mode.csv": ("input_mode", "backbone_model_alias"),
    "by_eval_mode.csv": ("eval_mode", "backbone_model_alias"),
    "by_tool_condition.csv": ("tool_condition", "backbone_model_alias"),
    "by_tool_condition_question_type.csv": ("tool_condition", "question_type", "backbone_model_alias"),
    "by_tool_condition_input_mode.csv": ("tool_condition", "input_mode", "backbone_model_alias"),
    "by_question_type.csv": ("question_type", "backbone_model_alias"),
    "by_question_type_input_mode.csv": ("question_type", "input_mode", "backbone_model_alias"),
    "by_category.csv": ("category", "backbone_model_alias"),
    "by_subcategory.csv": ("category", "subcategory", "backbone_model_alias"),
    "by_edit_type.csv": ("edit_type", "backbone_model_alias"),
    "by_source_variant.csv": ("source_variant", "backbone_model_alias"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate if_exist run outputs into CSV tables.")
    parser.add_argument("--input-root", type=Path, default=DEFAULT_RAW_RUNS_ROOT, help="Run dir or root containing runs.")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_TABLES_ROOT, help="Directory for CSV tables.")
    parser.add_argument("--questions", type=Path, default=DEFAULT_QUESTIONS_PATH, help="Benchmark JSON path.")
    parser.add_argument(
        "--include-stale-records",
        action="store_true",
        help="Include records whose image pair is no longer present in the current dataset.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print planned table sizes without writing CSV files.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run_dirs = discover_run_dirs([args.input_root])
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
    records = enrich_records_with_edit_type(records, args.questions)

    print(f"Loaded {len(records)} records from {len(run_dirs)} run directory/directories.")
    print_readable_summary(summarize_records(records))

    for filename, fields in TABLE_SPECS.items():
        rows = aggregate_rows(records, fields)
        rows = add_raw_delta_columns(rows) if "tool_condition" in fields else rows
        path = args.output_root / filename
        print(f"{filename}: {len(rows)} rows")
        if not args.dry_run:
            write_csv(path, rows)
            print(f"  wrote {path}")
    return 0


def add_raw_delta_columns(rows: list[dict]) -> list[dict]:
    if not rows or "Tool_Condition" not in rows[0]:
        return rows
    metric_columns = ("Accuracy%", "Bias%", "Other%")
    delta_columns = ("Delta_Accuracy_vs_Raw", "Delta_Bias_vs_Raw", "Delta_Other_vs_Raw")
    baseline_by_key: dict[tuple[tuple[str, str], ...], dict] = {}
    for row in rows:
        key = comparable_row_key(row)
        if row.get("Tool_Condition") == "raw":
            baseline_by_key[key] = row
    updated_rows: list[dict] = []
    for row in rows:
        updated = dict(row)
        baseline = baseline_by_key.get(comparable_row_key(row))
        for metric, delta in zip(metric_columns, delta_columns):
            if baseline is None:
                updated[delta] = ""
                continue
            updated[delta] = round(float(row.get(metric, 0.0)) - float(baseline.get(metric, 0.0)), 4)
        updated_rows.append(updated)
    return updated_rows


def comparable_row_key(row: dict) -> tuple[tuple[str, str], ...]:
    ignored = {
        "Tool_Condition",
        "N",
        "Errors",
        "Correct",
        "Biased",
        "Other",
        "Accuracy%",
        "Bias%",
        "Other%",
        "Delta_Accuracy_vs_Raw",
        "Delta_Bias_vs_Raw",
        "Delta_Other_vs_Raw",
    }
    return tuple(sorted((str(key), str(value)) for key, value in row.items() if key not in ignored))


def enrich_records_with_edit_type(records: list[dict], questions_path: Path) -> list[dict]:
    edit_type_map = load_edit_type_map(questions_path)
    enriched = []
    for record in records:
        if record.get("edit_type"):
            enriched.append(record)
            continue
        category = record.get("category") or record.get("domain")
        subcategory = record.get("subcategory") or record.get("key")
        updated = dict(record)
        updated["edit_type"] = edit_type_map.get((category, subcategory), "")
        enriched.append(updated)
    return enriched


def load_edit_type_map(questions_path: Path) -> dict[tuple[str, str], str]:
    benchmark = load_benchmark(questions_path)
    mapping: dict[tuple[str, str], str] = {}
    for category_entry in benchmark["categories"]:
        category = category_entry["category"]
        for subcategory_entry in category_entry["subcategories"]:
            mapping[(category, subcategory_entry["subcategory"])] = subcategory_entry.get("edit_type", "")
    return mapping


if __name__ == "__main__":
    sys.exit(main())
