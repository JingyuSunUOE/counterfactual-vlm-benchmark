#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

from eval_common import (
    DEFAULT_QUESTIONS_PATH,
    DEFAULT_RAW_RUNS_ROOT,
    DEFAULT_TABLES_ROOT,
    discover_run_dirs,
    filter_records_to_current_dataset,
    load_records_from_run_dirs,
    record_field_value,
    summarize_bucket,
    write_csv,
)


DEFAULT_OUTPUT_PATH = DEFAULT_TABLES_ROOT / "by_image_difficulty.csv"
ITEM_KEY_FIELDS = ("category", "subcategory", "source_variant", "original_image_path", "edited_image_path")
QUESTION_TYPES = ("yes_no", "multiple_choice", "open")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate if_exist records by image pair to identify easy or low-quality samples."
    )
    parser.add_argument(
        "--input-root",
        type=Path,
        default=DEFAULT_RAW_RUNS_ROOT,
        help="Run directory or root containing run directories.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="Output CSV path for per-image difficulty rows.",
    )
    parser.add_argument("--questions", type=Path, default=DEFAULT_QUESTIONS_PATH, help="Benchmark JSON path.")
    parser.add_argument(
        "--include-stale-records",
        action="store_true",
        help="Include records whose image pair is no longer present in the current dataset.",
    )
    parser.add_argument(
        "--min-cf-responses",
        type=int,
        default=3,
        help="Minimum successful cf_only responses required before assigning remove/review recommendations.",
    )
    parser.add_argument(
        "--easy-accuracy",
        type=float,
        default=0.80,
        help="cf_only accuracy threshold for consider_remove.",
    )
    parser.add_argument(
        "--easy-bias-rate",
        type=float,
        default=0.10,
        help="Maximum cf_only bias rate for consider_remove.",
    )
    parser.add_argument(
        "--hard-accuracy",
        type=float,
        default=0.20,
        help="cf_only accuracy threshold for review_quality.",
    )
    parser.add_argument("--top-k", type=int, default=20, help="Number of easiest and hardest samples to print.")
    parser.add_argument("--dry-run", action="store_true", help="Print summary and previews without writing CSV.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    validate_args(args)

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

    rows = build_item_difficulty_rows(
        records,
        min_cf_responses=args.min_cf_responses,
        easy_accuracy=args.easy_accuracy,
        easy_bias_rate=args.easy_bias_rate,
        hard_accuracy=args.hard_accuracy,
    )
    if not rows:
        raise SystemExit("No image-pair rows could be built from selected records.")

    print_summary(
        records=records,
        run_count=len(run_dirs),
        rows=rows,
        top_k=args.top_k,
    )

    if not args.dry_run:
        write_csv(args.output, rows)
        print(f"Wrote {len(rows)} image difficulty rows to {args.output}")
    else:
        print("Dry run: no CSV written.")
    return 0


def validate_args(args: argparse.Namespace) -> None:
    if args.min_cf_responses < 1:
        raise SystemExit("--min-cf-responses must be >= 1.")
    for name in ("easy_accuracy", "easy_bias_rate", "hard_accuracy"):
        value = getattr(args, name)
        if value < 0.0 or value > 1.0:
            raise SystemExit(f"--{name.replace('_', '-')} must be between 0 and 1.")
    if args.top_k < 0:
        raise SystemExit("--top-k must be >= 0.")


def build_item_difficulty_rows(
    records: Sequence[Dict[str, Any]],
    *,
    min_cf_responses: int,
    easy_accuracy: float,
    easy_bias_rate: float,
    hard_accuracy: float,
) -> List[Dict[str, Any]]:
    grouped: Dict[Tuple[str, str, str, str, str], List[Dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[item_key(record)].append(record)

    rows = [
        build_item_row(
            key=key,
            bucket=bucket,
            min_cf_responses=min_cf_responses,
            easy_accuracy=easy_accuracy,
            easy_bias_rate=easy_bias_rate,
            hard_accuracy=hard_accuracy,
        )
        for key, bucket in grouped.items()
    ]
    return sorted(
        rows,
        key=lambda row: (
            row["Category"],
            row["Subcategory"],
            row["Source_Variant"],
            row["Original_Image"],
            row["Edited_Image"],
        ),
    )


def item_key(record: Dict[str, Any]) -> Tuple[str, str, str, str, str]:
    category = str(record_field_value(record, "category") or record_field_value(record, "domain"))
    subcategory = str(record_field_value(record, "subcategory") or record_field_value(record, "key"))
    source_variant = str(record_field_value(record, "source_variant") or record_field_value(record, "source"))
    original_image_path = str(record.get("original_image_path") or "")
    edited_image_path = str(record.get("edited_image_path") or "")
    return category, subcategory, source_variant, original_image_path, edited_image_path


def build_item_row(
    *,
    key: Tuple[str, str, str, str, str],
    bucket: Sequence[Dict[str, Any]],
    min_cf_responses: int,
    easy_accuracy: float,
    easy_bias_rate: float,
    hard_accuracy: float,
) -> Dict[str, Any]:
    category, subcategory, source_variant, original_path, edited_path = key
    overall = summarize_bucket(bucket)
    cf_only = summarize_bucket(filter_by_input_mode(bucket, "cf_only"))
    both = summarize_bucket(filter_by_input_mode(bucket, "both"))
    orig_only = summarize_bucket(filter_by_input_mode(bucket, "orig_only"))

    question_type_summaries = {
        question_type: summarize_bucket(filter_by_question_type(bucket, question_type))
        for question_type in QUESTION_TYPES
    }
    cf_accuracy = float(cf_only["accuracy"])
    cf_bias_rate = float(cf_only["bias_rate"])
    easy_score = cf_accuracy - cf_bias_rate
    recommended_action = recommend_action(
        cf_response_count=int(cf_only["response_count"]),
        cf_accuracy=cf_accuracy,
        cf_bias_rate=cf_bias_rate,
        min_cf_responses=min_cf_responses,
        easy_accuracy=easy_accuracy,
        easy_bias_rate=easy_bias_rate,
        hard_accuracy=hard_accuracy,
    )

    return {
        "Category": category,
        "Subcategory": subcategory,
        "Source_Variant": source_variant,
        "Original_Image": Path(original_path).name if original_path else "",
        "Edited_Image": Path(edited_path).name if edited_path else "",
        "Original_Image_Path": original_path,
        "Edited_Image_Path": edited_path,
        "Model_Count": count_distinct(bucket, ("backbone_model_alias", "model")),
        "Question_Count": count_distinct(bucket, ("question_id",)),
        "Total_Responses": overall["response_count"],
        "Total_Errors": overall["error_count"],
        "Overall_Accuracy%": percent(overall["accuracy"]),
        "Overall_Bias%": percent(overall["bias_rate"]),
        "Overall_Other%": percent(overall["other_rate"]),
        "CF_Only_N": cf_only["response_count"],
        "CF_Only_Accuracy%": percent(cf_accuracy),
        "CF_Only_Bias%": percent(cf_bias_rate),
        "CF_Only_Other%": percent(cf_only["other_rate"]),
        "Both_N": both["response_count"],
        "Both_Accuracy%": percent(both["accuracy"]),
        "Both_Bias%": percent(both["bias_rate"]),
        "Orig_Only_N": orig_only["response_count"],
        "Orig_Only_Accuracy%": percent(orig_only["accuracy"]),
        "Yes_No_Accuracy%": percent_or_blank(question_type_summaries["yes_no"]),
        "Multiple_Choice_Accuracy%": percent_or_blank(question_type_summaries["multiple_choice"]),
        "Open_Accuracy%": percent_or_blank(question_type_summaries["open"]),
        "Easy_Score": round(easy_score, 4),
        "Recommended_Action": recommended_action,
    }


def filter_by_input_mode(records: Sequence[Dict[str, Any]], input_mode: str) -> List[Dict[str, Any]]:
    return [record for record in records if input_mode_for_record(record) == input_mode]


def input_mode_for_record(record: Dict[str, Any]) -> str:
    return str(record.get("input_mode") or record.get("eval_mode") or "")


def filter_by_question_type(records: Sequence[Dict[str, Any]], question_type: str) -> List[Dict[str, Any]]:
    return [record for record in records if record.get("question_type") == question_type]


def recommend_action(
    *,
    cf_response_count: int,
    cf_accuracy: float,
    cf_bias_rate: float,
    min_cf_responses: int,
    easy_accuracy: float,
    easy_bias_rate: float,
    hard_accuracy: float,
) -> str:
    if cf_response_count < min_cf_responses:
        return "keep"
    if cf_accuracy >= easy_accuracy and cf_bias_rate <= easy_bias_rate:
        return "consider_remove"
    if cf_accuracy <= hard_accuracy:
        return "review_quality"
    return "keep"


def count_distinct(records: Sequence[Dict[str, Any]], fields: Iterable[str]) -> int:
    values = set()
    for record in records:
        for field in fields:
            value = record.get(field)
            if value:
                values.add(str(value))
                break
    return len(values)


def percent(value: Any) -> float:
    return round(float(value) * 100.0, 2)


def percent_or_blank(summary: Dict[str, Any]) -> float | str:
    if int(summary["response_count"]) == 0:
        return ""
    return percent(summary["accuracy"])


def print_summary(
    *,
    records: Sequence[Dict[str, Any]],
    run_count: int,
    rows: Sequence[Dict[str, Any]],
    top_k: int,
) -> None:
    action_counts = Counter(row["Recommended_Action"] for row in rows)
    print(f"Loaded {len(records)} records from {run_count} run directory/directories.")
    print(f"Image pairs: {len(rows)}")
    print(
        "Recommendations: "
        f"consider_remove={action_counts.get('consider_remove', 0)} "
        f"review_quality={action_counts.get('review_quality', 0)} "
        f"keep={action_counts.get('keep', 0)}"
    )
    if top_k == 0:
        return
    print_top_rows(
        title=f"Top {top_k} easiest samples",
        rows=sorted(rows, key=lambda row: (numeric(row["Easy_Score"]), numeric(row["CF_Only_N"])), reverse=True),
        top_k=top_k,
    )
    print_top_rows(
        title=f"Top {top_k} lowest-accuracy samples",
        rows=sorted(rows, key=lambda row: (numeric(row["CF_Only_Accuracy%"]), -numeric(row["CF_Only_N"]))),
        top_k=top_k,
    )


def print_top_rows(*, title: str, rows: Sequence[Dict[str, Any]], top_k: int) -> None:
    print()
    print(title)
    print("Action           CF_N  CF_Acc  CF_Bias  Easy  Category/Subcategory/Source  Edited_Image")
    for row in rows[:top_k]:
        print(
            f"{row['Recommended_Action']:<15} "
            f"{int(row['CF_Only_N']):>4} "
            f"{float(row['CF_Only_Accuracy%']):>7.2f} "
            f"{float(row['CF_Only_Bias%']):>8.2f} "
            f"{float(row['Easy_Score']):>5.2f} "
            f"{row['Category']}/{row['Subcategory']}/{row['Source_Variant']:<8} "
            f"{row['Edited_Image']}"
        )


def numeric(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


if __name__ == "__main__":
    sys.exit(main())
