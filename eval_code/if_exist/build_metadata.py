#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

from eval_common import (
    DEFAULT_METADATA_ROOT,
    DEFAULT_QUESTIONS_PATH,
    build_group_specs,
    build_image_pairs,
    load_benchmark,
    target_texts_for_record,
    write_json,
)


DEFAULT_OUTPUT_PATH = DEFAULT_METADATA_ROOT / "if_exist_cf_metadata.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build if_exist metadata for cross-benchmark analysis and reporting."
    )
    parser.add_argument(
        "--questions",
        type=Path,
        default=DEFAULT_QUESTIONS_PATH,
        help="Benchmark question JSON path.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="Metadata JSON output path.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Preview metadata records without writing a file.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    benchmark = load_benchmark(args.questions)
    subcategory_metadata = build_subcategory_metadata(benchmark)
    groups = build_group_specs(benchmark)
    image_pairs = build_image_pairs(groups)
    questions_by_key = {
        (group.category, group.subcategory, group.source_variant): group.questions for group in groups
    }

    records: List[Dict[str, Any]] = []
    for pair in image_pairs:
        questions = questions_by_key[(pair.category, pair.subcategory, pair.source_variant)]
        metadata_source_question = select_metadata_question(questions)
        answers = build_answers_payload(metadata_source_question)
        subcategory_info = subcategory_metadata[(pair.category, pair.subcategory)]
        records.append(
            {
                "category": pair.category,
                "subcategory": pair.subcategory,
                "display_name": subcategory_info["display_name"],
                "source": pair.source_variant,
                "source_variant": pair.source_variant,
                "original_image": pair.original_path.name,
                "cf_filename": pair.edited_path.name,
                "original_path": str(pair.original_path),
                "cf_path": str(pair.edited_path),
                "edit_type": subcategory_info["edit_type"],
                "target_visual_cue": subcategory_info["target_visual_cue"],
                "bias_axis": subcategory_info["bias_axis"],
                "answers": answers,
                "question_ids": [question.question_id for question in questions],
            }
        )

    if args.dry_run:
        print(f"metadata_records={len(records)}")
        for record in records[:5]:
            print(json.dumps(record, ensure_ascii=False))
        if len(records) > 5:
            print(f"... {len(records) - 5} additional records not shown")
        return 0

    write_json(args.output, records)
    print(f"Wrote {len(records)} metadata records to {args.output}")
    return 0


def build_subcategory_metadata(benchmark: Dict[str, Any]) -> Dict[Tuple[str, str], Dict[str, str]]:
    metadata: Dict[Tuple[str, str], Dict[str, str]] = {}
    for category_entry in benchmark["categories"]:
        category = category_entry["category"]
        for subcategory_entry in category_entry["subcategories"]:
            subcategory = subcategory_entry["subcategory"]
            metadata[(category, subcategory)] = {
                "display_name": subcategory_entry["display_name"],
                "edit_type": subcategory_entry["edit_type"],
                "target_visual_cue": subcategory_entry["target_visual_cue"],
                "bias_axis": subcategory_entry["bias_axis"],
            }
    return metadata


def select_metadata_question(questions: List[Any]) -> Any:
    for question in questions:
        if question.question_type == "open":
            return question
    return questions[0]


def build_answers_payload(question: Any) -> Dict[str, Dict[str, str | None]]:
    answers: Dict[str, Dict[str, str | None]] = {}
    for input_mode in ("cf_only", "both", "orig_only"):
        correct, biased = target_texts_for_record(
            question_type=question.question_type,
            answers=question.answers_by_mode[input_mode],
            eval_mode=input_mode,
        )
        answers[input_mode] = {
            "correct": correct,
            "biased": None if input_mode == "orig_only" else biased,
        }
    return answers


if __name__ == "__main__":
    sys.exit(main())
