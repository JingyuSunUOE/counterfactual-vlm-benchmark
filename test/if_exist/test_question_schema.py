#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
QUESTIONS_PATH = REPO_ROOT / "cf_dataset" / "if_exist_cf_questions.json"

EXPECTED_SUBCATEGORIES = {
    "if_exist": ["camel", "elephant_trunk", "fish_fin", "rabbit"],
}
QUESTION_TYPES = ["yes_no", "multiple_choice", "open"]
ANSWER_MODES = ["cf_only", "both", "orig_only"]


def main() -> int:
    data = json.loads(QUESTIONS_PATH.read_text(encoding="utf-8"))
    text = QUESTIONS_PATH.read_text(encoding="utf-8")

    require_equal(data.get("input_modes"), ANSWER_MODES, "input_modes")
    require_equal(data.get("category_order"), ["if_exist"], "category_order")
    require_equal(data.get("question_type_order"), QUESTION_TYPES, "question_type_order")
    require_equal(data.get("source_variants"), ["real", "ai"], "source_variants")

    categories = data.get("categories")
    if not isinstance(categories, list) or len(categories) != 1:
        raise SystemExit("Expected exactly one category.")

    question_count = 0
    open_count = 0
    for category_entry in categories:
        category = require_str(category_entry.get("category"), "category")
        if category not in EXPECTED_SUBCATEGORIES:
            raise SystemExit(f"Unexpected category: {category}")
        subcategories = category_entry.get("subcategories")
        if not isinstance(subcategories, list):
            raise SystemExit(f"Category {category} has invalid subcategories.")
        require_equal(
            [item.get("subcategory") for item in subcategories],
            EXPECTED_SUBCATEGORIES[category],
            f"{category} subcategory order",
        )
        for subcategory_entry in subcategories:
            require_str(subcategory_entry.get("display_name"), f"{category}/{subcategory_entry.get('subcategory')} display_name")
            require_equal(subcategory_entry.get("source_variants"), ["real", "ai"], "subcategory source_variants")
            questions = subcategory_entry.get("questions")
            if not isinstance(questions, list) or len(questions) != 3:
                raise SystemExit(f"{category}/{subcategory_entry.get('subcategory')} should have three questions.")
            require_equal([item.get("type") for item in questions], QUESTION_TYPES, "question type order")
            for question in questions:
                question_count += 1
                validate_question(question)
                if question["type"] == "open":
                    open_count += 1

    require_equal(question_count, 12, "total question count")
    require_equal(open_count, 4, "open question count")
    validate_prompt_quality(text)
    print("if_exist question schema tests passed.")
    return 0


def validate_question(question: dict[str, Any]) -> None:
    for field in ("question_id", "type", "prompt", "evaluation_target", "bias_rationale", "answers"):
        if field not in question:
            raise SystemExit(f"Question missing required field {field}: {question}")
    question_type = question["type"]
    answers = question["answers"]
    require_equal(list(answers.keys()), ANSWER_MODES, f"{question['question_id']} answer modes")

    if question_type == "yes_no":
        for mode in ANSWER_MODES:
            payload = answers[mode]
            require_keys(payload, {"correct_answer", "biased_answer"}, f"{question['question_id']} {mode}")
            for key in ("correct_answer", "biased_answer"):
                value = payload[key]
                if value not in {"yes", "no"}:
                    raise SystemExit(f"{question['question_id']} {mode} {key} must be yes/no, got {value!r}")
    elif question_type == "multiple_choice":
        choices = question.get("choices")
        if not isinstance(choices, list) or not 2 <= len(choices) <= 4:
            raise SystemExit(f"{question['question_id']} must have 2 to 4 choices.")
        expected_ids = [chr(ord("A") + index) for index in range(len(choices))]
        require_equal([choice.get("id") for choice in choices], expected_ids, f"{question['question_id']} choice ids")
        for choice in choices:
            require_str(choice.get("text"), f"{question['question_id']} choice text")
        for mode in ANSWER_MODES:
            payload = answers[mode]
            require_keys(
                payload,
                {"correct_option_id", "correct_option_text", "biased_option_id", "biased_option_text"},
                f"{question['question_id']} {mode}",
            )
            if payload["correct_option_id"] not in expected_ids or payload["biased_option_id"] not in expected_ids:
                raise SystemExit(f"{question['question_id']} {mode} option id is outside choices.")
    elif question_type == "open":
        prompt = question["prompt"]
        if "supports or contradicts" not in prompt:
            raise SystemExit(f"{question['question_id']} open prompt must ask for a supports/contradicts judgment.")
        if not prompt.endswith("Answer in one short sentence."):
            raise SystemExit(f"{question['question_id']} open prompt must end with the standard short-answer suffix.")
        legacy_keys = {"reference_answer", "accepted_keywords", "biased_answer", "biased_keywords"}
        for mode in ANSWER_MODES:
            payload = answers[mode]
            if legacy_keys.intersection(payload):
                raise SystemExit(f"{question['question_id']} {mode} still contains legacy open-answer keys.")
            require_keys(payload, {"correct_rubric", "bias_rubric", "other_rubric"}, f"{question['question_id']} {mode}")
            validate_open_rubric(question["question_id"], mode, payload)
    else:
        raise SystemExit(f"Unexpected question type: {question_type}")


def validate_prompt_quality(text: str) -> None:
    stale_fragments = [
        "ecologically typical",
        "most plausibly infer",
        "at a glance",
        "What single visible cue",
        "cool rocky coast",
        "cool rocky coastline",
        "cold rocky shore",
        "cold rocky shoreline",
        "rocky-shore cue",
    ]
    found = [fragment for fragment in stale_fragments if fragment in text]
    if found:
        raise SystemExit(f"Question file still contains stale prompt/rubric fragments: {found}")


def validate_open_rubric(question_id: str, mode: str, payload: dict[str, Any]) -> None:
    correct = payload["correct_rubric"]
    bias = payload["bias_rubric"]
    other = payload["other_rubric"]
    require_keys(correct, {"target_claim", "required_visual_evidence", "reject_if"}, f"{question_id} {mode} correct")
    require_keys(bias, {"target_claim", "prior_basis", "qualify_if", "reject_if"}, f"{question_id} {mode} bias")
    require_keys(other, {"label_when"}, f"{question_id} {mode} other")
    require_str(correct["target_claim"], f"{question_id} {mode} correct target_claim")
    require_non_empty_list(correct["required_visual_evidence"], f"{question_id} {mode} required_visual_evidence")
    require_non_empty_list(correct["reject_if"], f"{question_id} {mode} correct reject_if")
    require_str(bias["target_claim"], f"{question_id} {mode} bias target_claim")
    require_str(bias["prior_basis"], f"{question_id} {mode} bias prior_basis")
    require_non_empty_list(bias["qualify_if"], f"{question_id} {mode} bias qualify_if")
    require_non_empty_list(bias["reject_if"], f"{question_id} {mode} bias reject_if")
    require_non_empty_list(other["label_when"], f"{question_id} {mode} other label_when")


def require_keys(payload: dict[str, Any], expected: set[str], label: str) -> None:
    missing = expected - set(payload)
    if missing:
        raise SystemExit(f"{label} missing keys: {sorted(missing)}")


def require_equal(actual: Any, expected: Any, label: str) -> None:
    if actual != expected:
        raise SystemExit(f"{label} mismatch: expected {expected!r}, got {actual!r}")


def require_str(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise SystemExit(f"{label} must be a non-empty string.")
    return value


def require_non_empty_list(value: Any, label: str) -> None:
    if not isinstance(value, list) or not value:
        raise SystemExit(f"{label} must be a non-empty list.")
    for item in value:
        require_str(item, label)


if __name__ == "__main__":
    sys.exit(main())
