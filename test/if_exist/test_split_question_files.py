#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
QUESTIONS_PATH = REPO_ROOT / "cf_dataset" / "if_exist_cf_questions.json"
QUESTION_TYPES = ["yes_no", "multiple_choice", "open"]
ANSWER_MODES = ["cf_only", "both", "orig_only"]
IF_EXIST_HARD_MC_SUBCATEGORIES = {"camel", "fish_fin", "rabbit"}
IF_EXIST_HARD_MC_CHOICES = [
    ("A", "The visible subject area clearly fits the normal expected appearance for this subject."),
    ("B", "The visible subject area clearly deviates from the normal expected appearance for this subject."),
    ("C", "The visible subject area is too cropped, covered, or unclear to judge reliably."),
    ("D", "The visible subject area looks unusual mainly because of pose, lighting, or viewing angle."),
]
IF_EXIST_ELEPHANT_MC_CHOICES = [
    ("A", "The expected feature is clearly present in the visible subject area"),
    ("B", "The expected feature is not clearly present, or appears reduced, blended, or incomplete"),
    ("C", "The relevant subject area is too cropped, covered, or unclear to judge reliably"),
    ("D", "The visible subject area is too distorted or non-diagnostic for a reliable feature judgment"),
]
IF_EXIST_STALE_PROMPT_FRAGMENTS = [
    "Does this image still look like",
    "Which visible description best matches this image",
    "visible hump",
    "clearly visible trunk",
    "prominent upper fin visible",
    "long ear visible",
]
IF_EXIST_HARD_MC_STALE_FRAGMENTS = [
    "expected feature is not clearly present",
    "reduced, blended, or incomplete",
    "clearly present in the visible subject area",
]


def main() -> int:
    if not QUESTIONS_PATH.exists():
        raise SystemExit(f"Missing question file: {QUESTIONS_PATH}")
    data = json.loads(QUESTIONS_PATH.read_text(encoding="utf-8"))
    require_equal(data.get("category_order"), ["if_exist"], f"{QUESTIONS_PATH.name} category_order")
    require_equal(data.get("input_modes"), ANSWER_MODES, f"{QUESTIONS_PATH.name} input_modes")
    require_equal(data.get("question_type_order"), QUESTION_TYPES, f"{QUESTIONS_PATH.name} question_type_order")
    categories = data.get("categories")
    if not isinstance(categories, list) or len(categories) != 1:
        raise SystemExit(f"{QUESTIONS_PATH.name} should contain exactly one category.")
    category_entry = categories[0]
    require_equal(category_entry.get("category"), "if_exist", f"{QUESTIONS_PATH.name} category")
    subcategories = category_entry.get("subcategories")
    if not isinstance(subcategories, list) or len(subcategories) != 4:
        raise SystemExit(f"{QUESTIONS_PATH.name} should contain four subcategories.")

    question_count = 0
    for subcategory_entry in subcategories:
        questions = subcategory_entry.get("questions")
        if not isinstance(questions, list) or len(questions) != 3:
            raise SystemExit(f"{QUESTIONS_PATH.name} {subcategory_entry.get('subcategory')} should have three questions.")
        require_equal([question.get("type") for question in questions], QUESTION_TYPES, "question order")
        for question in questions:
            question_count += 1
            prompt = question.get("prompt", "")
            if not isinstance(prompt, str):
                raise SystemExit(f"{question.get('question_id')} prompt should be a string.")
            answers = question.get("answers")
            if not isinstance(answers, dict):
                raise SystemExit(f"{question.get('question_id')} missing answers.")
            require_equal(list(answers), ANSWER_MODES, f"{question.get('question_id')} answer modes")
            assert_if_exist_question(question, subcategory=str(subcategory_entry.get("subcategory")))
    require_equal(question_count, 12, f"{QUESTIONS_PATH.name} question count")

    print("if_exist split question file tests passed.")
    return 0


def require_equal(actual: Any, expected: Any, label: str) -> None:
    if actual != expected:
        raise SystemExit(f"{label} mismatch: expected {expected!r}, got {actual!r}")


def assert_if_exist_question(question: dict[str, Any], *, subcategory: str) -> None:
    question_id = question.get("question_id")
    prompt = question.get("prompt", "")
    visible_text = prompt
    if question.get("type") == "multiple_choice":
        visible_text = "\n".join([visible_text, *[choice.get("text", "") for choice in question.get("choices") or []]])
    for fragment in IF_EXIST_STALE_PROMPT_FRAGMENTS:
        if fragment in visible_text:
            raise SystemExit(f"{question_id} model-visible text still contains stale direct fragment: {fragment!r}")

    if question.get("type") != "multiple_choice":
        return

    choices = question.get("choices")
    actual_choices = [(choice.get("id"), choice.get("text")) for choice in choices or []]
    expected_choices = (
        IF_EXIST_HARD_MC_CHOICES if subcategory in IF_EXIST_HARD_MC_SUBCATEGORIES else IF_EXIST_ELEPHANT_MC_CHOICES
    )
    require_equal(actual_choices, expected_choices, f"{question_id} MC choices")
    if subcategory in IF_EXIST_HARD_MC_SUBCATEGORIES:
        for fragment in IF_EXIST_HARD_MC_STALE_FRAGMENTS:
            if fragment in visible_text:
                raise SystemExit(f"{question_id} hard MC text still contains stale fragment: {fragment!r}")

    answers = question.get("answers", {})
    require_equal(
        (
            answers["cf_only"]["correct_option_id"],
            answers["cf_only"]["biased_option_id"],
            answers["both"]["correct_option_id"],
            answers["both"]["biased_option_id"],
        ),
        ("B", "A", "B", "A"),
        f"{question_id} cf/both MC mapping",
    )
    require_equal(
        (
            answers["orig_only"]["correct_option_id"],
            answers["orig_only"]["biased_option_id"],
        ),
        ("A", "A"),
        f"{question_id} orig MC mapping",
    )


if __name__ == "__main__":
    sys.exit(main())
