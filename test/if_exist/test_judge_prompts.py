#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "eval_code" / "if_exist"))

from eval_common import (  # noqa: E402
    QuestionSpec,
    build_judge_prompt,
    build_model_prompt,
    build_yes_no_judge_prompt,
    parse_multiple_choice,
    score_closed_form,
)


def main() -> int:
    prompt = build_yes_no_judge_prompt(
        question_prompt="Does the visible animal appear to be in snow? Answer only yes or no.",
        eval_mode="cf_only",
        candidate_answer="No, it is on a beach.",
        correct_answer="no",
        biased_answer="yes",
    )
    require_all(
        "yes_no judge prompt",
        prompt,
        [
            "correct_answer",
            "biased_answer",
            "candidate_answer",
            "No, it is on a beach.",
            "correct|biased|other",
            "strict JSON",
        ],
    )

    label = score_closed_form(
        "multiple_choice",
        "A",
        {
            "correct_option_id": "A",
            "biased_option_id": "B",
        },
    )
    if label != "correct":
        raise SystemExit(f"multiple_choice deterministic scoring regressed: {label}")
    mc_parse_cases = {
        "B": "B",
        "B. The visible subject area deviates.": "B",
        "Answer: C": "C",
        "The answer is D because the view is unclear.": "D",
        "Looking at the second image group, I can see a camel. The answer is B.": "B",
        "Looking at the second image group, the feature is absent.\n\nB": "B",
        "The visible area appears normal.\nFinal answer: A.": "A",
        "Looking at the second image group, I can see a camel in profile view.": None,
        "This is a camel.": None,
    }
    for raw, expected in mc_parse_cases.items():
        parsed = parse_multiple_choice(raw)
        if parsed != expected:
            raise SystemExit(f"strict multiple_choice parsing failed for {raw!r}: expected {expected!r}, got {parsed!r}")

    question = QuestionSpec(
        category="if_exist",
        subcategory="camel",
        display_name="camel",
        question_id="tool_prompt_yes_no",
        question_type="yes_no",
        prompt="Does this image show the kind of appearance normally expected for a camel? Answer only yes or no.",
        choices=[],
        answers_by_mode={},
    )
    tool_prompt = build_model_prompt(
        question=question,
        prime="off",
        eval_mode="both",
        tool_condition="crop",
    )
    require_all(
        "tool prompt",
        tool_prompt,
        [
            "If two image groups are provided, answer about the second image group.",
            "alternative visual views",
        ],
    )
    forbidden_prefix_terms = ["counterfactual", "edited", "bias", "segmentation", "evidence"]
    prefix = tool_prompt.split("\n\n", 1)[0]
    leaked = [term for term in forbidden_prefix_terms if term.lower() in prefix.lower()]
    if " sam " in f" {prefix.lower()} ":
        leaked.append("SAM")
    if leaked:
        raise SystemExit(f"Tool prompt prefix contains forbidden terms: {leaked}")

    answers = {
        "correct_rubric": {
            "target_claim": "The target image contradicts the expected appearance.",
            "required_visual_evidence": ["visible feature is absent"],
            "reject_if": ["feature is clearly present"],
        },
        "bias_rubric": {
            "target_claim": "The target image supports the expected appearance from prior.",
            "qualify_if": ["feature is treated as present from prior"],
            "reject_if": ["feature is described as absent"],
        },
        "other_rubric": {"label_when": ["ambiguous or unsupported"]},
    }
    cf_judge_prompt = build_judge_prompt(
        question_prompt="Prompt",
        eval_mode="cf_only",
        candidate_answer="The shape supports the expected appearance.",
        answers=answers,
        tool_condition="crop",
    )
    if "In both mode" in cf_judge_prompt or "the target is the second image" in cf_judge_prompt:
        raise SystemExit("cf_only open judge prompt unexpectedly contains both-mode second-image rules.")
    require_all(
        "cf_only open judge prompt",
        cf_judge_prompt,
        ["single target image group", "Do not require the answer to mention a second image"],
    )

    both_tool_judge_prompt = build_judge_prompt(
        question_prompt="Prompt",
        eval_mode="both",
        candidate_answer="The second group contradicts the expected appearance.",
        answers=answers,
        tool_condition="crop",
    )
    require_all(
        "both tool open judge prompt",
        both_tool_judge_prompt,
        ["target is the second image group", "first image group"],
    )

    both_raw_judge_prompt = build_judge_prompt(
        question_prompt="Prompt",
        eval_mode="both",
        candidate_answer="The two images differ in the expected appearance.",
        answers=answers,
        tool_condition="raw",
    )
    require_all(
        "both raw open judge prompt",
        both_raw_judge_prompt,
        ["compares the two related images", "Do not require or reward identifying image order"],
    )
    forbidden_raw_both = ["second image", "first image", "target image", "counterfactual", "edited"]
    present_raw_both = [fragment for fragment in forbidden_raw_both if fragment in both_raw_judge_prompt.lower()]
    if present_raw_both:
        raise SystemExit(f"raw both open judge prompt leaked target/order terminology: {present_raw_both}")

    orig_judge_prompt = build_judge_prompt(
        question_prompt="Prompt",
        eval_mode="orig_only",
        candidate_answer="The visible feature is present.",
        answers=answers,
        tool_condition="raw",
    )
    if "In both mode" in orig_judge_prompt:
        raise SystemExit("orig_only open judge prompt unexpectedly contains both-mode rules.")
    require_all("orig_only open judge prompt", orig_judge_prompt, ["single original image", "Apply the orig_only rubrics"])

    print("if_exist judge prompt tests passed.")
    return 0


def require_all(name: str, text: str, expected: list[str]) -> None:
    missing = [item for item in expected if item not in text]
    if missing:
        raise SystemExit(f"{name} missing expected fragments: {missing}")


if __name__ == "__main__":
    sys.exit(main())
