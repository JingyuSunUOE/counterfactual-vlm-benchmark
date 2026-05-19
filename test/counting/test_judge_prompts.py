#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
COMMON_PATH = REPO_ROOT / "eval_code/counting/eval_common.py"


def main() -> int:
    common = import_common()
    benchmark = common.load_benchmark(REPO_ROOT / "cf_dataset/counting_cf_questions.json")
    annotations = common.load_annotations(REPO_ROOT / "eval_results/counting/metadata/counting_count_annotations.json")
    groups = common.build_group_specs(benchmark, annotations=annotations)
    questions_by_key = common.build_questions_by_key(groups)
    pair = next(pair for pair in common.build_image_pairs(groups) if pair.edit_type == "wings_add")
    questions = questions_by_key[pair.question_group_id]

    yes_no = next(question for question in questions if question.question_type == "yes_no")
    yes_no_prompt = common.build_model_prompt(question=yes_no, prime="off", eval_mode="both")
    require_all(
        yes_no_prompt,
        ["Two related images are provided", "Compare the visible target attribute"],
    )
    lowered = yes_no_prompt.lower()
    forbidden = ["counterfactual", "edited", "first image", "second image", "target image"]
    present = [term for term in forbidden if term in lowered]
    if present:
        raise SystemExit(f"Model prompt leaks forbidden language: {present}")
    judge_prompt = common.build_yes_no_judge_prompt(
        question_prompt=yes_no_prompt,
        eval_mode="cf_only",
        candidate_answer="Yes.",
        correct_answer=yes_no.answers_by_mode["cf_only"]["correct_answer"],
        biased_answer=yes_no.answers_by_mode["cf_only"]["biased_answer"],
    )
    require_all(judge_prompt, ["candidate_answer", "correct_answer", "biased_answer", "strict JSON"])

    open_question = next(question for question in questions if question.question_type == "open")
    open_prompt = common.build_judge_prompt(
        question_prompt=common.build_model_prompt(question=open_question, prime="off", eval_mode="cf_only"),
        eval_mode="cf_only",
        candidate_answer="The visible count contradicts the expected count because more wings are visible.",
        answers=open_question.answers_by_mode["cf_only"],
    )
    require_all(
        open_prompt,
        [
            "visual counting benchmark",
            "correct_rubric",
            "bias_rubric",
            "other_rubric",
            "required_visual_evidence",
            "prior-aligned normal-count answer can be biased",
        ],
    )
    forbidden_cf_only = ["second image", "counterfactual target", "edited image"]
    present = [fragment for fragment in forbidden_cf_only if fragment in open_prompt.lower()]
    if present:
        raise SystemExit(f"cf_only open judge prompt contains forbidden target language: {present}")
    if "accepted_keywords" in open_prompt or "biased_keywords" in open_prompt:
        raise SystemExit("Open judge prompt should not use keyword scoring.")

    both_open_prompt = common.build_judge_prompt(
        question_prompt=common.build_model_prompt(question=open_question, prime="off", eval_mode="both"),
        eval_mode="both",
        candidate_answer="The two images show different visible wing counts.",
        answers=common.answers_for_mode(open_question, "both", "raw"),
    )
    require_all(
        both_open_prompt,
        [
            "neutral comparison of the two related images",
            "visible count differs or stays the same",
        ],
    )
    forbidden_both = ["counterfactual target", "edited image", "second image", "first image"]
    present_both = [fragment for fragment in forbidden_both if fragment in both_open_prompt.lower()]
    if present_both:
        raise SystemExit(f"both open judge prompt leaks forbidden target/order language: {present_both}")

    orig_open_prompt = common.build_judge_prompt(
        question_prompt=common.build_model_prompt(question=open_question, prime="off", eval_mode="orig_only"),
        eval_mode="orig_only",
        candidate_answer="The visible count supports the expected count because two wings are clearly visible.",
        answers=open_question.answers_by_mode["orig_only"],
    )
    require_all(
        orig_open_prompt,
        [
            "Decision procedure for orig_only",
            "two wings clearly visible",
            "five fingers visible",
            "four toes clearly seen",
            "Label biased only if the answer says the count is normal or expected but gives no visible counting cue",
        ],
    )
    if "prior-aligned normal-count answer can be biased" in orig_open_prompt:
        raise SystemExit("orig_only open judge prompt should not use cf_only/both prior-aligned bias rule.")

    print("counting judge prompt tests passed.")
    return 0


def import_common():
    spec = importlib.util.spec_from_file_location("counting_eval_common_for_test", COMMON_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {COMMON_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def require_all(text: str, expected: list[str]) -> None:
    missing = [fragment for fragment in expected if fragment not in text]
    if missing:
        raise SystemExit(f"Missing expected fragments: {missing}\n{text[:1000]}")


if __name__ == "__main__":
    raise SystemExit(main())
