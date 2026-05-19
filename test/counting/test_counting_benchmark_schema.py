#!/usr/bin/env python3
from __future__ import annotations

import json
import subprocess
import sys
from collections import Counter
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
ANNOTATIONS_PATH = REPO_ROOT / "eval_results/counting/metadata/counting_count_annotations.json"
QUESTIONS_PATH = REPO_ROOT / "cf_dataset/counting_cf_questions.json"
DIRECT_CUE_FRAGMENTS = [
    "extra wing",
    "missing wing",
    "six fingers",
    "one wing",
    "three wings",
]


def main() -> int:
    run(
        "build counting benchmark dry-run",
        [sys.executable, "gen_code/counting/build_counting_benchmark.py", "--dry-run"],
        ["annotations=256", "question_groups=44", "questions=132"],
    )
    if not ANNOTATIONS_PATH.exists() or not QUESTIONS_PATH.exists():
        run(
            "build counting benchmark",
            [sys.executable, "gen_code/counting/build_counting_benchmark.py", "--overwrite"],
            ["annotations=256", "question_groups=44", "questions=132"],
        )

    annotations = json.loads(ANNOTATIONS_PATH.read_text(encoding="utf-8"))
    questions = json.loads(QUESTIONS_PATH.read_text(encoding="utf-8"))
    validate_annotations(annotations)
    validate_questions(questions)
    print("counting benchmark schema tests passed.")
    return 0


def validate_annotations(annotations: list[dict]) -> None:
    if len(annotations) != 256:
        raise SystemExit(f"Expected 256 annotations, found {len(annotations)}")
    by_subset = Counter(record["subset"] for record in annotations)
    if by_subset != {"bird": 90, "insect": 66, "hand_paw": 100}:
        raise SystemExit(f"Unexpected subset counts: {by_subset}")
    for record in annotations:
        for key in ("original_path", "cf_path"):
            if not Path(record[key]).exists():
                raise SystemExit(f"Annotation path does not exist: {record[key]}")
        if record["subset"] == "bird":
            require_counts(record, normal=2, add=4, remove=1)
        if record["subset"] == "insect":
            if record["subcategory"] == "diptera":
                require_counts(record, normal=2, add=4, remove=1)
            else:
                require_counts(record, normal=4, add=6, remove=3)
        if record["subset"] == "hand_paw":
            if record["edit_type"] == "add_to_six":
                if record["normal_count"] != 5 or record["cf_count"] != 6 or record["biased_count"] != 5:
                    raise SystemExit(f"Unexpected add_to_six counts: {record['pair_key']}")
            elif record["edit_type"] == "add_to_five":
                if record["normal_count"] != 4 or record["cf_count"] != 5 or record["biased_count"] != 4:
                    raise SystemExit(f"Unexpected add_to_five counts: {record['pair_key']}")
            else:
                raise SystemExit(f"Unexpected hand_paw edit_type: {record['edit_type']}")


def require_counts(record: dict, *, normal: int, add: int, remove: int) -> None:
    expected_cf = add if record["edit_type"] == "wings_add" else remove
    if record["normal_count"] != normal or record["cf_count"] != expected_cf or record["biased_count"] != normal:
        raise SystemExit(f"Unexpected counts for {record['pair_key']}: {record}")


def validate_questions(questions: dict) -> None:
    if questions["input_modes"] != ["cf_only", "both", "orig_only"]:
        raise SystemExit("Unexpected input_modes.")
    if questions["category_order"] != ["counting"]:
        raise SystemExit("Unexpected category_order.")
    if questions["question_type_order"] != ["yes_no", "multiple_choice", "open"]:
        raise SystemExit("Unexpected question_type_order.")

    groups = questions["categories"][0]["question_groups"]
    if len(groups) != 44:
        raise SystemExit(f"Expected 44 question groups, found {len(groups)}")
    total_questions = sum(len(group["questions"]) for group in groups)
    if total_questions != 132:
        raise SystemExit(f"Expected 132 questions, found {total_questions}")

    for group in groups:
        qtypes = [question["type"] for question in group["questions"]]
        if qtypes != ["yes_no", "multiple_choice", "open"]:
            raise SystemExit(f"Unexpected question type order for {group['question_group_id']}: {qtypes}")
        validate_group_counts(group)
        for question in group["questions"]:
            prompt_lower = question["prompt"].lower()
            stale = [fragment for fragment in DIRECT_CUE_FRAGMENTS if fragment in prompt_lower]
            if stale:
                raise SystemExit(f"Prompt contains direct cue fragments {stale}: {question['question_id']}")
            if set(question["answers"]) != {"cf_only", "both", "orig_only"}:
                raise SystemExit(f"Question missing answer modes: {question['question_id']}")
            validate_question_answers(question, group)


def validate_group_counts(group: dict) -> None:
    if group["subset"] == "insect":
        if group["subcategory"] == "diptera" and group["normal_count"] != 2:
            raise SystemExit("Diptera normal_count should be 2.")
        if group["subcategory"] != "diptera" and group["normal_count"] != 4:
            raise SystemExit(f"Non-Diptera insect normal_count should be 4: {group['subcategory']}")


def validate_question_answers(question: dict, group: dict) -> None:
    answers = question["answers"]
    if question["type"] == "yes_no":
        direction_word = "higher than" if group["count_direction"] == "more_than_expected" else "lower than"
        if direction_word not in question["prompt"]:
            raise SystemExit(f"{question['question_id']} prompt should ask about {direction_word} expected count.")
        if "show the normal expected count" in question["prompt"]:
            raise SystemExit(f"{question['question_id']} still uses stale normal-count yes/no wording.")
        for mode in ("cf_only", "both"):
            if answers[mode]["correct_answer"] != "yes" or answers[mode]["biased_answer"] != "no":
                raise SystemExit(f"{question['question_id']} {mode} should map to yes/no.")
            if answers[mode]["correct_answer"] == answers[mode]["biased_answer"]:
                raise SystemExit(f"{question['question_id']} {mode} correct and biased should differ.")
        if answers["orig_only"]["correct_answer"] != "no" or answers["orig_only"]["biased_answer"] != "no":
            raise SystemExit(f"{question['question_id']} orig_only should map to no/no.")
    elif question["type"] == "multiple_choice":
        expected = "B" if group["count_direction"] == "more_than_expected" else "C"
        for mode in ("cf_only", "both"):
            if answers[mode]["correct_option_id"] != expected:
                raise SystemExit(f"{question['question_id']} {mode} should use correct option {expected}.")
            if answers[mode]["biased_option_id"] != "A":
                raise SystemExit(f"{question['question_id']} {mode} biased option should be A.")
            if answers[mode]["correct_option_id"] == answers[mode]["biased_option_id"]:
                raise SystemExit(f"{question['question_id']} {mode} correct and biased should differ.")
        if answers["orig_only"]["correct_option_id"] != "A" or answers["orig_only"]["biased_option_id"] != "A":
            raise SystemExit(f"{question['question_id']} orig_only should map to A/A.")
    elif question["type"] == "open":
        for mode in ("cf_only", "both", "orig_only"):
            payload = answers[mode]
            for key in ("correct_rubric", "bias_rubric", "other_rubric"):
                if key not in payload:
                    raise SystemExit(f"{question['question_id']} {mode} missing {key}.")
            serialized = json.dumps(payload, ensure_ascii=False).lower()
            if "counterfactual edit" in serialized:
                raise SystemExit(f"{question['question_id']} {mode} should not use counterfactual-edit wording.")
            if "clearly resolves the second image" in serialized:
                raise SystemExit(f"{question['question_id']} {mode} should not require explicit second-image wording.")
    else:
        raise SystemExit(f"Unexpected question type: {question['type']}")


def run(name: str, command: list[str], expected: list[str]) -> str:
    print(f"[RUN] {name}: {' '.join(command)}")
    completed = subprocess.run(
        command,
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        timeout=120,
        check=False,
    )
    output = completed.stdout + completed.stderr
    if completed.returncode != 0:
        print(output)
        raise SystemExit(f"{name} failed with exit code {completed.returncode}")
    missing = [fragment for fragment in expected if fragment not in output]
    if missing:
        raise SystemExit(f"{name} missing expected fragments: {missing}")
    return output


if __name__ == "__main__":
    sys.exit(main())
