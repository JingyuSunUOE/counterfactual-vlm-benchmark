#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


REPO_ROOT = Path(__file__).resolve().parents[2]
INPUT_MODES = ["cf_only", "both", "orig_only"]
QUESTION_TYPES = ["yes_no", "multiple_choice", "open"]
MC_CHOICES = [
    ("A", "The visible count clearly matches the normal expected count for this subject."),
    ("B", "The visible count is clearly higher than the normal expected count for this subject."),
    ("C", "The visible count is clearly lower than the normal expected count for this subject."),
    ("D", "The relevant body parts are too cropped, overlapped, or unclear to judge reliably."),
]
DIRECT_CUE_FRAGMENTS = [
    "extra wing",
    "missing wing",
    "six fingers",
    "one wing",
    "three wings",
]


INSECT_PRIORS = {
    "diptera": ("Flies and mosquitoes are widely known as two-winged insects.", "fly or mosquito"),
    "lepidoptera": ("Butterflies and moths are widely known for having two pairs of wings.", "butterfly or moth"),
    "odonata": ("Dragonflies and damselflies are widely known for having two pairs of wings.", "dragonfly or damselfly"),
    "coleoptera": ("Beetles and ladybugs are commonly expected to have two pairs of wings.", "beetle or ladybug"),
    "hemiptera": ("True bugs and stink bugs are commonly expected to have two pairs of wings.", "true bug or stink bug"),
    "orthoptera": ("Grasshoppers and crickets are commonly expected to have two pairs of wings.", "grasshopper or cricket"),
    "blattodea": ("Cockroaches are commonly expected to have two pairs of wings.", "cockroach"),
}


HAND_PAW_DISPLAY = {
    "chimpanzee_hand": "chimpanzee hand",
    "human_hand": "human hand",
    "panda_paw": "panda paw",
    "Woodpecker_feet": "woodpecker foot",
    "chicken_feet": "chicken foot",
}


def main() -> int:
    args = parse_args()
    metadata_path = resolve_path(args.metadata)
    annotations_output = resolve_path(args.annotations_output)
    annotations_csv = resolve_path(args.annotations_csv)
    questions_output = resolve_path(args.questions_output)
    report_output = resolve_path(args.report_output)

    metadata = load_metadata(metadata_path)
    annotations = build_annotations(metadata)
    questions = build_questions(annotations)
    validate_outputs(annotations, questions)
    print_summary(annotations, questions)

    if args.dry_run:
        print("Dry run complete. No files were written.")
        return 0

    ensure_writable(annotations_output, args.overwrite)
    ensure_writable(annotations_csv, args.overwrite)
    ensure_writable(questions_output, args.overwrite)
    ensure_writable(report_output, args.overwrite)
    write_json(annotations_output, annotations)
    write_csv(annotations_csv, annotations)
    write_json(questions_output, questions)
    report_output.parent.mkdir(parents=True, exist_ok=True)
    report_output.write_text(build_report(annotations, questions), encoding="utf-8")
    print("Counting benchmark annotations, questions, and report written.")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build counting benchmark annotations and question JSON.")
    parser.add_argument("--metadata", default="eval_results/counting/metadata/counting_metadata.json")
    parser.add_argument("--questions-output", default="cf_dataset/counting_cf_questions.json")
    parser.add_argument(
        "--annotations-output",
        default="eval_results/counting/metadata/counting_count_annotations.json",
    )
    parser.add_argument(
        "--annotations-csv",
        default="eval_results/counting/metadata/counting_count_annotations.csv",
    )
    parser.add_argument(
        "--report-output",
        default="eval_results/counting/reports/experiment_design.md",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def resolve_path(value: str | Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return REPO_ROOT / path


def load_metadata(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise SystemExit(f"Missing counting metadata: {path}")
    records = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(records, list) or not records:
        raise SystemExit("Counting metadata must be a non-empty JSON list.")
    return records


def build_annotations(metadata: list[dict[str, Any]]) -> list[dict[str, Any]]:
    annotations: list[dict[str, Any]] = []
    for record in metadata:
        rule = count_rule(record)
        question_group_id = group_id(record)
        display_name = display_name_for(record)
        annotation = dict(record)
        annotation.update(
            {
                "normal_count": rule["normal_count"],
                "cf_count": rule["cf_count"],
                "biased_count": rule["biased_count"],
                "count_direction": rule["count_direction"],
                "question_group_id": question_group_id,
                "prior_statement": prior_statement_for(record),
                "display_name": display_name,
            }
        )
        annotations.append(annotation)
    return sorted(annotations, key=lambda item: item["pair_key"])


def count_rule(record: dict[str, Any]) -> dict[str, Any]:
    subset = record["subset"]
    subcategory = record["subcategory"]
    edit_type = record["edit_type"]
    if subset == "bird":
        normal = 2
    elif subset == "insect":
        normal = 2 if subcategory == "diptera" else 4
    elif subset == "hand_paw":
        normal = 5 if edit_type == "add_to_six" else 4
    else:
        raise SystemExit(f"Unsupported counting subset: {subset}")

    if edit_type in {"wings_add", "add_to_six", "add_to_five"}:
        direction = "more_than_expected"
        cf_count = normal + 2 if edit_type == "wings_add" else normal + 1
    elif edit_type == "wings_remove":
        direction = "fewer_than_expected"
        cf_count = normal - 1
    else:
        raise SystemExit(f"Unsupported edit_type: {edit_type}")

    return {
        "normal_count": normal,
        "cf_count": cf_count,
        "biased_count": normal,
        "count_direction": direction,
    }


def group_id(record: dict[str, Any]) -> str:
    return f"counting_{record['subset']}_{safe_id(record['subcategory'])}_{record['edit_type']}"


def safe_id(value: str) -> str:
    return value.replace(" ", "_").replace("-", "_")


def display_name_for(record: dict[str, Any]) -> str:
    subset = record["subset"]
    subcategory = strip_ai_suffix(record["subcategory"])
    if subset == "bird":
        return f"{subcategory.replace('_', ' ')} bird"
    if subset == "insect":
        return INSECT_PRIORS.get(subcategory, ("", subcategory.replace("_", " ")))[1]
    if subset == "hand_paw":
        return HAND_PAW_DISPLAY.get(subcategory, HAND_PAW_DISPLAY.get(strip_ai_suffix(subcategory), subcategory.replace("_", " ")))
    raise SystemExit(f"Unsupported subset for display name: {subset}")


def strip_ai_suffix(value: str) -> str:
    return value[:-3] if value.endswith("_ai") else value


def prior_statement_for(record: dict[str, Any]) -> str:
    subset = record["subset"]
    subcategory = strip_ai_suffix(record["subcategory"])
    if subset == "bird":
        return "Birds are widely known for having one normal pair of wings."
    if subset == "insect":
        if subcategory not in INSECT_PRIORS:
            raise SystemExit(f"Missing insect prior for {subcategory}")
        return INSECT_PRIORS[subcategory][0]
    if subset == "hand_paw":
        if record["edit_type"] == "add_to_six":
            if subcategory.startswith("human_hand"):
                return "Human hands are widely known for having five fingers."
            if subcategory.startswith("chimpanzee_hand"):
                return "Chimpanzee hands are commonly expected to have five fingers."
            if subcategory.startswith("panda_paw"):
                return "Panda paws are commonly expected to have five visible digits."
        if record["edit_type"] == "add_to_five":
            if subcategory.startswith("chicken_feet"):
                return "Chicken feet are commonly expected to have four visible toes."
            if subcategory.startswith("Woodpecker_feet"):
                return "Woodpecker feet are commonly expected to have four visible toes."
        raise SystemExit(f"Missing hand/paw prior for {record['subcategory']} {record['edit_type']}")
    raise SystemExit(f"Unsupported subset for prior: {subset}")


def build_questions(annotations: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for annotation in annotations:
        grouped[annotation["question_group_id"]].append(annotation)

    question_groups = []
    for group_key, group_records in sorted(grouped.items()):
        representative = group_records[0]
        question_groups.append(
            {
                "question_group_id": group_key,
                "subset": representative["subset"],
                "subcategory": representative["subcategory"],
                "display_name": representative["display_name"],
                "source_variants": sorted({item["source_variant"] for item in group_records}),
                "count_attribute": representative["count_attribute"],
                "edit_type": representative["edit_type"],
                "normal_count": representative["normal_count"],
                "cf_count": representative["cf_count"],
                "biased_count": representative["biased_count"],
                "count_direction": representative["count_direction"],
                "prior_statement": representative["prior_statement"],
                "evaluation_target": "Visible count of the relevant countable body parts.",
                "bias_axis": "A model may rely on the normal category count instead of counting the visible edited body parts.",
                "questions": build_group_questions(representative),
            }
        )

    return {
        "benchmark_name": "counting_counterfactual_vlm_benchmark_questions",
        "version": "1.0",
        "language": "en",
        "input_modes": INPUT_MODES,
        "category_order": ["counting"],
        "question_type_order": QUESTION_TYPES,
        "source_variants": ["real", "ai"],
        "categories": [
            {
                "category": "counting",
                "question_groups": question_groups,
            }
        ],
    }


def build_group_questions(record: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        build_yes_no_question(record),
        build_multiple_choice_question(record),
        build_open_question(record),
    ]


def build_yes_no_question(record: dict[str, Any]) -> dict[str, Any]:
    direction = record["count_direction"]
    if direction == "more_than_expected":
        direction_text = "higher than"
    elif direction == "fewer_than_expected":
        direction_text = "lower than"
    else:
        raise SystemExit(f"Unsupported count_direction: {direction}")
    prompt = (
        f"{record['prior_statement']} Based on the visible countable body parts, "
        f"does this image show a count that is {direction_text} the normal expected count for this subject? "
        "Answer only yes or no."
    )
    return {
        "question_id": f"{record['question_group_id']}_yes_no",
        "type": "yes_no",
        "prompt": prompt,
        "evaluation_target": f"Whether the visible count is {direction_text} the normal expected count.",
        "bias_rationale": "A prior-driven model may answer no because the subject usually has the normal count.",
        "answers": {
            "cf_only": {"correct_answer": "yes", "biased_answer": "no"},
            "both": {"correct_answer": "yes", "biased_answer": "no"},
            "orig_only": {"correct_answer": "no", "biased_answer": "no"},
        },
    }


def build_multiple_choice_question(record: dict[str, Any]) -> dict[str, Any]:
    direction = record["count_direction"]
    if direction == "more_than_expected":
        cf_correct = ("B", MC_CHOICES[1][1])
    elif direction == "fewer_than_expected":
        cf_correct = ("C", MC_CHOICES[2][1])
    else:
        raise SystemExit(f"Unsupported count_direction: {direction}")
    biased = ("A", MC_CHOICES[0][1])
    prompt = (
        f"{record['prior_statement']} Which option best describes whether the visible count "
        "matches the normal expected count for this subject? Answer with only the option letter."
    )
    return {
        "question_id": f"{record['question_group_id']}_multiple_choice",
        "type": "multiple_choice",
        "prompt": prompt,
        "evaluation_target": "Relationship between the visible count and the normal expected count.",
        "bias_rationale": "The normal count prior can pull the model toward the matching-count option.",
        "choices": [{"id": option_id, "text": text} for option_id, text in MC_CHOICES],
        "answers": {
            "cf_only": option_answer(correct=cf_correct, biased=biased),
            "both": option_answer(correct=cf_correct, biased=biased),
            "orig_only": option_answer(correct=biased, biased=biased),
        },
    }


def option_answer(*, correct: tuple[str, str], biased: tuple[str, str]) -> dict[str, str]:
    return {
        "correct_option_id": correct[0],
        "correct_option_text": correct[1],
        "biased_option_id": biased[0],
        "biased_option_text": biased[1],
    }


def build_open_question(record: dict[str, Any]) -> dict[str, Any]:
    prompt = (
        f"{record['prior_statement']} State whether the visible count supports or contradicts "
        "the expected count, and cite one visible counting cue. Answer in one short sentence."
    )
    return {
        "question_id": f"{record['question_group_id']}_open",
        "type": "open",
        "prompt": prompt,
        "evaluation_target": "One-sentence visible-count judgment with evidence.",
        "bias_rationale": "A prior-driven model may describe the count as normal without counting visible body parts.",
        "answers": {
            "cf_only": open_mode_rubric(record, mode="cf_only"),
            "both": open_mode_rubric(record, mode="both"),
            "orig_only": open_mode_rubric(record, mode="orig_only"),
        },
    }


def open_mode_rubric(record: dict[str, Any], *, mode: str) -> dict[str, Any]:
    direction_text = "higher than" if record["count_direction"] == "more_than_expected" else "lower than"
    count_attribute = record["count_attribute"]
    if mode == "orig_only":
        correct_claim = "The image supports the expected count."
        correct_evidence = [
            f"The answer explicitly says the visible {count_attribute} count matches the expected count.",
            "The answer cites concrete visible counting evidence rather than only category knowledge.",
        ]
        bias_claim = "The image is judged as normal-count only from category prior rather than visible evidence."
        bias_qualify = [
            "The answer states that the count is normal but gives no concrete visible counting cue.",
            "The answer relies on what the subject normally has rather than what is visible in the image.",
        ]
    else:
        target = "second image" if mode == "both" else "image"
        correct_claim = f"The {target} contradicts the expected count because the visible count is {direction_text} expected."
        correct_evidence = [
            f"The answer explicitly says the visible {count_attribute} count contradicts the expected count.",
            f"The answer cites concrete visible counting evidence showing a {direction_text} expected count.",
        ]
        bias_claim = f"The {target} is judged as having the normal expected count because of the category prior."
        bias_qualify = [
            "The answer says the visible count is normal or expected despite the target count direction.",
            "The answer says the visible count supports the expected count or normal category template.",
            "The answer gives a prior-aligned normal-count judgment even if it claims to see normal-count evidence.",
        ]
    other_labels = [
        "The answer is vague, hedged, contradictory, or unsupported.",
        "The answer gives no count judgment or no visible counting cue.",
        "The answer only names the subject or expected body part.",
        "The answer says the relevant body parts are cropped, overlapped, or unclear.",
    ]
    if mode == "both":
        other_labels.append("The answer only discusses the first image or mixes both images without resolving the target image.")
    return {
        "correct_rubric": {
            "target_claim": correct_claim,
            "required_visual_evidence": correct_evidence,
            "reject_if": [
                "The answer gives no concrete visible count cue.",
                "The answer is mainly about object identity, species, pose, or image quality rather than count.",
                "The answer says the relevant body parts are too cropped, overlapped, or unclear to judge reliably.",
            ],
        },
        "bias_rubric": {
            "target_claim": bias_claim,
            "prior_basis": record["prior_statement"],
            "qualify_if": bias_qualify,
            "reject_if": [
                "The answer explicitly identifies a visible count that contradicts the expected count.",
                "The answer says the count is unclear, occluded, or not reliably countable.",
            ],
        },
        "other_rubric": {
            "label_when": other_labels
        },
    }


def ensure_writable(path: Path, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise SystemExit(f"Output exists. Use --overwrite to replace it: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_csv(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def validate_outputs(annotations: list[dict[str, Any]], questions: dict[str, Any]) -> None:
    if len(annotations) != 256:
        raise SystemExit(f"Expected 256 annotations, found {len(annotations)}")
    for annotation in annotations:
        for key in ("original_path", "cf_path"):
            if not Path(annotation[key]).exists():
                raise SystemExit(f"Annotation path does not exist: {annotation[key]}")
    groups = questions["categories"][0]["question_groups"]
    if len(groups) != 44:
        raise SystemExit(f"Expected 44 question groups, found {len(groups)}")
    total_questions = sum(len(group["questions"]) for group in groups)
    if total_questions != 132:
        raise SystemExit(f"Expected 132 questions, found {total_questions}")
    for group in groups:
        types = [question["type"] for question in group["questions"]]
        if types != QUESTION_TYPES:
            raise SystemExit(f"Question types are out of order for {group['question_group_id']}: {types}")
        for question in group["questions"]:
            prompt = question["prompt"].lower()
            stale = [fragment for fragment in DIRECT_CUE_FRAGMENTS if fragment in prompt]
            if stale:
                raise SystemExit(f"Prompt contains direct cue fragments {stale}: {question['question_id']}")
            validate_answer_modes(question)


def validate_answer_modes(question: dict[str, Any]) -> None:
    answers = question["answers"]
    if sorted(answers) != sorted(INPUT_MODES):
        raise SystemExit(f"Question missing answer modes: {question['question_id']}")
    if question["type"] == "yes_no":
        for mode in ("cf_only", "both"):
            if answers[mode]["correct_answer"] == answers[mode]["biased_answer"]:
                raise SystemExit(f"{question['question_id']} {mode} correct and biased answers should differ.")
        if answers["orig_only"]["correct_answer"] != answers["orig_only"]["biased_answer"]:
            raise SystemExit(f"{question['question_id']} orig_only correct and biased answers should match.")
    if question["type"] == "multiple_choice":
        for mode in ("cf_only", "both"):
            if answers[mode]["correct_option_id"] == answers[mode]["biased_option_id"]:
                raise SystemExit(f"{question['question_id']} {mode} correct and biased options should differ.")
        if answers["orig_only"]["correct_option_id"] != answers["orig_only"]["biased_option_id"]:
            raise SystemExit(f"{question['question_id']} orig_only correct and biased options should match.")


def print_summary(annotations: list[dict[str, Any]], questions: dict[str, Any]) -> None:
    by_subset: dict[str, int] = defaultdict(int)
    by_edit: dict[str, int] = defaultdict(int)
    for annotation in annotations:
        by_subset[annotation["subset"]] += 1
        by_edit[annotation["edit_type"]] += 1
    print("Counting benchmark build")
    print(f"  annotations={len(annotations)}")
    print(f"  question_groups={len(questions['categories'][0]['question_groups'])}")
    print(f"  questions={sum(len(group['questions']) for group in questions['categories'][0]['question_groups'])}")
    print(f"  by_subset={dict(sorted(by_subset.items()))}")
    print(f"  by_edit_type={dict(sorted(by_edit.items()))}")


def build_report(annotations: list[dict[str, Any]], questions: dict[str, Any]) -> str:
    group_count = len(questions["categories"][0]["question_groups"])
    question_count = sum(len(group["questions"]) for group in questions["categories"][0]["question_groups"])
    return f"""# Counting Counterfactual Benchmark Design

## 1. Purpose

The counting benchmark tests whether VLMs rely on normal category-count priors instead of counting visible body parts in counterfactual images. The benchmark currently covers birds, insects, and hand/paw images.

## 2. Dataset Scale

| Subset | Original Images | Counterfactual Images | Count Attribute |
|---|---:|---:|---|
| `bird` | 45 | 90 | wings |
| `insect` | 33 | 66 | wings |
| `hand_paw` | 100 | 100 | fingers/toes |
| **Total** | **178** | **256** |  |

All counting images are stored as `512x512` RGB PNG files. The standardized originals are under `dataset/counting`, and the counterfactual images are under `cf_dataset/counting_cf`.

## 3. Count Priors and Counterfactual Counts

Birds are treated as normally having one pair of wings. Bird counterfactuals either increase the visible count to four wings or reduce it to one wing.

For insects, `diptera` is treated as a two-winged order. The retained non-Diptera insect orders are treated as having two pairs of wings. The counterfactual counts are four or one wings for `diptera`, and six or three wings for the other retained insect orders.

For hand/paw images, human, chimpanzee, and panda examples are treated as normally having five visible fingers or digits and are edited to six. Chicken and woodpecker feet are treated as normally having four visible toes and are edited to five.

## 4. Question Design

The benchmark uses three input modes: `cf_only`, `both`, and `orig_only`. The `both` mode should be paired with a neutral runner instruction: `If two images are provided, answer about the second image.`

Each question group corresponds to one `(subset, subcategory, edit_type)` combination. The current file contains `{group_count}` question groups and `{question_count}` questions.

The benchmark uses three question types:

| Type | Scoring | Design |
|---|---|---|
| `yes_no` | LLM judge | Asks whether the visible body-part count is higher or lower than the normal expected count, depending on the edit direction. |
| `multiple_choice` | Deterministic option matching | Uses normal / higher / lower / unclear options rather than explicit numeric choices. |
| `open` | Rubric-based LLM judge | Requires a conclusion plus one visible counting cue. |

The multiple-choice design intentionally avoids listing concrete counts such as one, three, four, or six. This preserves the prior-bias pressure while still allowing deterministic scoring.

## 5. Artifacts

| Artifact | Path |
|---|---|
| Count annotations | `eval_results/counting/metadata/counting_count_annotations.json` |
| Count annotations CSV | `eval_results/counting/metadata/counting_count_annotations.csv` |
| Question file | `cf_dataset/counting_cf_questions.json` |
| Original metadata | `eval_results/counting/metadata/counting_metadata.json` |

## 6. Limitations

Some insects may have partially folded, overlapped, or visually subtle wings. Rubrics therefore allow `other` when the relevant body parts are not reliably countable from the image. This avoids treating visual ambiguity as model bias.
"""


if __name__ == "__main__":
    sys.exit(main())
