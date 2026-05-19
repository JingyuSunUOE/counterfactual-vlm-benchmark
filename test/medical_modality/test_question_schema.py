#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
QUESTIONS = ROOT / "medical" / "modality_swapping" / "medical_modality_questions.json"


def main() -> int:
    data = json.loads(QUESTIONS.read_text(encoding="utf-8"))
    assert data["benchmark_name"] == "medical_modality"
    assert data["input_modes"] == ["cf_only", "both", "orig_only"]
    assert data["question_type_order"] == ["yes_no", "multiple_choice", "open"]
    groups = data["categories"][0]["question_groups"]
    assert len(groups) == 4
    cf_only_yes_no_answers = []
    for group in groups:
        questions = group["questions"]
        assert {question["type"] for question in questions} == {"yes_no", "multiple_choice", "open"}
        yes_no = next(question for question in questions if question["type"] == "yes_no")
        prompt = yes_no["prompt"].lower()
        assert "does the tumor region look more flair-like than native t1-like" in prompt
        assert "flair or t2-flair" not in prompt
        assert "same mri sequence as the surrounding brain tissue" not in yes_no["prompt"].lower()
        cf_only_yes_no_answers.append(yes_no["answers"]["cf_only"]["correct_answer"])
        for mode in ("cf_only", "both"):
            assert yes_no["answers"][mode]["correct_answer"] != yes_no["answers"][mode]["biased_answer"]
        assert yes_no["answers"]["orig_only"]["correct_answer"] == yes_no["answers"]["orig_only"]["biased_answer"]
        mc = next(question for question in questions if question["type"] == "multiple_choice")
        assert [choice["id"] for choice in mc["choices"]] == ["A", "B", "C", "D"]
        for question in questions:
            assert set(question["answers"]) == {"cf_only", "both", "orig_only"}
        for mode in ("cf_only", "both"):
            assert mc["answers"][mode]["correct_option_id"] != mc["answers"][mode]["biased_option_id"]
        assert mc["answers"]["orig_only"]["correct_option_id"] == mc["answers"]["orig_only"]["biased_option_id"]
    assert sorted(cf_only_yes_no_answers) == ["no", "no", "yes", "yes"]
    serialized = json.dumps(data, ensure_ascii=False).lower()
    assert "counterfactual" not in serialized
    assert "edited image" not in serialized
    print("medical modality question schema ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
