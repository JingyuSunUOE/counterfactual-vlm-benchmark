#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "eval_code" / "fashion_industry"))

from eval_questions import (  # noqa: E402
    QUESTION_DESIGN_VERSION,
    QUESTION_DESIGNS,
    build_question_payload,
)
from eval_pipeline import maybe_override_both_raw_open_judge_label  # noqa: E402


QUESTION_TYPES = ("Q1", "Q2", "Q3", "Q_MC")
FORBIDDEN_LEAKAGE_WORDS = ("counterfactual", "edited", "bias", "ground truth")


def main() -> int:
    check_all_questions_have_question_level_targets()
    check_yes_no_mapping()
    check_both_raw_pair_difference_design()
    check_both_raw_open_override()
    check_prior_prefix_and_forbidden_words()
    check_high_risk_question_fixes()
    check_selected_open_prompts_do_not_leak_cf_answer()
    print("Fashion/industry question design checks passed.")
    return 0


def check_all_questions_have_question_level_targets() -> None:
    for key, mod_id in QUESTION_DESIGNS:
        for question_type in QUESTION_TYPES:
            payload = build_question_payload(
                key=key,
                mod_id=mod_id,
                question_type=question_type,
                input_mode="cf_only",
                prime=True,
            )
            assert_equal(payload["question_design_version"], QUESTION_DESIGN_VERSION)
            assert_non_empty(payload["prior_statement"], f"{key} mod{mod_id} prior")
            assert_non_empty(payload["prompt"], f"{key} mod{mod_id} {question_type} prompt")
            assert_non_empty(payload["correct"], f"{key} mod{mod_id} {question_type} correct")
            assert_non_empty(payload["biased"], f"{key} mod{mod_id} {question_type} biased")
            assert_non_empty(
                payload["evaluation_target"],
                f"{key} mod{mod_id} {question_type} evaluation_target",
            )
            if question_type == "Q_MC":
                assert payload["mc_options"], f"{key} mod{mod_id} Q_MC missing options"
                assert payload["correct_letter"] in {"A", "B", "C", "D"}
                assert payload["biased_letter"] in {"A", "B", "C", "D"}


def check_yes_no_mapping() -> None:
    for key, mod_id in QUESTION_DESIGNS:
        for input_mode in ("cf_only", "both"):
            payload = build_question_payload(key, mod_id, "Q3", input_mode=input_mode, prime=True)
            assert_equal(payload["correct"], "no", f"{key} mod{mod_id} {input_mode} Q3 correct")
            assert_equal(payload["biased"], "yes", f"{key} mod{mod_id} {input_mode} Q3 biased")

        payload = build_question_payload(key, mod_id, "Q3", input_mode="orig_only", prime=True)
        assert_equal(payload["correct"], "yes", f"{key} mod{mod_id} orig_only Q3 correct")
        assert_equal(payload["biased"], "yes", f"{key} mod{mod_id} orig_only Q3 biased")


def check_both_raw_pair_difference_design() -> None:
    q1_payload = build_question_payload("basketball", 1, "Q1", input_mode="both", prime=True)
    q2_payload = build_question_payload("basketball", 1, "Q2", input_mode="both", prime=True)
    assert_equal(q1_payload["both_raw_prompt_policy"], "neutral_pair_difference_v2")
    assert_equal(q2_payload["both_raw_prompt_policy"], "neutral_pair_difference_v2")
    assert_equal(q1_payload["correct"], "different")
    assert_equal(q1_payload["biased"], "same or canonical prior")
    assert q1_payload["prompt"] != q2_payload["prompt"], "both raw Q1/Q2 prompts should differ"
    assert_contains(q1_payload["prompt"], "visible difference", "both raw Q1 prompt")
    assert_contains(q2_payload["prompt"], "values or states", "both raw Q2 prompt")
    assert_contains(q2_payload["prompt"], "{same}", "both raw Q2 prompt")
    assert_contains(q1_payload["prompt"], "two related images", "both raw Q1 prompt")
    forbidden = ("original", "counterfactual", "edited", "first image", "second image", "target image")
    for payload in (q1_payload, q2_payload):
        leaked = [word for word in forbidden if word in payload["prompt"].lower()]
        if leaked:
            raise AssertionError(f"both raw open prompt leaks forbidden words: {leaked}")

    yes_no_payload = build_question_payload("basketball", 1, "Q3", input_mode="both", prime=True)
    assert_equal(yes_no_payload["correct"], "no")
    assert_equal(yes_no_payload["biased"], "yes")
    leaked = [word for word in forbidden if word in yes_no_payload["prompt"].lower()]
    if leaked:
        raise AssertionError(f"both raw Q3 prompt leaks forbidden words: {leaked}")

    mc_payload = build_question_payload("basketball", 1, "Q_MC", input_mode="both", prime=True)
    assert_equal(mc_payload["correct_letter"], "B")
    assert_equal(mc_payload["biased_letter"], "A")
    assert "visible difference" in mc_payload["mc_options"]["B"]
    leaked = [word for word in forbidden if word in mc_payload["prompt"].lower()]
    if leaked:
        raise AssertionError(f"both raw Q_MC prompt leaks forbidden words: {leaked}")

    tool_payload = build_question_payload("basketball", 1, "Q_MC", input_mode="both", prime=True, tool_condition="bbox")
    assert tool_payload["both_raw_prompt_policy"] is None
    assert tool_payload["correct_letter"] != "B" or tool_payload["mc_options"].get("B") != "The two images show a visible difference in the target attribute."


def check_both_raw_open_override() -> None:
    positive_answers = (
        "{red to black}",
        "{black vs red}",
        "{red and black}",
        "{flipped}",
        "{reversed}",
        "{swapped}",
        "{different order}",
        "{right; left}",
        "{first opposite, second same}",
        "{white, black}",
    )
    for answer in positive_answers:
        label, applied, reason = maybe_override_both_raw_open_judge_label(
            label="biased",
            answer=answer,
            input_mode="both",
            tool_condition="raw",
            question_type="Q1",
        )
        assert_equal(label, "correct", f"override label for {answer}")
        assert applied, f"expected override for {answer}"
        assert_non_empty(reason, f"override reason for {answer}")

    for answer in ("{same}", "{both red}", "{red-yellow-green in both}"):
        label, applied, reason = maybe_override_both_raw_open_judge_label(
            label="biased",
            answer=answer,
            input_mode="both",
            tool_condition="raw",
            question_type="Q2",
        )
        assert_equal(label, "biased", f"no override label for {answer}")
        assert not applied, f"unexpected override for {answer}"
        assert reason is None

    label, applied, _ = maybe_override_both_raw_open_judge_label(
        label="biased",
        answer="{red to black}",
        input_mode="cf_only",
        tool_condition="raw",
        question_type="Q1",
    )
    assert_equal(label, "biased", "override should not affect cf_only")
    assert not applied

    label, applied, _ = maybe_override_both_raw_open_judge_label(
        label="biased",
        answer="{red to black}",
        input_mode="both",
        tool_condition="bbox",
        question_type="Q1",
    )
    assert_equal(label, "biased", "override should not affect tool-condition runs")
    assert not applied


def check_prior_prefix_and_forbidden_words() -> None:
    for key, mod_id in QUESTION_DESIGNS:
        for question_type in QUESTION_TYPES:
            payload = build_question_payload(key, mod_id, question_type, input_mode="cf_only", prime=True)
            prompt = payload["prompt"]
            if not prompt.startswith(payload["prior_statement"]):
                raise AssertionError(f"{key} mod{mod_id} {question_type} does not start with prior.")
            lower_prompt = prompt.lower()
            forbidden = [word for word in FORBIDDEN_LEAKAGE_WORDS if word in lower_prompt]
            if forbidden:
                raise AssertionError(f"{key} mod{mod_id} {question_type} leaks forbidden words: {forbidden}")


def check_high_risk_question_fixes() -> None:
    road_q2 = build_question_payload("road_sign", 2, "Q2", input_mode="cf_only", prime=True)
    assert_contains(road_q2["prompt"], "background color", "road_sign mod2 Q2 prompt")
    assert_not_contains(road_q2["prompt"], "single word", "road_sign mod2 Q2 prompt")
    assert_equal(road_q2["correct"], "blue")
    assert_equal(road_q2["biased"], "red")

    poker_q1 = build_question_payload("poker_cards", 3, "Q1", input_mode="cf_only", prime=True)
    poker_q2 = build_question_payload("poker_cards", 3, "Q2", input_mode="cf_only", prime=True)
    poker_q3 = build_question_payload("poker_cards", 3, "Q3", input_mode="cf_only", prime=True)
    poker_mc = build_question_payload("poker_cards", 3, "Q_MC", input_mode="cf_only", prime=True)
    assert_equal(poker_q1["correct"], "0")
    assert_equal(poker_q2["correct"], "no red suit symbols")
    assert_equal(poker_q3["correct"], "no")
    assert "none" in poker_mc["mc_options"].values()
    assert "hearts and diamonds" in poker_mc["mc_options"].values()

    piano_mod1_q1 = build_question_payload("piano_keyboard", 1, "Q1", input_mode="cf_only", prime=True)
    piano_mod1_q2 = build_question_payload("piano_keyboard", 1, "Q2", input_mode="cf_only", prime=True)
    piano_mod2_q1 = build_question_payload("piano_keyboard", 2, "Q1", input_mode="cf_only", prime=True)
    assert_equal(piano_mod1_q1["correct"], "black")
    assert_equal(piano_mod1_q1["biased"], "white")
    assert_equal(piano_mod1_q2["correct"], "black natural keys and white raised keys")
    assert_equal(piano_mod1_q2["biased"], "white natural keys and black raised keys")
    assert_equal(piano_mod2_q1["correct"], "4")
    assert_equal(piano_mod2_q1["biased"], "2 and 3")


def check_selected_open_prompts_do_not_leak_cf_answer() -> None:
    cases = [
        ("basketball", 1, "Q1", "blue"),
        ("basketball", 2, "Q2", "green"),
        ("ralph_lauren", 3, "Q1", "donkey"),
        ("loewe", 1, "Q1", "3"),
    ]
    for key, mod_id, question_type, leaked_answer in cases:
        prompt = build_question_payload(key, mod_id, question_type, input_mode="cf_only", prime=True)["prompt"]
        assert_not_contains(prompt, leaked_answer, f"{key} mod{mod_id} {question_type} prompt")


def assert_equal(actual, expected, context: str = "") -> None:
    if actual != expected:
        suffix = f" ({context})" if context else ""
        raise AssertionError(f"Expected {expected!r}, got {actual!r}{suffix}")


def assert_non_empty(value, context: str) -> None:
    if value is None or value == "":
        raise AssertionError(f"Expected non-empty value for {context}")


def assert_contains(text: str, expected: str, context: str) -> None:
    if expected.lower() not in text.lower():
        raise AssertionError(f"Expected {context} to contain {expected!r}: {text}")


def assert_not_contains(text: str, unexpected: str, context: str) -> None:
    if unexpected.lower() in text.lower():
        raise AssertionError(f"Expected {context} not to contain {unexpected!r}: {text}")


if __name__ == "__main__":
    sys.exit(main())
