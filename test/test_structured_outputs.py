#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "eval_code"))

from structured_outputs import (  # noqa: E402
    JUDGE_LABEL_SCHEMA_ID,
    MC_ANSWER_SCHEMA_ID,
    YES_NO_ANSWER_SCHEMA_ID,
    anthropic_tool,
    openai_chat_response_format,
    parse_structured_judge_response,
    parse_structured_mc_answer,
    parse_structured_yes_no_answer,
    schema_for,
)


def main() -> int:
    judge_schema = schema_for(JUDGE_LABEL_SCHEMA_ID)
    if judge_schema["properties"]["label"]["enum"] != ["correct", "biased", "other"]:
        raise SystemExit("judge schema label enum regressed")
    if not judge_schema.get("additionalProperties") is False:
        raise SystemExit("judge schema must reject additional properties")

    mc_schema = schema_for(MC_ANSWER_SCHEMA_ID)
    if mc_schema["properties"]["answer"]["enum"] != ["A", "B", "C", "D"]:
        raise SystemExit("MC schema answer enum regressed")

    yes_no_schema = schema_for(YES_NO_ANSWER_SCHEMA_ID)
    if yes_no_schema["properties"]["answer"]["enum"] != ["yes", "no"]:
        raise SystemExit("yes/no schema answer enum regressed")

    response_format = openai_chat_response_format(JUDGE_LABEL_SCHEMA_ID)
    if response_format["type"] != "json_schema" or not response_format["json_schema"]["strict"]:
        raise SystemExit("OpenAI response_format builder regressed")

    tool = anthropic_tool(MC_ANSWER_SCHEMA_ID)
    if tool["input_schema"]["properties"]["answer"]["enum"] != ["A", "B", "C", "D"]:
        raise SystemExit("Anthropic tool schema builder regressed")

    parsed_judge = parse_structured_judge_response('{"label":"correct","reason":"visible cue"}')
    if parsed_judge != {"label": "correct", "reason": "visible cue"}:
        raise SystemExit(f"structured judge parse failed: {parsed_judge}")

    if parse_structured_mc_answer('{"answer":"B"}') != "B":
        raise SystemExit("structured MC answer parse failed")
    if parse_structured_mc_answer('{"answer":"E"}') is not None:
        raise SystemExit("structured MC parser accepted invalid option")
    if parse_structured_yes_no_answer('{"answer":"no"}') != "no":
        raise SystemExit("structured yes/no answer parse failed")

    print("structured output helper tests passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
