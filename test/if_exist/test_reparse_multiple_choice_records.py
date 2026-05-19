#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = REPO_ROOT / "eval_code/if_exist/reparse_multiple_choice_records.py"


def main() -> int:
    module = import_script()
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        input_run = root / "input_run"
        output_run = root / "output_run"
        input_run.mkdir()
        records = synthetic_records()
        write_jsonl(input_run / "records.jsonl", records)
        (input_run / "run_config.json").write_text(json.dumps({"run_id": "synthetic"}, indent=2), encoding="utf-8")

        dry_output = root / "dry_output"
        run_main(
            module,
            [
                "reparse_multiple_choice_records.py",
                "--input-run",
                str(input_run),
                "--output-run",
                str(dry_output),
                "--dry-run",
            ],
        )
        if dry_output.exists():
            raise SystemExit("Dry run should not create output directory.")

        run_main(
            module,
            [
                "reparse_multiple_choice_records.py",
                "--input-run",
                str(input_run),
                "--output-run",
                str(output_run),
            ],
        )
        output_records = read_jsonl(output_run / "records.jsonl")
        if len(output_records) != len(records):
            raise SystemExit("Reparsed output record count changed.")
        if output_records[0]["parsed_response"] is not None or output_records[0]["scored_label"] != "other":
            raise SystemExit(f"Strict parser did not reject prose-only MC answer: {output_records[0]}")
        if output_records[1]["parsed_response"] != "B" or output_records[1]["scored_label"] != "correct":
            raise SystemExit(f"Strict parser did not preserve explicit MC answer: {output_records[1]}")
        if not output_records[0].get("reparsed_multiple_choice"):
            raise SystemExit("MC record missing reparsed_multiple_choice marker.")
        if output_records[2] != records[2]:
            raise SystemExit("Non-MC record should be copied exactly.")
        summary = json.loads((output_run / "summary.json").read_text(encoding="utf-8"))
        if summary["by_question_type"]["multiple_choice"]["other_count"] != 1:
            raise SystemExit(f"Summary was not recomputed from reparsed records: {summary['by_question_type']}")
        config = json.loads((output_run / "run_config.json").read_text(encoding="utf-8"))
        if config.get("reparse_question_types") != ["multiple_choice"]:
            raise SystemExit("Run config missing reparse metadata.")
    print("if_exist reparse multiple-choice records tests passed.")
    return 0


def synthetic_records() -> list[dict]:
    base = {
        "run_id": "synthetic",
        "script_type": "closed",
        "backbone_model_alias": "sonnet-4",
        "backbone_model_id": "claude-sonnet-4-20250514",
        "judge_model_alias": "gpt-4o-mini",
        "judge_model_id": "gpt-4o-mini",
        "eval_mode": "cf_only",
        "input_mode": "cf_only",
        "tool_condition": "crop",
        "category": "if_exist",
        "domain": "if_exist",
        "subcategory": "camel",
        "source_variant": "real",
        "source": "real",
        "original_image_path": "/tmp/original.png",
        "edited_image_path": "/tmp/cf.png",
        "used_image_paths": ["/tmp/cf.png"],
        "prompt": "Which option best describes the image?",
        "ground_truth_target": {"option_id": "B", "option_text": "deviates"},
        "biased_target": {"option_id": "A", "option_text": "fits"},
        "error_message": None,
        "error_kind": None,
        "fatal_error": False,
    }
    prose_mc = dict(
        base,
        question_id="if_exist_camel_multiple_choice_prose",
        question_type="multiple_choice",
        raw_response_text="Looking at the second image group, I can see a camel in profile view.",
        parsed_response="A",
        scored_label="biased",
        judge_model="letter_match",
        judge_label=None,
        judge_raw_output=None,
        judge_reason=None,
    )
    explicit_mc = dict(
        base,
        question_id="if_exist_camel_multiple_choice_explicit",
        question_type="multiple_choice",
        raw_response_text="Answer: B",
        parsed_response="B",
        scored_label="correct",
        judge_model="letter_match",
        judge_label=None,
        judge_raw_output=None,
        judge_reason=None,
    )
    yes_no = dict(
        base,
        question_id="if_exist_camel_yes_no",
        question_type="yes_no",
        raw_response_text="No",
        parsed_response="no",
        scored_label="correct",
        judge_label="correct",
        judge_raw_output='{"label":"correct","reason":"mock"}',
        judge_reason="mock",
    )
    return [prose_mc, explicit_mc, yes_no]


def import_script():
    sys.path.insert(0, str(SCRIPT_PATH.parent))
    spec = importlib.util.spec_from_file_location("if_exist_reparse_multiple_choice_records_for_test", SCRIPT_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {SCRIPT_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def run_main(module, argv: list[str]) -> None:
    old_argv = sys.argv
    try:
        sys.argv = argv
        code = module.main()
    finally:
        sys.argv = old_argv
    if code != 0:
        raise SystemExit(f"reparse main failed with exit code {code}")


def write_jsonl(path: Path, records: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(record) for record in records) + "\n", encoding="utf-8")


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


if __name__ == "__main__":
    raise SystemExit(main())
