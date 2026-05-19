#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = REPO_ROOT / "eval_code/counting/rejudge_open_records.py"


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
                "rejudge_open_records.py",
                "--input-run",
                str(input_run),
                "--output-run",
                str(dry_output),
                "--dry-run",
            ],
        )
        if dry_output.exists():
            raise SystemExit("Dry run should not create output directory.")

        install_fake_judge(module)
        run_main(
            module,
            [
                "rejudge_open_records.py",
                "--input-run",
                str(input_run),
                "--output-run",
                str(output_run),
                "--judge-model",
                "gpt-4o-mini",
            ],
        )
        output_records = read_jsonl(output_run / "records.jsonl")
        if len(output_records) != len(records):
            raise SystemExit("Rejudged output record count changed.")
        if output_records[0]["scored_label"] != "correct" or output_records[0]["judge_label"] != "correct":
            raise SystemExit("Open record was not rejudged to the mocked label.")
        if not output_records[0].get("rejudged_open"):
            raise SystemExit("Open record missing rejudged_open marker.")
        if output_records[1] != records[1]:
            raise SystemExit("Non-open record should be copied exactly.")
        summary = json.loads((output_run / "summary.json").read_text(encoding="utf-8"))
        if summary["overall"]["correct_count"] != 2:
            raise SystemExit(f"Summary was not recomputed from rejudged records: {summary['overall']}")
        config = json.loads((output_run / "run_config.json").read_text(encoding="utf-8"))
        if config.get("rejudge_question_types") != ["open"]:
            raise SystemExit("Run config missing rejudge metadata.")
        filtered_output = root / "filtered_output"
        run_main(
            module,
            [
                "rejudge_open_records.py",
                "--input-run",
                str(input_run),
                "--output-run",
                str(filtered_output),
                "--judge-model",
                "gpt-4o-mini",
                "--input-mode",
                "orig_only",
            ],
        )
        filtered_records = read_jsonl(filtered_output / "records.jsonl")
        if filtered_records != records:
            raise SystemExit("Input-mode filter should copy non-matching records without rejudging them.")
        filtered_config = json.loads((filtered_output / "run_config.json").read_text(encoding="utf-8"))
        if filtered_config.get("rejudge_input_mode") != "orig_only":
            raise SystemExit("Run config missing rejudge_input_mode metadata.")
    print("counting rejudge open records tests passed.")
    return 0


def synthetic_records() -> list[dict]:
    base = {
        "run_id": "synthetic",
        "script_type": "closed",
        "backbone_model_alias": "gpt-4.1",
        "backbone_model_id": "gpt-4.1",
        "judge_model_alias": "gpt-4o-mini",
        "judge_model_id": "gpt-4o-mini",
        "eval_mode": "cf_only",
        "input_mode": "cf_only",
        "category": "counting",
        "subset": "bird",
        "subcategory": "accipitriformes",
        "source_variant": "real",
        "edit_type": "wings_add",
        "count_direction": "more_than_expected",
        "original_image_path": "/tmp/original.png",
        "edited_image_path": "/tmp/cf.png",
        "used_image_paths": ["/tmp/cf.png"],
        "prompt": "Birds are widely known for having one normal pair of wings.",
        "raw_response_text": "The visible count contradicts the expected count because four wings are visible.",
        "parsed_response": None,
        "ground_truth_target": {},
        "biased_target": {},
        "error_message": None,
        "error_kind": None,
        "fatal_error": False,
    }
    open_record = dict(
        base,
        question_id="counting_bird_accipitriformes_wings_add_open",
        question_type="open",
        scored_label="other",
        judge_label="other",
        judge_raw_output='{"label":"other","reason":"old"}',
        judge_reason="old",
    )
    mc_record = dict(
        base,
        question_id="counting_bird_accipitriformes_wings_add_multiple_choice",
        question_type="multiple_choice",
        raw_response_text="B",
        parsed_response="B",
        scored_label="correct",
        judge_label=None,
        judge_raw_output=None,
        judge_reason=None,
    )
    return [open_record, mc_record]


def import_script():
    sys.path.insert(0, str(SCRIPT_PATH.parent))
    spec = importlib.util.spec_from_file_location("counting_rejudge_open_records_for_test", SCRIPT_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {SCRIPT_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def install_fake_judge(module) -> None:
    module.resolve_closed_model_ref = lambda raw: SimpleNamespace(alias=raw, provider="openai", model_id=raw)
    module.validate_provider_envs = lambda providers: None

    class FakeAdapter:
        last_retry_count = 0

        def __init__(self, judge_ref) -> None:
            self.judge_ref = judge_ref

        def generate(self, *args, **kwargs) -> str:
            return '{"label":"correct","reason":"mocked visible counting evidence."}'

    module.ClosedProviderAdapter = FakeAdapter


def run_main(module, argv: list[str]) -> None:
    old_argv = sys.argv
    try:
        sys.argv = argv
        code = module.main()
    finally:
        sys.argv = old_argv
    if code != 0:
        raise SystemExit(f"rejudge main failed with exit code {code}")


def write_jsonl(path: Path, records: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(record) for record in records) + "\n", encoding="utf-8")


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


if __name__ == "__main__":
    raise SystemExit(main())
