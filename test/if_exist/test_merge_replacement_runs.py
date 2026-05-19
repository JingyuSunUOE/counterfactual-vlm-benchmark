#!/usr/bin/env python3
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = "eval_code/if_exist/merge_replacement_runs.py"


def main() -> int:
    help_output = run([sys.executable, SCRIPT, "--help"])
    for fragment in ["--base-run", "--replacement-run", "--replace-question-types", "--output-run"]:
        if fragment not in help_output:
            raise SystemExit(f"merge help missing {fragment}")

    with tempfile.TemporaryDirectory(prefix="if_exist_merge_") as tmp_dir:
        tmp_root = Path(tmp_dir)
        base_run = tmp_root / "base"
        replacement_run = tmp_root / "replacement"
        output_run = tmp_root / "merged"
        write_run(base_run, [record("yes_no", "old-yes"), record("multiple_choice", "old-mc"), record("open", "old-open")])
        write_run(replacement_run, [record("multiple_choice", "new-mc")], question_types=["multiple_choice"])

        run(
            [
                sys.executable,
                SCRIPT,
                "--base-run",
                str(base_run),
                "--replacement-run",
                str(replacement_run),
                "--replace-question-types",
                "multiple_choice",
                "--output-run",
                str(output_run),
            ]
        )

        records = [json.loads(line) for line in (output_run / "records.jsonl").read_text(encoding="utf-8").splitlines()]
        raw_texts = [record["raw_response_text"] for record in records]
        if raw_texts != ["old-yes", "new-mc", "old-open"]:
            raise SystemExit(f"Unexpected merged record order/content: {raw_texts}")
        if {record["run_id"] for record in records} != {"merged"}:
            raise SystemExit("Merged records should use the output run id.")
        summary = json.loads((output_run / "summary.json").read_text(encoding="utf-8"))
        if summary["overall"]["response_count"] != 3:
            raise SystemExit("Merged summary should count three records.")
        config = json.loads((output_run / "run_config.json").read_text(encoding="utf-8"))
        if config["merge"]["replacement_used_count"] != 1:
            raise SystemExit("Merged run_config should record one replacement.")

    print("if_exist merge replacement run tests passed.")
    return 0


def write_run(path: Path, records: list[dict[str, Any]], *, question_types: list[str] | None = None) -> None:
    path.mkdir(parents=True, exist_ok=True)
    with (path / "records.jsonl").open("w", encoding="utf-8") as handle:
        for item in records:
            handle.write(json.dumps(item) + "\n")
    (path / "run_config.json").write_text(
        json.dumps(
            {
                "run_id": path.name,
                "script_type": "closed",
                "questions_path": str(REPO_ROOT / "cf_dataset" / "if_exist_cf_questions.json"),
                "eval_mode": "cf_only",
                "input_mode": "cf_only",
                "backbone_model_alias": "gpt-4.1",
                "backbone_model_id": "gpt-4.1",
                "judge_model_alias": "gpt-4o-mini",
                "judge_model_id": "gpt-4o-mini",
                "source_variant": "both",
                "prime": "off",
                "temperature": 0.0,
                "max_output_tokens": 128,
                "question_types": question_types or ["yes_no", "multiple_choice", "open"],
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def record(question_type: str, raw_text: str) -> dict[str, Any]:
    question_id = f"if_exist_camel_{question_type}"
    return {
        "run_id": "source",
        "script_type": "closed",
        "backbone_model_alias": "gpt-4.1",
        "backbone_model_id": "gpt-4.1",
        "judge_model_alias": "gpt-4o-mini",
        "judge_model_id": "gpt-4o-mini",
        "eval_mode": "cf_only",
        "input_mode": "cf_only",
        "category": "if_exist",
        "subcategory": "camel",
        "source_variant": "real",
        "question_id": question_id,
        "question_type": question_type,
        "original_image_path": str(REPO_ROOT / "dataset" / "if_exist" / "camel" / "camel_1.png"),
        "edited_image_path": str(REPO_ROOT / "cf_dataset" / "if_exist_cf" / "camel" / "camel_1_remove_hump.png"),
        "raw_response_text": raw_text,
        "scored_label": "correct",
        "task_signature": f"cf_only|if_exist|camel|real|{question_id}|camel_1.png|camel_1_remove_hump.png",
    }


def run(command: list[str]) -> str:
    completed = subprocess.run(command, cwd=REPO_ROOT, text=True, capture_output=True, timeout=120, check=False)
    output = completed.stdout + completed.stderr
    if completed.returncode != 0:
        print(output)
        raise SystemExit(f"Command failed with exit code {completed.returncode}: {' '.join(command)}")
    return output


if __name__ == "__main__":
    sys.exit(main())
