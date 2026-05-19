#!/usr/bin/env python3
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
REQUIRED_FIELDS = {
    "category",
    "subcategory",
    "display_name",
    "source",
    "original_image",
    "cf_filename",
    "original_path",
    "cf_path",
    "edit_type",
    "target_visual_cue",
    "bias_axis",
    "answers",
    "question_ids",
}


def main() -> int:
    dry_output = run(
        "metadata dry-run",
        [
            sys.executable,
            "eval_code/if_exist/build_metadata.py",
            "--dry-run",
        ],
    )
    require_all("metadata dry-run", dry_output, ["metadata_records=", "category", "cf_filename"])

    with tempfile.TemporaryDirectory(prefix="if_exist_metadata_") as tmp_dir:
        output_path = Path(tmp_dir) / "metadata.json"
        run(
            "metadata write",
            [
                sys.executable,
                "eval_code/if_exist/build_metadata.py",
                "--output",
                str(output_path),
            ],
        )
        if not output_path.exists():
            raise SystemExit("Metadata output file was not written.")
        records = json.loads(output_path.read_text(encoding="utf-8"))
        validate_records(records)

    print("if_exist metadata CLI tests passed.")
    return 0


def validate_records(records: Any) -> None:
    if not isinstance(records, list) or not records:
        raise SystemExit("Metadata output should be a non-empty list.")
    first = records[0]
    missing = REQUIRED_FIELDS - set(first)
    if missing:
        raise SystemExit(f"Metadata record missing fields: {sorted(missing)}")
    for path_key in ("original_path", "cf_path"):
        if not Path(first[path_key]).exists():
            raise SystemExit(f"Metadata path does not exist: {first[path_key]}")
    answers = first["answers"]
    for mode in ("cf_only", "both", "orig_only"):
        if mode not in answers:
            raise SystemExit(f"Metadata answers missing mode: {mode}")
        if set(answers[mode]) != {"correct", "biased"}:
            raise SystemExit(f"Metadata answers for {mode} should contain correct and biased.")
    if answers["orig_only"]["biased"] is not None:
        raise SystemExit("Metadata orig_only biased answer should be null.")


def run(name: str, command: list[str]) -> str:
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
    return output


def require_all(name: str, text: str, expected: list[str]) -> None:
    missing = [item for item in expected if item not in text]
    if missing:
        raise SystemExit(f"{name} missing expected fragments: {missing}")


if __name__ == "__main__":
    sys.exit(main())
