#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import subprocess
import sys
import tempfile
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="counting_difficulty_") as tmp:
        root = Path(tmp)
        run_dir = root / "raw_runs" / "synthetic"
        output_csv = root / "by_image_difficulty.csv"
        run_dir.mkdir(parents=True)
        write_records(run_dir / "records.jsonl")

        dry = run(
            "difficulty dry-run",
            [
                sys.executable,
                "eval_code/counting/analyze_item_difficulty.py",
                "--input-root",
                str(root / "raw_runs"),
                "--output",
                str(output_csv),
                "--include-stale-records",
                "--min-cf-responses",
                "3",
                "--dry-run",
            ],
            ["Dry run: no CSV written."],
        )
        if output_csv.exists():
            raise SystemExit("Dry-run should not write item difficulty CSV.")

        run(
            "difficulty write",
            [
                sys.executable,
                "eval_code/counting/analyze_item_difficulty.py",
                "--input-root",
                str(root / "raw_runs"),
                "--output",
                str(output_csv),
                "--include-stale-records",
                "--min-cf-responses",
                "3",
            ],
            ["Wrote"],
        )
        rows = list(csv.DictReader(output_csv.open(encoding="utf-8")))
        actions = {row["Edited_Image"]: row["Recommended_Action"] for row in rows}
        if actions.get("easy.png") != "consider_remove":
            raise SystemExit(f"easy sample should be consider_remove: {actions}")
        if actions.get("hard.png") != "review_quality":
            raise SystemExit(f"hard sample should be review_quality: {actions}")

    print("counting item difficulty CLI tests passed.")
    return 0


def write_records(path: Path) -> None:
    rows = []
    rows.extend(sample_records(edited="easy.png", labels=["correct", "correct", "correct"]))
    rows.extend(sample_records(edited="hard.png", labels=["biased", "biased", "biased"]))
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def sample_records(*, edited: str, labels: list[str]) -> list[dict]:
    rows = []
    for idx, label in enumerate(labels, start=1):
        rows.append(
            {
                "run_id": "synthetic",
                "script_type": "closed",
                "backbone_model_alias": "gpt-4.1",
                "eval_mode": "cf_only",
                "input_mode": "cf_only",
                "category": "counting",
                "subset": "bird",
                "subcategory": "accipitriformes",
                "source_variant": "real",
                "edit_type": "wings_add",
                "count_direction": "more_than_expected",
                "question_id": f"{edited}_{idx}",
                "question_type": ("yes_no", "multiple_choice", "open")[idx - 1],
                "original_image_path": "/tmp/original.png",
                "edited_image_path": f"/tmp/{edited}",
                "cf_image_path": f"/tmp/{edited}",
                "scored_label": label,
                "error_message": None,
            }
        )
    return rows


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
    raise SystemExit(main())
