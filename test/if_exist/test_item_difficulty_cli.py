#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]


def main() -> int:
    help_output = run(
        "item difficulty help",
        [sys.executable, "eval_code/if_exist/analyze_item_difficulty.py", "--help"],
    )
    require_all(
        "item difficulty help",
        help_output,
        ["--input-root", "--output", "--min-cf-responses", "--easy-accuracy", "--dry-run"],
    )

    with tempfile.TemporaryDirectory(prefix="if_exist_item_difficulty_") as tmp_dir:
        tmp_root = Path(tmp_dir)
        input_root = tmp_root / "raw_runs"
        output_path = tmp_root / "tables" / "by_image_difficulty.csv"
        dry_output_path = tmp_root / "tables" / "dry_run_should_not_exist.csv"
        write_synthetic_run(input_root / "synthetic_run")

        dry_output = run(
            "item difficulty dry-run",
            [
                sys.executable,
                "eval_code/if_exist/analyze_item_difficulty.py",
                "--input-root",
                str(input_root),
                "--output",
                str(dry_output_path),
                "--include-stale-records",
                "--dry-run",
            ],
        )
        require_all("item difficulty dry-run", dry_output, ["Image pairs:", "Dry run: no CSV written."])
        if dry_output_path.exists():
            raise SystemExit("Dry-run unexpectedly wrote an output CSV.")

        output = run(
            "item difficulty generation",
            [
                sys.executable,
                "eval_code/if_exist/analyze_item_difficulty.py",
                "--input-root",
                str(input_root),
                "--output",
                str(output_path),
                "--include-stale-records",
                "--top-k",
                "2",
            ],
        )
        require_all(
            "item difficulty generation",
            output,
            ["Recommendations:", "consider_remove=", "review_quality=", "Top 2 easiest samples"],
        )
        if not output_path.exists():
            raise SystemExit("Item difficulty CSV was not written.")
        rows = read_rows(output_path)
        assert_action(rows, "easy_cf.png", "consider_remove")
        assert_action(rows, "hard_cf.png", "review_quality")
        assert_action(rows, "fallback_cf.png", "consider_remove")

    print("if_exist item difficulty CLI tests passed.")
    return 0


def write_synthetic_run(run_dir: Path) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    records.extend(image_records("easy", "easy_cf.png", ["correct", "correct", "correct"], include_input_mode=True))
    records.extend(image_records("hard", "hard_cf.png", ["biased", "biased", "other"], include_input_mode=True))
    records.extend(image_records("fallback", "fallback_cf.png", ["correct", "correct", "correct"], include_input_mode=False))
    records.extend(image_records("easy", "easy_cf.png", ["correct"], include_input_mode=True, input_mode="both"))
    records.extend(image_records("easy", "easy_cf.png", ["correct"], include_input_mode=True, input_mode="orig_only"))
    records.append(
        synthetic_record(
            stem="easy",
            edited_name="easy_cf.png",
            scored_label="error",
            question_index=99,
            include_input_mode=True,
        )
    )
    with (run_dir / "records.jsonl").open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")
    (run_dir / "run_config.json").write_text(
        json.dumps({"run_id": "synthetic_run", "script_type": "closed"}, indent=2),
        encoding="utf-8",
    )


def image_records(
    stem: str,
    edited_name: str,
    labels: list[str],
    *,
    include_input_mode: bool,
    input_mode: str = "cf_only",
) -> list[dict[str, Any]]:
    return [
        synthetic_record(
            stem=stem,
            edited_name=edited_name,
            scored_label=label,
            question_index=index,
            include_input_mode=include_input_mode,
            input_mode=input_mode,
        )
        for index, label in enumerate(labels, start=1)
    ]


def synthetic_record(
    *,
    stem: str,
    edited_name: str,
    scored_label: str,
    question_index: int,
    include_input_mode: bool,
    input_mode: str = "cf_only",
) -> dict[str, Any]:
    question_types = ["yes_no", "multiple_choice", "open"]
    record = {
        "run_id": "synthetic_run",
        "script_type": "closed",
        "backbone_model_alias": "gpt-4.1",
        "model": "gpt-4.1",
        "eval_mode": input_mode,
        "category": "if_exist",
        "domain": "if_exist",
        "subcategory": "camel",
        "key": "camel",
        "source_variant": "real",
        "source": "real",
        "original_image_path": f"/tmp/{stem}.png",
        "edited_image_path": f"/tmp/{edited_name}",
        "question_id": f"{stem}_q{question_index}",
        "question_type": question_types[(question_index - 1) % len(question_types)],
        "scored_label": scored_label,
    }
    if include_input_mode:
        record["input_mode"] = input_mode
    return record


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def assert_action(rows: list[dict[str, str]], edited_image: str, expected_action: str) -> None:
    matches = [row for row in rows if row["Edited_Image"] == edited_image]
    if not matches:
        raise SystemExit(f"No row found for edited image {edited_image}")
    actual = matches[0]["Recommended_Action"]
    if actual != expected_action:
        raise SystemExit(f"{edited_image} expected {expected_action}, got {actual}")


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
