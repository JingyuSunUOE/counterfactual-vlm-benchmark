#!/usr/bin/env python3
from __future__ import annotations

import csv
import subprocess
import sys
import tempfile
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = "eval_code/if_exist/build_visual_qc_gallery.py"
REQUIRED_COLUMNS = {
    "category",
    "subcategory",
    "source_variant",
    "original_path",
    "cf_path",
    "question_risk_note",
    "suggested_review_status",
}


def main() -> int:
    help_output = run([sys.executable, SCRIPT, "--help"])
    for fragment in ["--output-root", "--category", "--subcategory", "--source-variant", "--dry-run"]:
        if fragment not in help_output:
            raise SystemExit(f"QC gallery help missing {fragment}")

    with tempfile.TemporaryDirectory() as tmpdir:
        output_root = Path(tmpdir) / "visual_qc"
        dry_output = run(
            [
                sys.executable,
                SCRIPT,
                "--category",
                "if_exist",
                "--subcategory",
                "camel",
                "--source-variant",
                "real",
                "--max-pairs",
                "2",
                "--output-root",
                str(output_root),
                "--dry-run",
            ]
        )
        if "dry_run=true" not in dry_output or output_root.exists():
            raise SystemExit("QC gallery dry-run should print dry_run=true and avoid writing files.")

        run(
            [
                sys.executable,
                SCRIPT,
                "--category",
                "if_exist",
                "--subcategory",
                "rabbit",
                "--source-variant",
                "real",
                "--max-pairs",
                "2",
                "--output-root",
                str(output_root),
            ]
        )
        csv_path = output_root / "visual_qc_pairs.csv"
        sheet_path = output_root / "visual_qc_contact_sheet.jpg"
        if not csv_path.exists() or not sheet_path.exists():
            raise SystemExit("QC gallery did not create the expected CSV and contact sheet.")
        with csv_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            rows = list(reader)
        if not rows:
            raise SystemExit("QC gallery CSV should contain at least one row.")
        missing_columns = REQUIRED_COLUMNS - set(rows[0])
        if missing_columns:
            raise SystemExit(f"QC gallery CSV missing columns: {sorted(missing_columns)}")
        for row in rows:
            if row["suggested_review_status"] not in {"keep", "review", "exclude_candidate"}:
                raise SystemExit(f"Unexpected suggested_review_status: {row['suggested_review_status']}")
            if not Path(row["original_path"]).exists() or not Path(row["cf_path"]).exists():
                raise SystemExit("QC gallery CSV should reference existing original and CF paths.")

    print("if_exist visual QC gallery CLI tests passed.")
    return 0


def run(command: list[str]) -> str:
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
        raise SystemExit(f"Command failed with exit code {completed.returncode}: {' '.join(command)}")
    return output


if __name__ == "__main__":
    sys.exit(main())
