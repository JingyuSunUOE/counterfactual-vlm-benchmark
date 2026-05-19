#!/usr/bin/env python3
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="counting_analysis_") as tmp:
        root = Path(tmp)
        run_dir = root / "raw_runs" / "synthetic"
        tables_dir = root / "tables"
        figures_dir = root / "figures"
        run_dir.mkdir(parents=True)
        write_records(run_dir / "records.jsonl")

        run(
            "analysis",
            [
                sys.executable,
                "eval_code/counting/analyze_results.py",
                "--input-root",
                str(root / "raw_runs"),
                "--output-root",
                str(tables_dir),
                "--include-stale-records",
            ],
            ["by_subset.csv", "by_edit_type.csv", "by_count_direction.csv"],
        )
        for filename in (
            "full_matrix.csv",
            "by_subset.csv",
            "by_edit_type.csv",
            "by_count_direction.csv",
            "by_tool_condition.csv",
            "by_tool_condition_question_type.csv",
            "by_tool_condition_input_mode.csv",
        ):
            if not (tables_dir / filename).exists():
                raise SystemExit(f"Missing expected table: {filename}")
        full_matrix = (tables_dir / "full_matrix.csv").read_text(encoding="utf-8")
        for fragment in ("Tool_Condition", "Evidence_QC_Filter", "Missing_Evidence_Policy", "Delta_Accuracy_vs_Raw"):
            if fragment not in full_matrix:
                raise SystemExit(f"full_matrix.csv missing {fragment}")

        run(
            "figures",
            [
                sys.executable,
                "eval_code/counting/make_figures.py",
                "--tables-root",
                str(tables_dir),
                "--figures-root",
                str(figures_dir),
            ],
            ["Figures written"],
        )
        for filename in (
            "fig1_cf_only_by_qtype.png",
            "fig2_input_mode_comparison.png",
            "fig3_bias_by_subset.png",
            "fig4_edit_type_comparison.png",
            "fig5_more_vs_fewer.png",
            "fig6_subcategory_heatmap.png",
        ):
            if not (figures_dir / filename).exists():
                raise SystemExit(f"Missing expected figure: {filename}")

    print("counting analysis CLI tests passed.")
    return 0


def write_records(path: Path) -> None:
    rows = []
    for idx, label in enumerate(("correct", "biased", "other", "correct"), start=1):
        tool_condition = "raw" if idx <= 2 else "crop"
        rows.append(
            {
                "run_id": "synthetic",
                "script_type": "closed",
                "backbone_model_alias": "gpt-4.1",
                "backbone_model_id": "gpt-4.1",
                "judge_model_alias": "gpt-4o-mini",
                "judge_model_id": "gpt-4o-mini",
                "eval_mode": "cf_only",
                "input_mode": "cf_only",
                "category": "counting",
                "subset": "bird" if idx < 4 else "insect",
                "subcategory": "accipitriformes" if idx < 4 else "diptera",
                "source_variant": "real",
                "edit_type": "wings_add" if idx < 4 else "wings_remove",
                "count_direction": "more_than_expected" if idx < 4 else "fewer_than_expected",
                "question_group_id": "synthetic_group",
                "annotation_pair_key": f"synthetic_{idx}",
                "question_id": f"q{idx}",
                "question_type": ("yes_no", "multiple_choice", "open", "multiple_choice")[idx - 1],
                "original_image_path": f"/tmp/original_{idx}.png",
                "edited_image_path": f"/tmp/cf_{idx}.png",
                "cf_image_path": f"/tmp/cf_{idx}.png",
                "used_image_paths": [f"/tmp/cf_{idx}.png"],
                "tool_condition": tool_condition,
                "evidence_qc_filter": "pass",
                "missing_evidence_policy": "skip",
                "prompt": "Prompt",
                "raw_response_text": "Answer",
                "parsed_response": "A",
                "ground_truth_target": {},
                "biased_target": {},
                "scored_label": label,
                "judge_raw_output": None,
                "judge_label": None,
                "judge_reason": None,
                "latency_ms": 1,
                "error_message": None,
            }
        )
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


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
