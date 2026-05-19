#!/usr/bin/env python3
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="if_exist_analysis_") as tmp_dir:
        tmp_root = Path(tmp_dir)
        input_root = tmp_root / "raw_runs"
        output_root = tmp_root / "tables"
        write_synthetic_run(input_root / "synthetic_run")

        output = run(
            "analysis table generation",
            [
                sys.executable,
                "eval_code/if_exist/analyze_results.py",
                "--input-root",
                str(input_root),
                "--output-root",
                str(output_root),
            ],
        )
        require_all(
            "analysis table generation",
            output,
            [
                "Loaded",
                "full_matrix.csv",
                "by_model.csv",
                "by_input_mode.csv",
                "by_eval_mode.csv",
                "by_tool_condition.csv",
                "by_tool_condition_question_type.csv",
                "by_tool_condition_input_mode.csv",
                "by_question_type.csv",
                "by_question_type_input_mode.csv",
                "by_category.csv",
                "by_subcategory.csv",
                "by_edit_type.csv",
                "by_source_variant.csv",
            ],
        )
        required_tables = [
            "full_matrix.csv",
            "by_model.csv",
            "by_input_mode.csv",
            "by_eval_mode.csv",
            "by_tool_condition.csv",
            "by_tool_condition_question_type.csv",
            "by_tool_condition_input_mode.csv",
            "by_question_type.csv",
            "by_question_type_input_mode.csv",
            "by_category.csv",
            "by_subcategory.csv",
            "by_edit_type.csv",
            "by_source_variant.csv",
        ]
        missing = [name for name in required_tables if not (output_root / name).exists()]
        if missing:
            raise SystemExit(f"Analysis did not write expected tables: {missing}")
        full_matrix_text = (output_root / "full_matrix.csv").read_text(encoding="utf-8")
        require_all(
            "full_matrix columns",
            full_matrix_text,
            [
                "Input_Mode",
                "Eval_Mode",
                "Tool_Condition",
                "Evidence_QC_Filter",
                "Missing_Evidence_Policy",
                "Delta_Accuracy_vs_Raw",
                "Delta_Bias_vs_Raw",
            ],
        )

        figures_root = tmp_root / "figures"
        figure_output = run(
            "figure generation",
            [
                sys.executable,
                "eval_code/if_exist/make_figures.py",
                "--tables-root",
                str(output_root),
                "--figures-root",
                str(figures_root),
            ],
        )
        require_all("figure generation", figure_output, ["Figures written"])
        required_figures = [
            "fig1_cf_only_by_qtype.png",
            "fig1_cf_only_by_question_type.png",
            "fig2_input_mode_comparison.png",
            "fig3_bias_by_edit_type.png",
            "fig4_category_heatmap.png",
            "fig4_subcategory_heatmap.png",
            "fig5_mc_vs_open.png",
            "fig5_source_variant_comparison.png",
            "fig6_model_comparison.png",
            "fig_tool_condition_overall.png",
            "fig_tool_condition_by_question_type.png",
            "fig_tool_condition_by_input_mode.png",
            "fig_bias_reduction_vs_raw.png",
        ]
        missing_figures = [name for name in required_figures if not (figures_root / name).exists()]
        if missing_figures:
            raise SystemExit(f"Figure generation did not write expected files: {missing_figures}")

    with tempfile.TemporaryDirectory(prefix="if_exist_stale_filter_") as tmp_dir:
        tmp_root = Path(tmp_dir)
        input_root = tmp_root / "raw_runs"
        write_stale_filter_run(input_root / "stale_filter_run")
        output = run(
            "analysis stale-record filtering",
            [
                sys.executable,
                "eval_code/if_exist/analyze_results.py",
                "--input-root",
                str(input_root),
                "--output-root",
                str(tmp_root / "tables"),
                "--dry-run",
            ],
        )
        require_all(
            "analysis stale-record filtering",
            output,
            ["Dataset filter:", "kept=1", "dropped_stale=1"],
        )

    print("if_exist analysis CLI tests passed.")
    return 0


def write_synthetic_run(run_dir: Path) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    records = [
        synthetic_record("yes_no", "correct", "if_exist", "camel", "real"),
        synthetic_record("yes_no", "biased", "if_exist", "camel", "real", tool_condition="crop"),
        synthetic_record("multiple_choice", "biased", "if_exist", "camel", "ai"),
        synthetic_record("open", "other", "if_exist", "rabbit", "real"),
        synthetic_record("open", "error", "if_exist", "rabbit", "ai"),
    ]
    with (run_dir / "records.jsonl").open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")
    (run_dir / "run_config.json").write_text(
        json.dumps(
            {
                "run_id": "synthetic_run",
                "script_type": "closed",
                "eval_mode": "cf_only",
                "input_mode": "cf_only",
                "backbone_model_alias": "gpt-4.1",
                "judge_model_alias": "gpt-4o-mini",
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def write_stale_filter_run(run_dir: Path) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    valid_original = REPO_ROOT / "dataset" / "if_exist" / "camel" / "camel_18.png"
    valid_cf = REPO_ROOT / "cf_dataset" / "if_exist_cf" / "camel" / "camel_18_remove_hump.png"
    if not valid_original.exists() or not valid_cf.exists():
        raise SystemExit("Expected camel fixture images are missing.")
    records = [
        {
            **synthetic_record("yes_no", "correct", "if_exist", "camel", "real"),
            "original_image_path": str(valid_original),
            "edited_image_path": str(valid_cf),
        },
        {
            **synthetic_record("yes_no", "biased", "if_exist", "camel", "real"),
            "original_image_path": "/tmp/deleted_original.png",
            "edited_image_path": "/tmp/deleted_cf.png",
        },
    ]
    with (run_dir / "records.jsonl").open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")
    (run_dir / "run_config.json").write_text(
        json.dumps({"run_id": "stale_filter_run", "script_type": "closed"}, indent=2),
        encoding="utf-8",
    )


def synthetic_record(
    question_type: str,
    scored_label: str,
    category: str,
    subcategory: str,
    source_variant: str,
    *,
    tool_condition: str = "raw",
) -> dict[str, str]:
    return {
        "run_id": "synthetic_run",
        "script_type": "closed",
        "backbone_model_alias": "gpt-4.1",
        "backbone_model_id": "gpt-4.1",
        "judge_model_alias": "gpt-4o-mini",
        "judge_model_id": "gpt-4o-mini",
        "eval_mode": "cf_only",
        "input_mode": "cf_only",
        "category": category,
        "domain": category,
        "subcategory": subcategory,
        "key": subcategory,
        "source_variant": source_variant,
        "source": source_variant,
        "tool_condition": tool_condition,
        "edit_type": "remove_feature",
        "question_id": f"synthetic_{question_type}",
        "question_type": question_type,
        "scored_label": scored_label,
        "judge_label": scored_label if question_type in {"yes_no", "open"} else None,
        "judge_model": "gpt-4o-mini" if question_type in {"yes_no", "open"} else "letter_match",
    }


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


def require_all(name: str, output: str, expected: list[str]) -> None:
    missing = [item for item in expected if item not in output]
    if missing:
        raise SystemExit(f"{name} missing expected output fragments: {missing}")


if __name__ == "__main__":
    sys.exit(main())
