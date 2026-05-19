#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from eval_common import DEFAULT_FIGURES_ROOT, DEFAULT_TABLES_ROOT


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create medical modality benchmark figures from analysis CSV tables.")
    parser.add_argument("--tables-root", type=Path, default=DEFAULT_TABLES_ROOT)
    parser.add_argument("--figures-root", type=Path, default=DEFAULT_FIGURES_ROOT)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.figures_root.mkdir(parents=True, exist_ok=True)
    specs = [
        ("by_question_type.csv", "Question_Type", "fig1_by_question_type.png", "Performance by Question Type"),
        ("by_input_mode.csv", "Input_Mode", "fig2_input_mode_comparison.png", "Performance by Input Mode"),
        ("by_dataset.csv", "Dataset", "fig3_dataset_comparison.png", "BraTS Dataset Comparison"),
        ("by_visual_version.csv", "Visual_Version", "fig4_clean_vs_overlay.png", "Clean vs Overlay Visualization"),
        ("by_swap_direction.csv", "Swap_Direction", "fig5_swap_direction_comparison.png", "Swap Direction Comparison"),
        ("by_model.csv", "Model", "fig6_model_comparison.png", "Model Comparison"),
    ]
    for csv_name, x_col, out_name, title in specs:
        table = args.tables_root / csv_name
        if not table.exists():
            raise SystemExit(f"Required table not found: {table}")
        df = pd.read_csv(table)
        if df.empty:
            raise SystemExit(f"Table is empty: {table}")
        print(f"{out_name}: {len(df)} rows")
        if args.dry_run:
            continue
        plot_grouped_metrics(df, x_col=x_col, title=title, output=args.figures_root / out_name)
    return 0


def plot_grouped_metrics(df: pd.DataFrame, *, x_col: str, title: str, output: Path) -> None:
    if x_col not in df.columns:
        raise SystemExit(f"Missing column {x_col} in table used for {output.name}")
    grouped = df.groupby(x_col, dropna=False)[["Accuracy%", "Bias%", "Other%"]].mean().reset_index()
    ax = grouped.plot(x=x_col, y=["Accuracy%", "Bias%", "Other%"], kind="bar", figsize=(10, 5), rot=25)
    ax.set_title(title)
    ax.set_ylabel("Rate (%)")
    ax.set_xlabel("")
    ax.set_ylim(0, 100)
    ax.legend(loc="best")
    plt.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output, dpi=180)
    plt.close()


if __name__ == "__main__":
    raise SystemExit(main())
