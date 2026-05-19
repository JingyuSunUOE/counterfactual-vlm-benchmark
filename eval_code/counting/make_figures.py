#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

from eval_common import DEFAULT_FIGURES_ROOT, DEFAULT_TABLES_ROOT


os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")

import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate publication-style figures from counting tables.")
    parser.add_argument("--tables-root", type=Path, default=DEFAULT_TABLES_ROOT, help="Directory containing CSV tables.")
    parser.add_argument("--figures-root", type=Path, default=DEFAULT_FIGURES_ROOT, help="Directory for PNG figures.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.figures_root.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")

    full_matrix = read_csv(args.tables_root / "full_matrix.csv")
    by_input_mode = read_csv(args.tables_root / "by_input_mode.csv")
    by_subset = read_csv(args.tables_root / "by_subset.csv")
    by_edit_type = read_csv(args.tables_root / "by_edit_type.csv")
    by_count_direction = read_csv(args.tables_root / "by_count_direction.csv")
    by_subcategory = read_csv(args.tables_root / "by_subcategory.csv")

    outputs = {
        "fig1": args.figures_root / "fig1_cf_only_by_qtype.png",
        "fig2": args.figures_root / "fig2_input_mode_comparison.png",
        "fig3": args.figures_root / "fig3_bias_by_subset.png",
        "fig4": args.figures_root / "fig4_edit_type_comparison.png",
        "fig5": args.figures_root / "fig5_more_vs_fewer.png",
        "fig6": args.figures_root / "fig6_subcategory_heatmap.png",
    }
    figure_cf_only_by_question_type(full_matrix, outputs["fig1"])
    figure_input_mode_comparison(by_input_mode, outputs["fig2"])
    figure_bias_by_subset(by_subset, outputs["fig3"])
    figure_bias_by_edit_type(by_edit_type, outputs["fig4"])
    figure_more_vs_fewer(by_count_direction, outputs["fig5"])
    figure_subcategory_heatmap(by_subcategory, outputs["fig6"])

    print(f"Figures written to {args.figures_root}")
    return 0


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise SystemExit(f"Missing table: {path}")
    df = pd.read_csv(path)
    if df.empty:
        raise SystemExit(f"Table is empty: {path}")
    return df


def save_current(path: Path) -> Path:
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()
    print(f"  wrote {path}")
    return path


def ensure_columns(df: pd.DataFrame, columns: list[str], *, figure_name: str) -> None:
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise SystemExit(f"{figure_name} missing columns: {missing}")


def require_nonempty(df: pd.DataFrame, *, figure_name: str, reason: str) -> pd.DataFrame:
    if df.empty:
        raise SystemExit(f"{figure_name} has no data: {reason}")
    return df


def metric_barplot(
    df: pd.DataFrame,
    *,
    x: str,
    title: str,
    path: Path,
    hue: str | None = "Model",
    metrics: tuple[str, ...] = ("Accuracy%", "Bias%", "Other%"),
) -> Path:
    ensure_columns(df, [x, *metrics], figure_name=title)
    id_vars = [col for col in (x, hue) if col and col in df.columns]
    melted = df.melt(
        id_vars=id_vars,
        value_vars=list(metrics),
        var_name="Metric",
        value_name="Percent",
    )
    plt.figure(figsize=(10, 5))
    sns.barplot(data=melted, x=x, y="Percent", hue="Metric" if hue is None else hue)
    plt.title(title)
    plt.ylabel("Percent")
    plt.xticks(rotation=25, ha="right")
    return save_current(path)


def figure_cf_only_by_question_type(df: pd.DataFrame, path: Path) -> Path:
    ensure_columns(df, ["Input_Mode", "Question_Type"], figure_name="CF-only by question type")
    subset = require_nonempty(
        df[df["Input_Mode"] == "cf_only"].copy(),
        figure_name="CF-only by question type",
        reason="no cf_only rows in full_matrix.csv",
    )
    return metric_barplot(
        subset,
        x="Question_Type",
        title="CF-Only Accuracy and Bias by Question Type",
        path=path,
    )


def figure_input_mode_comparison(df: pd.DataFrame, path: Path) -> Path:
    return metric_barplot(df, x="Input_Mode", title="Input Mode Comparison", path=path)


def figure_bias_by_edit_type(df: pd.DataFrame, path: Path) -> Path:
    ensure_columns(df, ["Edit_Type", "Accuracy%", "Bias%"], figure_name="Bias by edit type")
    subset = require_nonempty(
        df[df["Edit_Type"].astype(str).str.len() > 0].copy(),
        figure_name="Bias by edit type",
        reason="no edit_type values in by_edit_type.csv",
    )
    return metric_barplot(
        subset,
        x="Edit_Type",
        title="Accuracy and Bias by Edit Type",
        path=path,
        hue=None,
        metrics=("Accuracy%", "Bias%"),
    )


def figure_bias_by_subset(df: pd.DataFrame, path: Path) -> Path:
    ensure_columns(df, ["Subset", "Accuracy%", "Bias%"], figure_name="Bias by subset")
    return metric_barplot(
        df,
        x="Subset",
        title="Accuracy and Bias by Counting Subset",
        path=path,
        hue=None,
        metrics=("Accuracy%", "Bias%"),
    )


def figure_more_vs_fewer(df: pd.DataFrame, path: Path) -> Path:
    ensure_columns(df, ["Count_Direction", "Accuracy%", "Bias%"], figure_name="More vs fewer")
    return metric_barplot(
        df,
        x="Count_Direction",
        title="More-than-Expected vs Fewer-than-Expected Counterfactuals",
        path=path,
        hue=None,
        metrics=("Accuracy%", "Bias%"),
    )


def figure_subcategory_heatmap(df: pd.DataFrame, path: Path) -> Path:
    ensure_columns(df, ["Subset", "Subcategory", "Model", "Accuracy%"], figure_name="Subcategory heatmap")
    subset = df.copy()
    subset["Row"] = subset["Subset"] + "/" + subset["Subcategory"]
    pivot = require_nonempty(
        subset.pivot_table(index="Row", columns="Model", values="Accuracy%", aggfunc="mean"),
        figure_name="Subcategory heatmap",
        reason="no subset/subcategory accuracy rows",
    )
    plt.figure(figsize=(9, max(5, len(pivot) * 0.4)))
    sns.heatmap(pivot, annot=True, fmt=".1f", cmap="YlGnBu", cbar_kws={"label": "Accuracy%"})
    plt.title("Counting Subcategory Accuracy Heatmap")
    return save_current(path)


def figure_mc_vs_open(df: pd.DataFrame, path: Path) -> Path:
    ensure_columns(df, ["Input_Mode", "Question_Type"], figure_name="MC vs open")
    subset = require_nonempty(
        df[df["Input_Mode"] == "cf_only"].copy(),
        figure_name="MC vs open",
        reason="no cf_only rows in full_matrix.csv",
    )
    subset["Format"] = subset["Question_Type"].map(
        lambda value: "multiple_choice" if value == "multiple_choice" else "non_mc"
    )
    grouped = (
        subset.groupby(["Model", "Format"], as_index=False)[["Accuracy%", "Bias%", "Other%"]]
        .mean(numeric_only=True)
        .sort_values(["Model", "Format"])
    )
    return metric_barplot(
        grouped,
        x="Format",
        title="Multiple Choice vs Non-Multiple Choice in CF-Only Mode",
        path=path,
    )


def make_source_variant_alias(df: pd.DataFrame, path: Path) -> Path:
    return metric_barplot(df, x="Source_Variant", title="Source Variant Comparison", path=path)


def make_model_alias(df: pd.DataFrame, path: Path) -> Path:
    return metric_barplot(df, x="Model", title="Model Comparison", path=path, hue=None)


def write_legacy_aliases(*, figures_root: Path, aliases: dict[str, Path]) -> None:
    for alias_name, source_path in aliases.items():
        alias_path = figures_root / alias_name
        if alias_path == source_path:
            continue
        if not source_path.exists():
            raise SystemExit(f"Cannot create legacy alias; source figure is missing: {source_path}")
        shutil.copyfile(source_path, alias_path)
        print(f"  wrote {alias_path}")


if __name__ == "__main__":
    sys.exit(main())
