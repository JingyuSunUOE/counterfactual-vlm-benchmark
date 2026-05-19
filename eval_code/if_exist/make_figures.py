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
    parser = argparse.ArgumentParser(description="Generate publication-style figures from if_exist tables.")
    parser.add_argument("--tables-root", type=Path, default=DEFAULT_TABLES_ROOT, help="Directory containing CSV tables.")
    parser.add_argument("--figures-root", type=Path, default=DEFAULT_FIGURES_ROOT, help="Directory for PNG figures.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.figures_root.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")

    full_matrix = read_csv(args.tables_root / "full_matrix.csv")
    by_input_mode = read_csv(args.tables_root / "by_input_mode.csv")
    by_edit_type = read_csv(args.tables_root / "by_edit_type.csv")
    by_subcategory = read_csv(args.tables_root / "by_subcategory.csv")
    by_source_variant = read_csv(args.tables_root / "by_source_variant.csv")
    by_model = read_csv(args.tables_root / "by_model.csv")
    by_tool_condition = read_optional_csv(args.tables_root / "by_tool_condition.csv")
    by_tool_condition_question_type = read_optional_csv(args.tables_root / "by_tool_condition_question_type.csv")
    by_tool_condition_input_mode = read_optional_csv(args.tables_root / "by_tool_condition_input_mode.csv")

    outputs = {
        "fig1": args.figures_root / "fig1_cf_only_by_qtype.png",
        "fig2": args.figures_root / "fig2_input_mode_comparison.png",
        "fig3": args.figures_root / "fig3_bias_by_edit_type.png",
        "fig4": args.figures_root / "fig4_category_heatmap.png",
        "fig5": args.figures_root / "fig5_mc_vs_open.png",
        "fig6": args.figures_root / "fig6_model_comparison.png",
    }
    figure_cf_only_by_question_type(full_matrix, outputs["fig1"])
    figure_input_mode_comparison(by_input_mode, outputs["fig2"])
    figure_bias_by_edit_type(by_edit_type, outputs["fig3"])
    figure_category_heatmap(by_subcategory, outputs["fig4"])
    figure_mc_vs_open(full_matrix, outputs["fig5"])
    make_model_alias(by_model, outputs["fig6"])
    if by_tool_condition is not None:
        figure_tool_condition_overall(
            by_tool_condition,
            args.figures_root / "fig_tool_condition_overall.png",
        )
        if has_non_raw_tool_condition(by_tool_condition):
            figure_bias_reduction_vs_raw(
                by_tool_condition,
                args.figures_root / "fig_bias_reduction_vs_raw.png",
            )
    if by_tool_condition_question_type is not None:
        figure_tool_condition_by_question_type(
            by_tool_condition_question_type,
            args.figures_root / "fig_tool_condition_by_question_type.png",
        )
    if by_tool_condition_input_mode is not None:
        figure_tool_condition_by_input_mode(
            by_tool_condition_input_mode,
            args.figures_root / "fig_tool_condition_by_input_mode.png",
        )

    write_legacy_aliases(
        figures_root=args.figures_root,
        aliases={
            "fig1_cf_only_by_question_type.png": outputs["fig1"],
            "fig4_subcategory_heatmap.png": outputs["fig4"],
            "fig5_source_variant_comparison.png": make_source_variant_alias(
                by_source_variant,
                args.figures_root / "fig5_source_variant_comparison.png",
            ),
        },
    )

    print(f"Figures written to {args.figures_root}")
    return 0


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise SystemExit(f"Missing table: {path}")
    df = pd.read_csv(path)
    if df.empty:
        raise SystemExit(f"Table is empty: {path}")
    return df


def read_optional_csv(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    df = pd.read_csv(path)
    if df.empty:
        return None
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


def figure_category_heatmap(df: pd.DataFrame, path: Path) -> Path:
    ensure_columns(df, ["Category", "Subcategory", "Model", "Accuracy%"], figure_name="Category heatmap")
    subset = df.copy()
    subset["Row"] = subset["Category"] + "/" + subset["Subcategory"]
    pivot = require_nonempty(
        subset.pivot_table(index="Row", columns="Model", values="Accuracy%", aggfunc="mean"),
        figure_name="Category heatmap",
        reason="no category/subcategory accuracy rows",
    )
    plt.figure(figsize=(9, max(5, len(pivot) * 0.4)))
    sns.heatmap(pivot, annot=True, fmt=".1f", cmap="YlGnBu", cbar_kws={"label": "Accuracy%"})
    plt.title("Category and Subcategory Accuracy Heatmap")
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


def figure_tool_condition_overall(df: pd.DataFrame, path: Path) -> Path:
    ensure_columns(df, ["Tool_Condition", "Accuracy%", "Bias%", "Other%"], figure_name="Tool condition overall")
    return metric_barplot(
        df,
        x="Tool_Condition",
        title="Agentic Vision Tool Conditions",
        path=path,
        hue=None,
    )


def figure_tool_condition_by_question_type(df: pd.DataFrame, path: Path) -> Path:
    ensure_columns(
        df,
        ["Tool_Condition", "Question_Type", "Accuracy%", "Bias%"],
        figure_name="Tool condition by question type",
    )
    subset = df.copy()
    subset["Tool_x_Q"] = subset["Tool_Condition"] + "\n" + subset["Question_Type"]
    return metric_barplot(
        subset,
        x="Tool_x_Q",
        title="Tool Conditions by Question Type",
        path=path,
        hue=None,
        metrics=("Accuracy%", "Bias%"),
    )


def figure_tool_condition_by_input_mode(df: pd.DataFrame, path: Path) -> Path:
    ensure_columns(
        df,
        ["Tool_Condition", "Input_Mode", "Accuracy%", "Bias%"],
        figure_name="Tool condition by input mode",
    )
    subset = df.copy()
    subset["Tool_x_Mode"] = subset["Tool_Condition"] + "\n" + subset["Input_Mode"]
    return metric_barplot(
        subset,
        x="Tool_x_Mode",
        title="Tool Conditions by Input Mode",
        path=path,
        hue=None,
        metrics=("Accuracy%", "Bias%"),
    )


def figure_bias_reduction_vs_raw(df: pd.DataFrame, path: Path) -> Path:
    ensure_columns(
        df,
        ["Tool_Condition", "Delta_Bias_vs_Raw"],
        figure_name="Bias reduction vs raw",
    )
    subset = require_nonempty(
        df[df["Tool_Condition"] != "raw"].copy(),
        figure_name="Bias reduction vs raw",
        reason="no non-raw tool condition rows",
    )
    plt.figure(figsize=(9, 4.5))
    sns.barplot(data=subset, x="Tool_Condition", y="Delta_Bias_vs_Raw", hue="Model" if "Model" in subset.columns else None)
    plt.axhline(0, color="black", linewidth=1)
    plt.title("Bias Change Relative to Raw")
    plt.ylabel("Delta Bias%")
    plt.xticks(rotation=25, ha="right")
    return save_current(path)


def has_non_raw_tool_condition(df: pd.DataFrame) -> bool:
    return "Tool_Condition" in df.columns and any(df["Tool_Condition"].astype(str) != "raw")


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
