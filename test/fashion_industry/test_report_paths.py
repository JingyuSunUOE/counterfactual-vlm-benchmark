#!/usr/bin/env python3
from __future__ import annotations

import re
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
REPORTS_DIR = REPO_ROOT / "eval_results" / "fashion_industry" / "reports"
HAN_RE = re.compile(r"[\u4e00-\u9fff]")

STALE_PATTERNS = [
    "vlm_biased",
    "eval_results/full_matrix.csv",
    "eval_results/by_category.csv",
    "eval_results/by_mod_type.csv",
    "`figures/",
    "`metadata/",
    "`eval_results/eval_cf_",
    "Gemini 3.1",
    "GPT-5.4",
    "Claude Opus",
    "Qwen 3.5",
    "DeepSeek",
    "Final",
    "Pending",
    "nearly invalid",
]

REQUIRED_ARTIFACTS = [
    "eval_results/fashion_industry/tables/full_matrix.csv",
    "eval_results/fashion_industry/tables/by_category.csv",
    "eval_results/fashion_industry/tables/by_mod_type.csv",
    "eval_results/fashion_industry/figures/fig1_cf_only_by_qtype.png",
    "eval_results/fashion_industry/figures/fig2_input_mode_comparison.png",
    "eval_results/fashion_industry/figures/fig3_bias_by_mod_type.png",
    "eval_results/fashion_industry/figures/fig4_category_heatmap.png",
    "eval_results/fashion_industry/figures/fig5_mc_vs_open.png",
    "eval_results/fashion_industry/figures/fig6_fashion_vs_industry.png",
    "eval_results/fashion_industry/metadata/fashion_cf_metadata.json",
    "eval_results/fashion_industry/metadata/industry_cf_metadata.json",
    "dataset/fashion_dataset",
    "dataset/industry_dataset",
    "cf_dataset/fashion_cf",
    "cf_dataset/industry_cf",
    "eval_code/fashion_industry/eval_pipeline.py",
    "eval_code/fashion_industry/eval_questions.py",
    "gen_code/fashion_industry/gen_fashion_cf.py",
    "gen_code/fashion_industry/gen_industry_cf.py",
]


def main() -> int:
    reports = sorted(REPORTS_DIR.glob("*.md"))
    if not reports:
        raise SystemExit(f"No Markdown reports found in {REPORTS_DIR}")

    for report in reports:
        text = report.read_text()
        if HAN_RE.search(text):
            raise SystemExit(f"{report} contains Chinese characters.")
        stale = [pattern for pattern in STALE_PATTERNS if pattern in text]
        if stale:
            raise SystemExit(f"{report} contains stale report patterns: {stale}")
        check_relative_figure_links(report, text)

    missing = [path for path in REQUIRED_ARTIFACTS if not (REPO_ROOT / path).exists()]
    if missing:
        raise SystemExit(f"Missing required artifacts: {missing}")

    print(f"Report path audit passed for {len(reports)} Markdown files.")
    return 0


def check_relative_figure_links(report: Path, text: str) -> None:
    for target in re.findall(r"\]\((\.\./figures/[^)]+)\)", text):
        if not (report.parent / target).resolve().exists():
            raise SystemExit(f"{report} references missing figure link: {target}")


if __name__ == "__main__":
    sys.exit(main())
