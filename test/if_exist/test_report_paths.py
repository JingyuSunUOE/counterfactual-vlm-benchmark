#!/usr/bin/env python3
from __future__ import annotations

import re
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
REPORTS_ROOT = REPO_ROOT / "eval_results" / "if_exist" / "reports"

REPORT_FILES = [
    REPORTS_ROOT / "experiment_design.md",
    REPORTS_ROOT / "evaluation_report.md",
    REPORTS_ROOT / "closed_model_evaluation_report.md",
    REPORTS_ROOT / "open_model_evaluation_report.md",
    REPORTS_ROOT / "agentic_vision_tool_report.md",
]

STALE_PATTERNS = [
    "images/if_exist",
    "Edited_images/if_exist",
    "dataset/" + "env" + "ironment",
    "cf_dataset/" + "env" + "ironment_cf",
    "cf_dataset/" + "env" + "ironment_if_exist_cf_questions.json",
    "eval_code/" + "env" + "ironment_if_exist",
    "eval_results/" + "env" + "ironment_if_exist",
    "counterfactual_vlm_benchmark_questions.json",
    "results/closed",
    "results/open",
    "20260405T082258Z",
    "if_exist_final_" + "v" + "2",
    "if_exist_prompt_" + "v" + "2",
    "if_exist_mc_" + "v" + "2",
    "merged_prompt_" + "v" + "2" + "_mc_" + "v" + "2",
]

REQUIRED_ARTIFACTS = [
    "cf_dataset/if_exist_cf_questions.json",
    "dataset/if_exist",
    "cf_dataset/if_exist_cf",
    "eval_code/if_exist/eval_closed_vlm.py",
    "eval_code/if_exist/eval_open_vlm.py",
    "eval_code/if_exist/eval_server_vlm.py",
    "eval_code/if_exist/analyze_results.py",
    "eval_code/if_exist/analyze_item_difficulty.py",
    "eval_code/if_exist/merge_replacement_runs.py",
    "eval_code/if_exist/make_figures.py",
    "eval_code/if_exist/build_metadata.py",
    "eval_code/if_exist/build_evidence_manifest.py",
    "eval_results/if_exist/metadata/if_exist_sam3_object_evidence_manifest.json",
    "eval_results/if_exist/metadata/if_exist_sam3_object_evidence_manifest.csv",
    "eval_results/if_exist/raw_runs",
    "eval_results/if_exist/tables",
    "eval_results/if_exist/figures",
    "eval_results/if_exist/metadata",
]


def main() -> int:
    for report_path in REPORT_FILES:
        if not report_path.exists():
            raise SystemExit(f"Missing report: {report_path}")
        text = report_path.read_text(encoding="utf-8")
        assert_english_only(report_path, text)
        assert_no_stale_paths(report_path, text)
        assert_markdown_links_exist(report_path, text)

    missing = [path for path in REQUIRED_ARTIFACTS if not (REPO_ROOT / path).exists()]
    if missing:
        raise SystemExit(f"Missing required referenced artifacts: {missing}")

    print("if_exist report path tests passed.")
    return 0


def assert_english_only(path: Path, text: str) -> None:
    if re.search(r"[\u4e00-\u9fff]", text):
        raise SystemExit(f"Report contains Chinese characters: {path}")


def assert_no_stale_paths(path: Path, text: str) -> None:
    stale = [pattern for pattern in STALE_PATTERNS if pattern in text]
    if stale:
        raise SystemExit(f"Report contains stale path references in {path}: {stale}")


def assert_markdown_links_exist(path: Path, text: str) -> None:
    links = re.findall(r"\]\(([^)]+)\)", text)
    missing: list[str] = []
    for link in links:
        if link.startswith(("http://", "https://", "#")):
            continue
        if link.startswith("../figures/"):
            continue
        target = (path.parent / link).resolve()
        if not target.exists():
            missing.append(link)
    if missing:
        raise SystemExit(f"Report has missing local markdown links in {path}: {missing}")


if __name__ == "__main__":
    sys.exit(main())
