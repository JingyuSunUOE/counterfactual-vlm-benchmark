#!/usr/bin/env python3
from __future__ import annotations

import re
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
REPORTS = [
    REPO_ROOT / "eval_results/counting/reports/experiment_design.md",
    REPO_ROOT / "eval_results/counting/reports/evaluation_report.md",
]
HAN_RE = re.compile(r"[\u4e00-\u9fff]")
REQUIRED = ["dataset/counting", "cf_dataset/counting_cf", "eval_results/counting"]
FORBIDDEN = ["counting/", "antlers", "bird v2", "12 insect orders", "legs_add", "legs_remove"]


def main() -> int:
    for path in REPORTS:
        if not path.exists():
            raise SystemExit(f"Missing report: {path}")
        text = path.read_text(encoding="utf-8")
        if HAN_RE.search(text):
            raise SystemExit(f"Report should be English-only: {path}")
        for fragment in REQUIRED:
            if fragment not in text:
                raise SystemExit(f"Report missing required path {fragment}: {path}")
        for fragment in FORBIDDEN:
            if fragment in text and fragment not in {"counting/"}:
                raise SystemExit(f"Report contains stale fragment {fragment}: {path}")
        if "`counting/`" in text:
            raise SystemExit(f"Report should not refer to old root counting/ source path: {path}")
    print("counting report path tests passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
