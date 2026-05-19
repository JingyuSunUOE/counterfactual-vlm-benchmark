#!/usr/bin/env python3
from __future__ import annotations

import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
REPORTS = ROOT / "eval_results" / "medical_modality" / "reports"


def main() -> int:
    files = [REPORTS / "experiment_design.md", REPORTS / "evaluation_report.md"]
    for path in files:
        assert path.exists(), f"missing report: {path}"
        text = path.read_text(encoding="utf-8")
        assert not re.search(r"[\u4e00-\u9fff]", text), f"Chinese text found in {path}"
        assert "results/modality_swap/" not in text
        assert "eval_results/medical_modality" in text
        assert "medical/modality_swapping" in text
    print("medical modality report paths ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
