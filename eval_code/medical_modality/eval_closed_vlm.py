#!/usr/bin/env python3
from __future__ import annotations

import runpy
import sys
from pathlib import Path


CURRENT_DIR = Path(__file__).resolve().parent
COUNTING_RUNNER = CURRENT_DIR.parent / "counting" / "eval_closed_vlm.py"

if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

sys.argv[0] = str(Path(__file__).resolve())
runpy.run_path(str(COUNTING_RUNNER), run_name="__main__")
