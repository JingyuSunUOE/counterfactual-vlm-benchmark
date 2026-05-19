#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import re
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
HAN_RE = re.compile(r"[\u4e00-\u9fff]")


def main() -> int:
    checks = [
        (
            "fashion cf help",
            [sys.executable, "gen_code/fashion_industry/gen_fashion_cf.py", "--help"],
            ["--brand", "--model", "--dry-run"],
        ),
        (
            "industry cf help",
            [sys.executable, "gen_code/fashion_industry/gen_industry_cf.py", "--help"],
            ["--category", "--model", "--dry-run"],
        ),
        (
            "fashion scrape help",
            [sys.executable, "gen_code/fashion_industry/scrape_fashion_logos.py", "--help"],
            ["--phase", "--brand", "--category", "--count", "--model", "--dry-run"],
        ),
        (
            "industry scrape help",
            [sys.executable, "gen_code/fashion_industry/scrape_industry.py", "--help"],
            ["--phase", "--category", "--count", "--model", "--dry-run"],
        ),
        (
            "organize fashion help",
            [sys.executable, "gen_code/fashion_industry/organize_fashion.py", "--help"],
            ["--exec"],
        ),
        (
            "organize industry help",
            [sys.executable, "gen_code/fashion_industry/organize_industry.py", "--help"],
            ["--exec"],
        ),
        (
            "fashion cf dry run",
            [
                sys.executable,
                "gen_code/fashion_industry/gen_fashion_cf.py",
                "--model",
                "both",
                "--brand",
                "chanel",
                "--dry-run",
            ],
            ["Chanel double-C", "SKIP", "DONE"],
        ),
        (
            "industry cf dry run",
            [
                sys.executable,
                "gen_code/fashion_industry/gen_industry_cf.py",
                "--model",
                "both",
                "--category",
                "basketball",
                "--dry-run",
            ],
            ["basketball", "SKIP", "DONE"],
        ),
    ]

    for name, command, expected in checks:
        output = run(name, command)
        require_all(name, output, expected)
        if name.endswith("dry run") and HAN_RE.search(output):
            raise SystemExit(f"{name} still contains Chinese display text.")

    check_display_strings()
    print("fashion_industry generation CLI smoke tests passed.")
    return 0


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


def check_display_strings() -> None:
    specs = [
        ("gen_code/fashion_industry/gen_fashion_cf.py", "FASHION_CF"),
        ("gen_code/fashion_industry/gen_industry_cf.py", "INDUSTRY_CF"),
    ]
    checked = 0
    for relative_path, config_name in specs:
        module = import_module(REPO_ROOT / relative_path)
        config = getattr(module, config_name)
        for key, cfg in config.items():
            checked += 1
            assert_no_han(f"{relative_path}:{key}:name", module.display_name(key, cfg))
            for mod in cfg["mods"]:
                checked += 1
                assert_no_han(f"{relative_path}:{key}:mod{mod['id']}", module.display_desc(key, mod))
    print(f"[CHECK] English display strings: {checked}")


def import_module(path: Path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def assert_no_han(label: str, value: str) -> None:
    if HAN_RE.search(value):
        raise SystemExit(f"{label} contains Chinese display text: {value}")


if __name__ == "__main__":
    sys.exit(main())
