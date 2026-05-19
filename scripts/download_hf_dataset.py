#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REPO_ID = "JingyuSun/counterfactual-vlm-benchmark-data"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download the benchmark data payload from a Hugging Face Dataset repo.")
    parser.add_argument("--repo-id", default=os.environ.get("HF_DATASET_REPO_ID", DEFAULT_REPO_ID))
    parser.add_argument("--local-dir", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--revision", default="main")
    parser.add_argument("--include", action="append", default=[], help="Optional include glob passed to hf download.")
    parser.add_argument("--exclude", action="append", default=[], help="Optional exclude glob passed to hf download.")
    parser.add_argument("--dry-run", action="store_true", help="Print the command without downloading.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.repo_id:
        raise SystemExit("Provide --repo-id or set HF_DATASET_REPO_ID.")
    local_dir = resolve_path(args.local_dir)
    command = [
        "hf",
        "download",
        args.repo_id,
        "--type",
        "dataset",
        "--revision",
        args.revision,
        "--local-dir",
        str(local_dir),
    ]
    for pattern in args.include:
        command.extend(["--include", pattern])
    for pattern in args.exclude:
        command.extend(["--exclude", pattern])

    print("download_command=" + " ".join(command))
    print(f"target_local_dir={local_dir}")
    if args.dry_run:
        print("dry_run=true; no files downloaded.")
        print_expected_layout(local_dir)
        return 0

    if shutil.which("hf") is None:
        raise SystemExit("hf CLI is not installed or not on PATH.")
    result = subprocess.run(command, cwd=PROJECT_ROOT, check=False)
    if result.returncode != 0:
        return result.returncode
    check_layout(local_dir)
    print_metadata_note()
    return 0


def resolve_path(path: Path) -> Path:
    if path.is_absolute():
        return path
    return (PROJECT_ROOT / path).resolve()


def print_expected_layout(local_dir: Path) -> None:
    print("expected_layout:")
    for rel in expected_paths():
        print(f"- {local_dir / rel}")
    print_metadata_note()


def check_layout(local_dir: Path) -> None:
    missing = [rel for rel in expected_paths() if not (local_dir / rel).exists()]
    if missing:
        print("warning=download completed, but expected paths are missing:")
        for rel in missing:
            print(f"- {local_dir / rel}")
    else:
        print("layout_check=ok")


def expected_paths() -> list[Path]:
    return [
        Path("dataset"),
        Path("cf_dataset"),
        Path("vision_dataset"),
        Path("medical/modality_swapping/medical_modality_questions.json"),
        Path("eval_results/if_exist/metadata"),
        Path("eval_results/counting/metadata"),
        Path("eval_results/fashion_industry/metadata"),
        Path("eval_results/medical_modality/metadata"),
    ]


def print_metadata_note() -> None:
    print(
        "metadata_note=This HF data package includes lightweight eval_results/*/metadata runtime artifacts, "
        "but intentionally excludes raw runs, reports, tables, figures, and provider caches."
    )


if __name__ == "__main__":
    sys.exit(main())
