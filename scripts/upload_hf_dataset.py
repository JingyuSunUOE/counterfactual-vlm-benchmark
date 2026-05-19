#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Sequence

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_STAGING_DIR = PROJECT_ROOT / "hf_dataset_release"
DEFAULT_LOG_DIR = PROJECT_ROOT / "release_logs"
CODE_REPO_URL = "https://github.com/JingyuSunUOE/counterfactual-vlm-benchmark"
DATASET_REPO_URL_PREFIX = "https://huggingface.co/datasets/"

EXCLUDED_DESCRIPTIONS = [
    ".env and other local secret files",
    ".venv/, .idea/, caches, __pycache__, and .DS_Store",
    "eval_results/ raw API runs, reports, tables, figures, and provider caches",
    "medical/BraTS2023_GLI/ and all BraTS source NIfTI volumes",
    "medical/modality_swapping/samples/",
]

PLACEHOLDER_NAMESPACES = {"YOUR_NAME", "USERNAME", "USER", "ORG_NAME", "YOUR_ORG"}


@dataclass
class CopyPlanItem:
    source: Path
    destination: Path
    required: bool
    description: str


@dataclass
class PathStats:
    files: int = 0
    bytes: int = 0


class Logger:
    def __init__(self, log_path: Path | None):
        self.log_path = log_path
        self._handle = None
        if log_path is not None:
            log_path.parent.mkdir(parents=True, exist_ok=True)
            self._handle = log_path.open("w", encoding="utf-8")

    def close(self) -> None:
        if self._handle is not None:
            self._handle.close()
            self._handle = None

    def log(self, message: str) -> None:
        print(message, flush=True)
        if self._handle is not None:
            self._handle.write(message + "\n")
            self._handle.flush()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build and upload the benchmark data payload to a Hugging Face Dataset repo."
    )
    parser.add_argument("--repo-id", required=True, help="HF dataset repo id, e.g. USER/vlm-counterfactual-benchmark-data.")
    parser.add_argument("--staging-dir", type=Path, default=DEFAULT_STAGING_DIR)
    parser.add_argument("--private", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--dry-run", action="store_true", help="Print the plan without copying, creating a repo, or uploading.")
    parser.add_argument("--overwrite-staging", action="store_true", help="Delete and rebuild the staging directory.")
    parser.add_argument("--reuse-staging", action="store_true", help="Upload the existing staging directory without rebuilding it.")
    parser.add_argument("--num-workers", type=int, default=8, help="Worker count passed to hf upload-large-folder.")
    parser.add_argument(
        "--include-medical-generated",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Include generated medical visualization PNGs under medical/modality_swapping/standardized if present.",
    )
    parser.add_argument("--log-every", type=int, default=500, help="Print copy progress every N files.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    staging_dir = resolve_path(args.staging_dir)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    logger = Logger(None if args.dry_run else DEFAULT_LOG_DIR / f"hf_upload_{timestamp}.log")
    try:
        return run(args=args, staging_dir=staging_dir, timestamp=timestamp, logger=logger)
    finally:
        logger.close()


def run(*, args: argparse.Namespace, staging_dir: Path, timestamp: str, logger: Logger) -> int:
    if args.reuse_staging and args.overwrite_staging:
        raise SystemExit("--reuse-staging and --overwrite-staging are mutually exclusive.")
    load_dotenv(PROJECT_ROOT / ".env")
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    copy_plan = build_copy_plan(staging_dir=staging_dir, include_medical_generated=args.include_medical_generated)

    logger.log(f"repo_id={args.repo_id}")
    logger.log(f"private={args.private}")
    logger.log(f"staging_dir={staging_dir}")
    logger.log(f"include_medical_generated={args.include_medical_generated}")
    logger.log("metadata_included=true")
    logger.log("reports_included=false")
    logger.log("excluded=" + json.dumps(EXCLUDED_DESCRIPTIONS, ensure_ascii=False))
    validate_repo_id(args.repo_id, dry_run=args.dry_run, logger=logger)

    validate_copy_plan(copy_plan)
    planned_stats = summarize_plan(copy_plan)
    logger.log("planned_payload=" + json.dumps(stats_to_json(planned_stats), indent=2))
    for item in copy_plan:
        logger.log(f"include={item.source} -> {item.destination} ({item.description})")

    upload_command = [
        "hf",
        "upload-large-folder",
        args.repo_id,
        str(staging_dir),
        "--type",
        "dataset",
        "--num-workers",
        str(args.num_workers),
    ]
    logger.log("upload_command=" + " ".join(upload_command))

    if args.dry_run:
        if not token:
            logger.log("warning=HF_TOKEN not found in environment or .env; actual upload would fail before repo creation.")
        if shutil.which("hf") is None:
            logger.log("warning=hf CLI not found; actual upload requires the Hugging Face CLI.")
        logger.log("dry_run=true; no files copied, no repo created, no upload started.")
        return 0

    if not token:
        raise SystemExit("HF_TOKEN is not set in .env or the environment.")
    if shutil.which("hf") is None:
        raise SystemExit("hf CLI is not installed or not on PATH. Install it before uploading.")
    create_dataset_repo(repo_id=args.repo_id, private=args.private, token=token, logger=logger)
    if args.reuse_staging:
        validate_existing_staging(staging_dir)
        logger.log(f"reuse_staging=true path={staging_dir}")
        logger.log("staging_rebuild_skipped=true")
    else:
        prepare_staging_dir(staging_dir=staging_dir, overwrite=args.overwrite_staging)
        copied_stats = copy_payload(copy_plan=copy_plan, logger=logger, log_every=args.log_every)
        write_dataset_card(staging_dir=staging_dir, repo_id=args.repo_id, timestamp=timestamp, stats=copied_stats)
        write_manifest(staging_dir=staging_dir, repo_id=args.repo_id, timestamp=timestamp, stats=copied_stats, args=args)

    env = os.environ.copy()
    env["HF_TOKEN"] = token
    logger.log("starting_upload=true")
    result = subprocess.run(upload_command, cwd=PROJECT_ROOT, env=env, check=False)
    logger.log(f"upload_exit_code={result.returncode}")
    logger.log(f"dataset_url=https://huggingface.co/datasets/{args.repo_id}")
    return result.returncode


def resolve_path(path: Path) -> Path:
    if path.is_absolute():
        return path
    return (PROJECT_ROOT / path).resolve()


def build_copy_plan(*, staging_dir: Path, include_medical_generated: bool) -> list[CopyPlanItem]:
    items = [
        CopyPlanItem(PROJECT_ROOT / "dataset", staging_dir / "dataset", True, "original images"),
        CopyPlanItem(PROJECT_ROOT / "cf_dataset", staging_dir / "cf_dataset", True, "counterfactual images and question JSON"),
        CopyPlanItem(PROJECT_ROOT / "vision_dataset", staging_dir / "vision_dataset", True, "generated visual evidence"),
        CopyPlanItem(
            PROJECT_ROOT / "eval_results" / "if_exist" / "metadata",
            staging_dir / "eval_results" / "if_exist" / "metadata",
            True,
            "if_exist runtime metadata and evidence manifests",
        ),
        CopyPlanItem(
            PROJECT_ROOT / "eval_results" / "counting" / "metadata",
            staging_dir / "eval_results" / "counting" / "metadata",
            True,
            "counting runtime metadata and evidence manifests",
        ),
        CopyPlanItem(
            PROJECT_ROOT / "eval_results" / "fashion_industry" / "metadata",
            staging_dir / "eval_results" / "fashion_industry" / "metadata",
            True,
            "fashion/industry runtime metadata and evidence manifests",
        ),
        CopyPlanItem(
            PROJECT_ROOT / "eval_results" / "medical_modality" / "metadata",
            staging_dir / "eval_results" / "medical_modality" / "metadata",
            True,
            "medical modality runtime metadata and evidence manifests",
        ),
        CopyPlanItem(
            PROJECT_ROOT / "medical" / "modality_swapping" / "medical_modality_questions.json",
            staging_dir / "medical" / "modality_swapping" / "medical_modality_questions.json",
            True,
            "medical question definitions",
        ),
    ]
    if include_medical_generated:
        items.append(
            CopyPlanItem(
                PROJECT_ROOT / "medical" / "modality_swapping" / "standardized",
                staging_dir / "medical" / "modality_swapping" / "standardized",
                False,
                "generated medical visualization images",
            )
        )
    return items


def validate_copy_plan(copy_plan: Sequence[CopyPlanItem]) -> None:
    missing = [item.source for item in copy_plan if item.required and not item.source.exists()]
    if missing:
        raise SystemExit("Required release inputs are missing:\n" + "\n".join(f"- {path}" for path in missing))


def validate_repo_id(repo_id: str, *, dry_run: bool, logger: Logger) -> None:
    if "/" not in repo_id:
        message = (
            f"Invalid --repo-id {repo_id!r}. Use the form NAMESPACE/DATASET_NAME, "
            "for example sunjingyu/vlm-counterfactual-benchmark-data."
        )
        raise SystemExit(message)
    namespace, name = repo_id.split("/", 1)
    if namespace in PLACEHOLDER_NAMESPACES or namespace.startswith("YOUR_"):
        message = (
            f"--repo-id still uses placeholder namespace {namespace!r}. Replace it with your Hugging Face "
            "username or an organization namespace where your token has write permission."
        )
        if dry_run:
            logger.log("warning=" + message)
        else:
            raise SystemExit(message)
    if not namespace or not name or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", namespace):
        raise SystemExit(f"Invalid Hugging Face namespace in --repo-id: {repo_id!r}")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", name):
        raise SystemExit(f"Invalid Hugging Face dataset name in --repo-id: {repo_id!r}")


def summarize_plan(copy_plan: Sequence[CopyPlanItem]) -> dict[str, PathStats]:
    summary: dict[str, PathStats] = {}
    for item in copy_plan:
        if item.source.exists():
            summary[str(item.source.relative_to(PROJECT_ROOT))] = collect_stats(item.source)
        else:
            summary[str(item.source.relative_to(PROJECT_ROOT))] = PathStats()
    return summary


def collect_stats(path: Path) -> PathStats:
    stats = PathStats()
    if path.is_file():
        if should_copy_file(path):
            stats.files += 1
            stats.bytes += path.stat().st_size
        return stats
    for child in path.rglob("*"):
        if child.is_file() and should_copy_file(child):
            stats.files += 1
            stats.bytes += child.stat().st_size
    return stats


def stats_to_json(stats: dict[str, PathStats]) -> dict[str, dict[str, int | str]]:
    return {
        key: {"files": value.files, "bytes": value.bytes, "human_size": human_bytes(value.bytes)}
        for key, value in stats.items()
    }


def prepare_staging_dir(*, staging_dir: Path, overwrite: bool) -> None:
    if staging_dir.exists():
        if not overwrite:
            raise SystemExit(f"Staging directory already exists: {staging_dir}. Use --overwrite-staging to rebuild it.")
        shutil.rmtree(staging_dir)
    staging_dir.mkdir(parents=True, exist_ok=True)


def validate_existing_staging(staging_dir: Path) -> None:
    required = [
        staging_dir,
        staging_dir / "README.md",
        staging_dir / "MANIFEST.json",
        staging_dir / "dataset",
        staging_dir / "cf_dataset",
        staging_dir / "vision_dataset",
        staging_dir / "eval_results" / "if_exist" / "metadata",
        staging_dir / "eval_results" / "counting" / "metadata",
        staging_dir / "eval_results" / "fashion_industry" / "metadata",
        staging_dir / "eval_results" / "medical_modality" / "metadata",
    ]
    missing = [path for path in required if not path.exists()]
    if missing:
        raise SystemExit(
            "--reuse-staging requested, but the staging directory is incomplete:\n"
            + "\n".join(f"- missing {path}" for path in missing)
        )


def copy_payload(*, copy_plan: Sequence[CopyPlanItem], logger: Logger, log_every: int) -> dict[str, PathStats]:
    copied: dict[str, PathStats] = {}
    for item in copy_plan:
        rel = str(item.source.relative_to(PROJECT_ROOT))
        if not item.source.exists():
            logger.log(f"skip_missing_optional={item.source}")
            copied[rel] = PathStats()
            continue
        logger.log(f"copy_start={item.source} -> {item.destination}")
        stats = copy_path(item.source, item.destination, logger=logger, log_every=log_every)
        copied[rel] = stats
        logger.log(f"copy_done={rel} files={stats.files} size={human_bytes(stats.bytes)}")
    return copied


def copy_path(source: Path, destination: Path, *, logger: Logger, log_every: int) -> PathStats:
    stats = PathStats()
    if source.is_file():
        copy_file(source, destination)
        stats.files = 1
        stats.bytes = source.stat().st_size
        return stats
    files = [path for path in source.rglob("*") if path.is_file() and should_copy_file(path)]
    total = len(files)
    for index, path in enumerate(files, start=1):
        rel = path.relative_to(source)
        copy_file(path, destination / rel)
        stats.files += 1
        stats.bytes += path.stat().st_size
        if log_every > 0 and (index == 1 or index == total or index % log_every == 0):
            logger.log(f"copy_progress source={source.name} files={index}/{total} size={human_bytes(stats.bytes)}")
    return stats


def copy_file(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def should_copy_file(path: Path) -> bool:
    parts = set(path.parts)
    if path.name == ".DS_Store":
        return False
    if "__pycache__" in parts:
        return False
    if "archive_before_rebuild" in parts:
        return False
    if path.suffix in {".nii"} or path.name.endswith(".nii.gz"):
        return False
    if ".env" in parts or path.name.startswith(".env"):
        return False
    return True


def write_dataset_card(*, staging_dir: Path, repo_id: str, timestamp: str, stats: dict[str, PathStats]) -> None:
    total_files = sum(item.files for item in stats.values())
    total_bytes = sum(item.bytes for item in stats.values())
    text = f"""---
license: other
pretty_name: Counterfactual VLM Benchmark Data
task_categories:
- visual-question-answering
---

# Counterfactual VLM Benchmark Data

This dataset repository contains the data payload for the Counterfactual VLM Benchmark.

Uploaded at: `{timestamp}`

- Code repository: [{CODE_REPO_URL}]({CODE_REPO_URL})
- Dataset repository: [{DATASET_REPO_URL_PREFIX}{repo_id}]({DATASET_REPO_URL_PREFIX}{repo_id})

Use this dataset together with the GitHub repository. From the repository root, run:

```bash
python scripts/download_hf_dataset.py \\
  --repo-id {repo_id} \\
  --local-dir .
```

## Contents

```text
dataset/                         Original images for non-medical benchmarks
cf_dataset/                      Counterfactual images and question JSON files
vision_dataset/                  Generated visual evidence
medical/modality_swapping/        Medical question JSON and generated medical visualizations, if included
eval_results/*/metadata/          Runtime metadata and evidence manifests needed by evaluators
MANIFEST.json                     Release manifest and payload summary
```

## Not Included

This release intentionally does not include:

- `eval_results/*/raw_runs`, reports, tables, figures, or raw API outputs
- BraTS source NIfTI files under `medical/BraTS2023_GLI/`
- API keys, provider caches, or local environment files

The included `eval_results/*/metadata/` directories are lightweight runtime artifacts used by the evaluation scripts. They are included so a fresh GitHub checkout plus this dataset payload has the same directory layout expected by the benchmark runners.

## Size Summary

Total files: `{total_files}`

Total size: `{human_bytes(total_bytes)}`

## Medical Data Note

BraTS source data is not redistributed. Any medical generated images in this dataset should be used only under the relevant source-data license constraints.
"""
    (staging_dir / "README.md").write_text(text, encoding="utf-8")


def write_manifest(
    *,
    staging_dir: Path,
    repo_id: str,
    timestamp: str,
    stats: dict[str, PathStats],
    args: argparse.Namespace,
) -> None:
    manifest = {
        "schema_version": "vlm_counterfactual_hf_release_v1",
        "repo_id": repo_id,
        "code_repo_url": CODE_REPO_URL,
        "dataset_repo_url": DATASET_REPO_URL_PREFIX + repo_id,
        "created_at_utc": timestamp,
        "private_requested": bool(args.private),
        "metadata_included": True,
        "reports_included": False,
        "include_medical_generated": bool(args.include_medical_generated),
        "staging_dir": str(staging_dir),
        "included": stats_to_json(stats),
        "excluded": EXCLUDED_DESCRIPTIONS,
    }
    (staging_dir / "MANIFEST.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def create_dataset_repo(*, repo_id: str, private: bool, token: str, logger: Logger) -> None:
    try:
        from huggingface_hub import HfApi
    except ImportError as exc:
        logger.log(f"warning=huggingface_hub import failed; falling back to hf CLI repo creation: {exc}")
        create_dataset_repo_with_cli(repo_id=repo_id, private=private, token=token, logger=logger)
        return
    try:
        from huggingface_hub.errors import HfHubHTTPError
    except ImportError:
        try:
            from huggingface_hub.utils import HfHubHTTPError  # type: ignore[attr-defined]
        except ImportError as exc:
            logger.log(f"warning=HfHubHTTPError import failed; falling back to hf CLI repo creation: {exc}")
            create_dataset_repo_with_cli(repo_id=repo_id, private=private, token=token, logger=logger)
            return
    logger.log("creating_or_reusing_dataset_repo=true")
    api = HfApi(token=token)
    try:
        whoami = api.whoami(token=token)
        username = whoami.get("name") or whoami.get("fullname") or "unknown"
        logger.log(f"hf_authenticated_as={username}")
        api.create_repo(repo_id=repo_id, repo_type="dataset", private=private, exist_ok=True)
    except HfHubHTTPError as exc:
        status = getattr(getattr(exc, "response", None), "status_code", None)
        if status == 403:
            raise SystemExit(
                "Hugging Face returned 403 Forbidden while creating the dataset repo.\n"
                f"repo_id={repo_id}\n"
                "Most likely causes:\n"
                "- --repo-id uses a namespace that is not your HF username or an org you can write to.\n"
                "- HF_TOKEN lacks write permission.\n"
                "- The target organization requires different permissions.\n"
                "Fix:\n"
                "1. Run `hf auth whoami` to confirm the logged-in user.\n"
                "2. Use `--repo-id <your_hf_username>/vlm-counterfactual-benchmark-data`, or create the dataset repo manually on HF.\n"
                "3. Ensure the token has write access."
            ) from exc
        raise


def create_dataset_repo_with_cli(*, repo_id: str, private: bool, token: str, logger: Logger) -> None:
    if shutil.which("hf") is None:
        raise SystemExit("hf CLI is not installed or not on PATH, and huggingface_hub API fallback is unavailable.")
    command = ["hf", "repos", "create", repo_id, "--type", "dataset", "--exist-ok"]
    if private:
        command.append("--private")
    env = os.environ.copy()
    env["HF_TOKEN"] = token
    logger.log("creating_or_reusing_dataset_repo_with_cli=true")
    logger.log("repo_create_command=" + " ".join(command))
    result = subprocess.run(command, cwd=PROJECT_ROOT, env=env, check=False)
    if result.returncode != 0:
        raise SystemExit(result.returncode)


def human_bytes(num_bytes: int) -> str:
    value = float(num_bytes)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if value < 1024 or unit == "TB":
            return f"{value:.1f}{unit}"
        value /= 1024
    return f"{value:.1f}TB"


if __name__ == "__main__":
    sys.exit(main())
