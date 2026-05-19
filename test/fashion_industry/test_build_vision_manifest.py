#!/usr/bin/env python3
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from eval_code.fashion_industry.build_vision_manifest import (  # noqa: E402
    ManifestBuildError,
    build_cf_manifest,
    build_original_manifest,
)


def main() -> int:
    test_synthetic_manifest_generation()
    test_missing_image_fails_strict()
    test_missing_prompt_key_fails()
    test_duplicate_sample_id_fails()
    test_cli_dry_run_writes_nothing()
    print("fashion_industry vision manifest tests passed.")
    return 0


def make_record(root: Path, *, suffix: str, key: str = "chanel", model: str = "gemini") -> dict:
    original = root / "dataset" / "fashion_dataset" / key / f"{key}_real_01.png"
    cf = root / "cf_dataset" / "fashion_cf" / key / f"{key}_real_01_cf_mod1_direction_flip_{model}{suffix}.png"
    original.parent.mkdir(parents=True, exist_ok=True)
    cf.parent.mkdir(parents=True, exist_ok=True)
    original.write_bytes(b"original")
    cf.write_bytes(b"cf")
    return {
        "filename": cf.name,
        "category": "fashion_monogram",
        "brand_key": key,
        "brand_name": "Chanel",
        "source_type": "cf_from_real",
        "model": model,
        "mod_id": 1,
        "mod_type": "direction_flip",
        "mod_desc": "flip",
        "ground_truth": {"orientation": "face-to-face"},
        "bias_answer": {"orientation": "back-to-back"},
        "original_image": original.name,
        "success": True,
        "original_path": str(original),
        "cf_path": str(cf),
    }


def test_synthetic_manifest_generation() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        metadata = [
            make_record(root, suffix=""),
            make_record(root, suffix="_second", model="gpt"),
        ]
        prompt_keys = {"chanel"}
        originals = build_original_manifest("fashion", metadata, prompt_keys, strict=True)
        cfs = build_cf_manifest("fashion", metadata, prompt_keys, strict=True)
        assert len(originals) == 1
        assert len(cfs) == 2
        assert originals[0]["prompt_key"] == "chanel"
        assert originals[0]["source_variant"] == "real"
        assert originals[0]["image_role"] == "original"
        assert cfs[0]["image_role"] == "cf"
        assert cfs[0]["benchmark"] == "fashion"
        assert cfs[0]["prompt_mode"] == "object"
        assert cfs[0]["subcategory"] == "chanel"


def test_missing_image_fails_strict() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        metadata = [make_record(root, suffix="")]
        Path(metadata[0]["cf_path"]).unlink()
        try:
            build_cf_manifest("fashion", metadata, {"chanel"}, strict=True)
        except ManifestBuildError as exc:
            assert "cf_path does not exist" in str(exc)
        else:
            raise AssertionError("missing CF image should fail in strict mode")


def test_missing_prompt_key_fails() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        metadata = [make_record(Path(tmp), suffix="", key="unknown_logo")]
        try:
            build_original_manifest("fashion", metadata, {"chanel"}, strict=True)
        except ManifestBuildError as exc:
            assert "prompt_key not found" in str(exc)
        else:
            raise AssertionError("missing prompt key should fail")


def test_duplicate_sample_id_fails() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        first = make_record(root, suffix="")
        second = dict(first)
        second["model"] = "gpt"
        second["filename"] = "different.png"
        second["cf_path"] = first["cf_path"]
        try:
            build_cf_manifest("fashion", [first, second], {"chanel"}, strict=True)
        except ManifestBuildError as exc:
            assert "Duplicate fashion cf sample_id" in str(exc)
        else:
            raise AssertionError("duplicate sample_id should fail")


def test_cli_dry_run_writes_nothing() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        output_dir = Path(tmp) / "manifest_output"
        completed = subprocess.run(
            [
                sys.executable,
                "eval_code/fashion_industry/build_vision_manifest.py",
                "--domain",
                "fashion",
                "--dry-run",
                "--output-dir",
                str(output_dir),
            ],
            cwd=REPO_ROOT,
            text=True,
            capture_output=True,
            timeout=120,
            check=False,
        )
        if completed.returncode != 0:
            raise AssertionError(completed.stdout + completed.stderr)
        assert "fashion: metadata=455 original_manifest=76 cf_manifest=455" in completed.stdout
        assert "dry_run=true" in completed.stdout
        assert not output_dir.exists()


if __name__ == "__main__":
    sys.exit(main())
