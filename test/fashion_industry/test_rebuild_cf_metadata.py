#!/usr/bin/env python3
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from eval_code.fashion_industry.rebuild_cf_metadata import (  # noqa: E402
    DomainSpec,
    MetadataRebuildError,
    activate_metadata,
    rebuild_domain,
)


def main() -> int:
    test_rebuild_synthetic_metadata()
    test_missing_expected_cf_fails_strict()
    test_duplicate_old_metadata_fails()
    test_activate_backs_up_active_metadata()
    test_cli_dry_run_writes_nothing()
    print("fashion_industry metadata rebuild tests passed.")
    return 0


def make_spec(root: Path, *, old_records: list[dict] | None = None, missing_gpt: bool = False) -> DomainSpec:
    dataset_root = root / "dataset"
    cf_root = root / "cf"
    metadata_root = root / "metadata"
    key = "sample_logo"
    original = dataset_root / key / "sample_logo_real_01.png"
    original.parent.mkdir(parents=True, exist_ok=True)
    original.write_bytes(b"not-a-real-image")
    cf_dir = cf_root / key
    cf_dir.mkdir(parents=True, exist_ok=True)
    (cf_dir / "sample_logo_real_01_cf_mod1_color_change_gemini.png").write_bytes(b"gemini")
    if not missing_gpt:
        (cf_dir / "sample_logo_real_01_cf_mod1_color_change_gpt.png").write_bytes(b"gpt")
    old_metadata_path = metadata_root / "sample_metadata.json"
    metadata_root.mkdir(parents=True, exist_ok=True)
    old_metadata_path.write_text(json.dumps(old_records or []))
    return DomainSpec(
        domain="fashion",
        definitions={
            key: {
                "name": "Sample logo",
                "category": "fashion_graphic",
                "mods": [
                    {
                        "id": 1,
                        "type": "color_change",
                        "desc": "blue logo",
                        "gt": {"color": "blue"},
                        "bias": {"color": "red"},
                        "edit_prompt": "edit prompt",
                        "prompts": ["gpt prompt"],
                    }
                ],
            }
        },
        dataset_root=dataset_root,
        cf_root=cf_root,
        old_metadata_path=old_metadata_path,
        rebuilt_filename="rebuilt.json",
        active_filename="active.json",
        key_field="brand_key",
        name_field="brand_name",
        expected_total=2,
    )


def test_rebuild_synthetic_metadata() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        old_records = [
            {
                "filename": "sample_logo_real_01_cf_mod1_color_change_gemini.png",
                "success": False,
                "timestamp": "old-ts",
            }
        ]
        spec = make_spec(root, old_records=old_records)
        records, report = rebuild_domain(spec, strict=True, rebuild_timestamp="new-ts")
        assert len(records) == 2
        by_name = {record["filename"]: record for record in records}
        gemini = by_name["sample_logo_real_01_cf_mod1_color_change_gemini.png"]
        gpt = by_name["sample_logo_real_01_cf_mod1_color_change_gpt.png"]
        assert gemini["success"] is True
        assert gemini["previous_metadata_status"] == "failed_but_file_exists"
        assert gemini["timestamp"] == "old-ts"
        assert gpt["previous_metadata_status"] == "missing_from_old_metadata"
        assert gpt["prompt"] == "gpt prompt"
        assert report["rebuilt_count"] == 2
        assert report["missing_cf"] == []


def test_missing_expected_cf_fails_strict() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        spec = make_spec(Path(tmp), missing_gpt=True)
        try:
            rebuild_domain(spec, strict=True, rebuild_timestamp="new-ts")
        except MetadataRebuildError as exc:
            assert "missing expected cf files=1" in str(exc)
        else:
            raise AssertionError("strict rebuild should fail when an expected CF file is missing")


def test_duplicate_old_metadata_fails() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        duplicate = {
            "filename": "sample_logo_real_01_cf_mod1_color_change_gemini.png",
            "success": True,
        }
        spec = make_spec(Path(tmp), old_records=[duplicate, duplicate])
        try:
            rebuild_domain(spec, strict=True, rebuild_timestamp="new-ts")
        except MetadataRebuildError as exc:
            assert "Duplicate filenames" in str(exc)
        else:
            raise AssertionError("duplicate old metadata should fail")


def test_activate_backs_up_active_metadata() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        spec = make_spec(root)
        output_dir = root / "out"
        backup_dir = root / "backup"
        output_dir.mkdir()
        (output_dir / spec.rebuilt_filename).write_text('[{"filename":"new"}]\n')
        (output_dir / spec.active_filename).write_text('[{"filename":"old"}]\n')
        actions = activate_metadata(
            [spec],
            output_dir,
            backup_dir,
            overwrite=True,
            timestamp_label="20260101T000000Z",
        )
        assert len(actions) == 1
        assert json.loads((output_dir / spec.active_filename).read_text())[0]["filename"] == "new"
        backups = list(backup_dir.glob("active_20260101T000000Z.json"))
        assert len(backups) == 1
        assert json.loads(backups[0].read_text())[0]["filename"] == "old"


def test_cli_dry_run_writes_nothing() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        output_dir = Path(tmp) / "dry_run_output"
        completed = subprocess.run(
            [
                sys.executable,
                "eval_code/fashion_industry/rebuild_cf_metadata.py",
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
        assert "dry_run=true" in completed.stdout
        assert not output_dir.exists()


if __name__ == "__main__":
    sys.exit(main())
