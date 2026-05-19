#!/usr/bin/env python3
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import nibabel as nib
import numpy as np
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[2]


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="medical_label_evidence_") as tmp:
        root = Path(tmp)
        metadata_path, brats_root = create_fixture(root, empty_mask=False)
        output_root = root / "evidence"
        manifest_json = root / "manifest.json"
        manifest_csv = root / "manifest.csv"
        output = run(
            [
                sys.executable,
                "gen_code/medical_modality/build_label_evidence.py",
                "--metadata",
                str(metadata_path),
                "--brats23-root",
                str(brats_root),
                "--output-root",
                str(output_root),
                "--manifest-json",
                str(manifest_json),
                "--manifest-csv",
                str(manifest_csv),
                "--overwrite",
            ],
            expect_success=True,
        )
        assert "records=1" in output
        records = json.loads(manifest_json.read_text(encoding="utf-8"))["records"]
        assert len(records) == 1
        record = records[0]
        assert record["schema_version"] == "medical_label_evidence_v1"
        assert record["qc_status"] == "pass"
        assert record["generator"]["source"] == "brats_segmentation_label"
        assert record["generator"]["mask_rule"] == "seg_gt_0"
        assert record["usable_tool_conditions"]["contour"] is True
        for role in ("original", "cf"):
            for field in ("raw", "bbox_overlay", "contour_overlay", "crop", "zoom_panel"):
                path = Path(record[role][field])
                assert path.exists(), f"missing {role}.{field}: {path}"
                assert Image.open(path).size == (512, 512)
            assert Path(record[role]["mask_binary"]).exists()
        assert manifest_csv.exists()

    with tempfile.TemporaryDirectory(prefix="medical_label_empty_") as tmp:
        root = Path(tmp)
        metadata_path, brats_root = create_fixture(root, empty_mask=True)
        output = run(
            [
                sys.executable,
                "gen_code/medical_modality/build_label_evidence.py",
                "--metadata",
                str(metadata_path),
                "--brats23-root",
                str(brats_root),
                "--output-root",
                str(root / "evidence"),
                "--manifest-json",
                str(root / "manifest.json"),
                "--manifest-csv",
                str(root / "manifest.csv"),
                "--overwrite",
            ],
            expect_success=False,
        )
        assert "segmentation mask is empty" in output
    print("medical label evidence tests passed.")
    return 0


def create_fixture(root: Path, *, empty_mask: bool) -> tuple[Path, Path]:
    case_id = "BraTS-GLI-99999-000"
    brats_root = root / "brats2023"
    case_dir = brats_root / case_id
    case_dir.mkdir(parents=True)
    seg = np.zeros((16, 16, 3), dtype=np.int16)
    if not empty_mask:
        seg[4:11, 5:13, 1] = 1
    nib.save(nib.Nifti1Image(seg, affine=np.eye(4)), case_dir / f"{case_id}-seg.nii.gz")

    image_dir = root / "images"
    image_dir.mkdir()
    original_path = image_dir / "original.png"
    cf_path = image_dir / "cf.png"
    Image.new("RGB", (512, 512), (90, 90, 90)).save(original_path)
    Image.new("RGB", (512, 512), (120, 120, 120)).save(cf_path)
    metadata = [
        {
            "benchmark": "medical_modality",
            "dataset": "brats2023",
            "case_id": case_id,
            "visual_version": "clean",
            "swap_direction": "t1n_background_t2f_tumor",
            "background_modality": "T1N",
            "tumor_region_modality": "T2F",
            "question_group_id": "brats2023_t1n_background_t2f_tumor",
            "pair_key": "brats2023:clean:BraTS-GLI-99999-000:t1n_background_t2f_tumor",
            "slice_index": 1,
            "original_path": str(original_path),
            "cf_path": str(cf_path),
        }
    ]
    metadata_path = root / "metadata.json"
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    return metadata_path, brats_root


def run(command: list[str], *, expect_success: bool) -> str:
    result = subprocess.run(command, cwd=REPO_ROOT, text=True, capture_output=True)
    output = result.stdout + result.stderr
    if expect_success and result.returncode != 0:
        raise AssertionError(f"command failed: {' '.join(command)}\n{output}")
    if not expect_success and result.returncode == 0:
        raise AssertionError(f"command unexpectedly succeeded: {' '.join(command)}\n{output}")
    return output


if __name__ == "__main__":
    raise SystemExit(main())
