#!/usr/bin/env python3
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "eval_code" / "fashion_industry" / "build_evidence_manifest.py"


def run_cmd(args: list[str], *, check: bool = True) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        [sys.executable, str(SCRIPT), *args],
        cwd=REPO_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if check and result.returncode != 0:
        raise AssertionError(
            f"Command failed with exit code {result.returncode}\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
        )
    return result


def write_file(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"x")


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def metadata_record(root: Path, *, filename: str = "brand_real_01_cf_mod1_color_gemini.png") -> dict:
    original = root / "dataset" / "fashion_dataset" / "brand" / "brand_real_01.png"
    cf = root / "cf_dataset" / "fashion_cf" / "brand" / filename
    write_file(original)
    write_file(cf)
    return {
        "filename": filename,
        "brand_key": "brand",
        "brand_name": "Brand",
        "source_type": "cf_from_real",
        "model": "gemini",
        "mod_id": 1,
        "mod_type": "color",
        "ground_truth": {"color": "blue"},
        "bias_answer": {"color": "red"},
        "original_image": original.name,
        "success": True,
        "original_path": str(original),
        "cf_path": str(cf),
    }


def vision_rows(raw: dict) -> tuple[dict, dict]:
    original_row = {
        "schema_version": "vision_manifest_v1",
        "benchmark": "fashion",
        "sample_id": "fashion_brand_real_brand_real_01_original",
        "image_path": raw["original_path"],
        "image_role": "original",
        "category": "fashion",
        "subcategory": "brand",
        "source_variant": "real",
        "prompt_key": "brand",
        "prompt_mode": "object",
        "domain": "fashion",
        "cf_filename": raw["filename"],
        "original_image": raw["original_image"],
    }
    cf_row = {
        **original_row,
        "sample_id": "fashion_brand_real_brand_real_01_cf_mod1_color_gemini_cf",
        "image_path": raw["cf_path"],
        "image_role": "cf",
        "cf_filename": raw["filename"],
    }
    return original_row, cf_row


def write_evidence(root: Path, *, role: str, sample_id: str, status: str, instances: list[dict] | None) -> None:
    sample_dir = root / role / "brand" / sample_id
    sample_dir.mkdir(parents=True, exist_ok=True)
    metadata = {
        "schema_version": "sam3_evidence_v1",
        "status": status,
        "sample_id": sample_id,
        "image_role": role,
        "source_image_path": "unused.png",
        "model_id": "facebook/sam3",
        "backend": "sam3_transformers",
        "backend_resolved": "sam3_transformers",
        "dtype": "float32",
        "crop_padding": 0.05,
        "max_instances": 3,
        "merge_instances_over": 5,
        "prompt_profile": "object",
        "num_instances": len(instances or []),
        "num_instances_saved": len(instances or []),
        "instances_merged": bool(instances and instances[0].get("instances_merged")),
        "selected_prompt_text": "brand logo",
    }
    (sample_dir / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    rows = []
    for item in instances or []:
        instance_id = int(item["instance_id"])
        suffix = "" if len(instances or []) == 1 else f"_{instance_id:03d}"
        files = {
            "bbox_overlay_file": f"overlays/bbox_overlay{suffix}.png",
            "crop_file": f"crops/crop_{instance_id:03d}.png",
            "zoom_panel_file": f"zooms/zoom_panel_{instance_id:03d}.png",
        }
        for rel in files.values():
            write_file(sample_dir / rel)
        rows.append({**item, **files})
    (sample_dir / "instances.json").write_text(
        json.dumps({"schema_version": "sam3_instances_v1", "instances": rows}),
        encoding="utf-8",
    )


def test_build_evidence_manifest_qc_and_usability() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        metadata_dir = root / "metadata"
        output_dir = root / "out"
        raw = metadata_record(root)
        metadata_dir.mkdir(parents=True, exist_ok=True)
        (metadata_dir / "fashion_cf_metadata.json").write_text(json.dumps([raw]), encoding="utf-8")
        original_row, cf_row = vision_rows(raw)
        write_jsonl(metadata_dir / "fashion_original_vision_manifest.jsonl", [original_row])
        write_jsonl(metadata_dir / "fashion_cf_vision_manifest.jsonl", [cf_row])
        original_root = root / "sam_original"
        cf_root = root / "sam_cf"
        write_evidence(
            original_root,
            role="original",
            sample_id=original_row["sample_id"],
            status="success",
            instances=[{"instance_id": 0, "mask_area_ratio": 0.2, "box_area_ratio": 0.3}],
        )
        write_evidence(
            cf_root,
            role="cf",
            sample_id=cf_row["sample_id"],
            status="success",
            instances=[
                {"instance_id": 0, "mask_area_ratio": 0.1, "box_area_ratio": 0.2},
                {"instance_id": 1, "mask_area_ratio": 0.4, "box_area_ratio": 0.5},
            ],
        )

        result = run_cmd(
            [
                "--domain",
                "fashion",
                "--metadata-dir",
                str(metadata_dir),
                "--output-dir",
                str(output_dir),
                "--fashion-original-root",
                str(original_root),
                "--fashion-cf-root",
                str(cf_root),
                "--overwrite",
            ]
        )
        assert "fashion: records=1" in result.stdout
        records = json.loads((output_dir / "fashion_sam3_object_evidence_manifest_pad005_max3_merge5_fallback.json").read_text())
        record = records[0]
        assert record["qc_status"] == "review"
        assert record["original"]["qc_status"] == "pass"
        assert record["cf"]["qc_status"] == "review"
        assert record["cf"]["selected_instance_id"] == 1
        assert record["usable_tool_conditions_by_input_mode"]["cf_only"]["crop"] is True
        assert record["usable_tool_conditions_by_input_mode"]["both"]["zoom_panel"] is True
        assert (output_dir / "fashion_sam3_object_evidence_manifest_pad005_max3_merge5_fallback.csv").exists()


def test_empty_role_is_not_usable_for_tool_conditions() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        metadata_dir = root / "metadata"
        output_dir = root / "out"
        raw = metadata_record(root)
        metadata_dir.mkdir(parents=True, exist_ok=True)
        (metadata_dir / "fashion_cf_metadata.json").write_text(json.dumps([raw]), encoding="utf-8")
        original_row, cf_row = vision_rows(raw)
        write_jsonl(metadata_dir / "fashion_original_vision_manifest.jsonl", [original_row])
        write_jsonl(metadata_dir / "fashion_cf_vision_manifest.jsonl", [cf_row])
        original_root = root / "sam_original"
        cf_root = root / "sam_cf"
        write_evidence(original_root, role="original", sample_id=original_row["sample_id"], status="empty", instances=None)
        write_evidence(
            cf_root,
            role="cf",
            sample_id=cf_row["sample_id"],
            status="success",
            instances=[{"instance_id": 0, "mask_area_ratio": 0.2, "box_area_ratio": 0.3}],
        )

        run_cmd(
            [
                "--domain",
                "fashion",
                "--metadata-dir",
                str(metadata_dir),
                "--output-dir",
                str(output_dir),
                "--fashion-original-root",
                str(original_root),
                "--fashion-cf-root",
                str(cf_root),
                "--overwrite",
            ]
        )
        records = json.loads((output_dir / "fashion_sam3_object_evidence_manifest_pad005_max3_merge5_fallback.json").read_text())
        record = records[0]
        assert record["qc_status"] == "empty"
        assert record["usable_tool_conditions_by_input_mode"]["cf_only"]["bbox"] is True
        assert record["usable_tool_conditions_by_input_mode"]["both"]["bbox"] is False


def main() -> int:
    tests = [
        test_build_evidence_manifest_qc_and_usability,
        test_empty_role_is_not_usable_for_tool_conditions,
    ]
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
