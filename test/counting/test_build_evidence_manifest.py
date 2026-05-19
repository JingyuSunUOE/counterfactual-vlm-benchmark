#!/usr/bin/env python3
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "eval_code" / "counting" / "build_evidence_manifest.py"


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


def write_image(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (32, 32), "white").save(path)


def sample_id(*, subset: str, subcategory: str, source_variant: str, stem: str, edit_type: str | None, role: str) -> str:
    parts = [subset, subcategory, source_variant, stem]
    if edit_type is not None:
        parts.append(edit_type)
    parts.append(role)
    return "_".join(parts)


def annotation(root: Path, *, index: int, edit_type: str = "wings_add") -> dict:
    original = root / "dataset" / "counting" / "bird" / "accipitriformes" / f"orig_{index}.png"
    cf = root / "cf_dataset" / "counting_cf" / "bird" / "accipitriformes" / edit_type / f"cf_{index}.png"
    write_image(original)
    write_image(cf)
    return {
        "benchmark": "counting",
        "subset": "bird",
        "group": "accipitriformes",
        "subcategory": "accipitriformes",
        "source_variant": "real",
        "count_attribute": "wings",
        "edit_type": edit_type,
        "original_path": str(original),
        "cf_path": str(cf),
        "pair_key": f"pair_{index}",
        "question_group_id": f"group_{edit_type}",
        "normal_count": 2,
        "cf_count": 4,
        "biased_count": 2,
        "count_direction": "more_than_expected",
    }


def write_evidence(
    root: Path,
    *,
    role: str,
    subcategory: str,
    sample_id_value: str,
    status: str,
    source_image_path: str,
    instances: list[dict] | None = None,
    used_fallback: bool = False,
    prompt: str = "bird of prey",
) -> None:
    sample_dir = root / role / subcategory / sample_id_value
    sample_dir.mkdir(parents=True, exist_ok=True)
    metadata = {
        "schema_version": "sam3_evidence_v1",
        "status": status,
        "benchmark": "counting",
        "sample_id": sample_id_value,
        "image_role": role,
        "source_image_path": source_image_path,
        "subcategory": subcategory,
        "source_variant": "real",
        "model_id": "facebook/sam3",
        "backend": "sam3",
        "backend_resolved": "sam3_transformers",
        "crop_padding": 0.1,
        "prompt_profile": "object",
        "prompt_text": prompt,
        "selected_prompt_text": prompt,
        "used_fallback_prompt": used_fallback,
        "num_instances": len(instances or []),
    }
    (sample_dir / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    records = []
    for item in instances or []:
        instance_id = int(item["instance_id"])
        suffix = "" if len(instances or []) == 1 else f"_{instance_id:03d}"
        files = {
            "bbox_overlay_file": f"overlays/bbox_overlay{suffix}.png",
            "outline_overlay_file": f"overlays/outline_overlay{suffix}.png",
            "crop_file": f"crops/crop_{instance_id:03d}.png",
            "zoom_panel_file": f"zooms/zoom_panel_{instance_id:03d}.png",
        }
        for rel in files.values():
            path = sample_dir / rel
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"x")
        records.append({**item, **files})
    (sample_dir / "instances.json").write_text(
        json.dumps({"schema_version": "sam3_instances_v1", "instances": records}),
        encoding="utf-8",
    )


def test_build_evidence_manifest_selection_and_fallback() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        anns = [annotation(root, index=1), annotation(root, index=2), annotation(root, index=3)]
        annotations_path = root / "annotations.json"
        annotations_path.write_text(json.dumps(anns), encoding="utf-8")
        primary_original = root / "primary_original"
        primary_cf = root / "primary_cf"
        fallback_original = root / "fallback_original"
        fallback_cf = root / "fallback_cf"
        output_json = root / "evidence.json"
        output_csv = root / "evidence.csv"

        orig1 = sample_id(subset="bird", subcategory="accipitriformes", source_variant="real", stem="orig_1", edit_type=None, role="original")
        cf1 = sample_id(subset="bird", subcategory="accipitriformes", source_variant="real", stem="cf_1", edit_type="wings_add", role="cf")
        orig2 = sample_id(subset="bird", subcategory="accipitriformes", source_variant="real", stem="orig_2", edit_type=None, role="original")
        cf2 = sample_id(subset="bird", subcategory="accipitriformes", source_variant="real", stem="cf_2", edit_type="wings_add", role="cf")
        orig3 = sample_id(subset="bird", subcategory="accipitriformes", source_variant="real", stem="orig_3", edit_type=None, role="original")
        cf3 = sample_id(subset="bird", subcategory="accipitriformes", source_variant="real", stem="cf_3", edit_type="wings_add", role="cf")

        write_evidence(
            primary_original,
            role="original",
            subcategory="accipitriformes",
            sample_id_value=orig1,
            status="success",
            source_image_path=anns[0]["original_path"],
            instances=[
                {"instance_id": 0, "mask_area_ratio": 0.03, "box_area_ratio": 0.10},
                {"instance_id": 1, "mask_area_ratio": 0.30, "box_area_ratio": 0.20},
            ],
        )
        write_evidence(
            fallback_original,
            role="original",
            subcategory="accipitriformes",
            sample_id_value=orig1,
            status="success",
            source_image_path=anns[0]["original_path"],
            instances=[{"instance_id": 0, "mask_area_ratio": 0.80, "box_area_ratio": 0.80}],
            used_fallback=True,
            prompt="fallback ignored",
        )
        write_evidence(
            primary_cf,
            role="cf",
            subcategory="accipitriformes",
            sample_id_value=cf1,
            status="success",
            source_image_path=anns[0]["cf_path"],
            instances=[{"instance_id": 0, "mask_area_ratio": 0.20, "box_area_ratio": 0.20}],
        )

        write_evidence(primary_original, role="original", subcategory="accipitriformes", sample_id_value=orig2, status="empty", source_image_path=anns[1]["original_path"])
        write_evidence(primary_cf, role="cf", subcategory="accipitriformes", sample_id_value=cf2, status="empty", source_image_path=anns[1]["cf_path"])
        write_evidence(
            fallback_original,
            role="original",
            subcategory="accipitriformes",
            sample_id_value=orig2,
            status="success",
            source_image_path=anns[1]["original_path"],
            instances=[{"instance_id": 0, "mask_area_ratio": 0.15, "box_area_ratio": 0.15}],
            used_fallback=True,
            prompt="bird",
        )
        write_evidence(
            fallback_cf,
            role="cf",
            subcategory="accipitriformes",
            sample_id_value=cf2,
            status="success",
            source_image_path=anns[1]["cf_path"],
            instances=[{"instance_id": 0, "mask_area_ratio": 0.15, "box_area_ratio": 0.15}],
            used_fallback=True,
            prompt="bird",
        )

        write_evidence(primary_original, role="original", subcategory="accipitriformes", sample_id_value=orig3, status="empty", source_image_path=anns[2]["original_path"])
        write_evidence(primary_cf, role="cf", subcategory="accipitriformes", sample_id_value=cf3, status="empty", source_image_path=anns[2]["cf_path"])

        result = run_cmd(
            [
                "--annotations",
                str(annotations_path),
                "--primary-original-root",
                str(primary_original),
                "--primary-cf-root",
                str(primary_cf),
                "--fallback-original-root",
                str(fallback_original),
                "--fallback-cf-root",
                str(fallback_cf),
                "--output-json",
                str(output_json),
                "--output-csv",
                str(output_csv),
                "--overwrite",
            ]
        )

        assert "records=3" in result.stdout
        records = json.loads(output_json.read_text(encoding="utf-8"))
        first, second, third = records
        assert first["original"]["evidence_source"] == "primary"
        assert first["original"]["selected_instance_id"] == 1
        assert first["original"]["instance_selection_method"] == "largest_mask_area_ratio"
        assert first["usable_tool_conditions"]["tool_bundle"] is True
        assert first["usable_tool_conditions"]["outline"] is True
        assert first["usable_tool_conditions_by_input_mode"]["cf_only"]["tool_bundle"] is True
        assert first["usable_tool_conditions_by_input_mode"]["both"]["outline"] is True
        assert first["qc_status"] == "review"
        assert second["original"]["evidence_source"] == "fallback"
        assert second["cf"]["evidence_source"] == "fallback"
        assert second["original"]["used_fallback_prompt"] is True
        assert second["usable_tool_conditions"]["crop"] is True
        assert second["qc_status"] == "pass"
        assert third["qc_status"] == "empty"
        assert third["usable_tool_conditions"]["crop"] is False
        assert third["usable_tool_conditions_by_input_mode"]["cf_only"]["crop"] is False
        assert third["original"]["bbox_overlay"] is None
        assert output_csv.exists()


def main() -> int:
    tests = [test_build_evidence_manifest_selection_and_fallback]
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
