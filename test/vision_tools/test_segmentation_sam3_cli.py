#!/usr/bin/env python3
"""Non-live tests for the SAM3 evidence CLI."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

from PIL import Image, ImageDraw


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "segmentation_sam3.py"


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


def write_image(path: Path, color: tuple[int, int, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.new("RGB", (96, 80), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle((20, 12, 74, 66), fill=color)
    image.save(path)


def write_prompt_config(path: Path, benchmark: str = "if_exist") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "schema_version": "sam3_prompt_config_v1",
                "benchmark": benchmark,
                "categories": {
                    "camel": {
                        "object_prompt": "camel",
                        "default_region_key": "back",
                        "region_prompts": {"back": "camel back"},
                    },
                    "rabbit": {
                        "object_prompt": "rabbit",
                        "default_region_key": "head",
                        "region_prompts": {"head": "rabbit head"},
                    },
                    "empty_case": {
                        "object_prompt": "empty object",
                        "default_region_key": "body",
                        "region_prompts": {"body": "empty body"},
                    },
                    "fallback_case": {
                        "object_prompt": "empty primary object",
                        "fallback_object_prompts": ["fallback animal"],
                    },
                    "all_empty_case": {
                        "object_prompt": "empty primary object",
                        "fallback_object_prompts": ["empty fallback animal"],
                    },
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def assert_exists(path: Path) -> None:
    if not path.exists():
        raise AssertionError(f"Expected path to exist: {path}")


def assert_not_exists(path: Path) -> None:
    if path.exists():
        raise AssertionError(f"Expected path not to exist: {path}")


def test_help() -> None:
    result = run_cmd(["--help"])
    assert "--manifest" in result.stdout
    assert "--backend" in result.stdout
    assert "sam3_transformers" in result.stdout
    assert "sam3_repo" in result.stdout
    assert "--evidence-types" in result.stdout
    assert "outline" in result.stdout
    assert "--prompt-modes" in result.stdout
    assert "--progress-style" in result.stdout
    assert "--log-every" in result.stdout
    assert "--log-seconds" in result.stdout
    assert "--events-log" in result.stdout
    assert "--debug-labels" in result.stdout
    assert "--max-instances" in result.stdout
    assert "--merge-instances-over" in result.stdout
    assert "default: 3" in result.stdout


def test_dry_run_creates_no_files() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        input_root = root / "input"
        output_root = root / "output"
        config = root / "prompts.json"
        write_image(input_root / "camel" / "camel_1.png", (180, 120, 40))
        write_prompt_config(config)

        result = run_cmd(
            [
                "--benchmark",
                "if_exist",
                "--input-root",
                str(input_root),
                "--output-root",
                str(output_root),
                "--prompt-config",
                str(config),
                "--backend",
                "mock",
                "--dry-run",
            ]
        )

        assert "dry_run=true" in result.stdout
        assert "samples=1" in result.stdout
        assert "max_instances=3" in result.stdout
        assert "merge_instances_over=0" in result.stdout
        assert_not_exists(output_root)


def test_directory_scan_mock_schema_and_artifacts() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        input_root = root / "input"
        output_root = root / "output"
        config = root / "prompts.json"
        write_image(input_root / "camel" / "camel_1.png", (180, 120, 40))
        write_image(input_root / "rabbit_ai" / "rabbit_1.png", (140, 80, 180))
        write_prompt_config(config)

        run_cmd(
            [
                "--benchmark",
                "if_exist",
                "--input-root",
                str(input_root),
                "--output-root",
                str(output_root),
                "--prompt-config",
                str(config),
                "--backend",
                "mock",
                "--evidence-types",
                "mask,bbox,crop,zoom,overlay",
                "--debug-labels",
            ]
        )

        camel_dir = output_root / "original" / "camel" / "camel_camel_1"
        rabbit_dir = output_root / "original" / "rabbit" / "rabbit_ai_rabbit_1"

        for sample_dir in [camel_dir, rabbit_dir]:
            assert_exists(sample_dir / "source.png")
            assert_exists(sample_dir / "metadata.json")
            assert_exists(sample_dir / "instances.json")
            assert_exists(sample_dir / "masks" / "mask_000.png")
            assert_exists(sample_dir / "masks" / "mask_combined.png")
            assert_exists(sample_dir / "crops" / "crop_000.png")
            assert_exists(sample_dir / "zooms" / "zoom_000.png")
            assert_exists(sample_dir / "zooms" / "zoom_panel_000.png")
            assert_exists(sample_dir / "overlays" / "bbox_overlay.png")
            assert_exists(sample_dir / "overlays" / "mask_overlay.png")
            assert_exists(sample_dir / "overlays" / "mask_bbox_overlay.png")
            assert_exists(sample_dir / "debug" / "debug_overlay_with_labels.png")

            metadata = load_json(sample_dir / "metadata.json")
            instances = load_json(sample_dir / "instances.json")
            assert metadata["schema_version"] == "sam3_evidence_v1"
            assert metadata["status"] == "success"
            assert metadata["prompt_profile"] == "object"
            assert metadata["prompt_modes_requested"] == ["object"]
            assert metadata["is_multi_prompt_run"] is False
            assert Path(metadata["source_image_path"]).exists()
            assert "images/if_exist" not in metadata["source_image_path"]
            assert metadata["evidence_files"]["source"] == "source.png"
            assert metadata["evidence_files"]["bbox_overlay"] == "overlays/bbox_overlay.png"
            assert metadata["evidence_files"]["bbox_overlays"] == ["overlays/bbox_overlay.png"]
            assert len(instances["instances"]) == 1
            assert instances["instances"][0]["bbox_overlay_file"] == "overlays/bbox_overlay.png"

        rabbit_meta = load_json(rabbit_dir / "metadata.json")
        assert rabbit_meta["source_variant"] == "ai"
        summary = load_json(output_root / "run_summary.json")
        assert summary["num_samples"] == 2
        assert summary["num_success"] == 2
        assert summary["num_empty"] == 0
        assert summary["num_failed"] == 0
        assert summary["prompt_profile_distribution"] == {"object": 2}
        assert summary["prompt_modes_requested"] == ["object"]
        assert summary["backend_resolved"] == "mock"
        assert summary["mean_latency_ms"] is not None
        assert "events_log" in summary

        events_path = Path(summary["events_log"])
        assert_exists(events_path)
        events = [json.loads(line) for line in events_path.read_text(encoding="utf-8").splitlines()]
        assert events[0]["event"] == "start"
        assert any(event["event"] == "sample_done" for event in events)
        assert events[-1]["event"] == "finish"


def test_manifest_mode_preserves_sample_ids() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        input_root = root / "input"
        output_root = root / "output"
        config = root / "prompts.json"
        manifest = root / "manifest.jsonl"
        camel = input_root / "camel" / "one.png"
        rabbit = input_root / "rabbit" / "two.png"
        write_image(camel, (180, 120, 40))
        write_image(rabbit, (140, 80, 180))
        write_prompt_config(config)
        manifest.write_text(
            "\n".join(
                [
                    json.dumps(
                        {
                            "schema_version": "vision_manifest_v1",
                            "benchmark": "if_exist",
                            "sample_id": "sample_b",
                            "pair_id": "pair_b",
                            "image_path": str(rabbit),
                            "image_role": "cf",
                            "category": "if_exist",
                            "subcategory": "rabbit",
                            "source_variant": "real",
                            "prompt_key": "rabbit",
                            "prompt_mode": "region",
                            "target_region_key": "head",
                        }
                    ),
                    json.dumps(
                        {
                            "schema_version": "vision_manifest_v1",
                            "benchmark": "if_exist",
                            "sample_id": "sample_a",
                            "pair_id": "pair_a",
                            "image_path": str(camel),
                            "image_role": "cf",
                            "category": "if_exist",
                            "subcategory": "camel",
                            "source_variant": "real",
                            "prompt_key": "camel",
                            "prompt_mode": "object",
                            "target_region_key": None,
                        }
                    ),
                ]
            )
            + "\n",
            encoding="utf-8",
        )

        result = run_cmd(
            [
                "--benchmark",
                "if_exist",
                "--manifest",
                str(manifest),
                "--output-root",
                str(output_root),
                "--prompt-config",
                str(config),
                "--backend",
                "mock",
            ]
        )

        assert "sample_b" in result.stdout
        assert "sample_a" in result.stdout
        assert result.stdout.index("sample_b") < result.stdout.index("sample_a")
        sample_b_meta = load_json(output_root / "cf" / "rabbit" / "sample_b" / "metadata.json")
        sample_a_meta = load_json(output_root / "cf" / "camel" / "sample_a" / "metadata.json")
        assert sample_b_meta["sample_id"] == "sample_b"
        assert sample_b_meta["prompt_profile"] == "region_head"
        assert sample_b_meta["prompt_text"] == "rabbit head"
        assert sample_a_meta["sample_id"] == "sample_a"
        assert sample_a_meta["prompt_profile"] == "object"
        assert sample_a_meta["prompt_text"] == "camel"


def test_multi_instance_bbox_overlays_are_separate() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        input_root = root / "input"
        output_root = root / "output"
        config = root / "prompts.json"
        write_image(input_root / "camel" / "one.png", (180, 120, 40))
        write_prompt_config(config)
        data = load_json(config)
        data["categories"]["camel"]["object_prompt"] = "multi camel"
        config.write_text(json.dumps(data), encoding="utf-8")

        run_cmd(
            [
                "--benchmark",
                "if_exist",
                "--input-root",
                str(input_root),
                "--output-root",
                str(output_root),
                "--prompt-config",
                str(config),
                "--backend",
                "mock",
                "--evidence-types",
                "bbox,crop,zoom",
            ]
        )

        sample_dir = output_root / "original" / "camel" / "camel_one"
        assert_not_exists(sample_dir / "overlays" / "bbox_overlay.png")
        assert_exists(sample_dir / "overlays" / "bbox_overlay_000.png")
        assert_exists(sample_dir / "overlays" / "bbox_overlay_001.png")
        metadata = load_json(sample_dir / "metadata.json")
        instances = load_json(sample_dir / "instances.json")
        assert metadata["num_instances"] == 2
        assert metadata["evidence_files"]["bbox_overlay"] == "overlays/bbox_overlay_000.png"
        assert metadata["evidence_files"]["bbox_overlays"] == [
            "overlays/bbox_overlay_000.png",
            "overlays/bbox_overlay_001.png",
        ]
        assert instances["instances"][0]["bbox_overlay_file"] == "overlays/bbox_overlay_000.png"
        assert instances["instances"][1]["bbox_overlay_file"] == "overlays/bbox_overlay_001.png"


def test_outline_overlay_schema_and_aliases() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        input_root = root / "input"
        output_root = root / "output"
        output_root_alias = root / "output_alias"
        config = root / "prompts.json"
        write_image(input_root / "camel" / "one.png", (180, 120, 40))
        write_prompt_config(config)

        run_cmd(
            [
                "--benchmark",
                "if_exist",
                "--input-root",
                str(input_root),
                "--output-root",
                str(output_root),
                "--prompt-config",
                str(config),
                "--backend",
                "mock",
                "--evidence-types",
                "outline",
            ]
        )

        sample_dir = output_root / "original" / "camel" / "camel_one"
        assert_exists(sample_dir / "overlays" / "outline_overlay.png")
        metadata = load_json(sample_dir / "metadata.json")
        instances = load_json(sample_dir / "instances.json")
        assert metadata["evidence_types"] == ["outline"]
        assert metadata["evidence_files"]["outline_overlay"] == "overlays/outline_overlay.png"
        assert metadata["evidence_files"]["outline_overlays"] == ["overlays/outline_overlay.png"]
        assert instances["instances"][0]["outline_overlay_file"] == "overlays/outline_overlay.png"
        assert_not_exists(sample_dir / "debug" / "debug_overlay_with_labels.png")

        run_cmd(
            [
                "--benchmark",
                "if_exist",
                "--input-root",
                str(input_root),
                "--output-root",
                str(output_root_alias),
                "--prompt-config",
                str(config),
                "--backend",
                "mock",
                "--evidence-types",
                "contour_overlay",
            ]
        )
        alias_meta = load_json(output_root_alias / "original" / "camel" / "camel_one" / "metadata.json")
        assert alias_meta["evidence_types"] == ["outline"]


def test_multi_instance_outline_overlays_are_separate() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        input_root = root / "input"
        output_root = root / "output"
        config = root / "prompts.json"
        write_image(input_root / "camel" / "one.png", (180, 120, 40))
        write_prompt_config(config)
        data = load_json(config)
        data["categories"]["camel"]["object_prompt"] = "multi camel"
        config.write_text(json.dumps(data), encoding="utf-8")

        run_cmd(
            [
                "--benchmark",
                "if_exist",
                "--input-root",
                str(input_root),
                "--output-root",
                str(output_root),
                "--prompt-config",
                str(config),
                "--backend",
                "mock",
                "--evidence-types",
                "outline",
            ]
        )

        sample_dir = output_root / "original" / "camel" / "camel_one"
        assert_not_exists(sample_dir / "overlays" / "outline_overlay.png")
        assert_exists(sample_dir / "overlays" / "outline_overlay_000.png")
        assert_exists(sample_dir / "overlays" / "outline_overlay_001.png")
        metadata = load_json(sample_dir / "metadata.json")
        instances = load_json(sample_dir / "instances.json")
        assert metadata["evidence_files"]["outline_overlay"] == "overlays/outline_overlay_000.png"
        assert metadata["evidence_files"]["outline_overlays"] == [
            "overlays/outline_overlay_000.png",
            "overlays/outline_overlay_001.png",
        ]
        assert instances["instances"][0]["outline_overlay_file"] == "overlays/outline_overlay_000.png"
        assert instances["instances"][1]["outline_overlay_file"] == "overlays/outline_overlay_001.png"


def test_max_instances_limits_saved_artifacts_and_summary() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        input_root = root / "input"
        output_root = root / "output"
        config = root / "prompts.json"
        write_image(input_root / "camel" / "one.png", (180, 120, 40))
        write_prompt_config(config)
        data = load_json(config)
        data["categories"]["camel"]["object_prompt"] = "multi5 camel"
        config.write_text(json.dumps(data), encoding="utf-8")

        base_args = [
            "--benchmark",
            "if_exist",
            "--input-root",
            str(input_root),
            "--output-root",
            str(output_root),
            "--prompt-config",
            str(config),
            "--backend",
            "mock",
            "--evidence-types",
            "bbox,crop,zoom,outline",
        ]
        run_cmd([*base_args, "--max-instances", "3"])

        sample_dir = output_root / "original" / "camel" / "camel_one"
        metadata = load_json(sample_dir / "metadata.json")
        instances = load_json(sample_dir / "instances.json")
        summary = load_json(output_root / "run_summary.json")
        assert metadata["num_instances"] == 3
        assert metadata["num_instances_detected"] == 5
        assert metadata["num_instances_saved"] == 3
        assert metadata["max_instances"] == 3
        assert metadata["truncated_instance_count"] == 2
        assert metadata["instance_truncation_method"] == "mask_area_desc_box_area_desc_score_desc"
        assert len(instances["instances"]) == 3
        assert_exists(sample_dir / "overlays" / "bbox_overlay_000.png")
        assert_exists(sample_dir / "overlays" / "bbox_overlay_001.png")
        assert_exists(sample_dir / "overlays" / "bbox_overlay_002.png")
        assert_not_exists(sample_dir / "overlays" / "bbox_overlay_003.png")
        assert_exists(sample_dir / "crops" / "crop_002.png")
        assert_not_exists(sample_dir / "crops" / "crop_003.png")
        assert_exists(sample_dir / "zooms" / "zoom_panel_002.png")
        assert_not_exists(sample_dir / "zooms" / "zoom_panel_003.png")
        assert_exists(sample_dir / "overlays" / "outline_overlay_002.png")
        assert_not_exists(sample_dir / "overlays" / "outline_overlay_003.png")
        assert summary["max_instances"] == 3
        assert summary["num_samples_with_truncated_instances"] == 1
        assert summary["total_truncated_instances"] == 2
        assert summary["instance_count_detected_distribution"] == {"5": 1}
        assert summary["instance_count_saved_distribution"] == {"3": 1}

        resume_with_changed_limit = run_cmd([*base_args, "--max-instances", "2", "--resume", "--fail-fast"], check=False)
        assert resume_with_changed_limit.returncode != 0
        assert "Existing output does not match resume criteria" in resume_with_changed_limit.stderr


def test_merge_instances_over_creates_single_union_instance() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        input_root = root / "input"
        output_root = root / "output"
        config = root / "prompts.json"
        write_image(input_root / "camel" / "one.png", (180, 120, 40))
        write_prompt_config(config)
        data = load_json(config)
        data["categories"]["camel"]["object_prompt"] = "multi6 camel"
        config.write_text(json.dumps(data), encoding="utf-8")

        base_args = [
            "--benchmark",
            "if_exist",
            "--input-root",
            str(input_root),
            "--output-root",
            str(output_root),
            "--prompt-config",
            str(config),
            "--backend",
            "mock",
            "--evidence-types",
            "bbox,crop,zoom,outline",
        ]
        run_cmd([*base_args, "--merge-instances-over", "5"])

        sample_dir = output_root / "original" / "camel" / "camel_one"
        metadata = load_json(sample_dir / "metadata.json")
        instances = load_json(sample_dir / "instances.json")
        summary = load_json(output_root / "run_summary.json")
        assert metadata["num_instances"] == 1
        assert metadata["num_instances_detected"] == 6
        assert metadata["num_instances_saved"] == 1
        assert metadata["merge_instances_over"] == 5
        assert metadata["instances_merged"] is True
        assert metadata["num_instances_before_merge"] == 6
        assert metadata["num_instances_after_merge"] == 1
        assert metadata["merged_source_instance_ids"] == [0, 1, 2, 3, 4, 5]
        assert metadata["instance_selection_method"] == "merged_instances_over_threshold"
        assert metadata["truncated_instance_count"] == 0
        assert len(instances["instances"]) == 1
        assert instances["instances"][0]["source_instance_ids"] == [0, 1, 2, 3, 4, 5]
        assert instances["instances"][0]["instance_selection_method"] == "merged_instances_over_threshold"
        assert instances["instances"][0]["bbox_overlay_file"] == "overlays/bbox_overlay.png"
        assert instances["instances"][0]["crop_file"] == "crops/crop_000.png"
        assert instances["instances"][0]["zoom_panel_file"] == "zooms/zoom_panel_000.png"
        assert_exists(sample_dir / "overlays" / "bbox_overlay.png")
        assert_not_exists(sample_dir / "overlays" / "bbox_overlay_001.png")
        assert_exists(sample_dir / "crops" / "crop_000.png")
        assert_not_exists(sample_dir / "crops" / "crop_001.png")
        assert_exists(sample_dir / "zooms" / "zoom_panel_000.png")
        assert_not_exists(sample_dir / "zooms" / "zoom_panel_001.png")
        assert summary["merge_instances_over"] == 5
        assert summary["num_samples_with_merged_instances"] == 1
        assert summary["total_instances_before_merge"] == 6
        assert summary["instance_count_detected_distribution"] == {"6": 1}
        assert summary["instance_count_saved_distribution"] == {"1": 1}

        resume_with_changed_merge = run_cmd(
            [*base_args, "--merge-instances-over", "4", "--resume", "--fail-fast"],
            check=False,
        )
        assert resume_with_changed_merge.returncode != 0
        assert "Existing output does not match resume criteria" in resume_with_changed_merge.stderr


def test_merge_threshold_not_triggered_still_limits_to_max_instances() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        input_root = root / "input"
        output_root = root / "output"
        config = root / "prompts.json"
        write_image(input_root / "camel" / "one.png", (180, 120, 40))
        write_prompt_config(config)
        data = load_json(config)
        data["categories"]["camel"]["object_prompt"] = "multi5 camel"
        config.write_text(json.dumps(data), encoding="utf-8")

        run_cmd(
            [
                "--benchmark",
                "if_exist",
                "--input-root",
                str(input_root),
                "--output-root",
                str(output_root),
                "--prompt-config",
                str(config),
                "--backend",
                "mock",
                "--evidence-types",
                "bbox,crop,zoom,outline",
                "--merge-instances-over",
                "5",
                "--max-instances",
                "3",
            ]
        )

        sample_dir = output_root / "original" / "camel" / "camel_one"
        metadata = load_json(sample_dir / "metadata.json")
        instances = load_json(sample_dir / "instances.json")
        assert metadata["instances_merged"] is False
        assert metadata["num_instances_before_merge"] == 5
        assert metadata["num_instances_after_merge"] == 5
        assert metadata["num_instances_saved"] == 3
        assert metadata["truncated_instance_count"] == 2
        assert len(instances["instances"]) == 3
        assert instances["instances"][0]["instance_selection_method"] == "mask_area_desc_box_area_desc_score_desc"
        assert_exists(sample_dir / "overlays" / "bbox_overlay_002.png")
        assert_not_exists(sample_dir / "overlays" / "bbox_overlay_003.png")


def test_multi_prompt_generates_object_and_region_profiles() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        input_root = root / "input"
        output_root = root / "output"
        config = root / "prompts.json"
        write_image(input_root / "camel" / "one.png", (180, 120, 40))
        write_image(input_root / "rabbit" / "two.png", (140, 80, 180))
        write_prompt_config(config)

        result = run_cmd(
            [
                "--benchmark",
                "if_exist",
                "--input-root",
                str(input_root),
                "--output-root",
                str(output_root),
                "--prompt-config",
                str(config),
                "--backend",
                "mock",
                "--prompt-modes",
                "object,region",
            ]
        )

        assert "prompt_modes=object,region" in result.stdout
        camel_object = output_root / "object" / "original" / "camel" / "camel_one"
        camel_region = output_root / "region_back" / "original" / "camel" / "camel_one"
        rabbit_object = output_root / "object" / "original" / "rabbit" / "rabbit_two"
        rabbit_region = output_root / "region_head" / "original" / "rabbit" / "rabbit_two"
        for sample_dir in [camel_object, camel_region, rabbit_object, rabbit_region]:
            assert_exists(sample_dir / "metadata.json")

        camel_region_meta = load_json(camel_region / "metadata.json")
        rabbit_region_meta = load_json(rabbit_region / "metadata.json")
        assert camel_region_meta["prompt_profile"] == "region_back"
        assert camel_region_meta["prompt_text"] == "camel back"
        assert camel_region_meta["target_region_key"] == "back"
        assert camel_region_meta["prompt_modes_requested"] == ["object", "region"]
        assert camel_region_meta["is_multi_prompt_run"] is True
        assert rabbit_region_meta["prompt_profile"] == "region_head"
        assert rabbit_region_meta["prompt_text"] == "rabbit head"

        summary = load_json(output_root / "run_summary.json")
        assert summary["num_samples"] == 4
        assert summary["prompt_profile_distribution"] == {"object": 2, "region_back": 1, "region_head": 1}
        assert summary["prompt_modes_requested"] == ["object", "region"]
        assert summary["status_by_prompt_profile"] == {
            "object": {"success": 2},
            "region_back": {"success": 1},
            "region_head": {"success": 1},
        }
        assert summary["empty_by_prompt_profile"] == {"object": 0, "region_back": 0, "region_head": 0}
        assert summary["failed_by_prompt_profile"] == {"object": 0, "region_back": 0, "region_head": 0}


def test_progress_lines_and_off_modes() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        input_root = root / "input"
        output_root = root / "output"
        output_root_off = root / "output_off"
        config = root / "prompts.json"
        write_image(input_root / "camel" / "one.png", (180, 120, 40))
        write_prompt_config(config)

        lines = run_cmd(
            [
                "--benchmark",
                "if_exist",
                "--input-root",
                str(input_root),
                "--output-root",
                str(output_root),
                "--prompt-config",
                str(config),
                "--backend",
                "mock",
                "--progress-style",
                "lines",
                "--log-every",
                "1",
            ]
        )
        assert "success=1" in lines.stdout
        assert "camel/one" in lines.stdout or "camel_one" in lines.stdout

        off = run_cmd(
            [
                "--benchmark",
                "if_exist",
                "--input-root",
                str(input_root),
                "--output-root",
                str(output_root_off),
                "--prompt-config",
                str(config),
                "--backend",
                "mock",
                "--progress-style",
                "off",
            ]
        )
        assert "success=1" not in off.stdout
        summary = load_json(output_root_off / "run_summary.json")
        assert_exists(Path(summary["events_log"]))


def test_backend_sam3_alias_resolves_to_transformers_in_dry_run() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        input_root = root / "input"
        output_root = root / "output"
        config = root / "prompts.json"
        write_image(input_root / "camel" / "one.png", (180, 120, 40))
        write_prompt_config(config)

        result = run_cmd(
            [
                "--benchmark",
                "if_exist",
                "--input-root",
                str(input_root),
                "--output-root",
                str(output_root),
                "--prompt-config",
                str(config),
                "--backend",
                "sam3",
                "--dry-run",
            ]
        )
        assert "backend=sam3" in result.stdout
        assert "backend_resolved=sam3_transformers" in result.stdout
        assert_not_exists(output_root)


def test_sam3_repo_preflight_fails_without_package() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        input_root = root / "input"
        output_root = root / "output"
        config = root / "prompts.json"
        write_image(input_root / "camel" / "one.png", (180, 120, 40))
        write_prompt_config(config)

        result = run_cmd(
            [
                "--benchmark",
                "if_exist",
                "--input-root",
                str(input_root),
                "--output-root",
                str(output_root),
                "--prompt-config",
                str(config),
                "--backend",
                "sam3_repo",
                "--model-id",
                "facebook/sam3.1",
            ],
            check=False,
        )
        if result.returncode == 0:
            # The optional Meta SAM3 package is installed in this environment.
            return
        assert (
            "requires the official Meta SAM3 package" in result.stderr
            or "ModuleNotFoundError" in result.stderr
        )
        if output_root.exists():
            assert not list(output_root.rglob("metadata.json"))


def test_multi_prompt_target_region_override() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        input_root = root / "input"
        output_root = root / "output"
        config = root / "prompts.json"
        write_image(input_root / "rabbit" / "one.png", (140, 80, 180))
        write_prompt_config(config)

        result = run_cmd(
            [
                "--benchmark",
                "if_exist",
                "--input-root",
                str(input_root),
                "--output-root",
                str(output_root),
                "--prompt-config",
                str(config),
                "--backend",
                "mock",
                "--prompt-modes",
                "region",
                "--target-region-key",
                "head",
                "--dry-run",
            ]
        )

        assert "prompt_profile=region_head" in result.stdout
        assert "prompt=rabbit head" in result.stdout


def test_multi_prompt_missing_default_region_fails_before_output() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        input_root = root / "input"
        output_root = root / "output"
        config = root / "prompts.json"
        write_image(input_root / "camel" / "one.png", (180, 120, 40))
        config.write_text(
            json.dumps(
                {
                    "schema_version": "sam3_prompt_config_v1",
                    "benchmark": "if_exist",
                    "categories": {
                        "camel": {
                            "object_prompt": "camel",
                            "region_prompts": {"back": "camel back"},
                        }
                    },
                }
            ),
            encoding="utf-8",
        )

        result = run_cmd(
            [
                "--benchmark",
                "if_exist",
                "--input-root",
                str(input_root),
                "--output-root",
                str(output_root),
                "--prompt-config",
                str(config),
                "--backend",
                "mock",
                "--prompt-modes",
                "object,region",
            ],
            check=False,
        )

        assert result.returncode != 0
        assert "region prompt mode requires" in result.stderr
        assert_not_exists(output_root)


def test_empty_prediction_schema() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        input_root = root / "input"
        output_root = root / "output"
        config = root / "prompts.json"
        write_image(input_root / "empty_case" / "one.png", (120, 120, 120))
        write_prompt_config(config)

        base_args = [
            "--benchmark",
            "if_exist",
            "--input-root",
            str(input_root),
            "--output-root",
            str(output_root),
            "--prompt-config",
            str(config),
            "--backend",
            "mock",
        ]
        run_cmd(base_args)

        sample_dir = output_root / "original" / "empty_case" / "empty_case_one"
        metadata = load_json(sample_dir / "metadata.json")
        instances = load_json(sample_dir / "instances.json")
        assert metadata["status"] == "empty"
        assert metadata["num_instances"] == 0
        assert instances["instances"] == []
        assert_exists(sample_dir / "source.png")
        assert_not_exists(sample_dir / "masks" / "mask_000.png")


def test_fallback_prompt_success_schema() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        input_root = root / "input"
        output_root = root / "output"
        config = root / "prompts.json"
        write_image(input_root / "fallback_case" / "one.png", (120, 120, 120))
        write_prompt_config(config)

        base_args = [
            "--benchmark",
            "if_exist",
            "--input-root",
            str(input_root),
            "--output-root",
            str(output_root),
            "--prompt-config",
            str(config),
            "--backend",
            "mock",
        ]
        run_cmd(base_args)

        sample_dir = output_root / "original" / "fallback_case" / "fallback_case_one"
        metadata = load_json(sample_dir / "metadata.json")
        summary = load_json(output_root / "run_summary.json")
        events = [json.loads(line) for line in Path(summary["events_log"]).read_text(encoding="utf-8").splitlines()]
        done = next(event for event in events if event["event"] == "sample_done")
        assert metadata["status"] == "success"
        assert metadata["prompt_text"] == "fallback animal"
        assert metadata["primary_prompt_text"] == "empty primary object"
        assert metadata["selected_prompt_text"] == "fallback animal"
        assert metadata["used_fallback_prompt"] is True
        assert metadata["fallback_prompt_index"] == 0
        assert [attempt["status"] for attempt in metadata["prompt_attempts"]] == ["empty", "success"]
        assert summary["num_used_fallback_prompt"] == 1
        assert done["selected_prompt_text"] == "fallback animal"
        assert done["used_fallback_prompt"] is True
        run_cmd([*base_args, "--resume"])
        resumed_summary = load_json(output_root / "run_summary.json")
        assert resumed_summary["num_skipped_resume"] == 1


def test_all_fallback_prompts_empty_remains_empty() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        input_root = root / "input"
        output_root = root / "output"
        config = root / "prompts.json"
        write_image(input_root / "all_empty_case" / "one.png", (120, 120, 120))
        write_prompt_config(config)

        run_cmd(
            [
                "--benchmark",
                "if_exist",
                "--input-root",
                str(input_root),
                "--output-root",
                str(output_root),
                "--prompt-config",
                str(config),
                "--backend",
                "mock",
            ]
        )

        sample_dir = output_root / "original" / "all_empty_case" / "all_empty_case_one"
        metadata = load_json(sample_dir / "metadata.json")
        summary = load_json(output_root / "run_summary.json")
        assert metadata["status"] == "empty"
        assert metadata["prompt_text"] == "empty fallback animal"
        assert metadata["selected_prompt_text"] == "empty fallback animal"
        assert metadata["used_fallback_prompt"] is False
        assert metadata["fallback_prompt_index"] is None
        assert [attempt["status"] for attempt in metadata["prompt_attempts"]] == ["empty", "empty"]
        assert summary["num_used_fallback_prompt"] == 0


def test_resume_and_changed_source_conflict() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        input_root = root / "input"
        output_root = root / "output"
        config = root / "prompts.json"
        image_path = input_root / "camel" / "one.png"
        write_image(image_path, (180, 120, 40))
        write_prompt_config(config)

        base_args = [
            "--benchmark",
            "if_exist",
            "--input-root",
            str(input_root),
            "--output-root",
            str(output_root),
            "--prompt-config",
            str(config),
            "--backend",
            "mock",
        ]
        run_cmd(base_args)
        run_cmd([*base_args, "--resume"])
        summary = load_json(output_root / "run_summary.json")
        assert summary["num_skipped_resume"] == 1

        write_image(image_path, (20, 120, 210))
        conflict = run_cmd([*base_args, "--resume", "--fail-fast"], check=False)
        assert conflict.returncode != 0
        assert "Existing output does not match resume criteria" in conflict.stderr

        run_cmd([*base_args, "--resume", "--overwrite"])
        summary = load_json(output_root / "run_summary.json")
        assert summary["num_success"] == 1


def test_multi_prompt_resume_keeps_profiles_separate() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        input_root = root / "input"
        output_root = root / "output"
        config = root / "prompts.json"
        write_image(input_root / "camel" / "one.png", (180, 120, 40))
        write_prompt_config(config)

        base_args = [
            "--benchmark",
            "if_exist",
            "--input-root",
            str(input_root),
            "--output-root",
            str(output_root),
            "--prompt-config",
            str(config),
            "--backend",
            "mock",
            "--prompt-modes",
            "object,region",
        ]
        run_cmd(base_args)
        run_cmd([*base_args, "--resume"])
        summary = load_json(output_root / "run_summary.json")
        assert summary["num_skipped_resume"] == 2
        assert summary["prompt_profile_distribution"] == {"object": 1, "region_back": 1}


def test_duplicate_manifest_ids_abort() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        input_root = root / "input"
        output_root = root / "output"
        config = root / "prompts.json"
        manifest = root / "manifest.jsonl"
        image_path = input_root / "camel" / "one.png"
        write_image(image_path, (180, 120, 40))
        write_prompt_config(config)
        row = {
            "schema_version": "vision_manifest_v1",
            "benchmark": "if_exist",
            "sample_id": "duplicate",
            "pair_id": "pair",
            "image_path": str(image_path),
            "image_role": "original",
            "category": "if_exist",
            "subcategory": "camel",
            "source_variant": "real",
            "prompt_key": "camel",
            "prompt_mode": "object",
            "target_region_key": None,
        }
        manifest.write_text(json.dumps(row) + "\n" + json.dumps(row) + "\n", encoding="utf-8")

        result = run_cmd(
            [
                "--benchmark",
                "if_exist",
                "--manifest",
                str(manifest),
                "--output-root",
                str(output_root),
                "--prompt-config",
                str(config),
                "--backend",
                "mock",
            ],
            check=False,
        )
        assert result.returncode != 0
        assert "Duplicate sample_id values within the same prompt profile" in result.stderr
        assert_not_exists(output_root)


def test_duplicate_manifest_ids_allowed_across_prompt_profiles() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        input_root = root / "input"
        output_root = root / "output"
        config = root / "prompts.json"
        manifest = root / "manifest.jsonl"
        image_path = input_root / "camel" / "one.png"
        write_image(image_path, (180, 120, 40))
        write_prompt_config(config)
        base = {
            "schema_version": "vision_manifest_v1",
            "benchmark": "if_exist",
            "sample_id": "same_image",
            "pair_id": "pair",
            "image_path": str(image_path),
            "image_role": "original",
            "category": "if_exist",
            "subcategory": "camel",
            "source_variant": "real",
            "prompt_key": "camel",
        }
        object_row = {**base, "prompt_mode": "object", "target_region_key": None}
        region_row = {**base, "prompt_mode": "region", "target_region_key": "back"}
        manifest.write_text(json.dumps(object_row) + "\n" + json.dumps(region_row) + "\n", encoding="utf-8")

        run_cmd(
            [
                "--benchmark",
                "if_exist",
                "--manifest",
                str(manifest),
                "--output-root",
                str(output_root),
                "--prompt-config",
                str(config),
                "--backend",
                "mock",
            ]
        )

        assert_exists(output_root / "object" / "original" / "camel" / "same_image" / "metadata.json")
        assert_exists(output_root / "region_back" / "original" / "camel" / "same_image" / "metadata.json")
        object_meta = load_json(output_root / "object" / "original" / "camel" / "same_image" / "metadata.json")
        region_meta = load_json(output_root / "region_back" / "original" / "camel" / "same_image" / "metadata.json")
        assert object_meta["prompt_profile"] == "object"
        assert region_meta["prompt_profile"] == "region_back"


def main() -> int:
    tests = [
        test_help,
        test_dry_run_creates_no_files,
        test_directory_scan_mock_schema_and_artifacts,
        test_manifest_mode_preserves_sample_ids,
        test_multi_instance_bbox_overlays_are_separate,
        test_outline_overlay_schema_and_aliases,
        test_multi_instance_outline_overlays_are_separate,
        test_max_instances_limits_saved_artifacts_and_summary,
        test_merge_instances_over_creates_single_union_instance,
        test_merge_threshold_not_triggered_still_limits_to_max_instances,
        test_multi_prompt_generates_object_and_region_profiles,
        test_progress_lines_and_off_modes,
        test_backend_sam3_alias_resolves_to_transformers_in_dry_run,
        test_sam3_repo_preflight_fails_without_package,
        test_multi_prompt_target_region_override,
        test_multi_prompt_missing_default_region_fails_before_output,
        test_empty_prediction_schema,
        test_fallback_prompt_success_schema,
        test_all_fallback_prompts_empty_remains_empty,
        test_resume_and_changed_source_conflict,
        test_multi_prompt_resume_keeps_profiles_separate,
        test_duplicate_manifest_ids_abort,
        test_duplicate_manifest_ids_allowed_across_prompt_profiles,
    ]
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
