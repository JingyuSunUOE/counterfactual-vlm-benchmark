#!/usr/bin/env python3
from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path
import json

from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[2]


def main() -> int:
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        test_content_based_mime_detection(tmp_path)
        evidence_manifest = build_temp_evidence_manifest(tmp_path)
        checks = [
        (
            "eval help",
            [
                sys.executable,
                "eval_code/fashion_industry/eval_pipeline.py",
                "--help",
            ],
            [
                "--phase",
                "--input-mode",
                "--model-set",
                "--server-url",
                "--open-backend",
                "--server-model-id",
                "--judge-provider",
                "--server-preflight",
                "--server-framework",
                "--server-gpu-ids",
                "--server-dtype",
                "--server-tensor-parallel-size",
                "--server-hf-model-id",
                "--dry-run",
                "--report",
                "--output-root",
                "--resume",
                "--max-retries",
                "--stop-on-quota",
                "--max-consecutive-errors",
                "--judge-structured-output",
                "--closed-form-structured-output",
                "--structured-output-fallback",
                "--tool-condition",
                "--evidence-manifest",
                "--evidence-qc-filter",
                "--missing-evidence-policy",
                "--image-group-labels",
            ],
        ),
        (
            "open server dry run",
            [
                sys.executable,
                "eval_code/fashion_industry/eval_pipeline.py",
                "--phase",
                "cf",
                "--input-mode",
                "cf_only",
                "--model",
                "qwen3-vl-8b",
                "--open-backend",
                "server",
                "--server-url",
                "http://localhost:8000",
                "--limit",
                "1",
                "--dry-run",
            ],
            [
                "model=qwen3-vl-8b",
                "open_backend=server",
                "server_model_id=Qwen/Qwen3-VL-8B-Instruct",
                "pending=",
            ],
        ),
        (
            "open dashscope dry run",
            [
                sys.executable,
                "eval_code/fashion_industry/eval_pipeline.py",
                "--phase",
                "cf",
                "--input-mode",
                "cf_only",
                "--model",
                "qwen3-vl-8b",
                "--open-backend",
                "dashscope",
                "--limit",
                "1",
                "--dry-run",
            ],
            ["model=qwen3-vl-8b", "open_backend=dashscope", "pending="],
        ),
        (
            "cf-only dry run",
            [
                sys.executable,
                "eval_code/fashion_industry/eval_pipeline.py",
                "--phase",
                "cf",
                "--input-mode",
                "cf_only",
                "--model",
                "gpt-4.1",
                "--question-types",
                "all",
                "--limit",
                "1",
                "--output-root",
                str(tmp_path / "cf_only_output_root"),
                "--dry-run",
            ],
            ["model=gpt-4.1", "pending=", "output_root=", "proposed_run_dir=", "image_paths"],
        ),
        (
            "fashion subset dry run",
            [
                sys.executable,
                "eval_code/fashion_industry/eval_pipeline.py",
                "--phase",
                "cf",
                "--input-mode",
                "both",
                "--model",
                "gpt-4.1",
                "--domain",
                "fashion",
                "--key",
                "chanel",
                "--question-types",
                "Q1,Q_MC",
                "--limit",
                "2",
                "--dry-run",
            ],
            ["model=gpt-4.1", "chanel", "question_type", "image_paths"],
        ),
        (
            "raw with evidence manifest dry run",
            [
                sys.executable,
                "eval_code/fashion_industry/eval_pipeline.py",
                "--phase",
                "cf",
                "--input-mode",
                "cf_only",
                "--model",
                "gpt-4.1",
                "--domain",
                "fashion",
                "--key",
                evidence_manifest["key"],
                "--tool-condition",
                "raw",
                "--evidence-manifest",
                str(evidence_manifest["path"]),
                "--evidence-qc-filter",
                "pass",
                "--missing-evidence-policy",
                "skip",
                "--limit",
                "1",
                "--dry-run",
            ],
            ["tool_condition=raw", "Evidence filter:", "eligible_tasks=", "\"image_paths\""],
            True,
        ),
        (
            "tool condition requires manifest",
            [
                sys.executable,
                "eval_code/fashion_industry/eval_pipeline.py",
                "--phase",
                "cf",
                "--input-mode",
                "cf_only",
                "--model",
                "gpt-4.1",
                "--domain",
                "fashion",
                "--tool-condition",
                "crop",
                "--limit",
                "1",
                "--dry-run",
            ],
            ["--evidence-manifest is required"],
            False,
        ),
        (
            "tool crop dry run",
            [
                sys.executable,
                "eval_code/fashion_industry/eval_pipeline.py",
                "--phase",
                "cf",
                "--input-mode",
                "cf_only",
                "--model",
                "gpt-4.1",
                "--domain",
                "fashion",
                "--key",
                evidence_manifest["key"],
                "--tool-condition",
                "crop",
                "--evidence-manifest",
                str(evidence_manifest["path"]),
                "--evidence-qc-filter",
                "pass",
                "--missing-evidence-policy",
                "skip",
                "--image-group-labels",
                "on",
                "--limit",
                "1",
                "--dry-run",
            ],
            ["tool_condition=crop", "Evidence filter:", "eligible_tasks=", "alternative visual views", "used_image_groups", "crop.png"],
            True,
        ),
        (
            "tool both dry run",
            [
                sys.executable,
                "eval_code/fashion_industry/eval_pipeline.py",
                "--phase",
                "cf",
                "--input-mode",
                "both",
                "--model",
                "gpt-4.1",
                "--domain",
                "fashion",
                "--key",
                evidence_manifest["key"],
                "--tool-condition",
                "bbox",
                "--evidence-manifest",
                str(evidence_manifest["path"]),
                "--evidence-qc-filter",
                "pass",
                "--missing-evidence-policy",
                "skip",
                "--limit",
                "1",
                "--dry-run",
            ],
            ["tool_condition=bbox", "second image group", "\"role\": \"original\"", "\"role\": \"target\"", "bbox.png"],
            True,
        ),
        (
            "custom output root report",
            [
                sys.executable,
                "eval_code/fashion_industry/eval_pipeline.py",
                "--report",
                "--output-root",
                str(tmp_path / "empty_report_root"),
            ],
            ["No eval_results directory found:"],
        ),
        (
            "report aggregation",
            [
                sys.executable,
                "eval_code/fashion_industry/eval_pipeline.py",
                "--report",
                "--phase",
                "cf",
                "--input-mode",
                "cf_only",
            ],
            ["Report rows=", "By input_mode", "By domain"],
        ),
        ]

        for item in checks:
            if len(item) == 4:
                name, command, expected, should_succeed = item
            else:
                name, command, expected = item
                should_succeed = True
            output = run(name, command, should_succeed=should_succeed)
            require_all(name, output, expected)

    print("fashion_industry eval CLI smoke tests passed.")
    return 0


def test_content_based_mime_detection(tmp: Path) -> None:
    sys.path.insert(0, str(REPO_ROOT / "eval_code" / "fashion_industry"))
    import eval_pipeline
    import requests

    jpeg_with_png_suffix = tmp / "jpeg_bytes.png"
    Image.new("RGB", (8, 8), color=(255, 0, 0)).save(jpeg_with_png_suffix, format="JPEG")
    if eval_pipeline.guess_mime_type(str(jpeg_with_png_suffix)) != "image/jpeg":
        raise SystemExit("guess_mime_type should detect JPEG content even with a .png suffix")

    captured = {}
    original_post = requests.post

    class FakeResponse:
        status_code = 200

        @staticmethod
        def json() -> dict:
            return {"content": [{"type": "text", "text": "{ok}"}], "usage": {}}

    def fake_post(url, *, headers=None, json=None, timeout=None):
        captured["payload"] = json
        return FakeResponse()

    try:
        requests.post = fake_post
        response, error = eval_pipeline.query_claude([str(jpeg_with_png_suffix)], "What is visible?", model="claude-sonnet-4")
    finally:
        requests.post = original_post

    if error:
        raise SystemExit(f"query_claude fake request failed unexpectedly: {error}")
    if response != "{ok}":
        raise SystemExit(f"query_claude fake response mismatch: {response!r}")
    media_type = captured["payload"]["messages"][0]["content"][0]["source"]["media_type"]
    if media_type != "image/jpeg":
        raise SystemExit(f"Claude payload should use image/jpeg, got {media_type!r}")


def build_temp_evidence_manifest(tmp: Path) -> dict:
    metadata_path = REPO_ROOT / "eval_results" / "fashion_industry" / "metadata" / "fashion_cf_metadata.json"
    records = json.loads(metadata_path.read_text())
    raw = records[0]
    key = raw["brand_key"]
    original_raw = REPO_ROOT / raw["original_path"]
    cf_raw = REPO_ROOT / raw["cf_path"]
    for rel in ("original_bbox.png", "original_crop.png", "original_zoom_panel.png", "cf_bbox.png", "cf_crop.png", "cf_zoom_panel.png"):
        (tmp / rel).write_bytes(b"x")
    manifest_path = tmp / "fashion_tool_manifest.json"
    manifest_path.write_text(
        json.dumps(
            [
                {
                    "schema_version": "fashion_industry_sam3_object_evidence_v1",
                    "benchmark": "fashion_industry",
                    "domain": "fashion",
                    "key": key,
                    "subcategory": key,
                    "source_variant": "real" if "_real_" in raw["original_image"] else "ai",
                    "pair_key": f"fashion/{key}/{Path(raw['original_image']).stem}/mod{raw['mod_id']}/{raw['model']}",
                    "original_image": raw["original_image"],
                    "cf_filename": raw["filename"],
                    "mod_id": raw["mod_id"],
                    "mod_type": raw["mod_type"],
                    "model_gen": raw["model"],
                    "generator": {"model_id": "facebook/sam3", "prompt_profile": "object"},
                    "qc_status": "pass",
                    "usable_tool_conditions": {"raw": True, "bbox": True, "crop": True, "zoom_panel": True},
                    "usable_tool_conditions_by_input_mode": {
                        "cf_only": {"raw": True, "bbox": True, "crop": True, "zoom_panel": True},
                        "both": {"raw": True, "bbox": True, "crop": True, "zoom_panel": True},
                        "orig_only": {"raw": True, "bbox": True, "crop": True, "zoom_panel": True},
                    },
                    "original": {
                        "sample_id": "original",
                        "raw": str(original_raw),
                        "qc_status": "pass",
                        "bbox_overlay": str(tmp / "original_bbox.png"),
                        "crop": str(tmp / "original_crop.png"),
                        "zoom_panel": str(tmp / "original_zoom_panel.png"),
                        "selected_instance_id": 0,
                        "instances_merged": False,
                    },
                    "cf": {
                        "sample_id": "cf",
                        "raw": str(cf_raw),
                        "qc_status": "pass",
                        "bbox_overlay": str(tmp / "cf_bbox.png"),
                        "crop": str(tmp / "cf_crop.png"),
                        "zoom_panel": str(tmp / "cf_zoom_panel.png"),
                        "selected_instance_id": 0,
                        "instances_merged": False,
                    },
                }
            ],
            indent=2,
        ),
        encoding="utf-8",
    )
    return {"path": manifest_path, "key": key}


def run(name: str, command: list[str], *, should_succeed: bool = True) -> str:
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
    if should_succeed and completed.returncode != 0:
        print(output)
        raise SystemExit(f"{name} failed with exit code {completed.returncode}")
    if not should_succeed and completed.returncode == 0:
        print(output)
        raise SystemExit(f"{name} unexpectedly succeeded")
    return output


def require_all(name: str, output: str, expected: list[str]) -> None:
    missing = [item for item in expected if item not in output]
    if missing:
        raise SystemExit(f"{name} missing expected output fragments: {missing}")


if __name__ == "__main__":
    sys.exit(main())
