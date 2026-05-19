#!/usr/bin/env python3
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="medical_tool_cli_") as tmp:
        manifest_path, selected = write_tool_manifest(Path(tmp))
        checks = [
            (
                "closed help",
                [sys.executable, "eval_code/medical_modality/eval_closed_vlm.py", "--help"],
                [
                    "--visual-version",
                    "--dataset",
                    "--sample-fraction",
                    "--tool-condition",
                    "--evidence-manifest",
                    "--evidence-qc-filter",
                    "--missing-evidence-policy",
                    "--image-group-labels",
                ],
                True,
            ),
            (
                "server help",
                [sys.executable, "eval_code/medical_modality/eval_server_vlm.py", "--help"],
                ["--server-url", "--judge-provider", "--tool-condition", "--evidence-manifest"],
                True,
            ),
            (
                "open help",
                [sys.executable, "eval_code/medical_modality/eval_open_vlm.py", "--help"],
                ["--dtype", "--load-in-4bit", "--tool-condition", "--evidence-manifest"],
                True,
            ),
            (
                "raw dry-run",
                [
                    sys.executable,
                    "eval_code/medical_modality/eval_closed_vlm.py",
                    "--input-mode",
                    "cf_only",
                    "--backbone-model",
                    "gpt-5",
                    "--judge-model",
                    "gpt-4o-mini",
                    "--visual-version",
                    "clean",
                    "--dataset",
                    "brats2023",
                    "--question-types",
                    "yes_no",
                    "--max-samples",
                    "1",
                    "--prime",
                    "off",
                    "--dry-run",
                ],
                ["tool_condition=raw", "response_target_count=1", "\"visual_version\": \"clean\""],
                True,
            ),
            (
                "contour missing manifest fails",
                [
                    sys.executable,
                    "eval_code/medical_modality/eval_closed_vlm.py",
                    "--input-mode",
                    "cf_only",
                    "--backbone-model",
                    "gpt-5",
                    "--judge-model",
                    "gpt-4o-mini",
                    "--tool-condition",
                    "contour",
                    "--max-samples",
                    "1",
                    "--dry-run",
                ],
                ["--evidence-manifest is required"],
                False,
            ),
            (
                "contour cf-only dry-run",
                [
                    sys.executable,
                    "eval_code/medical_modality/eval_closed_vlm.py",
                    "--input-mode",
                    "cf_only",
                    "--backbone-model",
                    "gpt-5",
                    "--judge-model",
                    "gpt-4o-mini",
                    "--visual-version",
                    selected["visual_version"],
                    "--dataset",
                    selected["dataset"],
                    "--subcategory",
                    selected["question_group_id"],
                    "--case-id",
                    selected["case_id"],
                    "--swap-direction",
                    selected["swap_direction"],
                    "--question-types",
                    "yes_no",
                    "--tool-condition",
                    "contour",
                    "--evidence-manifest",
                    str(manifest_path),
                    "--evidence-qc-filter",
                    "pass",
                    "--missing-evidence-policy",
                    "skip",
                    "--image-group-labels",
                    "on",
                    "--max-samples",
                    "1",
                    "--dry-run",
                ],
                [
                    "tool_condition=contour",
                    "eligible_pairs=1",
                    "same MRI slice",
                    "contour_overlay.png",
                    "Image group 1, view 1.",
                ],
                True,
            ),
            (
                "zoom both dry-run",
                [
                    sys.executable,
                    "eval_code/medical_modality/eval_closed_vlm.py",
                    "--input-mode",
                    "both",
                    "--backbone-model",
                    "gpt-5",
                    "--judge-model",
                    "gpt-4o-mini",
                    "--visual-version",
                    selected["visual_version"],
                    "--dataset",
                    selected["dataset"],
                    "--subcategory",
                    selected["question_group_id"],
                    "--case-id",
                    selected["case_id"],
                    "--swap-direction",
                    selected["swap_direction"],
                    "--question-types",
                    "yes_no",
                    "--tool-condition",
                    "zoom_panel",
                    "--evidence-manifest",
                    str(manifest_path),
                    "--evidence-qc-filter",
                    "pass",
                    "--missing-evidence-policy",
                    "skip",
                    "--image-group-labels",
                    "on",
                    "--max-samples",
                    "1",
                    "--dry-run",
                ],
                ["tool_condition=zoom_panel", "same MRI slice", "zoom_panel.png", "Image group 2, view 2."],
                True,
            ),
        ]
        for name, command, expected, expect_success in checks:
            output = run(name, command, expect_success=expect_success)
            for needle in expected:
                assert needle in output, f"{name}: missing {needle!r}\n{output}"
    print("medical modality eval CLI tests passed.")
    return 0


def write_tool_manifest(root: Path) -> tuple[Path, dict]:
    metadata = json.loads(
        (REPO_ROOT / "eval_results/medical_modality/metadata/medical_modality_metadata.json").read_text(encoding="utf-8")
    )
    selected = metadata[0]
    evidence_files: dict[tuple[str, str], str] = {}
    for role in ("original", "cf"):
        role_dir = root / role
        role_dir.mkdir(parents=True, exist_ok=True)
        for field in ("bbox_overlay", "contour_overlay", "crop", "zoom_panel"):
            path = role_dir / f"{field}.png"
            path.write_bytes(b"x")
            evidence_files[(role, field)] = str(path)
    manifest = [
        {
            "schema_version": "medical_label_evidence_v1",
            "benchmark": "medical_modality",
            "pair_key": selected["pair_key"],
            "question_group_id": selected["question_group_id"],
            "dataset": selected["dataset"],
            "case_id": selected["case_id"],
            "visual_version": selected["visual_version"],
            "swap_direction": selected["swap_direction"],
            "background_modality": selected["background_modality"],
            "tumor_region_modality": selected["tumor_region_modality"],
            "generator": {
                "source": "brats_segmentation_label",
                "mask_rule": "seg_gt_0",
                "crop_padding": 0.35,
            },
            "qc_status": "pass",
            "original": {
                "raw": selected["original_path"],
                "bbox_overlay": evidence_files[("original", "bbox_overlay")],
                "contour_overlay": evidence_files[("original", "contour_overlay")],
                "crop": evidence_files[("original", "crop")],
                "zoom_panel": evidence_files[("original", "zoom_panel")],
            },
            "cf": {
                "raw": selected["cf_path"],
                "bbox_overlay": evidence_files[("cf", "bbox_overlay")],
                "contour_overlay": evidence_files[("cf", "contour_overlay")],
                "crop": evidence_files[("cf", "crop")],
                "zoom_panel": evidence_files[("cf", "zoom_panel")],
            },
            "usable_tool_conditions": {
                "raw": True,
                "bbox": True,
                "contour": True,
                "crop": True,
                "zoom_panel": True,
            },
        }
    ]
    manifest_path = root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return manifest_path, selected


def run(name: str, command: list[str], *, expect_success: bool) -> str:
    result = subprocess.run(command, cwd=REPO_ROOT, text=True, capture_output=True)
    output = result.stdout + result.stderr
    if expect_success and result.returncode != 0:
        raise AssertionError(f"{name} failed: {' '.join(command)}\n{output}")
    if not expect_success and result.returncode == 0:
        raise AssertionError(f"{name} unexpectedly succeeded: {' '.join(command)}\n{output}")
    return output


if __name__ == "__main__":
    raise SystemExit(main())
