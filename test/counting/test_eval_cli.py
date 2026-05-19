#!/usr/bin/env python3
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="counting_tool_cli_") as tmp:
        manifest_path, selected = write_tool_manifest(Path(tmp))
        checks = [
            (
                "closed help",
                [sys.executable, "eval_code/counting/eval_closed_vlm.py", "--help"],
                [
                    "--annotations",
                    "--subset",
                    "--edit-type",
                    "--count-direction",
                    "--tool-condition",
                    "--evidence-manifest",
                    "--evidence-qc-filter",
                    "--missing-evidence-policy",
                    "--image-group-labels",
                    "--provider-cache",
                    "--judge-structured-output",
                    "--closed-form-structured-output",
                    "--structured-output-fallback",
                ],
                True,
            ),
            (
                "server help",
                [sys.executable, "eval_code/counting/eval_server_vlm.py", "--help"],
                [
                    "--annotations",
                    "--server-url",
                    "--judge-provider",
                    "--subset",
                    "--count-direction",
                    "--tool-condition",
                    "--evidence-manifest",
                    "--judge-structured-output",
                    "--closed-form-structured-output",
                    "--structured-output-fallback",
                ],
                True,
            ),
            (
                "open help",
                [sys.executable, "eval_code/counting/eval_open_vlm.py", "--help"],
                ["--annotations", "--dtype", "--load-in-4bit", "--subset", "--edit-type", "--tool-condition"],
                True,
            ),
            (
                "closed filtered dry-run",
                [
                    sys.executable,
                    "eval_code/counting/eval_closed_vlm.py",
                    "--input-mode",
                    "cf_only",
                    "--backbone-model",
                    "gpt-4.1",
                    "--judge-model",
                    "gpt-4o-mini",
                    "--subset",
                    "insect",
                    "--subcategory",
                    "diptera",
                    "--edit-type",
                    "wings_remove",
                    "--count-direction",
                    "fewer_than_expected",
                    "--question-types",
                    "multiple_choice",
                    "--max-samples",
                    "2",
                    "--prime",
                    "off",
                    "--dry-run",
                ],
                [
                    "image_pairs=2",
                    "response_target_count=2",
                    "counting_insect_diptera_wings_remove",
                    "Answer with only A, B, C, or D.",
                ],
                True,
            ),
            (
                "server both dry-run",
                [
                    sys.executable,
                    "eval_code/counting/eval_server_vlm.py",
                    "--input-mode",
                    "both",
                    "--backbone-model",
                    "qwen3-vl-8b",
                    "--server-url",
                    "http://localhost:8000",
                    "--judge-provider",
                    "openai",
                    "--judge-model",
                    "gpt-4o-mini",
                    "--subset",
                    "bird",
                    "--edit-type",
                    "wings_add",
                    "--question-types",
                    "multiple_choice",
                    "--max-samples",
                    "1",
                    "--prime",
                    "off",
                    "--dry-run",
                ],
                ["Two related images are provided", "Compare the visible target attribute", "image_pairs=1", "response_target_count=1"],
                True,
            ),
            (
                "closed tool-condition dry-run",
                [
                    sys.executable,
                    "eval_code/counting/eval_closed_vlm.py",
                    "--input-mode",
                    "cf_only",
                    "--backbone-model",
                    "gpt-4.1",
                    "--judge-model",
                    "gpt-4o-mini",
                    "--tool-condition",
                    "outline",
                    "--evidence-manifest",
                    str(manifest_path),
                    "--evidence-qc-filter",
                    "pass",
                    "--missing-evidence-policy",
                    "skip",
                    "--image-group-labels",
                    "on",
                    "--subset",
                    selected["subset"],
                    "--subcategory",
                    selected["subcategory"],
                    "--edit-type",
                    selected["edit_type"],
                    "--source-variant",
                    selected["source_variant"],
                    "--question-types",
                    "multiple_choice",
                    "--max-samples",
                    "1",
                    "--prime",
                    "off",
                    "--dry-run",
                ],
                [
                    "tool_condition=outline",
                    "eligible_pairs=1",
                    "alternative visual views",
                    "\"tool_condition\": \"outline\"",
                    "outline_overlay.png",
                    "Image group 1, view 1.",
                ],
                True,
            ),
            (
                "closed crop missing manifest fails",
                [
                    sys.executable,
                    "eval_code/counting/eval_closed_vlm.py",
                    "--input-mode",
                    "cf_only",
                    "--backbone-model",
                    "gpt-4.1",
                    "--judge-model",
                    "gpt-4o-mini",
                    "--tool-condition",
                    "crop",
                    "--max-samples",
                    "1",
                    "--dry-run",
                ],
                ["--evidence-manifest is required"],
                False,
            ),
        ]
        for name, command, expected, expect_success in checks:
            output = run(name, command, expect_success=expect_success)
            require_all(name, output, expected)
    print("counting eval CLI tests passed.")
    return 0


def write_tool_manifest(root: Path) -> tuple[Path, dict]:
    annotations = json.loads(
        (REPO_ROOT / "eval_results/counting/metadata/counting_count_annotations.json").read_text(encoding="utf-8")
    )
    selected = next(item for item in annotations if item["subset"] == "hand_paw")
    evidence_files: dict[tuple[str, str], str] = {}
    for role in ("original", "cf"):
        role_dir = root / role
        role_dir.mkdir(parents=True, exist_ok=True)
        for field in ("bbox_overlay", "crop", "zoom_panel", "outline_overlay"):
            path = role_dir / f"{field}.png"
            path.write_bytes(b"x")
            evidence_files[(role, field)] = str(path)
    manifest = [
        {
            "schema_version": "counting_sam3_object_evidence_v1",
            "benchmark": "counting",
            "pair_key": selected["pair_key"],
            "question_group_id": selected["question_group_id"],
            "subset": selected["subset"],
            "subcategory": selected["subcategory"],
            "source_variant": selected["source_variant"],
            "edit_type": selected["edit_type"],
            "count_direction": selected["count_direction"],
            "generator": {"model_id": "facebook/sam3", "prompt_profile": "object"},
            "qc_status": "pass",
            "usable_tool_conditions": {
                "raw": True,
                "bbox": True,
                "crop": True,
                "zoom_panel": True,
                "outline": True,
                "tool_bundle": True,
            },
            "usable_tool_conditions_by_input_mode": {
                mode: {
                    "raw": True,
                    "bbox": True,
                    "crop": True,
                    "zoom_panel": True,
                    "outline": True,
                    "tool_bundle": True,
                }
                for mode in ("cf_only", "both", "orig_only")
            },
            "original": {
                "raw": selected["original_path"],
                "qc_status": "pass",
                "selected_instance_id": 0,
                "selected_prompt_text": "hand",
                "bbox_overlay": evidence_files[("original", "bbox_overlay")],
                "crop": evidence_files[("original", "crop")],
                "zoom_panel": evidence_files[("original", "zoom_panel")],
                "outline_overlay": evidence_files[("original", "outline_overlay")],
            },
            "cf": {
                "raw": selected["cf_path"],
                "qc_status": "pass",
                "selected_instance_id": 0,
                "selected_prompt_text": "hand",
                "bbox_overlay": evidence_files[("cf", "bbox_overlay")],
                "crop": evidence_files[("cf", "crop")],
                "zoom_panel": evidence_files[("cf", "zoom_panel")],
                "outline_overlay": evidence_files[("cf", "outline_overlay")],
            },
        }
    ]
    manifest_path = root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return manifest_path, selected


def run(name: str, command: list[str], *, expect_success: bool = True) -> str:
    print(f"[RUN] {name}: {' '.join(command)}")
    completed = subprocess.run(
        command,
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        timeout=180,
        check=False,
    )
    output = completed.stdout + completed.stderr
    if expect_success and completed.returncode != 0:
        print(output)
        raise SystemExit(f"{name} failed with exit code {completed.returncode}")
    if not expect_success and completed.returncode == 0:
        raise SystemExit(f"{name} unexpectedly succeeded")
    return output


def require_all(name: str, output: str, expected: list[str]) -> None:
    missing = [fragment for fragment in expected if fragment not in output]
    if missing:
        raise SystemExit(f"{name} missing expected fragments: {missing}")


if __name__ == "__main__":
    sys.exit(main())
