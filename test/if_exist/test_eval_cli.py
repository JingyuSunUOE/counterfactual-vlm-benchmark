#!/usr/bin/env python3
from __future__ import annotations

import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


def main() -> int:
    checks = [
        (
            "closed runner help",
            [sys.executable, "eval_code/if_exist/eval_closed_vlm.py", "--help"],
            [
                "--eval-mode",
                "--input-mode",
                "--backbone-model",
                "--judge-model",
                "--output-root",
                "--dry-run",
                "--category",
                "--subcategory",
                "--source-variant",
                "--question-types",
                "--tool-condition",
                "--evidence-manifest",
                "--evidence-qc-filter",
                "--missing-evidence-policy",
                "--image-group-labels",
                "--report",
                "--no-progress",
                "--prime",
                "--resume",
                "--max-retries",
                "--stop-on-quota",
                "--max-consecutive-errors",
                "--judge-structured-output",
                "--closed-form-structured-output",
                "--structured-output-fallback",
            ],
        ),
        (
            "open runner help",
            [sys.executable, "eval_code/if_exist/eval_open_vlm.py", "--help"],
            [
                "--eval-mode",
                "--input-mode",
                "--backbone-model",
                "--judge-model",
                "--load-in-4bit",
                "--dry-run",
                "--category",
                "--subcategory",
                "--source-variant",
                "--question-types",
                "--tool-condition",
                "--evidence-manifest",
                "--evidence-qc-filter",
                "--missing-evidence-policy",
                "--image-group-labels",
                "--report",
                "--no-progress",
                "--prime",
                "--resume",
                "--max-retries",
                "--stop-on-quota",
                "--max-consecutive-errors",
                "--judge-structured-output",
                "--closed-form-structured-output",
                "--structured-output-fallback",
            ],
        ),
        (
            "server runner help",
            [sys.executable, "eval_code/if_exist/eval_server_vlm.py", "--help"],
            [
                "--input-mode",
                "--backbone-model",
                "--server-url",
                "--server-api-key",
                "--server-model-id",
                "--judge-provider",
                "--judge-server-url",
                "--server-preflight",
                "--tool-condition",
                "--evidence-manifest",
                "--evidence-qc-filter",
                "--missing-evidence-policy",
                "--image-group-labels",
                "--server-framework",
                "--server-gpu-ids",
                "--server-dtype",
                "--server-tensor-parallel-size",
                "--server-hf-model-id",
                "--dry-run",
                "--prime",
                "--resume",
                "--max-retries",
                "--stop-on-quota",
                "--max-consecutive-errors",
                "--judge-structured-output",
                "--closed-form-structured-output",
                "--structured-output-fallback",
            ],
        ),
        (
            "closed runner filtered dry-run",
            [
                sys.executable,
                "eval_code/if_exist/eval_closed_vlm.py",
                "--input-mode",
                "cf_only",
                "--backbone-model",
                "gpt-4.1",
                "--judge-model",
                "gpt-4o-mini",
                "--category",
                "if_exist",
                "--subcategory",
                "camel",
                "--source-variant",
                "real",
                "--question-types",
                "yes_no,multiple_choice",
                "--max-samples",
                "2",
                "--prime",
                "on",
                "--dry-run",
            ],
            [
                "dry_run script_type=closed",
                "eval_mode=cf_only",
                "image_pairs=2",
                "response_target_count=4",
                "\"subcategory\": \"camel\"",
                "\"source_variant\": \"real\"",
                "\"question_type\": \"yes_no\"",
                "\"question_type\": \"multiple_choice\"",
                "This is a well-known, classic camel",
            ],
        ),
        (
            "closed runner tool-condition dry-run",
            [
                sys.executable,
                "eval_code/if_exist/eval_closed_vlm.py",
                "--input-mode",
                "cf_only",
                "--backbone-model",
                "gpt-4.1",
                "--judge-model",
                "gpt-4o-mini",
                "--tool-condition",
                "crop",
                "--evidence-manifest",
                "eval_results/if_exist/metadata/if_exist_sam3_object_evidence_manifest.json",
                "--category",
                "if_exist",
                "--subcategory",
                "camel",
                "--source-variant",
                "real",
                "--question-types",
                "yes_no",
                "--max-samples",
                "1",
                "--prime",
                "off",
                "--image-group-labels",
                "on",
                "--dry-run",
            ],
            [
                "tool_condition=crop",
                "Evidence filter:",
                "eligible_pairs=1",
                "\"image_group_labels\": \"on\"",
                "Image group 1, view 1.",
                "Image group 1, view 2.",
                "alternative visual views",
                "\"tool_condition\": \"crop\"",
                "\"used_image_groups\"",
                "crops/crop_000.png",
            ],
        ),
        (
            "open runner filtered dry-run",
            [
                sys.executable,
                "eval_code/if_exist/eval_open_vlm.py",
                "--input-mode",
                "both",
                "--backbone-model",
                "qwen3-vl-8b",
                "--judge-model",
                "qwen3-vl-8b",
                "--category",
                "if_exist",
                "--subcategory",
                "elephant_trunk",
                "--source-variant",
                "ai",
                "--question-types",
                "yes_no",
                "--max-samples",
                "1",
                "--prime",
                "off",
                "--dry-run",
            ],
            [
                "dry_run script_type=open",
                "eval_mode=both",
                "image_pairs=1",
                "response_target_count=1",
                "\"subcategory\": \"elephant_trunk\"",
                "\"source_variant\": \"ai\"",
                "\"question_type\": \"yes_no\"",
                "Two related images are provided",
                "Compare the visible target attribute",
            ],
        ),
        (
            "server runner cf-only dry-run",
            [
                sys.executable,
                "eval_code/if_exist/eval_server_vlm.py",
                "--input-mode",
                "cf_only",
                "--backbone-model",
                "qwen3-vl-8b",
                "--server-url",
                "http://localhost:8000",
                "--server-model-id",
                "local-qwen",
                "--category",
                "if_exist",
                "--subcategory",
                "camel",
                "--question-types",
                "multiple_choice",
                "--max-samples",
                "1",
                "--dry-run",
            ],
            [
                "server_model_id=local-qwen",
                "dry_run script_type=server",
                "eval_mode=cf_only",
                "\"question_type\": \"multiple_choice\"",
            ],
        ),
        (
            "server runner both dry-run",
            [
                sys.executable,
                "eval_code/if_exist/eval_server_vlm.py",
                "--input-mode",
                "both",
                "--backbone-model",
                "qwen3-vl-8b",
                "--server-url",
                "http://localhost:8000",
                "--category",
                "if_exist",
                "--subcategory",
                "rabbit",
                "--question-types",
                "yes_no",
                "--max-samples",
                "1",
                "--dry-run",
            ],
            [
                "dry_run script_type=server",
                "eval_mode=both",
                "Qwen/Qwen3-VL-8B-Instruct",
                "Two related images are provided",
                "Compare the visible target attribute",
            ],
        ),
        (
            "server runner orig-only dry-run",
            [
                sys.executable,
                "eval_code/if_exist/eval_server_vlm.py",
                "--input-mode",
                "orig_only",
                "--backbone-model",
                "qwen2.5-vl-32b",
                "--server-url",
                "http://localhost:8000",
                "--category",
                "if_exist",
                "--subcategory",
                "fish_fin",
                "--question-types",
                "open",
                "--max-samples",
                "1",
                "--dry-run",
            ],
            ["dry_run script_type=server", "eval_mode=orig_only", "Qwen/Qwen2.5-VL-32B-Instruct"],
        ),
        (
            "closed runner subset dry-run",
            [
                sys.executable,
                "eval_code/if_exist/eval_closed_vlm.py",
                "--input-mode",
                "orig_only",
                "--backbone-model",
                "openai:gpt-4.1",
                "--judge-model",
                "openai:gpt-4o-mini",
                "--subset-path",
                "dataset/if_exist/camel",
                "--question-types",
                "open",
                "--max-samples",
                "1",
                "--dry-run",
            ],
            [
                "dry_run script_type=closed",
                "eval_mode=orig_only",
                "image_pairs=1",
                "response_target_count=1",
                "\"subcategory\": \"camel\"",
                "\"question_type\": \"open\"",
            ],
        ),
        (
            "closed runner report",
            [
                sys.executable,
                "eval_code/if_exist/eval_closed_vlm.py",
                "--input-mode",
                "cf_only",
                "--backbone-model",
                "gpt-4.1",
                "--judge-model",
                "gpt-4o-mini",
                "--output-root",
                str(REPO_ROOT / "test" / "_nonexistent_if_exist_runs"),
                "--report",
            ],
            [
                "No run directories found.",
            ],
        ),
    ]
    for name, command, expected in checks:
        output = run(name, command)
        require_all(name, output, expected)
        if name == "open runner filtered dry-run" and "This is a well-known, classic" in output:
            raise SystemExit("--prime off dry-run unexpectedly included the prior-prime prefix.")
        if name in {"closed runner filtered dry-run", "server runner cf-only dry-run", "server runner orig-only dry-run"}:
            if "If two images are provided, answer about the second image." in output:
                raise SystemExit(f"{name} unexpectedly included the both-mode second-image instruction.")
        if name in {"open runner filtered dry-run", "server runner both dry-run"}:
            forbidden = ["If two images are provided, answer about the second image.", "first image", "second image"]
            present = [fragment for fragment in forbidden if fragment in output]
            if present:
                raise SystemExit(f"{name} leaked legacy both raw target/order wording: {present}")

    required_paths = [
        "cf_dataset/if_exist_cf_questions.json",
        "dataset/if_exist",
        "cf_dataset/if_exist_cf",
        "eval_code/if_exist/eval_closed_vlm.py",
        "eval_code/if_exist/eval_open_vlm.py",
        "eval_code/if_exist/eval_server_vlm.py",
    ]
    missing = [path for path in required_paths if not (REPO_ROOT / path).exists()]
    if missing:
        raise SystemExit(f"Missing required if_exist paths: {missing}")

    print("if_exist eval CLI smoke tests passed.")
    return 0


def run(name: str, command: list[str]) -> str:
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
    if completed.returncode != 0:
        print(output)
        raise SystemExit(f"{name} failed with exit code {completed.returncode}")
    return output


def require_all(name: str, output: str, expected: list[str]) -> None:
    missing = [item for item in expected if item not in output]
    if missing:
        raise SystemExit(f"{name} missing expected output fragments: {missing}")


if __name__ == "__main__":
    sys.exit(main())
