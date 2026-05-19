#!/usr/bin/env python3
from __future__ import annotations

import subprocess
import sys
import os
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


def main() -> int:
    fake_two_gpu_env = os.environ.copy()
    fake_two_gpu_env["VLM_EVAL_FAKE_CUDA_SNAPSHOT"] = (
        '[{"index":0,"name":"H100","free_gb":78,"total_gb":80},'
        '{"index":1,"name":"H100","free_gb":78,"total_gb":80}]'
    )
    output = run(
        "counting wrapper dry-run",
        [
            sys.executable,
            "eval_code/run_server_eval.py",
            "--framework",
            "vllm",
            "--model",
            "Qwen/Qwen3-VL-8B-Instruct",
            "--served-model-name",
            "qwen3-vl-8b",
            "--benchmark",
            "counting",
            "--dry-run",
            "--",
            "--input-mode",
            "cf_only",
            "--backbone-model",
            "qwen3-vl-8b",
            "--judge-model",
            "gpt-4o-mini",
            "--judge-provider",
            "openai",
            "--max-samples",
            "1",
        ],
        ["eval_code/counting/eval_server_vlm.py", "--server-url", "--server-model-id", "eval_results/counting/server_logs"],
        env=fake_two_gpu_env,
    )
    if "--open-backend server" in output:
        raise SystemExit("Counting wrapper should not inject fashion --open-backend.")

    conflict = subprocess.run(
        [
            sys.executable,
            "eval_code/run_server_eval.py",
            "--framework",
            "vllm",
            "--model",
            "Qwen/Qwen3-VL-8B-Instruct",
            "--served-model-name",
            "qwen3-vl-8b",
            "--benchmark",
            "counting",
            "--dry-run",
            "--",
            "--input-mode",
            "cf_only",
            "--backbone-model",
            "qwen3-vl-8b",
            "--server-url",
            "http://localhost:8000",
        ],
        cwd=REPO_ROOT,
        env=fake_two_gpu_env,
        text=True,
        capture_output=True,
        timeout=120,
        check=False,
    )
    if conflict.returncode == 0 or "Do not pass --server-url" not in (conflict.stdout + conflict.stderr):
        raise SystemExit("Expected downstream --server-url conflict to fail.")

    print("counting run_server_eval CLI tests passed.")
    return 0


def run(name: str, command: list[str], expected: list[str], *, env: dict[str, str] | None = None) -> str:
    print(f"[RUN] {name}: {' '.join(command)}")
    completed = subprocess.run(
        command,
        cwd=REPO_ROOT,
        env=env,
        text=True,
        capture_output=True,
        timeout=120,
        check=False,
    )
    output = completed.stdout + completed.stderr
    if completed.returncode != 0:
        print(output)
        raise SystemExit(f"{name} failed with exit code {completed.returncode}")
    missing = [fragment for fragment in expected if fragment not in output]
    if missing:
        raise SystemExit(f"{name} missing expected fragments: {missing}")
    return output


if __name__ == "__main__":
    raise SystemExit(main())
