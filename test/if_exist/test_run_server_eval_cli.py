#!/usr/bin/env python3
from __future__ import annotations

import subprocess
import sys
import tempfile
import os
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
RUNNER = "eval_code/run_server_eval.py"
sys.path.insert(0, str(REPO_ROOT / "eval_code"))
import run_server_eval  # noqa: E402


def run_cmd(args: list[str], *, env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, *args],
        cwd=REPO_ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )


def assert_contains(text: str, expected: list[str]) -> None:
    missing = [item for item in expected if item not in text]
    if missing:
        raise AssertionError(f"Missing expected output fragments: {missing}\nOutput:\n{text}")


def main() -> int:
    help_result = run_cmd([RUNNER, "--help"])
    if help_result.returncode != 0:
        raise AssertionError(help_result.stderr)
    assert_contains(
        help_result.stdout,
        [
            "--framework",
            "--model",
            "--served-model-name",
            "--benchmark",
            "--server-startup-timeout",
            "--health-poll-seconds",
            "--server-log-mode",
            "--if-server-running",
            "--keep-server-alive",
            "--judge-server-model",
            "--judge-served-model-name",
            "--judge-port",
            "--judge-gpu-ids",
            "--gpu-placement-policy",
            "--shared-gpu-total-utilization",
            "--shared-gpu-min-server-utilization",
            "--keep-judge-server-alive",
        ],
    )

    fake_two_gpu_env = os.environ.copy()
    fake_two_gpu_env["VLM_EVAL_FAKE_CUDA_SNAPSHOT"] = (
        '[{"index":0,"name":"H100","free_gb":78,"total_gb":80},'
        '{"index":1,"name":"H100","free_gb":78,"total_gb":80}]'
    )
    fake_one_gpu_env = os.environ.copy()
    fake_one_gpu_env["VLM_EVAL_FAKE_CUDA_SNAPSHOT"] = '[{"index":0,"name":"H100","free_gb":78,"total_gb":80}]'

    with tempfile.TemporaryDirectory() as tmp_dir:
        log_dir = Path(tmp_dir) / "server_logs"
        dry_result = run_cmd(
            [
                RUNNER,
                "--framework",
                "vllm",
                "--model",
                "Qwen/Qwen3-VL-8B-Instruct",
                "--served-model-name",
                "qwen3-vl-8b",
                "--benchmark",
                "if_exist",
                "--server-log-dir",
                str(log_dir),
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
            env=fake_two_gpu_env,
        )
        if dry_result.returncode != 0:
            raise AssertionError(dry_result.stderr)
        assert_contains(
            dry_result.stdout,
            [
                "[1/5] Preflight",
                "CUDA GPU placement was planned",
                "joint_gpu_placement_plan=",
                "[2/5] Launch Server",
                "launch_command=",
                "vllm.entrypoints.openai.api_server",
                "backbone_CUDA_VISIBLE_DEVICES=0",
                "[3/5] Wait Ready",
                "health_url=http://127.0.0.1:8000/v1/models",
                "[4/5] Run Benchmark",
                "eval_code/if_exist/eval_server_vlm.py",
                "--server-url http://127.0.0.1:8000",
                "--server-model-id qwen3-vl-8b",
                "[5/5] Cleanup",
            ],
        )
        if log_dir.exists():
            raise AssertionError("Dry-run should not create the server log directory.")

    shared_result = run_cmd(
        [
            RUNNER,
            "--framework",
            "vllm",
            "--model",
            "Qwen/Qwen3-VL-8B-Instruct",
            "--served-model-name",
            "qwen3-vl-8b",
            "--judge-server-model",
            "Qwen/Qwen3-8B",
            "--judge-served-model-name",
            "qwen3-8b",
            "--benchmark",
            "if_exist",
            "--dry-run",
            "--",
            "--input-mode",
            "cf_only",
            "--backbone-model",
            "qwen3-vl-8b",
            "--max-samples",
            "1",
        ],
        env=fake_one_gpu_env,
    )
    if shared_result.returncode != 0:
        raise AssertionError(shared_result.stderr)
    assert_contains(
        shared_result.stdout,
        [
            '"placement_mode": "shared"',
            "backbone_CUDA_VISIBLE_DEVICES=0",
            "judge_CUDA_VISIBLE_DEVICES=0",
            '"gpu_memory_utilization"',
        ],
    )

    fail_env = os.environ.copy()
    fail_env["VLM_EVAL_FAKE_CUDA_SNAPSHOT"] = '[{"index":0,"name":"Tiny","free_gb":1,"total_gb":1}]'
    placement_fail = run_cmd(
        [
            RUNNER,
            "--framework",
            "vllm",
            "--model",
            "Qwen/Qwen3-VL-8B-Instruct",
            "--judge-server-model",
            "Qwen/Qwen3-8B",
            "--benchmark",
            "if_exist",
            "--dry-run",
            "--",
            "--input-mode",
            "cf_only",
            "--backbone-model",
            "qwen3-vl-8b",
            "--max-samples",
            "1",
        ],
        env=fail_env,
    )
    if placement_fail.returncode == 0:
        raise AssertionError("Expected automatic placement to fail on a tiny fake GPU.")
    assert_contains(placement_fail.stderr + placement_fail.stdout, ["Joint GPU placement failed", "failure_reason"])

    conflict_result = run_cmd(
        [
            RUNNER,
            "--framework",
            "vllm",
            "--model",
            "Qwen/Qwen3-VL-8B-Instruct",
            "--benchmark",
            "if_exist",
            "--dry-run",
            "--",
            "--input-mode",
            "cf_only",
            "--backbone-model",
            "qwen3-vl-8b",
            "--server-url",
            "http://localhost:9999",
        ],
        env=fake_two_gpu_env,
    )
    if conflict_result.returncode == 0:
        raise AssertionError("Expected downstream --server-url conflict to fail.")
    assert_contains(
        conflict_result.stderr + conflict_result.stdout,
        ["Do not pass --server-url after '--'"],
    )

    judge_dry_result = run_cmd(
        [
            RUNNER,
            "--framework",
            "vllm",
            "--model",
            "Qwen/Qwen3-VL-8B-Instruct",
            "--served-model-name",
            "qwen3-vl-8b",
            "--judge-server-model",
            "Qwen/Qwen3-8B",
            "--judge-served-model-name",
            "qwen3-8b",
            "--judge-port",
            "8001",
            "--benchmark",
            "if_exist",
            "--dry-run",
            "--",
            "--input-mode",
            "cf_only",
            "--backbone-model",
            "qwen3-vl-8b",
            "--max-samples",
            "1",
        ],
        env=fake_two_gpu_env,
    )
    if judge_dry_result.returncode != 0:
        raise AssertionError(judge_dry_result.stderr)
    assert_contains(
        judge_dry_result.stdout,
        [
            "backbone_launch_command=",
            "judge_launch_command=",
            "backbone_health_url=http://127.0.0.1:8000/v1/models",
            "judge_health_url=http://127.0.0.1:8001/v1/models",
            "--judge-provider server",
            "--judge-model qwen3-8b",
            "--judge-server-url http://127.0.0.1:8001",
        ],
    )

    judge_conflict_result = run_cmd(
        [
            RUNNER,
            "--framework",
            "vllm",
            "--model",
            "Qwen/Qwen3-VL-8B-Instruct",
            "--judge-server-model",
            "Qwen/Qwen3-8B",
            "--benchmark",
            "if_exist",
            "--dry-run",
            "--",
            "--input-mode",
            "cf_only",
            "--backbone-model",
            "qwen3-vl-8b",
            "--judge-provider",
            "server",
        ]
    )
    if judge_conflict_result.returncode == 0:
        raise AssertionError("Expected downstream --judge-provider conflict to fail.")
    assert_contains(
        judge_conflict_result.stderr + judge_conflict_result.stdout,
        ["Do not pass --judge-provider after '--'"],
    )

    fashion_result = run_cmd(
        [
            RUNNER,
            "--framework",
            "sglang",
            "--model",
            "Qwen/Qwen3-VL-8B-Instruct",
            "--served-model-name",
            "qwen3-vl-8b",
            "--judge-server-model",
            "Qwen/Qwen3-8B",
            "--judge-served-model-name",
            "qwen3-8b",
            "--benchmark",
            "fashion_industry",
            "--dry-run",
            "--",
            "--phase",
            "cf",
            "--input-mode",
            "cf_only",
            "--model",
            "qwen3-vl-8b",
            "--limit",
            "1",
        ],
        env=fake_two_gpu_env,
    )
    if fashion_result.returncode != 0:
        raise AssertionError(fashion_result.stderr)
    assert_contains(
        fashion_result.stdout,
        [
            "eval_code/fashion_industry/eval_pipeline.py",
            "--open-backend server",
            "--judge-provider server",
            "--judge qwen3-8b",
            "--judge-server-url http://127.0.0.1:8001",
            "--server-url http://127.0.0.1:8000",
            "--server-model-id qwen3-vl-8b",
        ],
    )

    class FakeProcess:
        returncode = None

        def poll(self):
            return None

    original_fetch_model_ids = run_server_eval.fetch_model_ids
    try:
        calls = {"count": 0}

        def eventually_ready(_server_url: str, *, timeout: float):
            calls["count"] += 1
            return ["mock-model"] if calls["count"] >= 2 else None

        run_server_eval.fetch_model_ids = eventually_ready
        ready_models = run_server_eval.wait_for_server_ready(
            server_url="http://127.0.0.1:8000",
            process=FakeProcess(),
            timeout_seconds=2,
            poll_seconds=0.01,
            label="backbone",
        )
        if ready_models != ["mock-model"]:
            raise AssertionError(f"Unexpected ready models: {ready_models}")

        calls["count"] = 0
        run_server_eval.fetch_model_ids = eventually_ready
        judge_ready_models = run_server_eval.wait_for_server_ready(
            server_url="http://127.0.0.1:8001",
            process=FakeProcess(),
            timeout_seconds=2,
            poll_seconds=0.01,
            label="judge",
        )
        if judge_ready_models != ["mock-model"]:
            raise AssertionError(f"Unexpected judge ready models: {judge_ready_models}")

        def never_ready(_server_url: str, *, timeout: float):
            return None

        run_server_eval.fetch_model_ids = never_ready
        try:
            run_server_eval.wait_for_server_ready(
                server_url="http://127.0.0.1:8000",
                process=FakeProcess(),
                timeout_seconds=0.01,
                poll_seconds=0.01,
                label="judge",
            )
        except SystemExit as exc:
            if "Timed out waiting for judge server readiness" not in str(exc):
                raise
        else:
            raise AssertionError("Expected readiness timeout to raise SystemExit.")
    finally:
        run_server_eval.fetch_model_ids = original_fetch_model_ids

    return 0


if __name__ == "__main__":
    sys.exit(main())
