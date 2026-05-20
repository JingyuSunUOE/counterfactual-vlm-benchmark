#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "eval_code"))

from server_vram_planner import BYTES_PER_GIB, CudaDeviceSnapshot, plan_dual_server_vram, plan_server_vram  # noqa: E402


def main() -> int:
    devices_40g = fake_devices([40])
    single = plan_server_vram(model_id="Qwen/Qwen3-VL-8B-Instruct", devices=devices_40g, dtype="bfloat16")
    require(single.ok, "8B model should fit on one 40GB fake GPU.")
    assert_equal("single TP", single.tensor_parallel_size, 1)

    devices_45g = fake_devices([45, 45])
    sharded = plan_server_vram(model_id="Qwen/Qwen3-VL-32B-Instruct", devices=devices_45g, dtype="bfloat16")
    require(sharded.ok, f"32B model should fit across two 45GB fake GPUs: {sharded.failure_reason}")
    assert_equal("auto TP", sharded.tensor_parallel_size, 2)

    devices_24g = fake_devices([24, 24])
    insufficient = plan_server_vram(model_id="Qwen/Qwen3-VL-32B-Instruct", devices=devices_24g, dtype="bfloat16")
    require(not insufficient.ok, "32B bf16 model should not fit across two 24GB fake GPUs.")

    quantized = plan_server_vram(
        model_id="Qwen/Qwen3-VL-32B-Instruct",
        devices=devices_40g,
        dtype="bfloat16",
        load_in_4bit=True,
    )
    require(quantized.ok, f"Explicit 4-bit estimate should fit on one 40GB fake GPU: {quantized.failure_reason}")
    assert_equal("4-bit explicit", quantized.load_in_4bit, True)

    unknown = plan_server_vram(model_id="local-qwen", devices=devices_40g, dtype="bfloat16")
    require(not unknown.ok, "Model ids without size suffix should fail unless an HF model id is provided.")

    test_dual_auto_separate()
    test_dual_auto_shared()
    test_dual_auto_fails_when_shared_too_large()
    test_dual_auto_fails_when_large_backbone_uses_all_gpus()
    test_dual_explicit_respects_manual_gpu_ids()
    test_dual_unknown_model_reports_size_hint()
    test_launcher_dry_run()
    test_if_exist_strict_preflight_dry_run_fails_before_outputs()
    print("server VRAM planner tests passed.")
    return 0


def fake_devices(free_gb_values: list[float]) -> list[CudaDeviceSnapshot]:
    return [
        CudaDeviceSnapshot(
            index=index,
            name=f"Fake GPU {index}",
            free_bytes=int(free_gb * BYTES_PER_GIB),
            total_bytes=int(free_gb * BYTES_PER_GIB),
        )
        for index, free_gb in enumerate(free_gb_values)
    ]


def test_dual_auto_separate() -> None:
    plan = plan_dual_server_vram(
        backbone_model_id="Qwen/Qwen3-VL-8B-Instruct",
        judge_model_id="Qwen/Qwen3-8B",
        devices=fake_devices([80, 80]),
        placement_policy="auto",
    )
    require(plan.ok, f"8B + 8B should auto-place on two 80GB GPUs: {plan.failure_reason}")
    assert_equal("dual separate mode", plan.placement_mode, "separate")
    require(plan.backbone_plan is not None and plan.judge_plan is not None, "dual plan should contain both server plans.")
    require(
        not (set(plan.backbone_plan.assigned_gpu_ids) & set(plan.judge_plan.assigned_gpu_ids)),
        "auto separate should not overlap backbone and judge GPUs.",
    )


def test_dual_auto_shared() -> None:
    plan = plan_dual_server_vram(
        backbone_model_id="Qwen/Qwen3-VL-8B-Instruct",
        judge_model_id="Qwen/Qwen3-8B",
        devices=fake_devices([80]),
        placement_policy="auto",
        shared_gpu_total_utilization=0.88,
    )
    require(plan.ok, f"8B + 8B should share one 80GB GPU: {plan.failure_reason}")
    assert_equal("dual shared mode", plan.placement_mode, "shared")
    require(plan.backbone_plan is not None and plan.judge_plan is not None, "shared plan should contain both server plans.")
    assert_equal("shared backbone gpu", plan.backbone_plan.assigned_gpu_ids, [0])
    assert_equal("shared judge gpu", plan.judge_plan.assigned_gpu_ids, [0])
    require(
        plan.backbone_plan.gpu_memory_utilization + plan.judge_plan.gpu_memory_utilization <= 0.880001,
        "shared utilization should not exceed configured total.",
    )


def test_dual_auto_fails_when_shared_too_large() -> None:
    plan = plan_dual_server_vram(
        backbone_model_id="Qwen/Qwen2.5-VL-72B-Instruct",
        judge_model_id="Qwen/Qwen3-8B",
        devices=fake_devices([80]),
        placement_policy="auto",
    )
    require(not plan.ok, "72B + 8B should not fit on one 80GB GPU under shared placement.")
    require("Shared failure" in (plan.failure_reason or ""), "failure should include shared placement reason.")


def test_dual_auto_fails_when_large_backbone_uses_all_gpus() -> None:
    plan = plan_dual_server_vram(
        backbone_model_id="Qwen/Qwen2.5-VL-72B-Instruct",
        judge_model_id="Qwen/Qwen3-8B",
        devices=fake_devices([80, 80]),
        placement_policy="auto",
    )
    require(not plan.ok, "72B + 8B should fail on two 80GB GPUs with conservative bf16 estimate.")
    require(plan.failure_reason, "large dual placement failure should explain why it failed.")


def test_dual_explicit_respects_manual_gpu_ids() -> None:
    plan = plan_dual_server_vram(
        backbone_model_id="Qwen/Qwen3-VL-8B-Instruct",
        judge_model_id="Qwen/Qwen3-8B",
        devices=fake_devices([80, 80]),
        placement_policy="auto",
        backbone_gpu_ids="0",
        judge_gpu_ids="1",
    )
    require(plan.ok, f"explicit 8B + 8B placement should pass: {plan.failure_reason}")
    assert_equal("explicit policy", plan.placement_policy, "explicit")
    assert_equal("explicit mode", plan.placement_mode, "explicit_separate")
    assert_equal("explicit backbone gpu", plan.backbone_plan.assigned_gpu_ids if plan.backbone_plan else None, [0])
    assert_equal("explicit judge gpu", plan.judge_plan.assigned_gpu_ids if plan.judge_plan else None, [1])


def test_dual_unknown_model_reports_size_hint() -> None:
    plan = plan_dual_server_vram(
        backbone_model_id="local-backbone",
        judge_model_id="Qwen/Qwen3-8B",
        devices=fake_devices([80, 80]),
        placement_policy="auto",
    )
    require(not plan.ok, "unknown backbone size should fail joint placement.")
    output = json.dumps(plan.to_json())
    require("8B/32B/72B" in output or "Unable to estimate" in output, "failure should contain a model-size hint.")


def test_launcher_dry_run() -> None:
    env = os.environ.copy()
    env["VLM_EVAL_FAKE_CUDA_SNAPSHOT"] = json.dumps(
        [{"index": 0, "name": "Fake GPU 0", "free_gb": 45, "total_gb": 45}]
    )
    completed = subprocess.run(
        [
            sys.executable,
            "eval_code/launch_vlm_server.py",
            "--framework",
            "vllm",
            "--model",
            "Qwen/Qwen3-VL-8B-Instruct",
            "--served-model-name",
            "qwen3-vl-8b",
            "--dry-run",
        ],
        cwd=REPO_ROOT,
        env=env,
        text=True,
        capture_output=True,
        timeout=120,
        check=False,
    )
    output = completed.stdout + completed.stderr
    if completed.returncode != 0:
        raise SystemExit(f"launcher dry-run failed:\n{output}")
    require("launch_command=" in output, "launcher dry-run did not print a launch command.")
    require("--tensor-parallel-size 1" in output, "launcher dry-run did not include the planned TP size.")


def test_if_exist_strict_preflight_dry_run_fails_before_outputs() -> None:
    env = os.environ.copy()
    env["VLM_EVAL_FAKE_CUDA_SNAPSHOT"] = json.dumps(
        [{"index": 0, "name": "Tiny GPU", "free_gb": 1, "total_gb": 1}]
    )
    with tempfile.TemporaryDirectory(prefix="server_preflight_fail_") as tmp_dir:
        output_root = Path(tmp_dir) / "runs"
        completed = subprocess.run(
            [
                sys.executable,
                "eval_code/if_exist/eval_server_vlm.py",
                "--input-mode",
                "cf_only",
                "--backbone-model",
                "qwen3-vl-32b",
                "--category",
                "if_exist",
                "--subcategory",
                "camel",
                "--question-types",
                "yes_no",
                "--max-samples",
                "1",
                "--server-preflight",
                "strict",
                "--output-root",
                str(output_root),
                "--dry-run",
            ],
            cwd=REPO_ROOT,
            env=env,
            text=True,
            capture_output=True,
            timeout=120,
            check=False,
        )
        output = completed.stdout + completed.stderr
        require(completed.returncode != 0, "strict insufficient preflight should fail.")
        require("Server VRAM preflight failed" in output, "strict preflight did not explain the failure.")
        require(not output_root.exists(), "strict dry-run preflight should not create output directories.")


def require(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(message)


def assert_equal(name: str, actual: object, expected: object) -> None:
    if actual != expected:
        raise SystemExit(f"{name}: expected {expected!r}, got {actual!r}")


if __name__ == "__main__":
    sys.exit(main())
