#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import subprocess
import sys
from typing import List

from server_vram_planner import (
    SUPPORTED_SERVER_DTYPES,
    plan_server_vram,
    preflight_or_warn,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Launch an OpenAI-compatible local VLM server with VRAM preflight.")
    parser.add_argument("--framework", choices=("vllm", "sglang"), required=True)
    parser.add_argument("--model", required=True, help="HF model id or local model path.")
    parser.add_argument("--served-model-name", default=None, help="Model name exposed by the OpenAI-compatible server.")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--gpu-ids", default=None, help="Comma-separated CUDA device ids exposed to the server.")
    parser.add_argument("--dtype", choices=SUPPORTED_SERVER_DTYPES, default="bfloat16")
    parser.add_argument("--tensor-parallel-size", default="auto")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=None,
        help="Optional vLLM --max-model-len. Use this to avoid very large model defaults when the benchmark does not need long context.",
    )
    parser.add_argument("--reserve-gb", type=float, default=4.0)
    parser.add_argument("--load-in-4bit", action="store_true", help="Use explicit 4-bit memory estimation for preflight.")
    parser.add_argument("--dry-run", action="store_true", help="Print the launch command without starting the server.")
    parser.add_argument(
        "--skip-package-check",
        action="store_true",
        help="Do not verify that the selected serving framework is importable before launch.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    plan = plan_server_vram(
        model_id=args.model,
        gpu_ids=args.gpu_ids,
        tensor_parallel_size=args.tensor_parallel_size,
        dtype=args.dtype,
        load_in_4bit=args.load_in_4bit,
        gpu_memory_utilization=args.gpu_memory_utilization,
        reserve_gb=args.reserve_gb,
    )
    preflight_or_warn("strict", [("server", plan)])
    command = build_command(args=args, tensor_parallel_size=plan.tensor_parallel_size or 1)
    env = os.environ.copy()
    if plan.assigned_gpu_ids:
        env["CUDA_VISIBLE_DEVICES"] = ",".join(str(device_id) for device_id in plan.assigned_gpu_ids)

    print("server_vram_plan=" + json.dumps(plan.to_json(), indent=2))
    print("launch_command=" + " ".join(command))
    if args.load_in_4bit:
        print(
            "warning=--load-in-4bit affects preflight estimation only. "
            "Use a quantized model or framework-specific quantization flags when required."
        )
    if args.dry_run:
        return 0
    if not args.skip_package_check:
        require_framework(args.framework)
    return subprocess.run(command, env=env, check=False).returncode


def build_command(*, args: argparse.Namespace, tensor_parallel_size: int) -> List[str]:
    served_name = args.served_model_name or args.model
    if args.framework == "vllm":
        command = [
            sys.executable,
            "-m",
            "vllm.entrypoints.openai.api_server",
            "--model",
            args.model,
            "--served-model-name",
            served_name,
            "--host",
            args.host,
            "--port",
            str(args.port),
            "--dtype",
            args.dtype,
            "--tensor-parallel-size",
            str(tensor_parallel_size),
            "--gpu-memory-utilization",
            str(args.gpu_memory_utilization),
        ]
        if args.max_model_len is not None:
            command.extend(["--max-model-len", str(args.max_model_len)])
    elif args.framework == "sglang":
        command = [
            sys.executable,
            "-m",
            "sglang.launch_server",
            "--model-path",
            args.model,
            "--served-model-name",
            served_name,
            "--host",
            args.host,
            "--port",
            str(args.port),
            "--dtype",
            args.dtype,
            "--tp",
            str(tensor_parallel_size),
            "--mem-fraction-static",
            str(args.gpu_memory_utilization),
        ]
    else:  # pragma: no cover - argparse prevents this.
        raise ValueError(f"Unsupported framework: {args.framework}")
    return command


def require_framework(framework: str) -> None:
    module_name = "vllm" if framework == "vllm" else "sglang"
    if importlib.util.find_spec(module_name) is None:
        raise SystemExit(
            f"{module_name} is not installed. Install it in the serving environment, "
            "or rerun with --skip-package-check if the module is provided externally."
        )


if __name__ == "__main__":
    sys.exit(main())
