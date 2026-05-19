#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from launch_vlm_server import build_command, require_framework
from server_vram_planner import (
    DEFAULT_SHARED_GPU_MIN_SERVER_UTILIZATION,
    DEFAULT_SHARED_GPU_TOTAL_UTILIZATION,
    SUPPORTED_GPU_PLACEMENT_POLICIES,
    SUPPORTED_SERVER_DTYPES,
    DualServerVramPlan,
    plan_dual_server_vram,
    plan_server_vram,
    preflight_or_warn,
)


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8000
DEFAULT_JUDGE_PORT = 8001
DEFAULT_JUDGE_SERVED_MODEL_NAME = "qwen3-8b"
DEFAULT_STARTUP_TIMEOUT = 1800.0
DEFAULT_HEALTH_POLL_SECONDS = 5.0
SERVER_CONNECTION_OPTIONS = {"--server-url", "--server-model-id", "--local-url"}
AUTO_JUDGE_OPTIONS = {"--judge-provider", "--judge-server-url", "--judge-model", "--judge"}


@dataclass(frozen=True)
class ServerSpec:
    label: str
    framework: str
    model: str
    served_model_name: str
    host: str
    port: int
    gpu_ids: Optional[str]
    dtype: str
    tensor_parallel_size: str
    gpu_memory_utilization: float
    reserve_gb: float
    load_in_4bit: bool
    keep_alive: bool
    log_path: Path

    @property
    def url(self) -> str:
        return build_server_url(self.host, self.port)


@dataclass
class ServerRuntime:
    spec: ServerSpec
    process: Optional[subprocess.Popen[Any]] = None
    log_handle: Any = None
    launched: bool = False
    existing_models: Optional[List[str]] = None
    ready_models: Optional[List[str]] = None


def parse_args(argv: Optional[Sequence[str]] = None) -> Tuple[argparse.Namespace, List[str]]:
    raw_args = list(sys.argv[1:] if argv is None else argv)
    wrapper_args, benchmark_args = split_wrapper_and_benchmark_args(raw_args)

    parser = argparse.ArgumentParser(
        description=(
            "Start local OpenAI-compatible VLM servers, wait until they are ready, "
            "run a benchmark, then clean up the server processes."
        )
    )
    parser.add_argument("--framework", choices=("vllm", "sglang"), required=True)
    parser.add_argument("--model", required=True, help="HF model id or local model path used to start the backbone server.")
    parser.add_argument("--served-model-name", default=None, help="Model name exposed by the backbone server.")
    parser.add_argument("--benchmark", choices=("if_exist", "fashion_industry", "counting", "medical_modality"), required=True)
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--gpu-ids", default=None, help="Comma-separated CUDA device ids exposed to the backbone server.")
    parser.add_argument("--dtype", choices=SUPPORTED_SERVER_DTYPES, default="bfloat16")
    parser.add_argument("--tensor-parallel-size", default="auto")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    parser.add_argument("--reserve-gb", type=float, default=4.0)
    parser.add_argument("--load-in-4bit", action="store_true", help="Use explicit 4-bit memory estimation for backbone preflight.")
    parser.add_argument("--judge-server-model", default=None, help="Optional HF model id or local path for a text-only judge server.")
    parser.add_argument("--judge-served-model-name", default=DEFAULT_JUDGE_SERVED_MODEL_NAME)
    parser.add_argument("--judge-framework", choices=("vllm", "sglang"), default=None)
    parser.add_argument("--judge-host", default=None)
    parser.add_argument("--judge-port", type=int, default=DEFAULT_JUDGE_PORT)
    parser.add_argument("--judge-gpu-ids", default=None)
    parser.add_argument("--judge-dtype", choices=SUPPORTED_SERVER_DTYPES, default=None)
    parser.add_argument("--judge-tensor-parallel-size", default=None)
    parser.add_argument("--judge-gpu-memory-utilization", type=float, default=None)
    parser.add_argument("--judge-reserve-gb", type=float, default=None)
    parser.add_argument("--judge-load-in-4bit", action="store_true")
    parser.add_argument(
        "--gpu-placement-policy",
        choices=SUPPORTED_GPU_PLACEMENT_POLICIES,
        default="auto",
        help="Automatically place backbone and judge servers across visible GPUs.",
    )
    parser.add_argument(
        "--shared-gpu-total-utilization",
        type=float,
        default=DEFAULT_SHARED_GPU_TOTAL_UTILIZATION,
        help="Maximum combined GPU memory utilization when backbone and judge share one GPU.",
    )
    parser.add_argument(
        "--shared-gpu-min-server-utilization",
        type=float,
        default=DEFAULT_SHARED_GPU_MIN_SERVER_UTILIZATION,
        help="Minimum utilization budget assigned to each server in shared-GPU placement.",
    )
    parser.add_argument("--keep-judge-server-alive", action="store_true")
    parser.add_argument("--server-startup-timeout", type=float, default=DEFAULT_STARTUP_TIMEOUT)
    parser.add_argument("--health-poll-seconds", type=float, default=DEFAULT_HEALTH_POLL_SECONDS)
    parser.add_argument("--server-log-dir", type=Path, default=None)
    parser.add_argument("--server-log-mode", choices=("file", "terminal"), default="file")
    parser.add_argument("--if-server-running", choices=("fail", "reuse"), default="fail")
    parser.add_argument("--keep-server-alive", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Print server and benchmark commands without running them.")
    parser.add_argument(
        "--skip-package-check",
        action="store_true",
        help="Do not verify that the selected serving framework is importable before launch.",
    )
    args = parser.parse_args(wrapper_args)
    return args, benchmark_args


def split_wrapper_and_benchmark_args(raw_args: Sequence[str]) -> Tuple[List[str], List[str]]:
    if "--" not in raw_args:
        return list(raw_args), []
    separator_index = list(raw_args).index("--")
    return list(raw_args[:separator_index]), list(raw_args[separator_index + 1 :])


def main(argv: Optional[Sequence[str]] = None) -> int:
    args, benchmark_args = parse_args(argv)
    backbone_spec = build_backbone_spec(args)
    judge_spec = build_judge_spec(args)
    runtimes = [ServerRuntime(backbone_spec)]
    if judge_spec is not None:
        runtimes.append(ServerRuntime(judge_spec))
    benchmark_command = build_benchmark_command(
        benchmark=args.benchmark,
        benchmark_args=benchmark_args,
        server_url=runtimes[0].spec.url,
        server_model_id=runtimes[0].spec.served_model_name,
        judge_server_url=runtimes[1].spec.url if len(runtimes) > 1 else None,
        judge_model_id=runtimes[1].spec.served_model_name if len(runtimes) > 1 else None,
    )
    if not args.dry_run and args.if_server_running == "reuse":
        detect_existing_runtimes(runtimes)
    joint_plan = apply_joint_gpu_placement(runtimes, args)

    if args.dry_run:
        print_step("1/5", "Preflight")
        print("dry_run=true")
        print("live_server_launch=skipped; CUDA GPU placement was planned without starting servers.")
        print("joint_gpu_placement_plan=" + json.dumps(joint_plan.to_json(), indent=2))
        for runtime in runtimes:
            print(f"{runtime.spec.label}_server_vram_request=" + json.dumps(server_vram_request(runtime.spec), indent=2))
        print_step("2/5", "Launch Server")
        for runtime in runtimes:
            dry_command, dry_env = build_server_launch_command(runtime.spec, tensor_parallel_size=dry_run_tensor_parallel_size(runtime.spec))
            print_server_launch_preview(runtime.spec, dry_command, dry_env)
        print_step("3/5", "Wait Ready")
        for runtime in runtimes:
            print(f"{runtime.spec.label}_health_url={models_url(runtime.spec.url)}")
        print_step("4/5", "Run Benchmark")
        print("benchmark_command=" + shell_join(benchmark_command))
        print_step("5/5", "Cleanup")
        print("dry_run=true; no server process was started.")
        return 0

    benchmark_exit_code = 1
    try:
        print_step("1/5", "Preflight")
        print("joint_gpu_placement_plan=" + json.dumps(joint_plan.to_json(), indent=2))
        for runtime in runtimes:
            preflight_runtime(runtime, args)

        if should_launch_and_wait_sequentially(joint_plan):
            print_step("2/5", "Launch Server")
            for runtime in runtimes:
                launch_runtime(runtime, args)
                wait_runtime_ready(runtime, args)
            print_step("3/5", "Wait Ready")
            print("shared_gpu_placement=true; each server was waited on immediately after launch.")
        else:
            print_step("2/5", "Launch Server")
            for runtime in runtimes:
                launch_runtime(runtime, args)

            print_step("3/5", "Wait Ready")
            for runtime in runtimes:
                wait_runtime_ready(runtime, args)

        print_step("4/5", "Run Benchmark")
        print("benchmark_command=" + shell_join(benchmark_command))
        benchmark_exit_code = run_benchmark(benchmark_command)
        return benchmark_exit_code
    except KeyboardInterrupt:
        print()
        print("interrupted=user")
        benchmark_exit_code = 130
        return benchmark_exit_code
    finally:
        print_step("5/5", "Cleanup")
        for runtime in reversed(runtimes):
            cleanup_runtime(runtime)
        print(f"benchmark_exit_code={benchmark_exit_code}")


def build_backbone_spec(args: argparse.Namespace) -> ServerSpec:
    served_name = args.served_model_name or args.model
    return ServerSpec(
        label="backbone",
        framework=args.framework,
        model=args.model,
        served_model_name=served_name,
        host=args.host,
        port=args.port,
        gpu_ids=args.gpu_ids,
        dtype=args.dtype,
        tensor_parallel_size=str(args.tensor_parallel_size),
        gpu_memory_utilization=args.gpu_memory_utilization,
        reserve_gb=args.reserve_gb,
        load_in_4bit=args.load_in_4bit,
        keep_alive=args.keep_server_alive,
        log_path=default_log_path(args, label="server"),
    )


def build_judge_spec(args: argparse.Namespace) -> Optional[ServerSpec]:
    if not args.judge_server_model:
        return None
    return ServerSpec(
        label="judge",
        framework=args.judge_framework or args.framework,
        model=args.judge_server_model,
        served_model_name=args.judge_served_model_name,
        host=args.judge_host or args.host,
        port=args.judge_port,
        gpu_ids=args.judge_gpu_ids,
        dtype=args.judge_dtype or args.dtype,
        tensor_parallel_size=str(args.judge_tensor_parallel_size or args.tensor_parallel_size),
        gpu_memory_utilization=args.judge_gpu_memory_utilization or args.gpu_memory_utilization,
        reserve_gb=args.judge_reserve_gb if args.judge_reserve_gb is not None else args.reserve_gb,
        load_in_4bit=args.judge_load_in_4bit,
        keep_alive=args.keep_judge_server_alive,
        log_path=default_log_path(args, label="judge_server"),
    )


def apply_joint_gpu_placement(runtimes: List[ServerRuntime], args: argparse.Namespace) -> DualServerVramPlan:
    existing_runtimes = [runtime for runtime in runtimes if runtime.existing_models is not None]
    if existing_runtimes:
        if len(existing_runtimes) == len(runtimes):
            return DualServerVramPlan(
                ok=True,
                placement_policy=args.gpu_placement_policy,
                placement_mode="reuse_existing",
                devices=[],
                backbone_plan=None,
                judge_plan=None,
                shared_gpu_total_utilization=args.shared_gpu_total_utilization,
                shared_gpu_min_server_utilization=args.shared_gpu_min_server_utilization,
                failure_reason=None,
            )
        raise SystemExit(
            "Partial existing server reuse is not supported with automatic GPU placement. "
            "Stop the existing server processes or provide explicit --gpu-ids/--judge-gpu-ids."
        )
    backbone = runtimes[0].spec
    judge = runtimes[1].spec if len(runtimes) > 1 else None
    placement_policy = "explicit" if args.gpu_ids or args.judge_gpu_ids else args.gpu_placement_policy
    plan = plan_dual_server_vram(
        backbone_model_id=backbone.model,
        judge_model_id=judge.model if judge is not None else None,
        placement_policy=placement_policy,
        backbone_gpu_ids=backbone.gpu_ids,
        judge_gpu_ids=judge.gpu_ids if judge is not None else None,
        backbone_tensor_parallel_size=backbone.tensor_parallel_size,
        judge_tensor_parallel_size=judge.tensor_parallel_size if judge is not None else "auto",
        backbone_dtype=backbone.dtype,
        judge_dtype=judge.dtype if judge is not None else backbone.dtype,
        backbone_load_in_4bit=backbone.load_in_4bit,
        judge_load_in_4bit=judge.load_in_4bit if judge is not None else False,
        backbone_gpu_memory_utilization=backbone.gpu_memory_utilization,
        judge_gpu_memory_utilization=judge.gpu_memory_utilization if judge is not None else backbone.gpu_memory_utilization,
        backbone_reserve_gb=backbone.reserve_gb,
        judge_reserve_gb=judge.reserve_gb if judge is not None else backbone.reserve_gb,
        shared_gpu_total_utilization=args.shared_gpu_total_utilization,
        shared_gpu_min_server_utilization=args.shared_gpu_min_server_utilization,
    )
    if not plan.ok:
        raise SystemExit("Joint GPU placement failed:\n" + json.dumps(plan.to_json(), indent=2))
    apply_server_plan_to_runtime(runtimes[0], plan.backbone_plan)
    if len(runtimes) > 1:
        apply_server_plan_to_runtime(runtimes[1], plan.judge_plan)
    return plan


def detect_existing_runtimes(runtimes: List[ServerRuntime]) -> None:
    for runtime in runtimes:
        runtime.existing_models = fetch_model_ids(runtime.spec.url, timeout=2.0)


def apply_server_plan_to_runtime(runtime: ServerRuntime, plan: Optional[Any]) -> None:
    if plan is None or not plan.ok:
        return
    runtime.spec = replace(
        runtime.spec,
        gpu_ids=",".join(str(device_id) for device_id in plan.assigned_gpu_ids),
        tensor_parallel_size=str(plan.tensor_parallel_size or runtime.spec.tensor_parallel_size),
        gpu_memory_utilization=plan.gpu_memory_utilization,
    )
    runtime._tensor_parallel_size = plan.tensor_parallel_size or 1  # type: ignore[attr-defined]


def should_launch_and_wait_sequentially(plan: DualServerVramPlan) -> bool:
    if plan.backbone_plan is None or plan.judge_plan is None:
        return False
    return bool(set(plan.backbone_plan.assigned_gpu_ids) & set(plan.judge_plan.assigned_gpu_ids))


def preflight_runtime(runtime: ServerRuntime, args: argparse.Namespace) -> None:
    existing_models = fetch_model_ids(runtime.spec.url, timeout=2.0)
    runtime.existing_models = existing_models
    if existing_models is not None:
        if args.if_server_running == "fail":
            raise SystemExit(
                f"{runtime.spec.label} server already running at {runtime.spec.url} with models={existing_models}. "
                "Use --if-server-running reuse to connect to existing servers."
            )
        print(f"{runtime.spec.label}_existing_server=reuse url={runtime.spec.url} models={existing_models}")
        return

    plan = plan_server_vram(
        model_id=runtime.spec.model,
        gpu_ids=runtime.spec.gpu_ids,
        tensor_parallel_size=runtime.spec.tensor_parallel_size,
        dtype=runtime.spec.dtype,
        load_in_4bit=runtime.spec.load_in_4bit,
        gpu_memory_utilization=runtime.spec.gpu_memory_utilization,
        reserve_gb=runtime.spec.reserve_gb,
    )
    preflight_or_warn("strict", [(f"{runtime.spec.label}_server", plan)])
    print(f"{runtime.spec.label}_server_vram_plan=" + json.dumps(plan.to_json(), indent=2))
    runtime._tensor_parallel_size = plan.tensor_parallel_size or 1  # type: ignore[attr-defined]
    if runtime.spec.load_in_4bit:
        print(
            f"{runtime.spec.label}_warning=--load-in-4bit affects preflight estimation only. "
            "Use a quantized model or framework-specific quantization flags when required."
        )


def launch_runtime(runtime: ServerRuntime, args: argparse.Namespace) -> None:
    if runtime.existing_models is not None:
        print(f"{runtime.spec.label}_launch=skipped; using existing server.")
        return
    if not args.skip_package_check:
        require_framework(runtime.spec.framework)
    tensor_parallel_size = getattr(runtime, "_tensor_parallel_size", 1)
    launch_command, env = build_server_launch_command(runtime.spec, tensor_parallel_size=tensor_parallel_size)
    runtime.log_handle = open_server_log(runtime.spec.log_path, args.server_log_mode)
    runtime.process = subprocess.Popen(
        launch_command,
        env=env,
        stdout=runtime.log_handle if runtime.log_handle is not None else None,
        stderr=subprocess.STDOUT if runtime.log_handle is not None else None,
        start_new_session=True,
    )
    runtime.launched = True
    print(f"{runtime.spec.label}_server_url={runtime.spec.url}")
    print(f"{runtime.spec.label}_served_model_name={runtime.spec.served_model_name}")
    print(f"{runtime.spec.label}_server_pid={runtime.process.pid}")
    print(f"{runtime.spec.label}_server_log_path={runtime.spec.log_path if runtime.log_handle is not None else 'terminal'}")
    print(f"{runtime.spec.label}_launch_command=" + shell_join(launch_command))


def wait_runtime_ready(runtime: ServerRuntime, args: argparse.Namespace) -> None:
    if runtime.existing_models is not None:
        print(f"{runtime.spec.label}_ready=true url={runtime.spec.url} models={runtime.existing_models}")
        runtime.ready_models = runtime.existing_models
        return
    if runtime.ready_models is not None:
        print(f"{runtime.spec.label}_ready=true url={runtime.spec.url} models={runtime.ready_models}")
        return
    if runtime.process is None:
        raise SystemExit(f"{runtime.spec.label} server process was not started.")
    ready_models = wait_for_server_ready(
        server_url=runtime.spec.url,
        process=runtime.process,
        timeout_seconds=args.server_startup_timeout,
        poll_seconds=args.health_poll_seconds,
        label=runtime.spec.label,
    )
    runtime.ready_models = ready_models
    print(f"{runtime.spec.label}_ready=true url={runtime.spec.url} models={ready_models}")


def cleanup_runtime(runtime: ServerRuntime) -> None:
    if runtime.launched and runtime.process is not None:
        if runtime.spec.keep_alive:
            print(f"{runtime.spec.label}_server_status=kept_alive pid={runtime.process.pid} url={runtime.spec.url}")
        else:
            terminate_process_group(runtime.process, label=f"{runtime.spec.label}_server")
            print(f"{runtime.spec.label}_server_status=terminated")
    else:
        print(f"{runtime.spec.label}_server_status=not_started_by_wrapper")
    if runtime.log_handle is not None:
        runtime.log_handle.close()
        print(f"{runtime.spec.label}_server_log_path={runtime.spec.log_path}")


def build_server_url(host: str, port: int) -> str:
    connect_host = "127.0.0.1" if host in {"0.0.0.0", "::"} else host
    return f"http://{connect_host}:{port}"


def models_url(server_url: str) -> str:
    return server_url.rstrip("/") + "/v1/models"


def print_step(step: str, title: str) -> None:
    print()
    print(f"[{step}] {title}")


def server_vram_request(spec: ServerSpec) -> Dict[str, Any]:
    return {
        "framework": spec.framework,
        "model": spec.model,
        "served_model_name": spec.served_model_name,
        "host": spec.host,
        "port": spec.port,
        "gpu_ids": spec.gpu_ids,
        "dtype": spec.dtype,
        "tensor_parallel_size": spec.tensor_parallel_size,
        "gpu_memory_utilization": spec.gpu_memory_utilization,
        "reserve_gb": spec.reserve_gb,
        "load_in_4bit": spec.load_in_4bit,
    }


def dry_run_tensor_parallel_size(spec: ServerSpec) -> int:
    if str(spec.tensor_parallel_size).lower() == "auto":
        return 1
    return int(spec.tensor_parallel_size)


def build_server_launch_command(spec: ServerSpec, *, tensor_parallel_size: int) -> Tuple[List[str], Dict[str, str]]:
    launch_args = argparse.Namespace(
        framework=spec.framework,
        model=spec.model,
        served_model_name=spec.served_model_name,
        host=spec.host,
        port=spec.port,
        dtype=spec.dtype,
        gpu_memory_utilization=spec.gpu_memory_utilization,
    )
    command = build_command(args=launch_args, tensor_parallel_size=tensor_parallel_size)
    env = os.environ.copy()
    if spec.gpu_ids:
        env["CUDA_VISIBLE_DEVICES"] = spec.gpu_ids
    return command, env


def print_server_launch_preview(spec: ServerSpec, command: Sequence[str], env: Dict[str, str]) -> None:
    print(f"{spec.label}_server_url={spec.url}")
    print(f"{spec.label}_served_model_name={spec.served_model_name}")
    print(f"{spec.label}_server_log_path={spec.log_path}")
    print(f"{spec.label}_launch_command=" + shell_join(command))
    if env.get("CUDA_VISIBLE_DEVICES"):
        print(f"{spec.label}_CUDA_VISIBLE_DEVICES=" + env["CUDA_VISIBLE_DEVICES"])


def default_log_path(args: argparse.Namespace, *, label: str) -> Path:
    if args.server_log_dir is not None:
        log_dir = args.server_log_dir
    else:
        result_name = args.benchmark
        log_dir = REPO_ROOT / "eval_results" / result_name / "server_logs"
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return log_dir / f"{label}_{timestamp}.log"


def open_server_log(log_path: Path, mode: str) -> Any:
    if mode == "terminal":
        return None
    log_path.parent.mkdir(parents=True, exist_ok=True)
    return log_path.open("a", encoding="utf-8", buffering=1)


def fetch_model_ids(server_url: str, *, timeout: float) -> Optional[List[str]]:
    request = urllib.request.Request(
        models_url(server_url),
        headers={"Authorization": "Bearer EMPTY"},
        method="GET",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, OSError):
        return None
    data = payload.get("data") if isinstance(payload, dict) else None
    if not isinstance(data, list):
        return []
    model_ids = []
    for item in data:
        if isinstance(item, dict) and item.get("id") is not None:
            model_ids.append(str(item["id"]))
    return model_ids


def wait_for_server_ready(
    *,
    server_url: str,
    process: subprocess.Popen[Any],
    timeout_seconds: float,
    poll_seconds: float,
    label: str = "server",
) -> List[str]:
    started = time.monotonic()
    next_report = 0.0
    server_label = "server" if label == "server" else f"{label} server"
    while True:
        elapsed = time.monotonic() - started
        if process.poll() is not None:
            raise SystemExit(f"{server_label} process exited before becoming ready. exit_code={process.returncode}")
        model_ids = fetch_model_ids(server_url, timeout=min(10.0, max(1.0, poll_seconds)))
        if model_ids is not None:
            return model_ids
        if elapsed >= timeout_seconds:
            raise SystemExit(f"Timed out waiting for {server_label} readiness after {timeout_seconds:.0f} seconds.")
        if elapsed >= next_report:
            print(f"{label}_waiting_for_server elapsed_seconds={int(elapsed)} health_url={models_url(server_url)}")
            next_report = elapsed + max(1.0, poll_seconds)
        time.sleep(max(1.0, poll_seconds))


def build_benchmark_command(
    *,
    benchmark: str,
    benchmark_args: Sequence[str],
    server_url: str,
    server_model_id: str,
    judge_server_url: Optional[str],
    judge_model_id: Optional[str],
) -> List[str]:
    validate_no_conflicting_downstream_options(benchmark_args, auto_judge_enabled=judge_server_url is not None)
    if benchmark in {"if_exist", "counting", "medical_modality"}:
        script = REPO_ROOT / "eval_code" / benchmark / "eval_server_vlm.py"
        command = [sys.executable, str(script), *benchmark_args]
        if judge_server_url is not None and judge_model_id is not None:
            command.extend(["--judge-provider", "server", "--judge-model", judge_model_id, "--judge-server-url", judge_server_url])
    elif benchmark == "fashion_industry":
        script = REPO_ROOT / "eval_code" / "fashion_industry" / "eval_pipeline.py"
        command = [sys.executable, str(script), *benchmark_args]
        command = ensure_fashion_server_backend(command)
        if judge_server_url is not None and judge_model_id is not None:
            command.extend(["--judge-provider", "server", "--judge", judge_model_id, "--judge-server-url", judge_server_url])
    else:  # pragma: no cover - argparse prevents this.
        raise ValueError(f"Unsupported benchmark: {benchmark}")
    command.extend(["--server-url", server_url, "--server-model-id", server_model_id])
    return command


def validate_no_conflicting_downstream_options(benchmark_args: Sequence[str], *, auto_judge_enabled: bool) -> None:
    conflicts = set(SERVER_CONNECTION_OPTIONS)
    if auto_judge_enabled:
        conflicts.update(AUTO_JUDGE_OPTIONS)
    for token in benchmark_args:
        option = token.split("=", 1)[0]
        if option in conflicts:
            raise SystemExit(
                f"Do not pass {option} after '--'. The wrapper supplies that option automatically."
            )


def ensure_fashion_server_backend(command: List[str]) -> List[str]:
    for token in command:
        if token.startswith("--open-backend="):
            value = token.split("=", 1)[1]
            if value != "server":
                raise SystemExit("run_server_eval.py requires --open-backend server for fashion_industry.")
            return command
    if "--open-backend" not in command:
        return [*command, "--open-backend", "server"]
    index = command.index("--open-backend")
    try:
        value = command[index + 1]
    except IndexError as exc:
        raise SystemExit("--open-backend requires a value.") from exc
    if value != "server":
        raise SystemExit("run_server_eval.py requires --open-backend server for fashion_industry.")
    return command


def run_benchmark(command: Sequence[str]) -> int:
    process = subprocess.Popen(list(command), start_new_session=True)
    try:
        return process.wait()
    except KeyboardInterrupt:
        terminate_process_group(process, label="benchmark")
        raise


def terminate_process_group(process: subprocess.Popen[Any], *, label: str, timeout: float = 30.0) -> None:
    if process.poll() is not None:
        return
    try:
        if hasattr(os, "killpg"):
            os.killpg(process.pid, signal.SIGTERM)
        else:
            process.terminate()
        process.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        print(f"{label}_termination=timeout; sending SIGKILL")
        if hasattr(os, "killpg"):
            os.killpg(process.pid, signal.SIGKILL)
        else:
            process.kill()
        process.wait(timeout=timeout)


def shell_join(command: Sequence[str]) -> str:
    return " ".join(shell_quote(part) for part in command)


def shell_quote(value: Any) -> str:
    text = str(value)
    if not text:
        return "''"
    safe_chars = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_+-./:=,@")
    if all(char in safe_chars for char in text):
        return text
    return "'" + text.replace("'", "'\"'\"'") + "'"


if __name__ == "__main__":
    sys.exit(main())
