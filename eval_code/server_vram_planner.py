#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple


BYTES_PER_GIB = 1024**3
DEFAULT_GPU_MEMORY_UTILIZATION = 0.90
DEFAULT_RESERVE_GB = 4.0
DEFAULT_MODEL_SAFETY_FACTOR = 1.10
FOUR_BIT_MEMORY_FACTOR = 0.35
SUPPORTED_SERVER_DTYPES = ("auto", "bfloat16", "float16", "float32")
SUPPORTED_SERVER_PREFLIGHT_MODES = ("off", "check", "strict")
SUPPORTED_SERVER_FRAMEWORKS = ("vllm", "sglang", "generic")
SUPPORTED_GPU_PLACEMENT_POLICIES = ("auto", "separate", "shared", "explicit")
DEFAULT_SHARED_GPU_TOTAL_UTILIZATION = 0.88
DEFAULT_SHARED_GPU_MIN_SERVER_UTILIZATION = 0.15
FAKE_CUDA_ENV = "VLM_EVAL_FAKE_CUDA_SNAPSHOT"


@dataclass(frozen=True)
class CudaDeviceSnapshot:
    index: int
    name: str
    free_bytes: int
    total_bytes: int

    def to_json(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ServerVramPlan:
    ok: bool
    model_id: str
    dtype: str
    load_in_4bit: bool
    tensor_parallel_size: Optional[int]
    tensor_parallel_request: str
    gpu_ids: List[int]
    assigned_gpu_ids: List[int]
    estimated_model_bytes: Optional[int]
    estimated_required_bytes: Optional[int]
    estimated_per_gpu_bytes: Optional[int]
    parameter_count_billions: Optional[float]
    parameter_source: Optional[str]
    gpu_memory_utilization: float
    reserve_gb: float
    available_bytes_by_device: Dict[str, int]
    devices: List[Dict[str, Any]]
    failure_reason: Optional[str] = None

    def to_json(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class DualServerVramPlan:
    ok: bool
    placement_policy: str
    placement_mode: str
    devices: List[Dict[str, Any]]
    backbone_plan: Optional[ServerVramPlan]
    judge_plan: Optional[ServerVramPlan]
    shared_gpu_total_utilization: float
    shared_gpu_min_server_utilization: float
    failure_reason: Optional[str] = None

    def to_json(self) -> Dict[str, Any]:
        return {
            "ok": self.ok,
            "placement_policy": self.placement_policy,
            "placement_mode": self.placement_mode,
            "devices": self.devices,
            "backbone_plan": self.backbone_plan.to_json() if self.backbone_plan is not None else None,
            "judge_plan": self.judge_plan.to_json() if self.judge_plan is not None else None,
            "shared_gpu_total_utilization": self.shared_gpu_total_utilization,
            "shared_gpu_min_server_utilization": self.shared_gpu_min_server_utilization,
            "failure_reason": self.failure_reason,
        }


def add_server_preflight_arguments(parser: argparse.ArgumentParser, *, include_judge: bool = True) -> None:
    parser.add_argument(
        "--server-preflight",
        choices=SUPPORTED_SERVER_PREFLIGHT_MODES,
        default="off",
        help="Run local CUDA VRAM planning for an OpenAI-compatible server before evaluation.",
    )
    parser.add_argument(
        "--server-framework",
        choices=SUPPORTED_SERVER_FRAMEWORKS,
        default="generic",
        help="Serving framework assumed by the preflight or launcher.",
    )
    parser.add_argument("--server-gpu-ids", default=None, help="Comma-separated CUDA device ids for the backbone server.")
    parser.add_argument(
        "--server-dtype",
        choices=SUPPORTED_SERVER_DTYPES,
        default="bfloat16",
        help="Weight dtype assumed by the backbone server preflight.",
    )
    parser.add_argument(
        "--server-tensor-parallel-size",
        default="auto",
        help="Tensor parallel size for the backbone server preflight; use auto or an integer.",
    )
    parser.add_argument(
        "--server-gpu-memory-utilization",
        type=float,
        default=DEFAULT_GPU_MEMORY_UTILIZATION,
        help="Fraction of each selected GPU that the server may use.",
    )
    parser.add_argument(
        "--server-reserve-gb",
        type=float,
        default=DEFAULT_RESERVE_GB,
        help="Additional GiB to reserve on each selected GPU.",
    )
    parser.add_argument("--server-load-in-4bit", action="store_true", help="Estimate backbone server memory as explicit 4-bit loading.")
    parser.add_argument(
        "--server-hf-model-id",
        default=None,
        help="HF model id used for VRAM estimation when the served model name is an alias.",
    )
    if not include_judge:
        return
    parser.add_argument("--judge-server-gpu-ids", default=None, help="Comma-separated CUDA device ids for a local judge server.")
    parser.add_argument(
        "--judge-server-dtype",
        choices=SUPPORTED_SERVER_DTYPES,
        default=None,
        help="Weight dtype assumed by the judge server preflight. Defaults to --server-dtype.",
    )
    parser.add_argument(
        "--judge-server-tensor-parallel-size",
        default=None,
        help="Tensor parallel size for judge server preflight. Defaults to --server-tensor-parallel-size.",
    )
    parser.add_argument(
        "--judge-server-gpu-memory-utilization",
        type=float,
        default=None,
        help="Judge server GPU utilization. Defaults to --server-gpu-memory-utilization.",
    )
    parser.add_argument(
        "--judge-server-reserve-gb",
        type=float,
        default=None,
        help="Judge server per-GPU reserve. Defaults to --server-reserve-gb.",
    )
    parser.add_argument("--judge-server-load-in-4bit", action="store_true", help="Estimate judge server memory as explicit 4-bit loading.")
    parser.add_argument("--judge-server-hf-model-id", default=None, help="HF model id used for judge server VRAM estimation.")


def capture_cuda_devices() -> List[CudaDeviceSnapshot]:
    fake_snapshot = os.getenv(FAKE_CUDA_ENV)
    if fake_snapshot:
        return parse_fake_cuda_snapshot(fake_snapshot)
    try:
        import torch
    except Exception:
        return []
    if not torch.cuda.is_available():
        return []
    devices: List[CudaDeviceSnapshot] = []
    torch.cuda.empty_cache()
    for index in range(torch.cuda.device_count()):
        free_bytes, total_bytes = torch.cuda.mem_get_info(index)
        properties = torch.cuda.get_device_properties(index)
        devices.append(
            CudaDeviceSnapshot(
                index=index,
                name=str(properties.name),
                free_bytes=int(free_bytes),
                total_bytes=int(total_bytes),
            )
        )
    return devices


def parse_fake_cuda_snapshot(raw: str) -> List[CudaDeviceSnapshot]:
    payload = json.loads(raw)
    if not isinstance(payload, list):
        raise ValueError(f"{FAKE_CUDA_ENV} must be a JSON list.")
    devices = []
    for item in payload:
        index = int(item["index"])
        name = str(item.get("name", f"Fake CUDA {index}"))
        if "free_bytes" in item:
            free_bytes = int(item["free_bytes"])
        else:
            free_bytes = int(float(item["free_gb"]) * BYTES_PER_GIB)
        if "total_bytes" in item:
            total_bytes = int(item["total_bytes"])
        else:
            total_bytes = int(float(item.get("total_gb", item.get("free_gb", 0))) * BYTES_PER_GIB)
        devices.append(CudaDeviceSnapshot(index=index, name=name, free_bytes=free_bytes, total_bytes=total_bytes))
    return devices


def dtype_element_size(dtype_name: str) -> int:
    normalized = dtype_name.lower()
    if normalized in {"auto", "bfloat16", "float16", "fp16", "bf16"}:
        return 2
    if normalized in {"float32", "fp32"}:
        return 4
    raise ValueError(f"Unsupported dtype for server preflight: {dtype_name}")


def parse_gpu_ids(raw_value: Optional[str], devices: Sequence[CudaDeviceSnapshot]) -> List[int]:
    available_ids = [device.index for device in devices]
    if raw_value is None or str(raw_value).strip() == "":
        return available_ids
    parsed = [int(part.strip()) for part in str(raw_value).split(",") if part.strip()]
    missing = [device_id for device_id in parsed if device_id not in available_ids]
    if missing:
        raise ValueError(f"Requested CUDA devices are not visible: {missing}; visible={available_ids}")
    return parsed


def parse_tensor_parallel_size(raw_value: Any, *, max_devices: int) -> Optional[int]:
    value = str(raw_value or "auto").strip().lower()
    if value == "auto":
        return None
    try:
        parsed = int(value)
    except ValueError as exc:
        raise ValueError(f"tensor_parallel_size must be auto or a positive integer, got {raw_value!r}") from exc
    if parsed <= 0:
        raise ValueError("tensor_parallel_size must be positive.")
    if parsed > max_devices:
        raise ValueError(f"tensor_parallel_size={parsed} exceeds selected CUDA device count={max_devices}.")
    return parsed


def infer_parameter_count(model_id: str) -> Tuple[Optional[float], Optional[str]]:
    match = re.search(r"[eE](\d+(?:\.\d+)?)\s*[bB](?:$|[-_/])", model_id)
    if match:
        return float(match.group(1)) * 1_000_000_000, "model_id_e_size_suffix"
    match = re.search(r"(?<![\w.])(\d+(?:\.\d+)?)\s*[bB](?![\w.])", model_id)
    if match:
        return float(match.group(1)) * 1_000_000_000, "model_id_size_suffix"
    match = re.search(r"(\d+(?:\.\d+)?)\s*[-_]?b(?:$|[-_/])", model_id.lower())
    if match:
        return float(match.group(1)) * 1_000_000_000, "model_id_size_suffix"
    if "/" not in model_id and not Path(model_id).exists():
        return None, None
    try:
        from transformers import AutoConfig
    except Exception:
        return None, None
    try:
        config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
    except Exception:
        return None, None
    for attr in ("num_parameters", "n_params", "parameter_count"):
        value = getattr(config, attr, None)
        if isinstance(value, (int, float)) and value > 0:
            return float(value), f"config.{attr}"
    return None, None


def estimate_model_bytes(model_id: str, *, dtype: str, load_in_4bit: bool) -> Tuple[Optional[int], Optional[float], Optional[str]]:
    parameter_count, source = infer_parameter_count(model_id)
    if parameter_count is None:
        return None, None, None
    if load_in_4bit:
        bytes_per_param = dtype_element_size("float16") * FOUR_BIT_MEMORY_FACTOR
    else:
        bytes_per_param = dtype_element_size(dtype)
    return int(parameter_count * bytes_per_param), parameter_count / 1_000_000_000, source


def plan_server_vram(
    *,
    model_id: str,
    devices: Optional[Sequence[CudaDeviceSnapshot]] = None,
    gpu_ids: Optional[str] = None,
    tensor_parallel_size: Any = "auto",
    dtype: str = "bfloat16",
    load_in_4bit: bool = False,
    gpu_memory_utilization: float = DEFAULT_GPU_MEMORY_UTILIZATION,
    reserve_gb: float = DEFAULT_RESERVE_GB,
) -> ServerVramPlan:
    device_list = list(devices) if devices is not None else capture_cuda_devices()
    all_device_json = [device.to_json() for device in device_list]
    if not device_list:
        return failed_plan(
            model_id=model_id,
            dtype=dtype,
            load_in_4bit=load_in_4bit,
            tensor_parallel_request=str(tensor_parallel_size),
            gpu_memory_utilization=gpu_memory_utilization,
            reserve_gb=reserve_gb,
            devices=all_device_json,
            failure_reason="No visible CUDA devices were found for local server preflight.",
        )
    try:
        selected_gpu_ids = parse_gpu_ids(gpu_ids, device_list)
        requested_tp = parse_tensor_parallel_size(tensor_parallel_size, max_devices=len(selected_gpu_ids))
    except ValueError as exc:
        return failed_plan(
            model_id=model_id,
            dtype=dtype,
            load_in_4bit=load_in_4bit,
            tensor_parallel_request=str(tensor_parallel_size),
            gpu_memory_utilization=gpu_memory_utilization,
            reserve_gb=reserve_gb,
            devices=all_device_json,
            failure_reason=str(exc),
        )
    selected_devices = [device for device in device_list if device.index in selected_gpu_ids]
    estimated_bytes, parameter_count_billions, parameter_source = estimate_model_bytes(model_id, dtype=dtype, load_in_4bit=load_in_4bit)
    if estimated_bytes is None:
        return failed_plan(
            model_id=model_id,
            dtype=dtype,
            load_in_4bit=load_in_4bit,
            tensor_parallel_request=str(tensor_parallel_size),
            gpu_ids=selected_gpu_ids,
            gpu_memory_utilization=gpu_memory_utilization,
            reserve_gb=reserve_gb,
            devices=all_device_json,
            failure_reason=(
                "Unable to estimate parameter count from model id or HF config. "
                "Use an HF/model id containing a size suffix such as 8B/32B/72B, "
                "or pass a model/config with a readable parameter count."
            ),
        )
    required_bytes = int(estimated_bytes * DEFAULT_MODEL_SAFETY_FACTOR)
    available = available_bytes_by_device(selected_devices, gpu_memory_utilization=gpu_memory_utilization, reserve_gb=reserve_gb)
    assigned_gpu_ids, chosen_tp, per_gpu_bytes = choose_tensor_parallel_assignment(
        selected_devices=selected_devices,
        available_bytes=available,
        required_bytes=required_bytes,
        requested_tp=requested_tp,
    )
    if not assigned_gpu_ids or chosen_tp is None or per_gpu_bytes is None:
        return failed_plan(
            model_id=model_id,
            dtype=dtype,
            load_in_4bit=load_in_4bit,
            tensor_parallel_request=str(tensor_parallel_size),
            gpu_ids=selected_gpu_ids,
            estimated_model_bytes=estimated_bytes,
            estimated_required_bytes=required_bytes,
            parameter_count_billions=parameter_count_billions,
            parameter_source=parameter_source,
            gpu_memory_utilization=gpu_memory_utilization,
            reserve_gb=reserve_gb,
            available_bytes_by_device={str(k): v for k, v in available.items()},
            devices=all_device_json,
            failure_reason=build_vram_failure_reason(required_bytes, available, selected_gpu_ids, requested_tp),
        )
    return ServerVramPlan(
        ok=True,
        model_id=model_id,
        dtype=dtype,
        load_in_4bit=load_in_4bit,
        tensor_parallel_size=chosen_tp,
        tensor_parallel_request=str(tensor_parallel_size),
        gpu_ids=selected_gpu_ids,
        assigned_gpu_ids=assigned_gpu_ids,
        estimated_model_bytes=estimated_bytes,
        estimated_required_bytes=required_bytes,
        estimated_per_gpu_bytes=per_gpu_bytes,
        parameter_count_billions=parameter_count_billions,
        parameter_source=parameter_source,
        gpu_memory_utilization=gpu_memory_utilization,
        reserve_gb=reserve_gb,
        available_bytes_by_device={str(k): v for k, v in available.items()},
        devices=all_device_json,
        failure_reason=None,
    )


def available_bytes_by_device(
    devices: Sequence[CudaDeviceSnapshot],
    *,
    gpu_memory_utilization: float,
    reserve_gb: float,
) -> Dict[int, int]:
    reserve_bytes = int(max(0.0, reserve_gb) * BYTES_PER_GIB)
    return {
        device.index: max(0, int(device.free_bytes * gpu_memory_utilization) - reserve_bytes)
        for device in devices
    }


def choose_tensor_parallel_assignment(
    *,
    selected_devices: Sequence[CudaDeviceSnapshot],
    available_bytes: Dict[int, int],
    required_bytes: int,
    requested_tp: Optional[int],
) -> Tuple[List[int], Optional[int], Optional[int]]:
    ordered = sorted(selected_devices, key=lambda device: (available_bytes.get(device.index, 0), -device.index), reverse=True)
    tp_candidates = [requested_tp] if requested_tp is not None else list(range(1, len(ordered) + 1))
    for tp in tp_candidates:
        if tp is None or tp > len(ordered):
            continue
        per_gpu_bytes = int(math.ceil(required_bytes / tp))
        assigned = [device.index for device in ordered[:tp]]
        if all(available_bytes.get(device_id, 0) >= per_gpu_bytes for device_id in assigned):
            return assigned, tp, per_gpu_bytes
    return [], None, None


def failed_plan(
    *,
    model_id: str,
    dtype: str,
    load_in_4bit: bool,
    tensor_parallel_request: str,
    gpu_memory_utilization: float,
    reserve_gb: float,
    devices: List[Dict[str, Any]],
    failure_reason: str,
    gpu_ids: Optional[List[int]] = None,
    estimated_model_bytes: Optional[int] = None,
    estimated_required_bytes: Optional[int] = None,
    parameter_count_billions: Optional[float] = None,
    parameter_source: Optional[str] = None,
    available_bytes_by_device: Optional[Dict[str, int]] = None,
) -> ServerVramPlan:
    return ServerVramPlan(
        ok=False,
        model_id=model_id,
        dtype=dtype,
        load_in_4bit=load_in_4bit,
        tensor_parallel_size=None,
        tensor_parallel_request=tensor_parallel_request,
        gpu_ids=gpu_ids or [],
        assigned_gpu_ids=[],
        estimated_model_bytes=estimated_model_bytes,
        estimated_required_bytes=estimated_required_bytes,
        estimated_per_gpu_bytes=None,
        parameter_count_billions=parameter_count_billions,
        parameter_source=parameter_source,
        gpu_memory_utilization=gpu_memory_utilization,
        reserve_gb=reserve_gb,
        available_bytes_by_device=available_bytes_by_device or {},
        devices=devices,
        failure_reason=failure_reason,
    )


def build_vram_failure_reason(required_bytes: int, available: Dict[int, int], gpu_ids: Sequence[int], requested_tp: Optional[int]) -> str:
    requested = f"requested tensor_parallel_size={requested_tp}" if requested_tp is not None else "auto tensor_parallel_size"
    available_gib = {device_id: round(bytes_value / BYTES_PER_GIB, 2) for device_id, bytes_value in available.items()}
    return (
        f"Insufficient VRAM for {requested}. Estimated required total={required_bytes / BYTES_PER_GIB:.2f} GiB; "
        f"selected GPUs={list(gpu_ids)} available_after_reserve_gib={available_gib}."
    )


def subtract_plan_from_devices(devices: Sequence[CudaDeviceSnapshot], plan: ServerVramPlan) -> List[CudaDeviceSnapshot]:
    if not plan.ok or not plan.estimated_per_gpu_bytes:
        return list(devices)
    assigned = set(plan.assigned_gpu_ids)
    updated = []
    for device in devices:
        if device.index in assigned:
            updated.append(
                CudaDeviceSnapshot(
                    index=device.index,
                    name=device.name,
                    free_bytes=max(0, device.free_bytes - plan.estimated_per_gpu_bytes),
                    total_bytes=device.total_bytes,
                )
            )
        else:
            updated.append(device)
    return updated


def plan_dual_server_vram(
    *,
    backbone_model_id: str,
    judge_model_id: Optional[str] = None,
    devices: Optional[Sequence[CudaDeviceSnapshot]] = None,
    placement_policy: str = "auto",
    backbone_gpu_ids: Optional[str] = None,
    judge_gpu_ids: Optional[str] = None,
    backbone_tensor_parallel_size: Any = "auto",
    judge_tensor_parallel_size: Any = "auto",
    backbone_dtype: str = "bfloat16",
    judge_dtype: str = "bfloat16",
    backbone_load_in_4bit: bool = False,
    judge_load_in_4bit: bool = False,
    backbone_gpu_memory_utilization: float = DEFAULT_GPU_MEMORY_UTILIZATION,
    judge_gpu_memory_utilization: float = DEFAULT_GPU_MEMORY_UTILIZATION,
    backbone_reserve_gb: float = DEFAULT_RESERVE_GB,
    judge_reserve_gb: float = DEFAULT_RESERVE_GB,
    shared_gpu_total_utilization: float = DEFAULT_SHARED_GPU_TOTAL_UTILIZATION,
    shared_gpu_min_server_utilization: float = DEFAULT_SHARED_GPU_MIN_SERVER_UTILIZATION,
) -> DualServerVramPlan:
    device_list = list(devices) if devices is not None else capture_cuda_devices()
    device_json = [device.to_json() for device in device_list]
    policy = placement_policy.strip().lower()
    if policy not in SUPPORTED_GPU_PLACEMENT_POLICIES:
        return failed_dual_plan(
            placement_policy=placement_policy,
            devices=device_json,
            shared_gpu_total_utilization=shared_gpu_total_utilization,
            shared_gpu_min_server_utilization=shared_gpu_min_server_utilization,
            failure_reason=f"Unsupported GPU placement policy: {placement_policy!r}.",
        )
    if not device_list:
        return failed_dual_plan(
            placement_policy=policy,
            devices=device_json,
            shared_gpu_total_utilization=shared_gpu_total_utilization,
            shared_gpu_min_server_utilization=shared_gpu_min_server_utilization,
            failure_reason="No visible CUDA devices were found for automatic GPU placement.",
        )
    if not judge_model_id:
        backbone_plan = plan_server_vram(
            model_id=backbone_model_id,
            devices=device_list,
            gpu_ids=backbone_gpu_ids,
            tensor_parallel_size=backbone_tensor_parallel_size,
            dtype=backbone_dtype,
            load_in_4bit=backbone_load_in_4bit,
            gpu_memory_utilization=backbone_gpu_memory_utilization,
            reserve_gb=backbone_reserve_gb,
        )
        return DualServerVramPlan(
            ok=backbone_plan.ok,
            placement_policy=policy,
            placement_mode="single" if backbone_plan.ok else "failed",
            devices=device_json,
            backbone_plan=backbone_plan,
            judge_plan=None,
            shared_gpu_total_utilization=shared_gpu_total_utilization,
            shared_gpu_min_server_utilization=shared_gpu_min_server_utilization,
            failure_reason=None if backbone_plan.ok else backbone_plan.failure_reason,
        )

    explicit_requested = bool(backbone_gpu_ids or judge_gpu_ids or policy == "explicit")
    if explicit_requested:
        return plan_dual_explicit(
            devices=device_list,
            placement_policy="explicit",
            backbone_model_id=backbone_model_id,
            judge_model_id=judge_model_id,
            backbone_gpu_ids=backbone_gpu_ids,
            judge_gpu_ids=judge_gpu_ids,
            backbone_tensor_parallel_size=backbone_tensor_parallel_size,
            judge_tensor_parallel_size=judge_tensor_parallel_size,
            backbone_dtype=backbone_dtype,
            judge_dtype=judge_dtype,
            backbone_load_in_4bit=backbone_load_in_4bit,
            judge_load_in_4bit=judge_load_in_4bit,
            backbone_gpu_memory_utilization=backbone_gpu_memory_utilization,
            judge_gpu_memory_utilization=judge_gpu_memory_utilization,
            backbone_reserve_gb=backbone_reserve_gb,
            judge_reserve_gb=judge_reserve_gb,
            shared_gpu_total_utilization=shared_gpu_total_utilization,
            shared_gpu_min_server_utilization=shared_gpu_min_server_utilization,
        )

    if policy == "separate":
        return plan_dual_separate(
            devices=device_list,
            placement_policy=policy,
            backbone_model_id=backbone_model_id,
            judge_model_id=judge_model_id,
            backbone_tensor_parallel_size=backbone_tensor_parallel_size,
            judge_tensor_parallel_size=judge_tensor_parallel_size,
            backbone_dtype=backbone_dtype,
            judge_dtype=judge_dtype,
            backbone_load_in_4bit=backbone_load_in_4bit,
            judge_load_in_4bit=judge_load_in_4bit,
            backbone_gpu_memory_utilization=backbone_gpu_memory_utilization,
            judge_gpu_memory_utilization=judge_gpu_memory_utilization,
            backbone_reserve_gb=backbone_reserve_gb,
            judge_reserve_gb=judge_reserve_gb,
            shared_gpu_total_utilization=shared_gpu_total_utilization,
            shared_gpu_min_server_utilization=shared_gpu_min_server_utilization,
        )
    if policy == "shared":
        return plan_dual_shared(
            devices=device_list,
            placement_policy=policy,
            backbone_model_id=backbone_model_id,
            judge_model_id=judge_model_id,
            backbone_tensor_parallel_size=backbone_tensor_parallel_size,
            judge_tensor_parallel_size=judge_tensor_parallel_size,
            backbone_dtype=backbone_dtype,
            judge_dtype=judge_dtype,
            backbone_load_in_4bit=backbone_load_in_4bit,
            judge_load_in_4bit=judge_load_in_4bit,
            backbone_reserve_gb=backbone_reserve_gb,
            judge_reserve_gb=judge_reserve_gb,
            shared_gpu_total_utilization=shared_gpu_total_utilization,
            shared_gpu_min_server_utilization=shared_gpu_min_server_utilization,
        )

    separate_plan = plan_dual_separate(
        devices=device_list,
        placement_policy=policy,
        backbone_model_id=backbone_model_id,
        judge_model_id=judge_model_id,
        backbone_tensor_parallel_size=backbone_tensor_parallel_size,
        judge_tensor_parallel_size=judge_tensor_parallel_size,
        backbone_dtype=backbone_dtype,
        judge_dtype=judge_dtype,
        backbone_load_in_4bit=backbone_load_in_4bit,
        judge_load_in_4bit=judge_load_in_4bit,
        backbone_gpu_memory_utilization=backbone_gpu_memory_utilization,
        judge_gpu_memory_utilization=judge_gpu_memory_utilization,
        backbone_reserve_gb=backbone_reserve_gb,
        judge_reserve_gb=judge_reserve_gb,
        shared_gpu_total_utilization=shared_gpu_total_utilization,
        shared_gpu_min_server_utilization=shared_gpu_min_server_utilization,
    )
    if separate_plan.ok:
        return separate_plan
    shared_plan = plan_dual_shared(
        devices=device_list,
        placement_policy=policy,
        backbone_model_id=backbone_model_id,
        judge_model_id=judge_model_id,
        backbone_tensor_parallel_size=backbone_tensor_parallel_size,
        judge_tensor_parallel_size=judge_tensor_parallel_size,
        backbone_dtype=backbone_dtype,
        judge_dtype=judge_dtype,
        backbone_load_in_4bit=backbone_load_in_4bit,
        judge_load_in_4bit=judge_load_in_4bit,
        backbone_reserve_gb=backbone_reserve_gb,
        judge_reserve_gb=judge_reserve_gb,
        shared_gpu_total_utilization=shared_gpu_total_utilization,
        shared_gpu_min_server_utilization=shared_gpu_min_server_utilization,
    )
    if shared_plan.ok:
        return shared_plan
    return failed_dual_plan(
        placement_policy=policy,
        devices=device_json,
        backbone_plan=separate_plan.backbone_plan,
        judge_plan=separate_plan.judge_plan or shared_plan.judge_plan,
        shared_gpu_total_utilization=shared_gpu_total_utilization,
        shared_gpu_min_server_utilization=shared_gpu_min_server_utilization,
        failure_reason=(
            "Automatic GPU placement failed. "
            f"Separate failure: {separate_plan.failure_reason}; "
            f"Shared failure: {shared_plan.failure_reason}"
        ),
    )


def plan_dual_separate(
    *,
    devices: Sequence[CudaDeviceSnapshot],
    placement_policy: str,
    backbone_model_id: str,
    judge_model_id: str,
    backbone_tensor_parallel_size: Any,
    judge_tensor_parallel_size: Any,
    backbone_dtype: str,
    judge_dtype: str,
    backbone_load_in_4bit: bool,
    judge_load_in_4bit: bool,
    backbone_gpu_memory_utilization: float,
    judge_gpu_memory_utilization: float,
    backbone_reserve_gb: float,
    judge_reserve_gb: float,
    shared_gpu_total_utilization: float,
    shared_gpu_min_server_utilization: float,
) -> DualServerVramPlan:
    device_json = [device.to_json() for device in devices]
    backbone_plan = plan_server_vram(
        model_id=backbone_model_id,
        devices=devices,
        gpu_ids=None,
        tensor_parallel_size=backbone_tensor_parallel_size,
        dtype=backbone_dtype,
        load_in_4bit=backbone_load_in_4bit,
        gpu_memory_utilization=backbone_gpu_memory_utilization,
        reserve_gb=backbone_reserve_gb,
    )
    if not backbone_plan.ok:
        return failed_dual_plan(
            placement_policy=placement_policy,
            placement_mode="separate_failed",
            devices=device_json,
            backbone_plan=backbone_plan,
            shared_gpu_total_utilization=shared_gpu_total_utilization,
            shared_gpu_min_server_utilization=shared_gpu_min_server_utilization,
            failure_reason=f"Backbone server cannot be placed separately: {backbone_plan.failure_reason}",
        )
    remaining_devices = [device for device in devices if device.index not in set(backbone_plan.assigned_gpu_ids)]
    if not remaining_devices:
        return failed_dual_plan(
            placement_policy=placement_policy,
            placement_mode="separate_failed",
            devices=device_json,
            backbone_plan=backbone_plan,
            shared_gpu_total_utilization=shared_gpu_total_utilization,
            shared_gpu_min_server_utilization=shared_gpu_min_server_utilization,
            failure_reason=(
                "Backbone placement uses all visible GPUs, leaving no separate GPU for judge server."
            ),
        )
    judge_plan = plan_server_vram(
        model_id=judge_model_id,
        devices=remaining_devices,
        gpu_ids=None,
        tensor_parallel_size=judge_tensor_parallel_size,
        dtype=judge_dtype,
        load_in_4bit=judge_load_in_4bit,
        gpu_memory_utilization=judge_gpu_memory_utilization,
        reserve_gb=judge_reserve_gb,
    )
    if not judge_plan.ok:
        return failed_dual_plan(
            placement_policy=placement_policy,
            placement_mode="separate_failed",
            devices=device_json,
            backbone_plan=backbone_plan,
            judge_plan=judge_plan,
            shared_gpu_total_utilization=shared_gpu_total_utilization,
            shared_gpu_min_server_utilization=shared_gpu_min_server_utilization,
            failure_reason=f"Judge server cannot be placed on remaining GPUs: {judge_plan.failure_reason}",
        )
    return DualServerVramPlan(
        ok=True,
        placement_policy=placement_policy,
        placement_mode="separate",
        devices=device_json,
        backbone_plan=backbone_plan,
        judge_plan=judge_plan,
        shared_gpu_total_utilization=shared_gpu_total_utilization,
        shared_gpu_min_server_utilization=shared_gpu_min_server_utilization,
    )


def plan_dual_shared(
    *,
    devices: Sequence[CudaDeviceSnapshot],
    placement_policy: str,
    backbone_model_id: str,
    judge_model_id: str,
    backbone_tensor_parallel_size: Any,
    judge_tensor_parallel_size: Any,
    backbone_dtype: str,
    judge_dtype: str,
    backbone_load_in_4bit: bool,
    judge_load_in_4bit: bool,
    backbone_reserve_gb: float,
    judge_reserve_gb: float,
    shared_gpu_total_utilization: float,
    shared_gpu_min_server_utilization: float,
) -> DualServerVramPlan:
    device_json = [device.to_json() for device in devices]
    if not devices:
        return failed_dual_plan(
            placement_policy=placement_policy,
            placement_mode="shared_failed",
            devices=device_json,
            shared_gpu_total_utilization=shared_gpu_total_utilization,
            shared_gpu_min_server_utilization=shared_gpu_min_server_utilization,
            failure_reason="No visible CUDA devices are available for shared placement.",
        )
    if str(backbone_tensor_parallel_size).strip().lower() not in {"auto", "1"}:
        return failed_dual_plan(
            placement_policy=placement_policy,
            placement_mode="shared_failed",
            devices=device_json,
            shared_gpu_total_utilization=shared_gpu_total_utilization,
            shared_gpu_min_server_utilization=shared_gpu_min_server_utilization,
            failure_reason="Shared placement only supports tensor_parallel_size auto or 1 for the backbone server.",
        )
    if str(judge_tensor_parallel_size).strip().lower() not in {"auto", "1", "none"}:
        return failed_dual_plan(
            placement_policy=placement_policy,
            placement_mode="shared_failed",
            devices=device_json,
            shared_gpu_total_utilization=shared_gpu_total_utilization,
            shared_gpu_min_server_utilization=shared_gpu_min_server_utilization,
            failure_reason="Shared placement only supports tensor_parallel_size auto or 1 for the judge server.",
        )
    device = max(devices, key=lambda item: item.free_bytes)
    backbone_required = estimate_required_bytes_for_model(
        backbone_model_id,
        dtype=backbone_dtype,
        load_in_4bit=backbone_load_in_4bit,
    )
    judge_required = estimate_required_bytes_for_model(
        judge_model_id,
        dtype=judge_dtype,
        load_in_4bit=judge_load_in_4bit,
    )
    if backbone_required[0] is None:
        return failed_dual_plan(
            placement_policy=placement_policy,
            placement_mode="shared_failed",
            devices=device_json,
            shared_gpu_total_utilization=shared_gpu_total_utilization,
            shared_gpu_min_server_utilization=shared_gpu_min_server_utilization,
            failure_reason=f"Unable to estimate backbone model size for shared placement: {backbone_model_id}",
        )
    if judge_required[0] is None:
        return failed_dual_plan(
            placement_policy=placement_policy,
            placement_mode="shared_failed",
            devices=device_json,
            shared_gpu_total_utilization=shared_gpu_total_utilization,
            shared_gpu_min_server_utilization=shared_gpu_min_server_utilization,
            failure_reason=f"Unable to estimate judge model size for shared placement: {judge_model_id}",
        )
    if shared_gpu_total_utilization <= 0 or shared_gpu_total_utilization > 1:
        return failed_dual_plan(
            placement_policy=placement_policy,
            placement_mode="shared_failed",
            devices=device_json,
            shared_gpu_total_utilization=shared_gpu_total_utilization,
            shared_gpu_min_server_utilization=shared_gpu_min_server_utilization,
            failure_reason="--shared-gpu-total-utilization must be in the range (0, 1].",
        )
    if shared_gpu_min_server_utilization < 0 or shared_gpu_min_server_utilization > shared_gpu_total_utilization:
        return failed_dual_plan(
            placement_policy=placement_policy,
            placement_mode="shared_failed",
            devices=device_json,
            shared_gpu_total_utilization=shared_gpu_total_utilization,
            shared_gpu_min_server_utilization=shared_gpu_min_server_utilization,
            failure_reason="--shared-gpu-min-server-utilization must be non-negative and no larger than total utilization.",
        )
    free_bytes = max(1, device.free_bytes)
    backbone_base = max(
        shared_gpu_min_server_utilization,
        (backbone_required[0] + int(backbone_reserve_gb * BYTES_PER_GIB)) / free_bytes,
    )
    judge_base = max(
        shared_gpu_min_server_utilization,
        (judge_required[0] + int(judge_reserve_gb * BYTES_PER_GIB)) / free_bytes,
    )
    base_total = backbone_base + judge_base
    if base_total > shared_gpu_total_utilization:
        return failed_dual_plan(
            placement_policy=placement_policy,
            placement_mode="shared_failed",
            devices=device_json,
            shared_gpu_total_utilization=shared_gpu_total_utilization,
            shared_gpu_min_server_utilization=shared_gpu_min_server_utilization,
            failure_reason=(
                f"Shared placement would require utilization={base_total:.3f}, "
                f"which exceeds limit={shared_gpu_total_utilization:.3f} on GPU {device.index}."
            ),
        )
    remaining = max(0.0, shared_gpu_total_utilization - base_total)
    backbone_weight = backbone_base / base_total if base_total > 0 else 0.5
    backbone_utilization = backbone_base + remaining * backbone_weight
    judge_utilization = judge_base + remaining * (1.0 - backbone_weight)
    gpu_id = str(device.index)
    backbone_plan = plan_server_vram(
        model_id=backbone_model_id,
        devices=devices,
        gpu_ids=gpu_id,
        tensor_parallel_size=1,
        dtype=backbone_dtype,
        load_in_4bit=backbone_load_in_4bit,
        gpu_memory_utilization=backbone_utilization,
        reserve_gb=backbone_reserve_gb,
    )
    judge_plan = plan_server_vram(
        model_id=judge_model_id,
        devices=devices,
        gpu_ids=gpu_id,
        tensor_parallel_size=1,
        dtype=judge_dtype,
        load_in_4bit=judge_load_in_4bit,
        gpu_memory_utilization=judge_utilization,
        reserve_gb=judge_reserve_gb,
    )
    if not backbone_plan.ok or not judge_plan.ok:
        return failed_dual_plan(
            placement_policy=placement_policy,
            placement_mode="shared_failed",
            devices=device_json,
            backbone_plan=backbone_plan,
            judge_plan=judge_plan,
            shared_gpu_total_utilization=shared_gpu_total_utilization,
            shared_gpu_min_server_utilization=shared_gpu_min_server_utilization,
            failure_reason=(
                f"Shared placement estimates passed but individual server planning failed: "
                f"backbone={backbone_plan.failure_reason}; judge={judge_plan.failure_reason}"
            ),
        )
    return DualServerVramPlan(
        ok=True,
        placement_policy=placement_policy,
        placement_mode="shared",
        devices=device_json,
        backbone_plan=backbone_plan,
        judge_plan=judge_plan,
        shared_gpu_total_utilization=shared_gpu_total_utilization,
        shared_gpu_min_server_utilization=shared_gpu_min_server_utilization,
    )


def plan_dual_explicit(
    *,
    devices: Sequence[CudaDeviceSnapshot],
    placement_policy: str,
    backbone_model_id: str,
    judge_model_id: str,
    backbone_gpu_ids: Optional[str],
    judge_gpu_ids: Optional[str],
    backbone_tensor_parallel_size: Any,
    judge_tensor_parallel_size: Any,
    backbone_dtype: str,
    judge_dtype: str,
    backbone_load_in_4bit: bool,
    judge_load_in_4bit: bool,
    backbone_gpu_memory_utilization: float,
    judge_gpu_memory_utilization: float,
    backbone_reserve_gb: float,
    judge_reserve_gb: float,
    shared_gpu_total_utilization: float,
    shared_gpu_min_server_utilization: float,
) -> DualServerVramPlan:
    device_json = [device.to_json() for device in devices]
    backbone_plan = plan_server_vram(
        model_id=backbone_model_id,
        devices=devices,
        gpu_ids=backbone_gpu_ids,
        tensor_parallel_size=backbone_tensor_parallel_size,
        dtype=backbone_dtype,
        load_in_4bit=backbone_load_in_4bit,
        gpu_memory_utilization=backbone_gpu_memory_utilization,
        reserve_gb=backbone_reserve_gb,
    )
    judge_plan = plan_server_vram(
        model_id=judge_model_id,
        devices=devices,
        gpu_ids=judge_gpu_ids,
        tensor_parallel_size=judge_tensor_parallel_size,
        dtype=judge_dtype,
        load_in_4bit=judge_load_in_4bit,
        gpu_memory_utilization=judge_gpu_memory_utilization,
        reserve_gb=judge_reserve_gb,
    )
    overlap = set(backbone_plan.assigned_gpu_ids) & set(judge_plan.assigned_gpu_ids)
    placement_mode = "explicit_shared" if overlap else "explicit_separate"
    failure_reason = None
    if not backbone_plan.ok:
        failure_reason = f"Explicit backbone placement failed: {backbone_plan.failure_reason}"
    elif not judge_plan.ok:
        failure_reason = f"Explicit judge placement failed: {judge_plan.failure_reason}"
    elif overlap:
        for device_id in overlap:
            device = next((item for item in devices if item.index == device_id), None)
            if device is None:
                continue
            required = (backbone_plan.estimated_per_gpu_bytes or 0) + (judge_plan.estimated_per_gpu_bytes or 0)
            allowed = int(device.free_bytes * shared_gpu_total_utilization)
            if required > allowed:
                failure_reason = (
                    f"Explicit shared GPU {device_id} would require {required / BYTES_PER_GIB:.2f} GiB, "
                    f"exceeding shared limit {allowed / BYTES_PER_GIB:.2f} GiB."
                )
                break
    return DualServerVramPlan(
        ok=failure_reason is None,
        placement_policy=placement_policy,
        placement_mode=placement_mode if failure_reason is None else "explicit_failed",
        devices=device_json,
        backbone_plan=backbone_plan,
        judge_plan=judge_plan,
        shared_gpu_total_utilization=shared_gpu_total_utilization,
        shared_gpu_min_server_utilization=shared_gpu_min_server_utilization,
        failure_reason=failure_reason,
    )


def estimate_required_bytes_for_model(
    model_id: str,
    *,
    dtype: str,
    load_in_4bit: bool,
) -> Tuple[Optional[int], Optional[float], Optional[str]]:
    estimated_bytes, parameter_count_billions, parameter_source = estimate_model_bytes(
        model_id,
        dtype=dtype,
        load_in_4bit=load_in_4bit,
    )
    if estimated_bytes is None:
        return None, parameter_count_billions, parameter_source
    return int(estimated_bytes * DEFAULT_MODEL_SAFETY_FACTOR), parameter_count_billions, parameter_source


def failed_dual_plan(
    *,
    placement_policy: str,
    devices: List[Dict[str, Any]],
    shared_gpu_total_utilization: float,
    shared_gpu_min_server_utilization: float,
    failure_reason: str,
    placement_mode: str = "failed",
    backbone_plan: Optional[ServerVramPlan] = None,
    judge_plan: Optional[ServerVramPlan] = None,
) -> DualServerVramPlan:
    return DualServerVramPlan(
        ok=False,
        placement_policy=placement_policy,
        placement_mode=placement_mode,
        devices=devices,
        backbone_plan=backbone_plan,
        judge_plan=judge_plan,
        shared_gpu_total_utilization=shared_gpu_total_utilization,
        shared_gpu_min_server_utilization=shared_gpu_min_server_utilization,
        failure_reason=failure_reason,
    )


def print_plan_summary(label: str, plan: ServerVramPlan) -> None:
    status = "PASS" if plan.ok else "FAIL"
    print(f"[{status}] {label}: model={plan.model_id} dtype={plan.dtype} load_in_4bit={plan.load_in_4bit}")
    if plan.parameter_count_billions is not None:
        print(f"  estimated_params={plan.parameter_count_billions:.2f}B source={plan.parameter_source}")
    if plan.estimated_required_bytes is not None:
        print(f"  estimated_required={plan.estimated_required_bytes / BYTES_PER_GIB:.2f} GiB")
    if plan.ok:
        print(f"  assigned_gpus={plan.assigned_gpu_ids} tensor_parallel_size={plan.tensor_parallel_size}")
        print(f"  estimated_per_gpu={plan.estimated_per_gpu_bytes / BYTES_PER_GIB:.2f} GiB")
    else:
        print(f"  reason={plan.failure_reason}")


def preflight_or_warn(mode: str, plans: Sequence[Tuple[str, ServerVramPlan]]) -> None:
    if mode == "off":
        return
    failed = [(label, plan) for label, plan in plans if not plan.ok]
    for label, plan in plans:
        print_plan_summary(label, plan)
    if failed and mode == "strict":
        lines = ["Server VRAM preflight failed:"]
        for label, plan in failed:
            lines.append(f"- {label}: {plan.failure_reason}")
        lines.append("Use a larger GPU set, higher tensor parallel size, smaller model, lower dtype, or explicit 4-bit quantization.")
        raise SystemExit("\n".join(lines))


def plan_pair_from_args(
    *,
    args: argparse.Namespace,
    backbone_model_id: str,
    judge_model_id: Optional[str] = None,
    judge_is_server: bool = False,
    same_judge_endpoint: bool = False,
) -> Tuple[Optional[ServerVramPlan], Optional[ServerVramPlan]]:
    if getattr(args, "server_preflight", "off") == "off":
        return None, None
    devices = capture_cuda_devices()
    backbone_hf_id = getattr(args, "server_hf_model_id", None) or backbone_model_id
    backbone_plan = plan_server_vram(
        model_id=backbone_hf_id,
        devices=devices,
        gpu_ids=getattr(args, "server_gpu_ids", None),
        tensor_parallel_size=getattr(args, "server_tensor_parallel_size", "auto"),
        dtype=getattr(args, "server_dtype", "bfloat16"),
        load_in_4bit=bool(getattr(args, "server_load_in_4bit", False)),
        gpu_memory_utilization=float(getattr(args, "server_gpu_memory_utilization", DEFAULT_GPU_MEMORY_UTILIZATION)),
        reserve_gb=float(getattr(args, "server_reserve_gb", DEFAULT_RESERVE_GB)),
    )
    if not judge_is_server or not judge_model_id:
        return backbone_plan, None
    judge_hf_id = getattr(args, "judge_server_hf_model_id", None) or judge_model_id
    if same_judge_endpoint and judge_hf_id == backbone_hf_id:
        return backbone_plan, None
    judge_devices = devices
    if not getattr(args, "judge_server_gpu_ids", None) and backbone_plan.ok:
        judge_devices = subtract_plan_from_devices(devices, backbone_plan)
    judge_plan = plan_server_vram(
        model_id=judge_hf_id,
        devices=judge_devices,
        gpu_ids=getattr(args, "judge_server_gpu_ids", None),
        tensor_parallel_size=getattr(args, "judge_server_tensor_parallel_size", None)
        or getattr(args, "server_tensor_parallel_size", "auto"),
        dtype=getattr(args, "judge_server_dtype", None) or getattr(args, "server_dtype", "bfloat16"),
        load_in_4bit=bool(getattr(args, "judge_server_load_in_4bit", False)),
        gpu_memory_utilization=float(
            getattr(args, "judge_server_gpu_memory_utilization", None)
            or getattr(args, "server_gpu_memory_utilization", DEFAULT_GPU_MEMORY_UTILIZATION)
        ),
        reserve_gb=float(getattr(args, "judge_server_reserve_gb", None) or getattr(args, "server_reserve_gb", DEFAULT_RESERVE_GB)),
    )
    return backbone_plan, judge_plan
