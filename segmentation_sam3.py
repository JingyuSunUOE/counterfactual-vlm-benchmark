#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import inspect
import json
import math
import os
import shutil
import sys
import time
import traceback
from collections import Counter
from contextlib import nullcontext
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from PIL import Image, ImageDraw, ImageFilter, ImageFont


REPO_ROOT = Path(__file__).resolve().parent
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
SAMPLE_SCHEMA_VERSION = "sam3_evidence_v1"
INSTANCE_SCHEMA_VERSION = "sam3_instances_v1"
PROMPT_CONFIG_SCHEMA_VERSION = "sam3_prompt_config_v1"
MANIFEST_SCHEMA_VERSION = "vision_manifest_v1"
RUN_SUMMARY_SCHEMA_VERSION = "sam3_run_summary_v1"
DEFAULT_EVIDENCE_TYPES = ("mask", "bbox", "crop", "zoom", "overlay")
EVIDENCE_TYPE_ALIASES = {
    "contour": "outline",
    "outline_overlay": "outline",
    "contour_overlay": "outline",
}
VALID_EVIDENCE_TYPES = set(DEFAULT_EVIDENCE_TYPES) | {"outline"} | set(EVIDENCE_TYPE_ALIASES)
BACKEND_ALIASES = {"sam3": "sam3_transformers"}


@dataclass(frozen=True)
class Sample:
    benchmark: str
    sample_id: str
    pair_id: Optional[str]
    image_path: Path
    image_role: str
    category: Optional[str]
    subcategory: str
    source_variant: str
    prompt_key: str
    prompt_mode: str
    target_region_key: Optional[str]
    target_part_key: Optional[str]
    prompt_profile: str
    prompt_modes_requested: Tuple[str, ...]
    is_multi_prompt_run: bool


@dataclass(frozen=True)
class InstancePrediction:
    mask: Image.Image
    box_xyxy_original: List[float]
    score: float
    source_instance_ids: Tuple[int, ...] = ()
    instance_selection_method: str = "single_instance"


@dataclass(frozen=True)
class PredictionResult:
    instances: List[InstancePrediction]
    original_size_wh: Tuple[int, int]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate SAM3-based visual evidence with a unified schema.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--benchmark", required=True, help="Benchmark name, for example if_exist.")
    parser.add_argument("--input-root", type=Path, default=None, help="Directory to scan when --manifest is absent.")
    parser.add_argument("--manifest", type=Path, default=None, help="JSONL manifest with explicit samples.")
    parser.add_argument("--output-root", type=Path, required=True, help="Root directory for generated evidence.")
    parser.add_argument("--prompt-config", type=Path, required=True, help="Prompt config JSON.")
    parser.add_argument("--image-role", choices=("original", "cf", "tool_input"), default="original")
    parser.add_argument("--prompt-mode", choices=("object", "region", "part"), default="object")
    parser.add_argument("--prompt-modes", default=None, help="Comma-separated prompt modes for multi-prompt runs, for example object,region.")
    parser.add_argument("--target-region-key", default=None)
    parser.add_argument("--target-part-key", default=None)
    parser.add_argument("--model-id", default="facebook/sam3")
    parser.add_argument("--device", choices=("auto", "cuda", "cpu", "mps"), default="auto")
    parser.add_argument("--dtype", choices=("auto", "float16", "bfloat16", "float32"), default="float32")
    parser.add_argument("--box-threshold", type=float, default=0.5)
    parser.add_argument("--mask-threshold", type=float, default=0.5)
    parser.add_argument("--min-score", type=float, default=None)
    parser.add_argument("--min-mask-area", type=int, default=0)
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument(
        "--evidence-types",
        default=",".join(DEFAULT_EVIDENCE_TYPES),
        help="Comma-separated evidence types: mask,bbox,crop,zoom,overlay,outline. Aliases: contour,outline_overlay,contour_overlay.",
    )
    parser.add_argument("--crop-padding", type=float, default=0.10)
    parser.add_argument("--no-text-labels", action="store_true", default=True, help="Formal overlays never include labels.")
    parser.add_argument("--debug-labels", action="store_true", help="Write debug/debug_overlay_with_labels.png.")
    parser.add_argument("--resume", action="store_true", help="Skip matching completed sample outputs.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing sample output directories.")
    parser.add_argument("--dry-run", action="store_true", help="Print planned work without loading a model or writing files.")
    parser.add_argument("--backend", choices=("sam3", "sam3_transformers", "sam3_repo", "mock"), default="sam3")
    parser.add_argument("--max-samples", type=int, default=None, help="Limit samples after scan/manifest resolution.")
    parser.add_argument("--max-instances", type=int, default=3, help="Maximum instances to save per sample; 0 keeps all instances.")
    parser.add_argument(
        "--merge-instances-over",
        type=int,
        default=0,
        help="Merge all detected instances into one union instance when detected count is greater than this value; 0 disables merging.",
    )
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--progress-style", choices=("auto", "bar", "lines", "off"), default="auto")
    parser.add_argument("--no-progress", action="store_true", help="Disable progress display.")
    parser.add_argument("--log-every", type=int, default=25, help="Emit progress at least every N samples in line mode.")
    parser.add_argument("--log-seconds", type=float, default=30.0, help="Emit progress at least every N seconds in line mode.")
    parser.add_argument("--events-log", type=Path, default=None, help="JSONL event log path. Defaults to <output-root>/run_events.jsonl.")
    return parser.parse_args()


def main() -> int:
    load_env_file()
    args = parse_args()
    validate_args(args)
    output_root = resolve_path(args.output_root)
    backend_resolved = resolve_backend(args.backend)
    events_log_path = resolve_path(args.events_log) if args.events_log else output_root / "run_events.jsonl"
    prompt_config_path = resolve_path(args.prompt_config)
    prompt_config = load_prompt_config(prompt_config_path, benchmark=args.benchmark)
    evidence_types = parse_evidence_types(args.evidence_types)
    prompt_modes = parse_prompt_modes(args)
    config_hash = build_config_hash(args, prompt_config, evidence_types)
    samples = load_samples(args=args, prompt_config=prompt_config, prompt_modes=prompt_modes)
    if args.max_samples is not None:
        samples = samples[: args.max_samples]
    ensure_unique_sample_ids(samples)
    validate_sample_prompts(samples, prompt_config)

    print(f"benchmark={args.benchmark}")
    print(f"backend={args.backend}")
    print(f"backend_resolved={backend_resolved}")
    print(f"samples={len(samples)}")
    print(f"output_root={output_root}")
    print(f"events_log={events_log_path}")
    print(f"evidence_types={','.join(evidence_types)}")
    print(f"prompt_modes={','.join(prompt_modes)}")
    print(f"max_instances={args.max_instances}")
    print(f"merge_instances_over={args.merge_instances_over}")
    for sample in samples[:20]:
        prompts = resolve_prompt_attempts(prompt_config, sample)
        prompt = prompts[0]
        fallback_preview = f" fallback_prompts={prompts[1:]}" if len(prompts) > 1 else ""
        print(
            f"sample_id={sample.sample_id} prompt_profile={sample.prompt_profile} "
            f"subcategory={sample.subcategory} prompt={prompt}{fallback_preview} "
            f"output={sample_output_dir(output_root, sample)}"
        )
    if len(samples) > 20:
        print(f"... {len(samples) - 20} additional samples not shown")
    if args.dry_run:
        print("dry_run=true; no files written")
        return 0

    preflight_backend(args, backend_resolved)
    output_root.mkdir(parents=True, exist_ok=True)
    event_logger = EventLogger(events_log_path)
    progress = ProgressReporter(
        total=len(samples),
        style="off" if args.no_progress else args.progress_style,
        log_every=args.log_every,
        log_seconds=args.log_seconds,
    )
    event_logger.write(
        "start",
        {
            "benchmark": args.benchmark,
            "backend": args.backend,
            "backend_resolved": backend_resolved,
            "model_id": args.model_id,
            "num_samples": len(samples),
            "output_root": str(output_root),
            "prompt_modes": list(prompt_modes),
            "evidence_types": list(evidence_types),
            "max_instances": args.max_instances,
            "merge_instances_over": args.merge_instances_over,
        },
    )
    runner = build_runner(args, backend_resolved=backend_resolved)
    results: List[Dict[str, Any]] = []
    started_at = time.time()
    for index, sample in enumerate(samples, start=1):
        prompt = resolve_prompt_text(prompt_config, sample)
        event_logger.write(
            "sample_start",
            {
                "index": index,
                "sample_id": sample.sample_id,
                "prompt_profile": sample.prompt_profile,
                "prompt_text": prompt,
                "output_dir": str(sample_output_dir(output_root, sample)),
            },
        )
        sample_started_at = time.time()
        progress.sample_start(index=index, sample=sample)
        try:
            result = process_sample(
                args=args,
                sample=sample,
                output_root=output_root,
                prompt_config=prompt_config,
                evidence_types=evidence_types,
                config_hash=config_hash,
                runner=runner,
            )
            latency_ms = int((time.time() - sample_started_at) * 1000)
            result["latency_ms"] = latency_ms
            results.append(result)
            event_logger.write(
                "sample_done",
                {
                    "index": index,
                    "sample_id": sample.sample_id,
                    "prompt_profile": sample.prompt_profile,
                    "prompt_text": prompt,
                    "status": result["status"],
                    "num_instances": result.get("num_instances", 0),
                    "num_instances_detected": result.get("num_instances_detected", result.get("num_instances", 0)),
                    "num_instances_saved": result.get("num_instances_saved", result.get("num_instances", 0)),
                    "truncated_instance_count": result.get("truncated_instance_count", 0),
                    "instances_merged": bool(result.get("instances_merged")),
                    "num_instances_before_merge": result.get("num_instances_before_merge"),
                    "num_instances_after_merge": result.get("num_instances_after_merge"),
                    "selected_prompt_text": result.get("selected_prompt_text") or result.get("prompt"),
                    "used_fallback_prompt": bool(result.get("used_fallback_prompt")),
                    "latency_ms": latency_ms,
                    "output_dir": result.get("output_dir"),
                },
            )
            progress.sample_done(index=index, sample=sample, result=result)
        except Exception as exc:
            failure = write_failed_sample(args=args, sample=sample, output_root=output_root, error=exc, config_hash=config_hash)
            latency_ms = int((time.time() - sample_started_at) * 1000)
            failure["latency_ms"] = latency_ms
            results.append(failure)
            event_logger.write(
                "sample_failed",
                {
                    "index": index,
                    "sample_id": sample.sample_id,
                    "prompt_profile": sample.prompt_profile,
                    "prompt_text": prompt,
                    "status": "failed",
                    "num_instances": 0,
                    "latency_ms": latency_ms,
                    "output_dir": str(sample_output_dir(output_root, sample)),
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                },
            )
            progress.sample_failed(index=index, sample=sample, error=exc)
            if args.fail_fast:
                raise

    progress.finish()
    summary = build_run_summary(
        args=args,
        samples=samples,
        results=results,
        output_root=output_root,
        prompt_config_path=prompt_config_path,
        config_hash=config_hash,
        backend_resolved=backend_resolved,
        events_log_path=events_log_path,
        elapsed_seconds=time.time() - started_at,
    )
    write_json(output_root / "run_summary.json", summary)
    event_logger.write(
        "finish",
        {
            "summary": str(output_root / "run_summary.json"),
            "num_success": summary["num_success"],
            "num_empty": summary["num_empty"],
            "num_failed": summary["num_failed"],
            "num_skipped_resume": summary["num_skipped_resume"],
            "num_samples_with_truncated_instances": summary["num_samples_with_truncated_instances"],
            "total_truncated_instances": summary["total_truncated_instances"],
            "elapsed_seconds": summary["elapsed_seconds"],
        },
    )
    print(f"summary={output_root / 'run_summary.json'}")
    return 0


def load_env_file() -> None:
    env_path = REPO_ROOT / ".env"
    if not env_path.exists():
        return
    try:
        from dotenv import load_dotenv
    except Exception:
        return
    load_dotenv(env_path, override=True)
    if not os.environ.get("HF_TOKEN"):
        for key in ("HUGGING_FACE_HUB_TOKEN", "HUGGINGFACE_TOKEN", "HUGGINGFACE_HUB_TOKEN"):
            value = os.environ.get(key)
            if value:
                os.environ["HF_TOKEN"] = value
                break


def validate_args(args: argparse.Namespace) -> None:
    if args.manifest is None and args.input_root is None:
        raise SystemExit("Either --input-root or --manifest is required.")
    if args.image_size <= 0:
        raise SystemExit("--image-size must be positive.")
    if args.crop_padding < 0:
        raise SystemExit("--crop-padding must be non-negative.")
    if args.max_samples is not None and args.max_samples <= 0:
        raise SystemExit("--max-samples must be positive.")
    if args.max_instances < 0:
        raise SystemExit("--max-instances must be non-negative.")
    if args.merge_instances_over < 0:
        raise SystemExit("--merge-instances-over must be non-negative.")
    if args.prompt_modes is not None and not args.prompt_modes.strip():
        raise SystemExit("--prompt-modes must not be empty when provided.")
    if args.log_every <= 0:
        raise SystemExit("--log-every must be positive.")
    if args.log_seconds < 0:
        raise SystemExit("--log-seconds must be non-negative.")


def resolve_backend(raw: str) -> str:
    return BACKEND_ALIASES.get(raw, raw)


def preflight_backend(args: argparse.Namespace, backend_resolved: str) -> None:
    if backend_resolved != "sam3_repo":
        return
    if importlib.util.find_spec("sam3") is None:
        raise SystemExit(
            "The sam3_repo backend requires the official Meta SAM3 package, but Python cannot import 'sam3'. "
            "Install the facebookresearch/sam3 repository in this environment before using "
            "--backend sam3_repo --model-id facebook/sam3.1."
        )
    if args.model_id == "facebook/sam3.1" and not (os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")):
        print(
            "warning=sam3.1 is a gated Hugging Face checkpoint; ensure access is approved and run `hf auth login` "
            "or set HF_TOKEN before loading.",
            file=sys.stderr,
        )


def resolve_torch_device(torch_module: Any, raw: str):
    if raw == "auto":
        if torch_module.cuda.is_available():
            return torch_module.device("cuda")
        if getattr(torch_module.backends, "mps", None) is not None and torch_module.backends.mps.is_available():
            return torch_module.device("mps")
        return torch_module.device("cpu")
    if raw == "cuda" and torch_module.cuda.is_available():
        return torch_module.device("cuda")
    if raw == "mps" and getattr(torch_module.backends, "mps", None) is not None and torch_module.backends.mps.is_available():
        return torch_module.device("mps")
    return torch_module.device("cpu")


def resolve_torch_dtype(torch_module: Any, raw: str):
    return {
        "auto": None,
        "float16": torch_module.float16,
        "bfloat16": torch_module.bfloat16,
        "float32": torch_module.float32,
    }[raw]


def resolve_path(path: Path) -> Path:
    return path.expanduser() if path.is_absolute() else (REPO_ROOT / path).resolve()


def load_prompt_config(path: Path, *, benchmark: str) -> Dict[str, Any]:
    if not path.exists():
        raise SystemExit(f"Prompt config not found: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("schema_version") != PROMPT_CONFIG_SCHEMA_VERSION:
        raise SystemExit(f"Prompt config schema_version must be {PROMPT_CONFIG_SCHEMA_VERSION}: {path}")
    if data.get("benchmark") != benchmark:
        raise SystemExit(f"Prompt config benchmark mismatch: expected {benchmark}, got {data.get('benchmark')}")
    if not isinstance(data.get("categories"), dict) or not data["categories"]:
        raise SystemExit("Prompt config must contain non-empty categories.")
    return data


def parse_evidence_types(raw: str) -> Tuple[str, ...]:
    values = tuple(value.strip() for value in raw.split(",") if value.strip())
    if not values:
        raise SystemExit("--evidence-types must not be empty.")
    invalid = sorted(set(values) - VALID_EVIDENCE_TYPES)
    if invalid:
        allowed = sorted((set(DEFAULT_EVIDENCE_TYPES) | {"outline"} | set(EVIDENCE_TYPE_ALIASES)))
        raise SystemExit(f"Unsupported evidence types: {invalid}. Allowed: {allowed}")
    normalized = tuple(EVIDENCE_TYPE_ALIASES.get(value, value) for value in values)
    duplicates = [value for value, count in Counter(normalized).items() if count > 1]
    if duplicates:
        raise SystemExit(f"Duplicate evidence types after alias normalization are not allowed: {duplicates}")
    return normalized


def parse_prompt_modes(args: argparse.Namespace) -> Tuple[str, ...]:
    raw = args.prompt_modes if args.prompt_modes is not None else args.prompt_mode
    modes = tuple(value.strip() for value in raw.split(",") if value.strip())
    if not modes:
        raise SystemExit("At least one prompt mode is required.")
    invalid = sorted(set(modes) - {"object", "region", "part"})
    if invalid:
        raise SystemExit(f"Unsupported prompt modes: {invalid}. Allowed: ['object', 'region', 'part']")
    duplicates = [mode for mode, count in Counter(modes).items() if count > 1]
    if duplicates:
        raise SystemExit(f"Duplicate prompt modes are not allowed: {duplicates}")
    return modes


def load_samples(*, args: argparse.Namespace, prompt_config: Dict[str, Any], prompt_modes: Tuple[str, ...]) -> List[Sample]:
    if args.manifest is not None:
        base_samples = load_manifest(resolve_path(args.manifest), benchmark=args.benchmark)
    else:
        assert args.input_root is not None
        base_samples = scan_input_root(
            input_root=resolve_path(args.input_root),
            benchmark=args.benchmark,
            image_role=args.image_role,
            prompt_mode=args.prompt_mode,
            target_region_key=args.target_region_key,
            target_part_key=args.target_part_key,
            prompt_config=prompt_config,
        )
    if args.prompt_modes is None:
        samples = [
            apply_prompt_defaults(
                sample,
                prompt_config=prompt_config,
                prompt_mode=sample.prompt_mode,
                target_region_key=sample.target_region_key,
                target_part_key=sample.target_part_key,
                prompt_modes_requested=prompt_modes,
                is_multi_prompt_run=False,
            )
            for sample in base_samples
        ]
        return mark_prompt_profile_layout(samples)
    expanded: List[Sample] = []
    for sample in base_samples:
        for prompt_mode in prompt_modes:
            expanded.append(
                apply_prompt_defaults(
                    sample,
                    prompt_config=prompt_config,
                    prompt_mode=prompt_mode,
                    target_region_key=args.target_region_key or sample.target_region_key,
                    target_part_key=args.target_part_key or sample.target_part_key,
                    prompt_modes_requested=prompt_modes,
                    is_multi_prompt_run=len(prompt_modes) > 1,
                )
            )
    return mark_prompt_profile_layout(expanded)


def load_manifest(path: Path, *, benchmark: str) -> List[Sample]:
    if not path.exists():
        raise SystemExit(f"Manifest not found: {path}")
    samples: List[Sample] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            raw = json.loads(line)
            if raw.get("schema_version") != MANIFEST_SCHEMA_VERSION:
                raise SystemExit(f"Invalid manifest schema at {path}:{line_number}")
            if raw.get("benchmark") != benchmark:
                raise SystemExit(f"Manifest benchmark mismatch at {path}:{line_number}")
            image_path = resolve_path(Path(str(raw["image_path"])))
            if not image_path.exists():
                raise SystemExit(f"Manifest image does not exist at {path}:{line_number}: {image_path}")
            samples.append(
                Sample(
                    benchmark=benchmark,
                    sample_id=sanitize_id(str(raw["sample_id"])),
                    pair_id=str(raw["pair_id"]) if raw.get("pair_id") else None,
                    image_path=image_path,
                    image_role=str(raw.get("image_role") or "tool_input"),
                    category=str(raw.get("category")) if raw.get("category") else None,
                    subcategory=str(raw["subcategory"]),
                    source_variant=str(raw.get("source_variant") or "real"),
                    prompt_key=str(raw.get("prompt_key") or raw["subcategory"]),
                    prompt_mode=str(raw.get("prompt_mode") or "object"),
                    target_region_key=str(raw["target_region_key"]) if raw.get("target_region_key") else None,
                    target_part_key=str(raw["target_part_key"]) if raw.get("target_part_key") else None,
                    prompt_profile="object",
                    prompt_modes_requested=("object",),
                    is_multi_prompt_run=False,
                )
            )
    return samples


def scan_input_root(
    *,
    input_root: Path,
    benchmark: str,
    image_role: str,
    prompt_mode: str,
    target_region_key: Optional[str],
    target_part_key: Optional[str],
    prompt_config: Dict[str, Any],
) -> List[Sample]:
    if not input_root.exists():
        raise SystemExit(f"Input root not found: {input_root}")
    paths = sorted(path for path in input_root.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS)
    samples: List[Sample] = []
    for path in paths:
        relative = path.relative_to(input_root)
        subcategory_raw = path.parent.name
        source_variant = "ai" if subcategory_raw.endswith("_ai") else "real"
        prompt_key = subcategory_raw[:-3] if subcategory_raw.endswith("_ai") else subcategory_raw
        if prompt_key not in prompt_config["categories"]:
            raise SystemExit(f"No prompt config for subcategory {prompt_key!r} from image {path}")
        sample_id = sanitize_id(relative.with_suffix("").as_posix())
        samples.append(
            Sample(
                benchmark=benchmark,
                sample_id=sample_id,
                pair_id=sample_id,
                image_path=path.resolve(),
                image_role=image_role,
                category=benchmark,
                subcategory=prompt_key,
                source_variant=source_variant,
                prompt_key=prompt_key,
                prompt_mode=prompt_mode,
                target_region_key=target_region_key,
                target_part_key=target_part_key,
                prompt_profile="object",
                prompt_modes_requested=(prompt_mode,),
                is_multi_prompt_run=False,
            )
        )
    return samples


def apply_prompt_defaults(
    sample: Sample,
    *,
    prompt_config: Dict[str, Any],
    prompt_mode: str,
    target_region_key: Optional[str],
    target_part_key: Optional[str],
    prompt_modes_requested: Tuple[str, ...],
    is_multi_prompt_run: bool,
) -> Sample:
    category = prompt_config["categories"].get(sample.prompt_key)
    if category is None:
        raise SystemExit(f"No prompt config for prompt_key={sample.prompt_key!r}")
    resolved_region_key = target_region_key
    resolved_part_key = target_part_key
    if prompt_mode == "region" and not resolved_region_key:
        resolved_region_key = category.get("default_region_key")
    if prompt_mode == "part" and not resolved_part_key:
        resolved_part_key = category.get("default_part_key")
    profile = prompt_profile_for(prompt_mode, resolved_region_key, resolved_part_key)
    return Sample(
        benchmark=sample.benchmark,
        sample_id=sample.sample_id,
        pair_id=sample.pair_id,
        image_path=sample.image_path,
        image_role=sample.image_role,
        category=sample.category,
        subcategory=sample.subcategory,
        source_variant=sample.source_variant,
        prompt_key=sample.prompt_key,
        prompt_mode=prompt_mode,
        target_region_key=resolved_region_key,
        target_part_key=resolved_part_key,
        prompt_profile=profile,
        prompt_modes_requested=prompt_modes_requested,
        is_multi_prompt_run=is_multi_prompt_run,
    )


def prompt_profile_for(prompt_mode: str, target_region_key: Optional[str], target_part_key: Optional[str]) -> str:
    if prompt_mode == "object":
        return "object"
    if prompt_mode == "region":
        if not target_region_key:
            raise SystemExit("region prompt mode requires --target-region-key or default_region_key in prompt config.")
        return f"region_{sanitize_id(target_region_key)}"
    if not target_part_key:
        raise SystemExit("part prompt mode requires --target-part-key or default_part_key in prompt config.")
    return f"part_{sanitize_id(target_part_key)}"


def mark_prompt_profile_layout(samples: Sequence[Sample]) -> List[Sample]:
    profiles_by_sample_id: Dict[str, set[str]] = {}
    for sample in samples:
        profiles_by_sample_id.setdefault(sample.sample_id, set()).add(sample.prompt_profile)
    needs_profile_layer = any(sample.is_multi_prompt_run for sample in samples) or any(
        len(profiles) > 1 for profiles in profiles_by_sample_id.values()
    )
    if not needs_profile_layer:
        return list(samples)
    return [replace(sample, is_multi_prompt_run=True) for sample in samples]


def sanitize_id(value: str) -> str:
    cleaned = []
    for char in value.replace("\\", "/"):
        cleaned.append(char if char.isalnum() or char in {"-", "_"} else "_")
    return "".join(cleaned).strip("_")


def ensure_unique_sample_ids(samples: Sequence[Sample]) -> None:
    counts = Counter((sample.prompt_profile, sample.sample_id) for sample in samples)
    duplicates = sorted(key for key, count in counts.items() if count > 1)
    if duplicates:
        raise SystemExit(f"Duplicate sample_id values within the same prompt profile: {duplicates[:10]}")


def validate_sample_prompts(samples: Sequence[Sample], prompt_config: Dict[str, Any]) -> None:
    for sample in samples:
        resolve_prompt_attempts(prompt_config, sample)


def resolve_prompt_text(prompt_config: Dict[str, Any], sample: Sample) -> str:
    return resolve_prompt_attempts(prompt_config, sample)[0]


def resolve_prompt_attempts(prompt_config: Dict[str, Any], sample: Sample) -> List[str]:
    category = prompt_config["categories"].get(sample.prompt_key)
    if category is None:
        raise SystemExit(f"No prompt config for prompt_key={sample.prompt_key!r}")
    if sample.prompt_mode == "object":
        prompt = category.get("object_prompt")
    elif sample.prompt_mode == "region":
        region_key = sample.target_region_key
        prompt = (category.get("region_prompts") or {}).get(region_key)
    else:
        prompt = (category.get("part_prompts") or {}).get(sample.target_part_key)
    if not prompt:
        raise SystemExit(
            f"Missing prompt for prompt_key={sample.prompt_key!r} "
            f"prompt_mode={sample.prompt_mode!r} target_region_key={sample.target_region_key!r} "
            f"target_part_key={sample.target_part_key!r}"
        )
    prompts = [str(prompt)]
    if sample.prompt_mode == "object":
        fallbacks = category.get("fallback_object_prompts") or []
        if not isinstance(fallbacks, list):
            raise SystemExit(f"fallback_object_prompts must be a list for prompt_key={sample.prompt_key!r}")
        for fallback in fallbacks:
            if not fallback:
                continue
            fallback_text = str(fallback)
            if fallback_text not in prompts:
                prompts.append(fallback_text)
    return prompts


def sample_output_dir(output_root: Path, sample: Sample) -> Path:
    if sample.is_multi_prompt_run:
        return output_root / sample.prompt_profile / sample.image_role / sample.subcategory / sample.sample_id
    return output_root / sample.image_role / sample.subcategory / sample.sample_id


def build_runner(args: argparse.Namespace, *, backend_resolved: str):
    if backend_resolved == "mock":
        return MockRunner()
    if backend_resolved == "sam3_repo":
        return Sam31RepoRunner(args)
    return Sam3TransformersRunner(args)


class MockRunner:
    def predict(self, image: Image.Image, prompt: str) -> PredictionResult:
        width, height = image.size
        if "empty" in prompt.lower():
            return PredictionResult(instances=[], original_size_wh=(width, height))
        x1 = width * 0.20
        y1 = height * 0.18
        x2 = width * 0.78
        y2 = height * 0.82
        mask = Image.new("L", (width, height), 0)
        draw = ImageDraw.Draw(mask)
        draw.rectangle([x1, y1, x2, y2], fill=255)
        if "multi5" in prompt.lower() or "multi6" in prompt.lower():
            specs = [
                (0.08, 0.08, 0.28, 0.30, 0.50),
                (0.12, 0.42, 0.48, 0.82, 0.91),
                (0.55, 0.10, 0.90, 0.45, 0.76),
                (0.62, 0.55, 0.86, 0.76, 0.88),
                (0.35, 0.22, 0.54, 0.40, 0.99),
            ]
            if "multi6" in prompt.lower():
                specs.append((0.02, 0.62, 0.22, 0.94, 0.69))
            instances = []
            for x1r, y1r, x2r, y2r, score in specs:
                candidate = Image.new("L", (width, height), 0)
                draw_candidate = ImageDraw.Draw(candidate)
                box = [width * x1r, height * y1r, width * x2r, height * y2r]
                draw_candidate.rectangle(box, fill=255)
                instances.append(InstancePrediction(mask=candidate, box_xyxy_original=box, score=score))
            return PredictionResult(instances=instances, original_size_wh=(width, height))
        if "multi" in prompt.lower():
            x1b = width * 0.55
            y1b = height * 0.10
            x2b = width * 0.90
            y2b = height * 0.45
            mask_b = Image.new("L", (width, height), 0)
            draw_b = ImageDraw.Draw(mask_b)
            draw_b.rectangle([x1b, y1b, x2b, y2b], fill=255)
            return PredictionResult(
                instances=[
                    InstancePrediction(mask=mask, box_xyxy_original=[x1, y1, x2, y2], score=0.9876),
                    InstancePrediction(mask=mask_b, box_xyxy_original=[x1b, y1b, x2b, y2b], score=0.7654),
                ],
                original_size_wh=(width, height),
            )
        return PredictionResult(
            instances=[InstancePrediction(mask=mask, box_xyxy_original=[x1, y1, x2, y2], score=0.9876)],
            original_size_wh=(width, height),
        )


class Sam3TransformersRunner:
    def __init__(self, args: argparse.Namespace):
        import torch
        from transformers import Sam3Model, Sam3Processor

        self.torch = torch
        self.device = resolve_torch_device(torch, args.device)
        dtype = resolve_torch_dtype(torch, args.dtype)
        model_kwargs: Dict[str, Any] = {}
        if dtype is not None:
            model_kwargs["torch_dtype"] = dtype
        print(f"loading_model={args.model_id} device={self.device} dtype={dtype if dtype is not None else 'auto'}")
        self.model = Sam3Model.from_pretrained(args.model_id, **model_kwargs).to(self.device)
        self.model.eval()
        self.processor = Sam3Processor.from_pretrained(args.model_id)
        self.box_threshold = args.box_threshold
        self.mask_threshold = args.mask_threshold
        self.min_score = args.min_score
        self.min_mask_area = args.min_mask_area

    def predict(self, image: Image.Image, prompt: str) -> PredictionResult:
        import numpy as np
        import torch

        rgb = image.convert("RGB")
        inputs_cpu = self.processor(images=rgb, text=prompt, return_tensors="pt")
        target_sizes = inputs_cpu["original_sizes"].tolist()
        inputs = {
            key: value.to(self.device) if isinstance(value, torch.Tensor) else value
            for key, value in inputs_cpu.items()
        }
        with torch.inference_mode():
            outputs = self.model(**inputs)
        post = self.processor.post_process_instance_segmentation(
            outputs,
            threshold=self.box_threshold,
            mask_threshold=self.mask_threshold,
            target_sizes=target_sizes,
        )[0]
        masks = tensor_or_array_to_masks(post.get("masks", []))
        boxes = tensor_or_array_to_boxes(post.get("boxes", []))
        scores = tensor_or_array_to_scores(post.get("scores", []))
        count = min(len(masks), len(boxes), len(scores))
        instances: List[InstancePrediction] = []
        for mask_arr, box, score in zip(masks[:count], boxes[:count], scores[:count]):
            if self.min_score is not None and score < self.min_score:
                continue
            mask_image = Image.fromarray((mask_arr > 0).astype("uint8") * 255, mode="L")
            if self.min_mask_area > 0 and positive_pixel_count(mask_image) < self.min_mask_area:
                continue
            instances.append(InstancePrediction(mask=mask_image, box_xyxy_original=box, score=score))
        height, width = target_sizes[0]
        return PredictionResult(instances=instances, original_size_wh=(int(width), int(height)))


class Sam31RepoRunner:
    def __init__(self, args: argparse.Namespace):
        import torch
        from sam3.model.sam3_image_processor import Sam3Processor
        from sam3.model_builder import build_sam3_image_model

        self.torch = torch
        self.device = resolve_torch_device(torch, args.device)
        dtype = resolve_torch_dtype(torch, args.dtype)
        print(f"loading_model={args.model_id} backend=sam3_repo device={self.device} dtype={dtype if dtype is not None else 'auto'}")
        self.model = self._build_model(build_sam3_image_model, args)
        self.model.to(self.device)
        if dtype is not None:
            self.model.to(dtype=dtype)
        self.model.eval()
        self.processor = Sam3Processor(self.model, confidence_threshold=args.box_threshold)
        self.min_score = args.min_score
        self.min_mask_area = args.min_mask_area
        self.autocast_dtype = torch.bfloat16 if self.device.type == "cuda" else None

    def _build_model(self, build_fn: Any, args: argparse.Namespace) -> Any:
        signature = inspect.signature(build_fn)
        params = signature.parameters
        kwargs: Dict[str, Any] = {}
        if "bpe_path" in params:
            bpe_path = find_sam3_bpe_path(build_fn)
            if bpe_path is not None:
                kwargs["bpe_path"] = str(bpe_path)
        if "model_id" in params:
            kwargs["model_id"] = args.model_id
        elif args.model_id == "facebook/sam3.1" and "checkpoint_path" in params:
            checkpoint_path = download_sam3_checkpoint("sam3.1")
            if checkpoint_path is not None:
                kwargs["checkpoint_path"] = checkpoint_path
                if "load_from_HF" in params:
                    kwargs["load_from_HF"] = False
        elif args.model_id not in {"facebook/sam3", "facebook/sam3.1", "auto", "default"}:
            model_path = Path(args.model_id)
            if "checkpoint_path" in params and model_path.exists():
                kwargs["checkpoint_path"] = str(model_path)
            else:
                print(
                    f"warning=sam3_repo build_sam3_image_model does not accept model_id; "
                    f"ignoring --model-id {args.model_id!r} and using the repo default checkpoint.",
                    file=sys.stderr,
                )
        if "device" in params:
            kwargs["device"] = str(self.device)
        if "eval_mode" in params:
            kwargs["eval_mode"] = True
        if "load_from_HF" in params and "checkpoint_path" not in kwargs:
            kwargs["load_from_HF"] = True
        return build_fn(**kwargs)

    def predict(self, image: Image.Image, prompt: str) -> PredictionResult:
        import numpy as np

        rgb = image.convert("RGB")
        with self._autocast_context():
            state = self.processor.set_image(rgb)
            output = self.processor.set_text_prompt(state=state, prompt=prompt)
        masks = tensor_or_array_to_masks(output.get("masks", []))
        boxes = tensor_or_array_to_boxes(output.get("boxes", []))
        scores = tensor_or_array_to_scores(output.get("scores", []))
        count = min(len(masks), len(boxes), len(scores))
        instances: List[InstancePrediction] = []
        for mask_arr, box, score in zip(masks[:count], boxes[:count], scores[:count]):
            if self.min_score is not None and score < self.min_score:
                continue
            mask_image = mask_array_to_image(mask_arr)
            if self.min_mask_area > 0 and positive_pixel_count(mask_image) < self.min_mask_area:
                continue
            instances.append(InstancePrediction(mask=mask_image, box_xyxy_original=box, score=score))
        return PredictionResult(instances=instances, original_size_wh=rgb.size)

    def _autocast_context(self):
        if self.autocast_dtype is None:
            return nullcontext()
        return self.torch.autocast(device_type=self.device.type, dtype=self.autocast_dtype)


def find_sam3_bpe_path(build_fn: Any) -> Optional[Path]:
    candidate_names = (
        "bpe_simple_vocab_16e6.txt.gz",
        "bpe_simple_vocab_16e6.txt",
    )
    roots: List[Path] = []
    try:
        roots.append(Path(inspect.getfile(build_fn)).resolve().parent)
    except TypeError:
        pass
    try:
        import sam3

        module_file = getattr(sam3, "__file__", None)
        if module_file:
            roots.append(Path(module_file).resolve().parent)
    except Exception:
        pass
    for root in roots:
        for base in (root, root.parent):
            for name in candidate_names:
                candidate = base / "assets" / name
                if candidate.exists():
                    return candidate
    return None


def download_sam3_checkpoint(version: str) -> Optional[str]:
    try:
        from sam3.model_builder import download_ckpt_from_hf
    except Exception:
        return None
    try:
        return str(download_ckpt_from_hf(version=version))
    except TypeError:
        return None


def tensor_or_array_to_masks(value: Any) -> List[Any]:
    import numpy as np
    import torch

    if isinstance(value, torch.Tensor):
        value = value.detach().cpu()
        if value.dtype == torch.bfloat16:
            value = value.float()
        value = value.numpy()
    elif isinstance(value, list):
        value = [item.detach().cpu().numpy() if isinstance(item, torch.Tensor) else np.asarray(item) for item in value]
    else:
        value = np.asarray(value)
    if isinstance(value, list):
        return [normalize_mask_array(item) for item in value]
    if value.size == 0:
        return []
    if value.ndim == 2:
        return [normalize_mask_array(value)]
    return [normalize_mask_array(value[index]) for index in range(value.shape[0])]


def normalize_mask_array(value: Any) -> Any:
    import numpy as np

    arr = np.asarray(value)
    arr = np.squeeze(arr)
    if arr.ndim == 0:
        return arr.reshape(1, 1)
    while arr.ndim > 2:
        arr = arr[0]
        arr = np.squeeze(arr)
    return arr


def mask_array_to_image(value: Any) -> Image.Image:
    import numpy as np

    arr = normalize_mask_array(value)
    if arr.ndim != 2:
        raise ValueError(f"Expected a 2D mask after squeeze, got shape {arr.shape}")
    return Image.fromarray((arr > 0).astype("uint8") * 255, mode="L")


def tensor_or_array_to_boxes(value: Any) -> List[List[float]]:
    import numpy as np
    import torch

    if isinstance(value, torch.Tensor):
        value = value.detach().cpu()
        if value.dtype == torch.bfloat16:
            value = value.float()
        value = value.numpy()
    arr = np.asarray(value, dtype="float32")
    if arr.size == 0:
        return []
    if arr.ndim == 1:
        arr = arr[None, :]
    return [[float(item) for item in row.tolist()] for row in arr]


def tensor_or_array_to_scores(value: Any) -> List[float]:
    import numpy as np
    import torch

    if isinstance(value, torch.Tensor):
        value = value.detach().cpu()
        if value.dtype == torch.bfloat16:
            value = value.float()
        value = value.numpy()
    arr = np.asarray(value, dtype="float32")
    if arr.size == 0:
        return []
    return [float(item) for item in arr.reshape(-1).tolist()]


def process_sample(
    *,
    args: argparse.Namespace,
    sample: Sample,
    output_root: Path,
    prompt_config: Dict[str, Any],
    evidence_types: Tuple[str, ...],
    config_hash: str,
    runner: Any,
) -> Dict[str, Any]:
    prompt_attempt_texts = resolve_prompt_attempts(prompt_config, sample)
    primary_prompt = prompt_attempt_texts[0]
    output_dir = sample_output_dir(output_root, sample)
    image_hash = sha256_file(sample.image_path)
    if args.resume and not args.overwrite and can_resume(
        output_dir,
        args=args,
        sample=sample,
        prompt_attempt_texts=prompt_attempt_texts,
        image_hash=image_hash,
        evidence_types=evidence_types,
        config_hash=config_hash,
    ):
        return {
            "status": "skipped_resume",
            "sample_id": sample.sample_id,
            "prompt_profile": sample.prompt_profile,
            **existing_instance_stats(output_dir),
        }
    if output_dir.exists():
        if args.overwrite:
            shutil.rmtree(output_dir)
        elif args.resume:
            if existing_status(output_dir) == "failed":
                shutil.rmtree(output_dir)
            else:
                raise RuntimeError(
                    f"Existing output does not match resume criteria; use --overwrite to regenerate: {output_dir}"
                )
        else:
            raise RuntimeError(f"Output directory already exists. Use --overwrite or --resume: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    source = Image.open(sample.image_path).convert("RGB")
    source_saved = resize_image(source, args.image_size)
    source_saved.save(output_dir / "source.png")
    prediction, selected_prompt, prompt_attempts, fallback_prompt_index = predict_with_fallbacks(
        runner=runner,
        image=source,
        prompt_attempt_texts=prompt_attempt_texts,
    )
    num_instances_detected = len(prediction.instances)
    prediction, merge_info = merge_prediction_instances(
        prediction,
        merge_instances_over=args.merge_instances_over,
    )
    prediction, truncation = limit_prediction_instances(prediction, max_instances=args.max_instances)
    status = "success" if prediction.instances else "empty"
    used_fallback_prompt = status == "success" and fallback_prompt_index is not None
    files = create_evidence_files(
        image=source,
        prediction=prediction,
        output_dir=output_dir,
        image_size=args.image_size,
        evidence_types=evidence_types,
        crop_padding=args.crop_padding,
        debug_labels=args.debug_labels,
        prompt=selected_prompt,
    )
    metadata = build_metadata(
        args=args,
        sample=sample,
        prompt=selected_prompt,
        primary_prompt=primary_prompt,
        prompt_attempts=prompt_attempts,
        used_fallback_prompt=used_fallback_prompt,
        fallback_prompt_index=fallback_prompt_index,
        status=status,
        image_hash=image_hash,
        prediction=prediction,
        num_instances_detected=num_instances_detected,
        merge_info=merge_info,
        truncation=truncation,
        files=files,
        evidence_types=evidence_types,
        config_hash=config_hash,
    )
    write_json(output_dir / "metadata.json", metadata)
    write_json(output_dir / "instances.json", build_instances_json(prediction, files, image_size=args.image_size))
    return {
        "status": status,
        "sample_id": sample.sample_id,
        "prompt_profile": sample.prompt_profile,
        "prompt": selected_prompt,
        "selected_prompt_text": selected_prompt,
        "used_fallback_prompt": used_fallback_prompt,
        "fallback_prompt_index": fallback_prompt_index,
        "prompt_attempts": prompt_attempts,
        "num_instances": len(prediction.instances),
        "num_instances_detected": num_instances_detected,
        "num_instances_saved": len(prediction.instances),
        "max_instances": args.max_instances,
        **merge_info,
        "truncated_instance_count": truncation["truncated_instance_count"],
        "instance_truncation_method": truncation["instance_truncation_method"],
        "scores": [instance.score for instance in prediction.instances],
        "output_dir": str(output_dir),
    }


def predict_with_fallbacks(
    *,
    runner: Any,
    image: Image.Image,
    prompt_attempt_texts: Sequence[str],
) -> Tuple[PredictionResult, str, List[Dict[str, Any]], Optional[int]]:
    if not prompt_attempt_texts:
        raise RuntimeError("No prompt attempts were resolved for this sample.")
    last_prediction: Optional[PredictionResult] = None
    attempt_records: List[Dict[str, Any]] = []
    selected_prompt = str(prompt_attempt_texts[-1])
    fallback_prompt_index: Optional[int] = None
    for attempt_index, prompt in enumerate(prompt_attempt_texts):
        prediction = runner.predict(image, prompt)
        last_prediction = prediction
        status = "success" if prediction.instances else "empty"
        attempt_records.append(
            {
                "attempt_index": attempt_index,
                "prompt": prompt,
                "status": status,
                "num_instances": len(prediction.instances),
                "scores": [instance.score for instance in prediction.instances],
            }
        )
        selected_prompt = prompt
        if prediction.instances:
            fallback_prompt_index = attempt_index - 1 if attempt_index > 0 else None
            return prediction, selected_prompt, attempt_records, fallback_prompt_index
    assert last_prediction is not None
    return last_prediction, selected_prompt, attempt_records, None


def merge_prediction_instances(
    prediction: PredictionResult,
    *,
    merge_instances_over: int,
) -> Tuple[PredictionResult, Dict[str, Any]]:
    detected = len(prediction.instances)
    if merge_instances_over <= 0 or detected <= merge_instances_over:
        return prediction, {
            "merge_instances_over": merge_instances_over,
            "instances_merged": False,
            "num_instances_before_merge": detected,
            "num_instances_after_merge": detected,
            "merged_source_instance_ids": [],
            "instance_selection_method": "none",
        }
    merged_mask = union_instance_masks(prediction.instances)
    x1 = min(float(instance.box_xyxy_original[0]) for instance in prediction.instances)
    y1 = min(float(instance.box_xyxy_original[1]) for instance in prediction.instances)
    x2 = max(float(instance.box_xyxy_original[2]) for instance in prediction.instances)
    y2 = max(float(instance.box_xyxy_original[3]) for instance in prediction.instances)
    source_ids = tuple(range(detected))
    merged = InstancePrediction(
        mask=merged_mask,
        box_xyxy_original=[x1, y1, x2, y2],
        score=max(float(instance.score) for instance in prediction.instances),
        source_instance_ids=source_ids,
        instance_selection_method="merged_instances_over_threshold",
    )
    merged_prediction = PredictionResult(instances=[merged], original_size_wh=prediction.original_size_wh)
    return merged_prediction, {
        "merge_instances_over": merge_instances_over,
        "instances_merged": True,
        "num_instances_before_merge": detected,
        "num_instances_after_merge": 1,
        "merged_source_instance_ids": list(source_ids),
        "instance_selection_method": "merged_instances_over_threshold",
    }


def union_instance_masks(instances: Sequence[InstancePrediction]) -> Image.Image:
    if not instances:
        return Image.new("L", (1, 1), 0)
    size = instances[0].mask.size
    combined = Image.new("L", size, 0)
    white = Image.new("L", size, 255)
    for instance in instances:
        mask = instance.mask.convert("L")
        if mask.size != size:
            mask = mask.resize(size, Image.Resampling.NEAREST)
        binary = mask.point(lambda value: 255 if value > 0 else 0)
        combined = Image.composite(white, combined, binary)
    return combined


def limit_prediction_instances(prediction: PredictionResult, *, max_instances: int) -> Tuple[PredictionResult, Dict[str, Any]]:
    detected = len(prediction.instances)
    method = "none"
    truncated = 0
    if max_instances > 0 and detected > max_instances:
        ranked = sorted(enumerate(prediction.instances), key=lambda item: instance_sort_key(item[1]), reverse=True)
        instances = [
            with_instance_audit(instance, source_index=source_index, method="mask_area_desc_box_area_desc_score_desc")
            for source_index, instance in ranked[:max_instances]
        ]
        method = "mask_area_desc_box_area_desc_score_desc"
        truncated = detected - len(instances)
    else:
        instances = [
            with_instance_audit(instance, source_index=index, method=instance.instance_selection_method)
            for index, instance in enumerate(prediction.instances)
        ]
    limited = PredictionResult(instances=list(instances), original_size_wh=prediction.original_size_wh)
    return limited, {
        "num_instances_detected": detected,
        "num_instances_saved": len(limited.instances),
        "max_instances": max_instances,
        "truncated_instance_count": truncated,
        "instance_truncation_method": method,
    }


def with_instance_audit(instance: InstancePrediction, *, source_index: int, method: str) -> InstancePrediction:
    source_ids = instance.source_instance_ids or (source_index,)
    selection_method = instance.instance_selection_method
    if selection_method == "single_instance":
        selection_method = method
    return replace(
        instance,
        source_instance_ids=tuple(source_ids),
        instance_selection_method=selection_method,
    )


def instance_sort_key(instance: InstancePrediction) -> Tuple[int, float, float]:
    x1, y1, x2, y2 = instance.box_xyxy_original
    box_area = max(0.0, float(x2) - float(x1)) * max(0.0, float(y2) - float(y1))
    return positive_pixel_count(instance.mask), box_area, float(instance.score)


def can_resume(
    output_dir: Path,
    *,
    args: argparse.Namespace,
    sample: Sample,
    prompt_attempt_texts: Sequence[str],
    image_hash: str,
    evidence_types: Tuple[str, ...],
    config_hash: str,
) -> bool:
    metadata_path = output_dir / "metadata.json"
    if not metadata_path.exists():
        return False
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return False
    recorded_attempts = metadata.get("prompt_attempts")
    if recorded_attempts:
        recorded_prompts = [str(item.get("prompt")) for item in recorded_attempts]
        expected_prompts = list(prompt_attempt_texts)
        if metadata.get("status") == "empty":
            prompts_match = recorded_prompts == expected_prompts
        else:
            prompts_match = expected_prompts[: len(recorded_prompts)] == recorded_prompts
    else:
        prompts_match = metadata.get("prompt_text") in set(prompt_attempt_texts)
    return all(
        (
            metadata.get("schema_version") == SAMPLE_SCHEMA_VERSION,
            metadata.get("status") in {"success", "empty"},
            metadata.get("source_image_sha256") == image_hash,
            metadata.get("model_id") == args.model_id,
            prompts_match,
            tuple(metadata.get("evidence_types") or []) == evidence_types,
            metadata.get("output_size_wh") == [args.image_size, args.image_size],
            metadata.get("config_hash") == config_hash,
            metadata.get("sample_id") == sample.sample_id,
            metadata.get("prompt_profile", sample.prompt_profile) == sample.prompt_profile,
        )
    )


def existing_status(output_dir: Path) -> Optional[str]:
    metadata_path = output_dir / "metadata.json"
    if not metadata_path.exists():
        return None
    try:
        return json.loads(metadata_path.read_text(encoding="utf-8")).get("status")
    except json.JSONDecodeError:
        return None


def existing_num_instances(output_dir: Path) -> int:
    metadata_path = output_dir / "metadata.json"
    if not metadata_path.exists():
        return 0
    try:
        return int(json.loads(metadata_path.read_text(encoding="utf-8")).get("num_instances") or 0)
    except Exception:
        return 0


def existing_instance_stats(output_dir: Path) -> Dict[str, Any]:
    metadata_path = output_dir / "metadata.json"
    if not metadata_path.exists():
        return {
            "num_instances": 0,
            "num_instances_detected": 0,
            "num_instances_saved": 0,
            "max_instances": None,
            "merge_instances_over": 0,
            "instances_merged": False,
            "num_instances_before_merge": 0,
            "num_instances_after_merge": 0,
            "truncated_instance_count": 0,
            "instance_truncation_method": "unknown",
        }
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except Exception:
        return {
            "num_instances": 0,
            "num_instances_detected": 0,
            "num_instances_saved": 0,
            "max_instances": None,
            "merge_instances_over": 0,
            "instances_merged": False,
            "num_instances_before_merge": 0,
            "num_instances_after_merge": 0,
            "truncated_instance_count": 0,
            "instance_truncation_method": "unknown",
        }
    num_instances = int(metadata.get("num_instances") or 0)
    num_saved = int(metadata.get("num_instances_saved") or num_instances)
    return {
        "num_instances": num_saved,
        "num_instances_detected": int(metadata.get("num_instances_detected") or num_saved),
        "num_instances_saved": num_saved,
        "max_instances": metadata.get("max_instances"),
        "merge_instances_over": metadata.get("merge_instances_over", 0),
        "instances_merged": bool(metadata.get("instances_merged")),
        "num_instances_before_merge": int(metadata.get("num_instances_before_merge") or num_saved),
        "num_instances_after_merge": int(metadata.get("num_instances_after_merge") or num_saved),
        "merged_source_instance_ids": metadata.get("merged_source_instance_ids") or [],
        "truncated_instance_count": int(metadata.get("truncated_instance_count") or 0),
        "instance_truncation_method": metadata.get("instance_truncation_method") or "none",
    }


def create_evidence_files(
    *,
    image: Image.Image,
    prediction: PredictionResult,
    output_dir: Path,
    image_size: int,
    evidence_types: Tuple[str, ...],
    crop_padding: float,
    debug_labels: bool,
    prompt: str,
) -> Dict[str, Any]:
    files: Dict[str, Any] = {
        "source": "source.png",
        "bbox_overlay": None,
        "bbox_overlays": [],
        "outline_overlay": None,
        "outline_overlays": [],
        "mask_overlay": None,
        "mask_bbox_overlay": None,
        "mask_combined": None,
        "crops": [],
        "zooms": [],
        "zoom_panels": [],
        "masks": [],
        "debug_overlay_with_labels": None,
    }
    resized = resize_image(image, image_size)
    scaled_instances = scale_instances(prediction, image_size=image_size)
    if "mask" in evidence_types:
        files["masks"] = save_masks(scaled_instances, output_dir / "masks", image_size=image_size)
        files["mask_combined"] = save_combined_mask(scaled_instances, output_dir / "masks", image_size=image_size)
    if "crop" in evidence_types:
        files["crops"] = save_crops(resized, scaled_instances, output_dir / "crops", crop_padding=crop_padding)
    if "zoom" in evidence_types:
        zooms, panels = save_zooms(resized, scaled_instances, output_dir / "zooms", image_size=image_size, crop_padding=crop_padding)
        files["zooms"] = zooms
        files["zoom_panels"] = panels
    if "bbox" in evidence_types:
        files["bbox_overlays"] = save_bbox_overlays(resized, scaled_instances, output_dir / "overlays")
        files["bbox_overlay"] = item_or_none(files["bbox_overlays"], 0)
    if "outline" in evidence_types:
        files["outline_overlays"] = save_outline_overlays(resized, scaled_instances, output_dir / "overlays")
        files["outline_overlay"] = item_or_none(files["outline_overlays"], 0)
    if "overlay" in evidence_types:
        files["mask_overlay"] = save_overlay(resized, scaled_instances, output_dir / "overlays" / "mask_overlay.png", draw_boxes=False, draw_masks=True)
        files["mask_bbox_overlay"] = save_overlay(resized, scaled_instances, output_dir / "overlays" / "mask_bbox_overlay.png", draw_boxes=True, draw_masks=True)
    if debug_labels:
        files["debug_overlay_with_labels"] = save_overlay(
            resized,
            scaled_instances,
            output_dir / "debug" / "debug_overlay_with_labels.png",
            draw_boxes=True,
            draw_masks=True,
            labels=[f"id:{index} | {prompt} | {instance.score:.3f}" for index, instance in enumerate(scaled_instances)],
        )
    return files


def resize_image(image: Image.Image, image_size: int) -> Image.Image:
    return image.convert("RGB").resize((image_size, image_size), Image.Resampling.BILINEAR)


def scale_instances(prediction: PredictionResult, *, image_size: int) -> List[InstancePrediction]:
    width, height = prediction.original_size_wh
    sx = image_size / max(width, 1)
    sy = image_size / max(height, 1)
    scaled: List[InstancePrediction] = []
    for instance in prediction.instances:
        x1, y1, x2, y2 = instance.box_xyxy_original
        mask = instance.mask.resize((image_size, image_size), Image.Resampling.NEAREST)
        scaled.append(
            InstancePrediction(
                mask=mask,
                box_xyxy_original=[float(x1), float(y1), float(x2), float(y2)],
                score=instance.score,
                source_instance_ids=instance.source_instance_ids,
                instance_selection_method=instance.instance_selection_method,
            )
        )
        object.__setattr__(scaled[-1], "_box_xyxy_saved", [float(x1 * sx), float(y1 * sy), float(x2 * sx), float(y2 * sy)])
    return scaled


def box_saved(instance: InstancePrediction) -> List[float]:
    return list(getattr(instance, "_box_xyxy_saved", instance.box_xyxy_original))


def save_masks(instances: Sequence[InstancePrediction], directory: Path, *, image_size: int) -> List[str]:
    directory.mkdir(parents=True, exist_ok=True)
    paths: List[str] = []
    for index, instance in enumerate(instances):
        path = directory / f"mask_{index:03d}.png"
        binary_mask(instance.mask, image_size=image_size).save(path)
        paths.append(f"masks/{path.name}")
    return paths


def save_combined_mask(instances: Sequence[InstancePrediction], directory: Path, *, image_size: int) -> Optional[str]:
    if not instances:
        return None
    directory.mkdir(parents=True, exist_ok=True)
    combined = Image.new("L", (image_size, image_size), 0)
    for instance in instances:
        combined = Image.composite(Image.new("L", combined.size, 255), combined, binary_mask(instance.mask, image_size=image_size))
    path = directory / "mask_combined.png"
    combined.save(path)
    return f"masks/{path.name}"


def binary_mask(mask: Image.Image, *, image_size: int) -> Image.Image:
    return mask.convert("L").resize((image_size, image_size), Image.Resampling.NEAREST).point(lambda value: 255 if value > 0 else 0)


def save_crops(image: Image.Image, instances: Sequence[InstancePrediction], directory: Path, *, crop_padding: float) -> List[str]:
    directory.mkdir(parents=True, exist_ok=True)
    paths: List[str] = []
    for index, instance in enumerate(instances):
        crop = image.crop(padded_box(box_saved(instance), image.size, crop_padding))
        path = directory / f"crop_{index:03d}.png"
        crop.save(path)
        paths.append(f"crops/{path.name}")
    return paths


def save_zooms(
    image: Image.Image,
    instances: Sequence[InstancePrediction],
    directory: Path,
    *,
    image_size: int,
    crop_padding: float,
) -> Tuple[List[str], List[str]]:
    directory.mkdir(parents=True, exist_ok=True)
    zooms: List[str] = []
    panels: List[str] = []
    for index, instance in enumerate(instances):
        crop = image.crop(padded_box(box_saved(instance), image.size, crop_padding))
        zoom = crop.resize((image_size, image_size), Image.Resampling.BILINEAR)
        zoom_path = directory / f"zoom_{index:03d}.png"
        zoom.save(zoom_path)
        zooms.append(f"zooms/{zoom_path.name}")

        panel = image.copy()
        inset_size = max(96, image_size // 3)
        inset = zoom.resize((inset_size, inset_size), Image.Resampling.BILINEAR)
        panel.paste(inset, (image_size - inset_size - 8, 8))
        draw = ImageDraw.Draw(panel)
        draw.rectangle([image_size - inset_size - 8, 8, image_size - 8, inset_size + 8], outline=(255, 180, 0), width=3)
        draw.rectangle(box_saved(instance), outline=(255, 180, 0), width=3)
        panel_path = directory / f"zoom_panel_{index:03d}.png"
        panel.save(panel_path)
        panels.append(f"zooms/{panel_path.name}")
    return zooms, panels


def save_bbox_overlays(image: Image.Image, instances: Sequence[InstancePrediction], directory: Path) -> List[str]:
    directory.mkdir(parents=True, exist_ok=True)
    paths: List[str] = []
    for index, instance in enumerate(instances):
        filename = "bbox_overlay.png" if len(instances) == 1 else f"bbox_overlay_{index:03d}.png"
        path = directory / filename
        save_overlay(image, [instance], path, draw_boxes=True, draw_masks=False)
        paths.append(f"overlays/{path.name}")
    return paths


def save_outline_overlays(image: Image.Image, instances: Sequence[InstancePrediction], directory: Path) -> List[str]:
    directory.mkdir(parents=True, exist_ok=True)
    paths: List[str] = []
    for index, instance in enumerate(instances):
        filename = "outline_overlay.png" if len(instances) == 1 else f"outline_overlay_{index:03d}.png"
        path = directory / filename
        save_outline_overlay(image, instance, path, color=color_for_instance(index))
        paths.append(f"overlays/{path.name}")
    return paths


def save_outline_overlay(image: Image.Image, instance: InstancePrediction, output: Path, *, color: Tuple[int, int, int]) -> str:
    output.parent.mkdir(parents=True, exist_ok=True)
    base = image.convert("RGBA")
    mask = binary_mask(instance.mask, image_size=base.size[0])
    edge = mask.filter(ImageFilter.FIND_EDGES).point(lambda value: 255 if value > 0 else 0)
    outline = Image.new("RGBA", base.size, color + (0,))
    outline.putalpha(edge)
    Image.alpha_composite(base, outline).convert("RGB").save(output)
    return str(output.relative_to(output.parents[1]).as_posix())


def padded_box(box: Sequence[float], size: Tuple[int, int], padding: float) -> Tuple[int, int, int, int]:
    width, height = size
    x1, y1, x2, y2 = [float(value) for value in box]
    pad_x = (x2 - x1) * padding
    pad_y = (y2 - y1) * padding
    return (
        max(0, int(math.floor(x1 - pad_x))),
        max(0, int(math.floor(y1 - pad_y))),
        min(width, int(math.ceil(x2 + pad_x))),
        min(height, int(math.ceil(y2 + pad_y))),
    )


def save_overlay(
    image: Image.Image,
    instances: Sequence[InstancePrediction],
    output: Path,
    *,
    draw_boxes: bool,
    draw_masks: bool,
    labels: Optional[Sequence[str]] = None,
) -> str:
    output.parent.mkdir(parents=True, exist_ok=True)
    base = image.convert("RGBA")
    if draw_masks:
        overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
        for index, instance in enumerate(instances):
            mask = binary_mask(instance.mask, image_size=base.size[0])
            color = color_for_instance(index)
            layer = Image.new("RGBA", base.size, color + (0,))
            layer.putalpha(mask.point(lambda value: int(value * 0.45)))
            overlay = Image.alpha_composite(overlay, layer)
        base = Image.alpha_composite(base, overlay)
    draw = ImageDraw.Draw(base)
    font = ImageFont.load_default()
    for index, instance in enumerate(instances):
        color = color_for_instance(index)
        box = box_saved(instance)
        if draw_boxes:
            draw.rectangle(box, outline=color, width=3)
        if labels:
            draw.text((box[0], max(0, box[1] - 14)), labels[index], fill=color, font=font)
    base.convert("RGB").save(output)
    return str(output.relative_to(output.parents[1]).as_posix())


def color_for_instance(index: int) -> Tuple[int, int, int]:
    palette = [
        (255, 99, 71),
        (60, 179, 113),
        (65, 105, 225),
        (255, 165, 0),
        (148, 0, 211),
        (220, 20, 60),
        (0, 206, 209),
        (255, 215, 0),
    ]
    return palette[index % len(palette)]


def build_metadata(
    *,
    args: argparse.Namespace,
    sample: Sample,
    prompt: str,
    primary_prompt: str,
    prompt_attempts: Sequence[Dict[str, Any]],
    used_fallback_prompt: bool,
    fallback_prompt_index: Optional[int],
    status: str,
    image_hash: str,
    prediction: PredictionResult,
    num_instances_detected: int,
    merge_info: Dict[str, Any],
    truncation: Dict[str, Any],
    files: Dict[str, Any],
    evidence_types: Tuple[str, ...],
    config_hash: str,
) -> Dict[str, Any]:
    return round_floats(
        {
            "schema_version": SAMPLE_SCHEMA_VERSION,
            "status": status,
            "benchmark": sample.benchmark,
            "sample_id": sample.sample_id,
            "pair_id": sample.pair_id,
            "image_role": sample.image_role,
            "source_image_path": str(sample.image_path),
            "source_image_sha256": image_hash,
            "category": sample.category,
            "subcategory": sample.subcategory,
            "source_variant": sample.source_variant,
            "prompt_key": sample.prompt_key,
            "prompt_profile": sample.prompt_profile,
            "prompt_text": prompt,
            "primary_prompt_text": primary_prompt,
            "selected_prompt_text": prompt,
            "prompt_attempts": list(prompt_attempts),
            "used_fallback_prompt": used_fallback_prompt,
            "fallback_prompt_index": fallback_prompt_index,
            "prompt_mode": sample.prompt_mode,
            "target_region_key": sample.target_region_key,
            "target_part_key": sample.target_part_key,
            "prompt_modes_requested": list(sample.prompt_modes_requested),
            "is_multi_prompt_run": sample.is_multi_prompt_run,
            "model_id": args.model_id,
            "backend": args.backend,
            "backend_resolved": resolve_backend(args.backend),
            "dtype": args.dtype,
            "thresholds": {
                "box_threshold": args.box_threshold,
                "mask_threshold": args.mask_threshold,
                "min_score": args.min_score,
                "min_mask_area": args.min_mask_area,
            },
            "output_size_wh": [args.image_size, args.image_size],
            "original_size_wh": list(prediction.original_size_wh),
            "num_instances": len(prediction.instances),
            "num_instances_detected": num_instances_detected,
            "num_instances_saved": len(prediction.instances),
            "max_instances": args.max_instances,
            "merge_instances_over": args.merge_instances_over,
            "instances_merged": merge_info["instances_merged"],
            "num_instances_before_merge": merge_info["num_instances_before_merge"],
            "num_instances_after_merge": merge_info["num_instances_after_merge"],
            "merged_source_instance_ids": merge_info["merged_source_instance_ids"],
            "instance_selection_method": merge_info["instance_selection_method"],
            "truncated_instance_count": truncation["truncated_instance_count"],
            "instance_truncation_method": truncation["instance_truncation_method"],
            "evidence_types": list(evidence_types),
            "crop_padding": args.crop_padding,
            "evidence_files": {
                "source": files["source"],
                "bbox_overlay": files["bbox_overlay"],
                "bbox_overlays": files["bbox_overlays"],
                "outline_overlay": files["outline_overlay"],
                "outline_overlays": files["outline_overlays"],
                "mask_overlay": files["mask_overlay"],
                "mask_bbox_overlay": files["mask_bbox_overlay"],
                "mask_combined": files["mask_combined"],
                "crops": files["crops"],
                "zooms": files["zooms"],
                "zoom_panels": files["zoom_panels"],
            },
            "debug_files": {"debug_overlay_with_labels": files["debug_overlay_with_labels"]},
            "config_hash": config_hash,
        }
    )


def build_instances_json(prediction: PredictionResult, files: Dict[str, Any], *, image_size: int) -> Dict[str, Any]:
    records = []
    for index, instance in enumerate(scale_instances(prediction, image_size=image_size)):
        x1, y1, x2, y2 = box_saved(instance)
        box_area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
        mask_area = positive_pixel_count(instance.mask)
        records.append(
            {
                "instance_id": index,
                "score": instance.score,
                "source_instance_ids": list(instance.source_instance_ids or (index,)),
                "instance_selection_method": instance.instance_selection_method,
                "box_xyxy_original": instance.box_xyxy_original,
                "box_xyxy_saved": box_saved(instance),
                "box_area_ratio": box_area / float(image_size * image_size),
                "mask_area_ratio": mask_area / float(image_size * image_size),
                "mask_file": item_or_none(files["masks"], index),
                "bbox_overlay_file": item_or_none(files["bbox_overlays"], index),
                "outline_overlay_file": item_or_none(files["outline_overlays"], index),
                "crop_file": item_or_none(files["crops"], index),
                "zoom_file": item_or_none(files["zooms"], index),
                "zoom_panel_file": item_or_none(files["zoom_panels"], index),
            }
        )
    return round_floats({"schema_version": INSTANCE_SCHEMA_VERSION, "instances": records})


def item_or_none(items: Sequence[str], index: int) -> Optional[str]:
    return items[index] if index < len(items) else None


def positive_pixel_count(mask: Image.Image) -> int:
    return int(sum(1 for value in mask.convert("L").getdata() if value > 0))


def write_failed_sample(
    *,
    args: argparse.Namespace,
    sample: Sample,
    output_root: Path,
    error: Exception,
    config_hash: str,
) -> Dict[str, Any]:
    output_dir = sample_output_dir(output_root, sample)
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": SAMPLE_SCHEMA_VERSION,
        "status": "failed",
        "benchmark": sample.benchmark,
        "sample_id": sample.sample_id,
        "pair_id": sample.pair_id,
        "image_role": sample.image_role,
        "source_image_path": str(sample.image_path),
        "source_image_sha256": sha256_file(sample.image_path) if sample.image_path.exists() else None,
        "subcategory": sample.subcategory,
        "source_variant": sample.source_variant,
        "prompt_profile": sample.prompt_profile,
        "prompt_mode": sample.prompt_mode,
        "target_region_key": sample.target_region_key,
        "target_part_key": sample.target_part_key,
        "prompt_modes_requested": list(sample.prompt_modes_requested),
        "is_multi_prompt_run": sample.is_multi_prompt_run,
        "model_id": args.model_id,
        "backend": args.backend,
        "backend_resolved": resolve_backend(args.backend),
        "num_instances": 0,
        "num_instances_detected": 0,
        "num_instances_saved": 0,
        "max_instances": args.max_instances,
        "merge_instances_over": args.merge_instances_over,
        "instances_merged": False,
        "num_instances_before_merge": 0,
        "num_instances_after_merge": 0,
        "merged_source_instance_ids": [],
        "instance_selection_method": "none",
        "truncated_instance_count": 0,
        "instance_truncation_method": "none",
        "error_type": type(error).__name__,
        "error_message": str(error),
        "traceback": traceback.format_exc(),
        "config_hash": config_hash,
    }
    write_json(output_dir / "metadata.json", payload)
    return {
        "status": "failed",
        "sample_id": sample.sample_id,
        "prompt_profile": sample.prompt_profile,
        "num_instances": 0,
        "num_instances_detected": 0,
        "num_instances_saved": 0,
        "max_instances": args.max_instances,
        "merge_instances_over": args.merge_instances_over,
        "instances_merged": False,
        "num_instances_before_merge": 0,
        "num_instances_after_merge": 0,
        "merged_source_instance_ids": [],
        "truncated_instance_count": 0,
        "instance_truncation_method": "none",
        "error_type": type(error).__name__,
    }


def build_run_summary(
    *,
    args: argparse.Namespace,
    samples: Sequence[Sample],
    results: Sequence[Dict[str, Any]],
    output_root: Path,
    prompt_config_path: Path,
    config_hash: str,
    backend_resolved: str,
    events_log_path: Path,
    elapsed_seconds: float,
) -> Dict[str, Any]:
    statuses = Counter(str(result.get("status")) for result in results)
    scores = [score for result in results for score in result.get("scores", [])]
    latencies = [float(result["latency_ms"]) for result in results if result.get("latency_ms") is not None]
    instance_counts = Counter(int(result.get("num_instances") or 0) for result in results)
    detected_instance_counts = Counter(int(result.get("num_instances_detected") or result.get("num_instances") or 0) for result in results)
    saved_instance_counts = Counter(int(result.get("num_instances_saved") or result.get("num_instances") or 0) for result in results)
    truncated_counts = [int(result.get("truncated_instance_count") or 0) for result in results]
    merged_results = [result for result in results if result.get("instances_merged")]
    fallback_used = [result for result in results if result.get("used_fallback_prompt")]
    selected_prompt_distribution = Counter(str(result.get("selected_prompt_text") or result.get("prompt") or "") for result in results)
    prompt_distribution = Counter()
    prompt_profile_distribution = Counter()
    status_by_prompt_profile: Dict[str, Counter[str]] = {}
    instance_count_by_prompt_profile: Dict[str, Counter[int]] = {}
    detected_instance_count_by_prompt_profile: Dict[str, Counter[int]] = {}
    saved_instance_count_by_prompt_profile: Dict[str, Counter[int]] = {}
    truncated_by_prompt_profile: Dict[str, int] = {}
    merged_by_prompt_profile: Dict[str, int] = {}
    merged_source_instances_by_prompt_profile: Dict[str, int] = {}
    for sample in samples:
        prompt_distribution[sample.prompt_key] += 1
        prompt_profile_distribution[sample.prompt_profile] += 1
    for result in results:
        profile = str(result.get("prompt_profile") or "unknown")
        status_by_prompt_profile.setdefault(profile, Counter())[str(result.get("status"))] += 1
        instance_count_by_prompt_profile.setdefault(profile, Counter())[int(result.get("num_instances") or 0)] += 1
        detected_instance_count_by_prompt_profile.setdefault(profile, Counter())[
            int(result.get("num_instances_detected") or result.get("num_instances") or 0)
        ] += 1
        saved_instance_count_by_prompt_profile.setdefault(profile, Counter())[
            int(result.get("num_instances_saved") or result.get("num_instances") or 0)
        ] += 1
        truncated_by_prompt_profile[profile] = truncated_by_prompt_profile.get(profile, 0) + int(
            result.get("truncated_instance_count") or 0
        )
        if result.get("instances_merged"):
            merged_by_prompt_profile[profile] = merged_by_prompt_profile.get(profile, 0) + 1
            merged_source_instances_by_prompt_profile[profile] = (
                merged_source_instances_by_prompt_profile.get(profile, 0)
                + int(result.get("num_instances_before_merge") or 0)
            )
    return round_floats(
        {
            "schema_version": RUN_SUMMARY_SCHEMA_VERSION,
            "run_id": str(int(time.time())),
            "benchmark": args.benchmark,
            "input_root": str(resolve_path(args.input_root)) if args.input_root else None,
            "manifest": str(resolve_path(args.manifest)) if args.manifest else None,
            "output_root": str(output_root),
            "model_id": args.model_id,
            "backend": args.backend,
            "backend_resolved": backend_resolved,
            "device": args.device,
            "dtype": args.dtype,
            "prompt_config": str(prompt_config_path),
            "prompt_modes_requested": list(samples[0].prompt_modes_requested) if samples else parse_prompt_modes(args),
            "num_samples": len(samples),
            "num_success": statuses.get("success", 0),
            "num_empty": statuses.get("empty", 0),
            "num_failed": statuses.get("failed", 0),
            "num_skipped_resume": statuses.get("skipped_resume", 0),
            "num_used_fallback_prompt": len(fallback_used),
            "selected_prompt_distribution": {
                prompt: count
                for prompt, count in sorted(selected_prompt_distribution.items())
                if prompt
            },
            "instance_count_distribution": dict(sorted(instance_counts.items())),
            "instance_count_detected_distribution": dict(sorted(detected_instance_counts.items())),
            "instance_count_saved_distribution": dict(sorted(saved_instance_counts.items())),
            "prompt_distribution": dict(sorted(prompt_distribution.items())),
            "prompt_profile_distribution": dict(sorted(prompt_profile_distribution.items())),
            "status_by_prompt_profile": {
                profile: dict(sorted(counter.items()))
                for profile, counter in sorted(status_by_prompt_profile.items())
            },
            "instance_count_by_prompt_profile": {
                profile: dict(sorted(counter.items()))
                for profile, counter in sorted(instance_count_by_prompt_profile.items())
            },
            "instance_count_detected_by_prompt_profile": {
                profile: dict(sorted(counter.items()))
                for profile, counter in sorted(detected_instance_count_by_prompt_profile.items())
            },
            "instance_count_saved_by_prompt_profile": {
                profile: dict(sorted(counter.items()))
                for profile, counter in sorted(saved_instance_count_by_prompt_profile.items())
            },
            "truncated_instances_by_prompt_profile": dict(sorted(truncated_by_prompt_profile.items())),
            "empty_by_prompt_profile": {
                profile: counter.get("empty", 0)
                for profile, counter in sorted(status_by_prompt_profile.items())
            },
            "failed_by_prompt_profile": {
                profile: counter.get("failed", 0)
                for profile, counter in sorted(status_by_prompt_profile.items())
            },
            "score_summary": summarize_scores(scores),
            "total_latency_seconds": sum(latencies) / 1000.0,
            "mean_latency_ms": (sum(latencies) / len(latencies)) if latencies else None,
            "crop_padding": args.crop_padding,
            "max_instances": args.max_instances,
            "merge_instances_over": args.merge_instances_over,
            "num_samples_with_merged_instances": len(merged_results),
            "total_instances_before_merge": sum(int(result.get("num_instances_before_merge") or 0) for result in merged_results),
            "merged_instances_by_prompt_profile": dict(sorted(merged_by_prompt_profile.items())),
            "merged_source_instances_by_prompt_profile": dict(sorted(merged_source_instances_by_prompt_profile.items())),
            "num_samples_with_truncated_instances": sum(1 for count in truncated_counts if count > 0),
            "total_truncated_instances": sum(truncated_counts),
            "events_log": str(events_log_path),
            "config_hash": config_hash,
            "elapsed_seconds": elapsed_seconds,
        }
    )


def summarize_scores(scores: Sequence[float]) -> Dict[str, Optional[float]]:
    if not scores:
        return {"count": 0, "min": None, "max": None, "mean": None}
    return {"count": len(scores), "min": min(scores), "max": max(scores), "mean": sum(scores) / len(scores)}


def build_config_hash(args: argparse.Namespace, prompt_config: Dict[str, Any], evidence_types: Tuple[str, ...]) -> str:
    payload = {
        "schema_version": SAMPLE_SCHEMA_VERSION,
        "model_id": args.model_id,
        "backend": args.backend,
        "box_threshold": args.box_threshold,
        "mask_threshold": args.mask_threshold,
        "min_score": args.min_score,
        "min_mask_area": args.min_mask_area,
        "image_size": args.image_size,
        "evidence_types": evidence_types,
        "crop_padding": args.crop_padding,
        "max_instances": args.max_instances,
        "merge_instances_over": args.merge_instances_over,
        "prompt_mode": args.prompt_mode,
        "prompt_modes": parse_prompt_modes(args),
        "target_region_key": args.target_region_key,
        "target_part_key": args.target_part_key,
        "prompt_config": prompt_config,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


class EventLogger:
    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text("", encoding="utf-8")

    def write(self, event: str, payload: Dict[str, Any]) -> None:
        record = {"event": event, "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), **payload}
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(round_floats(record), ensure_ascii=False) + "\n")


class ProgressReporter:
    def __init__(self, *, total: int, style: str, log_every: int, log_seconds: float):
        self.total = total
        self.style = "lines" if style == "auto" and not sys.stdout.isatty() else style
        if self.style == "auto":
            self.style = "bar"
        self.log_every = log_every
        self.log_seconds = log_seconds
        self.started_at = time.time()
        self.last_log_at = self.started_at
        self.counts: Counter[str] = Counter()
        self.current_index = 0

    def sample_start(self, *, index: int, sample: Sample) -> None:
        self.current_index = index
        if self.style == "bar":
            self._print_bar(index=index - 1, current=f"{sample.prompt_profile}/{sample.sample_id}", end="\r")

    def sample_done(self, *, index: int, sample: Sample, result: Dict[str, Any]) -> None:
        self.counts[str(result.get("status", "unknown"))] += 1
        self._maybe_print(index=index, current=f"{sample.prompt_profile}/{sample.sample_id}", force=False)

    def sample_failed(self, *, index: int, sample: Sample, error: Exception) -> None:
        self.counts["failed"] += 1
        self._maybe_print(index=index, current=f"{sample.prompt_profile}/{sample.sample_id} failed={type(error).__name__}", force=True)

    def finish(self) -> None:
        if self.style == "bar":
            self._print_bar(index=self.total, current="done", end="\n")
        elif self.style == "lines":
            self._print_line(index=self.total, current="done")

    def _maybe_print(self, *, index: int, current: str, force: bool) -> None:
        if self.style == "off":
            return
        now = time.time()
        should_log = force or index == self.total or index % self.log_every == 0 or (now - self.last_log_at) >= self.log_seconds
        if not should_log:
            return
        if self.style == "bar":
            self._print_bar(index=index, current=current, end="\r" if index < self.total else "\n")
        else:
            self._print_line(index=index, current=current)
        self.last_log_at = now

    def _print_line(self, *, index: int, current: str) -> None:
        print(self._message(index=index, current=current), flush=True)

    def _print_bar(self, *, index: int, current: str, end: str) -> None:
        width = 28
        ratio = 0.0 if self.total == 0 else min(1.0, max(0.0, index / self.total))
        filled = int(width * ratio)
        bar = "#" * filled + "-" * (width - filled)
        print(f"Generating [{bar}] {self._message(index=index, current=current)}", end=end, flush=True)

    def _message(self, *, index: int, current: str) -> str:
        elapsed = time.time() - self.started_at
        rate = index / elapsed if elapsed > 0 and index > 0 else 0.0
        remaining = max(0, self.total - index)
        eta = remaining / rate if rate > 0 else 0.0
        pct = 0.0 if self.total == 0 else (index / self.total) * 100.0
        return (
            f"{index}/{self.total} ({pct:5.1f}%) elapsed {format_duration(elapsed)} eta {format_duration(eta)} "
            f"| success={self.counts.get('success', 0)} empty={self.counts.get('empty', 0)} "
            f"failed={self.counts.get('failed', 0)} skipped={self.counts.get('skipped_resume', 0)} | {current}"
        )


def format_duration(seconds: float) -> str:
    seconds = max(0, int(seconds))
    minutes, sec = divmod(seconds, 60)
    hours, minute = divmod(minutes, 60)
    if hours:
        return f"{hours}h{minute:02d}m{sec:02d}s"
    if minutes:
        return f"{minutes}m{sec:02d}s"
    return f"{sec}s"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def round_floats(value: Any, ndigits: int = 4) -> Any:
    if isinstance(value, float):
        return round(value, ndigits) if math.isfinite(value) else value
    if isinstance(value, list):
        return [round_floats(item, ndigits) for item in value]
    if isinstance(value, dict):
        return {str(key): round_floats(item, ndigits) for key, item in value.items()}
    return value


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    sys.exit(main())
