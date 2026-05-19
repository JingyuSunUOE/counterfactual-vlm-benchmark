from __future__ import annotations

import hashlib
import argparse
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


OPENAI_EXTENDED_RETENTION_PREFIXES = (
    "gpt-4.1",
    "gpt-5",
    "gpt-5.1",
    "gpt-5.2",
)


def add_provider_cache_arguments(parser: Any) -> None:
    parser.add_argument(
        "--provider-cache",
        choices=("off", "auto", "on"),
        default="auto",
        help="Enable provider-level prompt/image caching when supported by the selected native provider.",
    )
    parser.add_argument(
        "--prompt-cache-retention",
        choices=("in_memory", "24h"),
        default="in_memory",
        help="OpenAI prompt cache retention policy. Ignored by non-OpenAI providers.",
    )
    parser.add_argument(
        "--cache-key-mode",
        choices=("image_pair", "image", "run"),
        default="image_pair",
        help="Granularity for stable provider cache keys.",
    )
    parser.add_argument(
        "--cache-usage-log",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Record provider cache usage metadata in raw result rows.",
    )

def build_provider_cache_key(
    *,
    benchmark_name: str,
    input_mode: str,
    image_paths: Sequence[Any],
    cache_key_mode: str,
) -> str:
    normalized_paths = [_normalize_path(path) for path in image_paths]
    if cache_key_mode == "image":
        key_paths = normalized_paths[-1:] if normalized_paths else []
    elif cache_key_mode == "run":
        key_paths = []
    else:
        key_paths = normalized_paths
    payload = "\n".join([benchmark_name, input_mode, cache_key_mode, *key_paths])
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:32]
    return f"vlm-eval-{digest}"


def _normalize_path(path: Any) -> str:
    try:
        return str(Path(path).expanduser().resolve())
    except Exception:
        return str(path)


def provider_cache_state(
    *,
    provider: str,
    provider_cache: str,
    model_id: str,
    prompt_cache_retention: str = "in_memory",
) -> dict[str, Any]:
    if provider_cache == "off":
        return {"enabled": False, "reason": "disabled", "provider": provider}
    if provider == "openai":
        state = {"enabled": True, "reason": "supported", "provider": provider}
        if prompt_cache_retention == "24h" and not supports_openai_extended_retention(model_id):
            if provider_cache == "on":
                raise ValueError(f"OpenAI model '{model_id}' does not support 24h prompt cache retention.")
            state["retention_warning"] = "24h retention unsupported for this model; using default in-memory retention."
        return state
    if provider == "anthropic":
        return {"enabled": True, "reason": "supported", "provider": provider}
    if provider == "google":
        return {"enabled": True, "reason": "implicit_only", "provider": provider}
    if provider_cache == "on":
        return {"enabled": False, "reason": f"unsupported_provider:{provider}", "provider": provider}
    return {"enabled": False, "reason": f"unsupported_provider:{provider}", "provider": provider}


def supports_openai_extended_retention(model_id: str) -> bool:
    normalized = model_id.lower()
    return any(normalized.startswith(prefix) for prefix in OPENAI_EXTENDED_RETENTION_PREFIXES)


def openai_cache_kwargs(
    *,
    cache_state: Mapping[str, Any],
    cache_key: str | None,
    model_id: str,
    prompt_cache_retention: str,
) -> dict[str, Any]:
    if not cache_state.get("enabled") or not cache_key:
        return {}
    kwargs: dict[str, Any] = {"prompt_cache_key": cache_key}
    if prompt_cache_retention == "24h" and supports_openai_extended_retention(model_id):
        kwargs["extra_body"] = {"prompt_cache_retention": "24h"}
    return kwargs


def add_anthropic_cache_control(content: list[dict[str, Any]], *, enabled: bool) -> None:
    if not enabled:
        return
    for block in reversed(content):
        if block.get("type") == "image":
            block["cache_control"] = {"type": "ephemeral"}
            return


def extract_cache_usage(provider: str, response: Any) -> dict[str, Any]:
    if response is None:
        return {"provider": provider}
    if provider == "google":
        usage = getattr(response, "usage_metadata", None)
    else:
        usage = getattr(response, "usage", None)
    usage_dict = object_to_plain(usage)
    return summarize_usage_dict(provider=provider, usage=usage_dict)


def summarize_usage_dict(*, provider: str, usage: Any) -> dict[str, Any]:
    if not isinstance(usage, Mapping):
        return {"provider": provider}
    flat = flatten_mapping(usage)
    interesting = {
        key: value
        for key, value in flat.items()
        if isinstance(value, (int, float, str, bool))
        and (
            "token" in key.lower()
            or "cache" in key.lower()
            or key.lower() in {"input_tokens", "output_tokens", "total_tokens", "prompt_tokens", "completion_tokens"}
        )
    }
    return {"provider": provider, **interesting}


def flatten_mapping(value: Mapping[str, Any], *, prefix: str = "") -> dict[str, Any]:
    flattened: dict[str, Any] = {}
    for key, item in value.items():
        full_key = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(item, Mapping):
            flattened.update(flatten_mapping(item, prefix=full_key))
        else:
            flattened[full_key] = item
    return flattened


def object_to_plain(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, Mapping):
        return {str(key): object_to_plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [object_to_plain(item) for item in value]
    if isinstance(value, (str, int, float, bool)):
        return value
    if hasattr(value, "model_dump"):
        try:
            return object_to_plain(value.model_dump())
        except Exception:
            pass
    if hasattr(value, "to_dict"):
        try:
            return object_to_plain(value.to_dict())
        except Exception:
            pass
    if hasattr(value, "__dict__"):
        return {
            key: object_to_plain(item)
            for key, item in vars(value).items()
            if not key.startswith("_") and isinstance(item, (str, int, float, bool, dict, list, tuple, type(None)))
        }
    return str(value)


def count_cache_usage(records: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    total = 0
    with_usage = 0
    numeric_totals: dict[str, float] = {}
    for record in records:
        total += 1
        usage = record.get("provider_cache_usage")
        if not isinstance(usage, Mapping) or len(usage) <= 1:
            continue
        with_usage += 1
        for key, value in usage.items():
            if key == "provider" or not isinstance(value, (int, float)):
                continue
            numeric_totals[key] = numeric_totals.get(key, 0.0) + float(value)
    return {"request_count": total, "requests_with_cache_usage": with_usage, "numeric_totals": numeric_totals}
