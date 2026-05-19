#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
EVAL_CODE_DIR = REPO_ROOT / "eval_code"
if str(EVAL_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(EVAL_CODE_DIR))

from provider_cache import (  # noqa: E402
    build_provider_cache_key,
    extract_cache_usage,
    provider_cache_state,
    supports_openai_extended_retention,
)


class UsageObject:
    def __init__(self) -> None:
        self.input_tokens = 100
        self.output_tokens = 5
        self.input_tokens_details = {"cached_tokens": 80}


class ResponseObject:
    usage = UsageObject()


def main() -> int:
    key_a = build_provider_cache_key(
        benchmark_name="if_exist",
        input_mode="cf_only",
        image_paths=["/tmp/a.png"],
        cache_key_mode="image_pair",
    )
    key_b = build_provider_cache_key(
        benchmark_name="if_exist",
        input_mode="cf_only",
        image_paths=["/tmp/a.png"],
        cache_key_mode="image_pair",
    )
    if key_a != key_b or not key_a.startswith("vlm-eval-"):
        raise SystemExit("provider cache key should be stable and prefixed.")

    if not provider_cache_state(provider="openai", provider_cache="auto", model_id="gpt-4.1")["enabled"]:
        raise SystemExit("OpenAI cache should be enabled in auto mode.")
    if not provider_cache_state(provider="anthropic", provider_cache="auto", model_id="claude-3-7-sonnet")["enabled"]:
        raise SystemExit("Anthropic cache should be enabled in auto mode.")
    if provider_cache_state(provider="openrouter", provider_cache="auto", model_id="x")["enabled"]:
        raise SystemExit("OpenRouter cache should not be marked enabled.")
    if not supports_openai_extended_retention("gpt-5.1"):
        raise SystemExit("gpt-5.1 should support extended retention.")

    usage = extract_cache_usage("openai", ResponseObject())
    if usage.get("input_tokens") != 100 or usage.get("input_tokens_details.cached_tokens") != 80:
        raise SystemExit(f"cache usage extraction failed: {usage}")

    print("provider cache tests passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
