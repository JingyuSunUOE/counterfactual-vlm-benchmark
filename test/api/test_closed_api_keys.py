#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from dotenv import load_dotenv

try:
    from anthropic import Anthropic
except Exception:  # pragma: no cover - import availability depends on env
    Anthropic = None

try:
    from google import genai
    from google.genai import types as genai_types
except Exception:  # pragma: no cover - import availability depends on env
    genai = None
    genai_types = None

try:
    from openai import OpenAI
except Exception:  # pragma: no cover - import availability depends on env
    OpenAI = None


REPO_ROOT = Path(__file__).resolve().parents[2]

PROVIDER_ENV_VARS = {
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "google": "GOOGLE_API_KEY",
}

DEFAULT_MODELS = {
    "openai": "gpt-4o-mini",
    "anthropic": "claude-3-7-sonnet-20250219",
    "google": "gemini-2.5-flash",
}

TEST_PROMPT = "Reply with exactly: ok"


@dataclass
class CheckResult:
    provider: str
    model: str
    status: str
    latency_ms: Optional[int]
    response_preview: Optional[str]
    error: Optional[str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "provider": self.provider,
            "model": self.model,
            "status": self.status,
            "latency_ms": self.latency_ms,
            "response_preview": self.response_preview,
            "error": self.error,
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Live API connectivity check for closed model providers.")
    parser.add_argument(
        "--provider",
        choices=("all", "openai", "anthropic", "google"),
        default="all",
        help="Which provider to test.",
    )
    parser.add_argument("--openai-model", default=DEFAULT_MODELS["openai"], help="OpenAI model used for the test.")
    parser.add_argument(
        "--anthropic-model",
        default=DEFAULT_MODELS["anthropic"],
        help="Anthropic Claude model used for the test.",
    )
    parser.add_argument("--google-model", default=DEFAULT_MODELS["google"], help="Gemini model used for the test.")
    parser.add_argument("--max-output-tokens", type=int, default=16, help="Max output tokens for each test request.")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature for each test request.")
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON instead of a table.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    load_dotenv(REPO_ROOT / ".env", override=True)

    provider_models = {
        "openai": args.openai_model,
        "anthropic": args.anthropic_model,
        "google": args.google_model,
    }
    providers = list(PROVIDER_ENV_VARS) if args.provider == "all" else [args.provider]

    checks: Dict[str, Callable[[str, float, int], str]] = {
        "openai": check_openai,
        "anthropic": check_anthropic,
        "google": check_google,
    }

    results = [
        run_check(
            provider=provider,
            model=provider_models[provider],
            check=checks[provider],
            temperature=args.temperature,
            max_output_tokens=args.max_output_tokens,
        )
        for provider in providers
    ]

    if args.json:
        print(json.dumps([result.to_dict() for result in results], indent=2, ensure_ascii=False))
    else:
        print_results(results)

    return 0 if all(result.status == "OK" for result in results) else 1


def run_check(
    *,
    provider: str,
    model: str,
    check: Callable[[str, float, int], str],
    temperature: float,
    max_output_tokens: int,
) -> CheckResult:
    env_var = PROVIDER_ENV_VARS[provider]
    if not os.getenv(env_var):
        return CheckResult(
            provider=provider,
            model=model,
            status="SKIP",
            latency_ms=None,
            response_preview=None,
            error=f"Missing {env_var}",
        )

    started = time.perf_counter()
    try:
        text = check(model, temperature, max_output_tokens)
        return CheckResult(
            provider=provider,
            model=model,
            status="OK",
            latency_ms=int((time.perf_counter() - started) * 1000),
            response_preview=preview(text),
            error=None,
        )
    except Exception as exc:
        return CheckResult(
            provider=provider,
            model=model,
            status="FAIL",
            latency_ms=int((time.perf_counter() - started) * 1000),
            response_preview=None,
            error=f"{type(exc).__name__}: {exc}",
        )


def check_openai(model: str, temperature: float, max_output_tokens: int) -> str:
    if OpenAI is None:
        raise RuntimeError("openai package is not installed.")
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    response = client.responses.create(
        model=model,
        input=[{"role": "user", "content": [{"type": "input_text", "text": TEST_PROMPT}]}],
        temperature=temperature,
        max_output_tokens=max_output_tokens,
    )
    text = getattr(response, "output_text", None)
    if text:
        return text.strip()
    parts: List[str] = []
    for item in getattr(response, "output", []) or []:
        for content in getattr(item, "content", []) or []:
            content_text = getattr(content, "text", None)
            if content_text:
                parts.append(content_text)
    if parts:
        return "\n".join(parts).strip()
    raise RuntimeError("OpenAI response did not contain text output.")


def check_anthropic(model: str, temperature: float, max_output_tokens: int) -> str:
    if Anthropic is None:
        raise RuntimeError("anthropic package is not installed.")
    client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    response = client.messages.create(
        model=model,
        max_tokens=max_output_tokens,
        temperature=temperature,
        messages=[{"role": "user", "content": TEST_PROMPT}],
    )
    parts = [block.text for block in response.content if getattr(block, "type", None) == "text"]
    if parts:
        return "\n".join(parts).strip()
    raise RuntimeError("Anthropic response did not contain text output.")


def check_google(model: str, temperature: float, max_output_tokens: int) -> str:
    if genai is None or genai_types is None:
        raise RuntimeError("google-genai package is not installed.")
    client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))
    response = client.models.generate_content(
        model=model,
        contents=[genai_types.Part.from_text(text=TEST_PROMPT)],
        config=genai_types.GenerateContentConfig(
            temperature=temperature,
            max_output_tokens=max_output_tokens,
        ),
    )
    text = getattr(response, "text", None)
    if text:
        return text.strip()
    candidates = getattr(response, "candidates", None) or []
    for candidate in candidates:
        content = getattr(candidate, "content", None)
        if not content:
            continue
        response_parts = getattr(content, "parts", None) or []
        texts = [getattr(part, "text", None) for part in response_parts if getattr(part, "text", None)]
        if texts:
            return "\n".join(texts).strip()
    raise RuntimeError("Google response did not contain text output.")


def preview(text: str, limit: int = 80) -> str:
    normalized = " ".join(text.split())
    if len(normalized) <= limit:
        return normalized
    return normalized[: limit - 3] + "..."


def print_results(results: List[CheckResult]) -> None:
    headers = ("Provider", "Model", "Status", "Latency", "Response / Error")
    rows = []
    for result in results:
        latency = "-" if result.latency_ms is None else f"{result.latency_ms} ms"
        detail = result.response_preview if result.status == "OK" else result.error
        rows.append((result.provider, result.model, result.status, latency, detail or ""))

    widths = [
        max(len(headers[index]), *(len(row[index]) for row in rows))
        for index in range(len(headers))
    ]
    print("Closed API connectivity check")
    print(format_row(headers, widths))
    print(format_row(tuple("-" * width for width in widths), widths))
    for row in rows:
        print(format_row(row, widths))


def format_row(values: tuple[str, ...], widths: List[int]) -> str:
    return "  ".join(value.ljust(widths[index]) for index, value in enumerate(values))


if __name__ == "__main__":
    sys.exit(main())
