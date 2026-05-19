#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "eval_code" / "if_exist"))

from eval_common import (  # noqa: E402
    classify_error,
    completed_success_count,
    is_fatal_error_kind,
    load_checkpoint,
    successful_task_signatures,
    write_checkpoint,
)


def main() -> int:
    assert_equal("OpenAI quota classification", classify_error("insufficient_quota: billing hard limit"), "quota_exceeded")
    assert_equal("OpenAI 429 classification", classify_error("Too many requests", status_code=429), "rate_limit")
    assert_equal("Anthropic auth classification", classify_error("401 Unauthorized", status_code=401), "auth_error")
    assert_equal(
        "adapter signature regression classification",
        classify_error("ClosedProviderAdapter.generate() missing 1 required keyword-only argument"),
        "auth_error",
    )
    assert_equal("Gemini quota classification", classify_error("RESOURCE_EXHAUSTED quota exceeded"), "quota_exceeded")
    assert_equal(
        "Gemini unavailable classification",
        classify_error("invalid literal for int() with base 10: 'UNAVAILABLE'"),
        "server_error",
    )
    assert_equal(
        "Gemini internal classification",
        classify_error("invalid literal for int() with base 10: 'INTERNAL'"),
        "server_error",
    )
    assert_equal("server classification", classify_error("upstream failure", status_code=503), "server_error")
    assert_equal("timeout classification", classify_error("Read timed out"), "timeout")
    if not is_fatal_error_kind("quota_exceeded", stop_on_quota=True):
        raise SystemExit("quota_exceeded should be fatal when stop_on_quota is enabled.")
    if is_fatal_error_kind("rate_limit", stop_on_quota=True):
        raise SystemExit("rate_limit should not be fatal.")

    records = [
        record("sig-success", "correct", None),
        record("sig-error", "error", "Backbone generation failed: insufficient_quota"),
        record("sig-judge-error", "error", "Judge scoring failed"),
    ]
    done = successful_task_signatures(records)
    if done != {"sig-success"}:
        raise SystemExit(f"success-only resume signatures regressed: {done}")
    assert_equal("completed success count", completed_success_count(records), 1)

    with tempfile.TemporaryDirectory(prefix="if_exist_checkpoint_") as tmp_dir:
        run_dir = Path(tmp_dir) / "run"
        checkpoint = write_checkpoint(
            run_dir=run_dir,
            run_id="run",
            run_status="interrupted_quota_exceeded",
            total_tasks=3,
            records=records,
            skipped_resume_count=1,
            pending_count=2,
            last_task_signature="sig-error",
            last_error_kind="quota_exceeded",
            last_error_message="quota exhausted",
        )
        loaded = load_checkpoint(run_dir)
        if loaded != checkpoint:
            raise SystemExit("checkpoint round-trip failed.")
        require_keys(
            checkpoint,
            [
                "run_id",
                "run_status",
                "total_tasks",
                "completed_success_count",
                "error_count",
                "skipped_resume_count",
                "pending_count",
                "last_task_signature",
                "last_error_kind",
                "last_error_message",
                "updated_at",
            ],
        )
        assert_equal("checkpoint completed_success_count", checkpoint["completed_success_count"], 1)
        assert_equal("checkpoint error_count", checkpoint["error_count"], 2)

    print("if_exist error recovery tests passed.")
    return 0


def record(signature: str, scored_label: str, error_message: str | None) -> dict[str, str | None]:
    return {
        "task_signature": signature,
        "scored_label": scored_label,
        "judge_label": scored_label if scored_label != "error" else None,
        "error_message": error_message,
    }


def assert_equal(name: str, actual: object, expected: object) -> None:
    if actual != expected:
        raise SystemExit(f"{name}: expected {expected!r}, got {actual!r}")


def require_keys(payload: dict[str, object], keys: list[str]) -> None:
    missing = [key for key in keys if key not in payload]
    if missing:
        raise SystemExit(f"checkpoint missing keys: {missing}\n{json.dumps(payload, indent=2)}")


if __name__ == "__main__":
    sys.exit(main())
