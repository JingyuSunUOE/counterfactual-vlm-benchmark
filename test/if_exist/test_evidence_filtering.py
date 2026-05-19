#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
EVAL_DIR = REPO_ROOT / "eval_code" / "if_exist"
if str(EVAL_DIR) not in sys.path:
    sys.path.insert(0, str(EVAL_DIR))

from eval_common import (  # noqa: E402
    EVIDENCE_SCHEMA_VERSION,
    ImagePair,
    build_image_groups,
    filter_image_pairs_for_evidence,
    load_evidence_manifest,
)


def write_file(path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"x")
    return str(path)


def make_pair(root: Path, *, key: str = "pair1") -> ImagePair:
    original = root / "dataset" / "if_exist" / "camel" / f"{key}.png"
    cf = root / "cf_dataset" / "if_exist_cf" / "camel" / f"{key}_remove_hump.png"
    write_file(original)
    write_file(cf)
    return ImagePair(
        category="if_exist",
        subcategory="camel",
        source_variant="real",
        key=key,
        original_path=original,
        edited_path=cf,
    )


def role_payload(root: Path, role: str, *, missing: tuple[str, ...] = ()) -> dict:
    payload = {"raw": write_file(root / role / "raw.png")}
    fields = {
        "bbox_overlay": root / role / "bbox.png",
        "crop": root / role / "crop.png",
        "zoom_panel": root / role / "zoom_panel.png",
    }
    for field, path in fields.items():
        payload[field] = None if field in missing else write_file(path)
    return payload


def write_manifest(path: Path, records: list[dict]) -> None:
    path.write_text(json.dumps(records, indent=2), encoding="utf-8")


def manifest_record(root: Path, *, qc_status: str = "pass", missing_original: tuple[str, ...] = (), missing_cf: tuple[str, ...] = ()) -> dict:
    return {
        "schema_version": EVIDENCE_SCHEMA_VERSION,
        "benchmark": "if_exist",
        "pair_key": "pair1",
        "subcategory": "camel",
        "source_variant": "real",
        "qc_status": qc_status,
        "generator": {"model_id": "mock", "prompt_profile": "object"},
        "original": role_payload(root, "original", missing=missing_original),
        "cf": role_payload(root, "cf", missing=missing_cf),
    }


def test_empty_qc_does_not_break_manifest_loader() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        manifest_path = root / "manifest.json"
        write_manifest(manifest_path, [manifest_record(root, qc_status="empty", missing_cf=("crop",))])
        manifest = load_evidence_manifest(manifest_path)
        assert len(manifest) == 1
        assert next(iter(manifest.values()))["qc_status"] == "empty"


def test_cf_only_crop_requires_only_cf_crop() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        pair = make_pair(root)
        manifest_path = root / "manifest.json"
        write_manifest(manifest_path, [manifest_record(root, missing_original=("crop",))])
        manifest = load_evidence_manifest(manifest_path)
        eligible, stats = filter_image_pairs_for_evidence(
            [pair],
            eval_mode="cf_only",
            tool_condition="crop",
            evidence_manifest=manifest,
            evidence_manifest_path=manifest_path,
            evidence_qc_filter="pass",
            missing_evidence_policy="skip",
        )
        assert eligible == [pair]
        assert stats["skipped_pair_count_due_to_evidence"] == 0


def test_both_tool_bundle_requires_both_roles() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        pair = make_pair(root)
        manifest_path = root / "manifest.json"
        write_manifest(manifest_path, [manifest_record(root, missing_original=("crop",))])
        manifest = load_evidence_manifest(manifest_path)
        eligible, stats = filter_image_pairs_for_evidence(
            [pair],
            eval_mode="both",
            tool_condition="tool_bundle",
            evidence_manifest=manifest,
            evidence_manifest_path=manifest_path,
            evidence_qc_filter="pass",
            missing_evidence_policy="skip",
        )
        assert eligible == []
        assert stats["skipped_by_missing_tool_condition"] == {"original.crop": 1}


def test_pass_review_and_all_qc_filters() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        pair = make_pair(root)
        manifest_path = root / "manifest.json"
        write_manifest(manifest_path, [manifest_record(root, qc_status="review")])
        manifest = load_evidence_manifest(manifest_path)
        eligible_pass, _stats = filter_image_pairs_for_evidence(
            [pair],
            eval_mode="cf_only",
            tool_condition="bbox",
            evidence_manifest=manifest,
            evidence_manifest_path=manifest_path,
            evidence_qc_filter="pass",
            missing_evidence_policy="skip",
        )
        eligible_review, _stats = filter_image_pairs_for_evidence(
            [pair],
            eval_mode="cf_only",
            tool_condition="bbox",
            evidence_manifest=manifest,
            evidence_manifest_path=manifest_path,
            evidence_qc_filter="pass_review",
            missing_evidence_policy="skip",
        )
        eligible_all, _stats = filter_image_pairs_for_evidence(
            [pair],
            eval_mode="cf_only",
            tool_condition="bbox",
            evidence_manifest=manifest,
            evidence_manifest_path=manifest_path,
            evidence_qc_filter="all",
            missing_evidence_policy="skip",
        )
        assert eligible_pass == []
        assert eligible_review == [pair]
        assert eligible_all == [pair]


def test_missing_policy_error_and_raw_fallback() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        pair = make_pair(root)
        manifest_path = root / "manifest.json"
        write_manifest(manifest_path, [manifest_record(root, missing_cf=("crop",))])
        manifest = load_evidence_manifest(manifest_path)
        try:
            filter_image_pairs_for_evidence(
                [pair],
                eval_mode="cf_only",
                tool_condition="crop",
                evidence_manifest=manifest,
                evidence_manifest_path=manifest_path,
                evidence_qc_filter="pass",
                missing_evidence_policy="error",
            )
        except SystemExit:
            pass
        else:
            raise AssertionError("missing_evidence_policy=error should fail.")
        eligible, stats = filter_image_pairs_for_evidence(
            [pair],
            eval_mode="cf_only",
            tool_condition="crop",
            evidence_manifest=manifest,
            evidence_manifest_path=manifest_path,
            evidence_qc_filter="pass",
            missing_evidence_policy="raw_fallback",
        )
        assert eligible == [pair]
        assert stats["raw_fallback_pair_count"] == 1
        image_groups, _record = build_image_groups(
            pair=pair,
            eval_mode="cf_only",
            tool_condition="crop",
            evidence_manifest=manifest,
            evidence_manifest_path=manifest_path,
            missing_evidence_policy="raw_fallback",
        )
        assert image_groups[0].evidence == ()


def main() -> int:
    tests = [
        test_empty_qc_does_not_break_manifest_loader,
        test_cf_only_crop_requires_only_cf_crop,
        test_both_tool_bundle_requires_both_roles,
        test_pass_review_and_all_qc_filters,
        test_missing_policy_error_and_raw_fallback,
    ]
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
