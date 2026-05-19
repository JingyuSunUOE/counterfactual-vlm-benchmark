#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
COUNTING_CODE = REPO_ROOT / "eval_code" / "counting"
if str(COUNTING_CODE) not in sys.path:
    sys.path.insert(0, str(COUNTING_CODE))

from eval_common import (  # noqa: E402
    CountingAnnotation,
    build_image_groups,
    filter_image_pairs_for_evidence,
    flatten_image_groups,
    load_evidence_manifest,
)


def main() -> int:
    tests = [
        test_qc_empty_loads_and_skips,
        test_cf_only_crop_requires_only_cf_side,
        test_both_tool_bundle_requires_both_sides,
        test_outline_builds_raw_plus_outline,
    ]
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
    return 0


def test_qc_empty_loads_and_skips() -> None:
    with tempfile.TemporaryDirectory(prefix="counting_evidence_filter_") as tmp:
        root = Path(tmp)
        pair = make_pair(root)
        manifest_path = root / "manifest.json"
        write_manifest(manifest_path, root, pair, qc_status="empty", missing_cf=("crop",))
        manifest = load_evidence_manifest(manifest_path)
        filtered, stats = filter_image_pairs_for_evidence(
            [pair],
            eval_mode="cf_only",
            tool_condition="crop",
            evidence_manifest=manifest,
            evidence_manifest_path=manifest_path,
            evidence_qc_filter="pass",
            missing_evidence_policy="skip",
        )
        assert filtered == []
        assert stats["skipped_by_qc_status"] == {"empty": 1}


def test_cf_only_crop_requires_only_cf_side() -> None:
    with tempfile.TemporaryDirectory(prefix="counting_evidence_filter_") as tmp:
        root = Path(tmp)
        pair = make_pair(root)
        manifest_path = root / "manifest.json"
        write_manifest(manifest_path, root, pair, missing_original=("crop",))
        manifest = load_evidence_manifest(manifest_path)
        filtered, stats = filter_image_pairs_for_evidence(
            [pair],
            eval_mode="cf_only",
            tool_condition="crop",
            evidence_manifest=manifest,
            evidence_manifest_path=manifest_path,
            evidence_qc_filter="pass",
            missing_evidence_policy="skip",
        )
        assert filtered == [pair]
        assert stats["skipped_pair_count_due_to_evidence"] == 0


def test_both_tool_bundle_requires_both_sides() -> None:
    with tempfile.TemporaryDirectory(prefix="counting_evidence_filter_") as tmp:
        root = Path(tmp)
        pair = make_pair(root)
        manifest_path = root / "manifest.json"
        write_manifest(manifest_path, root, pair, missing_original=("outline_overlay",))
        manifest = load_evidence_manifest(manifest_path)
        filtered, stats = filter_image_pairs_for_evidence(
            [pair],
            eval_mode="both",
            tool_condition="tool_bundle",
            evidence_manifest=manifest,
            evidence_manifest_path=manifest_path,
            evidence_qc_filter="pass",
            missing_evidence_policy="skip",
        )
        assert filtered == []
        assert stats["skipped_by_missing_tool_condition"] == {"original.outline_overlay": 1}


def test_outline_builds_raw_plus_outline() -> None:
    with tempfile.TemporaryDirectory(prefix="counting_evidence_filter_") as tmp:
        root = Path(tmp)
        pair = make_pair(root)
        manifest_path = root / "manifest.json"
        write_manifest(manifest_path, root, pair)
        manifest = load_evidence_manifest(manifest_path)
        groups, evidence_record = build_image_groups(
            pair=pair,
            eval_mode="cf_only",
            tool_condition="outline",
            evidence_manifest=manifest,
            evidence_manifest_path=manifest_path,
            missing_evidence_policy="error",
        )
        paths = flatten_image_groups(groups)
        assert evidence_record is not None
        assert len(paths) == 2
        assert paths[0] == pair.edited_path
        assert paths[1].name == "cf_outline_overlay.png"


def make_pair(root: Path) -> CountingAnnotation:
    original = root / "dataset/counting/hand_paw/five/human_hand/original.png"
    cf = root / "cf_dataset/counting_cf/hand_paw/add_to_six/human_hand/cf.png"
    original.parent.mkdir(parents=True, exist_ok=True)
    cf.parent.mkdir(parents=True, exist_ok=True)
    original.write_bytes(b"x")
    cf.write_bytes(b"x")
    return CountingAnnotation(
        category="counting",
        subset="hand_paw",
        subcategory="human_hand",
        source_variant="real",
        key="pair_1",
        original_path=original,
        edited_path=cf,
        question_group_id="counting_hand_paw_human_hand_add_to_six",
        edit_type="add_to_six",
        count_direction="more_than_expected",
        normal_count=5,
        cf_count=6,
        biased_count=5,
        count_attribute="fingers",
        display_name="human hand",
        annotation_pair_key="pair_1",
    )


def write_manifest(
    path: Path,
    root: Path,
    pair: CountingAnnotation,
    *,
    qc_status: str = "pass",
    missing_original: tuple[str, ...] = (),
    missing_cf: tuple[str, ...] = (),
) -> None:
    record = {
        "schema_version": "counting_sam3_object_evidence_v1",
        "benchmark": "counting",
        "pair_key": pair.annotation_pair_key,
        "question_group_id": pair.question_group_id,
        "subset": pair.subset,
        "subcategory": pair.subcategory,
        "source_variant": pair.source_variant,
        "edit_type": pair.edit_type,
        "count_direction": pair.count_direction,
        "qc_status": qc_status,
        "generator": {"model_id": "facebook/sam3", "prompt_profile": "object"},
        "original": role_payload(root, "original", pair.original_path, missing_original),
        "cf": role_payload(root, "cf", pair.edited_path, missing_cf),
    }
    path.write_text(json.dumps([record]), encoding="utf-8")


def role_payload(root: Path, role: str, raw_path: Path, missing: tuple[str, ...]) -> dict:
    payload = {
        "raw": str(raw_path),
        "qc_status": "pass",
        "selected_instance_id": 0,
        "selected_prompt_text": "hand",
    }
    for field in ("bbox_overlay", "crop", "zoom_panel", "outline_overlay"):
        if field in missing:
            continue
        evidence_path = root / f"{role}_{field}.png"
        evidence_path.write_bytes(b"x")
        payload[field] = str(evidence_path)
    return payload


if __name__ == "__main__":
    raise SystemExit(main())
