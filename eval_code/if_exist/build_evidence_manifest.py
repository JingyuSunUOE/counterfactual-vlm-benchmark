#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


EVAL_CODE_DIR = Path(__file__).resolve().parents[1]
if str(EVAL_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(EVAL_CODE_DIR))

from eval_common import (  # noqa: E402
    DEFAULT_METADATA_ROOT,
    EVIDENCE_SCHEMA_VERSION,
    REPO_ROOT,
    natural_key,
    normalize_image_key,
)


DEFAULT_ORIGINAL_ROOT = REPO_ROOT / "vision_dataset" / "if_exist" / "sam3_object_original_pad005"
DEFAULT_CF_ROOT = REPO_ROOT / "vision_dataset" / "if_exist" / "sam3_object_cf_pad005"
DEFAULT_OUTPUT_JSON = DEFAULT_METADATA_ROOT / "if_exist_sam3_object_evidence_manifest.json"
DEFAULT_OUTPUT_CSV = DEFAULT_METADATA_ROOT / "if_exist_sam3_object_evidence_manifest.csv"
EXPECTED_PAIR_COUNT = 160


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the if_exist object-only SAM evidence manifest.")
    parser.add_argument("--original-root", type=Path, default=DEFAULT_ORIGINAL_ROOT)
    parser.add_argument("--cf-root", type=Path, default=DEFAULT_CF_ROOT)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_OUTPUT_CSV)
    parser.add_argument("--dry-run", action="store_true", help="Validate and preview without writing files.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing output files.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    original_root = args.original_root.expanduser().resolve()
    cf_root = args.cf_root.expanduser().resolve()
    if not original_root.exists():
        raise SystemExit(f"Original evidence root not found: {original_root}")
    if not cf_root.exists():
        raise SystemExit(f"CF evidence root not found: {cf_root}")

    original_records = collect_evidence_records(original_root, expected_role="original")
    cf_records = collect_evidence_records(cf_root, expected_role="cf")
    manifest = build_manifest_records(original_records, cf_records)
    if len(manifest) != EXPECTED_PAIR_COUNT:
        raise SystemExit(f"Expected {EXPECTED_PAIR_COUNT} evidence pairs, found {len(manifest)}.")

    print(f"original_success={len(original_records)} cf_success={len(cf_records)} pairs={len(manifest)}")
    print(f"output_json={args.output_json}")
    print(f"output_csv={args.output_csv}")
    if manifest:
        preview = manifest[0]
        print(
            "preview="
            + json.dumps(
                {
                    "pair_key": preview["pair_key"],
                    "subcategory": preview["subcategory"],
                    "source_variant": preview["source_variant"],
                    "original": preview["original"],
                    "cf": preview["cf"],
                },
                ensure_ascii=False,
            )
        )

    if args.dry_run:
        return 0
    for output_path in (args.output_json, args.output_csv):
        if output_path.exists() and not args.overwrite:
            raise SystemExit(f"Output exists; pass --overwrite to replace: {output_path}")
        output_path.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_csv(args.output_csv, manifest)
    return 0


def collect_evidence_records(root: Path, *, expected_role: str) -> Dict[Tuple[str, str, str], Dict[str, Any]]:
    records: Dict[Tuple[str, str, str], Dict[str, Any]] = {}
    problems: List[str] = []
    metadata_paths = sorted(root.rglob("metadata.json"), key=lambda item: natural_key(item.as_posix()))
    if not metadata_paths:
        raise SystemExit(f"No metadata.json files found under {root}")
    for metadata_path in metadata_paths:
        metadata = load_json(metadata_path)
        if metadata.get("image_role") != expected_role:
            continue
        if metadata.get("status") != "success":
            continue
        if metadata.get("prompt_profile") != "object":
            continue
        try:
            record = build_role_record(metadata, metadata_path=metadata_path, role=expected_role)
        except ValueError as exc:
            problems.append(f"{metadata_path}: {exc}")
            continue
        key = (record["subcategory"], record["source_variant"], record["pair_key"])
        if key in records:
            problems.append(f"{metadata_path}: duplicate evidence key {key}")
            continue
        records[key] = record
    if problems:
        preview = "\n- ".join(problems[:20])
        suffix = "" if len(problems) <= 20 else f"\n- ... {len(problems) - 20} additional problems"
        raise SystemExit(f"Evidence collection failed:\n- {preview}{suffix}")
    return records


def build_role_record(metadata: Dict[str, Any], *, metadata_path: Path, role: str) -> Dict[str, Any]:
    evidence_files = metadata.get("evidence_files") or {}
    source_path = rebase_known_path(str(metadata.get("source_image_path") or ""))
    if not source_path.exists():
        raise ValueError(f"source image does not exist after rebasing: {source_path}")
    source_stem = source_path.stem
    pair_key = normalize_image_key(source_stem, is_edited=(role == "cf"))
    bbox_overlay = first_available_evidence(evidence_files, keys=("bbox_overlays", "bbox_overlay"))
    crop = first_available_evidence(evidence_files, keys=("crops",))
    zoom_panel = first_available_evidence(evidence_files, keys=("zoom_panels",))
    role_payload = {
        "raw": source_path.as_posix(),
        "bbox_overlay": resolve_relative(metadata_path.parent, bbox_overlay).as_posix(),
        "crop": resolve_relative(metadata_path.parent, crop).as_posix(),
        "zoom_panel": resolve_relative(metadata_path.parent, zoom_panel).as_posix(),
    }
    missing = [path for path in role_payload.values() if not Path(path).exists()]
    if missing:
        raise ValueError("missing evidence files: " + ", ".join(missing))
    role_payload["evidence_available"] = {
        "raw": True,
        "bbox": bool(role_payload["bbox_overlay"]),
        "crop": bool(role_payload["crop"]),
        "zoom_panel": bool(role_payload["zoom_panel"]),
        "tool_bundle": all(bool(role_payload[field]) for field in ("bbox_overlay", "crop", "zoom_panel")),
    }
    return {
        "pair_key": pair_key,
        "subcategory": str(metadata.get("subcategory") or ""),
        "source_variant": str(metadata.get("source_variant") or ""),
        "generator": {
            "model_id": metadata.get("model_id"),
            "backend": metadata.get("backend_resolved") or metadata.get("backend"),
            "dtype": metadata.get("dtype"),
            "crop_padding": metadata.get("crop_padding"),
            "prompt_profile": metadata.get("prompt_profile"),
        },
        role: role_payload,
    }


def first_available_evidence(evidence_files: Dict[str, Any], *, keys: Tuple[str, ...]) -> str:
    for key in keys:
        value = evidence_files.get(key)
        if isinstance(value, list) and value:
            return str(value[0])
        if isinstance(value, str) and value:
            return value
    raise ValueError(f"missing evidence file for keys={keys}")


def resolve_relative(base_dir: Path, value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path.resolve(strict=False)
    return (base_dir / path).resolve(strict=False)


def rebase_known_path(value: str) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute() and path.exists():
        return path.resolve()
    for marker in ("vision_dataset/if_exist/", "cf_dataset/if_exist_cf/", "dataset/if_exist/"):
        marker_index = value.find(marker)
        if marker_index != -1:
            return (REPO_ROOT / value[marker_index:]).resolve(strict=False)
    if path.is_absolute():
        return path.resolve(strict=False)
    return (REPO_ROOT / path).resolve(strict=False)


def build_manifest_records(
    original_records: Dict[Tuple[str, str, str], Dict[str, Any]],
    cf_records: Dict[Tuple[str, str, str], Dict[str, Any]],
) -> List[Dict[str, Any]]:
    original_keys = set(original_records)
    cf_keys = set(cf_records)
    missing_cf = sorted(original_keys - cf_keys)
    missing_original = sorted(cf_keys - original_keys)
    if missing_cf or missing_original:
        messages = []
        if missing_cf:
            messages.append(f"missing CF evidence for {len(missing_cf)} keys: {missing_cf[:10]}")
        if missing_original:
            messages.append(f"missing original evidence for {len(missing_original)} keys: {missing_original[:10]}")
        raise SystemExit("Evidence pairing failed:\n- " + "\n- ".join(messages))

    manifest: List[Dict[str, Any]] = []
    for key in sorted(original_keys, key=lambda item: natural_key("|".join(item))):
        original = original_records[key]
        cf = cf_records[key]
        manifest.append(
            {
                "schema_version": EVIDENCE_SCHEMA_VERSION,
                "benchmark": "if_exist",
                "pair_key": original["pair_key"],
                "subcategory": original["subcategory"],
                "source_variant": original["source_variant"],
                "generator": merge_generator(original.get("generator", {}), cf.get("generator", {})),
                "qc_status": "pass",
                "usable_tool_conditions": {
                    "raw": True,
                    "bbox": True,
                    "crop": True,
                    "zoom_panel": True,
                    "tool_bundle": True,
                },
                "original": original["original"],
                "cf": cf["cf"],
            }
        )
    return manifest


def merge_generator(original: Dict[str, Any], cf: Dict[str, Any]) -> Dict[str, Any]:
    merged = dict(original)
    for key, value in cf.items():
        if merged.get(key) in (None, "") and value not in (None, ""):
            merged[key] = value
    return merged


def write_csv(path: Path, manifest: List[Dict[str, Any]]) -> None:
    columns = [
        "schema_version",
        "benchmark",
        "pair_key",
        "subcategory",
        "source_variant",
        "qc_status",
        "usable_raw",
        "usable_bbox",
        "usable_crop",
        "usable_zoom_panel",
        "usable_tool_bundle",
        "model_id",
        "backend",
        "dtype",
        "crop_padding",
        "prompt_profile",
        "original_raw",
        "original_bbox_overlay",
        "original_crop",
        "original_zoom_panel",
        "cf_raw",
        "cf_bbox_overlay",
        "cf_crop",
        "cf_zoom_panel",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for record in manifest:
            generator = record.get("generator") or {}
            usable = record.get("usable_tool_conditions") or {}
            writer.writerow(
                {
                    "schema_version": record.get("schema_version"),
                    "benchmark": record.get("benchmark"),
                    "pair_key": record.get("pair_key"),
                    "subcategory": record.get("subcategory"),
                    "source_variant": record.get("source_variant"),
                    "qc_status": record.get("qc_status"),
                    "usable_raw": usable.get("raw"),
                    "usable_bbox": usable.get("bbox"),
                    "usable_crop": usable.get("crop"),
                    "usable_zoom_panel": usable.get("zoom_panel"),
                    "usable_tool_bundle": usable.get("tool_bundle"),
                    "model_id": generator.get("model_id"),
                    "backend": generator.get("backend"),
                    "dtype": generator.get("dtype"),
                    "crop_padding": generator.get("crop_padding"),
                    "prompt_profile": generator.get("prompt_profile"),
                    "original_raw": record["original"]["raw"],
                    "original_bbox_overlay": record["original"]["bbox_overlay"],
                    "original_crop": record["original"]["crop"],
                    "original_zoom_panel": record["original"]["zoom_panel"],
                    "cf_raw": record["cf"]["raw"],
                    "cf_bbox_overlay": record["cf"]["bbox_overlay"],
                    "cf_crop": record["cf"]["crop"],
                    "cf_zoom_panel": record["cf"]["zoom_panel"],
                }
            )


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


if __name__ == "__main__":
    sys.exit(main())
