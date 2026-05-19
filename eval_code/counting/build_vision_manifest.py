#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ANNOTATIONS = REPO_ROOT / "eval_results" / "counting" / "metadata" / "counting_count_annotations.json"
DEFAULT_ORIGINAL_OUTPUT = REPO_ROOT / "eval_results" / "counting" / "metadata" / "counting_original_vision_manifest.jsonl"
DEFAULT_CF_OUTPUT = REPO_ROOT / "eval_results" / "counting" / "metadata" / "counting_cf_vision_manifest.jsonl"
MANIFEST_SCHEMA_VERSION = "vision_manifest_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build object-only SAM vision manifests for counting.")
    parser.add_argument("--annotations", type=Path, default=DEFAULT_ANNOTATIONS)
    parser.add_argument("--original-output", type=Path, default=DEFAULT_ORIGINAL_OUTPUT)
    parser.add_argument("--cf-output", type=Path, default=DEFAULT_CF_OUTPUT)
    parser.add_argument("--dry-run", action="store_true", help="Validate and preview without writing files.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing manifest outputs.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    annotations_path = resolve_path(args.annotations)
    annotations = load_annotations(annotations_path)
    original_manifest = build_original_manifest(annotations)
    cf_manifest = build_cf_manifest(annotations)
    ensure_unique_sample_ids(original_manifest, label="original")
    ensure_unique_sample_ids(cf_manifest, label="cf")

    print(f"annotations={len(annotations)}")
    print(f"original_manifest={len(original_manifest)} output={resolve_path(args.original_output)}")
    print(f"cf_manifest={len(cf_manifest)} output={resolve_path(args.cf_output)}")
    if original_manifest:
        print("original_preview=" + json.dumps(original_manifest[0], ensure_ascii=False))
    if cf_manifest:
        print("cf_preview=" + json.dumps(cf_manifest[0], ensure_ascii=False))

    if args.dry_run:
        return 0

    for output_path in (resolve_path(args.original_output), resolve_path(args.cf_output)):
        if output_path.exists() and not args.overwrite:
            raise SystemExit(f"Output exists; pass --overwrite to replace: {output_path}")
        output_path.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(resolve_path(args.original_output), original_manifest)
    write_jsonl(resolve_path(args.cf_output), cf_manifest)
    return 0


def resolve_path(path: Path) -> Path:
    path = path.expanduser()
    return path if path.is_absolute() else (REPO_ROOT / path).resolve()


def load_annotations(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        raise SystemExit(f"Counting annotation file not found: {path}")
    records = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(records, list) or not records:
        raise SystemExit("Counting annotation file must contain a non-empty JSON list.")
    return records


def build_original_manifest(annotations: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    by_original: Dict[str, Dict[str, Any]] = {}
    for raw in annotations:
        original_path = rebase_counting_path(str(raw["original_path"]))
        if not original_path.exists():
            raise SystemExit(f"Original image path does not exist: {original_path}")
        key = original_path.as_posix()
        if key in by_original:
            continue
        by_original[key] = build_manifest_record(
            raw=raw,
            image_path=original_path,
            image_role="original",
            sample_id=sample_id_for(raw, image_path=original_path, image_role="original", include_edit=False),
            pair_id=pair_id_for_original(raw, original_path),
        )
    return sorted(by_original.values(), key=lambda item: natural_key(item["sample_id"]))


def build_cf_manifest(annotations: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    for raw in annotations:
        cf_path = rebase_counting_path(str(raw["cf_path"]))
        if not cf_path.exists():
            raise SystemExit(f"CF image path does not exist: {cf_path}")
        records.append(
            build_manifest_record(
                raw=raw,
                image_path=cf_path,
                image_role="cf",
                sample_id=sample_id_for(raw, image_path=cf_path, image_role="cf", include_edit=True),
                pair_id=str(raw["pair_key"]),
            )
        )
    return sorted(records, key=lambda item: natural_key(item["sample_id"]))


def build_manifest_record(
    *,
    raw: Dict[str, Any],
    image_path: Path,
    image_role: str,
    sample_id: str,
    pair_id: str,
) -> Dict[str, Any]:
    subcategory = str(raw["subcategory"])
    return {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "benchmark": "counting",
        "sample_id": sample_id,
        "pair_id": pair_id,
        "image_path": path_for_manifest(image_path),
        "image_role": image_role,
        "category": "counting",
        "subcategory": subcategory,
        "source_variant": str(raw.get("source_variant") or "real"),
        "prompt_key": subcategory,
        "prompt_mode": "object",
        "target_region_key": None,
        "target_part_key": None,
    }


def sample_id_for(raw: Dict[str, Any], *, image_path: Path, image_role: str, include_edit: bool) -> str:
    parts = [
        str(raw["subset"]),
        str(raw["subcategory"]),
        str(raw.get("source_variant") or "real"),
        image_path.stem,
    ]
    if include_edit:
        parts.append(str(raw["edit_type"]))
    parts.append(image_role)
    return sanitize_id("_".join(parts))


def pair_id_for_original(raw: Dict[str, Any], image_path: Path) -> str:
    return "/".join(
        [
            "original",
            str(raw["subset"]),
            str(raw["subcategory"]),
            str(raw.get("source_variant") or "real"),
            image_path.stem,
        ]
    )


def rebase_counting_path(value: str) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute() and path.exists():
        return path.resolve()
    markers = (
        "dataset/counting/",
        "cf_dataset/counting_cf/",
        "eval_results/counting/",
    )
    for marker in markers:
        marker_index = value.find(marker)
        if marker_index != -1:
            return (REPO_ROOT / value[marker_index:]).resolve(strict=False)
    if path.is_absolute():
        return path.resolve(strict=False)
    return (REPO_ROOT / path).resolve(strict=False)


def path_for_manifest(path: Path) -> str:
    try:
        return path.resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def ensure_unique_sample_ids(records: List[Dict[str, Any]], *, label: str) -> None:
    seen: Dict[str, int] = {}
    duplicates: List[str] = []
    for record in records:
        sample_id = str(record["sample_id"])
        seen[sample_id] = seen.get(sample_id, 0) + 1
        if seen[sample_id] == 2:
            duplicates.append(sample_id)
    if duplicates:
        raise SystemExit(f"Duplicate {label} sample_id values: {duplicates[:20]}")


def write_jsonl(path: Path, records: List[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def sanitize_id(value: str) -> str:
    cleaned = []
    for char in value.replace("\\", "/"):
        cleaned.append(char if char.isalnum() or char in {"-", "_"} else "_")
    return "".join(cleaned).strip("_")


def natural_key(value: str) -> List[Any]:
    parts: List[Any] = []
    current = ""
    is_digit = False
    for char in value:
        if char.isdigit():
            if current and not is_digit:
                parts.append(current.lower())
                current = ""
            current += char
            is_digit = True
        else:
            if current and is_digit:
                parts.append(int(current))
                current = ""
            current += char
            is_digit = False
    if current:
        parts.append(int(current) if is_digit else current.lower())
    return parts


if __name__ == "__main__":
    sys.exit(main())
