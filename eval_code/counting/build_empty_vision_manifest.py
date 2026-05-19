#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List


REPO_ROOT = Path(__file__).resolve().parents[2]
MANIFEST_SCHEMA_VERSION = "vision_manifest_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a counting vision manifest containing only primary-empty SAM samples.")
    parser.add_argument("--input-manifest", type=Path, required=True, help="Full counting vision manifest JSONL.")
    parser.add_argument("--evidence-root", type=Path, required=True, help="Primary SAM evidence root to inspect.")
    parser.add_argument("--output", type=Path, required=True, help="Output empty-only manifest JSONL.")
    parser.add_argument("--dry-run", action="store_true", help="Preview without writing output.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing output.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_manifest = resolve_path(args.input_manifest)
    evidence_root = resolve_path(args.evidence_root)
    output = resolve_path(args.output)
    rows = read_manifest(input_manifest)
    metadata_by_sample = read_metadata_index(evidence_root)
    empty_rows: List[Dict[str, Any]] = []
    missing: List[str] = []
    for row in rows:
        sample_id = str(row["sample_id"])
        metadata = metadata_by_sample.get(sample_id)
        if metadata is None:
            missing.append(sample_id)
            continue
        if metadata.get("status") == "empty":
            empty_rows.append(row)
    if missing:
        raise SystemExit(
            "Primary evidence metadata missing for manifest sample ids: "
            + ", ".join(missing[:20])
            + (f" ... ({len(missing)} total)" if len(missing) > 20 else "")
        )

    print(f"input_manifest={input_manifest}")
    print(f"evidence_root={evidence_root}")
    print(f"input_samples={len(rows)}")
    print(f"empty_samples={len(empty_rows)}")
    print(f"output={output}")
    if empty_rows:
        print("preview=" + json.dumps(empty_rows[0], ensure_ascii=False))
    if args.dry_run:
        return 0
    if output.exists() and not args.overwrite:
        raise SystemExit(f"Output exists; pass --overwrite to replace: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(output, empty_rows)
    return 0


def resolve_path(path: Path) -> Path:
    path = path.expanduser()
    return path if path.is_absolute() else (REPO_ROOT / path).resolve()


def read_manifest(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        raise SystemExit(f"Input manifest not found: {path}")
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if row.get("schema_version") != MANIFEST_SCHEMA_VERSION:
                raise SystemExit(f"Invalid manifest schema at {path}:{line_number}")
            rows.append(row)
    if not rows:
        raise SystemExit(f"Input manifest is empty: {path}")
    return rows


def read_metadata_index(root: Path) -> Dict[str, Dict[str, Any]]:
    if not root.exists():
        raise SystemExit(f"Evidence root not found: {root}")
    index: Dict[str, Dict[str, Any]] = {}
    duplicates: List[str] = []
    for metadata_path in sorted(root.rglob("metadata.json")):
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        sample_id = str(metadata.get("sample_id") or "")
        if not sample_id:
            continue
        if sample_id in index:
            duplicates.append(sample_id)
            continue
        metadata["_metadata_path"] = str(metadata_path)
        index[sample_id] = metadata
    if not index:
        raise SystemExit(f"No metadata.json files found under evidence root: {root}")
    if duplicates:
        raise SystemExit(f"Duplicate sample_id values in evidence metadata: {duplicates[:20]}")
    return index


def write_jsonl(path: Path, records: List[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    sys.exit(main())
