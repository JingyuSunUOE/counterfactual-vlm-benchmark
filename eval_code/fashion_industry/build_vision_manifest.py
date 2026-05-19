#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_METADATA_DIR = REPO_ROOT / "eval_results" / "fashion_industry" / "metadata"
PROMPT_CONFIG_DIR = REPO_ROOT / "vision_configs" / "sam3_prompts"
MANIFEST_SCHEMA_VERSION = "vision_manifest_v1"
DOMAIN_CONFIG = {
    "fashion": {
        "metadata": "fashion_cf_metadata.json",
        "original_output": "fashion_original_vision_manifest.jsonl",
        "cf_output": "fashion_cf_vision_manifest.jsonl",
        "prompt_config": "fashion.json",
        "key_field": "brand_key",
    },
    "industry": {
        "metadata": "industry_cf_metadata.json",
        "original_output": "industry_original_vision_manifest.jsonl",
        "cf_output": "industry_cf_vision_manifest.jsonl",
        "prompt_config": "industry.json",
        "key_field": "sub_category",
    },
}


class ManifestBuildError(RuntimeError):
    pass


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build object-only SAM vision manifests for fashion/industry.")
    parser.add_argument("--domain", choices=["fashion", "industry", "both"], default="both")
    parser.add_argument("--metadata-dir", type=Path, default=DEFAULT_METADATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_METADATA_DIR)
    parser.add_argument("--dry-run", action="store_true", help="Validate and preview without writing files.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing manifest outputs.")
    parser.add_argument("--strict", dest="strict", action="store_true", default=True)
    parser.add_argument("--no-strict", dest="strict", action="store_false")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    metadata_dir = resolve_path(args.metadata_dir)
    output_dir = resolve_path(args.output_dir)
    domains = ["fashion", "industry"] if args.domain == "both" else [args.domain]

    outputs: list[tuple[str, Path, list[dict[str, Any]], Path, list[dict[str, Any]]]] = []
    for domain in domains:
        metadata = load_metadata(metadata_dir / DOMAIN_CONFIG[domain]["metadata"])
        prompt_keys = load_prompt_keys(PROMPT_CONFIG_DIR / DOMAIN_CONFIG[domain]["prompt_config"])
        original_manifest = build_original_manifest(domain, metadata, prompt_keys, strict=args.strict)
        cf_manifest = build_cf_manifest(domain, metadata, prompt_keys, strict=args.strict)
        validate_manifest_counts(domain, metadata, original_manifest, cf_manifest, strict=args.strict)
        original_output = output_dir / DOMAIN_CONFIG[domain]["original_output"]
        cf_output = output_dir / DOMAIN_CONFIG[domain]["cf_output"]
        outputs.append((domain, original_output, original_manifest, cf_output, cf_manifest))

        print(f"{domain}: metadata={len(metadata)} original_manifest={len(original_manifest)} cf_manifest={len(cf_manifest)}")
        if original_manifest:
            print(f"{domain}_original_preview=" + json.dumps(original_manifest[0], ensure_ascii=False))
        if cf_manifest:
            print(f"{domain}_cf_preview=" + json.dumps(cf_manifest[0], ensure_ascii=False))

    if args.dry_run:
        print("dry_run=true, no files written")
        return 0

    output_dir.mkdir(parents=True, exist_ok=True)
    for _domain, original_output, original_manifest, cf_output, cf_manifest in outputs:
        write_jsonl(original_output, original_manifest, overwrite=args.overwrite)
        write_jsonl(cf_output, cf_manifest, overwrite=args.overwrite)
    return 0


def resolve_path(path: Path) -> Path:
    path = path.expanduser()
    return path if path.is_absolute() else (REPO_ROOT / path).resolve()


def load_metadata(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise ManifestBuildError(f"Metadata file not found: {path}")
    records = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(records, list) or not records:
        raise ManifestBuildError(f"Metadata must contain a non-empty JSON list: {path}")
    return records


def load_prompt_keys(path: Path) -> set[str]:
    if not path.exists():
        raise ManifestBuildError(f"Prompt config not found: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    categories = data.get("categories")
    if not isinstance(categories, dict) or not categories:
        raise ManifestBuildError(f"Prompt config must contain categories: {path}")
    return set(categories)


def build_original_manifest(
    domain: str,
    metadata: list[dict[str, Any]],
    prompt_keys: set[str],
    *,
    strict: bool,
) -> list[dict[str, Any]]:
    by_original: dict[str, dict[str, Any]] = {}
    for raw in metadata:
        validate_raw_record(domain, raw, prompt_keys, strict=strict)
        original_path = resolve_metadata_path(str(raw["original_path"]))
        key = original_path.as_posix()
        if key in by_original:
            continue
        by_original[key] = build_manifest_record(
            domain=domain,
            raw=raw,
            image_path=original_path,
            image_role="original",
            sample_id=sample_id_for(domain, raw, image_path=original_path, image_role="original"),
            pair_id=f"original/{domain}/{subcategory_for(domain, raw)}/{source_variant_for(raw)}/{original_path.stem}",
        )
    records = sorted(by_original.values(), key=lambda item: natural_key(str(item["sample_id"])))
    ensure_unique_sample_ids(records, label=f"{domain} original")
    return records


def build_cf_manifest(
    domain: str,
    metadata: list[dict[str, Any]],
    prompt_keys: set[str],
    *,
    strict: bool,
) -> list[dict[str, Any]]:
    records = []
    for raw in metadata:
        validate_raw_record(domain, raw, prompt_keys, strict=strict)
        cf_path = resolve_metadata_path(str(raw["cf_path"]))
        records.append(
            build_manifest_record(
                domain=domain,
                raw=raw,
                image_path=cf_path,
                image_role="cf",
                sample_id=sample_id_for(domain, raw, image_path=cf_path, image_role="cf"),
                pair_id=pair_id_for(domain, raw),
            )
        )
    records = sorted(records, key=lambda item: natural_key(str(item["sample_id"])))
    ensure_unique_sample_ids(records, label=f"{domain} cf")
    return records


def validate_raw_record(domain: str, raw: dict[str, Any], prompt_keys: set[str], *, strict: bool) -> None:
    required = ["filename", "original_image", "original_path", "cf_path", "model", "mod_id", "mod_type"]
    missing = [field for field in required if field not in raw]
    key_field = DOMAIN_CONFIG[domain]["key_field"]
    if key_field not in raw:
        missing.append(key_field)
    if missing:
        raise ManifestBuildError(f"{domain} metadata record missing fields {missing}: {raw.get('filename')}")
    if raw.get("success") is False:
        raise ManifestBuildError(f"{domain} metadata contains success=false record: {raw.get('filename')}")
    prompt_key = subcategory_for(domain, raw)
    if prompt_key not in prompt_keys:
        raise ManifestBuildError(f"{domain} prompt_key not found in prompt config: {prompt_key}")
    if strict:
        for field in ("original_path", "cf_path"):
            path = resolve_metadata_path(str(raw[field]))
            if not path.exists():
                raise ManifestBuildError(f"{domain} {field} does not exist: {path}")


def validate_manifest_counts(
    domain: str,
    metadata: list[dict[str, Any]],
    original_manifest: list[dict[str, Any]],
    cf_manifest: list[dict[str, Any]],
    *,
    strict: bool,
) -> None:
    if not strict:
        return
    expected_original_count = len({str(resolve_metadata_path(str(raw["original_path"]))) for raw in metadata})
    if len(cf_manifest) != len(metadata):
        raise ManifestBuildError(
            f"{domain} cf manifest count {len(cf_manifest)} != metadata count {len(metadata)}"
        )
    if len(original_manifest) != expected_original_count:
        raise ManifestBuildError(
            f"{domain} original manifest count {len(original_manifest)} != unique metadata originals "
            f"{expected_original_count}"
        )


def build_manifest_record(
    *,
    domain: str,
    raw: dict[str, Any],
    image_path: Path,
    image_role: str,
    sample_id: str,
    pair_id: str,
) -> dict[str, Any]:
    subcategory = subcategory_for(domain, raw)
    return {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "benchmark": domain,
        "sample_id": sample_id,
        "pair_id": pair_id,
        "image_path": path_for_manifest(image_path),
        "image_role": image_role,
        "category": domain,
        "subcategory": subcategory,
        "source_variant": source_variant_for(raw),
        "prompt_key": subcategory,
        "prompt_mode": "object",
        "target_region_key": None,
        "target_part_key": None,
        "domain": domain,
        "mod_id": raw.get("mod_id"),
        "mod_type": raw.get("mod_type"),
        "model_gen": raw.get("model"),
        "cf_filename": raw.get("filename"),
        "original_image": raw.get("original_image"),
    }


def subcategory_for(domain: str, raw: dict[str, Any]) -> str:
    return str(raw[DOMAIN_CONFIG[domain]["key_field"]])


def source_variant_for(raw: dict[str, Any]) -> str:
    original_image = str(raw.get("original_image") or "")
    source_type = str(raw.get("source_type") or "")
    if "_real_" in original_image or source_type.endswith("_real"):
        return "real"
    return "ai"


def sample_id_for(domain: str, raw: dict[str, Any], *, image_path: Path, image_role: str) -> str:
    parts = [
        domain,
        subcategory_for(domain, raw),
        source_variant_for(raw),
        image_path.stem,
        image_role,
    ]
    return sanitize_id("_".join(parts))


def pair_id_for(domain: str, raw: dict[str, Any]) -> str:
    original_stem = Path(str(raw["original_image"])).stem
    return "/".join(
        [
            domain,
            subcategory_for(domain, raw),
            original_stem,
            f"mod{raw['mod_id']}",
            str(raw["model"]),
        ]
    )


def resolve_metadata_path(value: str) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path.resolve(strict=False)
    return (REPO_ROOT / path).resolve(strict=False)


def path_for_manifest(path: Path) -> str:
    try:
        return path.resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def ensure_unique_sample_ids(records: list[dict[str, Any]], *, label: str) -> None:
    seen: dict[str, int] = {}
    duplicates = []
    for record in records:
        sample_id = str(record["sample_id"])
        seen[sample_id] = seen.get(sample_id, 0) + 1
        if seen[sample_id] == 2:
            duplicates.append(sample_id)
    if duplicates:
        raise ManifestBuildError(f"Duplicate {label} sample_id values: {duplicates[:20]}")


def write_jsonl(path: Path, records: list[dict[str, Any]], *, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise ManifestBuildError(f"Output exists; pass --overwrite to replace: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def sanitize_id(value: str) -> str:
    cleaned = []
    for char in value.replace("\\", "/"):
        cleaned.append(char if char.isalnum() or char in {"-", "_"} else "_")
    return "".join(cleaned).strip("_")


def natural_key(value: str) -> list[Any]:
    parts: list[Any] = []
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
    try:
        sys.exit(main())
    except ManifestBuildError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
