#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
import shutil
import sys
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
METADATA_DIR = REPO_ROOT / "eval_results" / "fashion_industry" / "metadata"
DEFAULT_EXCLUSIONS_PATH = METADATA_DIR / "cf_metadata_exclusions.json"
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp"}
GEN_MODELS = ("gemini", "gpt")


class MetadataRebuildError(RuntimeError):
    pass


@dataclass(frozen=True)
class DomainSpec:
    domain: str
    definitions: dict[str, dict[str, Any]]
    dataset_root: Path
    cf_root: Path
    old_metadata_path: Path
    rebuilt_filename: str
    active_filename: str
    key_field: str
    name_field: str
    expected_total: int | None = None


def load_module(path: Path, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise MetadataRebuildError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_domain_specs() -> dict[str, DomainSpec]:
    fashion_module = load_module(
        REPO_ROOT / "gen_code" / "fashion_industry" / "gen_fashion_cf.py",
        "fashion_cf_definitions",
    )
    industry_module = load_module(
        REPO_ROOT / "gen_code" / "fashion_industry" / "gen_industry_cf.py",
        "industry_cf_definitions",
    )
    return {
        "fashion": DomainSpec(
            domain="fashion",
            definitions=fashion_module.FASHION_CF,
            dataset_root=REPO_ROOT / "dataset" / "fashion_dataset",
            cf_root=REPO_ROOT / "cf_dataset" / "fashion_cf",
            old_metadata_path=METADATA_DIR / "fashion_cf_metadata.json",
            rebuilt_filename="fashion_cf_metadata_rebuilt.json",
            active_filename="fashion_cf_metadata.json",
            key_field="brand_key",
            name_field="brand_name",
        ),
        "industry": DomainSpec(
            domain="industry",
            definitions=industry_module.INDUSTRY_CF,
            dataset_root=REPO_ROOT / "dataset" / "industry_dataset",
            cf_root=REPO_ROOT / "cf_dataset" / "industry_cf",
            old_metadata_path=METADATA_DIR / "industry_cf_metadata.json",
            rebuilt_filename="industry_cf_metadata_rebuilt.json",
            active_filename="industry_cf_metadata.json",
            key_field="sub_category",
            name_field="name",
        ),
    }


def load_json_records(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    data = json.loads(path.read_text())
    if not isinstance(data, list):
        raise MetadataRebuildError(f"Expected list metadata: {path}")
    return data


def load_exclusions(path: Path) -> dict[str, set[str]]:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text())
    if isinstance(payload, dict):
        entries = payload.get("exclusions", [])
    elif isinstance(payload, list):
        entries = payload
    else:
        raise MetadataRebuildError(f"Expected exclusion list or object: {path}")
    by_domain: dict[str, set[str]] = {}
    for entry in entries:
        if not isinstance(entry, dict):
            raise MetadataRebuildError(f"Invalid exclusion entry in {path}: {entry!r}")
        domain = entry.get("domain")
        filename = entry.get("filename")
        if not domain or not filename:
            raise MetadataRebuildError(f"Exclusion requires domain and filename: {entry!r}")
        by_domain.setdefault(str(domain), set()).add(str(filename))
    return by_domain


def index_old_records(records: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    counts = Counter(record.get("filename") for record in records)
    duplicates = sorted(filename for filename, count in counts.items() if filename and count > 1)
    if duplicates:
        raise MetadataRebuildError(f"Duplicate filenames in old metadata: {duplicates[:10]}")
    return {record["filename"]: record for record in records}


def image_files(root: Path) -> list[Path]:
    if not root.exists():
        return []
    return sorted(
        path
        for path in root.iterdir()
        if path.is_file() and not path.name.startswith(".") and path.suffix.lower() in IMAGE_SUFFIXES
    )


def relative_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(REPO_ROOT.resolve()).as_posix()
    except ValueError:
        return resolved.as_posix()


def source_type_for(original_name: str) -> str:
    return "real" if "_real_" in original_name else "ai"


def prompt_for_original(original: Path, mod: dict[str, Any]) -> str:
    try:
        original_index = int(original.stem.split("_")[-1]) - 1
    except ValueError as exc:
        raise MetadataRebuildError(f"Cannot infer prompt index from original filename: {original.name}") from exc
    prompts = mod.get("prompts") or []
    if not prompts:
        raise MetadataRebuildError(f"Missing GPT prompts for mod {mod.get('id')}")
    return prompts[original_index % len(prompts)]


def expected_filename(original: Path, mod: dict[str, Any], model_name: str) -> str:
    return f"{original.stem}_cf_mod{mod['id']}_{mod['type']}_{model_name}.png"


def previous_status(filename: str, old_index: dict[str, dict[str, Any]]) -> str:
    old = old_index.get(filename)
    if not old:
        return "missing_from_old_metadata"
    if old.get("success") is False:
        return "failed_but_file_exists"
    return "active_success"


def build_record(
    spec: DomainSpec,
    key: str,
    cfg: dict[str, Any],
    original: Path,
    mod: dict[str, Any],
    model_name: str,
    cf_path: Path,
    old_index: dict[str, dict[str, Any]],
    rebuild_timestamp: str,
) -> dict[str, Any]:
    filename = expected_filename(original, mod, model_name)
    source_type = source_type_for(original.name)
    record: dict[str, Any] = {
        "filename": filename,
        "category": cfg.get("category", spec.domain),
        spec.key_field: key,
        spec.name_field: cfg["name"],
        "source_type": f"cf_from_{source_type}",
        "model": model_name,
        "mod_id": mod["id"],
        "mod_type": mod["type"],
        "mod_desc": mod["desc"],
        "ground_truth": mod["gt"],
        "bias_answer": mod["bias"],
        "original_image": original.name,
        "success": True,
        "timestamp": old_index.get(filename, {}).get("timestamp") or rebuild_timestamp,
        "metadata_rebuilt": True,
        "repair_source": "rebuilt_from_disk_and_generator_definitions",
        "previous_metadata_status": previous_status(filename, old_index),
        "cf_path": relative_path(cf_path),
        "original_path": relative_path(original),
        "rebuild_timestamp": rebuild_timestamp,
    }
    if model_name == "gemini":
        record["edit_prompt"] = mod["edit_prompt"]
    else:
        record["prompt"] = prompt_for_original(original, mod)
    return record


def rebuild_domain(
    spec: DomainSpec,
    *,
    strict: bool = True,
    rebuild_timestamp: str | None = None,
    excluded_filenames: set[str] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rebuild_timestamp = rebuild_timestamp or datetime.now(timezone.utc).isoformat()
    excluded_filenames = excluded_filenames or set()
    old_records = load_json_records(spec.old_metadata_path)
    old_index = index_old_records(old_records)
    old_names = set(old_index)

    disk_cf_paths = {
        path.name: path
        for path in spec.cf_root.glob("*/*")
        if path.is_file() and not path.name.startswith(".") and path.suffix.lower() in IMAGE_SUFFIXES
    }
    expected_names: set[str] = set()
    records: list[dict[str, Any]] = []
    missing_original_dirs: list[str] = []
    missing_originals: list[str] = []
    missing_cf: list[str] = []

    for key, cfg in sorted(spec.definitions.items()):
        original_dir = spec.dataset_root / key
        if not original_dir.exists():
            missing_original_dirs.append(relative_path(original_dir))
            continue
        originals = image_files(original_dir)
        if not originals:
            missing_originals.append(relative_path(original_dir))
            continue
        for original in originals:
            for mod in cfg["mods"]:
                for model_name in GEN_MODELS:
                    filename = expected_filename(original, mod, model_name)
                    expected_names.add(filename)
                    if filename in excluded_filenames:
                        continue
                    cf_path = spec.cf_root / key / filename
                    if not cf_path.exists():
                        missing_cf.append(relative_path(cf_path))
                        continue
                    records.append(
                        build_record(
                            spec=spec,
                            key=key,
                            cfg=cfg,
                            original=original,
                            mod=mod,
                            model_name=model_name,
                            cf_path=cf_path,
                            old_index=old_index,
                            rebuild_timestamp=rebuild_timestamp,
                        )
                    )

    rebuilt_names = [record["filename"] for record in records]
    duplicate_rebuilt = sorted(
        filename for filename, count in Counter(rebuilt_names).items() if count > 1
    )
    disk_extra = sorted(set(disk_cf_paths) - expected_names)
    old_extra = sorted(old_names - expected_names)
    previous_counts = Counter(record["previous_metadata_status"] for record in records)

    report = {
        "domain": spec.domain,
        "old_metadata_path": relative_path(spec.old_metadata_path),
        "old_record_count": len(old_records),
        "old_success_count": sum(1 for record in old_records if record.get("success") is not False),
        "old_failed_count": sum(1 for record in old_records if record.get("success") is False),
        "disk_cf_count": len(disk_cf_paths),
        "expected_count": len(expected_names),
        "rebuilt_count": len(records),
        "excluded_count": len(excluded_filenames),
        "excluded_filenames": sorted(excluded_filenames),
        "expected_total": spec.expected_total,
        "previous_metadata_status_counts": dict(sorted(previous_counts.items())),
        "missing_original_dirs": missing_original_dirs,
        "missing_originals": missing_originals,
        "missing_cf": missing_cf,
        "disk_extra_not_expected": [relative_path(disk_cf_paths[name]) for name in disk_extra],
        "old_metadata_not_expected": old_extra,
        "duplicate_rebuilt_filenames": duplicate_rebuilt,
    }

    strict_errors = []
    if spec.expected_total is not None and len(records) != spec.expected_total:
        strict_errors.append(
            f"{spec.domain}: rebuilt_count={len(records)} expected_total={spec.expected_total}"
        )
    if missing_original_dirs:
        strict_errors.append(f"{spec.domain}: missing original dirs={len(missing_original_dirs)}")
    if missing_originals:
        strict_errors.append(f"{spec.domain}: original dirs without images={len(missing_originals)}")
    if missing_cf:
        strict_errors.append(f"{spec.domain}: missing expected cf files={len(missing_cf)}")
    if disk_extra:
        strict_errors.append(f"{spec.domain}: disk files not expected={len(disk_extra)}")
    if duplicate_rebuilt:
        strict_errors.append(f"{spec.domain}: duplicate rebuilt filenames={len(duplicate_rebuilt)}")
    if strict and strict_errors:
        report["strict_errors"] = strict_errors
        raise MetadataRebuildError("; ".join(strict_errors))

    return sorted(records, key=lambda record: record["filename"]), report


def write_json(path: Path, payload: Any, *, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise MetadataRebuildError(f"Refusing to overwrite existing file without --overwrite: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")


def activate_metadata(
    specs: list[DomainSpec],
    output_dir: Path,
    backup_dir: Path,
    *,
    overwrite: bool,
    timestamp_label: str,
) -> list[dict[str, str]]:
    actions = []
    backup_dir.mkdir(parents=True, exist_ok=True)
    for spec in specs:
        rebuilt_path = output_dir / spec.rebuilt_filename
        active_path = output_dir / spec.active_filename
        if not rebuilt_path.exists():
            raise MetadataRebuildError(f"Cannot activate missing rebuilt metadata: {rebuilt_path}")
        if active_path.exists():
            backup_path = backup_dir / f"{active_path.stem}_{timestamp_label}{active_path.suffix}"
            if backup_path.exists() and not overwrite:
                raise MetadataRebuildError(f"Backup exists and --overwrite was not set: {backup_path}")
            shutil.copy2(active_path, backup_path)
        else:
            backup_path = None
        shutil.copy2(rebuilt_path, active_path)
        actions.append({
            "domain": spec.domain,
            "rebuilt_path": relative_path(rebuilt_path),
            "active_path": relative_path(active_path),
            "backup_path": relative_path(backup_path) if backup_path else "",
        })
    return actions


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rebuild fashion/industry CF metadata from disk files.")
    parser.add_argument("--domain", choices=["fashion", "industry", "both"], default="both")
    parser.add_argument("--output-dir", type=Path, default=METADATA_DIR)
    parser.add_argument("--backup-dir", type=Path, default=METADATA_DIR / "archive_before_rebuild")
    parser.add_argument("--exclusions", type=Path, default=DEFAULT_EXCLUSIONS_PATH)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--activate", action="store_true")
    parser.add_argument("--strict", dest="strict", action="store_true", default=True)
    parser.add_argument("--no-strict", dest="strict", action="store_false")
    return parser.parse_args(argv)


def selected_specs(domain: str, specs: dict[str, DomainSpec]) -> list[DomainSpec]:
    if domain == "both":
        return [specs["fashion"], specs["industry"]]
    return [specs[domain]]


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    specs = selected_specs(args.domain, load_domain_specs())
    output_dir = args.output_dir if args.output_dir.is_absolute() else REPO_ROOT / args.output_dir
    backup_dir = args.backup_dir if args.backup_dir.is_absolute() else REPO_ROOT / args.backup_dir
    exclusions_path = args.exclusions if args.exclusions.is_absolute() else REPO_ROOT / args.exclusions
    exclusions = load_exclusions(exclusions_path)
    timestamp = datetime.now(timezone.utc).isoformat()
    timestamp_label = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    report = {
        "schema_version": "fashion_industry_cf_metadata_rebuild_v1",
        "rebuild_timestamp": timestamp,
        "strict": args.strict,
        "dry_run": args.dry_run,
        "activate": args.activate,
        "exclusions_path": relative_path(exclusions_path) if exclusions_path.exists() else "",
        "domains": {},
        "activation_actions": [],
    }
    outputs: list[tuple[DomainSpec, list[dict[str, Any]]]] = []

    for spec in specs:
        records, domain_report = rebuild_domain(
            spec,
            strict=args.strict,
            rebuild_timestamp=timestamp,
            excluded_filenames=exclusions.get(spec.domain, set()),
        )
        report["domains"][spec.domain] = domain_report
        outputs.append((spec, records))

    for spec, records in outputs:
        print(
            f"{spec.domain}: old={report['domains'][spec.domain]['old_record_count']} "
            f"disk={report['domains'][spec.domain]['disk_cf_count']} rebuilt={len(records)}"
        )
        print(f"  previous_status={report['domains'][spec.domain]['previous_metadata_status_counts']}")
        if args.dry_run:
            print(f"  dry_run_output={output_dir / spec.rebuilt_filename}")
        else:
            write_json(output_dir / spec.rebuilt_filename, records, overwrite=args.overwrite)

    if args.dry_run:
        print("dry_run=true, no files written")
        print(json.dumps(report, indent=2, ensure_ascii=False))
        return 0

    report_path = output_dir / "cf_metadata_rebuild_report.json"
    write_json(report_path, report, overwrite=args.overwrite)

    if args.activate:
        actions = activate_metadata(
            [spec for spec, _records in outputs],
            output_dir,
            backup_dir,
            overwrite=args.overwrite,
            timestamp_label=timestamp_label,
        )
        report["activation_actions"] = actions
        write_json(report_path, report, overwrite=True)
        print(f"activated={len(actions)} backup_dir={backup_dir}")

    print(f"report={report_path}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except MetadataRebuildError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
