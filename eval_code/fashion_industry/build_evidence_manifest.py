#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_METADATA_DIR = REPO_ROOT / "eval_results" / "fashion_industry" / "metadata"
SCHEMA_VERSION = "fashion_industry_sam3_object_evidence_v1"
TOOL_CONDITIONS = ("raw", "bbox", "crop", "zoom_panel")
DOMAIN_CONFIG = {
    "fashion": {
        "metadata": "fashion_cf_metadata.json",
        "original_manifest": "fashion_original_vision_manifest.jsonl",
        "cf_manifest": "fashion_cf_vision_manifest.jsonl",
        "output_json": "fashion_sam3_object_evidence_manifest_pad005_max3_merge5_fallback.json",
        "output_csv": "fashion_sam3_object_evidence_manifest_pad005_max3_merge5_fallback.csv",
        "key_field": "brand_key",
    },
    "industry": {
        "metadata": "industry_cf_metadata.json",
        "original_manifest": "industry_original_vision_manifest.jsonl",
        "cf_manifest": "industry_cf_vision_manifest.jsonl",
        "output_json": "industry_sam3_object_evidence_manifest_pad005_max3_merge5_fallback.json",
        "output_csv": "industry_sam3_object_evidence_manifest_pad005_max3_merge5_fallback.csv",
        "key_field": "sub_category",
    },
}


class EvidenceManifestError(RuntimeError):
    pass


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build selected SAM evidence manifests for fashion/industry.")
    parser.add_argument("--domain", choices=["fashion", "industry", "both"], default="both")
    parser.add_argument("--metadata-dir", type=Path, default=DEFAULT_METADATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_METADATA_DIR)
    parser.add_argument("--fashion-original-root", type=Path, default=None)
    parser.add_argument("--fashion-cf-root", type=Path, default=None)
    parser.add_argument("--industry-original-root", type=Path, default=None)
    parser.add_argument("--industry-cf-root", type=Path, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    metadata_dir = resolve_path(args.metadata_dir)
    output_dir = resolve_path(args.output_dir)
    domains = ["fashion", "industry"] if args.domain == "both" else [args.domain]
    outputs: list[tuple[str, Path, Path, list[dict[str, Any]]]] = []

    for domain in domains:
        original_root = resolve_domain_root(args, domain, "original")
        cf_root = resolve_domain_root(args, domain, "cf")
        records = build_domain_manifest(
            domain=domain,
            metadata_dir=metadata_dir,
            original_root=original_root,
            cf_root=cf_root,
        )
        json_path = output_dir / DOMAIN_CONFIG[domain]["output_json"]
        csv_path = output_dir / DOMAIN_CONFIG[domain]["output_csv"]
        outputs.append((domain, json_path, csv_path, records))
        print(f"{domain}: records={len(records)} output_json={json_path}")
        print("qc_status_counts=" + json.dumps(count_by(records, "qc_status"), sort_keys=True))
        print("cf_only_usable_crop=" + str(sum(1 for row in records if row["usable_tool_conditions_by_input_mode"]["cf_only"]["crop"])))
        print("both_usable_crop=" + str(sum(1 for row in records if row["usable_tool_conditions_by_input_mode"]["both"]["crop"])))
        if records:
            print("preview=" + json.dumps(records[0], ensure_ascii=False))

    if args.dry_run:
        print("dry_run=true, no files written")
        return 0

    output_dir.mkdir(parents=True, exist_ok=True)
    for _domain, json_path, csv_path, records in outputs:
        for path in (json_path, csv_path):
            if path.exists() and not args.overwrite:
                raise EvidenceManifestError(f"Output exists; pass --overwrite to replace: {path}")
        json_path.write_text(json.dumps(records, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        write_csv(csv_path, records)
    return 0


def resolve_path(path: Path) -> Path:
    path = path.expanduser()
    return path if path.is_absolute() else (REPO_ROOT / path).resolve()


def resolve_domain_root(args: argparse.Namespace, domain: str, role: str) -> Path:
    value = getattr(args, f"{domain}_{role}_root")
    if value is None:
        raise EvidenceManifestError(f"--{domain}-{role}-root is required for --domain {domain} or both.")
    root = resolve_path(value)
    if not root.exists():
        raise EvidenceManifestError(f"Evidence root not found: {root}")
    return root


def build_domain_manifest(
    *,
    domain: str,
    metadata_dir: Path,
    original_root: Path,
    cf_root: Path,
) -> list[dict[str, Any]]:
    config = DOMAIN_CONFIG[domain]
    metadata = load_json_list(metadata_dir / config["metadata"])
    original_manifest = index_vision_manifest(
        metadata_dir / config["original_manifest"],
        key_fields=("subcategory", "original_image"),
    )
    cf_manifest = index_vision_manifest(
        metadata_dir / config["cf_manifest"],
        key_fields=("subcategory", "cf_filename"),
    )
    original_evidence = load_sam_metadata_index(original_root)
    cf_evidence = load_sam_metadata_index(cf_root)

    records: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str, int, str]] = set()
    for raw in metadata:
        if raw.get("success") is False:
            continue
        key = str(raw[config["key_field"]])
        cf_filename = str(raw["filename"])
        original_image = str(raw["original_image"])
        model_gen = str(raw.get("model") or "")
        mod_id = int(raw["mod_id"])
        source_variant = source_variant_for(raw)
        dedupe_key = (domain, key, original_image, mod_id, model_gen)
        if (domain, key, cf_filename, mod_id, model_gen) in seen:
            raise EvidenceManifestError(f"Duplicate metadata record for {domain}/{key}/{cf_filename}")
        seen.add((domain, key, cf_filename, mod_id, model_gen))

        original_vm = original_manifest.get((key, original_image))
        cf_vm = cf_manifest.get((key, cf_filename))
        if original_vm is None:
            raise EvidenceManifestError(f"Missing original vision manifest row for {domain}/{key}/{original_image}")
        if cf_vm is None:
            raise EvidenceManifestError(f"Missing CF vision manifest row for {domain}/{key}/{cf_filename}")

        original_payload = build_role_payload(
            vision_record=original_vm,
            evidence_metadata=original_evidence.get(str(original_vm["sample_id"])),
        )
        cf_payload = build_role_payload(
            vision_record=cf_vm,
            evidence_metadata=cf_evidence.get(str(cf_vm["sample_id"])),
        )
        qc_status = combine_qc_status(original_payload["qc_status"], cf_payload["qc_status"])
        usable_by_input_mode = usable_tool_conditions_by_input_mode(original_payload, cf_payload)
        records.append(
            {
                "schema_version": SCHEMA_VERSION,
                "benchmark": "fashion_industry",
                "domain": domain,
                "key": key,
                "subcategory": key,
                "source_variant": source_variant,
                "pair_key": pair_key_for(domain, key, original_image, mod_id, model_gen),
                "dedupe_key": "/".join(map(str, dedupe_key)),
                "original_image": original_image,
                "cf_filename": cf_filename,
                "mod_id": mod_id,
                "mod_type": raw.get("mod_type"),
                "model_gen": model_gen,
                "generator": generator_summary(original_payload, cf_payload),
                "qc_status": qc_status,
                "usable_tool_conditions": usable_for_roles(original_payload, cf_payload),
                "usable_tool_conditions_by_input_mode": usable_by_input_mode,
                "original": original_payload,
                "cf": cf_payload,
            }
        )
    return sorted(records, key=lambda row: natural_key(row["pair_key"]))


def load_json_list(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise EvidenceManifestError(f"Metadata file not found: {path}")
    records = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(records, list):
        raise EvidenceManifestError(f"Expected JSON list: {path}")
    return records


def index_vision_manifest(path: Path, *, key_fields: tuple[str, str]) -> dict[tuple[str, str], dict[str, Any]]:
    if not path.exists():
        raise EvidenceManifestError(f"Vision manifest not found: {path}")
    index: dict[tuple[str, str], dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            key = tuple(str(row.get(field) or "") for field in key_fields)
            if not all(key):
                raise EvidenceManifestError(f"Missing {key_fields} in {path}:{line_number}")
            if key in index:
                raise EvidenceManifestError(f"Duplicate vision manifest key {key} in {path}")
            index[key] = row
    return index


def load_sam_metadata_index(root: Path) -> dict[str, dict[str, Any]]:
    index: dict[str, dict[str, Any]] = {}
    duplicates: list[str] = []
    for metadata_path in sorted(root.rglob("metadata.json")):
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        sample_id = str(metadata.get("sample_id") or "")
        if not sample_id:
            continue
        if sample_id in index:
            duplicates.append(sample_id)
            continue
        metadata["_metadata_path"] = str(metadata_path)
        metadata["_sample_dir"] = str(metadata_path.parent)
        instances_path = metadata_path.parent / "instances.json"
        metadata["_instances_path"] = str(instances_path) if instances_path.exists() else None
        index[sample_id] = metadata
    if not index:
        raise EvidenceManifestError(f"No SAM metadata.json files found under {root}")
    if duplicates:
        raise EvidenceManifestError(f"Duplicate SAM sample_id values under {root}: {duplicates[:20]}")
    return index


def build_role_payload(*, vision_record: dict[str, Any], evidence_metadata: dict[str, Any] | None) -> dict[str, Any]:
    sample_id = str(vision_record["sample_id"])
    raw_path = path_for_output(vision_record["image_path"])
    if evidence_metadata is None:
        return empty_role_payload(sample_id=sample_id, raw=raw_path, status="missing_metadata")
    sample_dir = Path(str(evidence_metadata["_sample_dir"]))
    metadata_path = Path(str(evidence_metadata["_metadata_path"]))
    status = str(evidence_metadata.get("status") or "missing")
    if status != "success":
        return empty_role_payload(
            sample_id=sample_id,
            raw=raw_path,
            status=status,
            metadata=evidence_metadata,
            metadata_path=metadata_path,
        )
    instances_path = Path(str(evidence_metadata.get("_instances_path") or ""))
    if not instances_path.exists():
        return empty_role_payload(
            sample_id=sample_id,
            raw=raw_path,
            status="missing_instances",
            metadata=evidence_metadata,
            metadata_path=metadata_path,
        )
    instances_doc = json.loads(instances_path.read_text(encoding="utf-8"))
    instances = instances_doc.get("instances") or []
    if not instances:
        return empty_role_payload(
            sample_id=sample_id,
            raw=raw_path,
            status="empty",
            metadata=evidence_metadata,
            metadata_path=metadata_path,
            instances_path=instances_path,
        )
    selected = select_instance(instances)
    selection_method = selection_method_for(instances, instances_merged=bool(evidence_metadata.get("instances_merged")))
    selected_instance_id = int(selected.get("instance_id", 0))
    files = {
        "bbox_overlay": sample_dir / selected["bbox_overlay_file"] if selected.get("bbox_overlay_file") else None,
        "crop": sample_dir / selected["crop_file"] if selected.get("crop_file") else None,
        "zoom_panel": sample_dir / selected["zoom_panel_file"] if selected.get("zoom_panel_file") else None,
    }
    missing_files = [name for name, value in files.items() if value is None or not value.exists()]
    num_saved = int(evidence_metadata.get("num_instances_saved") or evidence_metadata.get("num_instances") or len(instances))
    instances_merged = bool(evidence_metadata.get("instances_merged"))
    qc_status = "empty" if missing_files else ("review" if instances_merged or num_saved > 1 else "pass")
    return {
        "sample_id": sample_id,
        "raw": raw_path,
        "metadata_path": path_for_output(metadata_path),
        "instances_path": path_for_output(instances_path),
        "status": status,
        "qc_status": qc_status,
        "bbox_overlay": path_for_output(files["bbox_overlay"]) if files["bbox_overlay"] else None,
        "crop": path_for_output(files["crop"]) if files["crop"] else None,
        "zoom_panel": path_for_output(files["zoom_panel"]) if files["zoom_panel"] else None,
        "selected_instance_id": selected_instance_id,
        "instance_selection_method": selection_method,
        "instances_merged": instances_merged,
        "num_instances_detected": int(evidence_metadata.get("num_instances_detected") or evidence_metadata.get("num_instances") or len(instances)),
        "num_instances_saved": num_saved,
        "selected_box_area_ratio": selected.get("box_area_ratio"),
        "selected_mask_area_ratio": selected.get("mask_area_ratio"),
        "used_fallback_prompt": bool(evidence_metadata.get("used_fallback_prompt")),
        "selected_prompt_text": evidence_metadata.get("selected_prompt_text") or evidence_metadata.get("prompt_text"),
        "evidence_available": {
            "raw": bool(raw_path),
            "bbox": "bbox_overlay" not in missing_files,
            "crop": "crop" not in missing_files,
            "zoom_panel": "zoom_panel" not in missing_files,
        },
    }


def empty_role_payload(
    *,
    sample_id: str,
    raw: str | None,
    status: str,
    metadata: dict[str, Any] | None = None,
    metadata_path: Path | None = None,
    instances_path: Path | None = None,
) -> dict[str, Any]:
    return {
        "sample_id": sample_id,
        "raw": raw,
        "metadata_path": path_for_output(metadata_path) if metadata_path else None,
        "instances_path": path_for_output(instances_path) if instances_path else None,
        "status": status,
        "qc_status": "empty",
        "bbox_overlay": None,
        "crop": None,
        "zoom_panel": None,
        "selected_instance_id": None,
        "instance_selection_method": None,
        "instances_merged": bool(metadata.get("instances_merged")) if metadata else False,
        "num_instances_detected": int(metadata.get("num_instances_detected") or metadata.get("num_instances") or 0) if metadata else 0,
        "num_instances_saved": int(metadata.get("num_instances_saved") or metadata.get("num_instances") or 0) if metadata else 0,
        "selected_box_area_ratio": None,
        "selected_mask_area_ratio": None,
        "used_fallback_prompt": bool(metadata.get("used_fallback_prompt")) if metadata else False,
        "selected_prompt_text": (metadata.get("selected_prompt_text") or metadata.get("prompt_text")) if metadata else None,
        "evidence_available": {"raw": bool(raw), "bbox": False, "crop": False, "zoom_panel": False},
    }


def select_instance(instances: list[dict[str, Any]]) -> dict[str, Any]:
    if len(instances) == 1:
        return instances[0]
    with_mask = [item for item in instances if item.get("mask_area_ratio") is not None]
    if with_mask:
        return max(with_mask, key=lambda item: float(item.get("mask_area_ratio") or 0.0))
    return max(instances, key=lambda item: float(item.get("box_area_ratio") or 0.0))


def selection_method_for(instances: list[dict[str, Any]], *, instances_merged: bool = False) -> str:
    if instances_merged:
        return "merged_instances_over_threshold"
    if len(instances) == 1:
        return "single_instance"
    if any(item.get("mask_area_ratio") is not None for item in instances):
        return "largest_mask_area_ratio"
    return "largest_box_area_ratio"


def combine_qc_status(*statuses: str) -> str:
    if any(status == "empty" for status in statuses):
        return "empty"
    if any(status == "review" for status in statuses):
        return "review"
    return "pass"


def usable_for_roles(*roles: dict[str, Any]) -> dict[str, bool]:
    usable = {"raw": True}
    for condition in ("bbox", "crop", "zoom_panel"):
        usable[condition] = all(bool((role.get("evidence_available") or {}).get(condition)) for role in roles)
    return usable


def usable_tool_conditions_by_input_mode(original: dict[str, Any], cf: dict[str, Any]) -> dict[str, dict[str, bool]]:
    return {
        "cf_only": usable_for_roles(cf),
        "both": usable_for_roles(original, cf),
        "orig_only": usable_for_roles(original),
    }


def generator_summary(*payloads: dict[str, Any]) -> dict[str, Any]:
    for payload in payloads:
        metadata_path = payload.get("metadata_path")
        if not metadata_path:
            continue
        path = resolve_maybe_repo_path(str(metadata_path))
        if not path.exists():
            continue
        metadata = json.loads(path.read_text(encoding="utf-8"))
        return {
            "model_id": metadata.get("model_id"),
            "backend": metadata.get("backend"),
            "backend_resolved": metadata.get("backend_resolved"),
            "dtype": metadata.get("dtype"),
            "crop_padding": metadata.get("crop_padding"),
            "max_instances": metadata.get("max_instances"),
            "merge_instances_over": metadata.get("merge_instances_over"),
            "prompt_profile": metadata.get("prompt_profile"),
            "schema_version": metadata.get("schema_version"),
        }
    return {
        "model_id": None,
        "backend": None,
        "backend_resolved": None,
        "dtype": None,
        "crop_padding": None,
        "max_instances": None,
        "merge_instances_over": None,
        "prompt_profile": "object",
        "schema_version": None,
    }


def source_variant_for(raw: dict[str, Any]) -> str:
    original_image = str(raw.get("original_image") or "")
    source_type = str(raw.get("source_type") or "")
    if "_real_" in original_image or source_type.endswith("_real"):
        return "real"
    return "ai"


def pair_key_for(domain: str, key: str, original_image: str, mod_id: int, model_gen: str) -> str:
    return "/".join([domain, key, Path(original_image).stem, f"mod{mod_id}", model_gen])


def path_for_output(value: Any) -> str | None:
    if value is None:
        return None
    raw = str(value)
    for marker in (
        "vision_dataset/fashion/",
        "vision_dataset/industry/",
        "eval_results/fashion_industry/",
        "dataset/fashion_dataset/",
        "dataset/industry_dataset/",
        "cf_dataset/fashion_cf/",
        "cf_dataset/industry_cf/",
    ):
        marker_index = raw.find(marker)
        if marker_index != -1:
            return raw[marker_index:]
    path = Path(raw)
    if not path.is_absolute():
        return path.as_posix()
    try:
        return path.resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def resolve_maybe_repo_path(value: str) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    return (REPO_ROOT / path).resolve(strict=False)


def count_by(records: list[dict[str, Any]], field: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for record in records:
        key = str(record.get(field) or "")
        counts[key] = counts.get(key, 0) + 1
    return counts


def write_csv(path: Path, records: list[dict[str, Any]]) -> None:
    fieldnames = [
        "domain",
        "key",
        "source_variant",
        "pair_key",
        "original_image",
        "cf_filename",
        "mod_id",
        "mod_type",
        "model_gen",
        "qc_status",
        "usable_cf_only_bbox",
        "usable_cf_only_crop",
        "usable_cf_only_zoom_panel",
        "usable_both_bbox",
        "usable_both_crop",
        "usable_both_zoom_panel",
        "original_qc_status",
        "cf_qc_status",
        "original_selected_instance_id",
        "cf_selected_instance_id",
        "original_instances_merged",
        "cf_instances_merged",
        "original_num_instances_saved",
        "cf_num_instances_saved",
        "original_bbox_overlay",
        "cf_bbox_overlay",
        "original_crop",
        "cf_crop",
        "original_zoom_panel",
        "cf_zoom_panel",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            usable = record.get("usable_tool_conditions_by_input_mode") or {}
            cf_only = usable.get("cf_only") or {}
            both = usable.get("both") or {}
            writer.writerow(
                {
                    "domain": record.get("domain"),
                    "key": record.get("key"),
                    "source_variant": record.get("source_variant"),
                    "pair_key": record.get("pair_key"),
                    "original_image": record.get("original_image"),
                    "cf_filename": record.get("cf_filename"),
                    "mod_id": record.get("mod_id"),
                    "mod_type": record.get("mod_type"),
                    "model_gen": record.get("model_gen"),
                    "qc_status": record.get("qc_status"),
                    "usable_cf_only_bbox": cf_only.get("bbox"),
                    "usable_cf_only_crop": cf_only.get("crop"),
                    "usable_cf_only_zoom_panel": cf_only.get("zoom_panel"),
                    "usable_both_bbox": both.get("bbox"),
                    "usable_both_crop": both.get("crop"),
                    "usable_both_zoom_panel": both.get("zoom_panel"),
                    "original_qc_status": record["original"].get("qc_status"),
                    "cf_qc_status": record["cf"].get("qc_status"),
                    "original_selected_instance_id": record["original"].get("selected_instance_id"),
                    "cf_selected_instance_id": record["cf"].get("selected_instance_id"),
                    "original_instances_merged": record["original"].get("instances_merged"),
                    "cf_instances_merged": record["cf"].get("instances_merged"),
                    "original_num_instances_saved": record["original"].get("num_instances_saved"),
                    "cf_num_instances_saved": record["cf"].get("num_instances_saved"),
                    "original_bbox_overlay": record["original"].get("bbox_overlay"),
                    "cf_bbox_overlay": record["cf"].get("bbox_overlay"),
                    "original_crop": record["original"].get("crop"),
                    "cf_crop": record["cf"].get("crop"),
                    "original_zoom_panel": record["original"].get("zoom_panel"),
                    "cf_zoom_panel": record["cf"].get("zoom_panel"),
                }
            )


def natural_key(value: str) -> list[Any]:
    import re

    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", value)]


if __name__ == "__main__":
    try:
        sys.exit(main())
    except EvidenceManifestError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
