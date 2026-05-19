#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from build_vision_manifest import rebase_counting_path, sample_id_for


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ANNOTATIONS = REPO_ROOT / "eval_results" / "counting" / "metadata" / "counting_count_annotations.json"
DEFAULT_OUTPUT_JSON = REPO_ROOT / "eval_results" / "counting" / "metadata" / "counting_sam3_object_evidence_manifest.json"
DEFAULT_OUTPUT_CSV = REPO_ROOT / "eval_results" / "counting" / "metadata" / "counting_sam3_object_evidence_manifest.csv"
SCHEMA_VERSION = "counting_sam3_object_evidence_v1"
REQUIRED_INSTANCE_FILES = {
    "bbox_overlay": "bbox_overlay_file",
    "crop": "crop_file",
    "zoom_panel": "zoom_panel_file",
    "outline_overlay": "outline_overlay_file",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the selected object-only SAM evidence manifest for counting.")
    parser.add_argument("--annotations", type=Path, default=DEFAULT_ANNOTATIONS)
    parser.add_argument("--primary-original-root", type=Path, required=True)
    parser.add_argument("--primary-cf-root", type=Path, required=True)
    parser.add_argument("--fallback-original-root", type=Path, default=None)
    parser.add_argument("--fallback-cf-root", type=Path, default=None)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_OUTPUT_CSV)
    parser.add_argument("--min-selected-area-ratio", type=float, default=0.01)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    annotations = load_annotations(resolve_path(args.annotations))
    primary_original = load_evidence_index(resolve_path(args.primary_original_root), required=True)
    primary_cf = load_evidence_index(resolve_path(args.primary_cf_root), required=True)
    fallback_original = load_evidence_index(resolve_path(args.fallback_original_root), required=False) if args.fallback_original_root else {}
    fallback_cf = load_evidence_index(resolve_path(args.fallback_cf_root), required=False) if args.fallback_cf_root else {}
    records: List[Dict[str, Any]] = []
    for annotation in annotations:
        original_path = rebase_counting_path(str(annotation["original_path"]))
        cf_path = rebase_counting_path(str(annotation["cf_path"]))
        original_sample_id = sample_id_for(annotation, image_path=original_path, image_role="original", include_edit=False)
        cf_sample_id = sample_id_for(annotation, image_path=cf_path, image_role="cf", include_edit=True)
        original_payload = select_role_payload(
            sample_id=original_sample_id,
            primary=primary_original.get(original_sample_id),
            fallback=fallback_original.get(original_sample_id),
            min_selected_area_ratio=args.min_selected_area_ratio,
        )
        cf_payload = select_role_payload(
            sample_id=cf_sample_id,
            primary=primary_cf.get(cf_sample_id),
            fallback=fallback_cf.get(cf_sample_id),
            min_selected_area_ratio=args.min_selected_area_ratio,
        )
        qc_status = combine_qc_status(original_payload["qc_status"], cf_payload["qc_status"])
        generator = generator_summary(original_payload, cf_payload)
        usable_tool_conditions = combine_usable_tool_conditions(original_payload, cf_payload, qc_status=qc_status)
        usable_by_input_mode = usable_tool_conditions_by_input_mode(
            original_payload,
            cf_payload,
            qc_status=qc_status,
        )
        records.append(
            {
                "schema_version": SCHEMA_VERSION,
                "benchmark": "counting",
                "pair_key": annotation.get("pair_key"),
                "question_group_id": annotation.get("question_group_id"),
                "subset": annotation.get("subset"),
                "subcategory": annotation.get("subcategory"),
                "source_variant": annotation.get("source_variant"),
                "edit_type": annotation.get("edit_type"),
                "count_direction": annotation.get("count_direction"),
                "normal_count": annotation.get("normal_count"),
                "cf_count": annotation.get("cf_count"),
                "biased_count": annotation.get("biased_count"),
                "generator": generator,
                "qc_status": qc_status,
                "usable_tool_conditions": usable_tool_conditions,
                "usable_tool_conditions_by_input_mode": usable_by_input_mode,
                "original": original_payload,
                "cf": cf_payload,
            }
        )

    counts = count_statuses(records)
    print(f"annotations={len(annotations)}")
    print(f"records={len(records)}")
    print("qc_status_counts=" + json.dumps(counts, sort_keys=True))
    if records:
        print("preview=" + json.dumps(records[0], ensure_ascii=False))
    output_json = resolve_path(args.output_json)
    output_csv = resolve_path(args.output_csv)
    print(f"output_json={output_json}")
    print(f"output_csv={output_csv}")
    if args.dry_run:
        return 0
    for path in (output_json, output_csv):
        if path.exists() and not args.overwrite:
            raise SystemExit(f"Output exists; pass --overwrite to replace: {path}")
        path.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(records, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    write_csv(output_csv, records)
    return 0


def resolve_path(path: Optional[Path]) -> Path:
    if path is None:
        raise ValueError("path must not be None")
    path = path.expanduser()
    return path if path.is_absolute() else (REPO_ROOT / path).resolve()


def load_annotations(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        raise SystemExit(f"Counting annotation file not found: {path}")
    records = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(records, list) or not records:
        raise SystemExit("Counting annotation file must contain a non-empty JSON list.")
    return records


def load_evidence_index(root: Path, *, required: bool) -> Dict[str, Dict[str, Any]]:
    if not root.exists():
        if required:
            raise SystemExit(f"Evidence root not found: {root}")
        return {}
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
        metadata["_sample_dir"] = str(metadata_path.parent)
        index[sample_id] = metadata
    if required and not index:
        raise SystemExit(f"No metadata.json files found under evidence root: {root}")
    if duplicates:
        raise SystemExit(f"Duplicate sample_id values in evidence metadata: {duplicates[:20]}")
    return index


def select_role_payload(
    *,
    sample_id: str,
    primary: Optional[Dict[str, Any]],
    fallback: Optional[Dict[str, Any]],
    min_selected_area_ratio: float,
) -> Dict[str, Any]:
    source = "missing"
    metadata = None
    if primary and primary.get("status") == "success":
        source = "primary"
        metadata = primary
    elif fallback and fallback.get("status") == "success":
        source = "fallback"
        metadata = fallback
    elif primary:
        source = "primary"
        metadata = primary
    elif fallback:
        source = "fallback"
        metadata = fallback

    if metadata is None:
        return empty_payload(sample_id=sample_id, evidence_source=source, status="missing")
    metadata_path = Path(str(metadata["_metadata_path"]))
    sample_dir = Path(str(metadata["_sample_dir"]))
    if metadata.get("status") != "success":
        return empty_payload(
            sample_id=sample_id,
            evidence_source=source,
            status=str(metadata.get("status") or "missing"),
            metadata=metadata,
            metadata_path=metadata_path,
        )
    instances_path = sample_dir / "instances.json"
    if not instances_path.exists():
        return empty_payload(
            sample_id=sample_id,
            evidence_source=source,
            status="missing_instances",
            metadata=metadata,
            metadata_path=metadata_path,
        )
    instances_doc = json.loads(instances_path.read_text(encoding="utf-8"))
    instances = instances_doc.get("instances") or []
    if not instances:
        return empty_payload(
            sample_id=sample_id,
            evidence_source=source,
            status="empty",
            metadata=metadata,
            metadata_path=metadata_path,
            instances_path=instances_path,
        )
    selected, method = select_instance(instances)
    selected_instance_id = int(selected.get("instance_id", 0))
    selected_files = selected_evidence_files(sample_dir, selected)
    files_exist = all(path and Path(path).exists() for path in selected_files.values())
    selected_mask_area = selected.get("mask_area_ratio")
    selected_box_area = selected.get("box_area_ratio")
    area_for_review = selected_mask_area if selected_mask_area is not None else selected_box_area
    qc_status = "pass"
    if not files_exist:
        qc_status = "empty"
    elif len(instances) > 1 or (area_for_review is not None and float(area_for_review) < min_selected_area_ratio):
        qc_status = "review"
    return {
        "sample_id": sample_id,
        "raw": path_for_output(metadata.get("source_image_path")),
        "metadata_path": path_for_output(metadata_path),
        "instances_path": path_for_output(instances_path),
        "status": metadata.get("status"),
        "evidence_source": source,
        "bbox_overlay": path_for_output(selected_files["bbox_overlay"]),
        "crop": path_for_output(selected_files["crop"]),
        "zoom_panel": path_for_output(selected_files["zoom_panel"]),
        "outline_overlay": path_for_output(selected_files["outline_overlay"]),
        "selected_instance_id": selected_instance_id,
        "instance_selection_method": method,
        "num_instances_total": len(instances),
        "selected_box_area_ratio": selected_box_area,
        "selected_mask_area_ratio": selected_mask_area,
        "used_fallback_prompt": bool(metadata.get("used_fallback_prompt")),
        "selected_prompt_text": metadata.get("selected_prompt_text") or metadata.get("prompt_text"),
        "prompt_text": metadata.get("prompt_text"),
        "evidence_available": {
            "raw": bool(metadata.get("source_image_path")),
            "bbox": selected_files["bbox_overlay"] is not None and Path(selected_files["bbox_overlay"]).exists(),
            "crop": selected_files["crop"] is not None and Path(selected_files["crop"]).exists(),
            "zoom_panel": selected_files["zoom_panel"] is not None and Path(selected_files["zoom_panel"]).exists(),
            "outline_overlay": selected_files["outline_overlay"] is not None and Path(selected_files["outline_overlay"]).exists(),
            "outline": selected_files["outline_overlay"] is not None and Path(selected_files["outline_overlay"]).exists(),
            "tool_bundle": all(
                selected_files[key] is not None and Path(selected_files[key]).exists()
                for key in ("bbox_overlay", "crop", "zoom_panel", "outline_overlay")
            ),
        },
        "qc_status": qc_status,
    }


def empty_payload(
    *,
    sample_id: str,
    evidence_source: str,
    status: str,
    metadata: Optional[Dict[str, Any]] = None,
    metadata_path: Optional[Path] = None,
    instances_path: Optional[Path] = None,
) -> Dict[str, Any]:
    return {
        "sample_id": sample_id,
        "raw": path_for_output(metadata.get("source_image_path")) if metadata else None,
        "metadata_path": path_for_output(metadata_path) if metadata_path else None,
        "instances_path": path_for_output(instances_path) if instances_path else None,
        "status": status,
        "evidence_source": evidence_source,
        "bbox_overlay": None,
        "crop": None,
        "zoom_panel": None,
        "outline_overlay": None,
        "selected_instance_id": None,
        "instance_selection_method": None,
        "num_instances_total": int(metadata.get("num_instances") or 0) if metadata else 0,
        "selected_box_area_ratio": None,
        "selected_mask_area_ratio": None,
        "used_fallback_prompt": bool(metadata.get("used_fallback_prompt")) if metadata else False,
        "selected_prompt_text": (metadata.get("selected_prompt_text") or metadata.get("prompt_text")) if metadata else None,
        "prompt_text": metadata.get("prompt_text") if metadata else None,
        "evidence_available": {
            "raw": bool(metadata.get("source_image_path")) if metadata else False,
            "bbox": False,
            "crop": False,
            "zoom_panel": False,
            "outline_overlay": False,
            "outline": False,
            "tool_bundle": False,
        },
        "qc_status": "empty",
    }


def combine_usable_tool_conditions(original: Dict[str, Any], cf: Dict[str, Any], *, qc_status: str) -> Dict[str, bool]:
    original_available = original.get("evidence_available") or {}
    cf_available = cf.get("evidence_available") or {}
    usable = {"raw": True}
    for condition in ("bbox", "crop", "zoom_panel", "tool_bundle", "outline"):
        usable[condition] = (
            qc_status != "empty"
            and bool(original_available.get(condition))
            and bool(cf_available.get(condition))
        )
    usable["outline_overlay"] = (
        qc_status != "empty"
        and bool(original_available.get("outline_overlay"))
        and bool(cf_available.get("outline_overlay"))
    )
    return usable


def usable_tool_conditions_for_roles(*roles: Dict[str, Any], qc_status: str) -> Dict[str, bool]:
    usable = {"raw": True}
    for condition in ("bbox", "crop", "zoom_panel", "outline", "tool_bundle"):
        usable[condition] = qc_status != "empty" and all(
            bool((role.get("evidence_available") or {}).get(condition)) for role in roles
        )
    usable["outline_overlay"] = usable["outline"]
    return usable


def usable_tool_conditions_by_input_mode(original: Dict[str, Any], cf: Dict[str, Any], *, qc_status: str) -> Dict[str, Dict[str, bool]]:
    return {
        "cf_only": usable_tool_conditions_for_roles(cf, qc_status=cf.get("qc_status", qc_status)),
        "both": usable_tool_conditions_for_roles(original, cf, qc_status=qc_status),
        "orig_only": usable_tool_conditions_for_roles(original, qc_status=original.get("qc_status", qc_status)),
    }


def select_instance(instances: List[Dict[str, Any]]) -> Tuple[Dict[str, Any], str]:
    if len(instances) == 1:
        return instances[0], "single_instance"
    with_mask = [item for item in instances if item.get("mask_area_ratio") is not None]
    if with_mask:
        return max(with_mask, key=lambda item: float(item.get("mask_area_ratio") or 0.0)), "largest_mask_area_ratio"
    return max(instances, key=lambda item: float(item.get("box_area_ratio") or 0.0)), "largest_box_area_ratio"


def selected_evidence_files(sample_dir: Path, instance: Dict[str, Any]) -> Dict[str, Optional[Path]]:
    output: Dict[str, Optional[Path]] = {}
    for key, field in REQUIRED_INSTANCE_FILES.items():
        rel = instance.get(field)
        output[key] = sample_dir / rel if rel else None
    return output


def combine_qc_status(*statuses: str) -> str:
    if any(status == "empty" for status in statuses):
        return "empty"
    if any(status == "review" for status in statuses):
        return "review"
    return "pass"


def generator_summary(*payloads: Dict[str, Any]) -> Dict[str, Any]:
    for payload in payloads:
        metadata_path = payload.get("metadata_path")
        if not metadata_path:
            continue
        metadata_file = resolve_maybe_repo_path(str(metadata_path))
        if not metadata_file.exists():
            continue
        metadata = json.loads(metadata_file.read_text(encoding="utf-8"))
        return {
            "model_id": metadata.get("model_id"),
            "backend": metadata.get("backend"),
            "backend_resolved": metadata.get("backend_resolved"),
            "dtype": metadata.get("dtype"),
            "crop_padding": metadata.get("crop_padding"),
            "prompt_profile": metadata.get("prompt_profile"),
            "schema_version": metadata.get("schema_version"),
        }
    return {
        "model_id": None,
        "backend": None,
        "backend_resolved": None,
        "dtype": None,
        "crop_padding": None,
        "prompt_profile": "object",
        "schema_version": None,
    }


def resolve_maybe_repo_path(value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return (REPO_ROOT / path).resolve()


def path_for_output(value: Any) -> Optional[str]:
    if value is None:
        return None
    raw = str(value)
    for marker in ("vision_dataset/counting/", "eval_results/counting/", "cf_dataset/counting_cf/", "dataset/counting/"):
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


def count_statuses(records: List[Dict[str, Any]]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for record in records:
        status = str(record["qc_status"])
        counts[status] = counts.get(status, 0) + 1
    return counts


def write_csv(path: Path, records: List[Dict[str, Any]]) -> None:
    fieldnames = [
        "pair_key",
        "subset",
        "subcategory",
        "source_variant",
        "edit_type",
        "count_direction",
        "qc_status",
        "usable_raw",
        "usable_bbox",
        "usable_crop",
        "usable_zoom_panel",
        "usable_tool_bundle",
        "usable_outline",
        "usable_outline_overlay",
        "usable_cf_only_tool_bundle",
        "usable_both_tool_bundle",
        "usable_orig_only_tool_bundle",
        "usable_cf_only_outline",
        "usable_both_outline",
        "usable_orig_only_outline",
        "original_qc_status",
        "cf_qc_status",
        "original_evidence_source",
        "cf_evidence_source",
        "original_selected_instance_id",
        "cf_selected_instance_id",
        "original_num_instances_total",
        "cf_num_instances_total",
        "original_selected_prompt_text",
        "cf_selected_prompt_text",
        "original_bbox_overlay",
        "cf_bbox_overlay",
        "original_crop",
        "cf_crop",
        "original_zoom_panel",
        "cf_zoom_panel",
        "original_outline_overlay",
        "cf_outline_overlay",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            usable = record.get("usable_tool_conditions") or {}
            usable_by_mode = record.get("usable_tool_conditions_by_input_mode") or {}
            writer.writerow(
                {
                    "pair_key": record.get("pair_key"),
                    "subset": record.get("subset"),
                    "subcategory": record.get("subcategory"),
                    "source_variant": record.get("source_variant"),
                    "edit_type": record.get("edit_type"),
                    "count_direction": record.get("count_direction"),
                    "qc_status": record.get("qc_status"),
                    "usable_raw": usable.get("raw"),
                    "usable_bbox": usable.get("bbox"),
                    "usable_crop": usable.get("crop"),
                    "usable_zoom_panel": usable.get("zoom_panel"),
                    "usable_tool_bundle": usable.get("tool_bundle"),
                    "usable_outline": usable.get("outline"),
                    "usable_outline_overlay": usable.get("outline_overlay"),
                    "usable_cf_only_tool_bundle": (usable_by_mode.get("cf_only") or {}).get("tool_bundle"),
                    "usable_both_tool_bundle": (usable_by_mode.get("both") or {}).get("tool_bundle"),
                    "usable_orig_only_tool_bundle": (usable_by_mode.get("orig_only") or {}).get("tool_bundle"),
                    "usable_cf_only_outline": (usable_by_mode.get("cf_only") or {}).get("outline"),
                    "usable_both_outline": (usable_by_mode.get("both") or {}).get("outline"),
                    "usable_orig_only_outline": (usable_by_mode.get("orig_only") or {}).get("outline"),
                    "original_qc_status": record["original"].get("qc_status"),
                    "cf_qc_status": record["cf"].get("qc_status"),
                    "original_evidence_source": record["original"].get("evidence_source"),
                    "cf_evidence_source": record["cf"].get("evidence_source"),
                    "original_selected_instance_id": record["original"].get("selected_instance_id"),
                    "cf_selected_instance_id": record["cf"].get("selected_instance_id"),
                    "original_num_instances_total": record["original"].get("num_instances_total"),
                    "cf_num_instances_total": record["cf"].get("num_instances_total"),
                    "original_selected_prompt_text": record["original"].get("selected_prompt_text"),
                    "cf_selected_prompt_text": record["cf"].get("selected_prompt_text"),
                    "original_bbox_overlay": record["original"].get("bbox_overlay"),
                    "cf_bbox_overlay": record["cf"].get("bbox_overlay"),
                    "original_crop": record["original"].get("crop"),
                    "cf_crop": record["cf"].get("crop"),
                    "original_zoom_panel": record["original"].get("zoom_panel"),
                    "cf_zoom_panel": record["cf"].get("zoom_panel"),
                    "original_outline_overlay": record["original"].get("outline_overlay"),
                    "cf_outline_overlay": record["cf"].get("outline_overlay"),
                }
            )


if __name__ == "__main__":
    sys.exit(main())
