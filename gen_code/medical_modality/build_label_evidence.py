#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np
from PIL import Image, ImageDraw


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_METADATA = REPO_ROOT / "eval_results" / "medical_modality" / "metadata" / "medical_modality_metadata.json"
DEFAULT_BRATS23_ROOT = (
    REPO_ROOT
    / "medical"
    / "BraTS2023_GLI"
    / "ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData"
)
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "vision_dataset" / "medical_modality" / "label_evidence_clean"
DEFAULT_MANIFEST_JSON = (
    REPO_ROOT
    / "eval_results"
    / "medical_modality"
    / "metadata"
    / "medical_label_evidence_manifest_clean.json"
)
DEFAULT_MANIFEST_CSV = (
    REPO_ROOT
    / "eval_results"
    / "medical_modality"
    / "metadata"
    / "medical_label_evidence_manifest_clean.csv"
)

SCHEMA_VERSION = "medical_label_evidence_v1"
EVIDENCE_TYPES = ("bbox", "contour", "crop", "zoom_panel")
ROLE_FILENAMES = {
    "bbox": "bbox_overlay.png",
    "contour": "contour_overlay.png",
    "crop": "crop.png",
    "zoom_panel": "zoom_panel.png",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build BraTS-label-derived medical modality visual evidence.")
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    parser.add_argument("--brats23-root", type=Path, default=DEFAULT_BRATS23_ROOT)
    parser.add_argument("--brats18-root", type=Path, default=None)
    parser.add_argument("--visual-version", default="clean", help="Visual version to process, usually clean.")
    parser.add_argument("--evidence-types", default="bbox,contour,crop,zoom_panel")
    parser.add_argument("--crop-padding", type=float, default=0.35)
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--manifest-json", type=Path, default=DEFAULT_MANIFEST_JSON)
    parser.add_argument("--manifest-csv", type=Path, default=DEFAULT_MANIFEST_CSV)
    parser.add_argument("--non-strict", action="store_true", help="Write empty records instead of failing on invalid inputs.")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    evidence_types = parse_evidence_types(args.evidence_types)
    if args.crop_padding < 0:
        raise SystemExit("--crop-padding must be non-negative.")
    records = load_metadata(args.metadata, visual_version=args.visual_version)
    print(f"metadata={args.metadata.resolve(strict=False)}")
    print(f"visual_version={args.visual_version} records={len(records)}")
    print(f"evidence_types={','.join(evidence_types)}")
    print(f"output_root={args.output_root.resolve(strict=False)}")
    if args.dry_run:
        print_preview(records)
        return 0
    if args.manifest_json.exists() and not args.overwrite:
        raise SystemExit(f"Manifest exists; pass --overwrite: {args.manifest_json}")
    if args.manifest_csv.exists() and not args.overwrite:
        raise SystemExit(f"Manifest exists; pass --overwrite: {args.manifest_csv}")
    args.output_root.mkdir(parents=True, exist_ok=True)
    args.manifest_json.parent.mkdir(parents=True, exist_ok=True)
    args.manifest_csv.parent.mkdir(parents=True, exist_ok=True)

    manifest_records: List[Dict[str, Any]] = []
    problems: List[str] = []
    for index, record in enumerate(records, start=1):
        try:
            manifest_records.append(build_pair_evidence(record, args, evidence_types))
        except Exception as exc:
            if args.non_strict:
                manifest_records.append(empty_manifest_record(record, args, reason=str(exc)))
            else:
                problems.append(f"{record.get('pair_key', f'record {index}')}: {exc}")
        if index % 100 == 0:
            print(f"processed {index}/{len(records)}")
    if problems:
        preview = "\n- ".join(problems[:20])
        suffix = "" if len(problems) <= 20 else f"\n- ... {len(problems) - 20} additional problems"
        raise SystemExit(f"Evidence generation failed:\n- {preview}{suffix}")

    payload = {
        "schema_version": SCHEMA_VERSION,
        "benchmark": "medical_modality",
        "visual_version": args.visual_version,
        "generator": generator_payload(args),
        "records": manifest_records,
    }
    args.manifest_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    write_csv(args.manifest_csv, manifest_records)
    print_summary(manifest_records)
    print(f"wrote {args.manifest_json}")
    print(f"wrote {args.manifest_csv}")
    return 0


def parse_evidence_types(raw: str) -> Tuple[str, ...]:
    values = tuple(item.strip() for item in raw.split(",") if item.strip())
    invalid = [item for item in values if item not in EVIDENCE_TYPES]
    if invalid or not values:
        raise SystemExit(f"--evidence-types must be a comma-separated subset of {EVIDENCE_TYPES}")
    return values


def load_metadata(path: Path, *, visual_version: str) -> List[Dict[str, Any]]:
    if not path.exists():
        raise SystemExit(f"Metadata not found: {path}")
    records = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(records, list):
        raise SystemExit("Medical metadata must be a JSON list.")
    selected = [record for record in records if str(record.get("visual_version")) == visual_version]
    if not selected:
        raise SystemExit(f"No metadata records found for visual_version={visual_version!r}")
    return selected


def print_preview(records: Sequence[Dict[str, Any]]) -> None:
    by_dataset = Counter(str(record.get("dataset")) for record in records)
    by_direction = Counter(str(record.get("swap_direction")) for record in records)
    print("by_dataset=" + json.dumps(dict(sorted(by_dataset.items())), sort_keys=True))
    print("by_swap_direction=" + json.dumps(dict(sorted(by_direction.items())), sort_keys=True))
    for record in records[:3]:
        print(json.dumps({key: record.get(key) for key in ("pair_key", "case_id", "slice_index", "original_path", "cf_path")}, ensure_ascii=False))


def build_pair_evidence(record: Dict[str, Any], args: argparse.Namespace, evidence_types: Sequence[str]) -> Dict[str, Any]:
    mask = load_display_mask(record, args)
    if not mask.any():
        raise ValueError("segmentation mask is empty on selected slice")
    bbox = bbox_from_mask(mask)
    pair_dir = args.output_root / str(record["dataset"]) / str(record["case_id"]) / sanitize(str(record["pair_key"]))
    original_payload = build_role_evidence(
        raw_path=Path(record["original_path"]),
        role_dir=pair_dir / "original",
        mask=mask,
        bbox=bbox,
        args=args,
        evidence_types=evidence_types,
    )
    cf_payload = build_role_evidence(
        raw_path=Path(record["cf_path"]),
        role_dir=pair_dir / "cf",
        mask=mask,
        bbox=bbox,
        args=args,
        evidence_types=evidence_types,
    )
    usable = usable_tool_conditions(original_payload, cf_payload)
    requested_usable = all(usable.get(condition, False) for condition in ("raw", *evidence_types))
    return {
        "schema_version": SCHEMA_VERSION,
        "benchmark": "medical_modality",
        "pair_key": str(record["pair_key"]),
        "question_group_id": str(record["question_group_id"]),
        "dataset": str(record["dataset"]),
        "case_id": str(record["case_id"]),
        "visual_version": str(record["visual_version"]),
        "swap_direction": str(record["swap_direction"]),
        "background_modality": str(record["background_modality"]),
        "tumor_region_modality": str(record["tumor_region_modality"]),
        "slice_index": int(record["slice_index"]),
        "generator": generator_payload(args),
        "qc_status": "pass" if requested_usable else "empty",
        "mask_bbox_xyxy": list(map(int, bbox)),
        "original": original_payload,
        "cf": cf_payload,
        "usable_tool_conditions": usable,
    }


def empty_manifest_record(record: Dict[str, Any], args: argparse.Namespace, *, reason: str) -> Dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "benchmark": "medical_modality",
        "pair_key": str(record.get("pair_key", "")),
        "question_group_id": str(record.get("question_group_id", "")),
        "dataset": str(record.get("dataset", "")),
        "case_id": str(record.get("case_id", "")),
        "visual_version": str(record.get("visual_version", "")),
        "swap_direction": str(record.get("swap_direction", "")),
        "background_modality": str(record.get("background_modality", "")),
        "tumor_region_modality": str(record.get("tumor_region_modality", "")),
        "slice_index": int(record.get("slice_index", -1)),
        "generator": generator_payload(args),
        "qc_status": "empty",
        "error_message": reason,
        "original": {"raw": str(record.get("original_path", ""))},
        "cf": {"raw": str(record.get("cf_path", ""))},
        "usable_tool_conditions": {"raw": True, "bbox": False, "contour": False, "crop": False, "zoom_panel": False},
    }


def generator_payload(args: argparse.Namespace) -> Dict[str, Any]:
    return {
        "source": "brats_segmentation_label",
        "mask_rule": "seg_gt_0",
        "crop_padding": args.crop_padding,
        "image_size": args.image_size,
    }


def load_display_mask(record: Dict[str, Any], args: argparse.Namespace) -> np.ndarray:
    import nibabel as nib

    dataset = str(record["dataset"])
    case_id = str(record["case_id"])
    if dataset == "brats2023":
        seg_path = args.brats23_root / case_id / f"{case_id}-seg.nii.gz"
    elif dataset == "brats2018" and args.brats18_root is not None:
        seg_path = args.brats18_root / case_id / f"{case_id}-seg.nii.gz"
    else:
        raise ValueError(f"No segmentation root configured for dataset={dataset}")
    if not seg_path.exists():
        raise FileNotFoundError(f"Missing segmentation label: {seg_path}")
    label = nib.load(str(seg_path)).get_fdata().astype(np.uint8)
    z_index = int(record["slice_index"])
    mask = label[:, :, z_index] > 0
    display_mask = np.flipud(mask.T)
    image = Image.fromarray((display_mask.astype(np.uint8) * 255), mode="L")
    if image.size != (args.image_size, args.image_size):
        image = image.resize((args.image_size, args.image_size), Image.Resampling.NEAREST)
    return np.array(image) > 0


def bbox_from_mask(mask: np.ndarray) -> Tuple[int, int, int, int]:
    ys, xs = np.where(mask)
    if xs.size == 0 or ys.size == 0:
        raise ValueError("cannot compute bbox from empty mask")
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def build_role_evidence(
    *,
    raw_path: Path,
    role_dir: Path,
    mask: np.ndarray,
    bbox: Tuple[int, int, int, int],
    args: argparse.Namespace,
    evidence_types: Sequence[str],
) -> Dict[str, str]:
    if not raw_path.exists():
        raise FileNotFoundError(f"Missing standardized PNG: {raw_path}")
    role_dir.mkdir(parents=True, exist_ok=True)
    raw = Image.open(raw_path).convert("RGB")
    if raw.size != (args.image_size, args.image_size):
        raw = raw.resize((args.image_size, args.image_size), Image.Resampling.BILINEAR)
    payload: Dict[str, str] = {"raw": repo_relative_or_abs(raw_path)}
    mask_path = role_dir / "mask_binary.png"
    Image.fromarray(mask.astype(np.uint8) * 255, mode="L").save(mask_path, format="PNG")
    payload["mask_binary"] = repo_relative_or_abs(mask_path)
    if "bbox" in evidence_types:
        bbox_path = role_dir / ROLE_FILENAMES["bbox"]
        make_bbox_overlay(raw, bbox).save(bbox_path, format="PNG")
        payload["bbox_overlay"] = repo_relative_or_abs(bbox_path)
    if "contour" in evidence_types:
        contour_path = role_dir / ROLE_FILENAMES["contour"]
        make_contour_overlay(raw, mask).save(contour_path, format="PNG")
        payload["contour_overlay"] = repo_relative_or_abs(contour_path)
    if "crop" in evidence_types or "zoom_panel" in evidence_types:
        crop = make_crop(raw, bbox, padding=args.crop_padding)
        if "crop" in evidence_types:
            crop_path = role_dir / ROLE_FILENAMES["crop"]
            crop.save(crop_path, format="PNG")
            payload["crop"] = repo_relative_or_abs(crop_path)
        if "zoom_panel" in evidence_types:
            zoom_path = role_dir / ROLE_FILENAMES["zoom_panel"]
            make_zoom_panel(raw, crop, bbox).save(zoom_path, format="PNG")
            payload["zoom_panel"] = repo_relative_or_abs(zoom_path)
    return payload


def make_bbox_overlay(image: Image.Image, bbox: Tuple[int, int, int, int]) -> Image.Image:
    out = image.copy()
    draw = ImageDraw.Draw(out)
    x0, y0, x1, y1 = bbox
    for offset in range(2):
        draw.rectangle((x0 - offset, y0 - offset, x1 + offset - 1, y1 + offset - 1), outline=(255, 210, 0))
    return out


def make_contour_overlay(image: Image.Image, mask: np.ndarray) -> Image.Image:
    contour = contour_from_mask(mask)
    out = np.array(image).copy()
    out[contour] = np.array([255, 210, 0], dtype=np.uint8)
    return Image.fromarray(out, mode="RGB")


def contour_from_mask(mask: np.ndarray) -> np.ndarray:
    padded = np.pad(mask.astype(bool), 1, mode="constant", constant_values=False)
    center = padded[1:-1, 1:-1]
    interior = (
        center
        & padded[:-2, 1:-1]
        & padded[2:, 1:-1]
        & padded[1:-1, :-2]
        & padded[1:-1, 2:]
    )
    return center & ~interior


def padded_bbox(bbox: Tuple[int, int, int, int], *, padding: float, width: int, height: int) -> Tuple[int, int, int, int]:
    x0, y0, x1, y1 = bbox
    pad_x = int(round((x1 - x0) * padding))
    pad_y = int(round((y1 - y0) * padding))
    return max(0, x0 - pad_x), max(0, y0 - pad_y), min(width, x1 + pad_x), min(height, y1 + pad_y)


def make_crop(image: Image.Image, bbox: Tuple[int, int, int, int], *, padding: float) -> Image.Image:
    crop_box = padded_bbox(bbox, padding=padding, width=image.width, height=image.height)
    return image.crop(crop_box).resize(image.size, Image.Resampling.BILINEAR)


def make_zoom_panel(image: Image.Image, crop: Image.Image, bbox: Tuple[int, int, int, int]) -> Image.Image:
    out = make_bbox_overlay(image, bbox)
    inset_size = max(128, image.width // 3)
    inset = crop.resize((inset_size, inset_size), Image.Resampling.BILINEAR)
    margin = 16
    x = image.width - inset_size - margin
    y = margin
    out.paste(inset, (x, y))
    draw = ImageDraw.Draw(out)
    draw.rectangle((x - 1, y - 1, x + inset_size, y + inset_size), outline=(255, 210, 0), width=2)
    return out


def usable_tool_conditions(original: Dict[str, str], cf: Dict[str, str]) -> Dict[str, bool]:
    return {
        "raw": bool(original.get("raw") and cf.get("raw")),
        "bbox": bool(original.get("bbox_overlay") and cf.get("bbox_overlay")),
        "contour": bool(original.get("contour_overlay") and cf.get("contour_overlay")),
        "crop": bool(original.get("crop") and cf.get("crop")),
        "zoom_panel": bool(original.get("zoom_panel") and cf.get("zoom_panel")),
    }


def repo_relative_or_abs(path: Path) -> str:
    resolved = path.expanduser().resolve(strict=False)
    try:
        return str(resolved.relative_to(REPO_ROOT))
    except ValueError:
        return str(resolved)


def sanitize(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in value)


def write_csv(path: Path, records: Sequence[Dict[str, Any]]) -> None:
    fieldnames = [
        "pair_key",
        "dataset",
        "case_id",
        "visual_version",
        "swap_direction",
        "qc_status",
        "bbox",
        "contour",
        "crop",
        "zoom_panel",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            usable = record.get("usable_tool_conditions", {})
            writer.writerow(
                {
                    "pair_key": record.get("pair_key", ""),
                    "dataset": record.get("dataset", ""),
                    "case_id": record.get("case_id", ""),
                    "visual_version": record.get("visual_version", ""),
                    "swap_direction": record.get("swap_direction", ""),
                    "qc_status": record.get("qc_status", ""),
                    "bbox": usable.get("bbox", False),
                    "contour": usable.get("contour", False),
                    "crop": usable.get("crop", False),
                    "zoom_panel": usable.get("zoom_panel", False),
                }
            )


def print_summary(records: Sequence[Dict[str, Any]]) -> None:
    print("records", len(records))
    print("qc_status=" + json.dumps(dict(Counter(record.get("qc_status", "") for record in records)), sort_keys=True))
    for condition in ("bbox", "contour", "crop", "zoom_panel"):
        ok = sum(bool(record.get("usable_tool_conditions", {}).get(condition)) for record in records)
        print(f"{condition}={ok}/{len(records)}")


if __name__ == "__main__":
    sys.exit(main())
