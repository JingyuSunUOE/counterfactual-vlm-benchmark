#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

import numpy as np
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "medical" / "modality_swapping" / "standardized"
DEFAULT_METADATA_ROOT = REPO_ROOT / "eval_results" / "medical_modality" / "metadata"
DEFAULT_QUESTIONS_OUTPUT = REPO_ROOT / "medical" / "modality_swapping" / "medical_modality_questions.json"

DEFAULT_BRATS18_IMG = Path("/scratch/jtu9/MR_Detection/nnunet_data/nnUNet_raw/Dataset001_BraTS2018/imagesTr")
DEFAULT_BRATS18_LBL = Path("/scratch/jtu9/MR_Detection/nnunet_data/nnUNet_raw/Dataset001_BraTS2018/labelsTr")
DEFAULT_BRATS23_IMG = Path("/scratch/jtu9/MR_Detection/nnunet_data/nnUNet_raw/Dataset002_BraTS2023/imagesTr")
DEFAULT_BRATS23_LBL = Path("/scratch/jtu9/MR_Detection/nnunet_data/nnUNet_raw/Dataset002_BraTS2023/labelsTr")
DEFAULT_BRATS23_CASE_ROOT = (
    REPO_ROOT
    / "medical"
    / "modality_swapping"
    / "BraTS2023_GLI"
    / "ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData"
)

QUESTION_TYPES = ("yes_no", "multiple_choice", "open")
INPUT_MODES = ("cf_only", "both", "orig_only")
VISUAL_VERSIONS = ("clean", "overlay")


@dataclass(frozen=True)
class DatasetSpec:
    dataset: str
    image_root: Path
    label_root: Path
    case_root: Path | None
    t1_channel: int
    flair_channel: int
    t1_suffix: str
    flair_suffix: str
    seg_suffix: str
    t1_name: str
    flair_name: str
    t1_file_prefix: str
    flair_file_prefix: str
    t1_group_id: str
    flair_group_id: str
    skip_incomplete_cases: bool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate standardized medical modality-swapping images, metadata, and question JSON."
    )
    parser.add_argument("--brats18-img-root", type=Path, default=DEFAULT_BRATS18_IMG)
    parser.add_argument("--brats18-label-root", type=Path, default=DEFAULT_BRATS18_LBL)
    parser.add_argument(
        "--brats18-case-root",
        type=Path,
        default=None,
        help="Optional BraTS2018 case-folder root with <case>/<case>-t1.nii.gz, <case>-flair.nii.gz, and <case>-seg.nii.gz.",
    )
    parser.add_argument("--brats23-img-root", type=Path, default=DEFAULT_BRATS23_IMG)
    parser.add_argument("--brats23-label-root", type=Path, default=DEFAULT_BRATS23_LBL)
    parser.add_argument(
        "--brats23-case-root",
        type=Path,
        default=DEFAULT_BRATS23_CASE_ROOT,
        help="Optional BraTS2023 case-folder root with <case>/<case>-t1n.nii.gz, <case>-t2f.nii.gz, and <case>-seg.nii.gz.",
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--metadata-root", type=Path, default=DEFAULT_METADATA_ROOT)
    parser.add_argument("--questions-output", type=Path, default=DEFAULT_QUESTIONS_OUTPUT)
    parser.add_argument("--datasets", choices=("all", "brats2018", "brats2023"), default="all")
    parser.add_argument("--visual-versions", default="clean,overlay", help="Comma-separated clean,overlay subset.")
    parser.add_argument("--image-size", type=int, default=750)
    parser.add_argument("--overlay-alpha", type=float, default=0.45)
    parser.add_argument(
        "--skip-incomplete-cases",
        action="store_true",
        help="Skip case folders missing required modality or segmentation files instead of aborting.",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    visual_versions = parse_visual_versions(args.visual_versions)
    specs = selected_dataset_specs(args)
    validate_roots(specs)

    if args.output_root.exists() and any(args.output_root.iterdir()) and not args.overwrite and not args.dry_run:
        raise SystemExit(f"Output root is not empty; pass --overwrite to replace: {args.output_root}")

    total_cases = sum(len(case_ids(spec)) for spec in specs)
    print(f"datasets={[spec.dataset for spec in specs]} cases={total_cases} visual_versions={visual_versions}")
    print(f"output_root={args.output_root}")
    print(f"metadata_root={args.metadata_root}")
    print(f"questions_output={args.questions_output}")
    if args.dry_run:
        return 0

    if args.overwrite:
        for path in (args.output_root, args.metadata_root):
            if path.exists():
                shutil.rmtree(path)
    args.output_root.mkdir(parents=True, exist_ok=True)
    args.metadata_root.mkdir(parents=True, exist_ok=True)

    annotations: List[Dict[str, Any]] = []
    for spec in specs:
        annotations.extend(process_dataset(spec, args.output_root, visual_versions, args.image_size, args.overlay_alpha))

    questions = build_questions_json()
    write_json(args.questions_output, questions)
    annotations_path = args.metadata_root / "medical_modality_metadata.json"
    annotations_csv = args.metadata_root / "medical_modality_metadata.csv"
    write_json(annotations_path, annotations)
    write_csv(annotations_csv, annotations)
    print(f"wrote {annotations_path} ({len(annotations)} records)")
    print(f"wrote {annotations_csv}")
    print(f"wrote {args.questions_output}")
    return 0


def selected_dataset_specs(args: argparse.Namespace) -> List[DatasetSpec]:
    specs = [
        DatasetSpec(
            dataset="brats2018",
            image_root=args.brats18_img_root,
            label_root=args.brats18_label_root,
            case_root=args.brats18_case_root,
            t1_channel=1,
            flair_channel=0,
            t1_suffix="t1",
            flair_suffix="flair",
            seg_suffix="seg",
            t1_name="T1",
            flair_name="FLAIR",
            t1_file_prefix="T1",
            flair_file_prefix="FLAIR",
            t1_group_id="brats2018_t1_background_flair_tumor",
            flair_group_id="brats2018_flair_background_t1_tumor",
            skip_incomplete_cases=args.skip_incomplete_cases,
        ),
        DatasetSpec(
            dataset="brats2023",
            image_root=args.brats23_img_root,
            label_root=args.brats23_label_root,
            case_root=args.brats23_case_root,
            t1_channel=1,
            flair_channel=2,
            t1_suffix="t1n",
            flair_suffix="t2f",
            seg_suffix="seg",
            t1_name="T1N",
            flair_name="T2F",
            t1_file_prefix="T1N",
            flair_file_prefix="T2F",
            t1_group_id="brats2023_t1n_background_t2f_tumor",
            flair_group_id="brats2023_t2f_background_t1n_tumor",
            skip_incomplete_cases=args.skip_incomplete_cases,
        ),
    ]
    if args.datasets == "all":
        return specs
    return [spec for spec in specs if spec.dataset == args.datasets]


def validate_roots(specs: Sequence[DatasetSpec]) -> None:
    missing = []
    for spec in specs:
        if spec.case_root is not None and spec.case_root.exists():
            continue
        if spec.case_root is not None and not spec.case_root.exists():
            missing.append(str(spec.case_root))
        for path in (spec.image_root, spec.label_root):
            if not path.exists():
                missing.append(str(path))
    if missing:
        raise SystemExit(
            "Original BraTS NIfTI roots are required to generate clean/overlay standardized images.\n"
            "Provide either a case-folder root or nnUNet-style imagesTr/labelsTr roots.\n"
            "Missing paths:\n- " + "\n- ".join(missing)
        )


def parse_visual_versions(raw: str) -> List[str]:
    values = [part.strip() for part in raw.split(",") if part.strip()]
    invalid = [value for value in values if value not in VISUAL_VERSIONS]
    if invalid or not values:
        raise SystemExit(f"--visual-versions must be a comma-separated subset of {VISUAL_VERSIONS}")
    return values


def process_dataset(
    spec: DatasetSpec,
    output_root: Path,
    visual_versions: Sequence[str],
    image_size: int,
    overlay_alpha: float,
) -> List[Dict[str, Any]]:
    annotations: List[Dict[str, Any]] = []
    ids = case_ids(spec)
    for index, case_id in enumerate(ids, start=1):
        t1 = load_volume(spec, case_id, "t1")
        flair = load_volume(spec, case_id, "flair")
        label = load_label(spec, case_id)
        z_index = pick_largest_tumor_slice(label)
        mask = label[:, :, z_index] > 0
        t1_slice = normalize_slice(t1[:, :, z_index])
        flair_slice = normalize_slice(flair[:, :, z_index])

        t1_with_flair = t1_slice.copy()
        t1_with_flair[mask] = flair_slice[mask]
        flair_with_t1 = flair_slice.copy()
        flair_with_t1[mask] = t1_slice[mask]

        for visual_version in visual_versions:
            case_dir = output_root / visual_version / spec.dataset / case_id
            case_dir.mkdir(parents=True, exist_ok=True)
            use_overlay = visual_version == "overlay"
            t1_original = save_png(t1_slice, mask, case_dir / f"{spec.t1_file_prefix}_original.png", image_size, overlay_alpha, use_overlay)
            flair_original = save_png(flair_slice, mask, case_dir / f"{spec.flair_file_prefix}_original.png", image_size, overlay_alpha, use_overlay)
            t1_cf = save_png(t1_with_flair, mask, case_dir / f"{spec.t1_file_prefix}_with_{spec.flair_file_prefix}_tumor.png", image_size, overlay_alpha, use_overlay)
            flair_cf = save_png(flair_with_t1, mask, case_dir / f"{spec.flair_file_prefix}_with_{spec.t1_file_prefix}_tumor.png", image_size, overlay_alpha, use_overlay)
            annotations.append(
                annotation_record(
                    spec=spec,
                    case_id=case_id,
                    visual_version=visual_version,
                    z_index=z_index,
                    original_path=t1_original,
                    cf_path=t1_cf,
                    group_id=spec.t1_group_id,
                    swap_direction=f"{spec.t1_name.lower()}_background_{spec.flair_name.lower()}_tumor",
                    background_modality=spec.t1_name,
                    tumor_region_modality=spec.flair_name,
                    biased_modality=spec.t1_name,
                )
            )
            annotations.append(
                annotation_record(
                    spec=spec,
                    case_id=case_id,
                    visual_version=visual_version,
                    z_index=z_index,
                    original_path=flair_original,
                    cf_path=flair_cf,
                    group_id=spec.flair_group_id,
                    swap_direction=f"{spec.flair_name.lower()}_background_{spec.t1_name.lower()}_tumor",
                    background_modality=spec.flair_name,
                    tumor_region_modality=spec.t1_name,
                    biased_modality=spec.flair_name,
                )
            )
        if index % 100 == 0:
            print(f"{spec.dataset}: processed {index}/{len(ids)} cases")
    return annotations


def annotation_record(
    *,
    spec: DatasetSpec,
    case_id: str,
    visual_version: str,
    z_index: int,
    original_path: Path,
    cf_path: Path,
    group_id: str,
    swap_direction: str,
    background_modality: str,
    tumor_region_modality: str,
    biased_modality: str,
) -> Dict[str, Any]:
    return {
        "benchmark": "medical_modality",
        "category": "medical_modality",
        "dataset": spec.dataset,
        "case_id": case_id,
        "visual_version": visual_version,
        "source_variant": visual_version,
        "question_group_id": group_id,
        "pair_key": f"{spec.dataset}:{visual_version}:{case_id}:{swap_direction}",
        "swap_direction": swap_direction,
        "background_modality": background_modality,
        "tumor_region_modality": tumor_region_modality,
        "biased_modality": biased_modality,
        "original_path": str(original_path.resolve()),
        "cf_path": str(cf_path.resolve()),
        "slice_index": z_index,
        "answers": {
            "cf_only": {"correct": "no", "biased": "yes"},
            "both": {"correct": "no", "biased": "yes"},
            "orig_only": {"correct": "yes", "biased": None},
        },
    }


def case_ids(spec: DatasetSpec) -> List[str]:
    if uses_case_root(spec):
        case_dirs = sorted(path for path in spec.case_root.iterdir() if path.is_dir())
        complete: List[str] = []
        incomplete: List[tuple[str, List[str]]] = []
        for case_dir in case_dirs:
            missing = missing_case_folder_files(spec, case_dir.name)
            if missing:
                incomplete.append((case_dir.name, missing))
            else:
                complete.append(case_dir.name)
        if incomplete and not spec.skip_incomplete_cases:
            examples = "\n".join(f"- {case_id}: {', '.join(missing)}" for case_id, missing in incomplete[:20])
            extra = "" if len(incomplete) <= 20 else f"\n... {len(incomplete) - 20} more"
            raise SystemExit(
                f"{spec.dataset} case-folder root contains {len(incomplete)} incomplete cases. "
                "Pass --skip-incomplete-cases to skip them, or redownload the missing files.\n"
                f"{examples}{extra}"
            )
        if incomplete:
            print(f"{spec.dataset}: skipping {len(incomplete)} incomplete cases")
        return complete
    return sorted({path.name.rsplit("_", 1)[0] for path in spec.image_root.glob("*.nii.gz")})


def uses_case_root(spec: DatasetSpec) -> bool:
    return spec.case_root is not None and spec.case_root.exists()


def missing_case_folder_files(spec: DatasetSpec, case_id: str) -> List[str]:
    assert spec.case_root is not None
    case_dir = spec.case_root / case_id
    required = [
        case_dir / f"{case_id}-{spec.t1_suffix}.nii.gz",
        case_dir / f"{case_id}-{spec.flair_suffix}.nii.gz",
        case_dir / f"{case_id}-{spec.seg_suffix}.nii.gz",
    ]
    return [str(path) for path in required if not path.exists()]


def load_volume(spec: DatasetSpec, case_id: str, modality: str) -> np.ndarray:
    import nibabel as nib

    if uses_case_root(spec):
        suffix = spec.t1_suffix if modality == "t1" else spec.flair_suffix
        path = spec.case_root / case_id / f"{case_id}-{suffix}.nii.gz"
    else:
        channel = spec.t1_channel if modality == "t1" else spec.flair_channel
        path = spec.image_root / f"{case_id}_{channel:04d}.nii.gz"
    if not path.exists():
        raise FileNotFoundError(f"Missing modality volume: {path}")
    return nib.load(str(path)).get_fdata().astype(np.float32)


def load_label(spec: DatasetSpec, case_id: str) -> np.ndarray:
    import nibabel as nib

    if uses_case_root(spec):
        path = spec.case_root / case_id / f"{case_id}-{spec.seg_suffix}.nii.gz"
    else:
        path = spec.label_root / f"{case_id}.nii.gz"
    if not path.exists():
        raise FileNotFoundError(f"Missing segmentation label: {path}")
    return nib.load(str(path)).get_fdata().astype(np.uint8)


def pick_largest_tumor_slice(label: np.ndarray) -> int:
    whole_tumor = label > 0
    return int(np.argmax(whole_tumor.sum(axis=(0, 1))))


def normalize_slice(slice_2d: np.ndarray) -> np.ndarray:
    brain = slice_2d[slice_2d > 0]
    if brain.size == 0:
        return np.zeros_like(slice_2d, dtype=np.float32)
    low, high = np.percentile(brain, [1, 99])
    if high <= low:
        high = low + 1
    return np.clip((slice_2d - low) / (high - low), 0.0, 1.0).astype(np.float32)


def save_png(
    image: np.ndarray,
    mask: np.ndarray,
    path: Path,
    image_size: int,
    overlay_alpha: float,
    use_overlay: bool,
) -> Path:
    display = np.flipud(image.T)
    gray = (display * 255).round().astype(np.uint8)
    rgb = np.stack([gray, gray, gray], axis=-1).astype(np.float32) / 255.0
    if use_overlay:
        display_mask = np.flipud(mask.T)
        rgb[display_mask, 0] = rgb[display_mask, 0] * (1 - overlay_alpha) + overlay_alpha
        rgb[display_mask, 1] = rgb[display_mask, 1] * (1 - overlay_alpha)
        rgb[display_mask, 2] = rgb[display_mask, 2] * (1 - overlay_alpha)
    out = Image.fromarray(np.clip(rgb * 255, 0, 255).astype(np.uint8), mode="RGB")
    if out.size != (image_size, image_size):
        out = out.resize((image_size, image_size), Image.Resampling.BILINEAR)
    out.save(path, format="PNG")
    return path


def build_questions_json() -> Dict[str, Any]:
    groups = []
    for dataset, t1_name, flair_name, t1_group, flair_group in [
        ("brats2018", "T1", "FLAIR", "brats2018_t1_background_flair_tumor", "brats2018_flair_background_t1_tumor"),
        ("brats2023", "T1N", "T2F", "brats2023_t1n_background_t2f_tumor", "brats2023_t2f_background_t1n_tumor"),
    ]:
        groups.append(question_group(dataset, t1_group, f"{t1_name} background with {flair_name} tumor", t1_name, flair_name, "B", "A"))
        groups.append(question_group(dataset, flair_group, f"{flair_name} background with {t1_name} tumor", flair_name, t1_name, "A", "B"))
    return {
        "benchmark_name": "medical_modality",
        "version": "1.1",
        "language": "en",
        "input_modes": list(INPUT_MODES),
        "category_order": ["medical_modality"],
        "question_type_order": list(QUESTION_TYPES),
        "visual_versions": list(VISUAL_VERSIONS),
        "categories": [{"name": "medical_modality", "question_groups": groups}],
    }


def question_group(
    dataset: str,
    group_id: str,
    display_name: str,
    background_modality: str,
    tumor_modality: str,
    cf_correct_option: str,
    cf_biased_option: str,
) -> Dict[str, Any]:
    choices = [
        {"id": "A", "text": "T1 or native T1"},
        {"id": "B", "text": "FLAIR or T2-FLAIR"},
        {"id": "C", "text": "T2-weighted but not clearly FLAIR"},
        {"id": "D", "text": "The tumor region is too unclear to judge reliably"},
    ]
    orig_option = "A" if background_modality.upper().startswith("T1") else "B"
    cf_correct_yes_no = "yes" if tumor_modality.upper() in {"FLAIR", "T2F"} else "no"
    cf_biased_yes_no = "yes" if background_modality.upper() in {"FLAIR", "T2F"} else "no"
    orig_yes_no = "yes" if background_modality.upper() in {"FLAIR", "T2F"} else "no"
    prior = "Brain MRI sequences have characteristic signal patterns across tumor and surrounding brain tissue."
    return {
        "question_group_id": group_id,
        "dataset": dataset,
        "subcategory": group_id,
        "display_name": display_name,
        "background_modality": background_modality,
        "tumor_region_modality": tumor_modality,
        "questions": [
            {
                "question_id": f"{group_id}_yes_no",
                "type": "yes_no",
                "prompt": f"{prior} Based on the tumor-region signal, does the tumor region look more FLAIR-like than native T1-like? Answer only yes or no.",
                "answers": {
                    "cf_only": {"correct_answer": cf_correct_yes_no, "biased_answer": cf_biased_yes_no},
                    "both": {"correct_answer": cf_correct_yes_no, "biased_answer": cf_biased_yes_no},
                    "orig_only": {"correct_answer": orig_yes_no, "biased_answer": orig_yes_no},
                },
            },
            {
                "question_id": f"{group_id}_multiple_choice",
                "type": "multiple_choice",
                "prompt": f"{prior} Which MRI sequence does the visible tumor region most closely resemble? Answer with only the option letter.",
                "choices": choices,
                "answers": {
                    "cf_only": option_answer(choices, cf_correct_option, cf_biased_option),
                    "both": option_answer(choices, cf_correct_option, cf_biased_option),
                    "orig_only": option_answer(choices, orig_option, orig_option),
                },
            },
            {
                "question_id": f"{group_id}_open",
                "type": "open",
                "prompt": f"{prior} State whether the tumor-region contrast supports or conflicts with the surrounding MRI sequence, and cite one visible signal cue. Answer in one short sentence.",
                "answers": {
                    "cf_only": open_answer(background_modality, tumor_modality, target="conflicts", second_image=False),
                    "both": open_answer(background_modality, tumor_modality, target="conflicts", second_image=True),
                    "orig_only": open_answer(background_modality, background_modality, target="supports", second_image=False),
                },
            },
        ],
    }


def option_answer(choices: Sequence[Dict[str, str]], correct: str, biased: str) -> Dict[str, str]:
    text_by_id = {choice["id"]: choice["text"] for choice in choices}
    return {
        "correct_option_id": correct,
        "correct_option_text": text_by_id[correct],
        "biased_option_id": biased,
        "biased_option_text": text_by_id[biased],
    }


def open_answer(background: str, tumor: str, *, target: str, second_image: bool) -> Dict[str, Any]:
    image_ref = "the second image" if second_image else "the image"
    if target == "conflicts":
        claim = f"The tumor-region contrast conflicts with the surrounding {background} sequence and is more consistent with {tumor}."
        evidence = f"Visible evidence should mention a tumor-region signal cue that differs from the surrounding {background} background, such as abnormal brightness/darkness or a local contrast discontinuity."
        bias = f"The answer treats the tumor region as consistent with the surrounding {background} sequence or identifies the tumor as {background}-like despite the swapped tumor contrast."
    else:
        claim = f"The tumor-region contrast supports or is consistent with the surrounding {background} sequence."
        evidence = "Visible evidence should mention no clear local signal discontinuity between the tumor region and surrounding MRI sequence, or a tumor-region cue consistent with the same sequence."
        bias = "The answer gives only generic MRI sequence knowledge without citing visible contrast evidence."
    return {
        "correct_rubric": {
            "target_claim": claim,
            "required_visual_evidence": evidence,
            "reject_if": f"Reject if the answer is not about {image_ref}, only names a diagnosis, or gives no visible signal cue.",
        },
        "bias_rubric": {
            "target_claim": bias,
            "prior_basis": f"Background sequence prior: the surrounding image appears {background}-like.",
            "qualify_if": "The answer follows the background-modality expectation instead of the visible tumor-region contrast.",
            "reject_if": "Reject if the answer explicitly identifies the correct tumor-region modality inconsistency with visible evidence.",
        },
        "other_rubric": {
            "label_when": "The answer is vague, contradictory, refuses, gives a clinical diagnosis instead of modality consistency, or lacks a visible MRI signal cue.",
        },
    }


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    raise SystemExit(main())
