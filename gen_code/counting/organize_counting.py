#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from PIL import Image, ImageOps


REPO_ROOT = Path(__file__).resolve().parents[2]
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}
RESIZE_POLICY = "pad_preserve_aspect_white"


@dataclass(frozen=True)
class ImageJob:
    source: Path
    destination: Path


@dataclass(frozen=True)
class MigrationPlan:
    original_jobs: tuple[ImageJob, ...]
    cf_jobs: tuple[ImageJob, ...]
    metadata_records: tuple[dict[str, Any], ...]


def main() -> int:
    args = parse_args()
    source_root = resolve_path(args.source_root)
    dataset_root = resolve_path(args.dataset_root)
    cf_root = resolve_path(args.cf_root)
    metadata_root = resolve_path(args.metadata_root)

    if args.delete_source_after_validate and outputs_exist(dataset_root, cf_root, metadata_root) and not args.overwrite:
        validate_existing_outputs(
            source_root=source_root,
            dataset_root=dataset_root,
            cf_root=cf_root,
            metadata_root=metadata_root,
            image_size=args.image_size,
            expected_originals=178,
            expected_cf=256,
            expected_metadata=256,
        )
        delete_source(source_root=source_root, dry_run=args.dry_run)
        return 0

    plan = build_migration_plan(
        source_root=source_root,
        dataset_root=dataset_root,
        cf_root=cf_root,
        image_size=args.image_size,
    )
    print_plan(plan)

    if args.dry_run:
        print("Dry run complete. No files were written.")
        return 0

    prepare_output_roots(dataset_root=dataset_root, cf_root=cf_root, metadata_root=metadata_root, overwrite=args.overwrite)
    padding_color = parse_padding_color(args.padding_color)
    for job in (*plan.original_jobs, *plan.cf_jobs):
        normalize_and_save(job.source, job.destination, image_size=args.image_size, padding_color=padding_color)

    write_metadata(metadata_root=metadata_root, records=plan.metadata_records)
    write_readmes(dataset_root=dataset_root, cf_root=cf_root)
    validate_existing_outputs(
        source_root=source_root,
        dataset_root=dataset_root,
        cf_root=cf_root,
        metadata_root=metadata_root,
        image_size=args.image_size,
        expected_originals=len(plan.original_jobs),
        expected_cf=len(plan.cf_jobs),
        expected_metadata=len(plan.metadata_records),
    )

    if args.delete_source_after_validate:
        delete_source(source_root=source_root, dry_run=False)

    print("Counting migration completed and validated.")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Organize counting originals/CF images into dataset/ and cf_dataset/ with 512x512 PNG normalization."
    )
    parser.add_argument("--source-root", default="counting", help="Existing counting source directory.")
    parser.add_argument("--dataset-root", default="dataset/counting", help="Output directory for original images.")
    parser.add_argument("--cf-root", default="cf_dataset/counting_cf", help="Output directory for counterfactual images.")
    parser.add_argument(
        "--metadata-root",
        default="eval_results/counting/metadata",
        help="Output directory for counting metadata JSON/CSV.",
    )
    parser.add_argument("--image-size", type=int, default=512, help="Square output image size in pixels.")
    parser.add_argument("--padding-color", default="white", help="Padding/background color. Use 'white' or 'R,G,B'.")
    parser.add_argument("--dry-run", action="store_true", help="Validate and print the migration plan without writing files.")
    parser.add_argument("--overwrite", action="store_true", help="Replace existing counting output directories/files.")
    parser.add_argument(
        "--delete-source-after-validate",
        action="store_true",
        help="Delete source-root only after migrated outputs pass validation.",
    )
    return parser.parse_args()


def resolve_path(value: str | Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return REPO_ROOT / path


def build_migration_plan(*, source_root: Path, dataset_root: Path, cf_root: Path, image_size: int) -> MigrationPlan:
    if not source_root.exists():
        raise SystemExit(f"Source root does not exist: {source_root}")
    if image_size <= 0:
        raise SystemExit("--image-size must be positive.")

    original_jobs: dict[Path, Path] = {}
    cf_jobs: dict[Path, Path] = {}
    records: list[dict[str, Any]] = []

    for subset in ("bird", "insect"):
        subset_root = source_root / subset
        if not subset_root.exists():
            raise SystemExit(f"Missing source subset: {subset_root}")
        for group_dir in sorted_directories(subset_root):
            group = group_dir.name
            originals = {path.stem: path for path in image_files(group_dir)}
            if not originals:
                raise SystemExit(f"No original images found in {group_dir}")
            for source_edit_dir_name, edit_type in (("_wings_add", "wings_add"), ("_wings_remove", "wings_remove")):
                cf_source_dir = group_dir / source_edit_dir_name
                cf_images = {path.stem: path for path in image_files(cf_source_dir)}
                validate_matching_stems(
                    label=f"{subset}/{group}/{edit_type}",
                    original_stems=set(originals),
                    cf_stems=set(cf_images),
                )
                for stem, original_source in sorted(originals.items()):
                    cf_source = cf_images[stem]
                    original_destination = dataset_root / subset / group / f"{stem}.png"
                    cf_destination = cf_root / subset / group / edit_type / f"{stem}.png"
                    add_job(original_jobs, original_destination, original_source)
                    add_job(cf_jobs, cf_destination, cf_source)
                    records.append(
                        build_record(
                            subset=subset,
                            group=group,
                            subcategory=group,
                            source_variant="real",
                            count_attribute="wings",
                            edit_type=edit_type,
                            original_path=original_destination,
                            cf_path=cf_destination,
                            original_source_path=original_source,
                            cf_source_path=cf_source,
                            pair_key=f"{subset}/{group}/{stem}/{edit_type}",
                            image_size=image_size,
                        )
                    )

    add_hand_paw_records(
        source_root=source_root,
        dataset_root=dataset_root,
        cf_root=cf_root,
        image_size=image_size,
        original_jobs=original_jobs,
        cf_jobs=cf_jobs,
        records=records,
    )

    return MigrationPlan(
        original_jobs=tuple(ImageJob(source=src, destination=dst) for dst, src in sorted(original_jobs.items())),
        cf_jobs=tuple(ImageJob(source=src, destination=dst) for dst, src in sorted(cf_jobs.items())),
        metadata_records=tuple(sorted(records, key=lambda item: item["pair_key"])),
    )


def add_hand_paw_records(
    *,
    source_root: Path,
    dataset_root: Path,
    cf_root: Path,
    image_size: int,
    original_jobs: dict[Path, Path],
    cf_jobs: dict[Path, Path],
    records: list[dict[str, Any]],
) -> None:
    specs = (
        ("five", "add_to_six", "_6fingers", "fingers"),
        ("four", "add_to_five", "_5toes", "toes"),
    )
    for group, edit_type, cf_suffix, count_attribute in specs:
        original_group_dir = source_root / "hand_paw" / group
        cf_group_dir = source_root / "hand_paw_cf" / edit_type
        if not original_group_dir.exists():
            raise SystemExit(f"Missing hand_paw original group: {original_group_dir}")
        if not cf_group_dir.exists():
            raise SystemExit(f"Missing hand_paw CF group: {cf_group_dir}")
        for subject_dir in sorted_directories(original_group_dir):
            subject = subject_dir.name
            cf_subject_dir = cf_group_dir / subject
            if not cf_subject_dir.exists():
                raise SystemExit(f"Missing hand_paw CF subject directory: {cf_subject_dir}")
            originals = {path.stem.lower(): path for path in image_files(subject_dir)}
            cf_images = {normalize_hand_paw_cf_stem(path.stem, cf_suffix).lower(): path for path in image_files(cf_subject_dir)}
            validate_matching_stems(
                label=f"hand_paw/{group}/{subject}/{edit_type}",
                original_stems=set(originals),
                cf_stems=set(cf_images),
            )
            for normalized_stem, original_source in sorted(originals.items()):
                cf_source = cf_images[normalized_stem]
                original_destination = dataset_root / "hand_paw" / group / subject / f"{original_source.stem}.png"
                cf_destination = cf_root / "hand_paw" / edit_type / subject / f"{cf_source.stem}.png"
                add_job(original_jobs, original_destination, original_source)
                add_job(cf_jobs, cf_destination, cf_source)
                records.append(
                    build_record(
                        subset="hand_paw",
                        group=group,
                        subcategory=subject,
                        source_variant=source_variant_from_name(subject),
                        count_attribute=count_attribute,
                        edit_type=edit_type,
                        original_path=original_destination,
                        cf_path=cf_destination,
                        original_source_path=original_source,
                        cf_source_path=cf_source,
                        pair_key=f"hand_paw/{group}/{subject}/{original_source.stem}/{edit_type}",
                        image_size=image_size,
                    )
                )


def build_record(
    *,
    subset: str,
    group: str,
    subcategory: str,
    source_variant: str,
    count_attribute: str,
    edit_type: str,
    original_path: Path,
    cf_path: Path,
    original_source_path: Path,
    cf_source_path: Path,
    pair_key: str,
    image_size: int,
) -> dict[str, Any]:
    return {
        "benchmark": "counting",
        "subset": subset,
        "group": group,
        "subcategory": subcategory,
        "source_variant": source_variant,
        "count_attribute": count_attribute,
        "edit_type": edit_type,
        "original_path": str(original_path),
        "cf_path": str(cf_path),
        "original_source_path": str(original_source_path),
        "cf_source_path": str(cf_source_path),
        "pair_key": pair_key,
        "output_size": [image_size, image_size],
        "output_format": "png",
        "resize_policy": RESIZE_POLICY,
    }


def source_variant_from_name(name: str) -> str:
    return "ai" if name.lower().endswith("_ai") else "real"


def normalize_hand_paw_cf_stem(stem: str, suffix: str) -> str:
    if not stem.endswith(suffix):
        raise SystemExit(f"Hand/paw CF file does not end with expected suffix {suffix}: {stem}")
    return stem[: -len(suffix)]


def validate_matching_stems(*, label: str, original_stems: set[str], cf_stems: set[str]) -> None:
    missing = sorted(original_stems - cf_stems)
    extra = sorted(cf_stems - original_stems)
    if missing or extra:
        raise SystemExit(f"Pairing mismatch for {label}: missing_cf={missing}, extra_cf={extra}")


def add_job(jobs: dict[Path, Path], destination: Path, source: Path) -> None:
    existing = jobs.get(destination)
    if existing is not None and existing != source:
        raise SystemExit(f"Destination collision: {destination} from {existing} and {source}")
    jobs[destination] = source


def image_files(directory: Path) -> tuple[Path, ...]:
    if not directory.exists():
        raise SystemExit(f"Missing image directory: {directory}")
    return tuple(
        sorted(path for path in directory.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS)
    )


def sorted_directories(directory: Path) -> tuple[Path, ...]:
    return tuple(sorted(path for path in directory.iterdir() if path.is_dir() and not path.name.startswith(".")))


def prepare_output_roots(*, dataset_root: Path, cf_root: Path, metadata_root: Path, overwrite: bool) -> None:
    for path in (dataset_root, cf_root):
        if path.exists() and any(path.iterdir()):
            if not overwrite:
                raise SystemExit(f"Output directory is not empty. Use --overwrite to replace it: {path}")
            shutil.rmtree(path)
    metadata_json = metadata_root / "counting_metadata.json"
    metadata_csv = metadata_root / "counting_metadata.csv"
    if metadata_json.exists() or metadata_csv.exists():
        if not overwrite:
            raise SystemExit(f"Metadata already exists. Use --overwrite to replace files in: {metadata_root}")
    dataset_root.mkdir(parents=True, exist_ok=True)
    cf_root.mkdir(parents=True, exist_ok=True)
    metadata_root.mkdir(parents=True, exist_ok=True)


def normalize_and_save(source: Path, destination: Path, *, image_size: int, padding_color: tuple[int, int, int]) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(source) as image:
        image = ImageOps.exif_transpose(image)
        if image.mode in {"RGBA", "LA"} or (image.mode == "P" and "transparency" in image.info):
            image = image.convert("RGBA")
            background = Image.new("RGBA", image.size, (*padding_color, 255))
            background.alpha_composite(image)
            image = background.convert("RGB")
        else:
            image = image.convert("RGB")
        image.thumbnail((image_size, image_size), Image.Resampling.LANCZOS)
        canvas = Image.new("RGB", (image_size, image_size), padding_color)
        left = (image_size - image.width) // 2
        top = (image_size - image.height) // 2
        canvas.paste(image, (left, top))
        canvas.save(destination, format="PNG")


def parse_padding_color(value: str) -> tuple[int, int, int]:
    if value.lower() == "white":
        return (255, 255, 255)
    parts = value.split(",")
    if len(parts) != 3:
        raise SystemExit("--padding-color must be 'white' or 'R,G,B'.")
    try:
        rgb = tuple(int(part.strip()) for part in parts)
    except ValueError as exc:
        raise SystemExit("--padding-color RGB values must be integers.") from exc
    if any(channel < 0 or channel > 255 for channel in rgb):
        raise SystemExit("--padding-color RGB values must be in [0, 255].")
    return rgb  # type: ignore[return-value]


def write_metadata(*, metadata_root: Path, records: Iterable[dict[str, Any]]) -> None:
    records = list(records)
    metadata_root.mkdir(parents=True, exist_ok=True)
    json_path = metadata_root / "counting_metadata.json"
    csv_path = metadata_root / "counting_metadata.csv"
    json_path.write_text(json.dumps(records, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    if records:
        with csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(records[0]))
            writer.writeheader()
            writer.writerows(records)


def write_readmes(*, dataset_root: Path, cf_root: Path) -> None:
    dataset_parent = dataset_root.parent
    cf_parent = cf_root.parent
    dataset_parent.mkdir(parents=True, exist_ok=True)
    cf_parent.mkdir(parents=True, exist_ok=True)
    (dataset_parent / "README.md").write_text(build_dataset_readme(dataset_parent), encoding="utf-8")
    (cf_parent / "README.md").write_text(build_cf_readme(cf_parent), encoding="utf-8")


def build_dataset_readme(dataset_parent: Path) -> str:
    rows = [
        ("fashion_dataset", "Fashion logo originals", count_images(dataset_parent / "fashion_dataset")),
        ("industry_dataset", "Industry/object originals", count_images(dataset_parent / "industry_dataset")),
        ("if_exist", "Feature-existence originals", count_images(dataset_parent / "if_exist")),
        ("counting", "Counting originals", count_images(dataset_parent / "counting")),
    ]
    lines = [
        "# Dataset Directory",
        "",
        "This directory stores the original images used by the benchmark suites.",
        "",
        "| Dataset | Description | Image Count |",
        "|---|---|---:|",
    ]
    lines.extend(f"| `{name}` | {description} | {count} |" for name, description, count in rows)
    lines.extend(
        [
            "",
            "## Counting Dataset",
            "",
            "The `counting` dataset is organized into three subsets and all images are normalized to `512x512` RGB PNG files.",
            "Normalization preserves aspect ratio and uses white padding; images are not stretched or center-cropped.",
            "",
            "| Subset | Original Images | Notes |",
            "|---|---:|---|",
            "| `bird` | 45 | Bird order images paired with wing-count counterfactuals. |",
            "| `insect` | 33 | Insect order images paired with wing-count counterfactuals. |",
            "| `hand_paw` | 100 | Human/chimpanzee/panda hands or paws and bird feet. |",
            "| **Total** | **178** |  |",
            "",
        ]
    )
    return "\n".join(lines)


def build_cf_readme(cf_parent: Path) -> str:
    rows = [
        ("fashion_cf", "Fashion logo counterfactual images", count_images(cf_parent / "fashion_cf")),
        ("industry_cf", "Industry/object counterfactual images", count_images(cf_parent / "industry_cf")),
        ("if_exist_cf", "Feature-existence counterfactual images", count_images(cf_parent / "if_exist_cf")),
        ("counting_cf", "Counting counterfactual images", count_images(cf_parent / "counting_cf")),
    ]
    lines = [
        "# Counterfactual Dataset Directory",
        "",
        "This directory stores counterfactual images corresponding to original datasets under `dataset/`.",
        "",
        "| Dataset | Description | Image Count |",
        "|---|---|---:|",
    ]
    lines.extend(f"| `{name}` | {description} | {count} |" for name, description, count in rows)
    lines.extend(
        [
            "",
            "## Counting Counterfactual Dataset",
            "",
            "The `counting_cf` dataset contains count-changing counterfactuals for birds, insects, and hand/paw images.",
            "All counting counterfactuals are normalized to `512x512` RGB PNG files using aspect-ratio-preserving white padding.",
            "",
            "| Subset | Counterfactual Images | Edit Types |",
            "|---|---:|---|",
            "| `bird` | 90 | `wings_add`, `wings_remove` |",
            "| `insect` | 66 | `wings_add`, `wings_remove` |",
            "| `hand_paw` | 100 | `add_to_six`, `add_to_five` |",
            "| **Total** | **256** |  |",
            "",
            "Counting metadata is stored in `eval_results/counting/metadata/counting_metadata.json` and `.csv`.",
            "",
        ]
    )
    return "\n".join(lines)


def validate_existing_outputs(
    *,
    source_root: Path,
    dataset_root: Path,
    cf_root: Path,
    metadata_root: Path,
    image_size: int,
    expected_originals: int,
    expected_cf: int,
    expected_metadata: int,
) -> None:
    original_count = count_images(dataset_root)
    cf_count = count_images(cf_root)
    if original_count != expected_originals:
        raise SystemExit(f"Expected {expected_originals} counting originals, found {original_count} in {dataset_root}")
    if cf_count != expected_cf:
        raise SystemExit(f"Expected {expected_cf} counting CF images, found {cf_count} in {cf_root}")

    metadata_path = metadata_root / "counting_metadata.json"
    if not metadata_path.exists():
        raise SystemExit(f"Missing metadata JSON: {metadata_path}")
    records = json.loads(metadata_path.read_text(encoding="utf-8"))
    if not isinstance(records, list) or len(records) != expected_metadata:
        raise SystemExit(f"Expected {expected_metadata} metadata records, found {len(records) if isinstance(records, list) else 'non-list'}")
    for record in records:
        for key in ("original_path", "cf_path"):
            path = Path(record[key])
            if not path.exists():
                raise SystemExit(f"Metadata path does not exist: {path}")
    validate_images(dataset_root, image_size=image_size)
    validate_images(cf_root, image_size=image_size)
    if not (dataset_root.parent / "README.md").exists():
        raise SystemExit(f"Missing dataset README: {dataset_root.parent / 'README.md'}")
    if not (cf_root.parent / "README.md").exists():
        raise SystemExit(f"Missing CF README: {cf_root.parent / 'README.md'}")
    if source_root.exists():
        print(f"Validated migrated outputs. Source root still exists: {source_root}")
    else:
        print("Validated migrated outputs. Source root has already been removed.")


def validate_images(root: Path, *, image_size: int) -> None:
    for path in all_image_files(root):
        with Image.open(path) as image:
            if image.format != "PNG":
                raise SystemExit(f"Output image is not PNG: {path}")
            if image.mode != "RGB":
                raise SystemExit(f"Output image is not RGB: {path}")
            if image.size != (image_size, image_size):
                raise SystemExit(f"Output image is not {image_size}x{image_size}: {path} size={image.size}")


def outputs_exist(dataset_root: Path, cf_root: Path, metadata_root: Path) -> bool:
    return dataset_root.exists() or cf_root.exists() or (metadata_root / "counting_metadata.json").exists()


def count_images(root: Path) -> int:
    if not root.exists():
        return 0
    return sum(1 for _ in all_image_files(root))


def all_image_files(root: Path) -> Iterable[Path]:
    if not root.exists():
        return ()
    return (path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS)


def delete_source(*, source_root: Path, dry_run: bool) -> None:
    if not source_root.exists():
        print(f"Source root already absent: {source_root}")
        return
    if dry_run:
        print(f"Dry run: would delete source root after validation: {source_root}")
        return
    shutil.rmtree(source_root)
    print(f"Deleted source root after validation: {source_root}")


def print_plan(plan: MigrationPlan) -> None:
    subset_counts: dict[str, dict[str, int]] = {}
    for record in plan.metadata_records:
        counts = subset_counts.setdefault(record["subset"], {"cf": 0})
        counts["cf"] += 1
    print("Counting migration plan")
    print(f"  original_images={len(plan.original_jobs)}")
    print(f"  cf_images={len(plan.cf_jobs)}")
    print(f"  metadata_records={len(plan.metadata_records)}")
    for subset, counts in sorted(subset_counts.items()):
        print(f"  {subset}: cf_records={counts['cf']}")


if __name__ == "__main__":
    sys.exit(main())
