#!/usr/bin/env python3
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[2]


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="counting_organize_") as tmp:
        root = Path(tmp)
        source_root = root / "counting"
        dataset_root = root / "dataset" / "counting"
        cf_root = root / "cf_dataset" / "counting_cf"
        metadata_root = root / "eval_results" / "counting" / "metadata"
        create_fixture(source_root)

        dry_output = run(
            "counting dry-run",
            [
                sys.executable,
                "gen_code/counting/organize_counting.py",
                "--source-root",
                str(source_root),
                "--dataset-root",
                str(dataset_root),
                "--cf-root",
                str(cf_root),
                "--metadata-root",
                str(metadata_root),
                "--dry-run",
            ],
        )
        require_all(dry_output, ["original_images=5", "cf_images=8", "metadata_records=8"])
        if dataset_root.exists() or cf_root.exists():
            raise SystemExit("Dry-run should not create output directories.")

        run(
            "counting organize fixture",
            [
                sys.executable,
                "gen_code/counting/organize_counting.py",
                "--source-root",
                str(source_root),
                "--dataset-root",
                str(dataset_root),
                "--cf-root",
                str(cf_root),
                "--metadata-root",
                str(metadata_root),
                "--overwrite",
            ],
        )
        validate_fixture_outputs(dataset_root, cf_root, metadata_root)

    print("counting organize tests passed.")
    return 0


def create_fixture(source_root: Path) -> None:
    create_bird_or_insect_fixture(source_root, "bird", "anseriformes", ["bird_0", "bird_1"])
    create_bird_or_insect_fixture(source_root, "insect", "diptera", ["fly_0"])
    create_hand_paw_fixture(
        source_root=source_root,
        group="five",
        subject="human_hand",
        stem="human_hand_1",
        cf_edit="add_to_six",
        cf_suffix="_6fingers",
        extension=".jpg",
    )
    create_hand_paw_fixture(
        source_root=source_root,
        group="four",
        subject="chicken_feet_ai",
        stem="chicken_feet_001",
        cf_edit="add_to_five",
        cf_suffix="_5toes",
        extension=".png",
    )
    (source_root / ".DS_Store").write_text("ignored", encoding="utf-8")


def create_bird_or_insect_fixture(source_root: Path, subset: str, order: str, stems: list[str]) -> None:
    base_dir = source_root / subset / order
    add_dir = base_dir / "_wings_add"
    remove_dir = base_dir / "_wings_remove"
    for directory in (base_dir, add_dir, remove_dir):
        directory.mkdir(parents=True, exist_ok=True)
    for index, stem in enumerate(stems):
        make_image(base_dir / f"{stem}.png", size=(80 + index * 10, 120), color=(80, 120, 160))
        make_image(add_dir / f"{stem}.png", size=(160, 90 + index * 10), color=(120, 160, 80))
        make_image(remove_dir / f"{stem}.png", size=(90, 90), color=(160, 80, 120))


def create_hand_paw_fixture(
    *,
    source_root: Path,
    group: str,
    subject: str,
    stem: str,
    cf_edit: str,
    cf_suffix: str,
    extension: str,
) -> None:
    original_dir = source_root / "hand_paw" / group / subject
    cf_dir = source_root / "hand_paw_cf" / cf_edit / subject
    original_dir.mkdir(parents=True, exist_ok=True)
    cf_dir.mkdir(parents=True, exist_ok=True)
    make_image(original_dir / f"{stem}{extension}", size=(70, 140), color=(200, 190, 180))
    make_image(cf_dir / f"{stem}{cf_suffix}.png", size=(140, 70), color=(180, 190, 200))


def make_image(path: Path, *, size: tuple[int, int], color: tuple[int, int, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.new("RGB", size, color)
    image.save(path)


def validate_fixture_outputs(dataset_root: Path, cf_root: Path, metadata_root: Path) -> None:
    originals = list(dataset_root.rglob("*.png"))
    cfs = list(cf_root.rglob("*.png"))
    if len(originals) != 5:
        raise SystemExit(f"Expected 5 fixture originals, found {len(originals)}")
    if len(cfs) != 8:
        raise SystemExit(f"Expected 8 fixture CF images, found {len(cfs)}")
    for image_path in [*originals, *cfs]:
        with Image.open(image_path) as image:
            if image.format != "PNG" or image.mode != "RGB" or image.size != (512, 512):
                raise SystemExit(f"Invalid normalized image: {image_path} {image.format} {image.mode} {image.size}")

    metadata_path = metadata_root / "counting_metadata.json"
    records = json.loads(metadata_path.read_text(encoding="utf-8"))
    if len(records) != 8:
        raise SystemExit(f"Expected 8 metadata records, found {len(records)}")
    required = {
        "benchmark",
        "subset",
        "group",
        "subcategory",
        "source_variant",
        "count_attribute",
        "edit_type",
        "original_path",
        "cf_path",
        "original_source_path",
        "cf_source_path",
        "pair_key",
        "output_size",
        "output_format",
        "resize_policy",
    }
    missing = required - set(records[0])
    if missing:
        raise SystemExit(f"Metadata record missing fields: {sorted(missing)}")
    if not any(record["source_variant"] == "ai" for record in records):
        raise SystemExit("Expected at least one AI source_variant record.")

    dataset_readme = dataset_root.parent / "README.md"
    cf_readme = cf_root.parent / "README.md"
    for readme in (dataset_readme, cf_readme):
        text = readme.read_text(encoding="utf-8")
        require_all(text, ["512x512", "Counting"])
        for stale in ("antlers", "bird v2", "12 insect orders", "legs_add", "legs_remove"):
            if stale in text:
                raise SystemExit(f"README contains stale fragment {stale}: {readme}")


def run(name: str, command: list[str]) -> str:
    print(f"[RUN] {name}: {' '.join(command)}")
    completed = subprocess.run(
        command,
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        timeout=120,
        check=False,
    )
    output = completed.stdout + completed.stderr
    if completed.returncode != 0:
        print(output)
        raise SystemExit(f"{name} failed with exit code {completed.returncode}")
    return output


def require_all(text: str, expected: list[str]) -> None:
    missing = [item for item in expected if item not in text]
    if missing:
        raise SystemExit(f"Missing expected fragments: {missing}")


if __name__ == "__main__":
    sys.exit(main())
