#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence

from PIL import Image, ImageDraw

from eval_common import (
    DEFAULT_METADATA_ROOT,
    DEFAULT_QUESTIONS_PATH,
    REPO_ROOT,
    SOURCE_VARIANTS,
    build_group_specs,
    build_image_pairs,
    filter_groups,
    load_benchmark,
)


DEFAULT_OUTPUT_ROOT = DEFAULT_METADATA_ROOT / "visual_qc"


def main() -> int:
    args = parse_args()
    benchmark = load_benchmark(args.questions)
    subcategory_info = build_subcategory_info(benchmark)
    groups = build_group_specs(benchmark, base_dir=REPO_ROOT)
    groups = filter_groups(
        groups,
        category=args.category,
        subcategory=args.subcategory,
        source_variant=args.source_variant,
        question_types=("yes_no", "multiple_choice", "open"),
    )
    pairs = build_image_pairs(groups)
    if args.max_pairs is not None:
        pairs = pairs[: args.max_pairs]
    if not pairs:
        raise SystemExit("No image pairs matched the selected filters.")

    csv_path = args.output_root / "visual_qc_pairs.csv"
    sheet_path = args.output_root / "visual_qc_contact_sheet.jpg"

    print(f"selected_pairs={len(pairs)}")
    print(f"csv={csv_path}")
    print(f"contact_sheet={sheet_path}")
    if args.dry_run:
        print("dry_run=true; no files written")
        return 0

    args.output_root.mkdir(parents=True, exist_ok=True)
    rows = [build_row(pair, subcategory_info) for pair in pairs]
    write_csv(csv_path, rows)
    write_contact_sheet(
        sheet_path,
        rows,
        thumb_width=args.thumb_width,
        thumb_height=args.thumb_height,
        columns=args.columns,
    )
    print("visual QC artifacts written.")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a non-destructive visual QC contact sheet and CSV for if_exist image pairs."
    )
    parser.add_argument("--questions", type=Path, default=DEFAULT_QUESTIONS_PATH, help="Benchmark question JSON path.")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT, help="Output directory.")
    parser.add_argument("--category", choices=("if_exist",), default=None, help="Restrict to the if_exist category.")
    parser.add_argument("--subcategory", default=None, help="Restrict to one semantic subcategory.")
    parser.add_argument(
        "--source-variant",
        choices=("both", *SOURCE_VARIANTS),
        default="both",
        help="Restrict to real images, AI images, or both.",
    )
    parser.add_argument("--max-pairs", type=int, default=None, help="Limit the number of image pairs for preview/testing.")
    parser.add_argument("--thumb-width", type=int, default=180, help="Thumbnail width for each original/CF image.")
    parser.add_argument("--thumb-height", type=int, default=140, help="Thumbnail height for each original/CF image.")
    parser.add_argument("--columns", type=int, default=5, help="Number of pair tiles per contact-sheet row.")
    parser.add_argument("--dry-run", action="store_true", help="Print selected outputs without writing files.")
    return parser.parse_args()


def build_subcategory_info(benchmark: Dict[str, Any]) -> Dict[tuple[str, str], Dict[str, Any]]:
    info: Dict[tuple[str, str], Dict[str, Any]] = {}
    for category_entry in benchmark["categories"]:
        category = category_entry["category"]
        for subcategory_entry in category_entry["subcategories"]:
            subcategory = subcategory_entry["subcategory"]
            info[(category, subcategory)] = {
                "display_name": subcategory_entry.get("display_name", subcategory.replace("_", " ")),
                "edit_type": subcategory_entry.get("edit_type", ""),
                "target_visual_cue": subcategory_entry.get("target_visual_cue", ""),
                "bias_axis": subcategory_entry.get("bias_axis", ""),
            }
    return info


def build_row(pair: Any, subcategory_info: Dict[tuple[str, str], Dict[str, Any]]) -> Dict[str, str]:
    info = subcategory_info[(pair.category, pair.subcategory)]
    return {
        "category": pair.category,
        "subcategory": pair.subcategory,
        "display_name": str(info["display_name"]),
        "source_variant": pair.source_variant,
        "original_path": str(pair.original_path),
        "cf_path": str(pair.edited_path),
        "original_filename": pair.original_path.name,
        "cf_filename": pair.edited_path.name,
        "edit_type": str(info["edit_type"]),
        "target_visual_cue": str(info["target_visual_cue"]),
        "bias_axis": str(info["bias_axis"]),
        "question_risk_note": question_risk_note(pair.category, pair.subcategory),
        "suggested_review_status": "review",
    }


def question_risk_note(category: str, subcategory: str) -> str:
    return "Review whether the removed feature is clearly absent, reduced, or ambiguous rather than still visibly present."


def write_csv(path: Path, rows: Sequence[Dict[str, str]]) -> None:
    fieldnames = [
        "category",
        "subcategory",
        "display_name",
        "source_variant",
        "original_path",
        "cf_path",
        "original_filename",
        "cf_filename",
        "edit_type",
        "target_visual_cue",
        "bias_axis",
        "question_risk_note",
        "suggested_review_status",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_contact_sheet(
    path: Path,
    rows: Sequence[Dict[str, str]],
    *,
    thumb_width: int,
    thumb_height: int,
    columns: int,
) -> None:
    columns = max(1, columns)
    label_height = 42
    tile_width = thumb_width * 2
    tile_height = thumb_height + label_height
    sheet_rows = (len(rows) + columns - 1) // columns
    sheet = Image.new("RGB", (tile_width * columns, tile_height * sheet_rows), "white")
    draw = ImageDraw.Draw(sheet)

    for index, row in enumerate(rows):
        col = index % columns
        sheet_row = index // columns
        x0 = col * tile_width
        y0 = sheet_row * tile_height
        paste_thumbnail(sheet, Path(row["original_path"]), x0, y0, thumb_width, thumb_height)
        paste_thumbnail(sheet, Path(row["cf_path"]), x0 + thumb_width, y0, thumb_width, thumb_height)
        label = f"{index + 1}. {row['category']}/{row['subcategory']} ({row['source_variant']})"
        draw.text((x0 + 4, y0 + thumb_height + 4), label[:70], fill=(0, 0, 0))
        draw.text((x0 + 4, y0 + thumb_height + 22), "left=original  right=cf", fill=(80, 80, 80))

    sheet.save(path, quality=92)


def paste_thumbnail(sheet: Image.Image, path: Path, x: int, y: int, width: int, height: int) -> None:
    with Image.open(path) as image:
        image = image.convert("RGB")
        image.thumbnail((width, height))
        canvas = Image.new("RGB", (width, height), (245, 245, 245))
        px = (width - image.width) // 2
        py = (height - image.height) // 2
        canvas.paste(image, (px, py))
        sheet.paste(canvas, (x, y))


if __name__ == "__main__":
    sys.exit(main())
