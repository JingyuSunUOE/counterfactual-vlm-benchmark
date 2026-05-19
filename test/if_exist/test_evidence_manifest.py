#!/usr/bin/env python3
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="if_exist_evidence_manifest_") as tmp_dir:
        output_json = Path(tmp_dir) / "manifest.json"
        output_csv = Path(tmp_dir) / "manifest.csv"
        output = run(
            "build evidence manifest",
            [
                sys.executable,
                "eval_code/if_exist/build_evidence_manifest.py",
                "--original-root",
                "vision_dataset/if_exist/sam3_object_original_pad005",
                "--cf-root",
                "vision_dataset/if_exist/sam3_object_cf_pad005",
                "--output-json",
                str(output_json),
                "--output-csv",
                str(output_csv),
            ],
        )
        require_all("build evidence manifest", output, ["original_success=160", "cf_success=160", "pairs=160"])
        if not output_json.exists() or not output_csv.exists():
            raise SystemExit("Evidence manifest outputs were not written.")
        records = json.loads(output_json.read_text(encoding="utf-8"))
        if len(records) != 160:
            raise SystemExit(f"Expected 160 manifest records, found {len(records)}")
        subcategories = {record["subcategory"] for record in records}
        if subcategories != {"camel", "elephant_trunk", "fish_fin", "rabbit"}:
            raise SystemExit(f"Unexpected subcategories: {sorted(subcategories)}")
        source_variants = {record["source_variant"] for record in records}
        if source_variants != {"real", "ai"}:
            raise SystemExit(f"Unexpected source variants: {sorted(source_variants)}")
        for record in records:
            if record["schema_version"] != "if_exist_sam3_object_evidence_v1":
                raise SystemExit("Unexpected evidence schema version.")
            if record["qc_status"] != "pass":
                raise SystemExit("Evidence manifest should default to qc_status=pass.")
            usable = record.get("usable_tool_conditions") or {}
            for condition in ("raw", "bbox", "crop", "zoom_panel", "tool_bundle"):
                if usable.get(condition) is not True:
                    raise SystemExit(f"Expected {condition} to be usable in if_exist manifest.")
            for role in ("original", "cf"):
                availability = record[role].get("evidence_available") or {}
                for condition in ("raw", "bbox", "crop", "zoom_panel", "tool_bundle"):
                    if availability.get(condition) is not True:
                        raise SystemExit(f"Expected {role}.{condition} evidence availability.")
                for field in ("raw", "bbox_overlay", "crop", "zoom_panel"):
                    path = Path(record[role][field])
                    if not path.exists():
                        raise SystemExit(f"Missing manifest file reference: {path}")

    print("if_exist evidence manifest tests passed.")
    return 0


def run(name: str, command: list[str]) -> str:
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


def require_all(name: str, output: str, expected: list[str]) -> None:
    missing = [item for item in expected if item not in output]
    if missing:
        raise SystemExit(f"{name} missing expected output fragments: {missing}")


if __name__ == "__main__":
    sys.exit(main())
