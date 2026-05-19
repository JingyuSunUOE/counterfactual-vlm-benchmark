#!/usr/bin/env python3
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "eval_code" / "counting" / "build_empty_vision_manifest.py"


def run_cmd(args: list[str], *, check: bool = True) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        [sys.executable, str(SCRIPT), *args],
        cwd=REPO_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if check and result.returncode != 0:
        raise AssertionError(
            f"Command failed with exit code {result.returncode}\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
        )
    return result


def write_manifest(path: Path, sample_ids: list[str]) -> None:
    rows = []
    for sample_id in sample_ids:
        rows.append(
            {
                "schema_version": "vision_manifest_v1",
                "benchmark": "counting",
                "sample_id": sample_id,
                "pair_id": sample_id,
                "image_path": f"dataset/counting/{sample_id}.png",
                "image_role": "cf",
                "category": "counting",
                "subcategory": "example",
                "source_variant": "real",
                "prompt_key": "example",
                "prompt_mode": "object",
                "target_region_key": None,
                "target_part_key": None,
            }
        )
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


def write_metadata(root: Path, sample_id: str, status: str) -> None:
    sample_dir = root / "cf" / "example" / sample_id
    sample_dir.mkdir(parents=True, exist_ok=True)
    (sample_dir / "metadata.json").write_text(
        json.dumps({"schema_version": "sam3_evidence_v1", "sample_id": sample_id, "status": status}),
        encoding="utf-8",
    )


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def test_filters_only_empty_and_preserves_order() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        manifest = root / "manifest.jsonl"
        evidence_root = root / "evidence"
        output = root / "empty.jsonl"
        write_manifest(manifest, ["sample_a", "sample_b", "sample_c"])
        write_metadata(evidence_root, "sample_a", "success")
        write_metadata(evidence_root, "sample_b", "empty")
        write_metadata(evidence_root, "sample_c", "empty")

        result = run_cmd(
            [
                "--input-manifest",
                str(manifest),
                "--evidence-root",
                str(evidence_root),
                "--output",
                str(output),
                "--overwrite",
            ]
        )

        assert "input_samples=3" in result.stdout
        assert "empty_samples=2" in result.stdout
        rows = load_jsonl(output)
        assert [row["sample_id"] for row in rows] == ["sample_b", "sample_c"]


def test_missing_primary_metadata_fails() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        manifest = root / "manifest.jsonl"
        evidence_root = root / "evidence"
        output = root / "empty.jsonl"
        write_manifest(manifest, ["sample_a", "sample_b"])
        write_metadata(evidence_root, "sample_a", "empty")

        result = run_cmd(
            [
                "--input-manifest",
                str(manifest),
                "--evidence-root",
                str(evidence_root),
                "--output",
                str(output),
            ],
            check=False,
        )

        assert result.returncode != 0
        assert "Primary evidence metadata missing" in result.stderr
        assert not output.exists()


def main() -> int:
    tests = [
        test_filters_only_empty_and_preserves_order,
        test_missing_primary_metadata_fails,
    ]
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
