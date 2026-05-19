#!/usr/bin/env python3
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "eval_code" / "counting" / "build_vision_manifest.py"
COUNTING_CONFIG = REPO_ROOT / "vision_configs" / "sam3_prompts" / "counting.json"


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


def write_image(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (32, 32), "white").save(path)


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def synthetic_record(
    *,
    original: Path,
    cf: Path,
    subset: str,
    subcategory: str,
    edit_type: str,
    source_variant: str = "real",
) -> dict:
    return {
        "benchmark": "counting",
        "subset": subset,
        "group": subcategory,
        "subcategory": subcategory,
        "source_variant": source_variant,
        "count_attribute": "wings",
        "edit_type": edit_type,
        "original_path": str(original),
        "cf_path": str(cf),
        "pair_key": f"{subset}/{subcategory}/{original.stem}/{edit_type}",
        "question_group_id": f"counting_{subset}_{subcategory}_{edit_type}",
        "normal_count": 2,
        "cf_count": 4,
        "biased_count": 2,
        "count_direction": "more_than_expected",
        "display_name": subcategory,
    }


def test_synthetic_manifest_generation() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        original = root / "dataset" / "counting" / "bird" / "accipitriformes" / "one.png"
        cf_add = root / "cf_dataset" / "counting_cf" / "bird" / "accipitriformes" / "wings_add" / "one.png"
        cf_remove = root / "cf_dataset" / "counting_cf" / "bird" / "accipitriformes" / "wings_remove" / "one.png"
        hand_original = root / "dataset" / "counting" / "hand_paw" / "five" / "human_hand_ai" / "hand.png"
        hand_cf = root / "cf_dataset" / "counting_cf" / "hand_paw" / "add_to_six" / "human_hand_ai" / "hand.png"
        for path in [original, cf_add, cf_remove, hand_original, hand_cf]:
            write_image(path)
        annotations = [
            synthetic_record(original=original, cf=cf_add, subset="bird", subcategory="accipitriformes", edit_type="wings_add"),
            synthetic_record(original=original, cf=cf_remove, subset="bird", subcategory="accipitriformes", edit_type="wings_remove"),
            synthetic_record(
                original=hand_original,
                cf=hand_cf,
                subset="hand_paw",
                subcategory="human_hand_ai",
                edit_type="add_to_six",
                source_variant="ai",
            ),
        ]
        annotations_path = root / "annotations.json"
        original_output = root / "original.jsonl"
        cf_output = root / "cf.jsonl"
        annotations_path.write_text(json.dumps(annotations), encoding="utf-8")

        result = run_cmd(
            [
                "--annotations",
                str(annotations_path),
                "--original-output",
                str(original_output),
                "--cf-output",
                str(cf_output),
                "--overwrite",
            ]
        )

        assert "annotations=3" in result.stdout
        assert "original_manifest=2" in result.stdout
        assert "cf_manifest=3" in result.stdout
        original_records = load_jsonl(original_output)
        cf_records = load_jsonl(cf_output)
        assert len(original_records) == 2
        assert len(cf_records) == 3
        assert {record["image_role"] for record in original_records} == {"original"}
        assert {record["image_role"] for record in cf_records} == {"cf"}
        assert all(record["schema_version"] == "vision_manifest_v1" for record in original_records + cf_records)
        assert all(record["benchmark"] == "counting" for record in original_records + cf_records)
        assert all(record["prompt_key"] == record["subcategory"] for record in original_records + cf_records)
        assert all(record["prompt_mode"] == "object" for record in original_records + cf_records)
        assert any(record["prompt_key"] == "human_hand_ai" for record in original_records + cf_records)
        assert not any(record["prompt_key"] in {"wings_add", "wings_remove"} for record in cf_records)
        assert len({record["sample_id"] for record in original_records}) == 2
        assert len({record["sample_id"] for record in cf_records}) == 3


def test_dry_run_does_not_write_outputs() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        original = root / "one.png"
        cf = root / "cf.png"
        write_image(original)
        write_image(cf)
        annotations_path = root / "annotations.json"
        original_output = root / "original.jsonl"
        cf_output = root / "cf.jsonl"
        annotations_path.write_text(
            json.dumps(
                [
                    synthetic_record(
                        original=original,
                        cf=cf,
                        subset="bird",
                        subcategory="accipitriformes",
                        edit_type="wings_add",
                    )
                ]
            ),
            encoding="utf-8",
        )

        run_cmd(
            [
                "--annotations",
                str(annotations_path),
                "--original-output",
                str(original_output),
                "--cf-output",
                str(cf_output),
                "--dry-run",
            ]
        )
        assert not original_output.exists()
        assert not cf_output.exists()


def test_counting_prompt_config_schema() -> None:
    data = json.loads(COUNTING_CONFIG.read_text(encoding="utf-8"))
    assert data["schema_version"] == "sam3_prompt_config_v1"
    assert data["benchmark"] == "counting"
    categories = data["categories"]
    assert len(categories) == 27
    forbidden = ("wing", "finger", "toe", "digit", "extra", "missing", "one", "two", "three", "four", "five", "six")
    for key, spec in categories.items():
        assert set(spec) <= {"object_prompt", "fallback_object_prompts"}, key
        prompt_texts = [spec["object_prompt"], *spec.get("fallback_object_prompts", [])]
        for prompt in prompt_texts:
            prompt = prompt.lower()
            assert not any(fragment in prompt for fragment in forbidden), (key, prompt)
    assert categories["chicken_feet"]["object_prompt"] == "chicken feet"
    assert categories["Woodpecker_feet"]["object_prompt"] == "woodpecker feet"
    assert "chicken foot" in categories["chicken_feet"]["fallback_object_prompts"]
    assert "bird claws" in categories["chicken_feet"]["fallback_object_prompts"]
    assert "bird foot illustration" in categories["Woodpecker_feet"]["fallback_object_prompts"]
    assert "talons" in categories["Woodpecker_feet_ai"]["fallback_object_prompts"]
    assert categories["panda_paw"]["fallback_object_prompts"] == [
        "bear paw",
        "panda foot",
        "bear foot",
        "paw close-up",
        "animal paw",
    ]
    assert categories["chimpanzee_hand_ai"]["fallback_object_prompts"] == ["primate hand", "hand"]


def main() -> int:
    tests = [
        test_synthetic_manifest_generation,
        test_dry_run_does_not_write_outputs,
        test_counting_prompt_config_schema,
    ]
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
