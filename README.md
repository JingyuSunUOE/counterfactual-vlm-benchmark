# Counterfactual VLM Benchmark

A counterfactual vision-language model benchmark for testing whether VLMs answer from visible evidence or from language and category priors, and whether agentic vision tools help reduce that bias.

This repository contains evaluation code, question definitions, metadata schemas, visual-evidence tooling, and experiment plans for five benchmark domains:

- `if_exist`: expected object parts are removed.
- `counting`: countable body parts are added or removed.
- `fashion`: logo and monogram layouts are altered.
- `industry`: common object colors, patterns, orders, and configurations are altered.
- `medical_modality`: tumor-region MRI modality is swapped against the surrounding sequence.

The project supports closed-source VLM APIs and open-source VLMs served through an OpenAI-compatible server. It also includes checkpointed evaluation, resume support, structured-output scoring, result aggregation, and agentic vision evidence generation.

## Why This Benchmark Exists

Many VLMs answer visual questions using what they expect to see. For example, if a camel image has its hump removed, a model may still answer as if the hump is present because camels are strongly associated with humped bodies.

This benchmark studies that failure mode with paired original and counterfactual images. It also evaluates a newer question: can extra visual views, such as bounding boxes, crops, zoom panels, outlines, or medical contours, help models rely more on visual evidence and less on priors?

The goal is not to propose a new agentic vision method. The goal is to evaluate which visual tool conditions help which models under controlled counterfactual settings.

## Benchmark Overview

Counts below refer to the current active metadata in this repository. `Originals` counts unique original images; `CF pairs` counts counterfactual metadata records.

| Benchmark | Visual prior being tested | Originals | CF pairs | AI-related originals | Main evidence types |
| --- | --- | ---: | ---: | ---: | --- |
| `if_exist` | Objects should contain canonical parts | 160 | 160 | 80/160 = 50.0% | `bbox`, `crop`, `zoom_panel` |
| `counting` | Objects should have normal part counts | 178 | 256 | 50/178 = 28.1% | `bbox`, `crop`, `zoom_panel`, `outline` |
| `fashion` | Logos and monograms should follow canonical layouts | 76 | 455 | 37/76 = 48.7% | `bbox`, `crop`, `zoom_panel` |
| `industry` | Common objects should have canonical colors, patterns, or orders | 77 | 462 | 37/77 = 48.1% | `bbox`, `crop`, `zoom_panel` |
| `medical_modality` | Tumor appearance should match global MRI sequence appearance | 2502 | 2502 | 0/2502 = 0.0% | `bbox`, `contour`, `crop`, `zoom_panel` |

Medical data is treated as a separate domain and is not stored under `dataset/` or `cf_dataset/`. Formal medical experiments use the `clean` BraTS2023 modality-swap pool and typically run on a stratified 10% sample.

## Task Format

Each non-medical benchmark uses paired images:

- `original`: the unmodified image.
- `counterfactual`: an edited image where a salient expected feature or visual property has been changed.

The evaluation supports three input modes:

| Input mode | Meaning | Main use |
| --- | --- | --- |
| `orig_only` | Show only the original image | Sanity check that the original image and question are answerable |
| `cf_only` | Show only the counterfactual image | Main bias test |
| `both` | Show an original/counterfactual pair or image groups, depending on condition | Tests whether extra context helps |

The evaluation uses three question types:

| Question type | Scoring |
| --- | --- |
| `yes_no` | Parsed or judged against the question-level target |
| `multiple_choice` | Strict option parsing and deterministic scoring |
| `open` | Rubric-based judge scoring |

Model answers are mapped to three primary labels:

| Label | Meaning |
| --- | --- |
| `correct` | The answer follows the visible evidence for the current question |
| `biased` | The answer follows the expected prior or canonical appearance instead of the image |
| `other` | The answer is ambiguous, unsupported, contradictory, unparseable, or visually undecidable |

Errors such as API failures, quota stops, empty responses after retry, and runtime failures are tracked separately.

## Agentic Vision Tool Conditions

Tool-condition experiments evaluate whether extra visual views help reduce prior-driven answers.

| Tool condition | Input form | Intended role |
| --- | --- | --- |
| `raw` | Base image only | Baseline |
| `bbox` | Base image plus bounding-box overlay | Localize the relevant object or region |
| `crop` | Base image plus object/region crop | Show local detail |
| `zoom_panel` | Base image plus full-image panel with local zoom | Preserve context and detail |
| `outline` | Base image plus object contour overlay | Counting-specific object boundary cue |
| `contour` | Base image plus tumor contour overlay | Medical label-derived tumor localization cue |

Main tool experiments do not use part prompts such as `wing`, `finger`, or `toe`, because they would directly reveal the target attribute. Medical evidence is derived from BraTS segmentation labels rather than SAM, because natural-image segmentation models are not appropriate for MRI tumor localization.

## Repository Layout

```text
.
  dataset/                  Downloaded original images for non-medical benchmarks
  cf_dataset/               Downloaded counterfactual images plus tracked question JSON files
  medical/                  Medical-domain workspace; raw BraTS data is not tracked
  vision_dataset/           Downloaded generated visual evidence; not tracked by Git
  vision_configs/           SAM prompt configs for evidence generation
  gen_code/                 Dataset preparation and evidence generation scripts
  eval_code/                Evaluation runners, analysis, and server wrapper
  eval_results/             Local/generated metadata, reports, raw runs, tables, and figures
  scripts/                  Hugging Face data upload/download helpers
  test/                     Non-live schema, CLI, and analysis tests
```

For GitHub distribution, large image assets, visual evidence, metadata, reports, raw API runs, generated figures, and restricted medical source data are excluded from Git. Question JSON files, code, configs, tests, and documentation are tracked.

## Installation

Create an environment and install the project in editable mode:

```bash
pip install -e .
```

Closed-model evaluation requires provider API keys, usually configured through environment variables or a local `.env` file:

```text
OPENAI_API_KEY=...
ANTHROPIC_API_KEY=...
GOOGLE_API_KEY=...
```

Do not commit `.env` files.

## Download Data

The GitHub repository does not include image data, generated visual evidence, or `eval_results/` metadata/reports. Download the data payload from the Hugging Face Dataset release into the project root:

```bash
python scripts/download_hf_dataset.py \
  --repo-id YOUR_NAME/vlm-counterfactual-benchmark-data \
  --local-dir .
```

After download, the expected layout is:

```text
dataset/
cf_dataset/
vision_dataset/
medical/modality_swapping/medical_modality_questions.json
medical/modality_swapping/standardized/      # if included in the HF release
```

The HF data package intentionally does not include `eval_results/*/metadata` or reports. Regenerate metadata with the repository scripts or obtain metadata through a separate release before running the full evaluation.

## Publish Data to Hugging Face

To publish the current local data payload to a private Hugging Face Dataset repo:

```bash
curl -LsSf https://hf.co/cli/install.sh | bash -s
```

```bash
python scripts/upload_hf_dataset.py \
  --repo-id YOUR_NAME/vlm-counterfactual-benchmark-data \
  --private \
  --overwrite-staging
```

The upload script reads `HF_TOKEN` from `.env`, builds `hf_dataset_release/`, writes a dataset card and `MANIFEST.json`, then calls `hf upload-large-folder` with progress output. It excludes `.env`, `eval_results/`, raw BraTS NIfTI data, local environments, and caches. Inspect the private HF repo before making it public.

## Quick Start

Closed-model dry run on `if_exist`:

```bash
python eval_code/if_exist/eval_closed_vlm.py \
  --input-mode cf_only \
  --backbone-model gpt-5 \
  --judge-model gpt-4o-mini \
  --tool-condition raw \
  --question-types all \
  --source-variant both \
  --prime off \
  --max-samples 2 \
  --dry-run
```

Closed-model tool-condition dry run:

```bash
python eval_code/if_exist/eval_closed_vlm.py \
  --input-mode cf_only \
  --backbone-model gpt-5 \
  --judge-model gpt-4o-mini \
  --tool-condition crop \
  --evidence-manifest eval_results/if_exist/metadata/if_exist_sam3_object_evidence_manifest.json \
  --evidence-qc-filter pass_review \
  --missing-evidence-policy skip \
  --image-group-labels on \
  --question-types all \
  --source-variant both \
  --prime off \
  --max-samples 2 \
  --dry-run
```

Open-source VLMs are evaluated through an OpenAI-compatible server. The wrapper can launch local backbone and judge servers and automatically plan GPU placement:

```bash
python eval_code/run_server_eval.py \
  --framework vllm \
  --model Qwen/Qwen3-VL-8B-Instruct \
  --served-model-name qwen3vl8b \
  --judge-server-model Qwen/Qwen3-8B-Instruct \
  --judge-served-model-name qwen3-8b \
  --benchmark if_exist \
  --gpu-placement-policy auto \
  --dry-run \
  -- \
  --input-mode cf_only \
  --backbone-model qwen3vl8b \
  --judge-model qwen3-8b \
  --question-types all \
  --max-samples 1 \
  --dry-run
```

## Reproducibility Workflow

The public repository is organized around a lightweight, reproducible workflow:

1. Download the image and visual-evidence payload from the Hugging Face Dataset release.
2. Install the benchmark dependencies with `pip install -e .`.
3. Regenerate or obtain metadata artifacts when exact evaluation splits are required.
4. Run closed-model or OpenAI-compatible server evaluations from `eval_code/`.
5. Aggregate results with the benchmark-specific analysis scripts.

Internal planning notes, cluster job files, and machine-specific Docker configurations are not part of the public GitHub release. This keeps the repository focused on reusable benchmark code, question definitions, data download utilities, and documentation.

## Data and Metadata

Metadata files are the source of truth once generated. They are not included in the GitHub repository or the default HF data package under the current release policy. Do not infer formal evaluation splits by scanning directories.

| Benchmark | Main metadata |
| --- | --- |
| `if_exist` | `eval_results/if_exist/metadata/if_exist_cf_metadata.json` |
| `counting` | `eval_results/counting/metadata/counting_count_annotations.json` |
| `fashion` | `eval_results/fashion_industry/metadata/fashion_cf_metadata.json` |
| `industry` | `eval_results/fashion_industry/metadata/industry_cf_metadata.json` |
| `medical_modality` | `eval_results/medical_modality/metadata/medical_modality_metadata.json` |

See `dataset/README.md` for original-image data and `cf_dataset/README.md` for counterfactual-image data. If the metadata paths above are missing after a fresh clone, regenerate them with the scripts under `gen_code/` and `eval_code/`, or obtain them from a separate metadata release.

## Evaluation Metrics

Primary metrics are computed over non-error responses:

- `Accuracy`: percentage of `correct` responses.
- `Bias Rate`: percentage of `biased` responses.
- `Other Rate`: percentage of `other` responses.
- `Error Count`: API, parsing, quota, judge, or runtime failures.

For tool experiments, compare each tool condition against the matching raw baseline:

- `Delta Accuracy = Tool Accuracy - Raw Accuracy`
- `Delta Bias = Tool Bias Rate - Raw Bias Rate`
- `Delta Other = Tool Other Rate - Raw Other Rate`

Raw-vs-tool comparisons should use the same model, benchmark, input mode, question type, source variant, QC filter, and sample subset.

## Validation

Useful non-live checks:

```bash
python -m py_compile \
  eval_code/if_exist/eval_closed_vlm.py \
  eval_code/counting/eval_closed_vlm.py \
  eval_code/fashion_industry/eval_pipeline.py \
  eval_code/medical_modality/eval_closed_vlm.py \
  segmentation_sam3.py
```

```bash
python test/if_exist/test_eval_cli.py
python test/counting/test_eval_cli.py
python test/fashion_industry/test_eval_cli.py
python test/medical_modality/test_eval_cli.py
python test/vision_tools/test_segmentation_sam3_cli.py
```

## Release Notes for GitHub

Before publishing:

- Keep `.env`, `.venv/`, `.idea/`, provider caches, and raw API outputs out of Git.
- Do not commit restricted BraTS source data or NIfTI files.
- Do not commit generated `vision_dataset/` evidence, image data, metadata, or reports to GitHub under the current release policy.
- Publish large image assets through the Hugging Face Dataset release helper, or use Zenodo/Git LFS only with clear licensing.
- Medical generated images should only be published after confirming source-data license constraints; `medical/BraTS2023_GLI/` must remain excluded.

## License and Citation

License and citation details should be finalized before public release.
