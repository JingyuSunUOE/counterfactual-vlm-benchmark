# Original Image Datasets

This directory contains original images for the non-medical benchmark domains. Each original image is paired with one or more counterfactual images through metadata files and question definitions.

Medical data is intentionally not stored here. The medical benchmark is a separate MRI-domain task under `medical/`, with metadata under `eval_results/medical_modality/metadata/`.

## Directory Layout

```text
dataset/
  if_exist/             Original images for removed-part counterfactuals
  counting/             Original images for count-changing counterfactuals
  fashion_dataset/      Original fashion logo and monogram images
  industry_dataset/     Original common-object and industry-domain images
```

Do not infer formal evaluation samples by scanning this directory alone. The official source of truth is the metadata listed below.

## Active Dataset Statistics

Counts reflect the current active metadata and manually filtered image directories.

| Dataset directory | Task focus | Original images | AI-related originals | Main subcategories |
| --- | --- | ---: | ---: | --- |
| `if_exist` | Expected object parts | 160 | 80/160 = 50.0% | camel, elephant trunk, fish fin, rabbit |
| `counting` | Countable body parts | 178 | 50/178 = 28.1% | bird, insect, hand/paw |
| `fashion_dataset` | Logo and monogram layouts | 76 | 37/76 = 48.7% | Burberry, Celine, Chanel, Gucci, Loewe, LV, Ralph Lauren, YSL |
| `industry_dataset` | Common object colors, patterns, and orders | 58 | 28/58 = 48.3% | balls, signs, traffic lights, cards, keyboards |

`AI-related originals` uses `source_variant=ai` for `if_exist` and `counting`. For `fashion_dataset` and `industry_dataset`, it is inferred from paired metadata where the original source is marked by `source_type=cf_from_ai`.

## Source-of-Truth Metadata

Use these files for active samples and original-to-counterfactual pairing:

| Benchmark | Metadata |
| --- | --- |
| `if_exist` | `eval_results/if_exist/metadata/if_exist_cf_metadata.json` |
| `counting` | `eval_results/counting/metadata/counting_count_annotations.json` |
| `fashion_dataset` | `eval_results/fashion_industry/metadata/fashion_cf_metadata.json` |
| `industry_dataset` | `eval_results/fashion_industry/metadata/industry_cf_metadata.json` |

Important notes:

- `counting` reuses some original images across multiple counterfactual edits, so 178 originals produce 256 CF pairs.
- `fashion_dataset` and `industry_dataset` have been manually quality-filtered; current active metadata and image counts are aligned.
- Formal runners consume metadata and question definitions. They should not guess pairs from folder names.

## Dataset Details

### `if_exist`

The `if_exist` benchmark tests whether models visually verify expected object parts instead of relying on category priors.

| Subcategory | Prior | Counterfactual target |
| --- | --- | --- |
| `camel` | Camels normally have humps | Hump removed |
| `elephant_trunk` | Elephants normally have trunks | Trunk removed |
| `fish_fin` | Fish normally have visible fins | Upper fin removed |
| `rabbit` | Rabbits normally have long ears | Long ears removed |

The active set has 160 originals: 80 real-source images and 80 AI-generated source images.

### `counting`

The `counting` benchmark tests whether models count visible parts rather than answering with normal-count priors.

| Subset | Original images | Counterfactual focus |
| --- | ---: | --- |
| `bird` | 45 | Wing count |
| `insect` | 33 | Wing count |
| `hand_paw` | 100 | Fingers, toes, or digits |
| **Total** | **178** |  |

Counting originals are standardized to `512x512` RGB PNG using aspect-ratio-preserving white padding. They are not stretched or center-cropped, because counting cues often appear near image edges.

### `fashion_dataset`

The `fashion` benchmark tests priors about canonical logo and monogram structures.

Active brand folders:

```text
burberry, celine, chanel, gucci_gg, loewe, lv, ralph_lauren, ysl
```

The active set has 76 originals after manual quality filtering. Counterfactual edits target properties such as orientation, element count, symmetry, separation, layer arrangement, and monogram order.

### `industry_dataset`

The `industry` benchmark tests priors about common object appearance.

Active category folders:

```text
basketball, piano_keyboard, poker_cards, road_sign, tennis_ball, traffic_light
```

The active set has 58 originals after manual quality filtering. Counterfactual edits target properties such as color, pattern, order, layout, and object configuration.

## Image Normalization

| Dataset | Format notes |
| --- | --- |
| `counting` | Standardized to `512x512` RGB PNG with white padding |
| `if_exist` | Curated benchmark images; formal pairing is metadata-driven |
| `fashion_dataset` | Curated brand/logo images; manually filtered active set |
| `industry_dataset` | Curated common-object images; manually filtered active set |

If you regenerate or replace images, update the corresponding metadata and README statistics before running evaluations.

## Relationship to `cf_dataset/`

Original images in this directory are paired with counterfactual images in `cf_dataset/`:

| Original directory | Counterfactual directory |
| --- | --- |
| `dataset/if_exist` | `cf_dataset/if_exist_cf` |
| `dataset/counting` | `cf_dataset/counting_cf` |
| `dataset/fashion_dataset` | `cf_dataset/fashion_cf` |
| `dataset/industry_dataset` | `cf_dataset/industry_cf` |

See `cf_dataset/README.md` for counterfactual images, question files, and answer semantics.

## Usage Notes

- These images are intended for evaluation, not model training.
- Use metadata as the official split and pairing source.
- Do not randomly mix files from this directory into a benchmark split unless metadata confirms they are active samples.
- If publishing the project on GitHub, consider distributing large image assets separately through a dataset release, Hugging Face Dataset, Zenodo, or Git LFS.
- Third-party source and redistribution constraints should be reviewed before public release.
