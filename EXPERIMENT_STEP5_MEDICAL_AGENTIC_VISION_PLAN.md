# 第五阶段实验计划：Medical Modality Label-Derived Vision Tools

本文档是内部实验执行计划，只覆盖 `medical_modality` 这一类 benchmark。它用于第五阶段 agentic vision tool 实验，不作为论文正文、数据集 README 或开源首页文件。

## 1. 实验目的

第五阶段的核心问题是：基于 BraTS segmentation label 生成的视觉证据，是否能帮助 VLM 克服 medical modality 任务中的 global MRI appearance / surrounding sequence prior。

Medical 与自然图像类别不同：

- 视觉证据不是由 SAM 生成，而是由 BraTS 原始 segmentation label 生成。
- 工具的目标不是识别物体，而是定位 tumor region。
- 任务不是临床诊断，而是判断 tumor-region modality/contrast 是否与 surrounding MRI sequence 一致。
- 主实验只使用 `clean` visualization，避免红色 overlay 或其它可视化差异引入 confound。

我们在第一阶段已经测试过 `raw + clean + brats2023 + 10%` 条件，用于证明 benchmark 问题成立。因此本阶段不再重复 `raw` run。本阶段只测试 label-derived 非 raw 视觉证据。

主要指标仍为：

- `correct`：回答与 tumor-region modality consistency 的可见事实一致。
- `biased`：回答符合 global MRI appearance / surrounding sequence prior，但忽略 tumor region 的局部 modality conflict。
- `other`：回答模糊、不可判定、格式错误、无可见 signal cue，或不属于 correct/biased。

## 2. 实验优先级

### 2.1 主实验：优先执行

主实验只测试 `cf_only`，并固定使用 `clean + brats2023 + 10% stratified sample`。

| Dataset | Visual Version | Input Mode | Tool Condition | 优先级 | 目的 |
| --- | --- | --- | --- | --- | --- |
| `brats2023` | `clean` | `cf_only` | `contour` | P0 | 用 tumor boundary 定位目标区域，不遮挡 intensity |
| `brats2023` | `clean` | `cf_only` | `zoom_panel` | P0 | 保留全图上下文，同时放大 tumor region |

### 2.2 低优先级：资源允许后执行

| Dataset | Visual Version | Input Mode | Tool Condition | 优先级 | 说明 |
| --- | --- | --- | --- | --- | --- |
| `brats2023` | `clean` | `cf_only` | `bbox` | P1 | localization control，不遮挡 intensity |
| `brats2023` | `clean` | `cf_only` | `crop` | P1 | 放大 tumor region，但可能丢失 surrounding sequence context |
| `brats2023` | `clean` | `both` | `contour` | P2 | 测试定位边界是否帮助比较 original/CF |
| `brats2023` | `clean` | `both` | `zoom_panel` | P2 | 测试全图 + 局部视图是否帮助比较差异 |
| `brats2023` | `clean` | `both` | `bbox` | P3 | 最低优先级定位框 ablation |
| `brats2023` | `clean` | `both` | `crop` | P3 | 最低优先级；可能缺少上下文 |

本阶段不测试 `orig_only`，也不重复 `raw`。第一阶段 raw 结果作为本阶段的 baseline；第五阶段只运行非 raw tool conditions。

## 3. 数据与 Evidence 设置

Medical tool evidence 使用 label-derived evidence manifest：

```bash
MEDICAL_EVIDENCE_MANIFEST=eval_results/medical_modality/metadata/medical_label_evidence_manifest_clean.json
MEDICAL_QC_FILTER=pass
MEDICAL_MISSING_POLICY=skip
MEDICAL_DATASET=brats2023
MEDICAL_VISUAL_VERSION=clean
MEDICAL_SAMPLE_FRACTION=0.10
MEDICAL_SAMPLE_STRATEGY=stratified
```

说明：

- `pass` 是正式默认，因为 evidence 来自 BraTS label，而不是视觉模型预测。
- `skip` 表示缺失 tool evidence 的 pair 不发 API，不写 response record，不污染 denominator。
- 本阶段不新跑 raw；正式分析时使用第一阶段 `raw + clean + brats2023 + 10%` baseline。
- 如果 sampling seed 或 matched subset 后续需要严格复现，应在 runner 支持后显式加 `--seed`，并让 raw/tool 使用同一采样集合。

## 4. 通用评测设置

统一设置：

```text
dataset = brats2023
visual_version = clean
sample_fraction = 0.10
sample_strategy = stratified
question_types = all
prime = off
temperature = 0
max_output_tokens = 2048
judge_max_output_tokens = 512
judge_structured_output = auto
closed_form_structured_output = auto
reasoning_effort = off
evidence_qc_filter = pass
missing_evidence_policy = skip
```

`prime off` 不表示没有语言先验；medical 的问题 prompt 本身已经包含 MRI modality prior / consistency framing。

## 5. 模型变量

### 5.1 闭源模型

每次运行前先复制其中一个模型变量块。

GPT-5：

```bash
MODEL_ALIAS=gpt-5
MODEL_TAG=gpt5
JUDGE_MODEL=gpt-4o-mini
```

Claude Sonnet 4：

```bash
MODEL_ALIAS=claude-sonnet-4
MODEL_TAG=claude_sonnet4
JUDGE_MODEL=gpt-4o-mini
```

Gemini 3.1 Pro Preview：

```bash
MODEL_ALIAS=gemini-3.1-pro-preview
MODEL_TAG=gemini31pro_preview
JUDGE_MODEL=gpt-4o-mini
```

### 5.2 开源模型

开源模型默认使用 `eval_code/run_server_eval.py` 自动启动 OpenAI-compatible backbone server 和 text-only judge server。不要在 `--` 后手动传 `--server-url`、`--server-model-id`、`--judge-provider`、`--judge-model` 或 `--judge-server-url`。

通用 server 变量：

```bash
FRAMEWORK=vllm
DTYPE=bfloat16
GPU_MEMORY_UTILIZATION=0.90
JUDGE_GPU_MEMORY_UTILIZATION=0.80
GPU_PLACEMENT_POLICY=auto
SHARED_GPU_TOTAL_UTILIZATION=0.88
JUDGE_HF_MODEL_ID=Qwen/Qwen3-8B-Instruct
JUDGE_SERVED_MODEL_NAME=qwen3-8b
```

然后复制其中一个 backbone 模型变量块。

Qwen3-VL 8B：

```bash
HF_MODEL_ID=Qwen/Qwen3-VL-8B-Instruct
SERVED_MODEL_NAME=qwen3vl8b
MODEL_ALIAS=qwen3vl8b
MODEL_TAG=qwen3vl8b
```

Qwen3-VL 32B：

```bash
HF_MODEL_ID=Qwen/Qwen3-VL-32B-Instruct
SERVED_MODEL_NAME=qwen3vl32b
MODEL_ALIAS=qwen3vl32b
MODEL_TAG=qwen3vl32b
```

InternVL3.5 8B：

```bash
HF_MODEL_ID=OpenGVLab/InternVL3_5-8B-HF
SERVED_MODEL_NAME=internvl35-8b
MODEL_ALIAS=internvl35-8b
MODEL_TAG=internvl35_8b
```

InternVL3.5 38B-HF：

```bash
HF_MODEL_ID=OpenGVLab/InternVL3_5-38B-HF
SERVED_MODEL_NAME=internvl35-38b-hf
MODEL_ALIAS=internvl35-38b-hf
MODEL_TAG=internvl35_38b_hf
```

Gemma 4 E4B-it：

```bash
HF_MODEL_ID=google/gemma-4-E4B-it
SERVED_MODEL_NAME=gemma4-e4b-it
MODEL_ALIAS=gemma4-e4b-it
MODEL_TAG=gemma4_e4b_it
```

Gemma 4 31B-it：

```bash
HF_MODEL_ID=google/gemma-4-31B-it
SERVED_MODEL_NAME=gemma4-31b-it
MODEL_ALIAS=gemma4-31b-it
MODEL_TAG=gemma4_31b_it
```

Qwen2.5-VL 72B Instruct upper-bound：

```bash
HF_MODEL_ID=Qwen/Qwen2.5-VL-72B-Instruct
SERVED_MODEL_NAME=qwen25vl72b
MODEL_ALIAS=qwen25vl72b
MODEL_TAG=qwen25vl72b
```

## 6. 闭源模型命令

以下命令每个都需要分别对 GPT-5、Claude Sonnet 4、Gemini 3.1 Pro Preview 三个变量块执行。

### 6.1 P0：cf_only contour

```bash
python eval_code/medical_modality/eval_closed_vlm.py \
  --input-mode cf_only \
  --backbone-model "$MODEL_ALIAS" \
  --judge-model "$JUDGE_MODEL" \
  --visual-version "$MEDICAL_VISUAL_VERSION" \
  --dataset "$MEDICAL_DATASET" \
  --sample-fraction "$MEDICAL_SAMPLE_FRACTION" \
  --sample-strategy "$MEDICAL_SAMPLE_STRATEGY" \
  --question-types all \
  --tool-condition contour \
  --evidence-manifest "$MEDICAL_EVIDENCE_MANIFEST" \
  --evidence-qc-filter "$MEDICAL_QC_FILTER" \
  --missing-evidence-policy "$MEDICAL_MISSING_POLICY" \
  --image-group-labels on \
  --prime off \
  --temperature 0 \
  --max-output-tokens 2048 \
  --judge-max-output-tokens 512 \
  --judge-structured-output auto \
  --closed-form-structured-output auto \
  --reasoning-effort off \
  --provider-cache auto \
  --cache-key-mode image_pair \
  --resume off \
  --output-root "eval_results/medical_modality/raw_runs/${MODEL_TAG}_cf_only_tool_contour_clean_brats2023_10pct_v1"
```

### 6.2 P0：cf_only zoom_panel

```bash
python eval_code/medical_modality/eval_closed_vlm.py \
  --input-mode cf_only \
  --backbone-model "$MODEL_ALIAS" \
  --judge-model "$JUDGE_MODEL" \
  --visual-version "$MEDICAL_VISUAL_VERSION" \
  --dataset "$MEDICAL_DATASET" \
  --sample-fraction "$MEDICAL_SAMPLE_FRACTION" \
  --sample-strategy "$MEDICAL_SAMPLE_STRATEGY" \
  --question-types all \
  --tool-condition zoom_panel \
  --evidence-manifest "$MEDICAL_EVIDENCE_MANIFEST" \
  --evidence-qc-filter "$MEDICAL_QC_FILTER" \
  --missing-evidence-policy "$MEDICAL_MISSING_POLICY" \
  --image-group-labels on \
  --prime off \
  --temperature 0 \
  --max-output-tokens 2048 \
  --judge-max-output-tokens 512 \
  --judge-structured-output auto \
  --closed-form-structured-output auto \
  --reasoning-effort off \
  --provider-cache auto \
  --cache-key-mode image_pair \
  --resume off \
  --output-root "eval_results/medical_modality/raw_runs/${MODEL_TAG}_cf_only_tool_zoom_panel_clean_brats2023_10pct_v1"
```

### 6.3 P1：cf_only bbox

```bash
python eval_code/medical_modality/eval_closed_vlm.py \
  --input-mode cf_only \
  --backbone-model "$MODEL_ALIAS" \
  --judge-model "$JUDGE_MODEL" \
  --visual-version "$MEDICAL_VISUAL_VERSION" \
  --dataset "$MEDICAL_DATASET" \
  --sample-fraction "$MEDICAL_SAMPLE_FRACTION" \
  --sample-strategy "$MEDICAL_SAMPLE_STRATEGY" \
  --question-types all \
  --tool-condition bbox \
  --evidence-manifest "$MEDICAL_EVIDENCE_MANIFEST" \
  --evidence-qc-filter "$MEDICAL_QC_FILTER" \
  --missing-evidence-policy "$MEDICAL_MISSING_POLICY" \
  --image-group-labels on \
  --prime off \
  --temperature 0 \
  --max-output-tokens 2048 \
  --judge-max-output-tokens 512 \
  --judge-structured-output auto \
  --closed-form-structured-output auto \
  --reasoning-effort off \
  --provider-cache auto \
  --cache-key-mode image_pair \
  --resume off \
  --output-root "eval_results/medical_modality/raw_runs/${MODEL_TAG}_cf_only_tool_bbox_clean_brats2023_10pct_low_priority_v1"
```

### 6.4 P1：cf_only crop

```bash
python eval_code/medical_modality/eval_closed_vlm.py \
  --input-mode cf_only \
  --backbone-model "$MODEL_ALIAS" \
  --judge-model "$JUDGE_MODEL" \
  --visual-version "$MEDICAL_VISUAL_VERSION" \
  --dataset "$MEDICAL_DATASET" \
  --sample-fraction "$MEDICAL_SAMPLE_FRACTION" \
  --sample-strategy "$MEDICAL_SAMPLE_STRATEGY" \
  --question-types all \
  --tool-condition crop \
  --evidence-manifest "$MEDICAL_EVIDENCE_MANIFEST" \
  --evidence-qc-filter "$MEDICAL_QC_FILTER" \
  --missing-evidence-policy "$MEDICAL_MISSING_POLICY" \
  --image-group-labels on \
  --prime off \
  --temperature 0 \
  --max-output-tokens 2048 \
  --judge-max-output-tokens 512 \
  --judge-structured-output auto \
  --closed-form-structured-output auto \
  --reasoning-effort off \
  --provider-cache auto \
  --cache-key-mode image_pair \
  --resume off \
  --output-root "eval_results/medical_modality/raw_runs/${MODEL_TAG}_cf_only_tool_crop_clean_brats2023_10pct_low_priority_v1"
```

### 6.5 P2：both contour

```bash
python eval_code/medical_modality/eval_closed_vlm.py \
  --input-mode both \
  --backbone-model "$MODEL_ALIAS" \
  --judge-model "$JUDGE_MODEL" \
  --visual-version "$MEDICAL_VISUAL_VERSION" \
  --dataset "$MEDICAL_DATASET" \
  --sample-fraction "$MEDICAL_SAMPLE_FRACTION" \
  --sample-strategy "$MEDICAL_SAMPLE_STRATEGY" \
  --question-types all \
  --tool-condition contour \
  --evidence-manifest "$MEDICAL_EVIDENCE_MANIFEST" \
  --evidence-qc-filter "$MEDICAL_QC_FILTER" \
  --missing-evidence-policy "$MEDICAL_MISSING_POLICY" \
  --image-group-labels on \
  --prime off \
  --temperature 0 \
  --max-output-tokens 2048 \
  --judge-max-output-tokens 512 \
  --judge-structured-output auto \
  --closed-form-structured-output auto \
  --reasoning-effort off \
  --provider-cache auto \
  --cache-key-mode image_pair \
  --resume off \
  --output-root "eval_results/medical_modality/raw_runs/${MODEL_TAG}_both_tool_contour_clean_brats2023_10pct_low_priority_v1"
```

### 6.6 P2：both zoom_panel

```bash
python eval_code/medical_modality/eval_closed_vlm.py \
  --input-mode both \
  --backbone-model "$MODEL_ALIAS" \
  --judge-model "$JUDGE_MODEL" \
  --visual-version "$MEDICAL_VISUAL_VERSION" \
  --dataset "$MEDICAL_DATASET" \
  --sample-fraction "$MEDICAL_SAMPLE_FRACTION" \
  --sample-strategy "$MEDICAL_SAMPLE_STRATEGY" \
  --question-types all \
  --tool-condition zoom_panel \
  --evidence-manifest "$MEDICAL_EVIDENCE_MANIFEST" \
  --evidence-qc-filter "$MEDICAL_QC_FILTER" \
  --missing-evidence-policy "$MEDICAL_MISSING_POLICY" \
  --image-group-labels on \
  --prime off \
  --temperature 0 \
  --max-output-tokens 2048 \
  --judge-max-output-tokens 512 \
  --judge-structured-output auto \
  --closed-form-structured-output auto \
  --reasoning-effort off \
  --provider-cache auto \
  --cache-key-mode image_pair \
  --resume off \
  --output-root "eval_results/medical_modality/raw_runs/${MODEL_TAG}_both_tool_zoom_panel_clean_brats2023_10pct_low_priority_v1"
```

### 6.7 P3：both bbox

```bash
python eval_code/medical_modality/eval_closed_vlm.py \
  --input-mode both \
  --backbone-model "$MODEL_ALIAS" \
  --judge-model "$JUDGE_MODEL" \
  --visual-version "$MEDICAL_VISUAL_VERSION" \
  --dataset "$MEDICAL_DATASET" \
  --sample-fraction "$MEDICAL_SAMPLE_FRACTION" \
  --sample-strategy "$MEDICAL_SAMPLE_STRATEGY" \
  --question-types all \
  --tool-condition bbox \
  --evidence-manifest "$MEDICAL_EVIDENCE_MANIFEST" \
  --evidence-qc-filter "$MEDICAL_QC_FILTER" \
  --missing-evidence-policy "$MEDICAL_MISSING_POLICY" \
  --image-group-labels on \
  --prime off \
  --temperature 0 \
  --max-output-tokens 2048 \
  --judge-max-output-tokens 512 \
  --judge-structured-output auto \
  --closed-form-structured-output auto \
  --reasoning-effort off \
  --provider-cache auto \
  --cache-key-mode image_pair \
  --resume off \
  --output-root "eval_results/medical_modality/raw_runs/${MODEL_TAG}_both_tool_bbox_clean_brats2023_10pct_low_priority_v1"
```

### 6.8 P3：both crop

```bash
python eval_code/medical_modality/eval_closed_vlm.py \
  --input-mode both \
  --backbone-model "$MODEL_ALIAS" \
  --judge-model "$JUDGE_MODEL" \
  --visual-version "$MEDICAL_VISUAL_VERSION" \
  --dataset "$MEDICAL_DATASET" \
  --sample-fraction "$MEDICAL_SAMPLE_FRACTION" \
  --sample-strategy "$MEDICAL_SAMPLE_STRATEGY" \
  --question-types all \
  --tool-condition crop \
  --evidence-manifest "$MEDICAL_EVIDENCE_MANIFEST" \
  --evidence-qc-filter "$MEDICAL_QC_FILTER" \
  --missing-evidence-policy "$MEDICAL_MISSING_POLICY" \
  --image-group-labels on \
  --prime off \
  --temperature 0 \
  --max-output-tokens 2048 \
  --judge-max-output-tokens 512 \
  --judge-structured-output auto \
  --closed-form-structured-output auto \
  --reasoning-effort off \
  --provider-cache auto \
  --cache-key-mode image_pair \
  --resume off \
  --output-root "eval_results/medical_modality/raw_runs/${MODEL_TAG}_both_tool_crop_clean_brats2023_10pct_low_priority_v1"
```

## 7. 开源模型命令

以下命令每个都需要分别对所有开源模型变量块执行。wrapper 会自动启动 backbone server 和 judge server，并自动规划 GPU placement。

### 7.1 P0：cf_only contour

```bash
python eval_code/run_server_eval.py \
  --framework "$FRAMEWORK" \
  --model "$HF_MODEL_ID" \
  --served-model-name "$SERVED_MODEL_NAME" \
  --benchmark medical_modality \
  --dtype "$DTYPE" \
  --tensor-parallel-size auto \
  --gpu-placement-policy "$GPU_PLACEMENT_POLICY" \
  --shared-gpu-total-utilization "$SHARED_GPU_TOTAL_UTILIZATION" \
  --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
  --judge-server-model "$JUDGE_HF_MODEL_ID" \
  --judge-served-model-name "$JUDGE_SERVED_MODEL_NAME" \
  --judge-gpu-memory-utilization "$JUDGE_GPU_MEMORY_UTILIZATION" \
  --if-server-running fail \
  -- \
  --input-mode cf_only \
  --backbone-model "$MODEL_ALIAS" \
  --visual-version "$MEDICAL_VISUAL_VERSION" \
  --dataset "$MEDICAL_DATASET" \
  --sample-fraction "$MEDICAL_SAMPLE_FRACTION" \
  --sample-strategy "$MEDICAL_SAMPLE_STRATEGY" \
  --question-types all \
  --tool-condition contour \
  --evidence-manifest "$MEDICAL_EVIDENCE_MANIFEST" \
  --evidence-qc-filter "$MEDICAL_QC_FILTER" \
  --missing-evidence-policy "$MEDICAL_MISSING_POLICY" \
  --image-group-labels on \
  --prime off \
  --temperature 0 \
  --max-output-tokens 2048 \
  --judge-max-output-tokens 512 \
  --judge-structured-output auto \
  --closed-form-structured-output auto \
  --reasoning-effort off \
  --resume off \
  --output-root "eval_results/medical_modality/raw_runs/${MODEL_TAG}_cf_only_tool_contour_clean_brats2023_10pct_v1"
```

### 7.2 P0：cf_only zoom_panel

```bash
python eval_code/run_server_eval.py \
  --framework "$FRAMEWORK" \
  --model "$HF_MODEL_ID" \
  --served-model-name "$SERVED_MODEL_NAME" \
  --benchmark medical_modality \
  --dtype "$DTYPE" \
  --tensor-parallel-size auto \
  --gpu-placement-policy "$GPU_PLACEMENT_POLICY" \
  --shared-gpu-total-utilization "$SHARED_GPU_TOTAL_UTILIZATION" \
  --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
  --judge-server-model "$JUDGE_HF_MODEL_ID" \
  --judge-served-model-name "$JUDGE_SERVED_MODEL_NAME" \
  --judge-gpu-memory-utilization "$JUDGE_GPU_MEMORY_UTILIZATION" \
  --if-server-running fail \
  -- \
  --input-mode cf_only \
  --backbone-model "$MODEL_ALIAS" \
  --visual-version "$MEDICAL_VISUAL_VERSION" \
  --dataset "$MEDICAL_DATASET" \
  --sample-fraction "$MEDICAL_SAMPLE_FRACTION" \
  --sample-strategy "$MEDICAL_SAMPLE_STRATEGY" \
  --question-types all \
  --tool-condition zoom_panel \
  --evidence-manifest "$MEDICAL_EVIDENCE_MANIFEST" \
  --evidence-qc-filter "$MEDICAL_QC_FILTER" \
  --missing-evidence-policy "$MEDICAL_MISSING_POLICY" \
  --image-group-labels on \
  --prime off \
  --temperature 0 \
  --max-output-tokens 2048 \
  --judge-max-output-tokens 512 \
  --judge-structured-output auto \
  --closed-form-structured-output auto \
  --reasoning-effort off \
  --resume off \
  --output-root "eval_results/medical_modality/raw_runs/${MODEL_TAG}_cf_only_tool_zoom_panel_clean_brats2023_10pct_v1"
```

### 7.3 P1：cf_only bbox

```bash
python eval_code/run_server_eval.py \
  --framework "$FRAMEWORK" \
  --model "$HF_MODEL_ID" \
  --served-model-name "$SERVED_MODEL_NAME" \
  --benchmark medical_modality \
  --dtype "$DTYPE" \
  --tensor-parallel-size auto \
  --gpu-placement-policy "$GPU_PLACEMENT_POLICY" \
  --shared-gpu-total-utilization "$SHARED_GPU_TOTAL_UTILIZATION" \
  --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
  --judge-server-model "$JUDGE_HF_MODEL_ID" \
  --judge-served-model-name "$JUDGE_SERVED_MODEL_NAME" \
  --judge-gpu-memory-utilization "$JUDGE_GPU_MEMORY_UTILIZATION" \
  --if-server-running fail \
  -- \
  --input-mode cf_only \
  --backbone-model "$MODEL_ALIAS" \
  --visual-version "$MEDICAL_VISUAL_VERSION" \
  --dataset "$MEDICAL_DATASET" \
  --sample-fraction "$MEDICAL_SAMPLE_FRACTION" \
  --sample-strategy "$MEDICAL_SAMPLE_STRATEGY" \
  --question-types all \
  --tool-condition bbox \
  --evidence-manifest "$MEDICAL_EVIDENCE_MANIFEST" \
  --evidence-qc-filter "$MEDICAL_QC_FILTER" \
  --missing-evidence-policy "$MEDICAL_MISSING_POLICY" \
  --image-group-labels on \
  --prime off \
  --temperature 0 \
  --max-output-tokens 2048 \
  --judge-max-output-tokens 512 \
  --judge-structured-output auto \
  --closed-form-structured-output auto \
  --reasoning-effort off \
  --resume off \
  --output-root "eval_results/medical_modality/raw_runs/${MODEL_TAG}_cf_only_tool_bbox_clean_brats2023_10pct_low_priority_v1"
```

### 7.4 P1：cf_only crop

```bash
python eval_code/run_server_eval.py \
  --framework "$FRAMEWORK" \
  --model "$HF_MODEL_ID" \
  --served-model-name "$SERVED_MODEL_NAME" \
  --benchmark medical_modality \
  --dtype "$DTYPE" \
  --tensor-parallel-size auto \
  --gpu-placement-policy "$GPU_PLACEMENT_POLICY" \
  --shared-gpu-total-utilization "$SHARED_GPU_TOTAL_UTILIZATION" \
  --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
  --judge-server-model "$JUDGE_HF_MODEL_ID" \
  --judge-served-model-name "$JUDGE_SERVED_MODEL_NAME" \
  --judge-gpu-memory-utilization "$JUDGE_GPU_MEMORY_UTILIZATION" \
  --if-server-running fail \
  -- \
  --input-mode cf_only \
  --backbone-model "$MODEL_ALIAS" \
  --visual-version "$MEDICAL_VISUAL_VERSION" \
  --dataset "$MEDICAL_DATASET" \
  --sample-fraction "$MEDICAL_SAMPLE_FRACTION" \
  --sample-strategy "$MEDICAL_SAMPLE_STRATEGY" \
  --question-types all \
  --tool-condition crop \
  --evidence-manifest "$MEDICAL_EVIDENCE_MANIFEST" \
  --evidence-qc-filter "$MEDICAL_QC_FILTER" \
  --missing-evidence-policy "$MEDICAL_MISSING_POLICY" \
  --image-group-labels on \
  --prime off \
  --temperature 0 \
  --max-output-tokens 2048 \
  --judge-max-output-tokens 512 \
  --judge-structured-output auto \
  --closed-form-structured-output auto \
  --reasoning-effort off \
  --resume off \
  --output-root "eval_results/medical_modality/raw_runs/${MODEL_TAG}_cf_only_tool_crop_clean_brats2023_10pct_low_priority_v1"
```

### 7.5 P2：both contour

```bash
python eval_code/run_server_eval.py \
  --framework "$FRAMEWORK" \
  --model "$HF_MODEL_ID" \
  --served-model-name "$SERVED_MODEL_NAME" \
  --benchmark medical_modality \
  --dtype "$DTYPE" \
  --tensor-parallel-size auto \
  --gpu-placement-policy "$GPU_PLACEMENT_POLICY" \
  --shared-gpu-total-utilization "$SHARED_GPU_TOTAL_UTILIZATION" \
  --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
  --judge-server-model "$JUDGE_HF_MODEL_ID" \
  --judge-served-model-name "$JUDGE_SERVED_MODEL_NAME" \
  --judge-gpu-memory-utilization "$JUDGE_GPU_MEMORY_UTILIZATION" \
  --if-server-running fail \
  -- \
  --input-mode both \
  --backbone-model "$MODEL_ALIAS" \
  --visual-version "$MEDICAL_VISUAL_VERSION" \
  --dataset "$MEDICAL_DATASET" \
  --sample-fraction "$MEDICAL_SAMPLE_FRACTION" \
  --sample-strategy "$MEDICAL_SAMPLE_STRATEGY" \
  --question-types all \
  --tool-condition contour \
  --evidence-manifest "$MEDICAL_EVIDENCE_MANIFEST" \
  --evidence-qc-filter "$MEDICAL_QC_FILTER" \
  --missing-evidence-policy "$MEDICAL_MISSING_POLICY" \
  --image-group-labels on \
  --prime off \
  --temperature 0 \
  --max-output-tokens 2048 \
  --judge-max-output-tokens 512 \
  --judge-structured-output auto \
  --closed-form-structured-output auto \
  --reasoning-effort off \
  --resume off \
  --output-root "eval_results/medical_modality/raw_runs/${MODEL_TAG}_both_tool_contour_clean_brats2023_10pct_low_priority_v1"
```

### 7.6 P2：both zoom_panel

```bash
python eval_code/run_server_eval.py \
  --framework "$FRAMEWORK" \
  --model "$HF_MODEL_ID" \
  --served-model-name "$SERVED_MODEL_NAME" \
  --benchmark medical_modality \
  --dtype "$DTYPE" \
  --tensor-parallel-size auto \
  --gpu-placement-policy "$GPU_PLACEMENT_POLICY" \
  --shared-gpu-total-utilization "$SHARED_GPU_TOTAL_UTILIZATION" \
  --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
  --judge-server-model "$JUDGE_HF_MODEL_ID" \
  --judge-served-model-name "$JUDGE_SERVED_MODEL_NAME" \
  --judge-gpu-memory-utilization "$JUDGE_GPU_MEMORY_UTILIZATION" \
  --if-server-running fail \
  -- \
  --input-mode both \
  --backbone-model "$MODEL_ALIAS" \
  --visual-version "$MEDICAL_VISUAL_VERSION" \
  --dataset "$MEDICAL_DATASET" \
  --sample-fraction "$MEDICAL_SAMPLE_FRACTION" \
  --sample-strategy "$MEDICAL_SAMPLE_STRATEGY" \
  --question-types all \
  --tool-condition zoom_panel \
  --evidence-manifest "$MEDICAL_EVIDENCE_MANIFEST" \
  --evidence-qc-filter "$MEDICAL_QC_FILTER" \
  --missing-evidence-policy "$MEDICAL_MISSING_POLICY" \
  --image-group-labels on \
  --prime off \
  --temperature 0 \
  --max-output-tokens 2048 \
  --judge-max-output-tokens 512 \
  --judge-structured-output auto \
  --closed-form-structured-output auto \
  --reasoning-effort off \
  --resume off \
  --output-root "eval_results/medical_modality/raw_runs/${MODEL_TAG}_both_tool_zoom_panel_clean_brats2023_10pct_low_priority_v1"
```

### 7.7 P3：both bbox

```bash
python eval_code/run_server_eval.py \
  --framework "$FRAMEWORK" \
  --model "$HF_MODEL_ID" \
  --served-model-name "$SERVED_MODEL_NAME" \
  --benchmark medical_modality \
  --dtype "$DTYPE" \
  --tensor-parallel-size auto \
  --gpu-placement-policy "$GPU_PLACEMENT_POLICY" \
  --shared-gpu-total-utilization "$SHARED_GPU_TOTAL_UTILIZATION" \
  --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
  --judge-server-model "$JUDGE_HF_MODEL_ID" \
  --judge-served-model-name "$JUDGE_SERVED_MODEL_NAME" \
  --judge-gpu-memory-utilization "$JUDGE_GPU_MEMORY_UTILIZATION" \
  --if-server-running fail \
  -- \
  --input-mode both \
  --backbone-model "$MODEL_ALIAS" \
  --visual-version "$MEDICAL_VISUAL_VERSION" \
  --dataset "$MEDICAL_DATASET" \
  --sample-fraction "$MEDICAL_SAMPLE_FRACTION" \
  --sample-strategy "$MEDICAL_SAMPLE_STRATEGY" \
  --question-types all \
  --tool-condition bbox \
  --evidence-manifest "$MEDICAL_EVIDENCE_MANIFEST" \
  --evidence-qc-filter "$MEDICAL_QC_FILTER" \
  --missing-evidence-policy "$MEDICAL_MISSING_POLICY" \
  --image-group-labels on \
  --prime off \
  --temperature 0 \
  --max-output-tokens 2048 \
  --judge-max-output-tokens 512 \
  --judge-structured-output auto \
  --closed-form-structured-output auto \
  --reasoning-effort off \
  --resume off \
  --output-root "eval_results/medical_modality/raw_runs/${MODEL_TAG}_both_tool_bbox_clean_brats2023_10pct_low_priority_v1"
```

### 7.8 P3：both crop

```bash
python eval_code/run_server_eval.py \
  --framework "$FRAMEWORK" \
  --model "$HF_MODEL_ID" \
  --served-model-name "$SERVED_MODEL_NAME" \
  --benchmark medical_modality \
  --dtype "$DTYPE" \
  --tensor-parallel-size auto \
  --gpu-placement-policy "$GPU_PLACEMENT_POLICY" \
  --shared-gpu-total-utilization "$SHARED_GPU_TOTAL_UTILIZATION" \
  --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
  --judge-server-model "$JUDGE_HF_MODEL_ID" \
  --judge-served-model-name "$JUDGE_SERVED_MODEL_NAME" \
  --judge-gpu-memory-utilization "$JUDGE_GPU_MEMORY_UTILIZATION" \
  --if-server-running fail \
  -- \
  --input-mode both \
  --backbone-model "$MODEL_ALIAS" \
  --visual-version "$MEDICAL_VISUAL_VERSION" \
  --dataset "$MEDICAL_DATASET" \
  --sample-fraction "$MEDICAL_SAMPLE_FRACTION" \
  --sample-strategy "$MEDICAL_SAMPLE_STRATEGY" \
  --question-types all \
  --tool-condition crop \
  --evidence-manifest "$MEDICAL_EVIDENCE_MANIFEST" \
  --evidence-qc-filter "$MEDICAL_QC_FILTER" \
  --missing-evidence-policy "$MEDICAL_MISSING_POLICY" \
  --image-group-labels on \
  --prime off \
  --temperature 0 \
  --max-output-tokens 2048 \
  --judge-max-output-tokens 512 \
  --judge-structured-output auto \
  --closed-form-structured-output auto \
  --reasoning-effort off \
  --resume off \
  --output-root "eval_results/medical_modality/raw_runs/${MODEL_TAG}_both_tool_crop_clean_brats2023_10pct_low_priority_v1"
```

## 8. Dry-Run 与检查命令

正式运行前建议先对每个模型做一个小 dry-run。示例：

```bash
python eval_code/medical_modality/eval_closed_vlm.py \
  --input-mode cf_only \
  --backbone-model "$MODEL_ALIAS" \
  --judge-model "$JUDGE_MODEL" \
  --visual-version "$MEDICAL_VISUAL_VERSION" \
  --dataset "$MEDICAL_DATASET" \
  --sample-fraction "$MEDICAL_SAMPLE_FRACTION" \
  --sample-strategy "$MEDICAL_SAMPLE_STRATEGY" \
  --question-types all \
  --tool-condition contour \
  --evidence-manifest "$MEDICAL_EVIDENCE_MANIFEST" \
  --evidence-qc-filter "$MEDICAL_QC_FILTER" \
  --missing-evidence-policy "$MEDICAL_MISSING_POLICY" \
  --image-group-labels on \
  --prime off \
  --max-samples 2 \
  --dry-run
```

已有结果汇总：

```bash
python eval_code/medical_modality/eval_closed_vlm.py \
  --report \
  --output-root eval_results/medical_modality/raw_runs
```

生成分析表：

```bash
python eval_code/medical_modality/analyze_results.py \
  --input-root eval_results/medical_modality/raw_runs \
  --output-root eval_results/medical_modality/tables/step5_medical_agentic_tools_v1
```

生成图表：

```bash
python eval_code/medical_modality/make_figures.py \
  --tables-root eval_results/medical_modality/tables/step5_medical_agentic_tools_v1 \
  --figures-root eval_results/medical_modality/figures/step5_medical_agentic_tools_v1
```

## 9. 报告建议

主文优先报告：

- 第一阶段 `cf_only raw clean brats2023 10%` baseline。
- 第五阶段 `cf_only contour clean brats2023 10%`。
- 第五阶段 `cf_only zoom_panel clean brats2023 10%`。

附录或 ablation 报告：

- `cf_only bbox`
- `cf_only crop`
- `both contour / zoom_panel`
- `both bbox / crop`

核心表格应包含：

- Accuracy
- Bias Rate
- Other Rate
- Delta Accuracy vs Raw
- Delta Bias vs Raw
- 按 `question_type` 拆分：`yes_no / multiple_choice / open`
- 按 `swap_direction` 拆分
- 按 `tool_condition` 拆分：`contour / zoom_panel / bbox / crop`
- 按 `model` 拆分
- Item difficulty 或 case-level review

如果某个 tool 降低 bias 但显著提高 `other`，不能简单解释为工具有效；应单独讨论它可能让模型更不确定，而不是真正理解 tumor-region modality consistency。

`crop` 的解释需要特别谨慎：它可能放大 tumor region，但也可能切掉 surrounding/background modality context。若 crop 表现差，不应直接说明 localization 无效，而可能说明该任务需要同时保留全局 sequence context。
