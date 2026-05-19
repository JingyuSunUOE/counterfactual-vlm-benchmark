# 第四阶段实验计划：Fashion / Industry Agentic Vision Tools

本文档是内部实验执行计划，覆盖 `fashion` 和 `industry` 两个 domain。它用于第四阶段 agentic vision tool 实验，不作为论文正文、数据集 README 或开源首页文件。

## 1. 实验目的

第四阶段的核心问题是：额外视觉证据是否能帮助 VLM 克服 fashion / industry 任务中的语言先验。

Fashion / Industry 的语言先验包括：

- 品牌 logo 的常见颜色、方向、布局、文字或图形结构。
- 日常/工业物体的常见颜色、形状、数量、文字或状态。
- 模型可能根据品牌或物体类别常识回答，而不是依据 CF 图像中的可见属性回答。

我们在第一阶段已经测试过 `raw` 条件，用于证明 benchmark 问题成立。因此本阶段不再重复 `raw` run。本阶段只测试已经生成并人工检查过的 SAM object-only 视觉证据。

主要指标仍为：

- `correct`：回答与可见目标属性一致。
- `biased`：回答符合常见品牌/物体先验，但忽略或误读反事实属性。
- `other`：回答模糊、不可判定、格式错误、无可见证据，或不属于 correct/biased。

## 2. 实验优先级

Fashion 和 industry 使用同一套实验矩阵，但结果必须按 `domain=fashion/industry` 分开报告。

### 2.1 主实验：优先执行

主实验只测试 `cf_only`，因为它最直接衡量 tool 是否能帮助模型理解反事实图像本身。

| Domain | Input Mode | Tool Condition | 优先级 | 目的 |
| --- | --- | --- | --- | --- |
| `fashion` | `cf_only` | `crop` | P0 | 放大主体或目标区域，减少背景干扰 |
| `fashion` | `cf_only` | `zoom_panel` | P0 | 同时保留全图和局部放大 |
| `industry` | `cf_only` | `crop` | P0 | 放大主体或目标区域，减少背景干扰 |
| `industry` | `cf_only` | `zoom_panel` | P0 | 同时保留全图和局部放大 |

### 2.2 低优先级：资源允许后执行

| Domain | Input Mode | Tool Condition | 优先级 | 说明 |
| --- | --- | --- | --- | --- |
| `fashion` | `cf_only` | `bbox` | P1 | localization control，信息增量弱于 crop/zoom_panel |
| `industry` | `cf_only` | `bbox` | P1 | localization control，信息增量弱于 crop/zoom_panel |
| `fashion` | `both` | `crop` | P2 | 测试局部放大是否帮助比较 original/CF |
| `fashion` | `both` | `zoom_panel` | P2 | 测试全图 + 局部视图是否帮助比较差异 |
| `industry` | `both` | `crop` | P2 | 测试局部放大是否帮助比较 original/CF |
| `industry` | `both` | `zoom_panel` | P2 | 测试全图 + 局部视图是否帮助比较差异 |
| `fashion` | `both` | `bbox` | P3 | 最低优先级，只作为定位框 ablation |
| `industry` | `both` | `bbox` | P3 | 最低优先级，只作为定位框 ablation |

本阶段不测试 `orig_only`，也不重复 `raw`。第一阶段 raw 结果作为本阶段的 baseline；第四阶段只运行非 raw tool conditions。

## 3. 数据与 Evidence 设置

Fashion / Industry tool evidence 使用已经生成并人工 QC 过的 SAM object-only evidence manifests：

```bash
FASHION_EVIDENCE_MANIFEST=eval_results/fashion_industry/metadata/fashion_sam3_object_evidence_manifest_pad005_max3_merge5_fallback.json
INDUSTRY_EVIDENCE_MANIFEST=eval_results/fashion_industry/metadata/industry_sam3_object_evidence_manifest_pad005_max3_merge5_fallback.json
FI_QC_FILTER=pass_review
FI_MISSING_POLICY=skip
```

说明：

- `pass_review` 表示使用人工检查后可接受的 `pass` 和 `review` 样本。
- `skip` 表示缺失 tool evidence 的 pair 不发 API，不写 response record，不污染 denominator。
- 本阶段不新跑 raw；正式分析时使用第一阶段 raw baseline，并在报告中注明如果 tool run 因 evidence skip 导致 denominator 不一致，应使用 analysis 中的 matched-subset delta 或单独说明。
- Fashion 当前 active set 约为 `76 originals / 456 CF`，industry 当前 active set 约为 `77 originals / 462 CF`；后续如果 metadata 再变动，应以 active metadata 和 evidence manifest 为准。

## 4. 通用评测设置

统一设置：

```text
phase = cf
question_types = all
prime = off
temperature = 0
max_output_tokens = 2048
judge_max_output_tokens = 512
judge_structured_output = auto
closed_form_structured_output = auto
reasoning_effort = off
evidence_qc_filter = pass_review
missing_evidence_policy = skip
```

`prime off` 不表示没有语言先验；fashion / industry 的问题 prompt 本身已经包含 category-specific prior statement。

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

开源模型默认使用 `eval_code/run_server_eval.py` 自动启动 OpenAI-compatible backbone server 和 text-only judge server。不要在 `--` 后手动传 `--server-url`、`--server-model-id`、`--judge-provider`、`--judge-model`、`--judge` 或 `--judge-server-url`。

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

### 6.1 P0：fashion cf_only crop

```bash
python eval_code/fashion_industry/eval_pipeline.py \
  --phase cf \
  --domain fashion \
  --input-mode cf_only \
  --model "$MODEL_ALIAS" \
  --question-types all \
  --tool-condition crop \
  --evidence-manifest "$FASHION_EVIDENCE_MANIFEST" \
  --evidence-qc-filter "$FI_QC_FILTER" \
  --missing-evidence-policy "$FI_MISSING_POLICY" \
  --image-group-labels on \
  --prime off \
  --judge "$JUDGE_MODEL" \
  --provider-cache auto \
  --cache-key-mode image_pair \
  --judge-structured-output auto \
  --closed-form-structured-output auto \
  --reasoning-effort off \
  --max-output-tokens 2048 \
  --judge-max-output-tokens 512 \
  --resume off \
  --output-root "eval_results/fashion_industry/raw_runs/${MODEL_TAG}_fashion_cf_only_tool_crop_sam3_object_v1"
```

### 6.2 P0：fashion cf_only zoom_panel

```bash
python eval_code/fashion_industry/eval_pipeline.py \
  --phase cf \
  --domain fashion \
  --input-mode cf_only \
  --model "$MODEL_ALIAS" \
  --question-types all \
  --tool-condition zoom_panel \
  --evidence-manifest "$FASHION_EVIDENCE_MANIFEST" \
  --evidence-qc-filter "$FI_QC_FILTER" \
  --missing-evidence-policy "$FI_MISSING_POLICY" \
  --image-group-labels on \
  --prime off \
  --judge "$JUDGE_MODEL" \
  --provider-cache auto \
  --cache-key-mode image_pair \
  --judge-structured-output auto \
  --closed-form-structured-output auto \
  --reasoning-effort off \
  --max-output-tokens 2048 \
  --judge-max-output-tokens 512 \
  --resume off \
  --output-root "eval_results/fashion_industry/raw_runs/${MODEL_TAG}_fashion_cf_only_tool_zoom_panel_sam3_object_v1"
```

### 6.3 P0：industry cf_only crop

```bash
python eval_code/fashion_industry/eval_pipeline.py \
  --phase cf \
  --domain industry \
  --input-mode cf_only \
  --model "$MODEL_ALIAS" \
  --question-types all \
  --tool-condition crop \
  --evidence-manifest "$INDUSTRY_EVIDENCE_MANIFEST" \
  --evidence-qc-filter "$FI_QC_FILTER" \
  --missing-evidence-policy "$FI_MISSING_POLICY" \
  --image-group-labels on \
  --prime off \
  --judge "$JUDGE_MODEL" \
  --provider-cache auto \
  --cache-key-mode image_pair \
  --judge-structured-output auto \
  --closed-form-structured-output auto \
  --reasoning-effort off \
  --max-output-tokens 2048 \
  --judge-max-output-tokens 512 \
  --resume off \
  --output-root "eval_results/fashion_industry/raw_runs/${MODEL_TAG}_industry_cf_only_tool_crop_sam3_object_v1"
```

### 6.4 P0：industry cf_only zoom_panel

```bash
python eval_code/fashion_industry/eval_pipeline.py \
  --phase cf \
  --domain industry \
  --input-mode cf_only \
  --model "$MODEL_ALIAS" \
  --question-types all \
  --tool-condition zoom_panel \
  --evidence-manifest "$INDUSTRY_EVIDENCE_MANIFEST" \
  --evidence-qc-filter "$FI_QC_FILTER" \
  --missing-evidence-policy "$FI_MISSING_POLICY" \
  --image-group-labels on \
  --prime off \
  --judge "$JUDGE_MODEL" \
  --provider-cache auto \
  --cache-key-mode image_pair \
  --judge-structured-output auto \
  --closed-form-structured-output auto \
  --reasoning-effort off \
  --max-output-tokens 2048 \
  --judge-max-output-tokens 512 \
  --resume off \
  --output-root "eval_results/fashion_industry/raw_runs/${MODEL_TAG}_industry_cf_only_tool_zoom_panel_sam3_object_v1"
```

### 6.5 P1：fashion cf_only bbox

```bash
python eval_code/fashion_industry/eval_pipeline.py \
  --phase cf \
  --domain fashion \
  --input-mode cf_only \
  --model "$MODEL_ALIAS" \
  --question-types all \
  --tool-condition bbox \
  --evidence-manifest "$FASHION_EVIDENCE_MANIFEST" \
  --evidence-qc-filter "$FI_QC_FILTER" \
  --missing-evidence-policy "$FI_MISSING_POLICY" \
  --image-group-labels on \
  --prime off \
  --judge "$JUDGE_MODEL" \
  --provider-cache auto \
  --cache-key-mode image_pair \
  --judge-structured-output auto \
  --closed-form-structured-output auto \
  --reasoning-effort off \
  --max-output-tokens 2048 \
  --judge-max-output-tokens 512 \
  --resume off \
  --output-root "eval_results/fashion_industry/raw_runs/${MODEL_TAG}_fashion_cf_only_tool_bbox_sam3_object_low_priority_v1"
```

### 6.6 P1：industry cf_only bbox

```bash
python eval_code/fashion_industry/eval_pipeline.py \
  --phase cf \
  --domain industry \
  --input-mode cf_only \
  --model "$MODEL_ALIAS" \
  --question-types all \
  --tool-condition bbox \
  --evidence-manifest "$INDUSTRY_EVIDENCE_MANIFEST" \
  --evidence-qc-filter "$FI_QC_FILTER" \
  --missing-evidence-policy "$FI_MISSING_POLICY" \
  --image-group-labels on \
  --prime off \
  --judge "$JUDGE_MODEL" \
  --provider-cache auto \
  --cache-key-mode image_pair \
  --judge-structured-output auto \
  --closed-form-structured-output auto \
  --reasoning-effort off \
  --max-output-tokens 2048 \
  --judge-max-output-tokens 512 \
  --resume off \
  --output-root "eval_results/fashion_industry/raw_runs/${MODEL_TAG}_industry_cf_only_tool_bbox_sam3_object_low_priority_v1"
```

### 6.7 P2：fashion both crop

```bash
python eval_code/fashion_industry/eval_pipeline.py \
  --phase cf \
  --domain fashion \
  --input-mode both \
  --model "$MODEL_ALIAS" \
  --question-types all \
  --tool-condition crop \
  --evidence-manifest "$FASHION_EVIDENCE_MANIFEST" \
  --evidence-qc-filter "$FI_QC_FILTER" \
  --missing-evidence-policy "$FI_MISSING_POLICY" \
  --image-group-labels on \
  --prime off \
  --judge "$JUDGE_MODEL" \
  --provider-cache auto \
  --cache-key-mode image_pair \
  --judge-structured-output auto \
  --closed-form-structured-output auto \
  --reasoning-effort off \
  --max-output-tokens 2048 \
  --judge-max-output-tokens 512 \
  --resume off \
  --output-root "eval_results/fashion_industry/raw_runs/${MODEL_TAG}_fashion_both_tool_crop_sam3_object_low_priority_v1"
```

### 6.8 P2：fashion both zoom_panel

```bash
python eval_code/fashion_industry/eval_pipeline.py \
  --phase cf \
  --domain fashion \
  --input-mode both \
  --model "$MODEL_ALIAS" \
  --question-types all \
  --tool-condition zoom_panel \
  --evidence-manifest "$FASHION_EVIDENCE_MANIFEST" \
  --evidence-qc-filter "$FI_QC_FILTER" \
  --missing-evidence-policy "$FI_MISSING_POLICY" \
  --image-group-labels on \
  --prime off \
  --judge "$JUDGE_MODEL" \
  --provider-cache auto \
  --cache-key-mode image_pair \
  --judge-structured-output auto \
  --closed-form-structured-output auto \
  --reasoning-effort off \
  --max-output-tokens 2048 \
  --judge-max-output-tokens 512 \
  --resume off \
  --output-root "eval_results/fashion_industry/raw_runs/${MODEL_TAG}_fashion_both_tool_zoom_panel_sam3_object_low_priority_v1"
```

### 6.9 P2：industry both crop

```bash
python eval_code/fashion_industry/eval_pipeline.py \
  --phase cf \
  --domain industry \
  --input-mode both \
  --model "$MODEL_ALIAS" \
  --question-types all \
  --tool-condition crop \
  --evidence-manifest "$INDUSTRY_EVIDENCE_MANIFEST" \
  --evidence-qc-filter "$FI_QC_FILTER" \
  --missing-evidence-policy "$FI_MISSING_POLICY" \
  --image-group-labels on \
  --prime off \
  --judge "$JUDGE_MODEL" \
  --provider-cache auto \
  --cache-key-mode image_pair \
  --judge-structured-output auto \
  --closed-form-structured-output auto \
  --reasoning-effort off \
  --max-output-tokens 2048 \
  --judge-max-output-tokens 512 \
  --resume off \
  --output-root "eval_results/fashion_industry/raw_runs/${MODEL_TAG}_industry_both_tool_crop_sam3_object_low_priority_v1"
```

### 6.10 P2：industry both zoom_panel

```bash
python eval_code/fashion_industry/eval_pipeline.py \
  --phase cf \
  --domain industry \
  --input-mode both \
  --model "$MODEL_ALIAS" \
  --question-types all \
  --tool-condition zoom_panel \
  --evidence-manifest "$INDUSTRY_EVIDENCE_MANIFEST" \
  --evidence-qc-filter "$FI_QC_FILTER" \
  --missing-evidence-policy "$FI_MISSING_POLICY" \
  --image-group-labels on \
  --prime off \
  --judge "$JUDGE_MODEL" \
  --provider-cache auto \
  --cache-key-mode image_pair \
  --judge-structured-output auto \
  --closed-form-structured-output auto \
  --reasoning-effort off \
  --max-output-tokens 2048 \
  --judge-max-output-tokens 512 \
  --resume off \
  --output-root "eval_results/fashion_industry/raw_runs/${MODEL_TAG}_industry_both_tool_zoom_panel_sam3_object_low_priority_v1"
```

### 6.11 P3：fashion both bbox

```bash
python eval_code/fashion_industry/eval_pipeline.py \
  --phase cf \
  --domain fashion \
  --input-mode both \
  --model "$MODEL_ALIAS" \
  --question-types all \
  --tool-condition bbox \
  --evidence-manifest "$FASHION_EVIDENCE_MANIFEST" \
  --evidence-qc-filter "$FI_QC_FILTER" \
  --missing-evidence-policy "$FI_MISSING_POLICY" \
  --image-group-labels on \
  --prime off \
  --judge "$JUDGE_MODEL" \
  --provider-cache auto \
  --cache-key-mode image_pair \
  --judge-structured-output auto \
  --closed-form-structured-output auto \
  --reasoning-effort off \
  --max-output-tokens 2048 \
  --judge-max-output-tokens 512 \
  --resume off \
  --output-root "eval_results/fashion_industry/raw_runs/${MODEL_TAG}_fashion_both_tool_bbox_sam3_object_low_priority_v1"
```

### 6.12 P3：industry both bbox

```bash
python eval_code/fashion_industry/eval_pipeline.py \
  --phase cf \
  --domain industry \
  --input-mode both \
  --model "$MODEL_ALIAS" \
  --question-types all \
  --tool-condition bbox \
  --evidence-manifest "$INDUSTRY_EVIDENCE_MANIFEST" \
  --evidence-qc-filter "$FI_QC_FILTER" \
  --missing-evidence-policy "$FI_MISSING_POLICY" \
  --image-group-labels on \
  --prime off \
  --judge "$JUDGE_MODEL" \
  --provider-cache auto \
  --cache-key-mode image_pair \
  --judge-structured-output auto \
  --closed-form-structured-output auto \
  --reasoning-effort off \
  --max-output-tokens 2048 \
  --judge-max-output-tokens 512 \
  --resume off \
  --output-root "eval_results/fashion_industry/raw_runs/${MODEL_TAG}_industry_both_tool_bbox_sam3_object_low_priority_v1"
```

## 7. 开源模型命令

以下命令每个都需要分别对所有开源模型变量块执行。wrapper 会自动启动 backbone server 和 judge server，并自动规划 GPU placement。

### 7.1 P0：fashion cf_only crop

```bash
python eval_code/run_server_eval.py \
  --framework "$FRAMEWORK" \
  --model "$HF_MODEL_ID" \
  --served-model-name "$SERVED_MODEL_NAME" \
  --benchmark fashion_industry \
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
  --phase cf \
  --domain fashion \
  --input-mode cf_only \
  --model "$MODEL_ALIAS" \
  --question-types all \
  --tool-condition crop \
  --evidence-manifest "$FASHION_EVIDENCE_MANIFEST" \
  --evidence-qc-filter "$FI_QC_FILTER" \
  --missing-evidence-policy "$FI_MISSING_POLICY" \
  --image-group-labels on \
  --prime off \
  --judge-structured-output auto \
  --closed-form-structured-output auto \
  --reasoning-effort off \
  --max-output-tokens 2048 \
  --judge-max-output-tokens 512 \
  --resume off \
  --output-root "eval_results/fashion_industry/raw_runs/${MODEL_TAG}_fashion_cf_only_tool_crop_sam3_object_v1"
```

### 7.2 P0：fashion cf_only zoom_panel

```bash
python eval_code/run_server_eval.py \
  --framework "$FRAMEWORK" \
  --model "$HF_MODEL_ID" \
  --served-model-name "$SERVED_MODEL_NAME" \
  --benchmark fashion_industry \
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
  --phase cf \
  --domain fashion \
  --input-mode cf_only \
  --model "$MODEL_ALIAS" \
  --question-types all \
  --tool-condition zoom_panel \
  --evidence-manifest "$FASHION_EVIDENCE_MANIFEST" \
  --evidence-qc-filter "$FI_QC_FILTER" \
  --missing-evidence-policy "$FI_MISSING_POLICY" \
  --image-group-labels on \
  --prime off \
  --judge-structured-output auto \
  --closed-form-structured-output auto \
  --reasoning-effort off \
  --max-output-tokens 2048 \
  --judge-max-output-tokens 512 \
  --resume off \
  --output-root "eval_results/fashion_industry/raw_runs/${MODEL_TAG}_fashion_cf_only_tool_zoom_panel_sam3_object_v1"
```

### 7.3 P0：industry cf_only crop

```bash
python eval_code/run_server_eval.py \
  --framework "$FRAMEWORK" \
  --model "$HF_MODEL_ID" \
  --served-model-name "$SERVED_MODEL_NAME" \
  --benchmark fashion_industry \
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
  --phase cf \
  --domain industry \
  --input-mode cf_only \
  --model "$MODEL_ALIAS" \
  --question-types all \
  --tool-condition crop \
  --evidence-manifest "$INDUSTRY_EVIDENCE_MANIFEST" \
  --evidence-qc-filter "$FI_QC_FILTER" \
  --missing-evidence-policy "$FI_MISSING_POLICY" \
  --image-group-labels on \
  --prime off \
  --judge-structured-output auto \
  --closed-form-structured-output auto \
  --reasoning-effort off \
  --max-output-tokens 2048 \
  --judge-max-output-tokens 512 \
  --resume off \
  --output-root "eval_results/fashion_industry/raw_runs/${MODEL_TAG}_industry_cf_only_tool_crop_sam3_object_v1"
```

### 7.4 P0：industry cf_only zoom_panel

```bash
python eval_code/run_server_eval.py \
  --framework "$FRAMEWORK" \
  --model "$HF_MODEL_ID" \
  --served-model-name "$SERVED_MODEL_NAME" \
  --benchmark fashion_industry \
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
  --phase cf \
  --domain industry \
  --input-mode cf_only \
  --model "$MODEL_ALIAS" \
  --question-types all \
  --tool-condition zoom_panel \
  --evidence-manifest "$INDUSTRY_EVIDENCE_MANIFEST" \
  --evidence-qc-filter "$FI_QC_FILTER" \
  --missing-evidence-policy "$FI_MISSING_POLICY" \
  --image-group-labels on \
  --prime off \
  --judge-structured-output auto \
  --closed-form-structured-output auto \
  --reasoning-effort off \
  --max-output-tokens 2048 \
  --judge-max-output-tokens 512 \
  --resume off \
  --output-root "eval_results/fashion_industry/raw_runs/${MODEL_TAG}_industry_cf_only_tool_zoom_panel_sam3_object_v1"
```

### 7.5 P1：fashion cf_only bbox

```bash
python eval_code/run_server_eval.py \
  --framework "$FRAMEWORK" \
  --model "$HF_MODEL_ID" \
  --served-model-name "$SERVED_MODEL_NAME" \
  --benchmark fashion_industry \
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
  --phase cf \
  --domain fashion \
  --input-mode cf_only \
  --model "$MODEL_ALIAS" \
  --question-types all \
  --tool-condition bbox \
  --evidence-manifest "$FASHION_EVIDENCE_MANIFEST" \
  --evidence-qc-filter "$FI_QC_FILTER" \
  --missing-evidence-policy "$FI_MISSING_POLICY" \
  --image-group-labels on \
  --prime off \
  --judge-structured-output auto \
  --closed-form-structured-output auto \
  --reasoning-effort off \
  --max-output-tokens 2048 \
  --judge-max-output-tokens 512 \
  --resume off \
  --output-root "eval_results/fashion_industry/raw_runs/${MODEL_TAG}_fashion_cf_only_tool_bbox_sam3_object_low_priority_v1"
```

### 7.6 P1：industry cf_only bbox

```bash
python eval_code/run_server_eval.py \
  --framework "$FRAMEWORK" \
  --model "$HF_MODEL_ID" \
  --served-model-name "$SERVED_MODEL_NAME" \
  --benchmark fashion_industry \
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
  --phase cf \
  --domain industry \
  --input-mode cf_only \
  --model "$MODEL_ALIAS" \
  --question-types all \
  --tool-condition bbox \
  --evidence-manifest "$INDUSTRY_EVIDENCE_MANIFEST" \
  --evidence-qc-filter "$FI_QC_FILTER" \
  --missing-evidence-policy "$FI_MISSING_POLICY" \
  --image-group-labels on \
  --prime off \
  --judge-structured-output auto \
  --closed-form-structured-output auto \
  --reasoning-effort off \
  --max-output-tokens 2048 \
  --judge-max-output-tokens 512 \
  --resume off \
  --output-root "eval_results/fashion_industry/raw_runs/${MODEL_TAG}_industry_cf_only_tool_bbox_sam3_object_low_priority_v1"
```

### 7.7 P2：fashion both crop

```bash
python eval_code/run_server_eval.py \
  --framework "$FRAMEWORK" \
  --model "$HF_MODEL_ID" \
  --served-model-name "$SERVED_MODEL_NAME" \
  --benchmark fashion_industry \
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
  --phase cf \
  --domain fashion \
  --input-mode both \
  --model "$MODEL_ALIAS" \
  --question-types all \
  --tool-condition crop \
  --evidence-manifest "$FASHION_EVIDENCE_MANIFEST" \
  --evidence-qc-filter "$FI_QC_FILTER" \
  --missing-evidence-policy "$FI_MISSING_POLICY" \
  --image-group-labels on \
  --prime off \
  --judge-structured-output auto \
  --closed-form-structured-output auto \
  --reasoning-effort off \
  --max-output-tokens 2048 \
  --judge-max-output-tokens 512 \
  --resume off \
  --output-root "eval_results/fashion_industry/raw_runs/${MODEL_TAG}_fashion_both_tool_crop_sam3_object_low_priority_v1"
```

### 7.8 P2：fashion both zoom_panel

```bash
python eval_code/run_server_eval.py \
  --framework "$FRAMEWORK" \
  --model "$HF_MODEL_ID" \
  --served-model-name "$SERVED_MODEL_NAME" \
  --benchmark fashion_industry \
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
  --phase cf \
  --domain fashion \
  --input-mode both \
  --model "$MODEL_ALIAS" \
  --question-types all \
  --tool-condition zoom_panel \
  --evidence-manifest "$FASHION_EVIDENCE_MANIFEST" \
  --evidence-qc-filter "$FI_QC_FILTER" \
  --missing-evidence-policy "$FI_MISSING_POLICY" \
  --image-group-labels on \
  --prime off \
  --judge-structured-output auto \
  --closed-form-structured-output auto \
  --reasoning-effort off \
  --max-output-tokens 2048 \
  --judge-max-output-tokens 512 \
  --resume off \
  --output-root "eval_results/fashion_industry/raw_runs/${MODEL_TAG}_fashion_both_tool_zoom_panel_sam3_object_low_priority_v1"
```

### 7.9 P2：industry both crop

```bash
python eval_code/run_server_eval.py \
  --framework "$FRAMEWORK" \
  --model "$HF_MODEL_ID" \
  --served-model-name "$SERVED_MODEL_NAME" \
  --benchmark fashion_industry \
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
  --phase cf \
  --domain industry \
  --input-mode both \
  --model "$MODEL_ALIAS" \
  --question-types all \
  --tool-condition crop \
  --evidence-manifest "$INDUSTRY_EVIDENCE_MANIFEST" \
  --evidence-qc-filter "$FI_QC_FILTER" \
  --missing-evidence-policy "$FI_MISSING_POLICY" \
  --image-group-labels on \
  --prime off \
  --judge-structured-output auto \
  --closed-form-structured-output auto \
  --reasoning-effort off \
  --max-output-tokens 2048 \
  --judge-max-output-tokens 512 \
  --resume off \
  --output-root "eval_results/fashion_industry/raw_runs/${MODEL_TAG}_industry_both_tool_crop_sam3_object_low_priority_v1"
```

### 7.10 P2：industry both zoom_panel

```bash
python eval_code/run_server_eval.py \
  --framework "$FRAMEWORK" \
  --model "$HF_MODEL_ID" \
  --served-model-name "$SERVED_MODEL_NAME" \
  --benchmark fashion_industry \
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
  --phase cf \
  --domain industry \
  --input-mode both \
  --model "$MODEL_ALIAS" \
  --question-types all \
  --tool-condition zoom_panel \
  --evidence-manifest "$INDUSTRY_EVIDENCE_MANIFEST" \
  --evidence-qc-filter "$FI_QC_FILTER" \
  --missing-evidence-policy "$FI_MISSING_POLICY" \
  --image-group-labels on \
  --prime off \
  --judge-structured-output auto \
  --closed-form-structured-output auto \
  --reasoning-effort off \
  --max-output-tokens 2048 \
  --judge-max-output-tokens 512 \
  --resume off \
  --output-root "eval_results/fashion_industry/raw_runs/${MODEL_TAG}_industry_both_tool_zoom_panel_sam3_object_low_priority_v1"
```

### 7.11 P3：fashion both bbox

```bash
python eval_code/run_server_eval.py \
  --framework "$FRAMEWORK" \
  --model "$HF_MODEL_ID" \
  --served-model-name "$SERVED_MODEL_NAME" \
  --benchmark fashion_industry \
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
  --phase cf \
  --domain fashion \
  --input-mode both \
  --model "$MODEL_ALIAS" \
  --question-types all \
  --tool-condition bbox \
  --evidence-manifest "$FASHION_EVIDENCE_MANIFEST" \
  --evidence-qc-filter "$FI_QC_FILTER" \
  --missing-evidence-policy "$FI_MISSING_POLICY" \
  --image-group-labels on \
  --prime off \
  --judge-structured-output auto \
  --closed-form-structured-output auto \
  --reasoning-effort off \
  --max-output-tokens 2048 \
  --judge-max-output-tokens 512 \
  --resume off \
  --output-root "eval_results/fashion_industry/raw_runs/${MODEL_TAG}_fashion_both_tool_bbox_sam3_object_low_priority_v1"
```

### 7.12 P3：industry both bbox

```bash
python eval_code/run_server_eval.py \
  --framework "$FRAMEWORK" \
  --model "$HF_MODEL_ID" \
  --served-model-name "$SERVED_MODEL_NAME" \
  --benchmark fashion_industry \
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
  --phase cf \
  --domain industry \
  --input-mode both \
  --model "$MODEL_ALIAS" \
  --question-types all \
  --tool-condition bbox \
  --evidence-manifest "$INDUSTRY_EVIDENCE_MANIFEST" \
  --evidence-qc-filter "$FI_QC_FILTER" \
  --missing-evidence-policy "$FI_MISSING_POLICY" \
  --image-group-labels on \
  --prime off \
  --judge-structured-output auto \
  --closed-form-structured-output auto \
  --reasoning-effort off \
  --max-output-tokens 2048 \
  --judge-max-output-tokens 512 \
  --resume off \
  --output-root "eval_results/fashion_industry/raw_runs/${MODEL_TAG}_industry_both_tool_bbox_sam3_object_low_priority_v1"
```

## 8. Dry-Run 与检查命令

正式运行前建议先对每个模型做一个小 dry-run。示例：

```bash
python eval_code/fashion_industry/eval_pipeline.py \
  --phase cf \
  --domain fashion \
  --input-mode cf_only \
  --model "$MODEL_ALIAS" \
  --question-types all \
  --tool-condition crop \
  --evidence-manifest "$FASHION_EVIDENCE_MANIFEST" \
  --evidence-qc-filter "$FI_QC_FILTER" \
  --missing-evidence-policy "$FI_MISSING_POLICY" \
  --image-group-labels on \
  --prime off \
  --judge "$JUDGE_MODEL" \
  --limit 8 \
  --dry-run
```

已有结果汇总：

```bash
python eval_code/fashion_industry/eval_pipeline.py \
  --report
```

## 9. 报告建议

主文优先报告：

- 第一阶段 `fashion cf_only raw` baseline。
- 第四阶段 `fashion cf_only crop`。
- 第四阶段 `fashion cf_only zoom_panel`。
- 第一阶段 `industry cf_only raw` baseline。
- 第四阶段 `industry cf_only crop`。
- 第四阶段 `industry cf_only zoom_panel`。

附录或 ablation 报告：

- `fashion cf_only bbox`
- `industry cf_only bbox`
- `fashion both crop / zoom_panel / bbox`
- `industry both crop / zoom_panel / bbox`

核心表格应包含：

- Accuracy
- Bias Rate
- Other Rate
- Delta Accuracy vs Raw
- Delta Bias vs Raw
- 按 `domain` 拆分：`fashion / industry`
- 按 `question_type` 拆分：`Q1 / Q2 / Q3 / Q_MC`
- 按 `source_variant` 拆分：`real / ai`
- 按 `key` 或 `subcategory` 拆分：品牌或物体类别
- 如 metadata 支持，按 `mod_type` 拆分：color / text / count / direction / layout / shape 等

如果某个 tool 降低 bias 但显著提高 `other`，不能简单解释为工具有效；应单独讨论它可能让模型更不确定，而不是真正理解目标属性。
