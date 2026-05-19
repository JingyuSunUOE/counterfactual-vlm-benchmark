# 第二阶段实验计划：Counting Agentic Vision Tools

本文档是内部实验执行计划，只覆盖 `counting` 这一类 benchmark。它用于第二阶段 agentic vision tool 实验，不作为论文正文、数据集 README 或开源首页文件。

## 1. 实验目的

第二阶段的核心问题是：额外视觉证据是否能帮助 VLM 克服 counting 任务中的语言先验。

Counting 的语言先验包括：

- 鸟通常有一对翅膀。
- Diptera 通常有 2 个翅膀，其它当前 insect orders 通常有 4 个翅膀。
- 人手、黑猩猩手、熊猫爪通常有 5 个手指或趾。
- 鸡脚、啄木鸟脚通常有 4 个脚趾。

我们不再重复第一阶段的完整 `orig_only / cf_only / both` raw validity 矩阵。第一阶段已经用于证明原图本身不是主要难点、CF 图会诱导模型产生 bias。第二阶段只关注：当模型看到额外视觉视图时，是否更少回答 normal expected count，是否更能依据可见数量线索作答。

本阶段明确不重复 `raw` run，也不测试 `orig_only`。第一阶段 raw 结果作为本阶段的 baseline；第二阶段只运行非 raw tool conditions。

主要指标仍为：

- `correct`：回答与可见数量事实一致。
- `biased`：回答符合正常数量先验，但忽略或误读反事实数量。
- `other`：回答模糊、不可判定、格式错误、无可见数量证据，或不属于 correct/biased。

## 2. 实验优先级

### 2.1 主实验：优先执行

主实验只测试 `cf_only`，因为它最直接衡量 tool 是否能帮助模型理解反事实图像本身。

主实验条件：

| Input Mode | Tool Condition | 优先级 | 目的 |
| --- | --- | --- | --- |
| `cf_only` | `zoom_panel` | P0 | 同时保留全图和局部放大，最接近 agentic visual inspection |
| `cf_only` | `outline` | P0 | 强调对象外轮廓，可能帮助判断翅膀、手指、脚趾边缘结构 |
| `cf_only` | `crop` | P0 | 放大主体，减少背景干扰 |

### 2.2 低优先级：资源允许后执行

| Input Mode | Tool Condition | 优先级 | 说明 |
| --- | --- | --- | --- |
| `cf_only` | `bbox` | P1 | localization control，信息增量较弱 |
| `both` | `zoom_panel` | P1 | 测试工具视图是否帮助比较 original/CF |
| `both` | `outline` | P1 | 测试轮廓证据是否帮助比较数量差异 |
| `both` | `crop` | P1 | 测试局部放大是否帮助比较数量差异 |
| `both` | `bbox` | P2 | 最低优先级，只在需要完整 ablation 时执行 |

本阶段不测试 `orig_only`，也不重复 `raw`。原因是第一阶段已经证明 original sanity baseline 和 raw 语言先验效应；第二阶段核心问题是 CF 图和语言先验冲突时，非 raw 视觉工具是否能降低 bias。

## 3. 数据与 Evidence 设置

Counting tool evidence 使用已经生成并人工 QC 过的 SAM object-only evidence manifest：

```bash
COUNTING_EVIDENCE_MANIFEST=eval_results/counting/metadata/counting_sam3_object_evidence_manifest_pad005_max3.json
COUNTING_QC_FILTER=pass_review
COUNTING_MISSING_POLICY=skip
```

说明：

- `pass_review` 表示使用人工检查后可接受的 `pass` 和 `review` 样本。
- `skip` 表示缺失 tool evidence 的 pair 不发 API，不写 response record，不污染 denominator。
- 本阶段不新跑 raw；正式分析时使用第一阶段 raw baseline，并在报告中注明如果 tool run 因 evidence skip 导致 denominator 不一致，应使用 analysis 中的 matched-subset delta 或单独说明。
- 正式分析 raw-vs-tool delta 时，只比较相同 `input_mode / model / question_type / subset / source_variant` 且可匹配样本集合一致的结果。

## 4. 通用评测设置

统一设置：

```text
question_types = all
subset = all
source_variant = both
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

`prime off` 不表示没有语言先验；counting 的问题 prompt 本身已经包含正常数量先验。

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

### 6.1 P0：cf_only zoom_panel

```bash
python eval_code/counting/eval_closed_vlm.py \
  --input-mode cf_only \
  --backbone-model "$MODEL_ALIAS" \
  --judge-model "$JUDGE_MODEL" \
  --question-types all \
  --subset all \
  --source-variant both \
  --tool-condition zoom_panel \
  --evidence-manifest "$COUNTING_EVIDENCE_MANIFEST" \
  --evidence-qc-filter "$COUNTING_QC_FILTER" \
  --missing-evidence-policy "$COUNTING_MISSING_POLICY" \
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
  --output-root "eval_results/counting/raw_runs/${MODEL_TAG}_cf_only_tool_zoom_panel_pad005_max3_v1"
```

### 6.2 P0：cf_only outline

```bash
python eval_code/counting/eval_closed_vlm.py \
  --input-mode cf_only \
  --backbone-model "$MODEL_ALIAS" \
  --judge-model "$JUDGE_MODEL" \
  --question-types all \
  --subset all \
  --source-variant both \
  --tool-condition outline \
  --evidence-manifest "$COUNTING_EVIDENCE_MANIFEST" \
  --evidence-qc-filter "$COUNTING_QC_FILTER" \
  --missing-evidence-policy "$COUNTING_MISSING_POLICY" \
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
  --output-root "eval_results/counting/raw_runs/${MODEL_TAG}_cf_only_tool_outline_pad005_max3_v1"
```

### 6.3 P0：cf_only crop

```bash
python eval_code/counting/eval_closed_vlm.py \
  --input-mode cf_only \
  --backbone-model "$MODEL_ALIAS" \
  --judge-model "$JUDGE_MODEL" \
  --question-types all \
  --subset all \
  --source-variant both \
  --tool-condition crop \
  --evidence-manifest "$COUNTING_EVIDENCE_MANIFEST" \
  --evidence-qc-filter "$COUNTING_QC_FILTER" \
  --missing-evidence-policy "$COUNTING_MISSING_POLICY" \
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
  --output-root "eval_results/counting/raw_runs/${MODEL_TAG}_cf_only_tool_crop_pad005_max3_v1"
```

### 6.4 P1：cf_only bbox

```bash
python eval_code/counting/eval_closed_vlm.py \
  --input-mode cf_only \
  --backbone-model "$MODEL_ALIAS" \
  --judge-model "$JUDGE_MODEL" \
  --question-types all \
  --subset all \
  --source-variant both \
  --tool-condition bbox \
  --evidence-manifest "$COUNTING_EVIDENCE_MANIFEST" \
  --evidence-qc-filter "$COUNTING_QC_FILTER" \
  --missing-evidence-policy "$COUNTING_MISSING_POLICY" \
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
  --output-root "eval_results/counting/raw_runs/${MODEL_TAG}_cf_only_tool_bbox_pad005_max3_low_priority_v1"
```

### 6.5 P1：both zoom_panel

```bash
python eval_code/counting/eval_closed_vlm.py \
  --input-mode both \
  --backbone-model "$MODEL_ALIAS" \
  --judge-model "$JUDGE_MODEL" \
  --question-types all \
  --subset all \
  --source-variant both \
  --tool-condition zoom_panel \
  --evidence-manifest "$COUNTING_EVIDENCE_MANIFEST" \
  --evidence-qc-filter "$COUNTING_QC_FILTER" \
  --missing-evidence-policy "$COUNTING_MISSING_POLICY" \
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
  --output-root "eval_results/counting/raw_runs/${MODEL_TAG}_both_tool_zoom_panel_pad005_max3_low_priority_v1"
```

### 6.6 P1：both outline

```bash
python eval_code/counting/eval_closed_vlm.py \
  --input-mode both \
  --backbone-model "$MODEL_ALIAS" \
  --judge-model "$JUDGE_MODEL" \
  --question-types all \
  --subset all \
  --source-variant both \
  --tool-condition outline \
  --evidence-manifest "$COUNTING_EVIDENCE_MANIFEST" \
  --evidence-qc-filter "$COUNTING_QC_FILTER" \
  --missing-evidence-policy "$COUNTING_MISSING_POLICY" \
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
  --output-root "eval_results/counting/raw_runs/${MODEL_TAG}_both_tool_outline_pad005_max3_low_priority_v1"
```

### 6.7 P1：both crop

```bash
python eval_code/counting/eval_closed_vlm.py \
  --input-mode both \
  --backbone-model "$MODEL_ALIAS" \
  --judge-model "$JUDGE_MODEL" \
  --question-types all \
  --subset all \
  --source-variant both \
  --tool-condition crop \
  --evidence-manifest "$COUNTING_EVIDENCE_MANIFEST" \
  --evidence-qc-filter "$COUNTING_QC_FILTER" \
  --missing-evidence-policy "$COUNTING_MISSING_POLICY" \
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
  --output-root "eval_results/counting/raw_runs/${MODEL_TAG}_both_tool_crop_pad005_max3_low_priority_v1"
```

### 6.8 P2：both bbox

```bash
python eval_code/counting/eval_closed_vlm.py \
  --input-mode both \
  --backbone-model "$MODEL_ALIAS" \
  --judge-model "$JUDGE_MODEL" \
  --question-types all \
  --subset all \
  --source-variant both \
  --tool-condition bbox \
  --evidence-manifest "$COUNTING_EVIDENCE_MANIFEST" \
  --evidence-qc-filter "$COUNTING_QC_FILTER" \
  --missing-evidence-policy "$COUNTING_MISSING_POLICY" \
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
  --output-root "eval_results/counting/raw_runs/${MODEL_TAG}_both_tool_bbox_pad005_max3_low_priority_v1"
```

## 7. 开源模型命令

以下命令每个都需要分别对所有开源模型变量块执行。wrapper 会自动启动 backbone server 和 judge server，并自动规划 GPU placement。

### 7.1 P0：cf_only zoom_panel

```bash
python eval_code/run_server_eval.py \
  --framework "$FRAMEWORK" \
  --model "$HF_MODEL_ID" \
  --served-model-name "$SERVED_MODEL_NAME" \
  --benchmark counting \
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
  --question-types all \
  --subset all \
  --source-variant both \
  --tool-condition zoom_panel \
  --evidence-manifest "$COUNTING_EVIDENCE_MANIFEST" \
  --evidence-qc-filter "$COUNTING_QC_FILTER" \
  --missing-evidence-policy "$COUNTING_MISSING_POLICY" \
  --image-group-labels on \
  --prime off \
  --temperature 0 \
  --max-output-tokens 2048 \
  --judge-max-output-tokens 512 \
  --judge-structured-output auto \
  --closed-form-structured-output auto \
  --reasoning-effort off \
  --resume off \
  --output-root "eval_results/counting/raw_runs/${MODEL_TAG}_cf_only_tool_zoom_panel_pad005_max3_v1"
```

### 7.2 P0：cf_only outline

```bash
python eval_code/run_server_eval.py \
  --framework "$FRAMEWORK" \
  --model "$HF_MODEL_ID" \
  --served-model-name "$SERVED_MODEL_NAME" \
  --benchmark counting \
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
  --question-types all \
  --subset all \
  --source-variant both \
  --tool-condition outline \
  --evidence-manifest "$COUNTING_EVIDENCE_MANIFEST" \
  --evidence-qc-filter "$COUNTING_QC_FILTER" \
  --missing-evidence-policy "$COUNTING_MISSING_POLICY" \
  --image-group-labels on \
  --prime off \
  --temperature 0 \
  --max-output-tokens 2048 \
  --judge-max-output-tokens 512 \
  --judge-structured-output auto \
  --closed-form-structured-output auto \
  --reasoning-effort off \
  --resume off \
  --output-root "eval_results/counting/raw_runs/${MODEL_TAG}_cf_only_tool_outline_pad005_max3_v1"
```

### 7.3 P0：cf_only crop

```bash
python eval_code/run_server_eval.py \
  --framework "$FRAMEWORK" \
  --model "$HF_MODEL_ID" \
  --served-model-name "$SERVED_MODEL_NAME" \
  --benchmark counting \
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
  --question-types all \
  --subset all \
  --source-variant both \
  --tool-condition crop \
  --evidence-manifest "$COUNTING_EVIDENCE_MANIFEST" \
  --evidence-qc-filter "$COUNTING_QC_FILTER" \
  --missing-evidence-policy "$COUNTING_MISSING_POLICY" \
  --image-group-labels on \
  --prime off \
  --temperature 0 \
  --max-output-tokens 2048 \
  --judge-max-output-tokens 512 \
  --judge-structured-output auto \
  --closed-form-structured-output auto \
  --reasoning-effort off \
  --resume off \
  --output-root "eval_results/counting/raw_runs/${MODEL_TAG}_cf_only_tool_crop_pad005_max3_v1"
```

### 7.4 P1：cf_only bbox

```bash
python eval_code/run_server_eval.py \
  --framework "$FRAMEWORK" \
  --model "$HF_MODEL_ID" \
  --served-model-name "$SERVED_MODEL_NAME" \
  --benchmark counting \
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
  --question-types all \
  --subset all \
  --source-variant both \
  --tool-condition bbox \
  --evidence-manifest "$COUNTING_EVIDENCE_MANIFEST" \
  --evidence-qc-filter "$COUNTING_QC_FILTER" \
  --missing-evidence-policy "$COUNTING_MISSING_POLICY" \
  --image-group-labels on \
  --prime off \
  --temperature 0 \
  --max-output-tokens 2048 \
  --judge-max-output-tokens 512 \
  --judge-structured-output auto \
  --closed-form-structured-output auto \
  --reasoning-effort off \
  --resume off \
  --output-root "eval_results/counting/raw_runs/${MODEL_TAG}_cf_only_tool_bbox_pad005_max3_low_priority_v1"
```

### 7.5 P1：both zoom_panel

```bash
python eval_code/run_server_eval.py \
  --framework "$FRAMEWORK" \
  --model "$HF_MODEL_ID" \
  --served-model-name "$SERVED_MODEL_NAME" \
  --benchmark counting \
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
  --question-types all \
  --subset all \
  --source-variant both \
  --tool-condition zoom_panel \
  --evidence-manifest "$COUNTING_EVIDENCE_MANIFEST" \
  --evidence-qc-filter "$COUNTING_QC_FILTER" \
  --missing-evidence-policy "$COUNTING_MISSING_POLICY" \
  --image-group-labels on \
  --prime off \
  --temperature 0 \
  --max-output-tokens 2048 \
  --judge-max-output-tokens 512 \
  --judge-structured-output auto \
  --closed-form-structured-output auto \
  --reasoning-effort off \
  --resume off \
  --output-root "eval_results/counting/raw_runs/${MODEL_TAG}_both_tool_zoom_panel_pad005_max3_low_priority_v1"
```

### 7.6 P1：both outline

```bash
python eval_code/run_server_eval.py \
  --framework "$FRAMEWORK" \
  --model "$HF_MODEL_ID" \
  --served-model-name "$SERVED_MODEL_NAME" \
  --benchmark counting \
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
  --question-types all \
  --subset all \
  --source-variant both \
  --tool-condition outline \
  --evidence-manifest "$COUNTING_EVIDENCE_MANIFEST" \
  --evidence-qc-filter "$COUNTING_QC_FILTER" \
  --missing-evidence-policy "$COUNTING_MISSING_POLICY" \
  --image-group-labels on \
  --prime off \
  --temperature 0 \
  --max-output-tokens 2048 \
  --judge-max-output-tokens 512 \
  --judge-structured-output auto \
  --closed-form-structured-output auto \
  --reasoning-effort off \
  --resume off \
  --output-root "eval_results/counting/raw_runs/${MODEL_TAG}_both_tool_outline_pad005_max3_low_priority_v1"
```

### 7.7 P1：both crop

```bash
python eval_code/run_server_eval.py \
  --framework "$FRAMEWORK" \
  --model "$HF_MODEL_ID" \
  --served-model-name "$SERVED_MODEL_NAME" \
  --benchmark counting \
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
  --question-types all \
  --subset all \
  --source-variant both \
  --tool-condition crop \
  --evidence-manifest "$COUNTING_EVIDENCE_MANIFEST" \
  --evidence-qc-filter "$COUNTING_QC_FILTER" \
  --missing-evidence-policy "$COUNTING_MISSING_POLICY" \
  --image-group-labels on \
  --prime off \
  --temperature 0 \
  --max-output-tokens 2048 \
  --judge-max-output-tokens 512 \
  --judge-structured-output auto \
  --closed-form-structured-output auto \
  --reasoning-effort off \
  --resume off \
  --output-root "eval_results/counting/raw_runs/${MODEL_TAG}_both_tool_crop_pad005_max3_low_priority_v1"
```

### 7.8 P2：both bbox

```bash
python eval_code/run_server_eval.py \
  --framework "$FRAMEWORK" \
  --model "$HF_MODEL_ID" \
  --served-model-name "$SERVED_MODEL_NAME" \
  --benchmark counting \
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
  --question-types all \
  --subset all \
  --source-variant both \
  --tool-condition bbox \
  --evidence-manifest "$COUNTING_EVIDENCE_MANIFEST" \
  --evidence-qc-filter "$COUNTING_QC_FILTER" \
  --missing-evidence-policy "$COUNTING_MISSING_POLICY" \
  --image-group-labels on \
  --prime off \
  --temperature 0 \
  --max-output-tokens 2048 \
  --judge-max-output-tokens 512 \
  --judge-structured-output auto \
  --closed-form-structured-output auto \
  --reasoning-effort off \
  --resume off \
  --output-root "eval_results/counting/raw_runs/${MODEL_TAG}_both_tool_bbox_pad005_max3_low_priority_v1"
```

## 8. Dry-Run 与检查命令

正式运行前建议先对每个模型做一个小 dry-run。示例：

```bash
python eval_code/counting/eval_closed_vlm.py \
  --input-mode cf_only \
  --backbone-model "$MODEL_ALIAS" \
  --judge-model "$JUDGE_MODEL" \
  --question-types all \
  --subset all \
  --source-variant both \
  --tool-condition zoom_panel \
  --evidence-manifest "$COUNTING_EVIDENCE_MANIFEST" \
  --evidence-qc-filter "$COUNTING_QC_FILTER" \
  --missing-evidence-policy "$COUNTING_MISSING_POLICY" \
  --image-group-labels on \
  --prime off \
  --max-samples 2 \
  --dry-run
```

已有结果汇总：

```bash
python eval_code/counting/eval_closed_vlm.py \
  --report \
  --output-root eval_results/counting/raw_runs
```

生成分析表：

```bash
python eval_code/counting/analyze_results.py \
  --input-root eval_results/counting/raw_runs \
  --output-root eval_results/counting/tables/step2_counting_agentic_tools_v1
```

逐图难度分析：

```bash
python eval_code/counting/analyze_item_difficulty.py \
  --input-root eval_results/counting/raw_runs \
  --output eval_results/counting/tables/step2_counting_agentic_tools_v1/by_image_difficulty.csv
```

生成图表：

```bash
python eval_code/counting/make_figures.py \
  --tables-root eval_results/counting/tables/step2_counting_agentic_tools_v1 \
  --figures-root eval_results/counting/figures/step2_counting_agentic_tools_v1
```

## 9. 报告建议

主文优先报告：

- 第一阶段 `cf_only raw` baseline。
- 第二阶段 `cf_only zoom_panel`。
- 第二阶段 `cf_only outline`。
- 第二阶段 `cf_only crop`。

附录或 ablation 报告：

- `cf_only bbox`
- `both zoom_panel / outline / crop`
- `both bbox`

核心表格应包含：

- Accuracy
- Bias Rate
- Other Rate
- Delta Accuracy vs Raw
- Delta Bias vs Raw
- 按 `subset` 拆分：`bird / insect / hand_paw`
- 按 `edit_type` 拆分：`wings_add / wings_remove / add_to_six / add_to_five`
- 按 `count_direction` 拆分：`more_than_expected / fewer_than_expected`

如果某个 tool 降低 bias 但显著提高 `other`，不能简单解释为工具有效；应单独讨论它可能让模型更不确定，而不是真正理解数量。
