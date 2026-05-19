# 第一阶段实验计划：验证 Benchmark 问题成立

本文档是内部实验执行计划，用于第一阶段 raw benchmark validity 实验。它不作为论文正文、数据集 README 或开源说明文件。

## 1. 实验目的

第一阶段只验证我们构建的五个反事实 benchmark 是否确实体现了语言先验对 VLM 的影响。该阶段不测试 agentic vision tools，不使用 `bbox`、`crop`、`zoom_panel`、`contour` 或 `tool_bundle`。

核心论证链条分三步：

1. `orig_only`：模型在原图上应该表现较好，说明原图质量、问题本身和评测流程不是主要瓶颈。
2. `cf_only`：模型在反事实图像上应该明显下降，且 `biased` rate 上升，说明语言先验会诱导模型忽略可见反事实内容。
3. `both + raw`：同时给原图和反事实图，在不泄露哪张是原图/反事实图的前提下，模型不应完全恢复到原图水平；如果提升有限或仍然高 bias，说明仅给参考图不能自然解决语言先验问题。

该阶段的主要指标仍为：

- `correct`：回答与当前图像或图像对的可见目标属性一致。
- `biased`：回答明显依赖语言先验、常识模板或原始类别/模态期望。
- `other`：回答模糊、不可判定、格式错误、无关或无法归入 correct/biased。

## 2. 实验范围

五个数据大类：

| 大类 | Runner | 说明 |
| --- | --- | --- |
| `if_exist` | `eval_code/if_exist/eval_closed_vlm.py` / `eval_server_vlm.py` | 物体局部属性存在性反事实 |
| `counting` | `eval_code/counting/eval_closed_vlm.py` / `eval_server_vlm.py` | wings / fingers / toes 等数量反事实 |
| `fashion` | `eval_code/fashion_industry/eval_pipeline.py` | 时尚品牌/物体属性反事实 |
| `industry` | `eval_code/fashion_industry/eval_pipeline.py` | 工业/日常物体属性反事实 |
| `medical` | `eval_code/medical_modality/eval_closed_vlm.py` / `eval_server_vlm.py` | MRI tumor-region modality consistency 反事实 |

输入模式：

| 输入模式 | 第一阶段含义 | 预期 |
| --- | --- | --- |
| `orig_only` | 只给原图 | 应明显好于 cf，作为 sanity baseline |
| `cf_only` | 只给反事实图 | 应出现低 accuracy / 高 bias |
| `both` | 给两张相关图像，raw 条件下不暴露身份 | 不应完全消除 bias |

统一设置：

```text
tool_condition = raw
prime = off
temperature = 0
max_output_tokens = 2048
judge_max_output_tokens = 512
judge_structured_output = auto
closed_form_structured_output = auto
reasoning_effort = off
```

说明：

- `prime off` 并不表示没有语言先验；当前问题文本本身已经包含 benchmark-specific prior statement。
- `both + raw` 使用 neutral pair-difference 语义，不在 prompt 中告诉模型哪张是 original 或 counterfactual。
- `medical` 使用 `clean` visualization，且正式报告建议使用固定 `10%` stratified sample。

## 3. 模型清单

### 3.1 闭源模型

闭源模型使用 native closed runner，judge 统一为 `gpt-4o-mini`。

运行每个命令前，先复制其中一个模型变量块。

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

### 3.2 开源模型

开源模型默认使用 `eval_code/run_server_eval.py` 自动启动 OpenAI-compatible backbone server 和 text-only judge server。该 wrapper 会：

- 启动 backbone VLM server。
- 可选启动 judge server。
- 等待 `/v1/models` ready。
- 自动向下游 runner 注入 `--server-url`、`--server-model-id`、`--judge-provider server`、`--judge-model/--judge` 和 `--judge-server-url`。
- benchmark 结束后默认清理 server 进程。

因此正式命令不要在 `--` 后手动传 `--server-url`、`--server-model-id`、`--judge-provider`、`--judge-model`、`--judge` 或 `--judge-server-url`。

如果你已经手动启动了 server，才使用各 benchmark 的 `eval_server_vlm.py` 或 `eval_pipeline.py --open-backend server` 直连模式；本文件默认不采用该模式。

运行每个开源命令前，先设置通用 server 变量。默认让代码自动检测 CUDA 设备和显存，并自动决定 backbone / judge 是分卡加载还是单卡共享。只有排查问题或固定复现实验时，才额外手动传 `--gpu-ids` / `--judge-gpu-ids`。

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

然后复制其中一个 backbone 模型变量块。所有 backbone 都应使用 instruct/chat/IT 版本，不使用 base 或 thinking-only 版本。

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

## 4. 闭源模型命令

以下命令每个都需要分别对 `gpt-5`、`claude-sonnet-4`、`gemini-3.1-pro-preview` 三个变量块执行。

### 4.1 if_exist

`orig_only`：

```bash
python eval_code/if_exist/eval_closed_vlm.py \
  --input-mode orig_only \
  --backbone-model "$MODEL_ALIAS" \
  --judge-model "$JUDGE_MODEL" \
  --question-types all \
  --source-variant both \
  --tool-condition raw \
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
  --output-root "eval_results/if_exist/raw_runs/${MODEL_TAG}_orig_only_raw_validity_v1"
```

`cf_only`：

```bash
python eval_code/if_exist/eval_closed_vlm.py \
  --input-mode cf_only \
  --backbone-model "$MODEL_ALIAS" \
  --judge-model "$JUDGE_MODEL" \
  --question-types all \
  --source-variant both \
  --tool-condition raw \
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
  --output-root "eval_results/if_exist/raw_runs/${MODEL_TAG}_cf_only_raw_validity_v1"
```

`both`：

```bash
python eval_code/if_exist/eval_closed_vlm.py \
  --input-mode both \
  --backbone-model "$MODEL_ALIAS" \
  --judge-model "$JUDGE_MODEL" \
  --question-types all \
  --source-variant both \
  --tool-condition raw \
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
  --output-root "eval_results/if_exist/raw_runs/${MODEL_TAG}_both_raw_validity_v1"
```

### 4.2 counting

`orig_only`：

```bash
python eval_code/counting/eval_closed_vlm.py \
  --input-mode orig_only \
  --backbone-model "$MODEL_ALIAS" \
  --judge-model "$JUDGE_MODEL" \
  --question-types all \
  --subset all \
  --source-variant both \
  --tool-condition raw \
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
  --output-root "eval_results/counting/raw_runs/${MODEL_TAG}_orig_only_raw_validity_v1"
```

`cf_only`：

```bash
python eval_code/counting/eval_closed_vlm.py \
  --input-mode cf_only \
  --backbone-model "$MODEL_ALIAS" \
  --judge-model "$JUDGE_MODEL" \
  --question-types all \
  --subset all \
  --source-variant both \
  --tool-condition raw \
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
  --output-root "eval_results/counting/raw_runs/${MODEL_TAG}_cf_only_raw_validity_v1"
```

`both`：

```bash
python eval_code/counting/eval_closed_vlm.py \
  --input-mode both \
  --backbone-model "$MODEL_ALIAS" \
  --judge-model "$JUDGE_MODEL" \
  --question-types all \
  --subset all \
  --source-variant both \
  --tool-condition raw \
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
  --output-root "eval_results/counting/raw_runs/${MODEL_TAG}_both_raw_validity_v1"
```

### 4.3 fashion

`orig_only`：

```bash
python eval_code/fashion_industry/eval_pipeline.py \
  --phase cf \
  --domain fashion \
  --input-mode orig_only \
  --model "$MODEL_ALIAS" \
  --provider native \
  --question-types all \
  --tool-condition raw \
  --prime off \
  --judge "$JUDGE_MODEL" \
  --provider-cache auto \
  --cache-key-mode image_pair \
  --judge-structured-output auto \
  --closed-form-structured-output auto \
  --reasoning-effort off \
  --max-output-tokens 2048 \
  --judge-max-output-tokens 512 \
  --resume off
```

`cf_only`：

```bash
python eval_code/fashion_industry/eval_pipeline.py \
  --phase cf \
  --domain fashion \
  --input-mode cf_only \
  --model "$MODEL_ALIAS" \
  --provider native \
  --question-types all \
  --tool-condition raw \
  --prime off \
  --judge "$JUDGE_MODEL" \
  --provider-cache auto \
  --cache-key-mode image_pair \
  --judge-structured-output auto \
  --closed-form-structured-output auto \
  --reasoning-effort off \
  --max-output-tokens 2048 \
  --judge-max-output-tokens 512 \
  --resume off
```

`both`：

```bash
python eval_code/fashion_industry/eval_pipeline.py \
  --phase cf \
  --domain fashion \
  --input-mode both \
  --model "$MODEL_ALIAS" \
  --provider native \
  --question-types all \
  --tool-condition raw \
  --prime off \
  --judge "$JUDGE_MODEL" \
  --provider-cache auto \
  --cache-key-mode image_pair \
  --judge-structured-output auto \
  --closed-form-structured-output auto \
  --reasoning-effort off \
  --max-output-tokens 2048 \
  --judge-max-output-tokens 512 \
  --resume off
```

### 4.4 industry

`orig_only`：

```bash
python eval_code/fashion_industry/eval_pipeline.py \
  --phase cf \
  --domain industry \
  --input-mode orig_only \
  --model "$MODEL_ALIAS" \
  --provider native \
  --question-types all \
  --tool-condition raw \
  --prime off \
  --judge "$JUDGE_MODEL" \
  --provider-cache auto \
  --cache-key-mode image_pair \
  --judge-structured-output auto \
  --closed-form-structured-output auto \
  --reasoning-effort off \
  --max-output-tokens 2048 \
  --judge-max-output-tokens 512 \
  --resume off
```

`cf_only`：

```bash
python eval_code/fashion_industry/eval_pipeline.py \
  --phase cf \
  --domain industry \
  --input-mode cf_only \
  --model "$MODEL_ALIAS" \
  --provider native \
  --question-types all \
  --tool-condition raw \
  --prime off \
  --judge "$JUDGE_MODEL" \
  --provider-cache auto \
  --cache-key-mode image_pair \
  --judge-structured-output auto \
  --closed-form-structured-output auto \
  --reasoning-effort off \
  --max-output-tokens 2048 \
  --judge-max-output-tokens 512 \
  --resume off
```

`both`：

```bash
python eval_code/fashion_industry/eval_pipeline.py \
  --phase cf \
  --domain industry \
  --input-mode both \
  --model "$MODEL_ALIAS" \
  --provider native \
  --question-types all \
  --tool-condition raw \
  --prime off \
  --judge "$JUDGE_MODEL" \
  --provider-cache auto \
  --cache-key-mode image_pair \
  --judge-structured-output auto \
  --closed-form-structured-output auto \
  --reasoning-effort off \
  --max-output-tokens 2048 \
  --judge-max-output-tokens 512 \
  --resume off
```

### 4.5 medical

Medical validity 使用 `clean` visualization 和固定 `10%` stratified sample。

`orig_only`：

```bash
python eval_code/medical_modality/eval_closed_vlm.py \
  --input-mode orig_only \
  --backbone-model "$MODEL_ALIAS" \
  --judge-model "$JUDGE_MODEL" \
  --visual-version clean \
  --dataset brats2023 \
  --sample-fraction 0.10 \
  --sample-strategy stratified \
  --seed 20260518 \
  --question-types all \
  --tool-condition raw \
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
  --output-root "eval_results/medical_modality/raw_runs/${MODEL_TAG}_orig_only_raw_10pct_validity_v1"
```

`cf_only`：

```bash
python eval_code/medical_modality/eval_closed_vlm.py \
  --input-mode cf_only \
  --backbone-model "$MODEL_ALIAS" \
  --judge-model "$JUDGE_MODEL" \
  --visual-version clean \
  --dataset brats2023 \
  --sample-fraction 0.10 \
  --sample-strategy stratified \
  --seed 20260518 \
  --question-types all \
  --tool-condition raw \
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
  --output-root "eval_results/medical_modality/raw_runs/${MODEL_TAG}_cf_only_raw_10pct_validity_v1"
```

`both`：

```bash
python eval_code/medical_modality/eval_closed_vlm.py \
  --input-mode both \
  --backbone-model "$MODEL_ALIAS" \
  --judge-model "$JUDGE_MODEL" \
  --visual-version clean \
  --dataset brats2023 \
  --sample-fraction 0.10 \
  --sample-strategy stratified \
  --seed 20260518 \
  --question-types all \
  --tool-condition raw \
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
  --output-root "eval_results/medical_modality/raw_runs/${MODEL_TAG}_both_raw_10pct_validity_v1"
```

## 5. 开源模型命令

以下命令每个都需要分别对 7 个开源模型变量块执行。所有命令默认通过 `eval_code/run_server_eval.py` 自动启动 backbone server 和 judge server；不要提前手动启动 server，也不要在 `--` 后手动传 server/judge 连接参数。

### 5.1 if_exist

`orig_only`：

```bash
python eval_code/run_server_eval.py \
  --framework "$FRAMEWORK" \
  --model "$HF_MODEL_ID" \
  --served-model-name "$SERVED_MODEL_NAME" \
  --benchmark if_exist \
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
  --input-mode orig_only \
  --backbone-model "$MODEL_ALIAS" \
  --question-types all \
  --source-variant both \
  --tool-condition raw \
  --prime off \
  --temperature 0 \
  --max-output-tokens 2048 \
  --judge-max-output-tokens 512 \
  --judge-structured-output auto \
  --closed-form-structured-output auto \
  --reasoning-effort off \
  --resume off \
  --output-root "eval_results/if_exist/raw_runs/${MODEL_TAG}_orig_only_raw_validity_v1"
```

`cf_only`：

```bash
python eval_code/run_server_eval.py \
  --framework "$FRAMEWORK" \
  --model "$HF_MODEL_ID" \
  --served-model-name "$SERVED_MODEL_NAME" \
  --benchmark if_exist \
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
  --source-variant both \
  --tool-condition raw \
  --prime off \
  --temperature 0 \
  --max-output-tokens 2048 \
  --judge-max-output-tokens 512 \
  --judge-structured-output auto \
  --closed-form-structured-output auto \
  --reasoning-effort off \
  --resume off \
  --output-root "eval_results/if_exist/raw_runs/${MODEL_TAG}_cf_only_raw_validity_v1"
```

`both`：

```bash
python eval_code/run_server_eval.py \
  --framework "$FRAMEWORK" \
  --model "$HF_MODEL_ID" \
  --served-model-name "$SERVED_MODEL_NAME" \
  --benchmark if_exist \
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
  --source-variant both \
  --tool-condition raw \
  --prime off \
  --temperature 0 \
  --max-output-tokens 2048 \
  --judge-max-output-tokens 512 \
  --judge-structured-output auto \
  --closed-form-structured-output auto \
  --reasoning-effort off \
  --resume off \
  --output-root "eval_results/if_exist/raw_runs/${MODEL_TAG}_both_raw_validity_v1"
```

### 5.2 counting

`orig_only`：

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
  --input-mode orig_only \
  --backbone-model "$MODEL_ALIAS" \
  --question-types all \
  --subset all \
  --source-variant both \
  --tool-condition raw \
  --prime off \
  --temperature 0 \
  --max-output-tokens 2048 \
  --judge-max-output-tokens 512 \
  --judge-structured-output auto \
  --closed-form-structured-output auto \
  --reasoning-effort off \
  --resume off \
  --output-root "eval_results/counting/raw_runs/${MODEL_TAG}_orig_only_raw_validity_v1"
```

`cf_only`：

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
  --tool-condition raw \
  --prime off \
  --temperature 0 \
  --max-output-tokens 2048 \
  --judge-max-output-tokens 512 \
  --judge-structured-output auto \
  --closed-form-structured-output auto \
  --reasoning-effort off \
  --resume off \
  --output-root "eval_results/counting/raw_runs/${MODEL_TAG}_cf_only_raw_validity_v1"
```

`both`：

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
  --tool-condition raw \
  --prime off \
  --temperature 0 \
  --max-output-tokens 2048 \
  --judge-max-output-tokens 512 \
  --judge-structured-output auto \
  --closed-form-structured-output auto \
  --reasoning-effort off \
  --resume off \
  --output-root "eval_results/counting/raw_runs/${MODEL_TAG}_both_raw_validity_v1"
```

### 5.3 fashion

`orig_only`：

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
  --input-mode orig_only \
  --model "$MODEL_ALIAS" \
  --question-types all \
  --tool-condition raw \
  --prime off \
  --judge-structured-output auto \
  --closed-form-structured-output auto \
  --reasoning-effort off \
  --max-output-tokens 2048 \
  --judge-max-output-tokens 512 \
  --resume off
```

`cf_only`：

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
  --tool-condition raw \
  --prime off \
  --judge-structured-output auto \
  --closed-form-structured-output auto \
  --reasoning-effort off \
  --max-output-tokens 2048 \
  --judge-max-output-tokens 512 \
  --resume off
```

`both`：

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
  --tool-condition raw \
  --prime off \
  --judge-structured-output auto \
  --closed-form-structured-output auto \
  --reasoning-effort off \
  --max-output-tokens 2048 \
  --judge-max-output-tokens 512 \
  --resume off
```

### 5.4 industry

`orig_only`：

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
  --input-mode orig_only \
  --model "$MODEL_ALIAS" \
  --question-types all \
  --tool-condition raw \
  --prime off \
  --judge-structured-output auto \
  --closed-form-structured-output auto \
  --reasoning-effort off \
  --max-output-tokens 2048 \
  --judge-max-output-tokens 512 \
  --resume off
```

`cf_only`：

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
  --tool-condition raw \
  --prime off \
  --judge-structured-output auto \
  --closed-form-structured-output auto \
  --reasoning-effort off \
  --max-output-tokens 2048 \
  --judge-max-output-tokens 512 \
  --resume off
```

`both`：

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
  --tool-condition raw \
  --prime off \
  --judge-structured-output auto \
  --closed-form-structured-output auto \
  --reasoning-effort off \
  --max-output-tokens 2048 \
  --judge-max-output-tokens 512 \
  --resume off
```

### 5.5 medical

Medical validity 使用 `clean` visualization 和固定 `10%` stratified sample。

`orig_only`：

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
  --input-mode orig_only \
  --backbone-model "$MODEL_ALIAS" \
  --visual-version clean \
  --dataset brats2023 \
  --sample-fraction 0.10 \
  --sample-strategy stratified \
  --seed 20260518 \
  --question-types all \
  --tool-condition raw \
  --prime off \
  --temperature 0 \
  --max-output-tokens 2048 \
  --judge-max-output-tokens 512 \
  --judge-structured-output auto \
  --closed-form-structured-output auto \
  --reasoning-effort off \
  --resume off \
  --output-root "eval_results/medical_modality/raw_runs/${MODEL_TAG}_orig_only_raw_10pct_validity_v1"
```

`cf_only`：

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
  --visual-version clean \
  --dataset brats2023 \
  --sample-fraction 0.10 \
  --sample-strategy stratified \
  --seed 20260518 \
  --question-types all \
  --tool-condition raw \
  --prime off \
  --temperature 0 \
  --max-output-tokens 2048 \
  --judge-max-output-tokens 512 \
  --judge-structured-output auto \
  --closed-form-structured-output auto \
  --reasoning-effort off \
  --resume off \
  --output-root "eval_results/medical_modality/raw_runs/${MODEL_TAG}_cf_only_raw_10pct_validity_v1"
```

`both`：

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
  --visual-version clean \
  --dataset brats2023 \
  --sample-fraction 0.10 \
  --sample-strategy stratified \
  --seed 20260518 \
  --question-types all \
  --tool-condition raw \
  --prime off \
  --temperature 0 \
  --max-output-tokens 2048 \
  --judge-max-output-tokens 512 \
  --judge-structured-output auto \
  --closed-form-structured-output auto \
  --reasoning-effort off \
  --resume off \
  --output-root "eval_results/medical_modality/raw_runs/${MODEL_TAG}_both_raw_10pct_validity_v1"
```

## 6. 执行顺序建议

建议不要直接一次性跑完所有模型。推荐顺序：

1. 对 `gpt-5` 跑五类数据的 `orig_only / cf_only / both`，确认第一阶段指标形态符合预期。
2. 对 `claude-sonnet-4` 和 `gemini-3.1-pro-preview` 跑同样矩阵，观察闭源模型差异。
3. 对一个开源代表模型先跑完整 raw validity，例如 `qwen3-vl-32b`。
4. 如果 runner 和 judge server 都稳定，再扩展到其余开源模型。
5. 最后跑 `qwen2.5-vl-72b` upper-bound。

每个 run 完成后至少检查：

```bash
python eval_code/if_exist/eval_closed_vlm.py --report --output-root eval_results/if_exist/raw_runs
python eval_code/counting/eval_closed_vlm.py --report --output-root eval_results/counting/raw_runs
python eval_code/medical_modality/eval_closed_vlm.py --report --output-root eval_results/medical_modality/raw_runs
python eval_code/fashion_industry/eval_pipeline.py --report
```

## 7. 结果判定标准

第一阶段不是追求所有模型都失败，而是验证数据集是否形成清晰、可解释的现象。

理想结果形态：

| 输入模式 | 期望现象 |
| --- | --- |
| `orig_only` | accuracy 较高，bias 较低，说明原图和问题不是过难 |
| `cf_only` | accuracy 明显下降，bias 明显上升，说明语言先验影响反事实判断 |
| `both` | 相比 `cf_only` 可有提升，但不应完全恢复到 `orig_only`；若仍高 bias，说明简单参考图不能解决问题 |

需要警惕的异常：

- `orig_only` 也很差：可能说明问题设计过难、图像质量不佳或 judge 过严。
- `cf_only` 表现很好：可能说明该子集反事实太明显，语言先验诱导不够强。
- `both` 几乎完美：可能说明两图差异过于简单，或者 prompt 无意中暴露了比较目标。
- `other` rate 极高：优先检查 structured output、MC parser、judge prompt 或 max-output-tokens。
