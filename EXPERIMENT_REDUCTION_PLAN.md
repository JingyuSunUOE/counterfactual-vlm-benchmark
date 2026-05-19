# Benchmark 实验总览与执行索引

本文档是内部实验安排总览，用于统一后续测试规划、模型选择、执行优先级和结果命名规则。它不作为论文正文、数据集文档或开源 README 的一部分。

具体可复制运行命令以五个 step plan 为准；本文档不再重复维护完整命令，避免和分阶段计划产生冲突。

## 1. 文档定位

当前 benchmark 覆盖五个大类：

- `if_exist`
- `counting`
- `fashion`
- `industry`
- `medical_modality`

整体实验分成两条论证线：

- Benchmark validity：证明数据集和问题设计确实暴露 VLM 的语言先验偏差。
- Agentic vision tool efficacy：评估额外视觉证据是否能帮助模型克服语言先验偏差。

本文档只回答上层问题：

- 每个阶段要验证什么。
- 使用哪些模型。
- 哪些实验是主实验，哪些是低优先级 ablation。
- 结果应该如何命名、过滤和比较。

## 2. 五阶段计划索引

| 阶段 | 文件 | 目标 | 主输入模式 | 主条件 |
| --- | --- | --- | --- | --- |
| Step 1 | `EXPERIMENT_STEP1_BENCHMARK_VALIDITY_PLAN.md` | 验证 benchmark 本身成立：原图可识别，CF 会被语言先验干扰，raw both 不能自然解决问题 | `orig_only`、`cf_only`、`both` | `raw` |
| Step 2 | `EXPERIMENT_STEP2_COUNTING_AGENTIC_VISION_PLAN.md` | 测试 counting 中视觉证据是否帮助模型做可见数量判断 | `cf_only` | `zoom_panel`、`outline`、`crop` |
| Step 3 | `EXPERIMENT_STEP3_IF_EXIST_AGENTIC_VISION_PLAN.md` | 测试 if_exist 中视觉证据是否帮助模型判断目标属性存在性 | `cf_only` | `crop`、`zoom_panel` |
| Step 4 | `EXPERIMENT_STEP4_FASHION_INDUSTRY_AGENTIC_VISION_PLAN.md` | 测试 fashion / industry 中视觉证据是否帮助模型判断品牌、颜色、形状、布局等属性 | `cf_only` | `crop`、`zoom_panel` |
| Step 5 | `EXPERIMENT_STEP5_MEDICAL_AGENTIC_VISION_PLAN.md` | 测试 medical 中 label-derived 定位证据是否帮助模型判断 MRI modality consistency | `cf_only` | `contour`、`zoom_panel` |

执行原则：

- Step 1 是所有后续 tool 实验的 raw baseline。
- Step 2 到 Step 5 不重复 raw，也不测试 `orig_only`。
- Step 2 到 Step 5 优先跑 `cf_only`，`both` 只作为低优先级 ablation。
- 每个 step 文件中的命令是该阶段的 source of truth。

## 3. 模型选择

### 3.1 闭源模型

| Provider | Formal model | Artifact tag |
| --- | --- | --- |
| OpenAI | `gpt-5` | `gpt5` |
| Anthropic | `claude-sonnet-4` / `sonnet-4` | `claude_sonnet4` |
| Google | `gemini-3.1-pro-preview` | `gemini31pro_preview` |

说明：

- OpenAI 固定为 `gpt-5`，不再保留 `gpt-4.1` 作为正式模型名。
- Claude 固定为 Sonnet 4，避免后续版本切换影响横向比较。
- Gemini 使用 preview 模型时，目录名、run config 和论文表格必须保留 `preview` 标记。
- 闭源 judge 默认使用 `gpt-4o-mini`。

### 3.2 开源 VLM

开源模型必须使用 instruct / instruction-tuned 版本，原因是评测依赖稳定格式遵循和 structured output。

| Family | Small model | Large model | Artifact tags |
| --- | --- | --- | --- |
| Qwen3-VL | `Qwen/Qwen3-VL-8B-Instruct` | `Qwen/Qwen3-VL-32B-Instruct` | `qwen3vl8b`、`qwen3vl32b` |
| InternVL | `OpenGVLab/InternVL3_5-8B` | `OpenGVLab/InternVL3_5-38B-HF` | `internvl35_8b`、`internvl35_38b` |
| Gemma | `google/gemma-4-E4B-it` | `google/gemma-4-31B-it` | `gemma4_e4b_it`、`gemma4_31b_it` |

额外 upper-bound 模型：

| Family | Model | Artifact tag | 用途 |
| --- | --- | --- | --- |
| Qwen2.5-VL | `Qwen/Qwen2.5-VL-72B-Instruct` | `qwen25vl72b` | 70B 级开源上界，不参与严格 family scaling |

开源 judge：

```text
Qwen/Qwen3-8B-Instruct
```

运行时建议 server 暴露名：

```text
qwen3-8b
```

说明：

- Judge 不需要视觉输入，不使用 VLM judge。
- 开源 backbone 和 judge 都通过 OpenAI-compatible server runner 运行。
- `eval_code/run_server_eval.py` 默认使用自动 GPU/显存联合分配；正式命令不需要手动指定 GPU。
- 如果服务端使用本地 alias，artifact 中必须记录真实 `server_model_id`、`served_model_name`、`judge_model` 和 `judge_provider`。

## 4. 全局评测协议

正式结果应尽量使用统一协议：

| 设置 | 默认值 | 说明 |
| --- | --- | --- |
| `--judge-structured-output` | `auto` | Judge 默认尝试结构化输出，失败时按 runner policy fallback |
| `--closed-form-structured-output` | `auto` | MC / yes-no / open answer 默认尝试结构化输出 |
| `--reasoning-effort` | `off` | 不主动开启 extended thinking |
| `--max-output-tokens` | `2048` | 降低空回答或 output-limit error |
| `--judge-max-output-tokens` | `512` | Judge 输出只需要 label 和短 reason |
| `--empty-response-policy` | `retry_once` | 空回答先重试一次，仍为空再按规则记为 `other` 或 error |

Tool-condition 正式默认：

| 设置 | 默认值 | 说明 |
| --- | --- | --- |
| `--evidence-qc-filter` | `pass` 或人工确认后的 `pass_review` | 只使用 QC 可接受的视觉证据 |
| `--missing-evidence-policy` | `skip` | 缺失 evidence 的样本不发 API，不污染 denominator |
| `--image-group-labels` | `on` | Tool runs 中显式分组辅助模型理解多图输入 |

## 5. 输入模式策略

| 输入模式 | Step 1 raw validity | Tool efficacy | 说明 |
| --- | --- | --- | --- |
| `orig_only` | 必须保留 | 不测试 | 证明原图和问题本身不是主要难点 |
| `cf_only` | 必须保留 | 主实验 | 最直接测试语言先验偏差和 tool 是否有效 |
| `both` | 必须保留 | 低优先级 ablation | 成本高、解释复杂；用于观察双图是否进一步帮助 |

Step 1 中 `both + raw` 的语义是中性双图差异比较，不暴露哪张是 original 或 CF。

Tool-condition 中的 `both` 仍使用 image group 语义：original group 在前，目标 CF group 在后。该设置用于评估“给定目标组和额外视觉视图时，tool 是否帮助纠正偏差”。

## 6. Tool 优先级总览

| Benchmark | P0 主实验 | P1 补充 | P2 低优先级 |
| --- | --- | --- | --- |
| `counting` | `cf_only + zoom_panel`、`cf_only + outline`、`cf_only + crop` | `cf_only + bbox` | `both + zoom_panel/outline/crop/bbox` |
| `if_exist` | `cf_only + crop`、`cf_only + zoom_panel` | `cf_only + bbox` | `both + crop/zoom_panel/bbox` |
| `fashion` / `industry` | `cf_only + crop`、`cf_only + zoom_panel` | `cf_only + bbox` | `both + crop/zoom_panel/bbox` |
| `medical_modality` | `cf_only + contour`、`cf_only + zoom_panel` | `cf_only + bbox/crop` | `both + contour/zoom_panel/bbox/crop` |

Tool 选择原则：

- `crop`：提供局部细节，适合自然图像大多数类别。
- `zoom_panel`：保留全局上下文和局部放大，适合需要空间定位或上下文的任务。
- `bbox`：轻量定位证据，但对细节理解帮助可能有限。
- `outline`：counting-specific，帮助模型看到目标物体边界，不直接泄露数量答案。
- `contour`：medical-specific，来自 BraTS segmentation label，只定位 tumor，不改变 tumor intensity。

不作为主实验的条件：

- `mask_overlay`：可能遮挡或改变目标视觉属性，引入 tool-quality confound。
- `region / part prompt evidence`：可能直接泄露任务关键属性，例如 wings、fingers、toes。
- `tool_bundle`：输入图过多、成本高，只可作为未来 upper-bound ablation，不进入当前主计划。

## 7. 结果命名规则

目录名应同时表达模型、benchmark、输入模式、条件和版本。建议规则：

```text
<model_tag>_<benchmark_or_domain>_<input_mode>_<condition>_<scope>_v<version>
```

示例：

```text
gpt5_counting_cf_only_tool_zoom_panel_v1
claude_sonnet4_if_exist_cf_only_tool_crop_v1
gemini31pro_preview_medical_cf_only_tool_contour_clean_brats2023_10pct_v1
qwen3vl32b_fashion_cf_only_tool_zoom_panel_v1
```

特殊标记：

| 标记 | 含义 |
| --- | --- |
| `pilot` | 探索性测试，不进入正式表格 |
| `smoke` | 小规模功能检查，不进入正式表格 |
| `debug` | 调试运行，不进入正式表格 |
| `legacy_second_image` | 旧 both raw 语义，不与新 neutral pair 结果合并 |
| `neutral_pair_difference` | 新 both raw 中性差异比较语义 |
| `10pct` | 抽样比例，例如 medical 的 10% 运行 |
| `clean_brats2023` | medical 主实验设置 |

模型名规范：

- 不再使用 `gpt41` 作为正式结果模型名。
- Gemini preview 结果统一使用 `gemini31pro_preview`。
- 如果历史结果只做模型名归一化，必须保留原始 run provenance 字段，避免误解为重新跑过。

## 8. 指标与分析规则

主标签保持三类：

- `correct`：模型基于可见图像给出符合当前问题目标的答案。
- `biased`：模型被语言先验、常见外观或 canonical attribute 带偏。
- `other`：回答模糊、拒答、无关、格式错误、视觉不可判定或自相矛盾。

论文主表建议报告：

```text
Accuracy
Bias Rate
Other Rate
Delta Accuracy vs Raw
Delta Bias vs Raw
```

比较规则：

- Raw-vs-tool delta 只在模型、benchmark、input mode、question type、source variant、QC filter 和 sample subset 一致时计算。
- 如果 tool run 因 evidence 缺失跳过样本，raw baseline 应使用同一个 evidence-valid subset。
- Pilot / smoke / debug / legacy runs 不进入正式聚合。
- `other` 不能简单视为模型无 bias；它代表格式、可判定性或任务理解失败，应单独报告。

## 9. 推荐执行顺序

1. 完成 Step 1 benchmark validity 的正式 raw 结果，覆盖闭源模型和主要开源模型。
2. 优先跑 Step 2 到 Step 5 的 P0 `cf_only` tool experiments。
3. 根据 P0 是否显示 bias reduction，再决定是否运行 P1 / P2。
4. 对每个 benchmark 先完成一个闭源代表模型的 smoke，再扩展到全部闭源模型。
5. 开源模型优先跑最有代表性的 P0 tool condition，避免对 6 个开源模型做全 tool 全输入模式矩阵。
6. 最后统一运行 analysis、figures 和 per-benchmark reports。

## 10. 维护规则

- 本文件只维护上层策略，不维护完整命令。
- 具体命令只写在五个 step plan 中。
- 如果某个 step 文件的实验设置变化，应同步更新本文档中的阶段摘要和优先级表。
- 如果模型版本变化，应先更新本文档的模型选择，再更新对应 step plan 的命令。
- 如果某个 pilot 结果被提升为正式结果，应规范目录名、确认 run config，并从正式统计中排除旧 pilot 重复项。
