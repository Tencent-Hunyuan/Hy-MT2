# 翻译指令遵循评测基准

本项目用于评测大语言模型在翻译任务中对复杂指令的遵循能力。评测覆盖 **6 种约束类型**，支持多语言，包含单约束和多约束场景。

## 约束类型

| 约束类型 | 评测方式 | 分数类型 |
|---|---|---|
| 术语表约束翻译 | 规则校验 + LLM Judge 保底 | 门控 (0/1) |
| 风格指令遵循 | LLM Judge | 连续 (0-5 → 0-1) |
| 带上下文背景翻译 | LLM Judge | 连续 (0-5 → 0-1) |
| 布局保留翻译 | 规则校验（分隔符切分） | 门控 (0/1) |
| 结构化数据翻译 | 规则校验（格式验证） | 门控 (0/1) |
| 代码标签保留翻译 | 规则校验（代码片段匹配） | 门控 (0/1) |

**多约束评分规则**: `final_score = gate_score × avg(continuous_scores)`
- `gate_score` = 所有门控分数相乘（任一为 0 则整体为 0）
- `continuous_scores` = 所有连续分数的平均值（无连续分数则为 1.0）

## 数据格式

### 测试数据

测试数据为 JSONL 格式，每行包含以下字段：

| 字段 | 必需 | 说明 |
|---|---|---|
| `input` | ✓ | 发送给模型的完整 prompt |
| `output` | ✓ | 参考翻译（ground truth） |
| `class` | ✓ | 约束类型列表，决定评分逻辑 |
| `md5` | ✓ | 唯一标识符 |
| `origin_text` | ✓ | 原始待翻译文本 |
| `meta_data` | * | 代码/布局类评测所需的元数据（extracted_assets、primary_delimiter 等） |
| `term_dict` | * | 术语映射表（术语表约束类型） |
| `selected_style` | * | 目标风格（风格约束类型） |
| `generated_background` | * | 背景上下文（背景约束类型） |
| `origin_language` | | 源语言 |
| `target_language` | | 目标语言 |
| `instruction_lang` | | 指令语言 |
| `original_input` | | 改写前的原始简短输入 |

标 `*` 的字段对特定约束类型是必需的。

### 模型输出格式

你的模型输出文件应为 JSONL 格式，每行包含：

```json
{"md5": "<与测试数据中的 md5 对应>", "response": "<模型的翻译输出>"}
```

- **`md5`**: 必须与测试数据中的 `md5` 字段匹配，用于标识对应的测试用例。
- **`response`**: 模型生成的原始翻译输出。

参见 `data/sample_output.jsonl` 了解示例格式。

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 配置 LLM Judge API

将 `.env.example` 复制为 `.env` 并填入 API 凭据：

```bash
cp .env.example .env
# 编辑 .env 填入你的 API 配置
```

LLM Judge 支持任何 OpenAI 兼容的 API 接口。

### 3. 准备模型输出

将测试数据中每条的 `input` 字段发送给你的模型，收集响应并保存为：

```jsonl
{"md5": "c320a518b633b006b48719b403253e64", "response": "你的模型翻译结果..."}
{"md5": "6953e58b3d8ddf3161cbb3ff5f0de7f0", "response": "另一条翻译..."}
...
```

### 4. 运行评测

```bash
# 评测单约束数据
python run_eval.py \
    --input_data data/test_single_constraint.jsonl \
    --input_response your_model_output.jsonl \
    --output_dir eval_results

# 同时评测单约束和多约束
python run_eval.py \
    --input_data data/test_single_constraint.jsonl data/test_multi_constraint.jsonl \
    --input_response your_model_output.jsonl \
    --output_dir eval_results

# 仅规则校验模式（跳过 LLM Judge，用于调试）
python run_eval.py \
    --input_data data/test_single_constraint.jsonl \
    --input_response your_model_output.jsonl \
    --output_dir eval_results \
    --skip-llm
```

### 5. 查看结果

结果保存在输出目录中：
- `eval_details.jsonl` — 每条数据的详细评分
- `eval_summary.json` — 汇总统计

## 项目结构

```
openbench/
├── README.md              # 英文文档
├── README_zh.md           # 中文文档
├── run_eval.py            # 评测入口
├── config.py              # 配置文件
├── .env.example           # 环境变量模板
├── requirements.txt       # 依赖
├── data/
│   ├── test_single_constraint.jsonl   # 单约束测试数据（4506 条）
│   ├── test_multi_constraint.jsonl    # 多约束测试数据（2838 条）
│   └── sample_output.jsonl            # 示例模型输出格式
├── eval/
│   ├── __init__.py
│   ├── scoring.py         # 评分核心逻辑
│   ├── rule_validators.py # 规则校验器
│   └── llm_judge.py       # LLM Judge 模块
└── scripts/
    └── prepare_data.py    # 数据预处理脚本
```

## 评测流程

```
测试数据 (JSONL)  +  模型输出 (JSONL)
        │                      │
        └──────────┬───────────┘
                   ▼
            run_eval.py
                   │
        ┌──────────┴──────────┐
        ▼                     ▼
    规则校验器            LLM Judge
  (术语表/布局/         (风格/背景/
   结构化/代码)          术语表保底)
        │                     │
        └──────────┬──────────┘
                   ▼
            分数合成
                   │
                   ▼
         eval_results/
         ├── eval_details.jsonl
         └── eval_summary.json
```

