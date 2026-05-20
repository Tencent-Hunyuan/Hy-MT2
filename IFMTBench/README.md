# Translation Instruction Following Benchmark

A benchmark for evaluating how well LLMs follow complex instructions during translation tasks. The benchmark covers **6 constraint types** across multiple languages, including single-constraint and multi-constraint scenarios.

## Constraint Types

| Constraint Type | Evaluation Method | Score Type |
|---|---|---|
| Glossary Compliance (术语表约束翻译) | Rule check + LLM Judge fallback | Gate (0/1) |
| Style Following (风格指令遵循) | LLM Judge | Continuous (0-5 → 0-1) |
| Background Disambiguation (带上下文背景翻译) | LLM Judge | Continuous (0-5 → 0-1) |
| Layout Preservation (布局保留翻译) | Rule check (delimiter-based) | Gate (0/1) |
| Structured Data (结构化数据翻译) | Rule check (format validation) | Gate (0/1) |
| Code/Tag Preservation (代码标签保留翻译) | Rule check (asset matching) | Gate (0/1) |

**Multi-constraint scoring**: `final_score = gate_score × avg(continuous_scores)`

## Data Format

### Test Data

Each line in the test JSONL files contains:

| Field | Required | Description |
|---|---|---|
| `input` | ✓ | Full prompt to send to the model |
| `output` | ✓ | Reference translation (ground truth) |
| `class` | ✓ | Constraint type(s), list format, determines scoring logic |
| `md5` | ✓ | Unique identifier |
| `origin_text` | ✓ | Original source text |
| `meta_data` | * | Metadata for code/layout evaluation (extracted_assets, primary_delimiter, etc.) |
| `term_dict` | * | Glossary mapping (for glossary constraint type) |
| `selected_style` | * | Target style (for style constraint type) |
| `generated_background` | * | Background context (for background constraint type) |
| `origin_language` | | Source language |
| `target_language` | | Target language |
| `instruction_lang` | | Language of the instruction |
| `original_input` | | Original short-form input before rewriting |

Fields marked with `*` are required for specific constraint types.

### Model Output Format

Your model output file should be a JSONL file where each line contains:

```json
{"md5": "<matching md5 from test data>", "response": "<model's translation output>"}
```

- **`md5`**: Must match the `md5` field from the test data to identify which test case this response corresponds to.
- **`response`**: The model's raw translation output (the text your model generated).

See `data/sample_output.jsonl` for an example.

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure LLM Judge API

Copy `.env.example` to `.env` and fill in your API credentials:

```bash
cp .env.example .env
# Edit .env with your API configuration
```

The LLM Judge supports any OpenAI-compatible API endpoint.

### 3. Prepare Your Model Output

Send each test item's `input` field to your model, collect the responses, and save them as:

```jsonl
{"md5": "c320a518b633b006b48719b403253e64", "response": "Your model's translation here..."}
{"md5": "6953e58b3d8ddf3161cbb3ff5f0de7f0", "response": "Another translation..."}
...
```

### 4. Run Evaluation

```bash
# Evaluate single-constraint data
python run_eval.py \
    --input_data data/test_single_constraint.jsonl \
    --input_response your_model_output.jsonl \
    --output_dir eval_results

# Evaluate both single and multi-constraint
python run_eval.py \
    --input_data data/test_single_constraint.jsonl data/test_multi_constraint.jsonl \
    --input_response your_model_output.jsonl \
    --output_dir eval_results

# Rule-only mode (no LLM Judge, for debugging)
python run_eval.py \
    --input_data data/test_single_constraint.jsonl \
    --input_response your_model_output.jsonl \
    --output_dir eval_results \
    --skip-llm
```

### 5. View Results

Results are saved to the output directory:
- `eval_details.jsonl` — Per-item scoring details
- `eval_summary.json` — Aggregated statistics

## Project Structure

```
openbench/
├── README.md              # English documentation
├── README_zh.md           # 中文文档
├── run_eval.py            # Evaluation entry point
├── config.py              # Configuration
├── .env.example           # Environment variable template
├── requirements.txt       # Dependencies
├── data/
│   ├── test_single_constraint.jsonl   # Single-constraint test data (4506 items)
│   ├── test_multi_constraint.jsonl    # Multi-constraint test data (2838 items)
│   └── sample_output.jsonl            # Example model output format
├── eval/
│   ├── __init__.py
│   ├── scoring.py         # Core scoring logic
│   ├── rule_validators.py # Rule-based validators
│   └── llm_judge.py       # LLM Judge module
└── scripts/
    └── prepare_data.py    # Data preprocessing script
```

## Evaluation Workflow

```
Test Data (JSONL)  +  Model Output (JSONL)
        │                      │
        └──────────┬───────────┘
                   ▼
            run_eval.py
                   │
        ┌──────────┴──────────┐
        ▼                     ▼
  Rule Validators        LLM Judge
  (glossary/layout/      (style/background/
   structured/code)       glossary fallback)
        │                     │
        └──────────┬──────────┘
                   ▼
          Score Composition
                   │
                   ▼
         eval_results/
         ├── eval_details.jsonl
         └── eval_summary.json
```

