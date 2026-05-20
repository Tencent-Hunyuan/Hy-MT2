# -*- coding: utf-8 -*-
"""
Evaluation configuration.

LLM Judge settings are configured via environment variables, see .env.example.
"""

import os

# LLM Judge API configuration
LLM_CONFIG = {
    "api_base": os.environ.get("LLM_API_BASE", ""),
    "api_key": os.environ.get("LLM_API_KEY", ""),
    "model_name": os.environ.get("LLM_MODEL_NAME", "gpt-4o"),
    "temperature": float(os.environ.get("LLM_TEMPERATURE", "0.0")),
    "max_tokens": int(os.environ.get("LLM_MAX_TOKENS", "8192")),
    "top_p": float(os.environ.get("LLM_TOP_P", "0.6")),
}

# Evaluation concurrency configuration
MAX_WORKERS = int(os.environ.get("EVAL_MAX_WORKERS", "20"))
MAX_RETRIES = int(os.environ.get("EVAL_MAX_RETRIES", "5"))
REQUEST_TIMEOUT = int(os.environ.get("EVAL_REQUEST_TIMEOUT", "360"))

# Class label -> evaluation dimension mapping
CLASS_TO_DIMENSION = {
    "机器翻译-术语表约束翻译": "glossary",
    "机器翻译-风格指令遵循": "style",
    "机器翻译-带上下文背景翻译": "background",
}

# Class categories grouped by evaluation method
CLASS_LLM_JUDGE = {"机器翻译-风格指令遵循", "机器翻译-带上下文背景翻译"}
CLASS_GLOSSARY = {"机器翻译-术语表约束翻译"}
CLASS_LAYOUT = {"机器翻译-布局保留翻译"}
CLASS_STRUCTURED = {"机器翻译-结构化数据翻译"}
CLASS_CODE = {"机器翻译-内联代码保留翻译", "机器翻译-代码标签保留翻译"}

# Gate classes (binary 0/1)
GATE_CLASSES = CLASS_GLOSSARY | CLASS_LAYOUT | CLASS_STRUCTURED | CLASS_CODE
# Continuous classes (0-5 normalized to 0-1)
CONTINUOUS_CLASSES = CLASS_LLM_JUDGE
