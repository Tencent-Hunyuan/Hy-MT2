# -*- coding: utf-8 -*-
"""
LLM Judge module: Calls LLM API to score style, background, and glossary compliance.

Supports any OpenAI-compatible API endpoint.
"""

import json
import re
import time
import random
import logging
from typing import Optional

import requests

from config import LLM_CONFIG, MAX_RETRIES, REQUEST_TIMEOUT

log = logging.getLogger(__name__)

# ============================================================
# Prompt Templates
# ============================================================

GLOSSARY_REWARD_PROMPT = '''
# ROLE
You are an expert Linguistic and Morphological Evaluator for a translation Reward Model. Your SOLE objective is to determine if the specified terminology from the instruction was correctly integrated into the target translation, accounting for complex morphological adaptations (declension, pluralization, tense, etc.).

# EVALUATION DATA
<instruction>
{user_instruction}
</instruction>

<ground_truth>
{ground_truth}
</ground_truth>

<model_output>
{target_translation}
</model_output>

# RUBRICS
### Glossary Compliance - [BINARY SCORING: 0 or 1]
Evaluate if the translation accurately incorporates the specific terminology provided in the instruction/background.
- [1] Perfect Adherence: Flawlessly integrated the required terms. Morphological adaptations (e.g., plurals, tense, part-of-speech, conjugations) are grammatically natural in the target language. It is acceptable if the term underwent necessary morphological changes compared to its base dictionary form.
- [0] Fatal Violation (Veto): Instant 0 if ANY of the following occur: unauthorized synonym substitution, fallback to generic dictionary translation, omission of the core concept, or severe grammatical corruption caused by forcing the term. 

# OUTPUT FORMAT
Output ONLY a single integer: `1` or `0`. 
Do NOT wrap it in JSON, Markdown, or any other formatting. Do NOT output any explanatory text.
'''

STYLE_AND_BACKGROUND_REWARD_PROMPT = '''
# ROLE
You are an advanced Reward Model designed for Reinforcement Learning (RL) of Large Language Models. Your primary function is to evaluate **Instruction Tracking and Constraint Satisfaction**. 
Do NOT evaluate basic translation fluency. Your SOLE objective is to score whether the model executed the specific holistic [Constraints] (Style and Background).

# EVALUATION DATA
<instruction>
{user_instruction}
</instruction>

<ground_truth>
{ground_truth}
</ground_truth>

<model_output>
{target_translation}
</model_output>

# RUBRICS
Analyze the <instruction>. If a constraint is NOT requested, output `null`. If activated, evaluate against the rubrics.

### 1. Style & Register (Style) - [0-5 SCALE]
- [Activation Condition]: Activate if the instruction requests a specific tone, persona, register, or formatting style.
- [5] Perfect Alignment: Tone and register are exceptionally distinct and consistent throughout.
- [4] Strong Alignment: Generally fits the required style, but 1-2 lexical choices feel slightly generic.
- [3] Marginal Pass: Follows the basic directional constraint, but leans heavily on standard, flavorless translation. 
- [2] Default/Generic: Ignored the stylistic constraint, reverting to a safe, bland machine translation tone.
- [1] Severe Deviation: Noticeable conflict with the requested style.
- [0] Rule Break: Wrong style AND included conversational filler/hallucinations, breaking the fourth wall.

### 2. Contextual Cohesion (Background) - [0-5 SCALE]
- [Activation Condition]: Activate if the instruction provides ANY preceding context, a background summary, or asks the translation to consider the "context" or "background".
- [5] Perfect Disambiguation: Masterfully leveraged the background summary to resolve potential ambiguities. Flawless logical cohesion.
- [4] Strong Utilization: Correctly used the summary to guide the translation, but feels slightly rigid when referencing the background.
- [3] Logically Consistent: Does not contradict the summary, but disambiguation is mediocre (literal translation).
- [2] Total Ignorance: Ignored the summary entirely, resulting in a disjointed literal translation.
- [1] Logical Contradiction: Directly contradicts the core logic or established facts in the background summary.
- [0] Severe Hallucination (Prompt Bleeding): Mistakenly translated the background summary itself as part of the target text.

# OUTPUT FORMAT
Output ONLY a valid JSON object. Do NOT wrap the JSON in Markdown code blocks (e.g., no ```json). 
{{
  "scores": {{
    "style": [0, 1, 2, 3, 4, 5, or null],
    "background": [0, 1, 2, 3, 4, 5, or null]
  }}
}}'''


# ============================================================
# API Calls
# ============================================================

def _call_llm(user_prompt: str) -> tuple:
    """Call LLM API (OpenAI-compatible endpoint)."""
    config = LLM_CONFIG
    url = config["api_base"].rstrip("/") + "/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {config['api_key']}",
    }
    json_data = {
        "model": config["model_name"],
        "messages": [
            {"role": "user", "content": user_prompt},
        ],
        "temperature": config["temperature"],
        "max_tokens": config["max_tokens"],
        "top_p": config["top_p"],
        "stream": False,
    }
    resp = requests.post(url, headers=headers, json=json_data, timeout=REQUEST_TIMEOUT)
    resp.raise_for_status()
    response_json = resp.json()
    content = response_json["choices"][0]["message"]["content"]
    reasoning = response_json["choices"][0]["message"].get("reasoning_content", "")
    return True, content.strip(), reasoning.strip() if reasoning else ""


def _call_llm_with_retry(user_prompt: str) -> tuple:
    """Call LLM with retry logic."""
    last_error = None
    for attempt in range(MAX_RETRIES + 1):
        try:
            success, content, reasoning = _call_llm(user_prompt)
            if success:
                return True, content, reasoning
            last_error = content
        except Exception as e:
            last_error = str(e)
        if attempt < MAX_RETRIES:
            time.sleep(random.uniform(1, min(2 ** attempt, 10)))
    return False, f"Failed after {MAX_RETRIES} retries: {last_error}", ""


def _parse_json_from_text(text: str) -> Optional[dict]:
    """Parse JSON from LLM output text."""
    if not text:
        return None
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    m = re.search(r'```(?:json)?\s*\n?(.*?)\n?\s*```', text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            pass
    s, e = text.find('{'), text.rfind('}')
    if s != -1 and e > s:
        try:
            return json.loads(text[s:e + 1])
        except json.JSONDecodeError:
            pass
    return None


# ============================================================
# Judge Scoring Functions
# ============================================================

def score_glossary_judge(user_instruction: str, ground_truth: str, target_translation: str) -> dict:
    """Glossary LLM Judge (binary 0/1)."""
    result = {"glossary": None, "if_score": None, "raw_response": None, "reasoning": None}
    prompt = GLOSSARY_REWARD_PROMPT.format(
        user_instruction=user_instruction,
        ground_truth=ground_truth,
        target_translation=target_translation,
    )
    success, text, reasoning = _call_llm_with_retry(prompt)
    if not success:
        log.error(f"Glossary Judge call failed: {text}")
        return result

    result["raw_response"] = text[:500]
    result["reasoning"] = reasoning[:500] if reasoning else None

    text_stripped = text.strip()
    score = None
    if text_stripped in ("0", "1"):
        score = int(text_stripped)
    else:
        m = re.search(r'\b([01])\b', text_stripped)
        if m:
            score = int(m.group(1))

    if score is not None:
        result["glossary"] = score
        result["if_score"] = float(score)
    else:
        log.warning(f"Glossary Judge parse failed: {text[:200]}")
    return result


def score_style_background_judge(user_instruction: str, ground_truth: str,
                                  target_translation: str, expected_dimension: str) -> dict:
    """Style/Background LLM Judge (0-5 scale, normalized to 0-1)."""
    result = {"style": None, "background": None, "classification_match": False,
              "if_score": None, "raw_response": None, "reasoning": None}
    prompt = STYLE_AND_BACKGROUND_REWARD_PROMPT.format(
        user_instruction=user_instruction,
        ground_truth=ground_truth,
        target_translation=target_translation,
    )
    success, text, reasoning = _call_llm_with_retry(prompt)
    if not success:
        log.error(f"Style/Background Judge call failed: {text}")
        return result

    result["raw_response"] = text[:500]
    result["reasoning"] = reasoning[:500] if reasoning else None
    parsed = _parse_json_from_text(text)
    if parsed is None:
        log.warning(f"Style/Background Judge JSON parse failed: {text[:200]}")
        return result

    scores = parsed.get("scores", parsed) if isinstance(parsed, dict) else parsed

    def _extract_score(val):
        if val is None:
            return None
        if isinstance(val, (int, float)):
            return val
        if isinstance(val, list) and len(val) > 0:
            return val[0] if isinstance(val[0], (int, float)) else None
        return None

    if not isinstance(scores, dict):
        score_val = _extract_score(scores)
        result["style"] = score_val if expected_dimension == "style" else None
        result["background"] = score_val if expected_dimension == "background" else None
    else:
        result["style"] = _extract_score(scores.get("style"))
        result["background"] = _extract_score(scores.get("background"))

    dim_value = result.get(expected_dimension)
    if dim_value is not None:
        result["classification_match"] = True
        result["if_score"] = dim_value / 5.0
    else:
        result["classification_match"] = False
        result["if_score"] = None
    return result
