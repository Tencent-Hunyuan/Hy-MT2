# -*- coding: utf-8 -*-
"""
Rule-based validators: glossary, layout preservation, structured data, and code snippet matching.

These are deterministic rule checks that do not require an LLM.
"""

import io
import csv
import json
import re
import logging
from html.parser import HTMLParser

log = logging.getLogger(__name__)


# ============================================================
# Glossary Validation
# ============================================================

def validate_glossary(model_response: str, term_dict_str: str, ground_truth: str = "") -> dict:
    """
    Glossary rule check: verifies that model_response contains the correct target terms
    from term_dict. For multi-candidate scenarios, cross-validates with ground_truth.
    """
    result = {"valid": True, "errors": [], "matched": 0, "total": 0}

    if not term_dict_str:
        result["valid"] = False
        result["errors"].append("term_dict is empty, cannot perform rule check")
        return result
    if not model_response:
        result["valid"] = False
        result["errors"].append("model_response is empty")
        return result

    try:
        term_dict = json.loads(term_dict_str) if isinstance(term_dict_str, str) else term_dict_str
    except (json.JSONDecodeError, TypeError):
        result["valid"] = False
        result["errors"].append(f"term_dict parse failed: {str(term_dict_str)[:100]}")
        return result

    if not isinstance(term_dict, dict) or not term_dict:
        result["valid"] = False
        result["errors"].append("term_dict is empty or has invalid format")
        return result

    for src_term, tgt_terms in term_dict.items():
        if not isinstance(tgt_terms, list):
            tgt_terms = [tgt_terms]
        result["total"] += 1
        model_hits = [tgt for tgt in tgt_terms if tgt and tgt in model_response]

        if not model_hits:
            result["valid"] = False
            result["errors"].append(f"Term not matched: {src_term} -> {tgt_terms}")
            continue

        if len(tgt_terms) > 1 and ground_truth:
            gt_hits = [tgt for tgt in tgt_terms if tgt and tgt in ground_truth]
            correct_hit = any(tgt in gt_hits for tgt in model_hits)
            if correct_hit:
                result["matched"] += 1
            else:
                result["valid"] = False
                result["errors"].append(
                    f"Wrong term choice: {src_term} -> model used {model_hits}, "
                    f"but correct term in ground_truth is {gt_hits}"
                )
        else:
            result["matched"] += 1

    return result


# ============================================================
# Layout Preservation Validation
# ============================================================

def validate_layout(model_response: str, meta_data: dict, origin_text: str) -> dict:
    """Layout preservation check: verifies chunk count consistency after delimiter splitting."""
    primary_delimiter = meta_data.get("primary_delimiter", "")
    source_chunks = meta_data.get("source_chunks", [])
    result = {"valid": False, "errors": []}

    if not primary_delimiter:
        result["errors"].append("primary_delimiter is empty")
        return result
    if not origin_text:
        result["errors"].append("origin_text is empty")
        return result
    if not model_response:
        result["errors"].append("model_response is empty")
        return result

    origin_chunks = origin_text.split(primary_delimiter)
    output_chunks = model_response.split(primary_delimiter)
    if len(origin_chunks) == len(output_chunks) == len(source_chunks):
        result["valid"] = True
    else:
        result["errors"].append(
            f"Chunk count mismatch: origin={len(origin_chunks)}, "
            f"response={len(output_chunks)}, source_chunks={len(source_chunks)}"
        )
    return result


# ============================================================
# Structured Data Validation
# ============================================================

class HTMLTagExtractor(HTMLParser):
    def __init__(self):
        super().__init__()
        self.tags = []

    def handle_starttag(self, tag, attrs):
        self.tags.append(("start", tag, sorted([k for k, v in attrs])))

    def handle_endtag(self, tag):
        self.tags.append(("end", tag, []))

    def error(self, message):
        pass


def _validate_json_struct(origin_text, output_text):
    errors = []
    try:
        oj = json.loads(origin_text)
    except Exception:
        errors.append("Origin JSON parse failed")
        oj = None
    try:
        mj = json.loads(output_text)
    except Exception:
        errors.append("Output JSON parse failed")
        mj = None
    if oj is not None and mj is not None:
        errors.extend(_check_json_keys(oj, mj, "$"))
    return len(errors) == 0, errors


def _check_json_keys(origin, output, path="$"):
    errors = []
    if type(origin) != type(output):
        return [f"Type mismatch @ {path}"]
    if isinstance(origin, dict):
        ok, ek = set(origin.keys()), set(output.keys())
        if ok != ek:
            if ok - ek:
                errors.append(f"Missing keys @ {path}: {ok - ek}")
            if ek - ok:
                errors.append(f"Extra keys @ {path}: {ek - ok}")
        for k in ok & ek:
            errors.extend(_check_json_keys(origin[k], output[k], f"{path}.{k}"))
    elif isinstance(origin, list):
        if len(origin) != len(output):
            errors.append(f"Array length mismatch @ {path}")
        for i in range(min(len(origin), len(output))):
            errors.extend(_check_json_keys(origin[i], output[i], f"{path}[{i}]"))
    return errors


def _validate_html_struct(origin_text, output_text):
    errors = []
    op = HTMLTagExtractor()
    try:
        op.feed(origin_text)
    except Exception:
        errors.append("Origin HTML parse failed")
    mp = HTMLTagExtractor()
    try:
        mp.feed(output_text)
    except Exception:
        errors.append("Output HTML parse failed")
    if not errors:
        if len(op.tags) != len(mp.tags):
            errors.append("Tag count mismatch")
        else:
            for i, (ot, mt) in enumerate(zip(op.tags, mp.tags)):
                if ot[0] != mt[0] or ot[1] != mt[1]:
                    errors.append(f"Tag #{i+1} mismatch")
    return len(errors) == 0, errors


def _validate_csv_struct(origin_text, output_text):
    errors = []
    try:
        or_ = list(csv.reader(io.StringIO(origin_text)))
    except Exception:
        errors.append("Origin CSV parse failed")
        or_ = None
    try:
        mr_ = list(csv.reader(io.StringIO(output_text)))
    except Exception:
        errors.append("Output CSV parse failed")
        mr_ = None
    if or_ is not None and mr_ is not None:
        if len(or_) != len(mr_):
            errors.append("Row count mismatch")
        else:
            for i, (a, b) in enumerate(zip(or_, mr_)):
                if len(a) != len(b):
                    errors.append(f"Column count mismatch at row {i+1}")
    return len(errors) == 0, errors


def _parse_md_table(text):
    lines = [l.strip() for l in text.strip().split("\n") if l.strip()]
    if len(lines) < 2:
        return None, None, "Insufficient table rows"

    def split_row(line):
        line = line.strip().strip("|")
        return [c.strip() for c in line.split("|")]

    header = split_row(lines[0])
    if not re.match(r"^[\|\s\-:]+$", lines[1]):
        return None, None, "Row 2 is not a valid separator row"
    return header, [split_row(l) for l in lines[2:]], None


def _validate_markdown_struct(origin_text, output_text):
    errors = []
    oh, orows, oerr = _parse_md_table(origin_text)
    if oerr:
        errors.append(f"origin: {oerr}")
    mh, mrows, merr = _parse_md_table(output_text)
    if merr:
        errors.append(f"output: {merr}")
    if oh and mh:
        if len(oh) != len(mh):
            errors.append("Header column count mismatch")
        if orows and mrows and len(orows) != len(mrows):
            errors.append("Data row count mismatch")
    return len(errors) == 0, errors


STRUCT_VALIDATORS = {
    "JSON": _validate_json_struct, "json": _validate_json_struct,
    "HTML片段": _validate_html_struct, "HTML": _validate_html_struct, "html": _validate_html_struct,
    "CSV": _validate_csv_struct, "csv": _validate_csv_struct,
    "Markdown表格": _validate_markdown_struct, "Markdown": _validate_markdown_struct,
    "markdown": _validate_markdown_struct,
}


def validate_structured(origin_text: str, model_response: str, data_format: str) -> dict:
    """Structured data validation: checks if translation preserves the original data structure."""
    result = {"valid": False, "errors": []}
    if not origin_text or not model_response:
        result["errors"].append("origin_text or model_response is empty")
        return result
    validator = STRUCT_VALIDATORS.get(data_format)
    if not validator:
        result["errors"].append(f"Unknown data_format: {data_format}")
        return result
    valid, errors = validator(origin_text, model_response)
    result["valid"] = valid
    result["errors"] = errors
    return result


# ============================================================
# Code Snippet Matching
# ============================================================

def validate_code_assets(model_response: str, extracted_assets: list) -> dict:
    """Code/tag preservation check: verifies model_response contains all extracted_assets."""
    result = {"valid": True, "errors": [], "matched": 0, "total": len(extracted_assets)}
    if not extracted_assets:
        return result
    if not model_response:
        result["valid"] = False
        result["errors"].append("model_response is empty")
        return result
    for asset in extracted_assets:
        if asset in model_response:
            result["matched"] += 1
        else:
            result["valid"] = False
            result["errors"].append(f"Not matched: {asset}")
    return result
