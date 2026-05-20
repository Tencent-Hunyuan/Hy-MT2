# -*- coding: utf-8 -*-
"""
Core scoring module: integrates single-constraint and multi-constraint scoring logic.

Single-constraint scoring:
  - Glossary: rule check first, falls back to LLM Judge if rule fails
  - Style/Background: LLM Judge (0-5 normalized to 0-1)
  - Layout/Structured/Code: rule check (0/1)

Multi-constraint scoring:
  final_score = gate_score × avg(continuous_scores)
  - gate_score = product of all gate scores (any 0 makes the final 0)
  - continuous_scores = average of all continuous scores (defaults to 1.0 if none)
"""

import json
import logging
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

from tqdm import tqdm

from config import (
    CLASS_GLOSSARY, CLASS_LLM_JUDGE, CLASS_LAYOUT, CLASS_STRUCTURED, CLASS_CODE,
    CLASS_TO_DIMENSION, MAX_WORKERS,
)
from eval.rule_validators import (
    validate_glossary, validate_layout, validate_structured, validate_code_assets,
)
from eval.llm_judge import score_glossary_judge, score_style_background_judge

log = logging.getLogger(__name__)


def get_class_list(item: dict) -> list:
    """Extract the class list from an item."""
    cls = item.get("class", [])
    if isinstance(cls, str):
        cls = [cls]
    return cls


def score_single_dimension(item: dict, cls: str, model_response: str) -> dict:
    """
    Score a single constraint dimension.

    Args:
        item: Test data entry (contains input, output, origin_text, meta_data, etc.)
        cls: The constraint class to evaluate
        model_response: The model's translation output

    Returns:
        {"class": str, "score_type": "gate"|"continuous", "score": float|None, "details": dict}
    """
    origin_text = item.get("origin_text", "")
    ground_truth = item.get("output", "")
    user_instruction = item.get("input", "")
    meta_data = item.get("meta_data", {})

    result = {"class": cls, "score_type": None, "score": None, "details": {}}

    if cls in CLASS_GLOSSARY:
        result["score_type"] = "gate"
        term_dict_str = item.get("term_dict", "")
        check = validate_glossary(model_response, term_dict_str, ground_truth)
        if check["valid"]:
            result["score"] = 1.0
            result["details"] = {"method": "rule", "valid": True,
                                 "matched": check["matched"], "total": check["total"]}
        else:
            judge = score_glossary_judge(user_instruction, ground_truth, model_response)
            result["score"] = judge.get("if_score", 0.0)
            result["details"] = {"method": "llm_judge_fallback",
                                 "rule_errors": check["errors"],
                                 "judge_score": judge.get("glossary")}

    elif cls in CLASS_STRUCTURED:
        result["score_type"] = "gate"
        data_format = item.get("data_format", meta_data.get("data_format", ""))
        check = validate_structured(origin_text, model_response, data_format)
        result["score"] = 1.0 if check["valid"] else 0.0
        result["details"] = {"method": "rule", "valid": check["valid"],
                             "data_format": data_format, "errors": check["errors"]}

    elif cls in CLASS_LAYOUT:
        result["score_type"] = "gate"
        check = validate_layout(model_response, meta_data, origin_text)
        result["score"] = 1.0 if check["valid"] else 0.0
        result["details"] = {"method": "rule", "valid": check["valid"],
                             "errors": check["errors"]}

    elif cls in CLASS_CODE:
        result["score_type"] = "gate"
        extracted_assets = meta_data.get("extracted_assets", [])
        check = validate_code_assets(model_response, extracted_assets)
        result["score"] = 1.0 if check["valid"] else 0.0
        result["details"] = {"method": "rule", "valid": check["valid"],
                             "matched": check["matched"], "total": check["total"]}

    elif cls in CLASS_LLM_JUDGE:
        result["score_type"] = "continuous"
        expected_dim = CLASS_TO_DIMENSION.get(cls, "")
        judge = score_style_background_judge(user_instruction, ground_truth, model_response, expected_dim)
        result["score"] = judge.get("if_score")
        result["details"] = {
            "method": "llm_judge",
            "style": judge.get("style"),
            "background": judge.get("background"),
            "classification_match": judge.get("classification_match", False),
            "expected_dim": expected_dim,
        }

    else:
        result["score_type"] = "unknown"
        result["details"] = {"error": f"Unknown class: {cls}"}

    return result


def compose_multi_scores(dim_scores: list) -> dict:
    """
    Compose multi-dimension scores:
    final = gate_score × avg(continuous_scores)
    """
    gate_scores = []
    continuous_scores = []

    for ds in dim_scores:
        if ds["score"] is None:
            continue
        if ds["score_type"] == "gate":
            gate_scores.append(ds["score"])
        elif ds["score_type"] == "continuous":
            continuous_scores.append(ds["score"])

    gate = 1.0
    for g in gate_scores:
        gate *= g

    if continuous_scores:
        continuous_avg = sum(continuous_scores) / len(continuous_scores)
    else:
        continuous_avg = 1.0

    final = round(gate * continuous_avg, 4)

    return {
        "gate_score": round(gate, 4),
        "continuous_avg": round(continuous_avg, 4),
        "final_score": final,
    }


def score_one_item(item: dict, model_response: str) -> dict:
    """
    Score a single data item.

    Args:
        item: Test data entry
        model_response: Model output

    Returns:
        Scoring result dictionary
    """
    cls_list = get_class_list(item)
    is_multi = len(cls_list) > 1

    entry = {
        "md5": item.get("md5", ""),
        "class": cls_list,
        "is_multi_constraint": is_multi,
    }

    if is_multi:
        dim_scores = []
        for cls in cls_list:
            ds = score_single_dimension(item, cls, model_response)
            dim_scores.append(ds)

        composition = compose_multi_scores(dim_scores)
        entry["dimension_scores"] = dim_scores
        entry["gate_score"] = composition["gate_score"]
        entry["continuous_avg"] = composition["continuous_avg"]
        entry["final_score"] = composition["final_score"]
    else:
        cls = cls_list[0] if cls_list else "unknown"
        ds = score_single_dimension(item, cls, model_response)
        entry["dimension_scores"] = [ds]
        entry["final_score"] = ds["score"]

    return entry


def batch_score(test_data: list, responses: dict, workers: int = None) -> list:
    """
    Batch scoring.

    Args:
        test_data: List of test data items
        responses: {md5: response_text} mapping
        workers: Number of concurrent threads

    Returns:
        List of scoring results
    """
    if workers is None:
        workers = MAX_WORKERS

    log.info(f"Starting evaluation: {len(test_data)} items, workers={workers}")

    results = [None] * len(test_data)
    skipped = 0

    def _worker(idx):
        item = test_data[idx]
        md5 = item.get("md5", "")
        response = responses.get(md5, "")
        if not response:
            return idx, {"md5": md5, "class": get_class_list(item),
                        "final_score": None, "error": "Model output not found"}
        return idx, score_one_item(item, response)

    with ThreadPoolExecutor(max_workers=min(workers, max(len(test_data), 1))) as executor:
        futures = {executor.submit(_worker, i): i for i in range(len(test_data))}
        with tqdm(total=len(test_data), desc="Scoring", unit="item") as pbar:
            for future in as_completed(futures):
                idx, entry = future.result()
                results[idx] = entry
                if entry.get("error"):
                    skipped += 1
                pbar.update(1)

    if skipped:
        log.warning(f"Skipped {skipped} items (no matching model output found)")

    return results


def compute_summary(results: list) -> dict:
    """Compute aggregated scoring statistics."""
    single_stats = defaultdict(lambda: {"count": 0, "scores": []})
    multi_stats = defaultdict(lambda: {"count": 0, "scores": []})

    for r in results:
        if r is None or r.get("error"):
            continue
        cls_key = str(r["class"])
        is_multi = r.get("is_multi_constraint", False)

        if is_multi:
            s = multi_stats[cls_key]
        else:
            s = single_stats[cls_key]

        s["count"] += 1
        if r["final_score"] is not None:
            s["scores"].append(r["final_score"])

    # Overall statistics
    all_scores = []
    for s in single_stats.values():
        all_scores.extend(s["scores"])
    for s in multi_stats.values():
        all_scores.extend(s["scores"])

    summary = {
        "total_items": len(results),
        "scored_items": len(all_scores),
        "overall_avg": round(sum(all_scores) / len(all_scores), 4) if all_scores else None,
        "single_constraint": {k: {"count": v["count"],
                                   "avg_score": round(sum(v["scores"]) / len(v["scores"]), 4) if v["scores"] else None}
                              for k, v in single_stats.items()},
        "multi_constraint": {k: {"count": v["count"],
                                  "avg_score": round(sum(v["scores"]) / len(v["scores"]), 4) if v["scores"] else None}
                             for k, v in multi_stats.items()},
    }
    return summary


def print_summary(results: list):
    """Print scoring summary."""
    summary = compute_summary(results)

    print("\n" + "=" * 85)
    print(f"  Evaluation Results Summary")
    print("=" * 85)
    print(f"  Total: {summary['total_items']}  Scored: {summary['scored_items']}  "
          f"Overall Avg: {summary['overall_avg']}")
    print("-" * 85)

    if summary["single_constraint"]:
        print("\n  [Single Constraint]")
        print(f"  {'CLASS':<50} {'COUNT':>6} {'AVG':>10}")
        for k, v in sorted(summary["single_constraint"].items()):
            print(f"  {k:<50} {v['count']:>6} {v['avg_score'] or 'N/A':>10}")

    if summary["multi_constraint"]:
        print("\n  [Multi Constraint] (final = gate × avg_continuous)")
        print(f"  {'CLASS COMBO':<50} {'COUNT':>6} {'AVG':>10}")
        for k, v in sorted(summary["multi_constraint"].items()):
            print(f"  {k:<50} {v['count']:>6} {v['avg_score'] or 'N/A':>10}")

    print("=" * 85)
