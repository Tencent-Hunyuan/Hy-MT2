# -*- coding: utf-8 -*-
"""
Translation Instruction Following Evaluation entry point.

Usage:
    python run_eval.py \
        --input_data data/test_single_constraint.jsonl \
        --input_response data/sample_output.jsonl \
        --output_dir eval_results

    # Evaluate both single and multi-constraint
    python run_eval.py \
        --input_data data/test_single_constraint.jsonl data/test_multi_constraint.jsonl \
        --input_response my_model_output.jsonl \
        --output_dir eval_results

    # Rule-only mode (skip LLM Judge, for debugging)
    python run_eval.py \
        --input_data data/test_single_constraint.jsonl \
        --input_response my_model_output.jsonl \
        --output_dir eval_results \
        --skip-llm
"""

import argparse
import json
import os
import sys
import logging

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dotenv import load_dotenv
load_dotenv()  # Load environment variables from .env file

from eval.scoring import batch_score, print_summary, compute_summary

logging.basicConfig(
    format="%(asctime)s : %(levelname)s : %(filename)s: %(message)s",
    level=logging.INFO,
)
log = logging.getLogger(__name__)


def load_jsonl(path: str) -> list:
    """Load a JSONL file."""
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def load_responses(path: str) -> dict:
    """
    Load model output file, returning a {(md5, instruction_lang): response} mapping.
    Also builds a fallback {md5: response} mapping for entries without instruction_lang.

    Model output file format: one JSON object per line, must contain:
      - md5: unique identifier matching the md5 field in test data
      - response: the model's generated translation
      - instruction_lang (optional): the instruction language used for this response
    """
    responses = {}
    fallback = {}
    data = load_jsonl(path)
    for item in data:
        md5 = item.get("md5", "")
        response = item.get("response", "")
        lang = item.get("instruction_lang", "")
        if md5:
            if lang:
                responses[(md5, lang)] = response
            fallback[md5] = response
    log.info(f"Loaded {len(responses)} lang-specific responses, {len(fallback)} unique md5 entries")
    return {"by_lang": responses, "fallback": fallback}


def main():
    parser = argparse.ArgumentParser(
        description="Translation Instruction Following Evaluation"
    )
    parser.add_argument(
        "--input_data", "-d", nargs="+", required=True,
        help="Test data file path(s) (JSONL), multiple files supported"
    )
    parser.add_argument(
        "--input_response", "-r", required=True,
        help="Model output file path (JSONL), each line must contain md5 and response fields"
    )
    parser.add_argument(
        "--output_dir", "-o", default="eval_results",
        help="Output directory for evaluation results (default: eval_results)"
    )
    parser.add_argument(
        "--workers", "-w", type=int, default=None,
        help="Number of concurrent LLM Judge threads (default: EVAL_MAX_WORKERS env var or 20)"
    )
    parser.add_argument(
        "--skip-llm", action="store_true",
        help="Skip LLM Judge evaluation (rule-only mode, for debugging)"
    )
    args = parser.parse_args()

    # Load test data
    test_data = []
    for path in args.input_data:
        items = load_jsonl(path)
        log.info(f"Loaded test data: {path} ({len(items)} items)")
        test_data.extend(items)
    log.info(f"Total: {len(test_data)} test items")

    # Load model responses
    response_data = load_responses(args.input_response)
    by_lang = response_data["by_lang"]
    fallback = response_data["fallback"]

    # Build a flat {(md5, lang): response} lookup for batch_score
    # For each test item, try (md5, instruction_lang) first, then fallback to md5-only
    responses_flat = {}
    matched = 0
    for item in test_data:
        md5 = item.get("md5", "")
        lang = item.get("instruction_lang", "")
        key = (md5, lang)
        if key in by_lang:
            responses_flat[key] = by_lang[key]
            matched += 1
        elif md5 in fallback:
            responses_flat[key] = fallback[md5]
            matched += 1

    coverage = matched / len(test_data) if test_data else 0
    log.info(f"Response coverage: {coverage:.1%} ({matched}/{len(test_data)})")

    if coverage < 0.5:
        log.warning("Response coverage is below 50%, please check if md5/instruction_lang fields match")

    # If skipping LLM, temporarily modify config
    if args.skip_llm:
        log.info("Skipping LLM Judge (rule-only mode)")
        # Remap LLM Judge classes to empty set so no LLM calls are made
        import config
        config.CLASS_LLM_JUDGE = set()

    # Run scoring
    results = batch_score(test_data, responses_flat, workers=args.workers)

    # Output results
    os.makedirs(args.output_dir, exist_ok=True)

    # Write detailed results
    detail_path = os.path.join(args.output_dir, "eval_details.jsonl")
    with open(detail_path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    log.info(f"Detailed results written to: {detail_path}")

    # Write summary
    summary = compute_summary(results)
    summary_path = os.path.join(args.output_dir, "eval_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    log.info(f"Summary written to: {summary_path}")

    # Print summary
    print_summary(results)


if __name__ == "__main__":
    main()
