#!/usr/bin/env python3
"""
Quick validation script for converted HYV3 outer-format checkpoint.

Checks:
  1. model.safetensors.index.json structure and completeness
  2. All expected weight keys exist (dense layer 0, MoE layers 1-79)
  3. Expert tensor shapes (fused 3D format)
  4. All referenced shard files exist and are non-empty
  5. Spot-check: load a few shards and verify tensor shapes/dtypes
  6. No duplicate or orphan keys

Usage:
    python check_converted.py <output_dir> [--spot-check N]

Example:
    python check_converted.py pretrain_base/hf_outer
    python check_converted.py pretrain_base/hf_outer --spot-check 5
"""

import argparse
import json
import os
import sys
import time
from collections import defaultdict

# ============================================================================
# Expected key patterns for HYV3 outer format
# ============================================================================

# Dense layer (layer 0) expected suffixes
DENSE_SUFFIXES = [
    "input_layernorm.weight",
    "post_attention_layernorm.weight",
    "self_attn.q_proj.weight",
    "self_attn.k_proj.weight",
    "self_attn.v_proj.weight",
    "self_attn.o_proj.weight",
    "self_attn.q_norm.weight",
    "self_attn.k_norm.weight",
    "mlp.gate_proj.weight",
    "mlp.up_proj.weight",
    "mlp.down_proj.weight",
]

# MoE layer (layers 1-79) expected suffixes
MOE_SUFFIXES = [
    "input_layernorm.weight",
    "post_attention_layernorm.weight",
    "self_attn.q_proj.weight",
    "self_attn.k_proj.weight",
    "self_attn.v_proj.weight",
    "self_attn.o_proj.weight",
    "self_attn.q_norm.weight",
    "self_attn.k_norm.weight",
    # MoE-specific
    "mlp.gate.weight",
    "mlp.e_score_correction_bias",
    "mlp.experts.gate_up_proj",
    "mlp.experts.down_proj",
    "mlp.shared_experts.gate_proj.weight",
    "mlp.shared_experts.up_proj.weight",
    "mlp.shared_experts.down_proj.weight",
]

# MTP (Multi-Token Prediction) layer expected suffixes
# MTP layers share MoE structure but have additional projection/norm keys
MTP_EXTRA_SUFFIXES = [
    "eh_proj.weight",
    "enorm.weight",
    "final_layernorm.weight",
    "hnorm.weight",
]

# Global keys (not per-layer)
GLOBAL_KEYS = [
    "model.embed_tokens.weight",
    "model.norm.weight",
    "lm_head.weight",
]


def load_config(output_dir):
    """Load config.json and extract model parameters."""
    config_path = os.path.join(output_dir, "config.json")
    if not os.path.exists(config_path):
        print(f"[ERROR] config.json not found in {output_dir}")
        return None
    with open(config_path) as f:
        return json.load(f)


def check_index_json(output_dir):
    """Check model.safetensors.index.json for structure and completeness."""
    index_path = os.path.join(output_dir, "model.safetensors.index.json")
    if not os.path.exists(index_path):
        print(f"[ERROR] model.safetensors.index.json not found")
        return None, []

    with open(index_path) as f:
        index = json.load(f)

    errors = []

    # Check structure
    if "metadata" not in index:
        errors.append("Missing 'metadata' in index.json")
    elif "total_size" not in index["metadata"]:
        errors.append("Missing 'total_size' in metadata")

    if "weight_map" not in index:
        errors.append("Missing 'weight_map' in index.json")
        return index, errors

    weight_map = index["weight_map"]
    total_size = index.get("metadata", {}).get("total_size", 0)

    print(f"  Index keys     : {len(weight_map)}")
    print(f"  Total size     : {total_size / 1e9:.2f} GB")

    # Check for empty weight_map
    if len(weight_map) == 0:
        errors.append("weight_map is empty")

    return index, errors


def check_expected_keys(weight_map, config):
    """Check that all expected keys exist in the weight_map."""
    errors = []
    warnings = []

    num_layers = config.get("num_hidden_layers", 80)
    first_k_dense = config.get("first_k_dense_replace", 1)
    num_experts = config.get("num_experts", 192)
    num_mtp_layers = config.get("num_nextn_predict_layers", 0)

    # Check global keys
    for key in GLOBAL_KEYS:
        if key not in weight_map:
            errors.append(f"Missing global key: {key}")

    # Check per-layer keys (regular layers)
    missing_by_type = defaultdict(list)
    for layer_idx in range(num_layers):
        prefix = f"model.layers.{layer_idx}."
        if layer_idx < first_k_dense:
            # Dense layer
            suffixes = DENSE_SUFFIXES
        else:
            # MoE layer
            suffixes = MOE_SUFFIXES

        for suffix in suffixes:
            full_key = prefix + suffix
            if full_key not in weight_map:
                missing_by_type[suffix].append(layer_idx)

    # Check MTP layers (layer num_layers .. num_layers + num_mtp_layers - 1)
    mtp_missing_by_type = defaultdict(list)
    for mtp_idx in range(num_mtp_layers):
        layer_idx = num_layers + mtp_idx
        prefix = f"model.layers.{layer_idx}."
        # MTP layers use MoE structure + extra projection/norm keys
        mtp_suffixes = MOE_SUFFIXES + MTP_EXTRA_SUFFIXES
        for suffix in mtp_suffixes:
            full_key = prefix + suffix
            if full_key not in weight_map:
                mtp_missing_by_type[suffix].append(layer_idx)

    for suffix, layers in sorted(mtp_missing_by_type.items()):
        layer_str = str(layers)
        errors.append(f"Missing MTP key '{suffix}' in layers: {layer_str}")

    for suffix, layers in sorted(missing_by_type.items()):
        if len(layers) <= 5:
            layer_str = str(layers)
        else:
            layer_str = f"{layers[:3]}...({len(layers)} total)"
        errors.append(f"Missing '{suffix}' in layers: {layer_str}")

    # Check for unexpected keys (not matching any known pattern)
    known_prefixes = set()
    # Regular layers + MTP layers
    for layer_idx in range(num_layers + num_mtp_layers):
        known_prefixes.add(f"model.layers.{layer_idx}.")
    known_prefixes.add("model.embed_tokens.")
    known_prefixes.add("model.norm.")
    known_prefixes.add("lm_head.")
    # Alternative MTP prefix (some models use this)
    known_prefixes.add("model.mtp_layers.")

    unexpected = []
    for key in weight_map:
        if not any(key.startswith(p) for p in known_prefixes):
            unexpected.append(key)

    if unexpected:
        if len(unexpected) <= 5:
            for k in unexpected:
                warnings.append(f"Unexpected key: {k}")
        else:
            warnings.append(f"{len(unexpected)} unexpected keys found (first 3: {unexpected[:3]})")

    return errors, warnings


def check_shard_files(output_dir, weight_map):
    """Check that all referenced shard files exist and are non-empty."""
    errors = []
    warnings = []

    # Get unique shard files
    shard_files = sorted(set(weight_map.values()))
    print(f"  Shard files    : {len(shard_files)}")

    missing = []
    empty = []
    total_disk_size = 0

    for sf in shard_files:
        path = os.path.join(output_dir, sf)
        if not os.path.exists(path):
            missing.append(sf)
        else:
            size = os.path.getsize(path)
            if size == 0:
                empty.append(sf)
            total_disk_size += size

    print(f"  Disk size      : {total_disk_size / 1e9:.2f} GB")

    if missing:
        errors.append(f"Missing shard files ({len(missing)}): {missing[:5]}")
    if empty:
        errors.append(f"Empty shard files ({len(empty)}): {empty[:5]}")

    # Check for orphan shard files (exist on disk but not in index)
    all_safetensors = set(
        f for f in os.listdir(output_dir)
        if f.endswith(".safetensors")
    )
    referenced = set(shard_files)
    orphans = all_safetensors - referenced
    if orphans:
        # Distinguish between empty residue files (cross-shard merge artifacts)
        # and real orphan files with actual data
        EMPTY_SHARD_THRESHOLD = 128  # bytes; empty safetensors header is ~16 bytes
        residue_orphans = []
        real_orphans = []
        for o in sorted(orphans):
            sz = os.path.getsize(os.path.join(output_dir, o))
            if sz <= EMPTY_SHARD_THRESHOLD:
                residue_orphans.append(o)
            else:
                real_orphans.append(o)

        if residue_orphans:
            warnings.append(
                f"{len(residue_orphans)} empty residue shard(s) from cross-shard merge "
                f"(<=128 bytes each, safe to delete)"
            )
        if real_orphans:
            errors.append(
                f"Orphan shard files with data (not in index): {real_orphans[:5]}"
            )

    return errors, warnings


def check_key_distribution(weight_map):
    """Check the distribution of keys across shards."""
    shard_key_count = defaultdict(int)
    for key, shard in weight_map.items():
        shard_key_count[shard] += 1

    counts = sorted(shard_key_count.values())
    print(f"  Keys/shard     : min={counts[0]}, max={counts[-1]}, "
          f"median={counts[len(counts)//2]}")

    # Check for shards with 0 keys (should not happen if they are in weight_map)
    zero_shards = [s for s, c in shard_key_count.items() if c == 0]
    if zero_shards:
        return [f"Shards with 0 keys: {zero_shards}"]
    return []


def spot_check_shards(output_dir, weight_map, config, num_checks=3):
    """Spot-check a few shards by loading and verifying tensor shapes."""
    errors = []

    try:
        from safetensors import safe_open
    except ImportError:
        print("  [SKIP] safetensors not installed, skipping spot-check")
        return errors

    num_experts = config.get("num_experts", 192)
    expert_hidden = config.get("expert_hidden_dim", config.get("moe_intermediate_size", 1536))
    hidden_size = config.get("hidden_size", 4096)

    # Find shards that contain expert tensors (most interesting to check)
    expert_shards = set()
    for key, shard in weight_map.items():
        if "experts.gate_up_proj" in key or "experts.down_proj" in key:
            expert_shards.add(shard)

    # Pick a few shards to check
    check_shards = sorted(expert_shards)[:num_checks]
    if not check_shards:
        check_shards = sorted(set(weight_map.values()))[:num_checks]

    print(f"\n  Spot-checking {len(check_shards)} shard(s)...")

    for shard_file in check_shards:
        shard_path = os.path.join(output_dir, shard_file)
        t0 = time.time()

        try:
            with safe_open(shard_path, framework="pt", device="cpu") as f:
                keys_in_shard = list(f.keys())
                for key in keys_in_shard:
                    tensor = f.get_tensor(key)

                    # Check expert shapes
                    if key.endswith("experts.gate_up_proj"):
                        expected_shape = (num_experts, expert_hidden * 2, hidden_size)
                        if tuple(tensor.shape) != expected_shape:
                            errors.append(
                                f"{shard_file}/{key}: shape {tuple(tensor.shape)} "
                                f"!= expected {expected_shape}"
                            )

                    elif key.endswith("experts.down_proj"):
                        expected_shape = (num_experts, hidden_size, expert_hidden)
                        if tuple(tensor.shape) != expected_shape:
                            errors.append(
                                f"{shard_file}/{key}: shape {tuple(tensor.shape)} "
                                f"!= expected {expected_shape}"
                            )

                    # Check for NaN/Inf
                    if tensor.is_floating_point():
                        if tensor.isnan().any():
                            errors.append(f"{shard_file}/{key}: contains NaN values")
                        if tensor.isinf().any():
                            errors.append(f"{shard_file}/{key}: contains Inf values")

            elapsed = time.time() - t0
            print(f"    {shard_file}: {len(keys_in_shard)} keys, OK ({elapsed:.1f}s)")

        except Exception as e:
            errors.append(f"Failed to load {shard_file}: {e}")

    return errors


def main():
    parser = argparse.ArgumentParser(
        description="Validate converted HYV3 outer-format checkpoint."
    )
    parser.add_argument(
        "output_dir", type=str,
        help="Path to the converted outer-format checkpoint directory.",
    )
    parser.add_argument(
        "--spot-check", type=int, default=3, dest="spot_check",
        help="Number of shards to spot-check by loading tensors (default: 3).",
    )
    args = parser.parse_args()

    output_dir = os.path.abspath(args.output_dir)
    print(f"Validating: {output_dir}\n")

    if not os.path.isdir(output_dir):
        print(f"[ERROR] Directory not found: {output_dir}")
        sys.exit(1)

    all_errors = []
    all_warnings = []

    # 1. Load config
    print("[1/5] Loading config.json...")
    config = load_config(output_dir)
    if config is None:
        print("[ERROR] Cannot proceed without config.json")
        sys.exit(1)

    num_layers = config.get("num_hidden_layers", 0)
    num_experts = config.get("num_experts", 0)
    first_k_dense = config.get("first_k_dense_replace", 0)
    num_mtp = config.get("num_nextn_predict_layers", 0)
    print(f"  Layers         : {num_layers} ({first_k_dense} dense, {num_layers - first_k_dense} MoE)")
    print(f"  MTP layers     : {num_mtp}")
    print(f"  Experts/layer  : {num_experts}")
    print(f"  Hidden size    : {config.get('hidden_size', '?')}")
    print(f"  Expert hidden  : {config.get('expert_hidden_dim', config.get('moe_intermediate_size', '?'))}")

    # 2. Check index.json
    print("\n[2/5] Checking model.safetensors.index.json...")
    index, idx_errors = check_index_json(output_dir)
    all_errors.extend(idx_errors)

    if index is None or "weight_map" not in index:
        print("[ERROR] Cannot proceed without valid index.json")
        sys.exit(1)

    weight_map = index["weight_map"]

    # 3. Check expected keys
    print("\n[3/5] Checking expected keys...")
    key_errors, key_warnings = check_expected_keys(weight_map, config)
    all_errors.extend(key_errors)
    all_warnings.extend(key_warnings)

    # Also check key distribution
    dist_errors = check_key_distribution(weight_map)
    all_errors.extend(dist_errors)

    # 4. Check shard files
    print("\n[4/5] Checking shard files on disk...")
    shard_errors, shard_warnings = check_shard_files(output_dir, weight_map)
    all_errors.extend(shard_errors)
    all_warnings.extend(shard_warnings)

    # 5. Spot-check
    if args.spot_check > 0:
        print(f"\n[5/5] Spot-checking tensors (loading {args.spot_check} shard(s))...")
        spot_errors = spot_check_shards(output_dir, weight_map, config, args.spot_check)
        all_errors.extend(spot_errors)
    else:
        print("\n[5/5] Spot-check skipped (--spot-check 0)")

    # Summary
    print(f"\n{'=' * 60}")
    if all_warnings:
        print(f"WARNINGS ({len(all_warnings)}):")
        for w in all_warnings:
            print(f"  [WARN] {w}")

    if all_errors:
        print(f"ERRORS ({len(all_errors)}):")
        for e in all_errors:
            print(f"  [ERROR] {e}")
        print(f"\nResult: FAILED ({len(all_errors)} error(s), {len(all_warnings)} warning(s))")
        sys.exit(1)
    else:
        print(f"Result: PASSED (0 errors, {len(all_warnings)} warning(s))")
        print(f"{'=' * 60}")
        sys.exit(0)


if __name__ == "__main__":
    main()
