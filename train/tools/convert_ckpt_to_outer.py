#!/usr/bin/env python3
"""
Memory-friendly checkpoint converter: inner -> outer format (v2).

Converts the HYV3 checkpoint from inner format (per-expert keys, old naming)
to outer format (fused 3D experts, new naming) shard by shard.

Handles the case where a single layer's experts may be split across
multiple shards (cross-shard experts) by deferring their fusion to a
post-processing step.

v2 improvements over v1:
  - Post-processing is shard-centric (each shard read/written only once)
    instead of prefix-centric (same shard read/written multiple times).
    This fixes Bus error (core dump) when there are many cross-shard groups.
  - Explicit memory management with gc.collect() to prevent memory bloat.
  - Better progress reporting during post-processing.

Supports multi-process parallelism for faster conversion.

Usage:
    # Default 8 workers
    python convert_ckpt_to_outer.py \\
        --input_dir pretrain_base/hf \\
        --output_dir pretrain_base/hf_outer

    # Custom worker count
    python convert_ckpt_to_outer.py \\
        --input_dir pretrain_base/hf \\
        --output_dir pretrain_base/hf_outer \\
        --workers 16

The script will:
  1. Pre-scan index.json to detect cross-shard expert groups
  2. Convert weights shard-by-shard in parallel (key rename + expert fuse)
  3. Post-process cross-shard expert groups (merge from multiple shards)
     - v2: shard-centric approach, each shard read/written only once
  4. Copy config.json as-is (already in outer format)
  5. Copy all other files (tokenizer, etc.)
  6. Rebuild model.safetensors.index.json
"""

import argparse
import gc
import json
import os
import re
import signal
import shutil
import sys
import time
import traceback
from collections import OrderedDict, defaultdict
from multiprocessing import Pool

import torch

try:
    from safetensors import safe_open
    from safetensors.torch import save_file
except ImportError:
    raise ImportError("Please install safetensors: pip install safetensors")

# ============================================================================
# Signal handling for Bus error (SIGBUS) and other fatal signals
# ============================================================================

def _fatal_signal_handler(signum, frame):
    """Handle fatal signals (SIGBUS, SIGSEGV) by logging before exit.

    These signals cannot be caught by try/except. This handler ensures
    the error message is written to stderr (captured by nohup redirection)
    before the process terminates.
    """
    sig_name = signal.Signals(signum).name if hasattr(signal, 'Signals') else str(signum)
    pid = os.getpid()
    msg = (
        f"\n[FATAL] Process {pid} received {sig_name} (signal {signum}).\n"
        f"This typically indicates an out-of-memory condition during mmap I/O.\n"
        f"Stack trace at time of signal:\n"
    )
    sys.stderr.write(msg)
    traceback.print_stack(frame, file=sys.stderr)
    sys.stderr.flush()
    # Re-raise with default handler to get proper exit code
    signal.signal(signum, signal.SIG_DFL)
    os.kill(pid, signum)


def _install_signal_handlers():
    """Install handlers for SIGBUS and SIGSEGV in the current process."""
    for sig in (signal.SIGBUS, signal.SIGSEGV):
        try:
            signal.signal(sig, _fatal_signal_handler)
        except (OSError, ValueError):
            # Some signals may not be available on all platforms
            pass


def _pool_worker_init():
    """Initializer for multiprocessing pool workers.

    Installs signal handlers so that Bus errors in worker processes
    are also logged before the process dies.
    """
    _install_signal_handlers()


# ============================================================================
# Key rename mapping (inner -> outer)
# ============================================================================

_KEY_RENAMES = [
    ("mlp.router.gate.", "mlp.gate."),
    ("mlp.expert_bias", "mlp.e_score_correction_bias"),
    ("mlp.shared_mlp.", "mlp.shared_experts."),
]

# Regex to match per-expert keys
_EXPERT_KEY_RE = re.compile(
    r"^(.*\.mlp\.experts\.)(\d+)\.(gate_proj|up_proj|down_proj)\.weight$"
)

def rename_key(key: str) -> str:
    """Rename a single key from inner to outer format."""
    for old_sub, new_sub in _KEY_RENAMES:
        if old_sub in key:
            key = key.replace(old_sub, new_sub)
            break
    return key

def scan_cross_shard_experts(index_path: str):
    """Pre-scan index.json to find expert groups that span multiple shards.

    Returns:
        cross_shard_prefixes: set of expert prefixes that span multiple shards
            e.g. {"model.layers.80.mlp.experts."}
    """
    with open(index_path) as f:
        index = json.load(f)
    wm = index["weight_map"]

    # prefix -> set of shards
    prefix_shards = defaultdict(set)
    for key in wm:
        m = _EXPERT_KEY_RE.match(key)
        if m:
            prefix = m.group(1)
            prefix_shards[prefix].add(wm[key])

    cross_shard_prefixes = set()
    for prefix, shards in prefix_shards.items():
        if len(shards) > 1:
            cross_shard_prefixes.add(prefix)

    return cross_shard_prefixes

def convert_shard(shard_path: str, cross_shard_prefixes: set = None):
    """Load a single shard, rename keys, and fuse experts.

    For expert groups in cross_shard_prefixes, the per-expert keys are
    kept as-is (just renamed) and returned separately as deferred items,
    to be merged later in a post-processing step.

    Returns:
        result: OrderedDict of converted tensors (ready to save)
        deferred_expert_keys: list of original expert keys that were deferred
            (these are kept in result with their original per-expert naming
             but with the outer rename applied, to be post-processed later)
    """
    if cross_shard_prefixes is None:
        cross_shard_prefixes = set()

    tensors = OrderedDict()
    with safe_open(shard_path, framework="pt", device="cpu") as f:
        for key in f.keys():
            tensors[key] = f.get_tensor(key)

    # Separate expert keys from non-expert keys
    expert_groups = {}  # prefix -> {expert_idx -> {proj_name -> tensor}}
    deferred_expert_keys = []  # keys that belong to cross-shard experts
    result = OrderedDict()

    for key, tensor in tensors.items():
        m = _EXPERT_KEY_RE.match(key)
        if m:
            prefix = m.group(1)
            expert_idx = int(m.group(2))
            proj_name = m.group(3)

            if prefix in cross_shard_prefixes:
                # Defer: keep the key as-is (with rename) for post-processing
                new_key = rename_key(key)
                result[new_key] = tensor
                deferred_expert_keys.append(new_key)
            else:
                # Normal: collect for fusion within this shard
                if prefix not in expert_groups:
                    expert_groups[prefix] = {}
                if expert_idx not in expert_groups[prefix]:
                    expert_groups[prefix][expert_idx] = {}
                expert_groups[prefix][expert_idx][proj_name] = tensor
        else:
            # Non-expert key: just rename
            new_key = rename_key(key)
            result[new_key] = tensor

    # Fuse expert weights for each non-cross-shard layer prefix
    for prefix in sorted(expert_groups.keys()):
        experts = expert_groups[prefix]
        num_experts = max(experts.keys()) + 1

        gate_up_list = []
        down_list = []
        for i in range(num_experts):
            if i not in experts:
                raise ValueError(
                    f"Missing expert {i} in {prefix}. "
                    f"Found: {sorted(experts.keys())}"
                )
            exp = experts[i]
            gate_up = torch.cat([exp["gate_proj"], exp["up_proj"]], dim=0)
            gate_up_list.append(gate_up)
            down_list.append(exp["down_proj"])

        fused_gate_up = torch.stack(gate_up_list, dim=0)
        fused_down = torch.stack(down_list, dim=0)

        for exp in experts.values():
            exp.clear()
        gate_up_list.clear()
        down_list.clear()

        result[f"{prefix}gate_up_proj"] = fused_gate_up
        result[f"{prefix}down_proj"] = fused_down

    return result, deferred_expert_keys

def _process_one_shard(args_tuple):
    """Worker function: convert a single shard and save to output dir.

    Args:
        args_tuple: (idx, num_shards, shard_file, input_dir, output_dir, cross_shard_prefixes)

    Returns:
        (shard_file, key_list, shard_size, elapsed, deferred_keys)
    """
    idx, num_shards, shard_file, input_dir, output_dir, cross_shard_prefixes = args_tuple
    shard_path = os.path.join(input_dir, shard_file)
    t0 = time.time()

    converted, deferred_keys = convert_shard(shard_path, cross_shard_prefixes)

    shard_size = sum(t.numel() * t.element_size() for t in converted.values())

    out_shard_path = os.path.join(output_dir, shard_file)
    save_file(converted, out_shard_path)

    elapsed = time.time() - t0
    num_keys = len(converted)
    key_list = list(converted.keys())

    del converted

    deferred_info = ""
    if deferred_keys:
        deferred_info = f", Deferred={len(deferred_keys)}"

    print(
        f"  [{idx + 1}/{num_shards}] {shard_file}: "
        f"Keys={num_keys}, Size={shard_size / 1e9:.2f} GB, "
        f"Time={elapsed:.1f}s{deferred_info}",
        flush=True,
    )

    return shard_file, key_list, shard_size, elapsed, deferred_keys


def post_process_cross_shard_experts(output_dir, cross_shard_prefixes, all_deferred):
    """Merge cross-shard expert groups (v2: shard-centric approach).

    Instead of iterating per-prefix (which causes the same shard to be
    loaded/saved multiple times), this v2 approach:
      1. Builds a mapping of which prefixes each shard is involved in
      2. Collects all expert tensors from all involved shards in ONE pass
      3. Fuses all prefixes
      4. Writes each shard only ONCE with all its updates applied

    This avoids the Bus error (core dump) caused by repeated mmap of
    large files and memory bloat.

    Args:
        output_dir: path to output directory
        cross_shard_prefixes: set of expert prefixes that span multiple shards
        all_deferred: dict of {shard_file: [deferred_key, ...]}

    Returns:
        updated_shards: dict of {shard_file: (key_list, shard_size)} for updated shards
    """
    if not cross_shard_prefixes:
        return {}

    print(f"\n  Post-processing {len(cross_shard_prefixes)} cross-shard expert group(s)...",
          flush=True)

    # ----------------------------------------------------------------
    # Step 1: Build mappings
    # ----------------------------------------------------------------
    # prefix -> ordered list of shards that contain its experts
    prefix_to_shards = defaultdict(set)
    # shard -> set of prefixes it is involved in
    shard_to_prefixes = defaultdict(set)

    for shard_file, deferred_keys in all_deferred.items():
        for key in deferred_keys:
            m = _EXPERT_KEY_RE.match(key)
            if m:
                prefix = m.group(1)
                if prefix in cross_shard_prefixes:
                    prefix_to_shards[prefix].add(shard_file)
                    shard_to_prefixes[shard_file].add(prefix)

    # For each prefix, decide which shard will hold the fused result
    # (use the first shard alphabetically)
    prefix_to_target_shard = {}
    for prefix in sorted(prefix_to_shards.keys()):
        target = sorted(prefix_to_shards[prefix])[0]
        prefix_to_target_shard[prefix] = target

    # All shards that need to be updated
    all_involved_shards = set()
    for shards in prefix_to_shards.values():
        all_involved_shards.update(shards)

    print(f"    Involved shards: {len(all_involved_shards)}", flush=True)
    print(f"    Expert groups: {len(prefix_to_shards)}", flush=True)

    # ----------------------------------------------------------------
    # Step 2: Collect all expert tensors from all involved shards
    #         (one pass per shard)
    # ----------------------------------------------------------------
    # prefix -> {expert_idx -> {proj_name -> tensor}}
    all_expert_data = defaultdict(dict)
    # shard -> OrderedDict of non-expert keys (to be re-saved)
    shard_non_expert = {}

    sorted_involved = sorted(all_involved_shards)
    for si, shard_file in enumerate(sorted_involved):
        shard_path = os.path.join(output_dir, shard_file)
        prefixes_in_shard = shard_to_prefixes[shard_file]

        print(f"    [{si+1}/{len(sorted_involved)}] Reading {shard_file} "
              f"({len(prefixes_in_shard)} prefix(es))...", flush=True)

        non_expert = OrderedDict()
        with safe_open(shard_path, framework="pt", device="cpu") as f:
            for key in f.keys():
                m = _EXPERT_KEY_RE.match(key)
                if m and m.group(1) in prefixes_in_shard:
                    # This is a deferred expert key
                    prefix = m.group(1)
                    expert_idx = int(m.group(2))
                    proj_name = m.group(3)
                    if expert_idx not in all_expert_data[prefix]:
                        all_expert_data[prefix][expert_idx] = {}
                    all_expert_data[prefix][expert_idx][proj_name] = f.get_tensor(key)
                else:
                    # Non-expert key: keep as-is
                    non_expert[key] = f.get_tensor(key)

        shard_non_expert[shard_file] = non_expert
        gc.collect()

    # ----------------------------------------------------------------
    # Step 3: Fuse all expert groups
    # ----------------------------------------------------------------
    # prefix -> {"gate_up_proj": tensor, "down_proj": tensor}
    fused_results = {}

    for pi, prefix in enumerate(sorted(all_expert_data.keys())):
        expert_data = all_expert_data[prefix]
        num_experts = max(expert_data.keys()) + 1

        print(f"    Fusing {prefix} ({num_experts} experts)...", flush=True)

        gate_up_list = []
        down_list = []
        for i in range(num_experts):
            if i not in expert_data:
                raise ValueError(
                    f"Missing expert {i} in {prefix} after cross-shard merge. "
                    f"Found: {sorted(expert_data.keys())}"
                )
            exp = expert_data[i]
            if "gate_proj" not in exp or "up_proj" not in exp:
                raise ValueError(
                    f"Expert {i} in {prefix} missing gate_proj/up_proj. "
                    f"Has: {sorted(exp.keys())}"
                )
            if "down_proj" not in exp:
                raise ValueError(
                    f"Expert {i} in {prefix} missing down_proj. "
                    f"Has: {sorted(exp.keys())}"
                )
            gate_up = torch.cat([exp["gate_proj"], exp["up_proj"]], dim=0)
            gate_up_list.append(gate_up)
            down_list.append(exp["down_proj"])

        fused_gate_up = torch.stack(gate_up_list, dim=0)
        fused_down = torch.stack(down_list, dim=0)

        fused_results[prefix] = {
            "gate_up_proj": fused_gate_up,
            "down_proj": fused_down,
        }

        # Free per-expert data for this prefix
        del gate_up_list, down_list
        for exp in expert_data.values():
            exp.clear()
        del all_expert_data[prefix]
        gc.collect()

    del all_expert_data
    gc.collect()

    # ----------------------------------------------------------------
    # Step 4: Write each involved shard ONCE with all updates applied
    # ----------------------------------------------------------------
    updated_shards = {}

    for si, shard_file in enumerate(sorted_involved):
        shard_path = os.path.join(output_dir, shard_file)
        non_expert = shard_non_expert[shard_file]

        # Add fused tensors for prefixes that target this shard
        fused_added = []
        for prefix, target_shard in prefix_to_target_shard.items():
            if target_shard == shard_file and prefix in fused_results:
                non_expert[f"{prefix}gate_up_proj"] = fused_results[prefix]["gate_up_proj"]
                non_expert[f"{prefix}down_proj"] = fused_results[prefix]["down_proj"]
                fused_added.append(prefix)

        save_file(non_expert, shard_path)
        shard_size = sum(t.numel() * t.element_size() for t in non_expert.values())
        updated_shards[shard_file] = (list(non_expert.keys()), shard_size)

        fused_info = ""
        if fused_added:
            fused_info = f", Fused {len(fused_added)} group(s)"

        print(f"    [{si+1}/{len(sorted_involved)}] Wrote {shard_file}: "
              f"{len(non_expert)} keys, {shard_size / 1e9:.2f} GB{fused_info}",
              flush=True)

        # Free memory for this shard
        del shard_non_expert[shard_file]
        for prefix in fused_added:
            del fused_results[prefix]
        del non_expert
        gc.collect()

    return updated_shards


def main():
    parser = argparse.ArgumentParser(
        description="Convert HYV3 checkpoint from inner to outer format (v2, shard-centric post-processing)."
    )
    parser.add_argument(
        "--input_dir", type=str, required=True,
        help="Path to the inner-format checkpoint directory.",
    )
    parser.add_argument(
        "--output_dir", type=str, required=True,
        help="Path to the output outer-format checkpoint directory.",
    )
    parser.add_argument(
        "--workers", type=int, default=8,
        help="Number of parallel worker processes (default: 8).",
    )
    args = parser.parse_args()

    input_dir = os.path.abspath(args.input_dir)
    output_dir = os.path.abspath(args.output_dir)
    num_workers = args.workers

    if not os.path.isdir(input_dir):
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    os.makedirs(output_dir, exist_ok=True)

    # Pre-scan for cross-shard expert groups
    index_path = os.path.join(input_dir, "model.safetensors.index.json")
    cross_shard_prefixes = set()
    if os.path.exists(index_path):
        cross_shard_prefixes = scan_cross_shard_experts(index_path)
        if cross_shard_prefixes:
            print(f"Detected {len(cross_shard_prefixes)} cross-shard expert group(s):")
            for p in sorted(cross_shard_prefixes):
                print(f"  - {p}")
            print()

    # Get all safetensors files
    shard_files = sorted(
        f for f in os.listdir(input_dir) if f.endswith(".safetensors")
    )
    if not shard_files:
        raise FileNotFoundError(f"No .safetensors files found in {input_dir}")

    # Skip already-converted shards (for resumability)
    # NOTE: if there are cross-shard experts, we cannot skip shards that
    # contain deferred keys (they need post-processing). For simplicity,
    # when cross-shard experts exist, we re-process all shards.
    remaining = []
    skipped = []
    if cross_shard_prefixes:
        # Re-process all shards when cross-shard experts exist
        remaining = list(shard_files)
    else:
        for sf in shard_files:
            out_path = os.path.join(output_dir, sf)
            if os.path.exists(out_path) and os.path.getsize(out_path) > 0:
                skipped.append(sf)
            else:
                remaining.append(sf)

    num_shards = len(shard_files)
    num_workers = min(num_workers, len(remaining)) if remaining else 1

    print(f"=" * 60)
    print(f"HYV3 Checkpoint Converter (inner -> outer, v2)")
    print(f"  Input  : {input_dir}")
    print(f"  Output : {output_dir}")
    print(f"  Shards : {num_shards} total, {len(skipped)} already done, {len(remaining)} to process")
    print(f"  Workers: {num_workers}")
    if cross_shard_prefixes:
        print(f"  Cross-shard experts: {len(cross_shard_prefixes)} group(s) (will post-process)")
    print(f"=" * 60)

    t_start = time.time()

    # Build task list for remaining shards
    tasks = [
        (i, len(remaining), sf, input_dir, output_dir, cross_shard_prefixes)
        for i, sf in enumerate(remaining)
    ]

    # Process in parallel
    results = []
    if tasks:
        with Pool(processes=num_workers, initializer=_pool_worker_init) as pool:
            results = pool.map(_process_one_shard, tasks)

    # Collect deferred keys info
    all_deferred = {}  # shard_file -> [deferred_keys]
    for shard_file, key_list, shard_size, elapsed, deferred_keys in results:
        if deferred_keys:
            all_deferred[shard_file] = deferred_keys

    # Post-process cross-shard expert groups (v2: shard-centric)
    updated_shards = {}
    if cross_shard_prefixes and all_deferred:
        updated_shards = post_process_cross_shard_experts(
            output_dir, cross_shard_prefixes, all_deferred
        )

    # Build weight_map and total_size
    weight_map = OrderedDict()
    total_size = 0

    # For skipped shards, read their keys from the output files
    for sf in skipped:
        out_path = os.path.join(output_dir, sf)
        with safe_open(out_path, framework="pt", device="cpu") as f:
            keys = list(f.keys())
            for key in keys:
                weight_map[key] = sf
                t = f.get_tensor(key)
                total_size += t.numel() * t.element_size()

    # Collect results from newly converted shards
    for shard_file, key_list, shard_size, elapsed, deferred_keys in results:
        if shard_file in updated_shards:
            # This shard was updated by post-processing
            updated_key_list, updated_size = updated_shards[shard_file]
            for key in updated_key_list:
                weight_map[key] = shard_file
            total_size += updated_size
        else:
            for key in key_list:
                weight_map[key] = shard_file
            total_size += shard_size

    # Build and save index
    sorted_weight_map = OrderedDict(sorted(weight_map.items()))
    index = {
        "metadata": {"total_size": total_size},
        "weight_map": sorted_weight_map,
    }
    index_path_out = os.path.join(output_dir, "model.safetensors.index.json")
    with open(index_path_out, "w") as f:
        json.dump(index, f, indent=2)
        f.write("\n")
    print(f"\nSaved {index_path_out}")

    # Copy non-safetensors files (config, tokenizer, etc.)
    skip_suffixes = {".safetensors"}
    skip_names = {"model.safetensors.index.json"}
    copied = []
    for fname in os.listdir(input_dir):
        if fname in skip_names:
            continue
        if any(fname.endswith(s) for s in skip_suffixes):
            continue
        src = os.path.join(input_dir, fname)
        dst = os.path.join(output_dir, fname)
        if os.path.isfile(src):
            shutil.copy2(src, dst)
            copied.append(fname)
        elif os.path.isdir(src):
            if os.path.exists(dst):
                shutil.rmtree(dst)
            shutil.copytree(src, dst)
            copied.append(fname + "/")

    if copied:
        print(f"\nCopied files: {', '.join(copied)}")

    t_total = time.time() - t_start
    print(f"\n{'=' * 60}")
    print(f"Conversion complete!")
    print(f"  Total keys : {len(weight_map)}")
    print(f"  Total size : {total_size / 1e9:.2f} GB")
    print(f"  Total time : {t_total:.1f}s ({t_total / 60:.1f} min)")
    print(f"  Output dir : {output_dir}")
    print(f"{'=' * 60}")

if __name__ == "__main__":
    _install_signal_handlers()
    main()
