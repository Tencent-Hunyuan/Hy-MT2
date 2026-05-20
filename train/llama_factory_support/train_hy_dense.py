"""
LLaMA Factory training entry-point wrapper for HunYuan Dense models.

This script:
  1. Registers the hy_dense_1_8b and hy_dense_7b chat templates
  2. Injects a lightweight PatchCallback (tokenizer copy + gradient checkpointing fix)
  3. Calls run_exp() to start LLaMA Factory training

How it works:
  - train_lf_dense.sh launches this script via torchrun directly:
        torchrun ... train_hy_dense.py hy_dense_1_8b_full_sft.yaml
  - Each torchrun worker executes this script, so all patches are applied
    in every worker process before training begins.
  - We call run_exp() directly (not the CLI launcher) to avoid the
    launcher re-spawning workers and losing our patches.

Note:
  Dense models do NOT need MoE-specific patches (router dtype fix, expert
  key rename, etc.). Only the tokenizer copy callback and gradient
  checkpointing fix are needed.

Usage:
    # Via launch script (recommended):
    bash train_lf_dense.sh

    # Direct single-node (1 GPU, 1.8B model):
    torchrun --nproc_per_node 1 train_hy_dense.py hy_dense_1_8b_full_sft.yaml

    # Direct single-node (2 GPUs, 7B model):
    torchrun --nproc_per_node 2 train_hy_dense.py hy_dense_7b_full_sft.yaml
"""

import sys
import os

# Add current directory to path so templates can be imported
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Step 1: Register Dense model templates (must be before training starts)
import hy_dense_template  # noqa: F401

# Step 2: Import the patch callback (reuse HYV3PatchCallback for tokenizer copy)
# The MoE router patch will be silently skipped since Dense models don't have
# HYV3TopKRouter. Only Patch 3 (gradient_checkpointing) and Patch 4 (tokenizer
# copy) will be effective.
import hy_v3_patches  # noqa: F401

# Step 3: Inject PatchCallback into LLaMA Factory's training flow
from llamafactory.train.sft.workflow import run_sft as _orig_run_sft


def _patched_run_sft(model_args, data_args, training_args, finetuning_args, generating_args, callbacks=None):
    """Wrap run_sft to inject HYV3PatchCallback for tokenizer copy."""
    if callbacks is None:
        callbacks = []

    # Determine tokenizer directory for the save callback
    tokenizer_dir = getattr(model_args, "model_name_or_path", None)
    callbacks.append(hy_v3_patches.HYV3PatchCallback(tokenizer_dir=tokenizer_dir))

    return _orig_run_sft(model_args, data_args, training_args, finetuning_args, generating_args, callbacks=callbacks)


# Monkey-patch the SFT workflow
import llamafactory.train.sft.workflow as _sft_wf
_sft_wf.run_sft = _patched_run_sft


def main():
    """Entry point: called by torchrun in each worker process.

    Since train_lf_dense.sh launches us via torchrun directly, all patches
    (template registration, tokenizer copy callback injection) are already
    applied in this process. We just call run_exp() to start training.
    """
    from llamafactory.train.tuner import run_exp
    run_exp()


if __name__ == "__main__":
    main()
