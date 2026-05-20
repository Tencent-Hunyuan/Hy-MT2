"""
LLaMA Factory training entry-point wrapper for HYV3.

This script:
  1. Registers the hy_v3 chat template
  2. Applies all HYV3 monkey-patches (checkpoint key rename, dtype fix, etc.)
  3. Injects HYV3PatchCallback into the training loop
  4. Calls run_exp() to start LLaMA Factory training

How it works:
  - train_lf.sh launches this script via torchrun directly:
        torchrun ... train_hy_v3.py hy_v3_full_sft.yaml
  - Each torchrun worker executes this script, so all patches are applied
    in every worker process before training begins.
  - We call run_exp() directly (not the CLI launcher) to avoid the
    launcher re-spawning workers and losing our patches.

Usage:
    # Via launch script (recommended):
    bash train_lf.sh

    # Direct single-node (8 GPUs):
    torchrun --nproc_per_node 8 train_hy_v3.py hy_v3_full_sft.yaml
"""

import sys
import os

# Add current directory to path so patches can be imported
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Step 1: Register HYV3 template (must be before training starts)
import hy_v3_template  # noqa: F401

# Step 2: Apply checkpoint key rename patch (must be before model loading)
import hy_v3_patches  # noqa: F401

# Step 3: Inject HYV3PatchCallback into LLaMA Factory's training flow
from llamafactory.train.sft.workflow import run_sft as _orig_run_sft


def _patched_run_sft(model_args, data_args, training_args, finetuning_args, generating_args, callbacks=None):
    """Wrap run_sft to inject HYV3PatchCallback."""
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

    Since train_lf.sh launches us via torchrun directly, all patches
    (template registration, checkpoint key rename, SFT callback injection)
    are already applied in this process.  We just call run_exp() to start
    training — no need to go through the CLI launcher.
    """
    from llamafactory.train.tuner import run_exp
    run_exp()


if __name__ == "__main__":
    main()
