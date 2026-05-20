"""
HYV3 monkey-patches for LLaMA Factory + DeepSpeed training.

This module applies all necessary runtime patches so that HYV3 (MoE)
can be trained correctly under LLaMA Factory with DeepSpeed.

Usage:
    Import this module **before** calling `llamafactory-cli train`:

        import hy_v3_patches          # applies patches on import
        # ... then start training

    Or add to the LLaMA Factory YAML via a custom entry-point wrapper.

Patches applied:
    1. (Removed) -- transformers 5.8.1+ has built-in conversion_mapping for
       hy_v3 that handles key renaming + expert fusing automatically.
    2. Router forward dtype fix (MoE router gate dtype alignment for ZeRO-3)
    3. gradient_checkpointing   (use_reentrant=True for ZeRO-3)
    4. Tokenizer file copy      (CustomSaveCallback)
    5. (Removed) -- was per-expert ModuleList, now using native 3D Parameters
    6. (Removed) -- transformers 5.8.1+ has built-in revert_weight_conversion
       in save_pretrained that handles outer->inner format automatically.
"""

import os
import logging
import shutil
from typing import Optional

import torch
import torch.nn.functional as _F

logger = logging.getLogger(__name__)

# ============================================================================
# Patch 2: Router forward dtype alignment for ZeRO-3
#
# The HYV3 MoE HYV3TopKRouter.forward() calls F.linear with .float().
# Under DeepSpeed ZeRO-3, F.linear is replaced by zero3_linear_wrap which
# internally does input.matmul(weight.t()) WITHOUT aligning dtypes.
# When ZeRO-3 stores the gate weight in bf16, the fp32 input causes a
# dtype mismatch RuntimeError.
#
# Fix: monkey-patch HYV3TopKRouter.forward to cast input to
# self.weight.dtype before F.linear, then cast the output back to float32.
# ============================================================================

_router_patch_applied = False

def _apply_router_dtype_patch():
    """Monkey-patch HYV3TopKRouter.forward to align gate input dtype with weight dtype."""
    global _router_patch_applied
    if _router_patch_applied:
        return

    try:
        from transformers.models.hy_v3.modeling_hy_v3 import HYV3TopKRouter
    except ImportError:
        try:
            from transformers.hy_v3.modeling_hy_v3 import HYV3TopKRouter
        except ImportError:
            logger.warning(
                "Could not import HYV3TopKRouter; "
                "router dtype patch NOT applied."
            )
            return

    def _patched_router_forward(
        self,
        hidden_states: torch.Tensor,
        e_score_correction_bias: torch.Tensor,
    ) -> tuple:
        hidden_states = hidden_states.reshape(-1, self.hidden_dim)
        # Cast input to match weight dtype (bf16 under ZeRO-3)
        # instead of hard-coding float32, to avoid matmul dtype mismatch.
        weight_dtype = self.weight.dtype
        router_logits = _F.linear(hidden_states.to(weight_dtype), self.weight.to(weight_dtype))
        # Cast back to float32 for numerically stable sigmoid
        router_logits = router_logits.to(torch.float32)
        routing_weights = torch.sigmoid(router_logits)

        scores_for_choice = routing_weights + e_score_correction_bias
        _, top_k_index = torch.topk(scores_for_choice, self.top_k, dim=-1, sorted=False)
        top_k_weights = routing_weights.gather(1, top_k_index)

        top_k_weights = top_k_weights / (top_k_weights.sum(dim=-1, keepdim=True) + 1e-20)
        top_k_weights = top_k_weights * self.router_scaling_factor

        return router_logits, top_k_weights, top_k_index

    HYV3TopKRouter.forward = _patched_router_forward
    _router_patch_applied = True
    logger.info("HYV3 patch applied: HYV3TopKRouter.forward dtype alignment for ZeRO-3.")

# ============================================================================
# Patch 3: gradient_checkpointing use_reentrant=True
#
# PyTorch's torch.utils.checkpoint with use_reentrant=False (the default
# in transformers) performs strict metadata checks on recomputed tensors.
# Under ZeRO-3, parameters are all-gathered during the first forward pass
# but may be partitioned back when the checkpoint recomputes, causing a
# CheckpointError.  Setting use_reentrant=True avoids this.
#
# This is applied via a Trainer callback that modifies training_args
# before training starts.
# ============================================================================

# ============================================================================
# Patch 4: Tokenizer file copy callback
#
# Ensures each checkpoint directory is self-contained for inference by
# copying all tokenizer-related files from the original tokenizer path.
# ============================================================================

# Tokenizer files that should be copied to each checkpoint
_TOKENIZER_FILES = [
    "generation_config.json",
    "hy.tiktoken",
    "tokenizer_config.json",
    "tokenization_hy.py",
    "tokenizer.json",
    "special_tokens_map.json",
    "chat_template.jinja",
]

def _copy_tokenizer_to_checkpoint(tokenizer_dir: str, checkpoint_dir: str):
    """Copy tokenizer files from tokenizer_dir to checkpoint_dir."""
    for fname in _TOKENIZER_FILES:
        src = os.path.join(tokenizer_dir, fname)
        if os.path.isfile(src):
            shutil.copy(src, os.path.join(checkpoint_dir, fname))

# ============================================================================
# LLaMA Factory Callback: integrates patches 3, 4 into the training loop
# ============================================================================

try:
    from transformers import TrainerCallback
    from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR

    class HYV3PatchCallback(TrainerCallback):
        """
        LLaMA Factory compatible callback that applies HYV3-specific patches.

        Add to your YAML or pass to Trainer:
            callbacks: [hy_v3_patches.HYV3PatchCallback]
        """

        def __init__(self, tokenizer_dir: Optional[str] = None):
            """
            Args:
                tokenizer_dir: Path to the original tokenizer directory.
                    If None, will try to use model_name_or_path from training args.
            """
            self._tokenizer_dir = tokenizer_dir

        def on_train_begin(self, args, state, control, **kwargs):
            # --- Patch 3: gradient_checkpointing use_reentrant ---
            if getattr(args, "gradient_checkpointing", False) and getattr(args, "deepspeed", None):
                if not hasattr(args, "gradient_checkpointing_kwargs") or not args.gradient_checkpointing_kwargs:
                    args.gradient_checkpointing_kwargs = {"use_reentrant": True}
                elif "use_reentrant" not in args.gradient_checkpointing_kwargs:
                    args.gradient_checkpointing_kwargs["use_reentrant"] = True
                logger.info("HYV3 patch applied: gradient_checkpointing use_reentrant=True.")

            return control

        def on_save(self, args, state, control, **kwargs):
            # --- Patch 4: Copy tokenizer files ---
            if torch.distributed.is_initialized() and torch.distributed.get_rank() != 0:
                return control

            checkpoint_dir = os.path.join(
                args.output_dir,
                f"{PREFIX_CHECKPOINT_DIR}-{state.global_step}",
            )

            # Determine tokenizer directory
            tokenizer_dir = self._tokenizer_dir
            if tokenizer_dir is None:
                # Try common locations
                tokenizer_dir = getattr(args, "tokenizer_name_or_path", None)
                if tokenizer_dir is None:
                    tokenizer_dir = getattr(args, "model_name_or_path", None)

            if tokenizer_dir and os.path.isdir(tokenizer_dir):
                _copy_tokenizer_to_checkpoint(tokenizer_dir, checkpoint_dir)
                logger.info(
                    "HYV3: Copied tokenizer files from %s to %s",
                    tokenizer_dir, checkpoint_dir
                )

            return control

except ImportError:
    logger.warning(
        "transformers not available; HYV3PatchCallback not defined."
    )

# ============================================================================
# Auto-apply patches on import
# ============================================================================

# Patch 2: Router dtype fix
_apply_router_dtype_patch()

# Patches 3, 4 are applied via HYV3PatchCallback during training.
# Users should add HYV3PatchCallback to their Trainer callbacks.

logger.info(
    "HYV3 patches module loaded. Patch 2 (Router dtype fix) applied. "
    "Remember to add HYV3PatchCallback to your Trainer callbacks "
    "for gradient_checkpointing and tokenizer copy support."
)
