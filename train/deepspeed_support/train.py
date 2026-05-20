# Copyright 2024 Tencent Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import os
import re
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import json
import torch
import shutil
import logging
from dataclasses import dataclass, field
import deepspeed
from typing import Optional, Dict

import transformers
from torch.utils.data import Dataset
from transformers import Trainer, TrainerCallback
from peft import LoraConfig, get_peft_model, PeftModel
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
from transformers.modeling_utils import unwrap_model


def print_args(args, name='arguments'):
    """Print arguments."""
    if torch.distributed.get_rank() == 0:
        print(f'------------------------ {name} ------------------------', flush=True)
        str_list = []
        for arg in vars(args):
            dots = '.' * (48 - len(arg))
            str_list.append('  {} {} {}'.format(arg, dots, getattr(args, arg)))
        for arg in sorted(str_list, key=lambda x: x.lower()):
            print(arg, flush=True)
        print(f'-------------------- end of {name} ---------------------', flush=True)


@dataclass
class ModelArguments:
    use_flash_attn: bool = field(
        default=False, 
        metadata={"help": "Enable FlashAttention-2 for faster training."}
    )
    use_lora: bool = field(default=False, metadata={"help": "Enable Lora for faster training."})
    hidden_size: int = field(default=2048, metadata={"help": "The hidden size of the model."})
    num_layers: int = field(default=24, metadata={"help": "The number of layers of the model."})
    num_attention_heads: int = field(default=16, metadata={"help": "The number of attention heads of the model."})
    intermediate_size: int = field(default=8192, metadata={"help": "The intermediate size of the model."})
    max_position_embeddings: int = field(
        default=2048, 
        metadata={"help": "The maximum sequence length that this model might ever be used with."}
    )
    vocab_size: int = field(default=50257, metadata={"help": "The vocabulary size of the model."})
    type_vocab_size: int = field(default=1, metadata={"help": "The vocabulary size of the model."})
    layer_norm_eps: float = field(
        default=1e-5, 
        metadata={"help": "The epsilon used by the layer normalization layers of the model."}
    )
    moe_topk: int = field(default=4, metadata={"help": "The topk for MOE."})
    num_experts: int = field(default=8, metadata={"help": "The number of experts for MOE."})
    num_key_value_heads: int = field(default=16, metadata={"help": "The number of key-value heads in GQA."})
    moe_intermediate_size: int = field(default=1536, metadata={"help": "The intermediate size of each MoE expert."})
    use_mixed_mlp_moe: bool = field(
        default=False, 
        metadata={"help": "Whether to use mixed MoE with shared expert."}
    )
    num_shared_expert: int = field(default=1, metadata={"help": "Number of shared experts."})
    use_qk_norm: bool = field(default=False, metadata={"help": "Whether to use qk norm."})
    moe_layer_num_skipped: int = field(default=1, metadata={"help": "Number of initial dense layers before MoE layers."})
    tie_word_embeddings: bool = field(
        default=True, 
        metadata={"help": "Whether to tie the word embeddings of the encoder and the decoder."}
    )
    lora_rank: int = field(default=64, metadata={"help": "The rank of lora."})
    lora_alpha: int = field(default=8, metadata={"help": "Lora alpha"})
    lora_dropout: float = field(default=0.0, metadata={"help": "Lora dropout"})
    train_attention_params_only: bool = field(default=False, metadata={
        "help": "Whether to train attention parameters only."}
    )


@dataclass
class DataArguments:
    train_data_file: str = field(default=None, metadata={"help": "Path to the training data."})
    max_seq_length: int = field(
        default=2048, 
        metadata={"help": "The max sequence length of the model inputs after tokenization."}
    )
    complex_data: Optional[str] = field(default=None)
    use_dummy_data: bool = field(default=False, metadata={"help": "Use dummy data."})


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=2048,
        metadata={"help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."},
    )
    tokenizer_name_or_path: Optional[str] = field(default=None)
    model_name_or_path: Optional[str] = field(default=None)
    min_lr: float = field(
        default=0.01, 
        metadata={"help": "The final learning rate at the end of the decay will be learning_rate * min_lr"}
    )


IGNORE_INDEX = -100


class DummyDataset(Dataset):
    def __init__(self, tokenizer, max_seq_length=512, length=1000):
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.length = length
    
    def __len__(self):
        return self.length
    
    def __getitem__(self, index):
        tokens = torch.randint(0, self.tokenizer.vocab_size, (self.max_seq_length, ))
        return {'input_ids': tokens, 'labels': tokens}


class SFTDataset(Dataset):
    def __init__(self, data_file, tokenizer, max_seq_length = 2048, prompt_format = 'mplus'):
        self.tokenizer = tokenizer
        self.prompt_format = prompt_format
        self.max_seq_length = max_seq_length

        self.data_list = self.load_data(data_file)

    def __len__(self):
        return len(self.data_list)

    def load_data(self, data_file):
        logging.info('Loading data: {}'.format(data_file))
        with open(data_file, 'r', encoding='utf8') as f:
            data_list = f.readlines()
        logging.info("there are {} data in dataset".format(len(data_list)))
        return data_list

    def encode_data(self, data_dict):
        model_inputs = {}
        reasoning_effort = data_dict.get('reasoning_effort', None)
        if reasoning_effort is None:
            reasoning_effort = 'no_think'
        template_output = self.tokenizer.apply_chat_template(data_dict['messages'], tokenize=True, return_dict=False, is_training=True, reasoning_effort=reasoning_effort)
        if isinstance(template_output, list) and len(template_output) > 0 and isinstance(template_output[0], list):
            template_output = template_output[0]
        message_tokens = torch.tensor(template_output, dtype=torch.long)

        # Use new HunYuan tokenizer special tokens
        assistant_token_id = self.tokenizer.convert_tokens_to_ids('<｜hy_Assistant｜>')
        eos_token_id = self.tokenizer.convert_tokens_to_ids(self.tokenizer.eos_token)
        pad_token_id = self.tokenizer.pad_token_id

        # Find assistant reply boundaries: starts at <｜hy_Assistant｜>, ends at eos_token
        loss_token_begins = (message_tokens == assistant_token_id).nonzero(as_tuple=True)[0].tolist()
        loss_token_ends = (message_tokens == eos_token_id).nonzero(as_tuple=True)[0].tolist()
        message_labels = torch.tensor([IGNORE_INDEX] * message_tokens.shape[0])
        for begin_idx, end_idx in zip(loss_token_begins, loss_token_ends):
            # Compute loss from the token after <｜hy_Assistant｜> to eos_token (inclusive)
            message_labels[begin_idx + 1:end_idx + 1] = message_tokens[begin_idx + 1:end_idx + 1]
        input_ids = message_tokens.to(torch.long)
        labels = message_labels.to(torch.long)

        input_ids = input_ids[:self.max_seq_length]
        labels = labels[:self.max_seq_length]
        attention_mask = [1 if val != pad_token_id else 0 for val in input_ids]
        model_inputs["input_ids"] = input_ids
        model_inputs["attention_mask"] = torch.tensor(attention_mask, dtype=torch.bool)
        model_inputs["labels"] = labels

        return model_inputs

    def __getitem__(self, index):
        data = self.data_list[index]
        data = json.loads(data)
        model_inputs = self.encode_data(data)

        return model_inputs


@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances):
        input_ids = [instance['input_ids'] for instance in instances]
        labels = [instance['labels'] for instance in instances]
        pad_token_id = self.tokenizer.pad_token_id
        input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=pad_token_id)
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)
        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(pad_token_id),
        )


def make_supervised_data_module(tokenizer, data_args) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    if data_args.use_dummy_data:
        train_dataset = DummyDataset(tokenizer, data_args.max_seq_length)
    else:
        train_dataset = SFTDataset(
            tokenizer=tokenizer, 
            data_file=data_args.train_data_file, 
            max_seq_length=data_args.max_seq_length
        )
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    return dict(train_dataset=train_dataset, eval_dataset=None, data_collator=data_collator)


# for full model training, change the config.json, copy the model and configuration to support Auto load
class CustomSaveCallback(TrainerCallback):
    def on_save(self, args, state, control, **kwargs):
        if torch.distributed.get_rank() == 0:
            output_dir = os.path.join(args.output_dir, f"{PREFIX_CHECKPOINT_DIR}-{state.global_step}")

            # Copy tokenizer files to checkpoint directory
            tokenizer_files = [
                'generation_config.json',
                'hy.tiktoken',
                'tokenizer_config.json',
                'tokenization_hy.py',
                'tokenizer.json',
                'special_tokens_map.json',
                'chat_template.jinja',
            ]
            for fname in tokenizer_files:
                src = os.path.join(args.tokenizer_name_or_path, fname)
                if os.path.isfile(src):
                    shutil.copy(src, os.path.join(output_dir, fname))

        return control


def train():
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    print_args(model_args, 'model arguments')
    print_args(data_args, 'data arguments')
    print_args(training_args, 'training arguments')

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        training_args.tokenizer_name_or_path,
        trust_remote_code = True
    )

    init_kwargs = {}
    if model_args.use_flash_attn:
        init_kwargs["attn_implementation"] = "flash_attention_2"
        # Workaround: transformers >= 5.x uses importlib.metadata.packages_distributions()
        # to verify flash-attn package name, which fails when the package is installed under
        # a custom distribution name (e.g. ptm-flash-attn). Patch the check to skip it.
        try:
            from transformers.modeling_flash_attention_utils import FLASH_ATTENTION_COMPATIBILITY_MATRIX
            _orig_pkg_check = FLASH_ATTENTION_COMPATIBILITY_MATRIX[2]["pkg_availability_check"]
            FLASH_ATTENTION_COMPATIBILITY_MATRIX[2]["pkg_availability_check"] = lambda *a, **kw: True
            print("[Patch] Bypassed flash_attn package distribution name check for FA2.")
        except Exception as e:
            print(f"[Patch] Could not patch FA2 pkg check (non-fatal): {e}")
    if training_args.bf16:
        init_kwargs["dtype"] = torch.bfloat16
    elif training_args.fp16:
        init_kwargs["dtype"] = torch.float16

    # Check if model weights exist (not just the directory)
    _has_weights = (
        training_args.model_name_or_path is not None
        and os.path.isdir(training_args.model_name_or_path)
        and any(
            os.path.isfile(os.path.join(training_args.model_name_or_path, f))
            for f in ("model.safetensors", "pytorch_model.bin", "model.safetensors.index.json", "pytorch_model.bin.index.json")
        )
    )

    # -----------------------------------------------------------------------
    # Fix: Rename checkpoint keys so that old-style weight names (e.g.
    # self_attn.q_norm) are mapped to the current model attribute names
    # (e.g. self_attn.query_layernorm).  The model's
    # _fix_state_dict_key_on_load hook is NOT invoked on the DeepSpeed
    # ZeRO-3 loading path, so we monkey-patch the ZeRO-3 loader instead.
    # -----------------------------------------------------------------------
    _CKPT_KEY_RENAMES = [
        ("mlp.gate.wg.", "mlp.router.gate."),
    ]

    from transformers.integrations.deepspeed import (
        _load_state_dict_into_zero3_model as _orig_load_zero3,
    )
    import transformers.integrations.deepspeed as _ds_mod
    import transformers.modeling_utils as _mu_mod

    def _patched_load_zero3(model_to_load, state_dict, load_config=None):
        new_sd = {}
        for k, v in state_dict.items():
            new_k = k
            for old_sub, new_sub in _CKPT_KEY_RENAMES:
                if old_sub in new_k:
                    new_k = new_k.replace(old_sub, new_sub)
                    break
            new_sd[new_k] = v

        # Call original ZeRO-3 loader for parameters
        result = _orig_load_zero3(model_to_load, new_sd, load_config)

        # -------------------------------------------------------------------
        # Patch: Manually load buffers (e.g. e_score_correction_bias).
        # ZeRO-3's loader only handles named_parameters, not named_buffers.
        # -------------------------------------------------------------------
        buffers_loaded = 0
        for name, buf in model_to_load.named_buffers():
            if name in new_sd:
                src_tensor = new_sd[name]
                if isinstance(src_tensor, torch.Tensor):
                    buf.data.copy_(src_tensor.to(buf.dtype))
                    buffers_loaded += 1
                    # Remove from unexpected keys if tracked
                    if isinstance(result, tuple) and len(result) >= 2:
                        if isinstance(result[1], set):
                            result[1].discard(name)
        if buffers_loaded > 0:
            print(f"[HYV3 Patch] Manually loaded {buffers_loaded} buffers "
                  f"(e.g. e_score_correction_bias) into model.")

        return result

    _ds_mod._load_state_dict_into_zero3_model = _patched_load_zero3
    _mu_mod._load_state_dict_into_zero3_model = _patched_load_zero3
    # -----------------------------------------------------------------------

    # -------------------------------------------------------------------
    # Patch: Save-time reverse key rename + 3D -> per-expert unfuse.
    #
    # When saving checkpoints, the model state_dict uses 3D fused experts
    # and new naming.  We reverse both for old checkpoint compatibility:
    #   - mlp.gate.           -> mlp.router.gate.
    #   - mlp.e_score_correction_bias -> mlp.expert_bias
    #   - mlp.shared_experts. -> mlp.shared_mlp.
    #   - experts.gate_up_proj -> experts.{N}.gate_proj.weight + up_proj
    #   - experts.down_proj    -> experts.{N}.down_proj.weight
    # -------------------------------------------------------------------
    _SAVE_KEY_RENAMES = [
        ("mlp.gate.", "mlp.router.gate."),
        ("mlp.e_score_correction_bias", "mlp.expert_bias"),
        ("mlp.shared_experts.", "mlp.shared_mlp."),
    ]
    _FUSED_EXPERT_KEY_RE = re.compile(
        r"^(.*\.mlp\.experts\.)(gate_up_proj|down_proj)$"
    )

    def _apply_save_reverse_rename_patch():
        try:
            from transformers.models.hy_v3.modeling_hy_v3 import HYV3ForCausalLM
        except ImportError:
            try:
                from transformers.hy_v3.modeling_hy_v3 import HYV3ForCausalLM
            except ImportError:
                print("[HYV3 Patch] Could not import HYV3ForCausalLM; "
                      "save reverse rename patch NOT applied.")
                return

        _orig_save_pretrained = HYV3ForCausalLM.save_pretrained

        def _patched_save_pretrained(self, *args, **kwargs):
            state_dict = kwargs.get("state_dict", None)
            if state_dict is not None:
                reversed_sd = {}
                for k, v in state_dict.items():
                    new_k = k
                    # Apply simple key renames
                    for new_sub, old_sub in _SAVE_KEY_RENAMES:
                        if new_sub in new_k:
                            new_k = new_k.replace(new_sub, old_sub)
                            break

                    # Check if this is a fused 3D expert key
                    m = _FUSED_EXPERT_KEY_RE.match(new_k)
                    if m:
                        prefix = m.group(1)  # e.g. "model.layers.1.mlp.experts."
                        proj_type = m.group(2)  # "gate_up_proj" or "down_proj"

                        if proj_type == "gate_up_proj":
                            # v shape: [num_experts, 2*intermediate, hidden]
                            num_experts = v.shape[0]
                            intermediate = v.shape[1] // 2
                            for i in range(num_experts):
                                gate = v[i, :intermediate, :]
                                up = v[i, intermediate:, :]
                                reversed_sd[f"{prefix}{i}.gate_proj.weight"] = gate
                                reversed_sd[f"{prefix}{i}.up_proj.weight"] = up
                        elif proj_type == "down_proj":
                            # v shape: [num_experts, hidden, intermediate]
                            num_experts = v.shape[0]
                            for i in range(num_experts):
                                reversed_sd[f"{prefix}{i}.down_proj.weight"] = v[i]
                    else:
                        reversed_sd[new_k] = v

                kwargs["state_dict"] = reversed_sd
                print(f"[HYV3 Patch] Reverse-renamed and unfused "
                      f"{len(state_dict)} -> {len(reversed_sd)} "
                      f"state_dict keys for old checkpoint compatibility.")
            return _orig_save_pretrained(self, *args, **kwargs)

        HYV3ForCausalLM.save_pretrained = _patched_save_pretrained
        print("[HYV3 Patch] Applied: save-time reverse key rename + "
              "3D -> per-expert unfuse for old ckpt compatibility.")

    _apply_save_reverse_rename_patch()
    # -------------------------------------------------------------------

    if _has_weights:
        print(f"Initializing model from local file: {training_args.model_name_or_path}")
        model = transformers.AutoModelForCausalLM.from_pretrained(
            training_args.model_name_or_path,
            trust_remote_code=True,
            **init_kwargs
        )
    else:
        from transformers import HYV3Config
        from transformers import HYV3ForCausalLM
        print(f"Model weights not found at: {training_args.model_name_or_path}, "
              f"using random initialized HYV3 model instead.")
        # Use len(tokenizer) to include added special tokens; tokenizer.vocab_size
        # may only return the base vocabulary size and miss special tokens whose
        # IDs exceed that range, causing index-out-of-bounds in the embedding layer.
        config = HYV3Config(
            vocab_size=len(tokenizer),
            hidden_size=model_args.hidden_size,
            intermediate_size=model_args.intermediate_size,
            max_position_embeddings=training_args.model_max_length,
            moe_topk=model_args.moe_topk,
            num_experts=model_args.num_experts,
            num_attention_heads=model_args.num_attention_heads,
            num_key_value_heads=model_args.num_key_value_heads,
            num_hidden_layers=model_args.num_layers,
            moe_intermediate_size=model_args.moe_intermediate_size,
            use_mixed_mlp_moe=model_args.use_mixed_mlp_moe,
            num_shared_expert=model_args.num_shared_expert,
            use_qk_norm=model_args.use_qk_norm,
            moe_layer_num_skipped=model_args.moe_layer_num_skipped,
            tie_word_embeddings=model_args.tie_word_embeddings,
        )
        with deepspeed.zero.Init(dtype=init_kwargs.get("torch_dtype", torch.bfloat16), config_dict_or_path=training_args.deepspeed):
            model = HYV3ForCausalLM(config)
    
    if model_args.train_attention_params_only:
        for name, param in model.named_parameters():
            if 'self_attn' not in name:
                param.requires_grad = False

    if model_args.use_lora:
        # define Lora configuration
        lora_config = LoraConfig(
            r=model_args.lora_rank,
            lora_alpha=model_args.lora_alpha,
            lora_dropout=model_args.lora_dropout,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config)

    data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args)
    # Tell Trainer not to attempt DataParallel
    model.is_parallelizable = True
    model.model_parallel = True

    training_args.lr_scheduler_kwargs = {
        'min_lr_rate': training_args.min_lr / training_args.learning_rate,
    }

    # -----------------------------------------------------------------------
    # Fix: DeepSpeed ZeRO-3 + gradient checkpointing compatibility.
    #
    # PyTorch's torch.utils.checkpoint with use_reentrant=False (the default
    # in transformers) performs strict metadata checks on recomputed tensors
    # during backward.  Under ZeRO-3, parameters are all-gathered during the
    # first forward pass (shape=[full_size]) but may be partitioned back
    # (shape=[0]) when the checkpoint recomputes, causing a CheckpointError.
    #
    # Setting use_reentrant=True avoids this strict metadata check.
    # -----------------------------------------------------------------------
    if training_args.gradient_checkpointing and training_args.deepspeed:
        training_args.gradient_checkpointing_kwargs = {"use_reentrant": True}

    trainer = Trainer(
        model=model, 
        processing_class=tokenizer, 
        args=training_args,
        callbacks=[CustomSaveCallback],
        **data_module
    )
    model.config.use_cache = False

    # -----------------------------------------------------------------------
    # Monkey-patch: fix dtype mismatch in DeepSpeed ZeRO-3 linear wrapper.
    #
    # By this point the DeepSpeed engine has been initialised by the Trainer
    # and torch.nn.functional.linear has been replaced with
    # zero3_linear_wrap.  That wrapper does NOT auto-align input/weight
    # dtypes before the matmul, causing "expected mat1 and mat2 to have the
    # same dtype" errors in mixed-precision paths (MoE router gate in fp32
    # with bf16 weights, expert FFN receiving fp32 routing-weighted input
    # with bf16 weights, etc.).
    #
    # We wrap F.linear HERE (after DeepSpeed init) so that:
    #   1. We are sure to capture the already-replaced function.
    #   2. The dtype cast happens *outside* the autograd.Function, so
    #      gradient-checkpointing recompute sees identical tensor metadata.
    # -----------------------------------------------------------------------
    import torch.nn.functional as _F
    _orig_F_linear = _F.linear

    def _dtype_safe_linear(input, weight, bias=None):
        if input.dtype != weight.dtype:
            input = input.to(weight.dtype)
        return _orig_F_linear(input, weight, bias)

    _F.linear = _dtype_safe_linear
    # -----------------------------------------------------------------------

    trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)


if __name__ == "__main__":
    train()
