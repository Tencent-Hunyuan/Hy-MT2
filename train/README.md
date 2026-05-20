<p align="left">
    <a href="README_CN.md">中文</a> ｜ English
</p>

# Model Training

Hy3 preview provides processes related to model training. This section details how to process training data for model training purposes.

## Training Data Format and Processing

The training data should be formatted as a list of messages. By default, the system prompt for both training and inference is empty, but you may customize it as needed.

Below is a training data example for a translation task:

```python
# Translation task example
{"messages": [{"role": "user", "content": "将以下中文翻译为英文，只输出翻译结果，不要额外解释：\n\n实验结果证明了假设的正确性。"}, {"role": "assistant", "content": "The experimental results demonstrate the correctness of the hypothesis."}]}

```

## Quick Start

You can quickly get started by following the instructions in the Quick Start Guide.

## Model Training

### Hardware Requirements

The following are the minimum hardware requirements for each model at max_seq_length = 8192:

#### Hy-MT2-1.8B (Dense)

| Training Method | DeepSpeed Strategy | Minimum GPU Requirement |
|----------------|-------------------|------------------------|
| LoRA Fine-tuning | ZeRO-2 (no offload) | 1 GPU (24GB+) |
| Full Fine-tuning | ZeRO-2 (no offload) | 1 GPU (24GB+) |

#### Hy-MT2-7B (Dense)

| Training Method | DeepSpeed Strategy | Minimum GPU Requirement |
|----------------|-------------------|------------------------|
| LoRA Fine-tuning | ZeRO-2 (no offload) | 1 GPU (80GB+) |
| Full Fine-tuning | ZeRO-3 (no offload) | 2 GPUs (80GB+ each) |

#### Hy-MT2-30B-A3B (MoE)

| Training Method | DeepSpeed Strategy | Minimum GPU Requirement |
|----------------|-------------------|------------------------|
| LoRA Fine-tuning | ZeRO-2 (no offload) | 8 GPUs on a single machine (80GB+ each) |
| Full Fine-tuning | ZeRO-3 + offload | 8 GPUs on a single machine (80GB+ each) |

### Configure Passwordless SSH Login Between Machines (Multi-Machine Training)

> If you only use single-machine training, you can skip this section.

The following instructions use two machines as an example, with their IPs denoted as `${ip1}` and `${ip2}`. All steps should be performed inside the Docker container.

First, configure passwordless SSH for each container on every machine:

```sh
ssh-keygen			# Generate id_rsa and id_rsa.pub for passwordless login
ssh-keygen -t rsa -A    # Generate /etc/ssh/ssh_host_rsa_key and ssh_host_ecdsa_key for SSH listening
/usr/sbin/sshd -p 36005 -o ListenAddress=0.0.0.0        # Start SSH listening
echo "Port 36005" > ~/.ssh/config   # Set SSH connection port to 36005
passwd root    # Set the root password to avoid monitoring platform alerts
```

Note: `36005` is an example port. You may use any available port, but ensure it is **open** and **not occupied by other processes**.

Next, in each machine's container, execute:

```sh
cat ~/.ssh/id_rsa.pub
```

**Copy the output SSH public key and paste it into the `~/.ssh/authorized_keys` file, one key per line. This must be done on every machine.** In the end, the `~/.ssh/authorized_keys` file on each machine should be identical and contain the public keys of all machines.

Please note that for multi-node training, the code executed on each node must be identical. It is recommended to mount a shared network drive. If this is not possible, you must manually copy the dataset, scripts, and code to the same directory on each machine.

### Launch Methods

This project provides two training methods. You can choose based on your needs:

- **DeepSpeed Native Training** (based on HuggingFace Transformers Trainer): Located in the `train/deepspeed_support` directory
- **LLaMA-Factory Training**: Located in the `train/llama_factory_support` directory

#### DeepSpeed Native Training

Reference: [HuggingFace Transformers Trainer](https://huggingface.co/docs/transformers/main/en/main_classes/trainer)

##### Training Scripts

In the `train/deepspeed_support` directory, the training scripts for each model are as follows:

| Model | Full Fine-tuning | LoRA Fine-tuning |
|-------|-----------------|-----------------|
| Hy-MT2-1.8B (Dense) | `bash train_dense.sh 1.8B` | `bash train_dense_lora.sh 1.8B` |
| Hy-MT2-7B (Dense) | `bash train_dense.sh 7B` | `bash train_dense_lora.sh 7B` |
| Hy-MT2-30B-A3B (MoE) | `bash train.sh` | `bash train_lora.sh` |

##### Single-Machine Training

In the `train/deepspeed_support` directory, install dependencies and execute the corresponding script:

```sh
pip install -r requirements.txt
# Example: Dense 1.8B full fine-tuning
bash train_dense.sh 1.8B
```

##### Multi-Machine Training

To launch training across multiple machines, please first complete the configuration in [Configure Passwordless SSH Login Between Machines](#configure-passwordless-ssh-login-between-machines-multi-machine-training), and ensure all machines are within the same cluster.

Confirm that dependencies are installed (if not, run `pip install -r requirements.txt`), then set the `IP_LIST` environment variable in the corresponding training script:

```shell
export HOST_GPU_NUM=8
# IP list, comma separated. e.g. "192.168.1.1,192.168.1.2" or single node "192.168.1.1"
IP_LIST=${IP_LIST:-"127.0.0.1"}
```

Note: If the `IP_LIST` environment variable is not set, replace `IP_LIST` with the IP list! The format is:
```
For a single IP:
IP_LIST=${ip_1}

For multiple IPs:
IP_LIST=${ip_1},${ip_2}

```

Replace `${ip_1}` and `${ip_2}` with the actual IP addresses.

Then, on the machine with `${ip1}`, execute the corresponding training script in the `train/deepspeed_support/` directory. On first launch, you may see the following output:

```ssh
The authenticity of host '[ip]:36005 ([ip]:36005)' can't be established.
ECDSA key fingerprint is xxxxxx.
ECDSA key fingerprint is MD5:xxxxxx.
Are you sure you want to continue connecting (yes/no)?
```

Type `yes` to continue.

##### Key Parameters

The key parameters in the script are as follows:

- `--deepspeed`: Path to the DeepSpeed configuration file. Three default DeepSpeed configuration files are provided in the `train/deepspeed_support` folder: `ds_zero2_no_offload.json`, `ds_zero3_no_offload.json`, and `ds_zero3_offload.json`, with decreasing memory requirements in that order.
- `--model_name_or_path`: Path to the Hy3 preview HF pre-trained model weights to load.
- `--tokenizer_name_or_path`: Path to the tokenizer folder.
- `--train_data_file`: Path to the training file, which should be a jsonl file.
- `--output_dir`: Output directory where logs, tensorboard files, and model weights will be stored.
- `--per_device_train_batch_size`: Batch size per GPU.
- `--gradient_accumulation_steps`: Number of gradient accumulation steps. The global batch size is `per_device_train_batch_size * gradient_accumulation_steps * dp_size`.
- `--max_steps`: Total number of training steps.
- `--save_steps`: Number of steps between saving checkpoints.
- `--use_lora`: Whether to use LoRA training. Also accepts `--lora_rank`, `--lora_alpha`, and `--lora_dropout` parameters. By default, LoRA is applied to "q_proj", "k_proj", "v_proj", and "o_proj". To change this, modify the code. Note: **When using LoRA training, only the LoRA weights are saved, not the base model weights.** To merge LoRA weights, see the "LoRA Weight Merging" section below.
- `--make_moe_param_leaf_module`: When using ZeRO-3 with MoE training, treat the MoE module as a leaf module, i.e., its parameters are not partitioned by ZeRO-3. This option is expected to significantly increase memory usage.
- `--gradient_checkpointing`: Enable gradient checkpointing.
- `--train_attention_params_only`: Whether to train only attention parameters.
- `--learning_rate`: Maximum learning rate during training.
- `--min_lr`: Minimum learning rate during training.
- `--use_flash_attn`: Enable flash-attention for accelerated training.

**Notes:**

- To resume training from a previously saved checkpoint rather than loading pre-trained weights, specify `--resume_from_checkpoint` with the path to the checkpoint. Do not specify `--model_name_or_path`; this will load only the weights without the training state.
- When resuming from a checkpoint, there may be minor differences in loss due to the randomness of some non-deterministic algorithms. This is normal. See: [HuggingFace Transformers Trainer Randomness](https://huggingface.co/docs/transformers/main/en/main_classes/trainer#randomness)
- When `--model_name_or_path` is specified, all model-related parameters will be ignored.
- Samples within a batch are padded to the length of the longest sample in the batch, but the maximum length of each sample is `max_seq_length`. Any excess will be truncated.
- If you see a warning about bias weights not being loaded, you can ignore it. Hunyuan-Large does not use bias.

##### What if GPU Memory is Insufficient?

Reference: [DeepSpeed Configuration](https://www.deepspeed.ai/docs/config-json/)

You can try modifying the DeepSpeed configuration by removing the `auto` attribute from the following parameters and reducing their values:

- `stage3_param_persistence_threshold`
- `stage3_prefetch_bucket_size`
- `stage3_max_reuse_distance`

##### LoRA Weight Merging

LoRA weights saved during training cannot be merged into the ZeRO-3 model at runtime, as ZeRO-3 partitions model weights across data parallel ranks. To merge LoRA weights into the base model, you can do so offline to obtain a merged weight file. Run `merge_lora_weight.sh` to merge the LoRA and base model weights. The parameters are:

- `--base_model_path`: Directory of the base model weights
- `--adapter_model_path`: Directory of the LoRA weights
- `--output_path`: Directory to save the merged weights
- `--save_dtype`: Data type for saving the merged weights; options are: fp16, bf16, fp32

#### LLaMA-Factory Training

If you are familiar with LLaMA-Factory, you may use it for fine-tuning. All scripts, code, and configuration files are archived in the `train/llama_factory_support` directory. Unless otherwise specified, all files mentioned below are located in this directory.

##### Installation

You can install LLaMA-Factory by downloading the source code from https://github.com/hiyouga/LLaMA-Factory/tree/main and following the instructions on the website.

##### Training Scripts and Configuration Files

The configuration files and launch scripts for each model are as follows:

| Model | Full Fine-tuning Config | LoRA Fine-tuning Config | Launch Script |
|-------|------------------------|------------------------|---------------|
| Hy-MT2-1.8B (Dense) | `hy_dense_1_8b_full_sft.yaml` | `hy_dense_1_8b_lora_sft.yaml` | `bash train_lf_dense.sh` |
| Hy-MT2-7B (Dense) | `hy_dense_7b_full_sft.yaml` | `hy_dense_7b_lora_sft.yaml` | `YAML_FILE=hy_dense_7b_full_sft.yaml bash train_lf_dense.sh` |
| Hy-MT2-30B-A3B (MoE) | `hy_v3_full_sft.yaml` | `hy_v3_lora_sft.yaml` | `bash train_lf.sh` |

> **Tip**: The Dense model launch script `train_lf_dense.sh` uses `hy_dense_1_8b_full_sft.yaml` by default. You can specify other configuration files via the `YAML_FILE` environment variable.

Key parameters in the configuration files are as follows:

**Model:**

- `model_name_or_path`: Path to the Hy-MT HF format pre-trained model weights
- `trust_remote_code`: Whether to trust remote code; Hy-MT requires this to be set to `true`

**Training Method:**

- `stage`: Training stage, currently `sft` (supervised fine-tuning)
- `finetuning_type`: Fine-tuning type, either `full` (full fine-tuning) or `lora` (LoRA fine-tuning)
- `deepspeed`: DeepSpeed configuration file path; `ds_zero3_offload.json` is recommended for full fine-tuning, `ds_zero2_offload_lora.json` for LoRA fine-tuning

**LoRA Parameters (only effective during LoRA fine-tuning):**

- `lora_rank`: LoRA rank, default `64`
- `lora_alpha`: LoRA alpha coefficient, default `128`
- `lora_dropout`: LoRA dropout ratio, default `0.05`
- `lora_target`: Target modules for LoRA, default `q_proj,k_proj,v_proj,o_proj`

**Dataset:**

- `dataset_dir`: Dataset directory path
- `dataset`: Dataset name, must be registered in `dataset_info.json` under `dataset_dir`
- `template`: Chat template; Hy-MT2-1.8B uses `hy_dense_1_8b`, Hy-MT2-7B uses `hy_dense_7b`, Hy-MT2-30B-A3B uses `hy_v3`
- `cutoff_len`: Maximum sequence length; sequences exceeding this will be truncated. For full fine-tuning, can be set to `262144` (262K); for LoRA fine-tuning, `8192` is recommended to save memory
- `max_samples`: Maximum number of samples per dataset
- `overwrite_cache`: Whether to overwrite cached preprocessed datasets

**Output:**

- `output_dir`: Output directory where logs, TensorBoard files, and weights will be stored
- `logging_steps`: Number of steps between logging
- `save_steps`: Number of steps between saving checkpoints
- `plot_loss`: Whether to plot the training loss curve
- `overwrite_output_dir`: Whether to overwrite the existing output directory
- `save_only_model`: Whether to save only model weights (excluding optimizer states, etc.)
- `report_to`: Logging tool, options: `none`, `wandb`, `tensorboard`, `swanlab`, `mlflow`

**Training Hyperparameters:**

- `per_device_train_batch_size`: Batch size per GPU
- `gradient_accumulation_steps`: Gradient accumulation steps; `per_device_train_batch_size * gradient_accumulation_steps * dp_size` equals the global batch size
- `learning_rate`: Maximum learning rate; `1.0e-5` recommended for full fine-tuning, `2.0e-4` for LoRA fine-tuning
- `num_train_epochs`: Number of training epochs
- `lr_scheduler_type`: Learning rate scheduler type; `cosine_with_min_lr` is recommended
- `lr_scheduler_kwargs.min_lr_rate`: Ratio of minimum to maximum learning rate; e.g., `0.1` means the minimum learning rate is 10% of the maximum
- `warmup_ratio`: Proportion of total training steps used for warmup
- `bf16`: Whether to use BFloat16 mixed precision training
- `gradient_checkpointing`: Whether to enable gradient checkpointing to save memory
- `ddp_timeout`: Distributed training timeout (milliseconds)
- `flash_attn`: Attention implementation; `fa2` (FlashAttention-2) is recommended, `sdpa` is also available; using `fa2` requires the flash-attn package
- `resume_from_checkpoint`: Resume training from a specified checkpoint path; set to `null` to start from scratch

##### Launch Training

For multi-machine training, please first complete the configuration in [Configure Passwordless SSH Login Between Machines](#configure-passwordless-ssh-login-between-machines-multi-machine-training) (single-machine training can skip this step).

Modify the following configuration at the beginning of the corresponding launch script:

```shell
export HOST_GPU_NUM=8
# IP list, comma separated. e.g. "192.168.1.1,192.168.1.2" or single node "192.168.1.1"
export IP_LIST=${IP_LIST:-"127.0.0.1"}
```

Note: If the `IP_LIST` environment variable is not set, replace `IP_LIST` with the IP list! The format is:
```
For a single IP:
IP_LIST=${ip_1}

For multiple IPs:
IP_LIST=${ip_1},${ip_2}

```

Replace `${ip_1}` and `${ip_2}` with the actual IP addresses.

Then, on each machine, run the corresponding launch script in the `train/llama_factory_support/` directory. For example:

```sh
# Dense 1.8B full fine-tuning
bash train_lf_dense.sh

# Dense 7B LoRA fine-tuning
YAML_FILE=hy_dense_7b_lora_sft.yaml bash train_lf_dense.sh

# MoE 30B-A3B full fine-tuning
bash train_lf.sh
```