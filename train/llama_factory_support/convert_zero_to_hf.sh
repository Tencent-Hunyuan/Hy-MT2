#!/bin/bash
# 将 DeepSpeed ZeRO 格式的 checkpoint 转换为 HuggingFace 格式
# 使用 zero_to_fp32.py 转换权重，然后保存为 HF 格式

set -e  # 遇到错误立即退出

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"

# 激活 conda 环境
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate llama_factory

# 设置环境变量
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
export DISABLE_VERSION_CHECK=1
export CUDA_VISIBLE_DEVICES=""  # 使用 CPU 进行转换，避免显存不足

CHECKPOINT_DIR="$SCRIPT_DIR/saves/hy_v3/full/sft/checkpoint-39"
OUTPUT_DIR="$SCRIPT_DIR/saves/hy_v3/full/sft/checkpoint-39/hf_converted"
TEMP_WEIGHTS_DIR="$OUTPUT_DIR/zero_fp32_output"  # 分片输出目录

echo "=========================================="
echo "Converting DeepSpeed ZeRO checkpoint to HF format"
echo "Input:  $CHECKPOINT_DIR"
echo "Output: $OUTPUT_DIR"
echo "=========================================="

# 创建输出目录
mkdir -p "$OUTPUT_DIR"
rm -rf "$TEMP_WEIGHTS_DIR"
mkdir -p "$TEMP_WEIGHTS_DIR"

# Step 1: 使用 zero_to_fp32.py 转换权重
echo ""
echo "[Step 1/3] Converting weights from ZeRO format to FP32..."
cd "$CHECKPOINT_DIR"
python3 zero_to_fp32.py . "$TEMP_WEIGHTS_DIR"

# 检查输出 - zero_to_fp32.py 可能输出单个文件或多个分片
if [ -d "$TEMP_WEIGHTS_DIR" ] && [ "$(ls -A "$TEMP_WEIGHTS_DIR" 2>/dev/null)" ]; then
    echo "Weight conversion completed! Output in: $TEMP_WEIGHTS_DIR"
    echo "Files: $(ls "$TEMP_WEIGHTS_DIR" | wc -l) files"
else
    echo "ERROR: Weight conversion failed!"
    exit 1
fi

# Step 2: 复制配置文件
echo ""
echo "[Step 2/3] Copying config files..."
BASE_MODEL_DIR="$PROJECT_ROOT/a3b_ckpt"

# 从基座模型复制完整的配置文件（checkpoint 中的 tokenizer_config 不完整）
cp "$BASE_MODEL_DIR/config.json" "$OUTPUT_DIR/" 2>/dev/null || true
cp "$BASE_MODEL_DIR/tokenizer_config.json" "$OUTPUT_DIR/" 2>/dev/null || true
cp "$BASE_MODEL_DIR/tokenizer.json" "$OUTPUT_DIR/" 2>/dev/null || true
cp "$BASE_MODEL_DIR/special_tokens_map.json" "$OUTPUT_DIR/" 2>/dev/null || true
cp "$BASE_MODEL_DIR/chat_template.jinja" "$OUTPUT_DIR/" 2>/dev/null || true
cp "$CHECKPOINT_DIR/generation_config.json" "$OUTPUT_DIR/" 2>/dev/null || true

echo "Config files copied from base model."

# Step 3: 加载权重并保存为 HF 格式
echo ""
echo "[Step 3/3] Converting to HuggingFace format..."

cat > /tmp/convert_to_hf.py << 'PYEOF'
import torch
import json
import os
import sys
import glob
sys.path.insert(0, os.environ.get("PROJECT_ROOT", "."))

# 设置目录
output_dir = os.environ["OUTPUT_DIR"]
checkpoint_dir = os.environ["CHECKPOINT_DIR"]
base_model_dir = os.environ["BASE_MODEL_DIR"]
temp_weights_dir = os.path.join(output_dir, "zero_fp32_output")

# 加载 tokenizer
from transformers import AutoTokenizer
print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(base_model_dir, trust_remote_code=True)
tokenizer.save_pretrained(output_dir)

# 从 base model 加载配置和模型结构
from transformers import AutoConfig, AutoModelForCausalLM
print("Loading config...")
config = AutoConfig.from_pretrained(base_model_dir, trust_remote_code=True)

# 创建模型（从基座模型加载结构和权重，然后用训练后的权重覆盖）
print("Loading base model...")
with torch.no_grad():
    model = AutoModelForCausalLM.from_pretrained(
        base_model_dir,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )

# 加载转换后的权重（可能分片）
print(f"Loading weights from {temp_weights_dir} ...")
weight_files = sorted(glob.glob(os.path.join(temp_weights_dir, "*.bin")) + 
                      glob.glob(os.path.join(temp_weights_dir, "*.safetensors")))

if not weight_files:
    print(f"ERROR: No weight files found in {temp_weights_dir}")
    sys.exit(1)

print(f"Found {len(weight_files)} weight files")
state_dict = {}
for wf in weight_files:
    print(f"  Loading {wf} ...")
    if wf.endswith('.safetensors'):
        from safetensors.torch import load_file
        state_dict.update(load_file(wf, device="cpu"))
    else:
        state_dict.update(torch.load(wf, map_location="cpu"))

# 加载权重到模型
print("Loading converted weights into model...")
model_state_dict = model.state_dict()
filtered_state_dict = {}
skipped = 0
matched = 0
for k, v in state_dict.items():
    if k in model_state_dict:
        # 转换 dtype
        if v.dtype != model_state_dict[k].dtype:
            v = v.to(model_state_dict[k].dtype)
        filtered_state_dict[k] = v
        matched += 1
    else:
        skipped += 1
        if skipped <= 10:  # 只打印前10个跳过的key
            print(f"  Skipping key: {k}")

print(f"Matched {matched} tensors, skipped {skipped}")

if matched == 0:
    print("ERROR: No weights matched! Something is wrong with the conversion.")
    sys.exit(1)

missing, unexpected = model.load_state_dict(filtered_state_dict, strict=False)
if missing:
    print(f"WARNING: {len(missing)} keys missing in converted weights (using base model weights)")
    for k in missing[:10]:
        print(f"  Missing: {k}")
    if len(missing) > 10:
        print(f"  ... and {len(missing) - 10} more")

# 保存为 HF 格式（使用 safetensors，更安全、更快）
print(f"Saving model to {output_dir}...")
model.save_pretrained(output_dir, safe_serialization=True)
print("Done!")

# 清理临时文件
print("Cleaning up temp files...")
import shutil
shutil.rmtree(temp_weights_dir)
PYEOF

export OUTPUT_DIR="$OUTPUT_DIR"
export CHECKPOINT_DIR="$CHECKPOINT_DIR"
export BASE_MODEL_DIR="$PROJECT_ROOT/a3b_ckpt"
export PROJECT_ROOT="$PROJECT_ROOT"

python3 /tmp/convert_to_hf.py

echo ""
echo "=========================================="
echo "Conversion completed!"
echo "HF format model saved to: $OUTPUT_DIR"
echo "=========================================="
