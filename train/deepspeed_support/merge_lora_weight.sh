SCRIPT_DIR=$(dirname "$0")
python3 ${SCRIPT_DIR}/merge_lora_weight.py \
    --base_model_path /xxx/hy_l_train/checkpoint-200 \
    --adapter_model_path /xxx/runs/hy_l_lora_train/checkpoint-200 \
    --output_path /xxx/ckpts/merged_hy_lora_weight \
    --save_dtype bf16