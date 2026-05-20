#!/bin/bash

NET_TYPE="high"
export NCCL_DEBUG=WARN
export NCCL_P2P_LEVEL=NVL
export NCCL_IB_TIMEOUT=24
export NCCL_NVLS_ENABLE=0
export NCCL_MPI_PROFILE_PRIMS_ENABLE=0
export CUDA_DEVICE_MAX_CONNECTIONS=1
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=3600
if [[ "${NET_TYPE}" = "low" ]]; then
    export NCCL_SOCKET_IFNAME=eth1
    export NCCL_IB_GID_INDEX=3
    export NCCL_IB_HCA=mlx5_2:1
    export NCCL_IB_SL=3
    export NCCL_CHECK_DISABLE=1
    export NCCL_P2P_DISABLE=0
    export NCCL_LL_THRESHOLD=16384
    export NCCL_IB_CUDA_SUPPORT=1
else
    export NCCL_IB_GID_INDEX=3
    export NCCL_IB_SL=3
    export NCCL_CHECK_DISABLE=1
    export NCCL_P2P_DISABLE=0
    export NCCL_IB_DISABLE=0
    export NCCL_LL_THRESHOLD=16384
    export NCCL_IB_CUDA_SUPPORT=1
    export NCCL_SOCKET_IFNAME=bond1
    export UCX_NET_DEVICES=bond1
    export NCCL_IB_HCA=mlx5_bond_1,mlx5_bond_5,mlx5_bond_3,mlx5_bond_7,mlx5_bond_4,mlx5_bond_8,mlx5_bond_2,mlx5_bond_6
    export NCCL_COLLNET_ENABLE=0
    export SHARP_COLL_ENABLE_SAT=0
    export NCCL_NET_GDR_LEVEL=2
    export NCCL_IB_QPS_PER_CONNECTION=4
    export NCCL_IB_TC=160
    export NCCL_PXN_DISABLE=1
fi

export HOST_GPU_NUM=8
# IP list, comma separated. e.g. "192.168.1.1,192.168.1.2" or single node "192.168.1.1"
IP_LIST=${IP_LIST:-"127.0.0.1"}

IFS=',' read -ra IP_ARRAY <<< "$IP_LIST"
export NODES=${#IP_ARRAY[@]}
export LOCAL_IP=${IP_ARRAY[0]}
NODE_IP_LIST=""
for ip in "${IP_ARRAY[@]}"; do
    if [ -n "$NODE_IP_LIST" ]; then
        NODE_IP_LIST="${NODE_IP_LIST},"
    fi
    NODE_IP_LIST="${NODE_IP_LIST}${ip}:${HOST_GPU_NUM}"
done
export NODE_IP_LIST
export NODE_NUM=$((${NODES} * ${HOST_GPU_NUM}))

SCRIPT_DIR=$(dirname "$0")

model_path=path_to_model_weight
tokenizer_path=../models
train_data_file=example_data.jsonl

# ds_config_file=ds_zero2_no_offload.json
# ds_config_file=ds_zero3_no_offload.json
# For LoRA, zero2_offload is recommended to save memory
ds_config_file=${SCRIPT_DIR}/ds_zero2_no_offload.json

output_path=/root/hf_train_output

mkdir -p ${output_path}

current_time=$(date "+%Y.%m.%d-%H.%M.%S")
log_file=${output_path}/"log_${current_time}.txt"

echo $NODE_IP_LIST > env.txt 2>&1
sed "s/:/ slots=/g" env.txt | sed "s/,/\n/g" >  "hostfile"
sed "s/:.//g" env.txt | sed "s/,/\n/g" >  "pssh.hosts"
export CHIEF_IP=$LOCAL_IP

if [ ${NODES} -gt 1 ]; then
    HOST_PATH=hostfile
    DS_ARGS="--hostfile=${HOST_PATH} --master_addr ${CHIEF_IP}"
else
    DS_ARGS=""
fi

echo "NODES: ${NODES}, LOCAL_IP: ${LOCAL_IP}, NODE_IP_LIST: ${NODE_IP_LIST}"

deepspeed ${DS_ARGS} \
    ${SCRIPT_DIR}/train.py \
    --do_train \
    --model_name_or_path ${model_path} \
    --tokenizer_name_or_path ${tokenizer_path} \
    --train_data_file ${train_data_file} \
    --deepspeed ${ds_config_file} \
    --output_dir ${output_path} \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --gradient_checkpointing \
    --lr_scheduler_type cosine_with_min_lr \
    --logging_steps 1 \
    --max_steps 200 \
    --save_steps 100 \
    --learning_rate 1e-5 \
    --min_lr 1e-6 \
    --warmup_ratio 0.01 \
    --save_strategy steps \
    --bf16 \
    --use_lora \
    --lora_rank 64 \
    --lora_alpha 128 \
    --lora_dropout 0.1 \
    --hidden_size 2048 \
    --intermediate_size 6912 \
    --model_max_length 8192 \
    --max_seq_length 8192 \
    --moe_topk 8 \
    --num_experts 128 \
    --moe_intermediate_size 768 \
    --moe_layer_num_skipped 1 \
    --num_attention_heads 32 \
    --num_key_value_heads 4 \
    --num_layers 48 \
    --use_mixed_mlp_moe \
    --num_shared_expert 1 \
    --use_qk_norm | tee ${log_file}
