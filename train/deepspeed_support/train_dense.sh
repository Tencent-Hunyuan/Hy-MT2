#!/bin/bash

# Unified Dense model full fine-tuning script
# Supports: 1.8B and 7B dense models
# Usage: bash train_dense.sh [1.8B|7B]
#   - 1.8B: 1x GPU (24GB+), DeepSpeed ZeRO-2 (no offload)
#   - 7B:   2x GPU (80GB+ each), DeepSpeed ZeRO-3 (no offload)

# ============== Model Size Selection ==============
MODEL_SIZE=${1:-"1.8B"}

if [[ "${MODEL_SIZE}" != "1.8B" && "${MODEL_SIZE}" != "7B" ]]; then
    echo "Error: MODEL_SIZE must be '1.8B' or '7B', got '${MODEL_SIZE}'"
    echo "Usage: bash train_dense.sh [1.8B|7B]"
    exit 1
fi

# ============== NCCL Configuration ==============
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

# ============== Model-specific Configuration ==============
SCRIPT_DIR=$(dirname "$0")

if [[ "${MODEL_SIZE}" == "1.8B" ]]; then
    export HOST_GPU_NUM=1
    model_path=path_to_dense_1_8b_model
    ds_config_file=${SCRIPT_DIR}/ds_zero2_no_offload.json
    output_path=./dense_1_8b_output
    HIDDEN_SIZE=2048
    INTERMEDIATE_SIZE=6144
    NUM_ATTENTION_HEADS=16
    NUM_KEY_VALUE_HEADS=4
    NUM_LAYERS=32
else
    export HOST_GPU_NUM=2
    model_path=path_to_dense_7b_model
    ds_config_file=${SCRIPT_DIR}/ds_zero3_no_offload.json
    output_path=./dense_7b_output
    HIDDEN_SIZE=4096
    INTERMEDIATE_SIZE=14336
    NUM_ATTENTION_HEADS=32
    NUM_KEY_VALUE_HEADS=8
    NUM_LAYERS=32
fi

tokenizer_path=${model_path}
train_data_file=../data/example_data.jsonl

# ============== Multi-node Configuration ==============
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

# ============== Output & Logging ==============
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

echo "============================================"
echo "Dense ${MODEL_SIZE} full fine-tuning"
echo "NODES: ${NODES}, LOCAL_IP: ${LOCAL_IP}, NODE_IP_LIST: ${NODE_IP_LIST}"
echo "DeepSpeed config: ${ds_config_file}"
echo "Model path: ${model_path}"
echo "Output path: ${output_path}"
echo "============================================"

# ============== Launch Training ==============
deepspeed ${DS_ARGS} \
    ${SCRIPT_DIR}/train_dense.py \
    --do_train \
    --model_size ${MODEL_SIZE} \
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
    --max_steps 30 \
    --save_steps 30 \
    --learning_rate 1e-5 \
    --min_lr 1e-6 \
    --warmup_ratio 0.01 \
    --save_strategy steps \
    --bf16 \
    --hidden_size ${HIDDEN_SIZE} \
    --intermediate_size ${INTERMEDIATE_SIZE} \
    --num_attention_heads ${NUM_ATTENTION_HEADS} \
    --num_key_value_heads ${NUM_KEY_VALUE_HEADS} \
    --num_layers ${NUM_LAYERS} \
    --model_max_length 4096 \
    --max_seq_length 4096 \
    --use_qk_norm | tee ${log_file}
