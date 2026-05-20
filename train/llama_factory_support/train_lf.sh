#!/bin/bash
# ============================================================================
# LLaMA Factory training launch script for HYV3
#
# This script sets up the environment and launches training via torchrun.
#
# We use train_hy_v3.py as the entry point (not llamafactory-cli)
# because we need to inject HYV3-specific monkey-patches and register
# the hy_v3 chat template BEFORE LLaMA Factory starts.
# train_hy_v3.py directly calls run_exp() in each torchrun worker,
# ensuring all patches are active.
#
# Usage:
#   Single node:  bash train_lf.sh
#   Multi-node:   Run this script on EACH node with the same IP_LIST.
#                 IP_LIST="10.0.0.1,10.0.0.2" bash train_lf.sh
# ============================================================================

set -euo pipefail

# -------------------- Network Configuration --------------------
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

# Skip LLaMA Factory version check (we use a newer transformers branch)
export DISABLE_VERSION_CHECK=1

# -------------------- Node Configuration --------------------
export HOST_GPU_NUM=8
# IP list, comma separated. e.g. "10.0.0.1,10.0.0.2" or single node "127.0.0.1"
export IP_LIST=${IP_LIST:-"127.0.0.1"}

MASTER_PORT=${MASTER_PORT:-29500}

IFS=',' read -ra IP_ARRAY <<< "$IP_LIST"
NODES=${#IP_ARRAY[@]}
MASTER_ADDR=${IP_ARRAY[0]}

# -------------------- Paths --------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
YAML_FILE="${YAML_FILE:-${SCRIPT_DIR}/hy_v3_full_sft.yaml}"
ENTRY_SCRIPT="${SCRIPT_DIR}/train_hy_v3.py"

# -------------------- Distributed Environment --------------------
export MASTER_ADDR="${MASTER_ADDR}"
export MASTER_PORT="${MASTER_PORT}"
export NNODES="${NODES}"

if [ ${NODES} -gt 1 ]; then
    # Determine local node rank by matching local IP against IP_LIST
    LOCAL_IP=$(hostname -i | awk '{print $1}')
    NODE_RANK=0
    for i in "${!IP_ARRAY[@]}"; do
        if [[ "${IP_ARRAY[$i]}" == "${LOCAL_IP}" ]]; then
            NODE_RANK=$i
            break
        fi
    done
    export RANK="${NODE_RANK}"
else
    export RANK=0
fi

echo "============================================"
echo "  HYV3 LLaMA Factory Training"
echo "  Nodes: ${NNODES}, Rank: ${RANK}"
echo "  Master: ${MASTER_ADDR}:${MASTER_PORT}"
echo "  GPUs per node: ${HOST_GPU_NUM}"
echo "  Total GPUs: $((NODES * HOST_GPU_NUM))"
echo "============================================"

# -------------------- Launch --------------------
# We launch torchrun directly (instead of FORCE_TORCHRUN) so that each
# worker process runs train_hy_v3.py with all HYV3 patches applied.
torchrun \
    --nnodes "${NNODES}" \
    --node_rank "${RANK}" \
    --nproc_per_node "${HOST_GPU_NUM}" \
    --master_addr "${MASTER_ADDR}" \
    --master_port "${MASTER_PORT}" \
    "${ENTRY_SCRIPT}" "${YAML_FILE}"
