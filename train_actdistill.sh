#!/usr/bin/env bash
# ActDistill Training Orchestrator
# Usage:
#   bash train_actdistill.sh DATA_ROOT PRETRAINED_CHECKPOINT RUN_ROOT BALANCE_WEIGHT [SEED] [GPU_IDS]
#
# Example:
#   bash train_actdistill.sh /data/bridge manifest/pretrained.pt runs/actdistill 0.3 43 0,1

set -euo pipefail

if [ "$#" -lt 4 ]; then
    echo "Usage: $0 DATA_ROOT PRETRAINED_CHECKPOINT RUN_ROOT BALANCE_WEIGHT [SEED] [GPU_IDS]"
    exit 1
fi

DATA_ROOT=$1
PRETRAINED_CHECKPOINT=$2
RUN_ROOT=$3
BALANCE_WEIGHT=$4
SEED=${5:-42}
GPU_IDS=${6:-0}

PRIMARY_GPU=${GPU_IDS%%,*}
WORLD_SIZE=$(echo "$GPU_IDS" | awk -F',' '{print NF}')
TEACHER_OUTPUT_DIR="${RUN_ROOT}/actdistill_stage1"
TEACHER_CACHE_DIR="${TEACHER_OUTPUT_DIR}/teacher_outputs_cache"

mkdir -p "$RUN_ROOT"

echo "[ActDistill] data_root      : $DATA_ROOT"
echo "[ActDistill] pretrained_ckpt: $PRETRAINED_CHECKPOINT"
echo "[ActDistill] run_root       : $RUN_ROOT"
echo "[ActDistill] stage1 cache   : $TEACHER_OUTPUT_DIR"
echo "[ActDistill] balance weight : $BALANCE_WEIGHT"
echo "[ActDistill] seed           : $SEED"
echo "[ActDistill] gpu ids        : $GPU_IDS (world size = $WORLD_SIZE)"

# ActDistill hyper-parameters (matches paper defaults)
ALPHA=1.0
BETA=1.0
ETA=0.5
GAMMA=2.0
SEMANTIC_DIM=512
GNN_TYPE="GAT"

export USE_ACTDISTILL=TRUE
export ALPHA=${ALPHA}
export BETA=${BETA}
export ETA=${ETA}
export GAMMA=${GAMMA}
export SEMANTIC_DIM=${SEMANTIC_DIM}
export GNN_TYPE=${GNN_TYPE}
export BALANCE=${BALANCE_WEIGHT}

run_stage1() {
    echo "========================================="
    echo "Stage 1: Teacher Semantic Probing"
    echo "========================================="
    CUDA_VISIBLE_DEVICES=${PRIMARY_GPU} python scripts/train_teacher_probing.py \
        --teacher_checkpoint "${PRETRAINED_CHECKPOINT}" \
        --data_root_dir "${DATA_ROOT}" \
        --output_dir "${TEACHER_OUTPUT_DIR}" \
        --epochs 1 \
        --lr 1e-6 \
        --batch_size 64 \
        --device cuda:0
}

run_stage2() {
    echo "========================================="
    echo "Stage 2: Student Distillation with ActDistill"
    echo "========================================="
    CUDA_VISIBLE_DEVICES=${GPU_IDS} torchrun --standalone --nnodes 1 --nproc-per-node ${WORLD_SIZE} scripts/train.py \
        --vla.type prism-dinosiglip-224px+oxe+diffusion \
        --vla.expected_world_size ${WORLD_SIZE} \
        --vla.global_batch_size 128 \
        --vla.per_device_batch_size $((128 / WORLD_SIZE)) \
        --vla.learning_rate 1e-6 \
        --vla.epochs 5 \
        --vla.freeze_vision_backbone true \
        --vla.freeze_llm_backbone false \
        --data_root_dir "${DATA_ROOT}" \
        --run_root_dir "${RUN_ROOT}" \
        --run_id "actdistill_seed${SEED}" \
        --image_aug false \
        --save_interval 1000 \
        --action_dim 7 \
        --repeated_diffusion_steps 8 \
        --future_action_window_size 15 \
        --load_dit true \
        --action_model_type DiT-B \
        --is_resume False \
        --pretrained_checkpoint "${PRETRAINED_CHECKPOINT}" \
        --teacher_cache_dir "${TEACHER_CACHE_DIR}"
}

if [ ! -f "${TEACHER_OUTPUT_DIR}/semantic_heads_final.pth" ]; then
    run_stage1
else
    echo "[Stage 1] cached outputs found at ${TEACHER_OUTPUT_DIR}, skipping."
fi

run_stage2

