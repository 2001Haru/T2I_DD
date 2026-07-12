#!/usr/bin/env bash
set -euo pipefail

RUN_ID="${RUN_ID:?Set RUN_ID to the completed guidance run directory name.}"
SPEC="${SPEC:-imageA}"
IPC="${IPC:-10}"
N_NEIGHBORS="${N_NEIGHBORS:-85}"
MIN_CLUSTER_SIZE="${MIN_CLUSTER_SIZE:-55}"
SAMPLE_STEP="${SAMPLE_STEP:-25}"
DF="${DF:-1.0}"
GTP="${GTP:-0.9}"
GAMMA="${GAMMA:-0.05}"
EVAL_SEED="${EVAL_SEED:-0}"
IMAGENET_VAL_FOLDER="${IMAGENET_VAL_FOLDER:-/linxi/dataset/imagenet/validation/val}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export CUDA_VISIBLE_DEVICES

RUN_DIR="./results/${SPEC}/Step-${SAMPLE_STEP}/IPC-${IPC}/DF-${DF}-GTP-${GTP}-gamma-${GAMMA}/n_${N_NEIGHBORS}_s_${MIN_CLUSTER_SIZE}/guidance_conflict_runs/${RUN_ID}"
SAVE_ROOT="./trained_results/guidance_conflict_runs/${SPEC}/${RUN_ID}"

METHODS=(
    coda_baseline
    v0_generic_caption
    v1_class_focused_caption
)

for method in "${METHODS[@]}"; do
    train_dir="${RUN_DIR}/generated_images_${method}"
    save_dir="${SAVE_ROOT}/${method}-resnet_ap"
    if [[ ! -d "$train_dir" ]]; then
        echo "Missing generated dataset: $train_dir" >&2
        exit 1
    fi

    echo "==> Training downstream classifier: ${method}"
    PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python python ./test/train.py \
        --dataset_dir "$train_dir" "$IMAGENET_VAL_FOLDER" \
        -d imagenet --spec "$SPEC" --nclass 10 --size 256 --ipc "$IPC" \
        -n resnet_ap --depth 10 --save-dir "$save_dir" \
        --seed "$EVAL_SEED" --workers 12 \
        --n_neighbors "$N_NEIGHBORS" --min_cluster_size "$MIN_CLUSTER_SIZE" \
        --experiment_method "$method" --tag "guidance_run_${RUN_ID}"
done

echo "All downstream runs completed: ${SAVE_ROOT}"
