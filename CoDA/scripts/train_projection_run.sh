#!/usr/bin/env bash
set -euo pipefail

RUN_ID="${RUN_ID:?Set RUN_ID to the completed projection run directory name.}"
SPEC="${SPEC:-imageA}"
IPC="${IPC:-10}"
N_NEIGHBORS="${N_NEIGHBORS:-85}"
MIN_CLUSTER_SIZE="${MIN_CLUSTER_SIZE:-55}"
SAMPLE_STEP="${SAMPLE_STEP:-25}"
DF="${DF:-1.0}"
GTP="${GTP:-0.9}"
GAMMA="${GAMMA:-0.05}"
GENERATION_SEEDS="${GENERATION_SEEDS:-0 1}"
PROJECTION_ALPHAS="${PROJECTION_ALPHAS:-0.5 1.0}"
EVAL_SEED="${EVAL_SEED:-0}"
IMAGENET_VAL_FOLDER="${IMAGENET_VAL_FOLDER:-/linxi/dataset/imagenet/validation/val}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export CUDA_VISIBLE_DEVICES

EXPERIMENT_DIR="./results/${SPEC}/Step-${SAMPLE_STEP}/IPC-${IPC}/DF-${DF}-GTP-${GTP}-gamma-${GAMMA}/n_${N_NEIGHBORS}_s_${MIN_CLUSTER_SIZE}"
RUN_DIR="${EXPERIMENT_DIR}/projection_runs/${RUN_ID}"
SAVE_ROOT="./trained_results/projection_runs/${SPEC}/${RUN_ID}"

alpha_tag() {
    printf '%s' "$1" | tr '.' 'p'
}

for generation_seed in $GENERATION_SEEDS; do
    for alpha in $PROJECTION_ALPHAS; do
        tag="alpha_$(alpha_tag "$alpha")"
        method="v1_focused_projection_${tag}"
        train_dir="${RUN_DIR}/seed_${generation_seed}/generated_images_${method}"
        save_dir="${SAVE_ROOT}/seed_${generation_seed}/${method}-resnet_ap"

        if [[ ! -d "$train_dir" ]]; then
            echo "Missing generated dataset: $train_dir" >&2
            exit 1
        fi

        echo "==> Training ${method}, generation seed ${generation_seed}"
        PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python python ./test/train.py \
            --dataset_dir "$train_dir" "$IMAGENET_VAL_FOLDER" \
            -d imagenet --spec "$SPEC" --nclass 10 --size 256 --ipc "$IPC" \
            -n resnet_ap --depth 10 --save-dir "$save_dir" \
            --seed "$EVAL_SEED" --workers 12 \
            --n_neighbors "$N_NEIGHBORS" --min_cluster_size "$MIN_CLUSTER_SIZE" \
            --experiment_method "$method" --tag "projection_gen_seed_${generation_seed}"
    done
done

echo "All projection downstream runs completed: ${SAVE_ROOT}"
