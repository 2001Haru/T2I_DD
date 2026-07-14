#!/usr/bin/env bash
set -euo pipefail

RUN_ID="${RUN_ID:?Set RUN_ID to the completed kappa-cap run directory name.}"
SPEC="${SPEC:-imageB}"
IPC="${IPC:-10}"
N_NEIGHBORS="${N_NEIGHBORS:-85}"
MIN_CLUSTER_SIZE="${MIN_CLUSTER_SIZE:-55}"
SAMPLE_STEP="${SAMPLE_STEP:-25}"
DF="${DF:-1.0}"
GTP="${GTP:-0.9}"
GAMMA="${GAMMA:-0.05}"
KAPPA_CAP="${KAPPA_CAP:-0.3}"
GENERATION_SEEDS="${GENERATION_SEEDS:-0 1}"
EVAL_SEED="${EVAL_SEED:-0}"
IMAGENET_VAL_FOLDER="${IMAGENET_VAL_FOLDER:-/linxi/dataset/imagenet/validation/val}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export CUDA_VISIBLE_DEVICES

EXPERIMENT_DIR="./results/${SPEC}/Step-${SAMPLE_STEP}/IPC-${IPC}/DF-${DF}-GTP-${GTP}-gamma-${GAMMA}/n_${N_NEIGHBORS}_s_${MIN_CLUSTER_SIZE}"
RUN_DIR="${EXPERIMENT_DIR}/kappa_cap_runs/${RUN_ID}"
SAVE_ROOT="./trained_results/kappa_cap_runs/${SPEC}/${RUN_ID}"
cap_tag="$(printf '%s' "$KAPPA_CAP" | tr '.' 'p')"
METHOD="v1_focused_kappa_cap_${cap_tag}"

for generation_seed in $GENERATION_SEEDS; do
    train_dir="${RUN_DIR}/seed_${generation_seed}/generated_images_${METHOD}"
    save_dir="${SAVE_ROOT}/seed_${generation_seed}/${METHOD}-resnet_ap"
    timing_file="${RUN_DIR}/seed_${generation_seed}/timings/${METHOD}.json"
    if [[ ! -d "$train_dir" ]]; then
        echo "Missing generated dataset: $train_dir" >&2
        exit 1
    fi
    PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python python ./test/train.py \
        --dataset_dir "$train_dir" "$IMAGENET_VAL_FOLDER" \
        -d imagenet --spec "$SPEC" --nclass 10 --size 256 --ipc "$IPC" \
        -n resnet_ap --depth 10 --save-dir "$save_dir" \
        --seed "$EVAL_SEED" --workers 12 \
        --n_neighbors "$N_NEIGHBORS" --min_cluster_size "$MIN_CLUSTER_SIZE" \
        --timing_file "$timing_file" --experiment_method "$METHOD" \
        --tag "kappa_cap_${cap_tag}_gen_seed_${generation_seed}"
done

echo "All kappa-cap downstream runs completed: ${SAVE_ROOT}"
