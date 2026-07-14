#!/usr/bin/env bash
set -euo pipefail

SPEC="${SPEC:-imageA}"
RUN_ID="${RUN_ID:?Set RUN_ID to the completed multiview run directory name.}"
IMAGENET_VAL_FOLDER="${IMAGENET_VAL_FOLDER:-/linxi/dataset/imagenet/validation/val}"
IPC="${IPC:-10}"
N_NEIGHBORS="${N_NEIGHBORS:-85}"
MIN_CLUSTER_SIZE="${MIN_CLUSTER_SIZE:-55}"
SAMPLE_STEP="${SAMPLE_STEP:-25}"
DF="${DF:-1.0}"
GTP="${GTP:-0.9}"
GAMMA="${GAMMA:-0.05}"
GENERATION_SEEDS="${GENERATION_SEEDS:-0 1}"
EVAL_SEED="${EVAL_SEED:-0}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export CUDA_VISIBLE_DEVICES

EXPERIMENT_DIR="./results/${SPEC}/Step-${SAMPLE_STEP}/IPC-${IPC}/DF-${DF}-GTP-${GTP}-gamma-${GAMMA}/n_${N_NEIGHBORS}_s_${MIN_CLUSTER_SIZE}"
RUN_DIR="${EXPERIMENT_DIR}/multiview_caption_runs/${RUN_ID}"
SAVE_ROOT="./trained_results/multiview_caption_runs/${SPEC}/${RUN_ID}"
METHODS=(coda_baseline single_focused montage_common_mode)

if [[ ! -d "$RUN_DIR" ]]; then
    echo "Multiview generation run not found: ${RUN_DIR}" >&2
    exit 1
fi

for generation_seed in $GENERATION_SEEDS; do
    for method in "${METHODS[@]}"; do
        dataset_dir="${RUN_DIR}/seed_${generation_seed}/generated_images_${method}"
        if [[ ! -d "$dataset_dir" ]]; then
            echo "Generated dataset not found: ${dataset_dir}" >&2
            exit 1
        fi
        echo "==> Training ${method}, generation seed ${generation_seed}"
        PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python python ./test/train.py \
            --dataset_dir "$dataset_dir" "$IMAGENET_VAL_FOLDER" \
            -d imagenet --spec "$SPEC" --nclass 10 --size 256 --ipc "$IPC" \
            -n resnet_ap --depth 10 \
            --save-dir "${SAVE_ROOT}/seed_${generation_seed}/${method}-resnet_ap" \
            --seed "$EVAL_SEED" --workers 12 \
            --n_neighbors "$N_NEIGHBORS" --min_cluster_size "$MIN_CLUSTER_SIZE" \
            --timing_file "${RUN_DIR}/seed_${generation_seed}/timings/${method}.json" \
            --experiment_method "$method" --tag "multiview_gen_seed_${generation_seed}"
    done
done

python summarize_multiview_caption_results.py \
    --run_dir "$RUN_DIR" --trained_root "$SAVE_ROOT" \
    --generation_seeds $GENERATION_SEEDS

echo "All multiview downstream runs completed: ${SAVE_ROOT}"
