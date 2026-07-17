#!/usr/bin/env bash
set -euo pipefail

export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION="${PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION:-python}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"

RUN_ID="${CLASS_ORACLE_RUN_ID:?Set CLASS_ORACLE_RUN_ID to a completed oracle build.}"
FINAL_CONTROL_RUN_ID="${FINAL_CONTROL_RUN_ID:-final_prompt_controls_v0}"
IMAGEA_REFERENCE_RUN_ID="${IMAGEA_REFERENCE_RUN_ID:-imageA_multiview_v0}"
IMAGENET_VAL_FOLDER="${IMAGENET_VAL_FOLDER:-/linxi/dataset/imagenet/validation/val}"
IMAGEA_SEED="${IMAGEA_SEED:-0}"
IMAGEB_SEED="${IMAGEB_SEED:-1}"
EVAL_SEED="${EVAL_SEED:-0}"
IPC="${IPC:-10}"
N_NEIGHBORS="${N_NEIGHBORS:-85}"
MIN_CLUSTER_SIZE="${MIN_CLUSTER_SIZE:-55}"

RUN_ROOT="./results/class_oracle_runs/${RUN_ID}"
SAVE_ROOT="./trained_results/class_oracle_runs/${RUN_ID}"
FINAL_TRAIN_ROOT="./trained_results/final_prompt_controls/${FINAL_CONTROL_RUN_ID}"
A_REFERENCE_TRAIN="./trained_results/multiview_caption_runs/imageA/${IMAGEA_REFERENCE_RUN_ID}/seed_${IMAGEA_SEED}"

A_REAL_RESULT="${FINAL_TRAIN_ROOT}/imageA/seed_${IMAGEA_SEED}/real_representative-resnet_ap/per_class_accuracy_all_seeds.json"
A_DIFFUSION_RESULT="${A_REFERENCE_TRAIN}/coda_baseline-resnet_ap/per_class_accuracy_all_seeds.json"
B_REAL_RESULT="${FINAL_TRAIN_ROOT}/imageB/seed_${IMAGEB_SEED}/real_representative-resnet_ap/per_class_accuracy_all_seeds.json"
B_DIFFUSION_RESULT="${FINAL_TRAIN_ROOT}/imageB/seed_${IMAGEB_SEED}/class_prompt-resnet_ap/per_class_accuracy_all_seeds.json"

for required in \
    "${RUN_ROOT}/imageA/class_oracle/oracle_manifest.json" \
    "${RUN_ROOT}/imageB/class_oracle/oracle_manifest.json" \
    "$A_REAL_RESULT" "$A_DIFFUSION_RESULT" "$B_REAL_RESULT" "$B_DIFFUSION_RESULT"; do
    if [[ ! -e "$required" ]]; then
        echo "Required class-oracle artifact was not found: ${required}" >&2
        exit 1
    fi
done
mkdir -p "$SAVE_ROOT" "${RUN_ROOT}/timings"

train_variant() {
    local spec=$1
    local dataset_dir="${RUN_ROOT}/${spec}/class_oracle"
    local save_dir="${SAVE_ROOT}/${spec}/class_oracle-resnet_ap"
    local result_file="${save_dir}/per_class_accuracy_all_seeds.json"
    if [[ -f "$result_file" ]]; then
        echo "==> Reusing completed class-oracle classifier: ${spec}"
        return
    fi
    if [[ -e "$save_dir" || -e "${save_dir}_gpu0" || -e "${save_dir}_gpu1" ]]; then
        echo "Incomplete class-oracle classifier output exists; refusing to mix runs: ${save_dir}" >&2
        exit 1
    fi
    echo "==> Training ${spec}/class_oracle"
    python ./test/train.py \
        --dataset_dir "$dataset_dir" "$IMAGENET_VAL_FOLDER" \
        -d imagenet --spec "$spec" --nclass 10 --size 256 --ipc "$IPC" \
        -n resnet_ap --depth 10 --save-dir "$save_dir" \
        --seed "$EVAL_SEED" --workers 12 \
        --n_neighbors "$N_NEIGHBORS" --min_cluster_size "$MIN_CLUSTER_SIZE" \
        --timing_file "${RUN_ROOT}/timings/${spec}_class_oracle.json" \
        --experiment_method class_oracle --tag "class_oracle_${RUN_ID}"
}

train_variant imageA
train_variant imageB

SUMMARY_DIR="${SAVE_ROOT}/summary"
if [[ ! -e "$SUMMARY_DIR" ]]; then
    python summarize_class_oracle.py \
        --output_dir "$SUMMARY_DIR" \
        --input imageA "${RUN_ROOT}/imageA/class_oracle/oracle_manifest.json" \
            "$A_REAL_RESULT" "$A_DIFFUSION_RESULT" \
            "${SAVE_ROOT}/imageA/class_oracle-resnet_ap/per_class_accuracy_all_seeds.json" \
        --input imageB "${RUN_ROOT}/imageB/class_oracle/oracle_manifest.json" \
            "$B_REAL_RESULT" "$B_DIFFUSION_RESULT" \
            "${SAVE_ROOT}/imageB/class_oracle-resnet_ap/per_class_accuracy_all_seeds.json"
else
    echo "==> Reusing existing class-oracle summary: ${SUMMARY_DIR}"
fi

echo "Class-oracle training completed: ${SAVE_ROOT}"
