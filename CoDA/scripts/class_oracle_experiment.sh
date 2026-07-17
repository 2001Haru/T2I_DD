#!/usr/bin/env bash
set -euo pipefail

export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION="${PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION:-python}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"

RUN_ID="${CLASS_ORACLE_RUN_ID:-class_oracle_$(date -u +%Y%m%dT%H%M%SZ)}"
FINAL_CONTROL_RUN_ID="${FINAL_CONTROL_RUN_ID:-final_prompt_controls_v0}"
IMAGEA_REFERENCE_RUN_ID="${IMAGEA_REFERENCE_RUN_ID:-imageA_multiview_v0}"
IMAGEA_SEED="${IMAGEA_SEED:-0}"
IMAGEB_SEED="${IMAGEB_SEED:-1}"
IPC="${IPC:-10}"
N_NEIGHBORS="${N_NEIGHBORS:-85}"
MIN_CLUSTER_SIZE="${MIN_CLUSTER_SIZE:-55}"
SAMPLE_STEP="${SAMPLE_STEP:-25}"
DF="${DF:-1.0}"
GTP="${GTP:-0.9}"
GAMMA="${GAMMA:-0.05}"
RUN_DOWNSTREAM_TRAINING="${RUN_DOWNSTREAM_TRAINING:-true}"

experiment_dir() {
    local spec=$1
    echo "./results/${spec}/Step-${SAMPLE_STEP}/IPC-${IPC}/DF-${DF}-GTP-${GTP}-gamma-${GAMMA}/n_${N_NEIGHBORS}_s_${MIN_CLUSTER_SIZE}"
}

A_EXPERIMENT_DIR="$(experiment_dir imageA)"
B_EXPERIMENT_DIR="$(experiment_dir imageB)"
A_REFERENCE_DATA="${A_EXPERIMENT_DIR}/multiview_caption_runs/${IMAGEA_REFERENCE_RUN_ID}/seed_${IMAGEA_SEED}"
A_REFERENCE_TRAIN="./trained_results/multiview_caption_runs/imageA/${IMAGEA_REFERENCE_RUN_ID}/seed_${IMAGEA_SEED}"
FINAL_TRAIN_ROOT="./trained_results/final_prompt_controls/${FINAL_CONTROL_RUN_ID}"
B_FINAL_DATA="${B_EXPERIMENT_DIR}/final_prompt_controls/${FINAL_CONTROL_RUN_ID}/seed_${IMAGEB_SEED}"

A_REAL_DATA="${A_EXPERIMENT_DIR}/real_images"
A_DIFFUSION_DATA="${A_REFERENCE_DATA}/generated_images_coda_baseline"
A_REAL_RESULT="${FINAL_TRAIN_ROOT}/imageA/seed_${IMAGEA_SEED}/real_representative-resnet_ap/per_class_accuracy_all_seeds.json"
A_DIFFUSION_RESULT="${A_REFERENCE_TRAIN}/coda_baseline-resnet_ap/per_class_accuracy_all_seeds.json"
B_REAL_DATA="${B_EXPERIMENT_DIR}/real_images"
B_DIFFUSION_DATA="${B_FINAL_DATA}/generated_images_class_prompt"
B_REAL_RESULT="${FINAL_TRAIN_ROOT}/imageB/seed_${IMAGEB_SEED}/real_representative-resnet_ap/per_class_accuracy_all_seeds.json"
B_DIFFUSION_RESULT="${FINAL_TRAIN_ROOT}/imageB/seed_${IMAGEB_SEED}/class_prompt-resnet_ap/per_class_accuracy_all_seeds.json"

RUN_ROOT="./results/class_oracle_runs/${RUN_ID}"
TRAIN_ROOT="./trained_results/class_oracle_runs/${RUN_ID}"
for output in "$RUN_ROOT" "$TRAIN_ROOT"; do
    if [[ -e "$output" ]]; then
        echo "Refusing to overwrite class-oracle output: ${output}" >&2
        exit 1
    fi
done
for required in \
    "$A_REAL_DATA" "$A_DIFFUSION_DATA" "$A_REAL_RESULT" "$A_DIFFUSION_RESULT" \
    "$B_REAL_DATA" "$B_DIFFUSION_DATA" "$B_REAL_RESULT" "$B_DIFFUSION_RESULT"; do
    if [[ ! -e "$required" ]]; then
        echo "Required oracle endpoint artifact was not found: ${required}" >&2
        exit 1
    fi
done

mkdir -p "$RUN_ROOT"
python build_class_oracle_dataset.py \
    --spec imageA --real_dir "$A_REAL_DATA" --diffusion_dir "$A_DIFFUSION_DATA" \
    --real_result "$A_REAL_RESULT" --diffusion_result "$A_DIFFUSION_RESULT" \
    --output_dir "${RUN_ROOT}/imageA/class_oracle" --ipc "$IPC"
python build_class_oracle_dataset.py \
    --spec imageB --real_dir "$B_REAL_DATA" --diffusion_dir "$B_DIFFUSION_DATA" \
    --real_result "$B_REAL_RESULT" --diffusion_result "$B_DIFFUSION_RESULT" \
    --output_dir "${RUN_ROOT}/imageB/class_oracle" --ipc "$IPC"

if [[ "$RUN_DOWNSTREAM_TRAINING" == "true" ]]; then
    CLASS_ORACLE_RUN_ID="$RUN_ID" \
    FINAL_CONTROL_RUN_ID="$FINAL_CONTROL_RUN_ID" \
    IMAGEA_REFERENCE_RUN_ID="$IMAGEA_REFERENCE_RUN_ID" \
    IMAGEA_SEED="$IMAGEA_SEED" IMAGEB_SEED="$IMAGEB_SEED" \
    bash scripts/train_class_oracle_run.sh
fi

echo "Class-oracle experiment completed: ${RUN_ROOT}"
