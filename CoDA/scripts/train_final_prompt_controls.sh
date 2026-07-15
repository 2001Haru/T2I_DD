#!/usr/bin/env bash
set -euo pipefail

export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION="${PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION:-python}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"

RUN_ID="${RUN_ID:?Set RUN_ID to the completed final prompt-control run ID.}"
IMAGENET_VAL_FOLDER="${IMAGENET_VAL_FOLDER:-/linxi/dataset/imagenet/validation/val}"
IPC="${IPC:-10}"
N_NEIGHBORS="${N_NEIGHBORS:-85}"
MIN_CLUSTER_SIZE="${MIN_CLUSTER_SIZE:-55}"
SAMPLE_STEP="${SAMPLE_STEP:-25}"
DF="${DF:-1.0}"
GTP="${GTP:-0.9}"
GAMMA="${GAMMA:-0.05}"
IMAGEA_SEED="${IMAGEA_SEED:-0}"
IMAGEB_SEED="${IMAGEB_SEED:-1}"
EVAL_SEED="${EVAL_SEED:-0}"
IMAGEA_REFERENCE_RUN_ID="${IMAGEA_REFERENCE_RUN_ID:-imageA_multiview_v0}"

experiment_dir() {
    local spec=$1
    echo "./results/${spec}/Step-${SAMPLE_STEP}/IPC-${IPC}/DF-${DF}-GTP-${GTP}-gamma-${GAMMA}/n_${N_NEIGHBORS}_s_${MIN_CLUSTER_SIZE}"
}

A_EXPERIMENT_DIR="$(experiment_dir imageA)"
B_EXPERIMENT_DIR="$(experiment_dir imageB)"
A_RUN_DIR="${A_EXPERIMENT_DIR}/final_prompt_controls/${RUN_ID}/seed_${IMAGEA_SEED}"
B_RUN_DIR="${B_EXPERIMENT_DIR}/final_prompt_controls/${RUN_ID}/seed_${IMAGEB_SEED}"
SAVE_ROOT="./trained_results/final_prompt_controls/${RUN_ID}"
A_REFERENCE_TRAIN="./trained_results/multiview_caption_runs/imageA/${IMAGEA_REFERENCE_RUN_ID}/seed_${IMAGEA_SEED}"
A_CLASS_RESULT="${A_REFERENCE_TRAIN}/coda_baseline-resnet_ap/per_class_accuracy_all_seeds.json"
A_MONTAGE_RESULT="${A_REFERENCE_TRAIN}/montage_common_mode-resnet_ap/per_class_accuracy_all_seeds.json"

for required in "$A_RUN_DIR" "$B_RUN_DIR" "$A_CLASS_RESULT" "$A_MONTAGE_RESULT"; do
    if [[ ! -e "$required" ]]; then
        echo "Required final-control artifact was not found: ${required}" >&2
        exit 1
    fi
done

train_variant() {
    local spec=$1
    local seed=$2
    local method=$3
    local dataset_dir=$4
    local timing_file=$5
    local save_dir="${SAVE_ROOT}/${spec}/seed_${seed}/${method}-resnet_ap"
    local result_file="${save_dir}/per_class_accuracy_all_seeds.json"
    if [[ -f "$result_file" ]]; then
        echo "==> Reusing completed classifier result: ${spec}/${method}"
        return
    fi
    if [[ ! -d "$dataset_dir" ]]; then
        echo "Dataset for ${spec}/${method} was not found: ${dataset_dir}" >&2
        exit 1
    fi
    echo "==> Training ${spec}/${method}"
    python ./test/train.py \
        --dataset_dir "$dataset_dir" "$IMAGENET_VAL_FOLDER" \
        -d imagenet --spec "$spec" --nclass 10 --size 256 --ipc "$IPC" \
        -n resnet_ap --depth 10 --save-dir "$save_dir" \
        --seed "$EVAL_SEED" --workers 12 \
        --n_neighbors "$N_NEIGHBORS" --min_cluster_size "$MIN_CLUSTER_SIZE" \
        --timing_file "$timing_file" --experiment_method "$method" \
        --tag "final_prompt_control_seed_${seed}"
}

train_variant imageA "$IMAGEA_SEED" real_representative "${A_EXPERIMENT_DIR}/real_images" "${A_RUN_DIR}/timings/real_representative.json"
train_variant imageA "$IMAGEA_SEED" vae_reconstruction "${A_RUN_DIR}/vae_reconstruction" "${A_RUN_DIR}/timings/vae_reconstruction.json"
train_variant imageA "$IMAGEA_SEED" empty_prompt "${A_RUN_DIR}/generated_images_empty_prompt" "${A_RUN_DIR}/timings/empty_prompt.json"
train_variant imageA "$IMAGEA_SEED" generic_prompt "${A_RUN_DIR}/generated_images_generic_prompt" "${A_RUN_DIR}/timings/generic_prompt.json"

train_variant imageB "$IMAGEB_SEED" real_representative "${B_EXPERIMENT_DIR}/real_images" "${B_RUN_DIR}/timings/real_representative.json"
train_variant imageB "$IMAGEB_SEED" vae_reconstruction "${B_RUN_DIR}/vae_reconstruction" "${B_RUN_DIR}/timings/vae_reconstruction.json"
train_variant imageB "$IMAGEB_SEED" empty_prompt "${B_RUN_DIR}/generated_images_empty_prompt" "${B_RUN_DIR}/timings/empty_prompt.json"
train_variant imageB "$IMAGEB_SEED" generic_prompt "${B_RUN_DIR}/generated_images_generic_prompt" "${B_RUN_DIR}/timings/generic_prompt.json"
train_variant imageB "$IMAGEB_SEED" class_prompt "${B_RUN_DIR}/generated_images_class_prompt" "${B_RUN_DIR}/timings/class_prompt.json"
train_variant imageB "$IMAGEB_SEED" montage_caption "${B_RUN_DIR}/generated_images_montage_caption" "${B_RUN_DIR}/timings/montage_caption.json"

python summarize_final_prompt_controls.py \
    --output_dir "${SAVE_ROOT}/summary" \
    --result imageA real_representative "${SAVE_ROOT}/imageA/seed_${IMAGEA_SEED}/real_representative-resnet_ap/per_class_accuracy_all_seeds.json" \
    --result imageA vae_reconstruction "${SAVE_ROOT}/imageA/seed_${IMAGEA_SEED}/vae_reconstruction-resnet_ap/per_class_accuracy_all_seeds.json" \
    --result imageA empty_prompt "${SAVE_ROOT}/imageA/seed_${IMAGEA_SEED}/empty_prompt-resnet_ap/per_class_accuracy_all_seeds.json" \
    --result imageA generic_prompt "${SAVE_ROOT}/imageA/seed_${IMAGEA_SEED}/generic_prompt-resnet_ap/per_class_accuracy_all_seeds.json" \
    --result imageA class_prompt "$A_CLASS_RESULT" \
    --result imageA montage_caption "$A_MONTAGE_RESULT" \
    --result imageB real_representative "${SAVE_ROOT}/imageB/seed_${IMAGEB_SEED}/real_representative-resnet_ap/per_class_accuracy_all_seeds.json" \
    --result imageB vae_reconstruction "${SAVE_ROOT}/imageB/seed_${IMAGEB_SEED}/vae_reconstruction-resnet_ap/per_class_accuracy_all_seeds.json" \
    --result imageB empty_prompt "${SAVE_ROOT}/imageB/seed_${IMAGEB_SEED}/empty_prompt-resnet_ap/per_class_accuracy_all_seeds.json" \
    --result imageB generic_prompt "${SAVE_ROOT}/imageB/seed_${IMAGEB_SEED}/generic_prompt-resnet_ap/per_class_accuracy_all_seeds.json" \
    --result imageB class_prompt "${SAVE_ROOT}/imageB/seed_${IMAGEB_SEED}/class_prompt-resnet_ap/per_class_accuracy_all_seeds.json" \
    --result imageB montage_caption "${SAVE_ROOT}/imageB/seed_${IMAGEB_SEED}/montage_caption-resnet_ap/per_class_accuracy_all_seeds.json"

echo "Final prompt-control training completed: ${SAVE_ROOT}"
