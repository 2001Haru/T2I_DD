#!/usr/bin/env bash
set -euo pipefail

export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION="${PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION:-python}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"

RUN_ID="${GRADIENT_SELECTION_RUN_ID:-gradient_selection_$(date -u +%Y%m%dT%H%M%SZ)}"
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
NEIGHBOR_COUNT="${NEIGHBOR_COUNT:-32}"
MODEL_SEEDS="${MODEL_SEEDS:-0,1,2,3}"
GM_AUGMENTATIONS="${GM_AUGMENTATIONS:-4}"
GM_BATCH_SIZE="${GM_BATCH_SIZE:-64}"
GM_WORKERS="${GM_WORKERS:-8}"
RANDOM_SELECTION_SEED="${RANDOM_SELECTION_SEED:-20260717}"
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

RUN_ROOT="./results/gradient_selection_runs/${RUN_ID}"
TRAIN_ROOT="./trained_results/gradient_selection_runs/${RUN_ID}"
for output in "$RUN_ROOT" "$TRAIN_ROOT"; do
    if [[ -e "$output" ]]; then
        echo "Refusing to overwrite gradient-selection output: ${output}" >&2
        exit 1
    fi
done
for required in \
    "$A_REAL_DATA" "$A_DIFFUSION_DATA" "$A_REAL_RESULT" "$A_DIFFUSION_RESULT" \
    "$B_REAL_DATA" "$B_DIFFUSION_DATA" "$B_REAL_RESULT" "$B_DIFFUSION_RESULT"; do
    if [[ ! -e "$required" ]]; then
        echo "Required candidate or accuracy artifact was not found: ${required}" >&2
        exit 1
    fi
done

mkdir -p "$RUN_ROOT"
echo "Gradient-selection run: ${RUN_ID}"

run_selection() {
    local spec=$1
    local gpu=$2
    local real_data=$3
    local diffusion_data=$4
    local real_result=$5
    local diffusion_result=$6
    local selection_seed="$RANDOM_SELECTION_SEED"
    if [[ "$spec" == "imageB" ]]; then
        selection_seed=$((RANDOM_SELECTION_SEED + 100000))
    fi
    python gradient_candidate_selection.py \
        --spec "$spec" --real_dir "$real_data" --diffusion_dir "$diffusion_data" \
        --real_accuracy "$real_result" --diffusion_accuracy "$diffusion_result" \
        --output_dir "${RUN_ROOT}/${spec}" --ipc "$IPC" \
        --n_neighbors "$N_NEIGHBORS" --min_cluster_size "$MIN_CLUSTER_SIZE" \
        --neighbor_count "$NEIGHBOR_COUNT" --neighbor_metric standardized_l2 \
        --neighbor_weighting uniform --model_seeds "$MODEL_SEEDS" \
        --augmentations "$GM_AUGMENTATIONS" --random_selection_seed "$selection_seed" \
        --batch_size "$GM_BATCH_SIZE" --workers "$GM_WORKERS" --device "cuda:${gpu}"
}

echo "==> Computing ImageA local gradient signal on GPU 0"
run_selection imageA 0 "$A_REAL_DATA" "$A_DIFFUSION_DATA" "$A_REAL_RESULT" "$A_DIFFUSION_RESULT" \
    >"${RUN_ROOT}/imageA.log" 2>&1 &
pid_a=$!
echo "==> Computing ImageB local gradient signal on GPU 1"
run_selection imageB 1 "$B_REAL_DATA" "$B_DIFFUSION_DATA" "$B_REAL_RESULT" "$B_DIFFUSION_RESULT" \
    >"${RUN_ROOT}/imageB.log" 2>&1 &
pid_b=$!

set +e
wait "$pid_a"; status_a=$?
wait "$pid_b"; status_b=$?
set -e
if (( status_a != 0 || status_b != 0 )); then
    echo "Gradient computation failed (ImageA=${status_a}, ImageB=${status_b})." >&2
    echo "Inspect ${RUN_ROOT}/imageA.log and ${RUN_ROOT}/imageB.log." >&2
    exit 1
fi

if [[ "$RUN_DOWNSTREAM_TRAINING" == "true" ]]; then
    GRADIENT_SELECTION_RUN_ID="$RUN_ID" \
    FINAL_CONTROL_RUN_ID="$FINAL_CONTROL_RUN_ID" \
    IMAGEA_REFERENCE_RUN_ID="$IMAGEA_REFERENCE_RUN_ID" \
    IMAGEA_SEED="$IMAGEA_SEED" IMAGEB_SEED="$IMAGEB_SEED" \
    bash scripts/train_gradient_selection_run.sh
fi

echo "Gradient-selection experiment completed: ${RUN_ROOT}"
