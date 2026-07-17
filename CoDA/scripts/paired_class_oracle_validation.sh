#!/usr/bin/env bash
set -euo pipefail

export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION="${PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION:-python}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"

CLASS_ORACLE_RUN_ID="${CLASS_ORACLE_RUN_ID:?Set CLASS_ORACLE_RUN_ID to the completed class-oracle run.}"
RUN_ID="${PAIRED_ORACLE_RUN_ID:-${CLASS_ORACLE_RUN_ID}_paired_$(date -u +%Y%m%dT%H%M%SZ)}"
FINAL_CONTROL_RUN_ID="${FINAL_CONTROL_RUN_ID:-final_prompt_controls_v0}"
IMAGEA_REFERENCE_RUN_ID="${IMAGEA_REFERENCE_RUN_ID:-imageA_multiview_v0}"
IMAGENET_VAL_FOLDER="${IMAGENET_VAL_FOLDER:-/linxi/dataset/imagenet/validation/val}"
IMAGEA_SEED="${IMAGEA_SEED:-0}"
IMAGEB_SEED="${IMAGEB_SEED:-1}"
EVAL_SEED_STARTS="${EVAL_SEED_STARTS:-2 4}"
RESUME_RUN="${RESUME_RUN:-false}"
IPC="${IPC:-10}"
N_NEIGHBORS="${N_NEIGHBORS:-85}"
MIN_CLUSTER_SIZE="${MIN_CLUSTER_SIZE:-55}"
SAMPLE_STEP="${SAMPLE_STEP:-25}"
DF="${DF:-1.0}"
GTP="${GTP:-0.9}"
GAMMA="${GAMMA:-0.05}"

experiment_dir() {
    local spec=$1
    echo "./results/${spec}/Step-${SAMPLE_STEP}/IPC-${IPC}/DF-${DF}-GTP-${GTP}-gamma-${GAMMA}/n_${N_NEIGHBORS}_s_${MIN_CLUSTER_SIZE}"
}

A_EXPERIMENT_DIR="$(experiment_dir imageA)"
B_EXPERIMENT_DIR="$(experiment_dir imageB)"
A_REFERENCE_DATA="${A_EXPERIMENT_DIR}/multiview_caption_runs/${IMAGEA_REFERENCE_RUN_ID}/seed_${IMAGEA_SEED}"
B_FINAL_DATA="${B_EXPERIMENT_DIR}/final_prompt_controls/${FINAL_CONTROL_RUN_ID}/seed_${IMAGEB_SEED}"
ORACLE_ROOT="./results/class_oracle_runs/${CLASS_ORACLE_RUN_ID}"
RUN_ROOT="./results/paired_class_oracle_runs/${RUN_ID}"
SAVE_ROOT="./trained_results/paired_class_oracle_runs/${RUN_ID}"

A_REAL_DATA="${A_EXPERIMENT_DIR}/real_images"
A_DIFFUSION_DATA="${A_REFERENCE_DATA}/generated_images_coda_baseline"
A_ORACLE_DATA="${ORACLE_ROOT}/imageA/class_oracle"
B_REAL_DATA="${B_EXPERIMENT_DIR}/real_images"
B_DIFFUSION_DATA="${B_FINAL_DATA}/generated_images_class_prompt"
B_ORACLE_DATA="${ORACLE_ROOT}/imageB/class_oracle"

read -r -a EVAL_SEED_START_ARRAY <<< "$EVAL_SEED_STARTS"
EVAL_SEED_STARTS="${EVAL_SEED_START_ARRAY[*]}"
IFS=',' read -r -a VISIBLE_GPU_ARRAY <<< "$CUDA_VISIBLE_DEVICES"
if [[ "${#VISIBLE_GPU_ARRAY[@]}" -ne 2 ]]; then
    echo "Paired validation expects exactly two visible GPUs; got CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}" >&2
    exit 1
fi

for required in \
    "$A_REAL_DATA" "$A_DIFFUSION_DATA" "$A_ORACLE_DATA/oracle_manifest.json" \
    "$B_REAL_DATA" "$B_DIFFUSION_DATA" "$B_ORACLE_DATA/oracle_manifest.json"; do
    if [[ ! -e "$required" ]]; then
        echo "Required paired-oracle artifact was not found: ${required}" >&2
        exit 1
    fi
done

if [[ -e "$RUN_ROOT" || -e "$SAVE_ROOT" ]]; then
    if [[ "$RESUME_RUN" != "true" ]]; then
        echo "Paired-oracle output already exists; set RESUME_RUN=true to reuse completed stages: ${RUN_ID}" >&2
        exit 1
    fi
else
    mkdir -p "$RUN_ROOT" "$SAVE_ROOT"
fi
mkdir -p "${RUN_ROOT}/timings"

CONFIG_FILE="${RUN_ROOT}/paired_run_config.txt"
CONFIG_CONTENT="CLASS_ORACLE_RUN_ID=${CLASS_ORACLE_RUN_ID}
FINAL_CONTROL_RUN_ID=${FINAL_CONTROL_RUN_ID}
IMAGEA_REFERENCE_RUN_ID=${IMAGEA_REFERENCE_RUN_ID}
IMAGEA_SEED=${IMAGEA_SEED}
IMAGEB_SEED=${IMAGEB_SEED}
EVAL_SEED_STARTS=${EVAL_SEED_STARTS}
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
if [[ -f "$CONFIG_FILE" ]]; then
    if [[ "$(<"$CONFIG_FILE")" != "$CONFIG_CONTENT" ]]; then
        echo "Resume configuration differs from the recorded paired run: ${CONFIG_FILE}" >&2
        exit 1
    fi
else
    printf '%s\n' "$CONFIG_CONTENT" > "$CONFIG_FILE"
fi

train_variant() {
    local spec=$1
    local method=$2
    local dataset_dir=$3
    local seed_start=$4
    local save_dir="${SAVE_ROOT}/${spec}/${method}/seed_start_${seed_start}/resnet_ap"
    local result_file="${save_dir}/per_class_accuracy_all_seeds.json"

    if [[ -f "$result_file" ]]; then
        echo "==> Reusing ${spec}/${method}, classifier seeds ${seed_start} and $((seed_start + 1))"
        return
    fi
    if [[ -e "$save_dir" || -e "${save_dir}_gpu${seed_start}" || -e "${save_dir}_gpu$((seed_start + 1))" ]]; then
        echo "Incomplete paired classifier output exists; refusing to mix runs: ${save_dir}" >&2
        exit 1
    fi

    echo "==> Training ${spec}/${method}, classifier seeds ${seed_start} and $((seed_start + 1))"
    python ./test/train.py \
        --dataset_dir "$dataset_dir" "$IMAGENET_VAL_FOLDER" \
        -d imagenet --spec "$spec" --nclass 10 --size 256 --ipc "$IPC" \
        -n resnet_ap --depth 10 --save-dir "$save_dir" \
        --seed "$seed_start" --workers 12 \
        --n_neighbors "$N_NEIGHBORS" --min_cluster_size "$MIN_CLUSTER_SIZE" \
        --timing_file "${RUN_ROOT}/timings/${spec}_${method}_seed_start_${seed_start}.json" \
        --experiment_method "paired_oracle_${method}" \
        --tag "paired_oracle_${RUN_ID}_${spec}_seed_${seed_start}"
}

for seed_start in "${EVAL_SEED_START_ARRAY[@]}"; do
    if ! [[ "$seed_start" =~ ^[0-9]+$ ]]; then
        echo "EVAL_SEED_STARTS must contain non-negative integers: ${seed_start}" >&2
        exit 1
    fi
    train_variant imageA real "$A_REAL_DATA" "$seed_start"
    train_variant imageA diffusion "$A_DIFFUSION_DATA" "$seed_start"
    train_variant imageA class_oracle "$A_ORACLE_DATA" "$seed_start"
    train_variant imageB real "$B_REAL_DATA" "$seed_start"
    train_variant imageB diffusion "$B_DIFFUSION_DATA" "$seed_start"
    train_variant imageB class_oracle "$B_ORACLE_DATA" "$seed_start"
done

SUMMARY_DIR="${SAVE_ROOT}/summary"
if [[ -e "$SUMMARY_DIR" ]]; then
    if [[ "$RESUME_RUN" != "true" ]]; then
        echo "Paired-oracle summary already exists: ${SUMMARY_DIR}" >&2
        exit 1
    fi
    echo "==> Reusing paired-oracle summary: ${SUMMARY_DIR}"
else
    python summarize_paired_class_oracle.py \
        --output_dir "$SUMMARY_DIR" \
        --input imageA "${A_ORACLE_DATA}/oracle_manifest.json" "$SAVE_ROOT" \
        --input imageB "${B_ORACLE_DATA}/oracle_manifest.json" "$SAVE_ROOT"
fi

echo "Paired class-oracle validation completed: ${SAVE_ROOT}"
