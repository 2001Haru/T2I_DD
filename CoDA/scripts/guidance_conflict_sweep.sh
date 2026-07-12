#!/usr/bin/env bash
set -euo pipefail

MODEL_FOLDER="${MODEL_FOLDER:-/linxi/models/CoDA/SDXL-Refiner}"
IMAGENET_VAL_FOLDER="${IMAGENET_VAL_FOLDER:-/linxi/dataset/imagenet/validation/val}"
GENERIC_CAPTION_FILE="${GENERIC_CAPTION_FILE:-}"
CLASS_FOCUSED_CAPTION_FILE="${CLASS_FOCUSED_CAPTION_FILE:-}"
REFERENCE_RUN_DIR="${REFERENCE_RUN_DIR:-}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export CUDA_VISIBLE_DEVICES

SPEC="${SPEC:-imageA}"
IPC="${IPC:-10}"
N_NEIGHBORS="${N_NEIGHBORS:-85}"
MIN_CLUSTER_SIZE="${MIN_CLUSTER_SIZE:-55}"
SAMPLE_STEP="${SAMPLE_STEP:-25}"
DF="${DF:-1.0}"
GTP="${GTP:-0.9}"
GAMMA="${GAMMA:-0.05}"
GENERATION_SEED="${GENERATION_SEED:-1}"
EVAL_SEED="${EVAL_SEED:-0}"
RUN_DOWNSTREAM_TRAINING="${RUN_DOWNSTREAM_TRAINING:-true}"
RUN_ID="${GUIDANCE_RUN_ID:-gen_seed${GENERATION_SEED}_$(date -u +%Y%m%dT%H%M%SZ)}"

EXPERIMENT_DIR="./results/${SPEC}/Step-${SAMPLE_STEP}/IPC-${IPC}/DF-${DF}-GTP-${GTP}-gamma-${GAMMA}/n_${N_NEIGHBORS}_s_${MIN_CLUSTER_SIZE}"
RUN_DIR="${EXPERIMENT_DIR}/guidance_conflict_runs/${RUN_ID}"
GENERIC_CAPTION_FILE="${GENERIC_CAPTION_FILE:-${EXPERIMENT_DIR}/cluster_captions.json}"
CLASS_FOCUSED_CAPTION_FILE="${CLASS_FOCUSED_CAPTION_FILE:-${EXPERIMENT_DIR}/cluster_captions_vlm_caption_class_focused.json}"

if [[ -e "$RUN_DIR" ]]; then
    echo "Refusing to overwrite existing guidance run: $RUN_DIR" >&2
    exit 1
fi

if [[ ! -f "$GENERIC_CAPTION_FILE" ]]; then
    echo "Missing v0 caption file: $GENERIC_CAPTION_FILE" >&2
    exit 1
fi
if [[ ! -f "$CLASS_FOCUSED_CAPTION_FILE" ]]; then
    echo "Missing v1 caption file: $CLASS_FOCUSED_CAPTION_FILE" >&2
    exit 1
fi

COMMON_ARGS=(
    --local_model_path "$MODEL_FOLDER"
    --spec "$SPEC"
    --IPC "$IPC"
    --n_neighbors "$N_NEIGHBORS"
    --min_cluster_size "$MIN_CLUSTER_SIZE"
    --sample_step "$SAMPLE_STEP"
    --denoising_factor "$DF"
    --guideTPercent "$GTP"
    --CoDA_guidance_scale "$GAMMA"
    --seed "$GENERATION_SEED"
    --generate_images
    --measure_guidance_conflict
)

run_variant() {
    local method=$1
    local output_dirname="guidance_conflict_runs/${RUN_ID}/generated_images_${method}"
    shift
    echo "==> Generating guidance diagnostics: ${method}"
    python CoDA_main.py \
        "${COMMON_ARGS[@]}" \
        --experiment_method "$method" \
        --generated_images_dirname "$output_dirname" \
        --timing_file "${RUN_DIR}/timings/${method}.json" \
        "$@"
}

run_variant "coda_baseline"

run_variant "v0_generic_caption" \
    --use_cluster_captions \
    --cluster_caption_file "$GENERIC_CAPTION_FILE" \
    --cluster_caption_prompt_template "A high-quality natural image of a {class_name}. {caption}"

run_variant "v1_class_focused_caption" \
    --use_cluster_captions \
    --cluster_caption_file "$CLASS_FOCUSED_CAPTION_FILE" \
    --cluster_caption_prompt_template "An natural photo of a {class_name}, {caption}, centered object."

python compare_guidance_metrics.py \
    --input "baseline=${RUN_DIR}/generated_images_coda_baseline/guidance_metrics/guidance_metrics_summary.json" \
    --input "v0_generic=${RUN_DIR}/generated_images_v0_generic_caption/guidance_metrics/guidance_metrics_summary.json" \
    --input "v1_class_focused=${RUN_DIR}/generated_images_v1_class_focused_caption/guidance_metrics/guidance_metrics_summary.json" \
    --output_dir "${RUN_DIR}/comparison"

if [[ -n "$REFERENCE_RUN_DIR" ]]; then
    python compare_guidance_metrics.py \
        --input "seed0_baseline=${REFERENCE_RUN_DIR}/generated_images_coda_baseline/guidance_metrics/guidance_metrics_summary.json" \
        --input "seed0_v0=${REFERENCE_RUN_DIR}/generated_images_v0_generic_caption/guidance_metrics/guidance_metrics_summary.json" \
        --input "seed0_v1=${REFERENCE_RUN_DIR}/generated_images_v1_class_focused_caption/guidance_metrics/guidance_metrics_summary.json" \
        --input "seed${GENERATION_SEED}_baseline=${RUN_DIR}/generated_images_coda_baseline/guidance_metrics/guidance_metrics_summary.json" \
        --input "seed${GENERATION_SEED}_v0=${RUN_DIR}/generated_images_v0_generic_caption/guidance_metrics/guidance_metrics_summary.json" \
        --input "seed${GENERATION_SEED}_v1=${RUN_DIR}/generated_images_v1_class_focused_caption/guidance_metrics/guidance_metrics_summary.json" \
        --output_dir "${RUN_DIR}/cross_seed_comparison"
fi

train_variant() {
    local method=$1
    local train_data_path="${RUN_DIR}/generated_images_${method}"
    local train_save_dir="./trained_results/guidance_conflict_runs/${SPEC}/${RUN_ID}/${method}-resnet_ap"
    echo "==> Training downstream classifier: ${method}"
    python ./test/train.py --dataset_dir "$train_data_path" "$IMAGENET_VAL_FOLDER" \
        -d imagenet --spec "$SPEC" --nclass 10 --size 256 --ipc "$IPC" \
        -n resnet_ap --depth 10 --save-dir "$train_save_dir" \
        --seed "$EVAL_SEED" --workers 12 \
        --n_neighbors "$N_NEIGHBORS" --min_cluster_size "$MIN_CLUSTER_SIZE" \
        --experiment_method "$method" --tag "guidance_seed_${GENERATION_SEED}"
}

if [[ "$RUN_DOWNSTREAM_TRAINING" == "true" ]]; then
    train_variant "coda_baseline"
    train_variant "v0_generic_caption"
    train_variant "v1_class_focused_caption"
fi

echo "Guidance conflict sweep completed: ${RUN_DIR}"
