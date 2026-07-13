#!/usr/bin/env bash
set -euo pipefail

MODEL_FOLDER="${MODEL_FOLDER:-/linxi/models/CoDA/SDXL-Refiner}"
VLM_MODEL="${VLM_MODEL:-/linxi/models/CoDA/llava-1.5-7b-hf}"
IMAGENET_TRAIN_FOLDER="${IMAGENET_TRAIN_FOLDER:-/zhangchi/imagenet_512/images}"
IMAGENET_VAL_FOLDER="${IMAGENET_VAL_FOLDER:-/linxi/dataset/imagenet/validation/val}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export CUDA_VISIBLE_DEVICES

SPEC="imageB"
IPC="${IPC:-10}"
N_NEIGHBORS="${N_NEIGHBORS:-85}"
MIN_CLUSTER_SIZE="${MIN_CLUSTER_SIZE:-55}"
SAMPLE_STEP="${SAMPLE_STEP:-25}"
DF="${DF:-1.0}"
GTP="${GTP:-0.9}"
GAMMA="${GAMMA:-0.05}"
GENERATION_SEEDS="${GENERATION_SEEDS:-0 1}"
EVAL_SEED="${EVAL_SEED:-0}"
CALCULATE_FEATURES="${CALCULATE_FEATURES:-true}"
CALCULATE_CLUSTER="${CALCULATE_CLUSTER:-true}"
GENERATE_CAPTIONS="${GENERATE_CAPTIONS:-true}"
RUN_DOWNSTREAM_TRAINING="${RUN_DOWNSTREAM_TRAINING:-true}"
RUN_ID="${IMAGEB_RUN_ID:-imageB_$(date -u +%Y%m%dT%H%M%SZ)}"

if [[ -z "${VLM_CAPTION_INSTRUCTION:-}" ]]; then
    VLM_CAPTION_INSTRUCTION='Describe the physical appearance of the {class_name} in the image. Include details about its shape, posture, color, and any distinct features.'
fi
if [[ -z "${SDXL_CAPTION_PROMPT_TEMPLATE:-}" ]]; then
    SDXL_CAPTION_PROMPT_TEMPLATE='An natural photo of a {class_name}, {caption}, centered object.'
fi

EXPERIMENT_DIR="./results/${SPEC}/Step-${SAMPLE_STEP}/IPC-${IPC}/DF-${DF}-GTP-${GTP}-gamma-${GAMMA}/n_${N_NEIGHBORS}_s_${MIN_CLUSTER_SIZE}"
RUN_DIR="${EXPERIMENT_DIR}/imageB_runs/${RUN_ID}"
SAVE_ROOT="./trained_results/imageB_runs/${RUN_ID}"
CAPTION_FILE="${EXPERIMENT_DIR}/cluster_captions_v1_class_focused.json"

if [[ -e "$RUN_DIR" || -e "$SAVE_ROOT" ]]; then
    echo "Refusing to overwrite an existing ImageB run: ${RUN_ID}" >&2
    exit 1
fi
mkdir -p "$RUN_DIR"

python validate_imagenet_subset.py \
    --train_dir "$IMAGENET_TRAIN_FOLDER" \
    --val_dir "$IMAGENET_VAL_FOLDER" \
    --spec "$SPEC" --nclass 10 --min_train_per_class "$IPC" \
    --output "${RUN_DIR}/dataset_validation.json"

PREPARATION_ARGS=()
if [[ "$CALCULATE_FEATURES" == "true" ]]; then
    PREPARATION_ARGS+=(--calcu_features)
fi
if [[ "$CALCULATE_CLUSTER" == "true" ]]; then
    PREPARATION_ARGS+=(--calcu_cluster --cluster_detial --cluster_logger)
fi
if [[ "$GENERATE_CAPTIONS" == "true" ]]; then
    PREPARATION_ARGS+=(--generate_cluster_captions)
fi

if (( ${#PREPARATION_ARGS[@]} > 0 )); then
    echo "==> Preparing fixed ImageB clusters and focused captions"
    python CoDA_main.py \
        --dataset_dir "$IMAGENET_TRAIN_FOLDER" \
        --local_model_path "$MODEL_FOLDER" \
        --spec "$SPEC" --IPC "$IPC" \
        --n_neighbors "$N_NEIGHBORS" --min_cluster_size "$MIN_CLUSTER_SIZE" \
        --sample_step "$SAMPLE_STEP" --denoising_factor "$DF" \
        --guideTPercent "$GTP" --CoDA_guidance_scale "$GAMMA" \
        --cluster_caption_model_path "$VLM_MODEL" \
        --cluster_caption_file "$CAPTION_FILE" \
        --cluster_caption_instruction "$VLM_CAPTION_INSTRUCTION" \
        --experiment_method "imageB_preparation" \
        --timing_file "${RUN_DIR}/timings/preparation.json" \
        "${PREPARATION_ARGS[@]}"
fi

if [[ ! -f "$CAPTION_FILE" ]]; then
    echo "Missing focused caption file after preparation: $CAPTION_FILE" >&2
    exit 1
fi

COMMON_GENERATION_ARGS=(
    --local_model_path "$MODEL_FOLDER"
    --spec "$SPEC" --IPC "$IPC"
    --n_neighbors "$N_NEIGHBORS" --min_cluster_size "$MIN_CLUSTER_SIZE"
    --sample_step "$SAMPLE_STEP" --denoising_factor "$DF"
    --guideTPercent "$GTP" --CoDA_guidance_scale "$GAMMA"
    --generate_images --measure_guidance_conflict
)

generate_variant() {
    local generation_seed=$1
    local method=$2
    local alpha=$3
    local use_captions=$4
    local output_dirname="imageB_runs/${RUN_ID}/seed_${generation_seed}/generated_images_${method}"
    local timing_file="${RUN_DIR}/seed_${generation_seed}/timings/${method}.json"
    local caption_args=()

    if [[ "$use_captions" == "true" ]]; then
        caption_args=(
            --use_cluster_captions
            --cluster_caption_file "$CAPTION_FILE"
            --cluster_caption_prompt_template "$SDXL_CAPTION_PROMPT_TEMPLATE"
        )
    fi

    echo "==> Generating ${method}, generation seed ${generation_seed}"
    python CoDA_main.py \
        "${COMMON_GENERATION_ARGS[@]}" \
        --seed "$generation_seed" \
        --conflict_projection_alpha "$alpha" \
        --experiment_method "$method" \
        --generated_images_dirname "$output_dirname" \
        --timing_file "$timing_file" \
        "${caption_args[@]}"
}

for generation_seed in $GENERATION_SEEDS; do
    generate_variant "$generation_seed" "coda_baseline" "0.0" "false"
    generate_variant "$generation_seed" "v1_focused_alpha_0" "0.0" "true"
    generate_variant "$generation_seed" "v1_focused_alpha_0p5" "0.5" "true"

    seed_dir="${RUN_DIR}/seed_${generation_seed}"
    python compare_guidance_metrics.py \
        --input "baseline=${seed_dir}/generated_images_coda_baseline/guidance_metrics/guidance_metrics_summary.json" \
        --input "focused_alpha_0=${seed_dir}/generated_images_v1_focused_alpha_0/guidance_metrics/guidance_metrics_summary.json" \
        --input "focused_alpha_0p5=${seed_dir}/generated_images_v1_focused_alpha_0p5/guidance_metrics/guidance_metrics_summary.json" \
        --output_dir "${seed_dir}/comparison"
done

train_variant() {
    local generation_seed=$1
    local method=$2
    local dataset_dir="${RUN_DIR}/seed_${generation_seed}/generated_images_${method}"
    local save_dir="${SAVE_ROOT}/seed_${generation_seed}/${method}-resnet_ap"
    local timing_file="${RUN_DIR}/seed_${generation_seed}/timings/${method}.json"

    echo "==> Training ${method}, generation seed ${generation_seed}"
    PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python python ./test/train.py \
        --dataset_dir "$dataset_dir" "$IMAGENET_VAL_FOLDER" \
        -d imagenet --spec "$SPEC" --nclass 10 --size 256 --ipc "$IPC" \
        -n resnet_ap --depth 10 --save-dir "$save_dir" \
        --seed "$EVAL_SEED" --workers 12 \
        --n_neighbors "$N_NEIGHBORS" --min_cluster_size "$MIN_CLUSTER_SIZE" \
        --timing_file "$timing_file" --experiment_method "$method" \
        --tag "imageB_gen_seed_${generation_seed}"
}

if [[ "$RUN_DOWNSTREAM_TRAINING" == "true" ]]; then
    for generation_seed in $GENERATION_SEEDS; do
        train_variant "$generation_seed" "coda_baseline"
        train_variant "$generation_seed" "v1_focused_alpha_0"
        train_variant "$generation_seed" "v1_focused_alpha_0p5"
    done
    python summarize_imageB_results.py \
        --run_dir "$RUN_DIR" --trained_root "$SAVE_ROOT" \
        --generation_seeds $GENERATION_SEEDS
fi

echo "ImageB projection experiment completed: ${RUN_DIR}"
echo "Downstream results: ${SAVE_ROOT}"
