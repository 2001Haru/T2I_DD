#!/usr/bin/env bash
set -euo pipefail

MODEL_FOLDER="${MODEL_FOLDER:-/linxi/models/CoDA/SDXL-Refiner}"
VLM_MODEL="${VLM_MODEL:-/linxi/models/CoDA/llava-1.5-7b-hf}"
IMAGENET_TRAIN_FOLDER="${IMAGENET_TRAIN_FOLDER:-/zhangchi/imagenet_512/images}"
IMAGENET_VAL_FOLDER="${IMAGENET_VAL_FOLDER:-/linxi/dataset/imagenet/validation/val}"
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
GENERATION_SEEDS="${GENERATION_SEEDS:-0 1}"
EVAL_SEED="${EVAL_SEED:-0}"
CALCULATE_FEATURES="${CALCULATE_FEATURES:-false}"
CALCULATE_CLUSTER="${CALCULATE_CLUSTER:-false}"
GENERATE_CAPTIONS="${GENERATE_CAPTIONS:-true}"
OVERWRITE_CAPTIONS="${OVERWRITE_CAPTIONS:-false}"
RUN_GENERATION="${RUN_GENERATION:-true}"
RUN_DOWNSTREAM_TRAINING="${RUN_DOWNSTREAM_TRAINING:-true}"
RESUME_RUN="${RESUME_RUN:-false}"
RUN_ID="${MULTIVIEW_RUN_ID:-${SPEC}_multiview_$(date -u +%Y%m%dT%H%M%SZ)}"

SINGLE_INSTRUCTION='Write one concise sentence of at most 35 words describing only the physical appearance of the {class_name}, including visible shape, parts, posture, colors, textures, and distinctive features. Output only attributes usable in a single-image generation prompt. Do not mention the image, background, people, or other objects.'
MONTAGE_INSTRUCTION='Identify recurring physical attributes of the {class_name} across the provided visual examples. Write one concise sentence of at most 35 words for a single-image generation prompt. Mention only shared shape, parts, posture, colors, textures, and distinctive features. Do not mention images, examples, panels, tiles, titles, layout, backgrounds, people, other objects, or one-off details.'
SDXL_PROMPT_TEMPLATE='An natural photo of a {class_name}, {caption}, centered object.'

EXPERIMENT_DIR="./results/${SPEC}/Step-${SAMPLE_STEP}/IPC-${IPC}/DF-${DF}-GTP-${GTP}-gamma-${GAMMA}/n_${N_NEIGHBORS}_s_${MIN_CLUSTER_SIZE}"
RUN_DIR="${EXPERIMENT_DIR}/multiview_caption_runs/${RUN_ID}"
SAVE_ROOT="./trained_results/multiview_caption_runs/${SPEC}/${RUN_ID}"
SINGLE_CAPTION_FILE="${RUN_DIR}/captions_single_focused.json"
MONTAGE_CAPTION_FILE="${RUN_DIR}/captions_montage_common_mode.json"

if [[ -e "$RUN_DIR" ]]; then
    if [[ "$RESUME_RUN" != "true" ]]; then
        echo "Run already exists; set RESUME_RUN=true to continue it explicitly: ${RUN_ID}" >&2
        exit 1
    fi
elif [[ -e "$SAVE_ROOT" ]]; then
    echo "Refusing to mix a new run with existing trained results: ${SAVE_ROOT}" >&2
    exit 1
fi
mkdir -p "$RUN_DIR"

if [[ ! -f "${RUN_DIR}/dataset_validation.json" ]]; then
    python validate_imagenet_subset.py \
        --train_dir "$IMAGENET_TRAIN_FOLDER" --val_dir "$IMAGENET_VAL_FOLDER" \
        --spec "$SPEC" --nclass 10 --min_train_per_class "$IPC" \
        --output "${RUN_DIR}/dataset_validation.json"
fi

PREPARATION_ARGS=()
if [[ "$CALCULATE_FEATURES" == "true" ]]; then
    PREPARATION_ARGS+=(--calcu_features)
fi
if [[ "$CALCULATE_CLUSTER" == "true" ]]; then
    PREPARATION_ARGS+=(--calcu_cluster --cluster_detial --cluster_logger)
fi

if (( ${#PREPARATION_ARGS[@]} > 0 )); then
    echo "==> Preparing ${SPEC} features and clusters"
    python CoDA_main.py \
        --dataset_dir "$IMAGENET_TRAIN_FOLDER" --local_model_path "$MODEL_FOLDER" \
        --spec "$SPEC" --IPC "$IPC" \
        --n_neighbors "$N_NEIGHBORS" --min_cluster_size "$MIN_CLUSTER_SIZE" \
        --sample_step "$SAMPLE_STEP" --denoising_factor "$DF" \
        --guideTPercent "$GTP" --CoDA_guidance_scale "$GAMMA" \
        --experiment_method "multiview_preparation" \
        --timing_file "${RUN_DIR}/timings/preparation.json" \
        "${PREPARATION_ARGS[@]}"
fi

if [[ "$GENERATE_CAPTIONS" == "true" ]]; then
    CAPTION_OVERWRITE_ARGS=()
    if [[ "$OVERWRITE_CAPTIONS" == "true" ]]; then
        CAPTION_OVERWRITE_ARGS+=(--overwrite_cluster_captions)
    fi
    echo "==> Captioning single representatives"
    python CoDA_main.py \
        --spec "$SPEC" --IPC "$IPC" \
        --n_neighbors "$N_NEIGHBORS" --min_cluster_size "$MIN_CLUSTER_SIZE" \
        --sample_step "$SAMPLE_STEP" --denoising_factor "$DF" \
        --guideTPercent "$GTP" --CoDA_guidance_scale "$GAMMA" \
        --generate_cluster_captions \
        --cluster_caption_model_path "$VLM_MODEL" \
        --cluster_caption_file "$SINGLE_CAPTION_FILE" \
        --cluster_caption_image_mode representative \
        --cluster_caption_max_words 35 \
        --cluster_caption_instruction "$SINGLE_INSTRUCTION" \
        --experiment_method "single_caption_generation" \
        --timing_file "${RUN_DIR}/timings/single_caption_generation.json" \
        "${CAPTION_OVERWRITE_ARGS[@]}"

    echo "==> Captioning four-neighbor common modes"
    python CoDA_main.py \
        --spec "$SPEC" --IPC "$IPC" \
        --n_neighbors "$N_NEIGHBORS" --min_cluster_size "$MIN_CLUSTER_SIZE" \
        --sample_step "$SAMPLE_STEP" --denoising_factor "$DF" \
        --guideTPercent "$GTP" --CoDA_guidance_scale "$GAMMA" \
        --generate_cluster_captions \
        --cluster_caption_model_path "$VLM_MODEL" \
        --cluster_caption_file "$MONTAGE_CAPTION_FILE" \
        --cluster_caption_image_mode montage_neighbors \
        --cluster_caption_neighbor_count 4 \
        --cluster_caption_max_words 35 \
        --cluster_caption_montage_dir "${RUN_DIR}/caption_montages" \
        --cluster_caption_instruction "$MONTAGE_INSTRUCTION" \
        --experiment_method "montage_caption_generation" \
        --timing_file "${RUN_DIR}/timings/montage_caption_generation.json" \
        "${CAPTION_OVERWRITE_ARGS[@]}"
fi

for caption_file in "$SINGLE_CAPTION_FILE" "$MONTAGE_CAPTION_FILE"; do
    if [[ ! -f "$caption_file" ]]; then
        echo "Missing caption manifest: ${caption_file}" >&2
        exit 1
    fi
done

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
    local caption_file=${3:-}
    local output_dir="${RUN_DIR}/seed_${generation_seed}/generated_images_${method}"
    local caption_args=()
    if [[ -e "$output_dir" ]]; then
        echo "Refusing to overwrite generated dataset: ${output_dir}" >&2
        exit 1
    fi
    if [[ -n "$caption_file" ]]; then
        caption_args=(
            --use_cluster_captions --cluster_caption_file "$caption_file"
            --cluster_caption_prompt_template "$SDXL_PROMPT_TEMPLATE"
        )
    fi

    echo "==> Generating ${method}, generation seed ${generation_seed}"
    python CoDA_main.py \
        "${COMMON_GENERATION_ARGS[@]}" --seed "$generation_seed" \
        --experiment_method "$method" \
        --generated_images_dirname "multiview_caption_runs/${RUN_ID}/seed_${generation_seed}/generated_images_${method}" \
        --timing_file "${RUN_DIR}/seed_${generation_seed}/timings/${method}.json" \
        "${caption_args[@]}"
}

if [[ "$RUN_GENERATION" == "true" ]]; then
    for generation_seed in $GENERATION_SEEDS; do
        generate_variant "$generation_seed" coda_baseline
        generate_variant "$generation_seed" single_focused "$SINGLE_CAPTION_FILE"
        generate_variant "$generation_seed" montage_common_mode "$MONTAGE_CAPTION_FILE"

        seed_dir="${RUN_DIR}/seed_${generation_seed}"
        python compare_guidance_metrics.py \
            --input "baseline=${seed_dir}/generated_images_coda_baseline/guidance_metrics/guidance_metrics_summary.json" \
            --input "single=${seed_dir}/generated_images_single_focused/guidance_metrics/guidance_metrics_summary.json" \
            --input "montage=${seed_dir}/generated_images_montage_common_mode/guidance_metrics/guidance_metrics_summary.json" \
            --output_dir "${seed_dir}/guidance_comparison"
    done
fi

train_variant() {
    local generation_seed=$1
    local method=$2
    local dataset_dir="${RUN_DIR}/seed_${generation_seed}/generated_images_${method}"
    local save_dir="${SAVE_ROOT}/seed_${generation_seed}/${method}-resnet_ap"
    echo "==> Training ${method}, generation seed ${generation_seed}"
    PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python python ./test/train.py \
        --dataset_dir "$dataset_dir" "$IMAGENET_VAL_FOLDER" \
        -d imagenet --spec "$SPEC" --nclass 10 --size 256 --ipc "$IPC" \
        -n resnet_ap --depth 10 --save-dir "$save_dir" \
        --seed "$EVAL_SEED" --workers 12 \
        --n_neighbors "$N_NEIGHBORS" --min_cluster_size "$MIN_CLUSTER_SIZE" \
        --timing_file "${RUN_DIR}/seed_${generation_seed}/timings/${method}.json" \
        --experiment_method "$method" --tag "multiview_gen_seed_${generation_seed}"
}

if [[ "$RUN_DOWNSTREAM_TRAINING" == "true" ]]; then
    for generation_seed in $GENERATION_SEEDS; do
        train_variant "$generation_seed" coda_baseline
        train_variant "$generation_seed" single_focused
        train_variant "$generation_seed" montage_common_mode
    done
    python summarize_multiview_caption_results.py \
        --run_dir "$RUN_DIR" --trained_root "$SAVE_ROOT" \
        --generation_seeds $GENERATION_SEEDS
fi

echo "Multiview caption experiment completed: ${RUN_DIR}"
echo "Downstream results: ${SAVE_ROOT}"
