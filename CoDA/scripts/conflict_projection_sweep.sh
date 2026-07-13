#!/usr/bin/env bash
set -euo pipefail

MODEL_FOLDER="${MODEL_FOLDER:-/linxi/models/CoDA/SDXL-Refiner}"
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
PROJECTION_ALPHAS="${PROJECTION_ALPHAS:-0.5 1.0}"
EVAL_SEED="${EVAL_SEED:-0}"
RUN_DOWNSTREAM_TRAINING="${RUN_DOWNSTREAM_TRAINING:-true}"
RUN_ID="${PROJECTION_RUN_ID:-projection_$(date -u +%Y%m%dT%H%M%SZ)}"

EXPERIMENT_DIR="./results/${SPEC}/Step-${SAMPLE_STEP}/IPC-${IPC}/DF-${DF}-GTP-${GTP}-gamma-${GAMMA}/n_${N_NEIGHBORS}_s_${MIN_CLUSTER_SIZE}"
RUN_DIR="${EXPERIMENT_DIR}/projection_runs/${RUN_ID}"
CAPTION_FILE="${CLASS_FOCUSED_CAPTION_FILE:-${EXPERIMENT_DIR}/cluster_captions_vlm_caption_class_focused.json}"
SAVE_ROOT="./trained_results/projection_runs/${SPEC}/${RUN_ID}"

if [[ -e "$RUN_DIR" || -e "$SAVE_ROOT" ]]; then
    echo "Refusing to overwrite an existing projection run: ${RUN_ID}" >&2
    exit 1
fi
if [[ ! -f "$CAPTION_FILE" ]]; then
    echo "Missing class-focused caption file: $CAPTION_FILE" >&2
    exit 1
fi

alpha_tag() {
    printf '%s' "$1" | tr '.' 'p'
}

train_variant() {
    local dataset_dir=$1
    local save_dir=$2
    local method=$3
    local generation_seed=$4

    PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python python ./test/train.py \
        --dataset_dir "$dataset_dir" "$IMAGENET_VAL_FOLDER" \
        -d imagenet --spec "$SPEC" --nclass 10 --size 256 --ipc "$IPC" \
        -n resnet_ap --depth 10 --save-dir "$save_dir" \
        --seed "$EVAL_SEED" --workers 12 \
        --n_neighbors "$N_NEIGHBORS" --min_cluster_size "$MIN_CLUSTER_SIZE" \
        --experiment_method "$method" --tag "projection_gen_seed_${generation_seed}"
}

mkdir -p "$RUN_DIR"
for generation_seed in $GENERATION_SEEDS; do
    seed_dir="${RUN_DIR}/seed_${generation_seed}"
    comparison_args=()
    reference_var="SEED${generation_seed}_REFERENCE_RUN_DIR"
    reference_dir="${!reference_var:-}"

    if [[ -n "$reference_dir" ]]; then
        baseline_summary="${reference_dir}/generated_images_coda_baseline/guidance_metrics/guidance_metrics_summary.json"
        focused_summary="${reference_dir}/generated_images_v1_class_focused_caption/guidance_metrics/guidance_metrics_summary.json"
        if [[ ! -f "$baseline_summary" || ! -f "$focused_summary" ]]; then
            echo "Reference run for seed ${generation_seed} is incomplete: ${reference_dir}" >&2
            exit 1
        fi
        comparison_args+=(
            --input "baseline_existing=${baseline_summary}"
            --input "focused_alpha_0_existing=${focused_summary}"
        )
    fi

    for alpha in $PROJECTION_ALPHAS; do
        tag="alpha_$(alpha_tag "$alpha")"
        method="v1_focused_projection_${tag}"
        output_dirname="projection_runs/${RUN_ID}/seed_${generation_seed}/generated_images_${method}"
        output_dir="${seed_dir}/generated_images_${method}"

        echo "==> Generating ${method}, generation seed ${generation_seed}"
        python CoDA_main.py \
            --local_model_path "$MODEL_FOLDER" \
            --spec "$SPEC" --IPC "$IPC" \
            --n_neighbors "$N_NEIGHBORS" --min_cluster_size "$MIN_CLUSTER_SIZE" \
            --sample_step "$SAMPLE_STEP" --denoising_factor "$DF" \
            --guideTPercent "$GTP" --CoDA_guidance_scale "$GAMMA" \
            --conflict_projection_alpha "$alpha" \
            --seed "$generation_seed" --generate_images --measure_guidance_conflict \
            --use_cluster_captions --cluster_caption_file "$CAPTION_FILE" \
            --cluster_caption_prompt_template "An natural photo of a {class_name}, {caption}, centered object." \
            --experiment_method "$method" \
            --generated_images_dirname "$output_dirname" \
            --timing_file "${seed_dir}/timings/${method}.json"

        comparison_args+=(
            --input "${tag}=${output_dir}/guidance_metrics/guidance_metrics_summary.json"
        )

        if [[ "$RUN_DOWNSTREAM_TRAINING" == "true" ]]; then
            echo "==> Training ${method}, generation seed ${generation_seed}"
            train_variant \
                "$output_dir" \
                "${SAVE_ROOT}/seed_${generation_seed}/${method}-resnet_ap" \
                "$method" "$generation_seed"
        fi
    done

    python compare_guidance_metrics.py \
        "${comparison_args[@]}" \
        --output_dir "${seed_dir}/comparison"
done

echo "Conflict-aware projection sweep completed: ${RUN_DIR}"
echo "Downstream results: ${SAVE_ROOT}"
