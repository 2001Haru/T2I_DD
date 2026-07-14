#!/usr/bin/env bash
set -euo pipefail

SPEC="${SPEC:-imageB}"
MODEL_FOLDER="${MODEL_FOLDER:-/linxi/models/CoDA/SDXL-Refiner}"
IMAGENET_VAL_FOLDER="${IMAGENET_VAL_FOLDER:-/linxi/dataset/imagenet/validation/val}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export CUDA_VISIBLE_DEVICES

IPC="${IPC:-10}"
N_NEIGHBORS="${N_NEIGHBORS:-85}"
MIN_CLUSTER_SIZE="${MIN_CLUSTER_SIZE:-55}"
SAMPLE_STEP="${SAMPLE_STEP:-25}"
DF="${DF:-1.0}"
GTP="${GTP:-0.9}"
GAMMA="${GAMMA:-0.05}"
KAPPA_CAP="${KAPPA_CAP:-0.3}"
GENERATION_SEEDS="${GENERATION_SEEDS:-0 1}"
EVAL_SEED="${EVAL_SEED:-0}"
RUN_DOWNSTREAM_TRAINING="${RUN_DOWNSTREAM_TRAINING:-true}"
RUN_ID="${KAPPA_CAP_RUN_ID:-${SPEC}_kappa_cap_$(date -u +%Y%m%dT%H%M%SZ)}"

EXPERIMENT_DIR="./results/${SPEC}/Step-${SAMPLE_STEP}/IPC-${IPC}/DF-${DF}-GTP-${GTP}-gamma-${GAMMA}/n_${N_NEIGHBORS}_s_${MIN_CLUSTER_SIZE}"
RUN_DIR="${EXPERIMENT_DIR}/kappa_cap_runs/${RUN_ID}"
SAVE_ROOT="./trained_results/kappa_cap_runs/${SPEC}/${RUN_ID}"

case "$SPEC" in
    imageA)
        DEFAULT_CAPTION_FILE="${EXPERIMENT_DIR}/cluster_captions_vlm_caption_class_focused.json"
        ;;
    imageB)
        DEFAULT_CAPTION_FILE="${EXPERIMENT_DIR}/cluster_captions_v1_class_focused.json"
        ;;
    *)
        echo "Unsupported SPEC for the current development experiment: ${SPEC}" >&2
        exit 1
        ;;
esac
CAPTION_FILE="${CLASS_FOCUSED_CAPTION_FILE:-$DEFAULT_CAPTION_FILE}"

if [[ -e "$RUN_DIR" || -e "$SAVE_ROOT" ]]; then
    echo "Refusing to overwrite an existing kappa-cap run: ${RUN_ID}" >&2
    exit 1
fi
if [[ ! -f "$CAPTION_FILE" ]]; then
    echo "Missing class-focused caption file: $CAPTION_FILE" >&2
    exit 1
fi
mkdir -p "$RUN_DIR"

cap_tag="$(printf '%s' "$KAPPA_CAP" | tr '.' 'p')"
METHOD="v1_focused_kappa_cap_${cap_tag}"

for generation_seed in $GENERATION_SEEDS; do
    output_dirname="kappa_cap_runs/${RUN_ID}/seed_${generation_seed}/generated_images_${METHOD}"
    output_dir="${RUN_DIR}/seed_${generation_seed}/generated_images_${METHOD}"
    timing_file="${RUN_DIR}/seed_${generation_seed}/timings/${METHOD}.json"

    echo "==> Generating ${SPEC}, kappa cap ${KAPPA_CAP}, seed ${generation_seed}"
    python CoDA_main.py \
        --local_model_path "$MODEL_FOLDER" \
        --spec "$SPEC" --IPC "$IPC" \
        --n_neighbors "$N_NEIGHBORS" --min_cluster_size "$MIN_CLUSTER_SIZE" \
        --sample_step "$SAMPLE_STEP" --denoising_factor "$DF" \
        --guideTPercent "$GTP" --CoDA_guidance_scale "$GAMMA" \
        --conflict_projection_kappa_cap "$KAPPA_CAP" \
        --seed "$generation_seed" --generate_images --measure_guidance_conflict \
        --use_cluster_captions --cluster_caption_file "$CAPTION_FILE" \
        --cluster_caption_prompt_template "An natural photo of a {class_name}, {caption}, centered object." \
        --experiment_method "$METHOD" \
        --generated_images_dirname "$output_dirname" \
        --timing_file "$timing_file"

done

if [[ "$RUN_DOWNSTREAM_TRAINING" == "true" ]]; then
    for generation_seed in $GENERATION_SEEDS; do
        output_dir="${RUN_DIR}/seed_${generation_seed}/generated_images_${METHOD}"
        timing_file="${RUN_DIR}/seed_${generation_seed}/timings/${METHOD}.json"
        save_dir="${SAVE_ROOT}/seed_${generation_seed}/${METHOD}-resnet_ap"
        echo "==> Training ${SPEC}, kappa cap ${KAPPA_CAP}, seed ${generation_seed}"
        PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python python ./test/train.py \
            --dataset_dir "$output_dir" "$IMAGENET_VAL_FOLDER" \
            -d imagenet --spec "$SPEC" --nclass 10 --size 256 --ipc "$IPC" \
            -n resnet_ap --depth 10 --save-dir "$save_dir" \
            --seed "$EVAL_SEED" --workers 12 \
            --n_neighbors "$N_NEIGHBORS" --min_cluster_size "$MIN_CLUSTER_SIZE" \
            --timing_file "$timing_file" --experiment_method "$METHOD" \
            --tag "kappa_cap_${cap_tag}_gen_seed_${generation_seed}"
    done
fi

echo "Kappa-cap experiment completed: ${RUN_DIR}"
echo "Downstream results: ${SAVE_ROOT}"
