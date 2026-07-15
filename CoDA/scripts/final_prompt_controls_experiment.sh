#!/usr/bin/env bash
set -euo pipefail

export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION="${PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION:-python}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"

MODEL_FOLDER="${MODEL_FOLDER:-/linxi/models/CoDA/SDXL-Refiner}"
VLM_MODEL="${VLM_MODEL:-/linxi/models/CoDA/llava-1.5-7b-hf}"
IMAGENET_TRAIN_FOLDER="${IMAGENET_TRAIN_FOLDER:-/zhangchi/imagenet_512/images}"
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
RUN_DOWNSTREAM_TRAINING="${RUN_DOWNSTREAM_TRAINING:-true}"
RUN_ID="${FINAL_CONTROL_RUN_ID:-final_prompt_controls_$(date -u +%Y%m%dT%H%M%SZ)}"

MONTAGE_INSTRUCTION='Identify recurring physical attributes of the {class_name} across the provided visual examples. Write one concise sentence of at most 35 words for a single-image generation prompt. Mention only shared shape, parts, posture, colors, textures, and distinctive features. Do not mention images, examples, panels, tiles, titles, layout, backgrounds, people, other objects, or one-off details.'
SDXL_CAPTION_TEMPLATE='An natural photo of a {class_name}, {caption}, centered object.'

experiment_dir() {
    local spec=$1
    echo "./results/${spec}/Step-${SAMPLE_STEP}/IPC-${IPC}/DF-${DF}-GTP-${GTP}-gamma-${GAMMA}/n_${N_NEIGHBORS}_s_${MIN_CLUSTER_SIZE}"
}

A_EXPERIMENT_DIR="$(experiment_dir imageA)"
B_EXPERIMENT_DIR="$(experiment_dir imageB)"
A_RUN_DIR="${A_EXPERIMENT_DIR}/final_prompt_controls/${RUN_ID}/seed_${IMAGEA_SEED}"
B_RUN_DIR="${B_EXPERIMENT_DIR}/final_prompt_controls/${RUN_ID}/seed_${IMAGEB_SEED}"
SAVE_ROOT="./trained_results/final_prompt_controls/${RUN_ID}"

A_REFERENCE_DATA="${A_EXPERIMENT_DIR}/multiview_caption_runs/${IMAGEA_REFERENCE_RUN_ID}/seed_${IMAGEA_SEED}"
A_REFERENCE_TRAIN="./trained_results/multiview_caption_runs/imageA/${IMAGEA_REFERENCE_RUN_ID}/seed_${IMAGEA_SEED}"
A_CLASS_DATA="${A_REFERENCE_DATA}/generated_images_coda_baseline"
A_MONTAGE_DATA="${A_REFERENCE_DATA}/generated_images_montage_common_mode"
A_CLASS_RESULT="${A_REFERENCE_TRAIN}/coda_baseline-resnet_ap/per_class_accuracy_all_seeds.json"
A_MONTAGE_RESULT="${A_REFERENCE_TRAIN}/montage_common_mode-resnet_ap/per_class_accuracy_all_seeds.json"

for required in "$A_CLASS_DATA" "$A_MONTAGE_DATA" "$A_CLASS_RESULT" "$A_MONTAGE_RESULT"; do
    if [[ ! -e "$required" ]]; then
        echo "Required reusable ImageA artifact was not found: ${required}" >&2
        exit 1
    fi
done
for output in "$A_RUN_DIR" "$B_RUN_DIR" "$SAVE_ROOT"; do
    if [[ -e "$output" ]]; then
        echo "Refusing to overwrite final-control output: ${output}" >&2
        exit 1
    fi
done
mkdir -p "$A_RUN_DIR" "$B_RUN_DIR"

for spec in imageA imageB; do
    run_dir="$A_RUN_DIR"
    if [[ "$spec" == "imageB" ]]; then run_dir="$B_RUN_DIR"; fi
    python validate_imagenet_subset.py \
        --train_dir "$IMAGENET_TRAIN_FOLDER" --val_dir "$IMAGENET_VAL_FOLDER" \
        --spec "$spec" --nclass 10 --min_train_per_class "$IPC" \
        --output "${run_dir}/dataset_validation.json"
done

echo "==> Decoding representative latents for ImageA"
python reconstruct_representatives.py \
    --local_model_path "$MODEL_FOLDER" --output_dir "${A_RUN_DIR}/vae_reconstruction" \
    --spec imageA --ipc "$IPC" --n_neighbors "$N_NEIGHBORS" \
    --min_cluster_size "$MIN_CLUSTER_SIZE" --output_size 256

echo "==> Decoding representative latents for ImageB"
python reconstruct_representatives.py \
    --local_model_path "$MODEL_FOLDER" --output_dir "${B_RUN_DIR}/vae_reconstruction" \
    --spec imageB --ipc "$IPC" --n_neighbors "$N_NEIGHBORS" \
    --min_cluster_size "$MIN_CLUSTER_SIZE" --output_size 256

B_CAPTION_FILE="${B_RUN_DIR}/captions_montage_common_mode.json"
echo "==> Generating ImageB montage captions"
python CoDA_main.py \
    --spec imageB --IPC "$IPC" \
    --n_neighbors "$N_NEIGHBORS" --min_cluster_size "$MIN_CLUSTER_SIZE" \
    --sample_step "$SAMPLE_STEP" --denoising_factor "$DF" \
    --guideTPercent "$GTP" --CoDA_guidance_scale "$GAMMA" \
    --generate_cluster_captions --cluster_caption_model_path "$VLM_MODEL" \
    --cluster_caption_file "$B_CAPTION_FILE" \
    --cluster_caption_image_mode montage_neighbors --cluster_caption_neighbor_count 4 \
    --cluster_caption_max_words 35 \
    --cluster_caption_montage_dir "${B_RUN_DIR}/caption_montages" \
    --cluster_caption_instruction "$MONTAGE_INSTRUCTION" \
    --experiment_method "final_control_montage_caption_generation" \
    --timing_file "${B_RUN_DIR}/timings/montage_caption_generation.json"

COMMON_GENERATION_ARGS=(
    --local_model_path "$MODEL_FOLDER" --IPC "$IPC"
    --n_neighbors "$N_NEIGHBORS" --min_cluster_size "$MIN_CLUSTER_SIZE"
    --sample_step "$SAMPLE_STEP" --denoising_factor "$DF"
    --guideTPercent "$GTP" --CoDA_guidance_scale "$GAMMA"
    --generate_images --measure_guidance_conflict
)

generate_base_prompt() {
    local spec=$1
    local seed=$2
    local method=$3
    local prompt_template=$4
    local run_dir=$5
    local output_dirname="final_prompt_controls/${RUN_ID}/seed_${seed}/generated_images_${method}"
    echo "==> Generating ${spec}/${method}, seed ${seed}"
    python CoDA_main.py \
        "${COMMON_GENERATION_ARGS[@]}" --spec "$spec" --seed "$seed" \
        --base_prompt_template "$prompt_template" \
        --experiment_method "$method" --generated_images_dirname "$output_dirname" \
        --timing_file "${run_dir}/timings/${method}.json"
}

generate_montage() {
    local spec=$1
    local seed=$2
    local caption_file=$3
    local run_dir=$4
    local method="montage_caption"
    echo "==> Generating ${spec}/${method}, seed ${seed}"
    python CoDA_main.py \
        "${COMMON_GENERATION_ARGS[@]}" --spec "$spec" --seed "$seed" \
        --use_cluster_captions --cluster_caption_file "$caption_file" \
        --cluster_caption_prompt_template "$SDXL_CAPTION_TEMPLATE" \
        --experiment_method "$method" \
        --generated_images_dirname "final_prompt_controls/${RUN_ID}/seed_${seed}/generated_images_${method}" \
        --timing_file "${run_dir}/timings/${method}.json"
}

# ImageA class-prompt and montage datasets are reused from imageA_multiview_v0.
generate_base_prompt imageA "$IMAGEA_SEED" empty_prompt "" "$A_RUN_DIR"
generate_base_prompt imageA "$IMAGEA_SEED" generic_prompt "a natural photo" "$A_RUN_DIR"

generate_base_prompt imageB "$IMAGEB_SEED" empty_prompt "" "$B_RUN_DIR"
generate_base_prompt imageB "$IMAGEB_SEED" generic_prompt "a natural photo" "$B_RUN_DIR"
generate_base_prompt imageB "$IMAGEB_SEED" class_prompt "{class_name}" "$B_RUN_DIR"
generate_montage imageB "$IMAGEB_SEED" "$B_CAPTION_FILE" "$B_RUN_DIR"

python compare_guidance_metrics.py \
    --input "empty=${A_RUN_DIR}/generated_images_empty_prompt/guidance_metrics/guidance_metrics_summary.json" \
    --input "generic=${A_RUN_DIR}/generated_images_generic_prompt/guidance_metrics/guidance_metrics_summary.json" \
    --input "class=${A_CLASS_DATA}/guidance_metrics/guidance_metrics_summary.json" \
    --input "montage=${A_MONTAGE_DATA}/guidance_metrics/guidance_metrics_summary.json" \
    --output_dir "${A_RUN_DIR}/guidance_comparison"

python compare_guidance_metrics.py \
    --input "empty=${B_RUN_DIR}/generated_images_empty_prompt/guidance_metrics/guidance_metrics_summary.json" \
    --input "generic=${B_RUN_DIR}/generated_images_generic_prompt/guidance_metrics/guidance_metrics_summary.json" \
    --input "class=${B_RUN_DIR}/generated_images_class_prompt/guidance_metrics/guidance_metrics_summary.json" \
    --input "montage=${B_RUN_DIR}/generated_images_montage_caption/guidance_metrics/guidance_metrics_summary.json" \
    --output_dir "${B_RUN_DIR}/guidance_comparison"

train_variant() {
    local spec=$1
    local seed=$2
    local method=$3
    local dataset_dir=$4
    local timing_file=$5
    local save_dir="${SAVE_ROOT}/${spec}/seed_${seed}/${method}-resnet_ap"
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

if [[ "$RUN_DOWNSTREAM_TRAINING" == "true" ]]; then
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
fi

echo "Final prompt-control generation completed: ${RUN_ID}"
echo "Training and summary root: ${SAVE_ROOT}"
