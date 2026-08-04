#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export TOKENIZERS_PARALLELISM=false

SPECS="${SPECS:-imageA imageB imageC}"
GENERATION_SEEDS="${GENERATION_SEEDS:-0 1}"
P4_RUN_ID="${P4_RUN_ID:-p4_text_execution_v0}"
RUN_ID="${P5_RUN_ID:-p5_continuous_guidance_v0}"
MODEL_FOLDER="${MODEL_FOLDER:-/linxi/models/CoDA/SDXL-Refiner}"
DINO_MODEL="${DINO_MODEL:-/linxi/models/DINOv2/dinov2-base}"
CLIP_MODEL="${CLIP_MODEL:-/linxi/models/CLIP/clip-vit-large-patch14}"
IPC="${IPC:-10}"
N_NEIGHBORS="${N_NEIGHBORS:-85}"
MIN_CLUSTER_SIZE="${MIN_CLUSTER_SIZE:-55}"
SAMPLE_STEP="${SAMPLE_STEP:-25}"
DF="${DF:-1.0}"
GTP="${GTP:-0.9}"
CFG="${CFG:-5.0}"
GUIDANCE_GAMMA="${P5_GUIDANCE_GAMMA:-0.05}"
PROTOTYPE_INIT_STRENGTH="${PROTOTYPE_INIT_STRENGTH:-0.7}"
BATCH_SIZE="${BATCH_SIZE:-64}"
BOOTSTRAP_SAMPLES="${BOOTSTRAP_SAMPLES:-5000}"
PERMUTATION_SAMPLES="${PERMUTATION_SAMPLES:-1000}"
RANDOM_SEED="${RANDOM_SEED:-20260804}"
RUN_GENERATION="${RUN_GENERATION:-true}"
RUN_EVALUATION="${RUN_EVALUATION:-true}"
RERUN_EVALUATION="${RERUN_EVALUATION:-false}"
RESUME="${RESUME:-false}"
ARCHIVE_INCOMPLETE_GENERATION="${ARCHIVE_INCOMPLETE_GENERATION:-false}"
PROMPT_TEMPLATE="${P5_DCS_PROMPT_TEMPLATE:-}"
if [[ -z "$PROMPT_TEMPLATE" ]]; then
    PROMPT_TEMPLATE='An natural photo of a {class_name}, {caption}, centered object.'
fi

P4_ROOT="./results/p4_text_execution_runs/${P4_RUN_ID}"
PREPARED_DIR="${P4_ROOT}/prepared"
P4_MANIFEST="${P4_ROOT}/generation_manifest.json"
META_ROOT="./results/p5_continuous_guidance_runs/${RUN_ID}"
MANIFEST_FILE="${META_ROOT}/generation_manifest.json"
ANALYSIS_DIR="${META_ROOT}/analysis"
CONFIG_FILE="${META_ROOT}/run_config.txt"
COMPLETE_FILE="${META_ROOT}/complete.json"

CONFIG_CONTENT="SPECS=${SPECS}
GENERATION_SEEDS=${GENERATION_SEEDS}
P4_RUN_ID=${P4_RUN_ID}
MODEL_FOLDER=${MODEL_FOLDER}
DINO_MODEL=${DINO_MODEL}
CLIP_MODEL=${CLIP_MODEL}
IPC=${IPC}
N_NEIGHBORS=${N_NEIGHBORS}
MIN_CLUSTER_SIZE=${MIN_CLUSTER_SIZE}
SAMPLE_STEP=${SAMPLE_STEP}
DF=${DF}
GTP=${GTP}
CFG=${CFG}
GUIDANCE_GAMMA=${GUIDANCE_GAMMA}
PROTOTYPE_INIT_STRENGTH=${PROTOTYPE_INIT_STRENGTH}
PROMPT_TEMPLATE=${PROMPT_TEMPLATE}
BATCH_SIZE=${BATCH_SIZE}
BOOTSTRAP_SAMPLES=${BOOTSTRAP_SAMPLES}
PERMUTATION_SAMPLES=${PERMUTATION_SAMPLES}
RANDOM_SEED=${RANDOM_SEED}"

for path in \
    "${PREPARED_DIR}/preparation_summary.json" \
    "${PREPARED_DIR}/pair_manifest.csv" \
    "${PREPARED_DIR}/frozen_real_image_probes.pkl" \
    "$P4_MANIFEST" \
    "${P4_ROOT}/complete.json"; do
    if [[ ! -f "$path" ]]; then
        echo "Missing required completed P4 artifact: ${path}" >&2
        exit 1
    fi
done
for path in "$MODEL_FOLDER" "$DINO_MODEL" "$CLIP_MODEL"; do
    if [[ ! -d "$path" ]]; then
        echo "Missing model directory: ${path}" >&2
        exit 1
    fi
done

if [[ -e "$META_ROOT" ]]; then
    if [[ "$RESUME" != "true" ]]; then
        echo "P5 run exists; set RESUME=true or choose another P5_RUN_ID: ${RUN_ID}" >&2
        exit 1
    fi
    if [[ ! -f "$CONFIG_FILE" ]] || [[ "$(<"$CONFIG_FILE")" != "$CONFIG_CONTENT" ]]; then
        echo "P5 resume configuration differs from ${CONFIG_FILE}" >&2
        exit 1
    fi
    if [[ -f "$COMPLETE_FILE" ]]; then
        if [[ "$RERUN_EVALUATION" != "true" ]]; then
            echo "P5 run already complete: ${META_ROOT}"
            cat "${ANALYSIS_DIR}/summary.json"
            exit 0
        fi
        echo "==> Re-running P5 evaluation from completed generated datasets" >&2
    fi
else
    mkdir -p "$META_ROOT"
    printf '%s\n' "$CONFIG_CONTENT" > "$CONFIG_FILE"
fi

experiment_root() {
    local spec=$1
    local gamma=$2
    echo "./results/${spec}/Step-${SAMPLE_STEP}/IPC-${IPC}/DF-${DF}-GTP-${GTP}-gamma-${gamma}/n_${N_NEIGHBORS}_s_${MIN_CLUSTER_SIZE}"
}

expected_images() {
    local spec=$1
    jq '[.[] | length] | add' "${PREPARED_DIR}/${spec}_cluster_indices.json"
}

validate_dataset() {
    local spec=$1
    local dataset_dir=$2
    local require_guidance=$3
    local seed=${4:-}
    local mode=${5:-}
    local expected actual
    expected="$(expected_images "$spec")"
    # Guidance diagnostics also save PNG plots at depth two. Count only images
    # stored below ImageNet synset directories (for example n02111129/3.png).
    actual="$(find "$dataset_dir" -mindepth 2 -maxdepth 2 -type f \
        -path '*/n[0-9][0-9][0-9][0-9][0-9][0-9][0-9][0-9]/*.png' | wc -l)"
    if [[ "$actual" -ne "$expected" ]]; then
        echo "Validation failed: ${dataset_dir} has ${actual} PNGs; expected ${expected}." >&2
        return 1
    fi
    if ! compgen -G "${dataset_dir}/prompt_records_gpu*.json" > /dev/null; then
        echo "Validation failed: no prompt_records_gpu*.json in ${dataset_dir}." >&2
        return 1
    fi
    if [[ "$require_guidance" == "true" ]]; then
        for required in \
            "${dataset_dir}/guidance_metrics/guidance_metrics_raw.csv" \
            "${dataset_dir}/guidance_metrics/guidance_metrics_summary.json" \
            "${dataset_dir}/prompt_config.json"; do
            if [[ ! -f "$required" ]]; then
                echo "Validation failed: missing ${required}." >&2
                return 1
            fi
        done
        if ! jq -e \
            --arg spec "$spec" --argjson seed "$seed" \
            --argjson gamma "$GUIDANCE_GAMMA" --argjson cfg "$CFG" --argjson gtp "$GTP" \
            '.metadata.spec == $spec and .metadata.seed == $seed
             and .metadata.coda_guidance_scale == $gamma
             and .metadata.cfg_guidance_scale == $cfg
             and .metadata.guide_t_percent == $gtp
             and .metadata.conflict_projection_alpha == 0
             and .metadata.conflict_projection_kappa_cap == null' \
            "${dataset_dir}/guidance_metrics/guidance_metrics_summary.json" > /dev/null; then
            echo "Validation failed: guidance metadata differs for ${dataset_dir}." >&2
            jq '.metadata' "${dataset_dir}/guidance_metrics/guidance_metrics_summary.json" >&2
            return 1
        fi
        if [[ "$mode" == "i1g1" ]]; then
            if ! jq -e --argjson strength "$PROTOTYPE_INIT_STRENGTH" \
                '.prototype_initialization_strength == $strength' \
                "${dataset_dir}/prompt_config.json" > /dev/null; then
                echo "Validation failed: prototype strength differs for ${dataset_dir}." >&2
                jq '.' "${dataset_dir}/prompt_config.json" >&2
                return 1
            fi
        else
            if ! jq -e '.prototype_initialization_strength == null' \
                "${dataset_dir}/prompt_config.json" > /dev/null; then
                echo "Validation failed: I0 dataset unexpectedly uses prototype initialization: ${dataset_dir}." >&2
                jq '.' "${dataset_dir}/prompt_config.json" >&2
                return 1
            fi
        fi
    fi
}

p4_dataset() {
    local spec=$1 seed=$2 mode=$3 prompt=$4
    local count dataset
    count="$(jq --arg spec "$spec" --argjson seed "$seed" --arg mode "$mode" --arg prompt "$prompt" \
        '[.[] | select(.spec==$spec and .generation_seed==$seed and .visual_mode==$mode and .prompt_condition==$prompt)] | length' \
        "$P4_MANIFEST")"
    if [[ "$count" -ne 1 ]]; then
        echo "Expected one P4 manifest entry for ${spec}/${seed}/${mode}/${prompt}, found ${count}" >&2
        exit 1
    fi
    dataset="$(jq -r --arg spec "$spec" --argjson seed "$seed" --arg mode "$mode" --arg prompt "$prompt" \
        '.[] | select(.spec==$spec and .generation_seed==$seed and .visual_mode==$mode and .prompt_condition==$prompt) | .dataset_dir' \
        "$P4_MANIFEST")"
    validate_dataset "$spec" "$dataset" false "$seed" "$mode" || {
        echo "P4 G0 dataset is incomplete: ${dataset}" >&2
        exit 1
    }
    printf '%s\n' "$dataset"
}

generate_g1() {
    local spec=$1 seed=$2 mode=$3 prompt=$4
    local condition="${mode}_${prompt}"
    local dirname="p5_continuous_guidance_runs/${RUN_ID}/seed_${seed}/generated_images_${condition}"
    local output_dir="$(experiment_root "$spec" "$GUIDANCE_GAMMA")/${dirname}"
    if [[ -e "$output_dir" ]]; then
        if validate_dataset "$spec" "$output_dir" true "$seed" "$mode"; then
            echo "==> Reusing ${spec}/${condition}, generation seed ${seed}" >&2
            printf '%s\n' "$output_dir"
            return
        fi
        if [[ "$ARCHIVE_INCOMPLETE_GENERATION" != "true" ]]; then
            echo "Incomplete P5 generation exists: ${output_dir}" >&2
            echo "Set ARCHIVE_INCOMPLETE_GENERATION=true to archive and rerun it." >&2
            exit 1
        fi
        local archive="${META_ROOT}/incomplete_archives/${spec}/seed_${seed}/${condition}/$(date -u +%Y%m%dT%H%M%SZ)"
        mkdir -p "$(dirname "$archive")"
        mv -- "$output_dir" "$archive"
    fi
    if [[ "$RUN_GENERATION" != "true" ]]; then
        echo "Missing P5 G1 dataset with RUN_GENERATION=false: ${output_dir}" >&2
        exit 1
    fi

    local args=(
        --local_model_path "$MODEL_FOLDER"
        --spec "$spec" --IPC "$IPC"
        --n_neighbors "$N_NEIGHBORS" --min_cluster_size "$MIN_CLUSTER_SIZE"
        --sample_step "$SAMPLE_STEP" --denoising_factor "$DF"
        --guideTPercent "$GTP" --cfg_guidance_scale "$CFG"
        --CoDA_guidance_scale "$GUIDANCE_GAMMA" --seed "$seed" --generate_images
        --measure_guidance_conflict
        --experiment_method "$condition" --generated_images_dirname "$dirname"
        --generation_cluster_indices_file "${PREPARED_DIR}/${spec}_cluster_indices.json"
        --base_prompt_template '{class_name}'
    )
    if [[ "$mode" == "i1g1" ]]; then
        args+=(--prototype_initialization_strength "$PROTOTYPE_INIT_STRENGTH")
    fi
    if [[ "$prompt" != "label" ]]; then
        args+=(
            --use_cluster_captions
            --cluster_caption_file "${PREPARED_DIR}/${spec}_dcs_captions.json"
            --cluster_caption_prompt_template "$PROMPT_TEMPLATE"
        )
    fi
    if [[ "$prompt" == "shuffled" ]]; then
        args+=(--cluster_caption_source_map "${PREPARED_DIR}/${spec}_shuffled_source_map.json")
    fi
    echo "==> Generating ${spec}/${condition}, generation seed ${seed}" >&2
    python CoDA_main.py "${args[@]}" >&2
    validate_dataset "$spec" "$output_dir" true "$seed" "$mode" || {
        echo "P5 generation did not produce a complete dataset and guidance record: ${output_dir}" >&2
        exit 1
    }
    printf '%s\n' "$output_dir"
}

manifest='[]'
for spec in $SPECS; do
    [[ -f "${PREPARED_DIR}/${spec}_cluster_indices.json" ]] || {
        echo "P4 prepared inputs do not include ${spec}" >&2
        exit 1
    }
    for seed in $GENERATION_SEEDS; do
        for mode in i0g0 i1g0; do
            for prompt in label correct shuffled; do
                dataset="$(p4_dataset "$spec" "$seed" "$mode" "$prompt")"
                manifest="$(jq --arg spec "$spec" --argjson seed "$seed" --arg mode "$mode" \
                    --arg prompt "$prompt" --arg dataset "$dataset" \
                    '. + [{spec:$spec,generation_seed:$seed,visual_mode:$mode,prompt_condition:$prompt,dataset_dir:$dataset}]' \
                    <<< "$manifest")"
            done
        done
        for mode in i0g1 i1g1; do
            for prompt in label correct shuffled; do
                dataset="$(generate_g1 "$spec" "$seed" "$mode" "$prompt")"
                manifest="$(jq --arg spec "$spec" --argjson seed "$seed" --arg mode "$mode" \
                    --arg prompt "$prompt" --arg dataset "$dataset" \
                    '. + [{spec:$spec,generation_seed:$seed,visual_mode:$mode,prompt_condition:$prompt,dataset_dir:$dataset}]' \
                    <<< "$manifest")"
            done
        done
    done
done
printf '%s\n' "$manifest" > "$MANIFEST_FILE"

if [[ "$RUN_EVALUATION" == "true" ]]; then
    python evaluate_p5_continuous_guidance.py \
        --prepared-dir "$PREPARED_DIR" \
        --generation-manifest "$MANIFEST_FILE" \
        --output-dir "$ANALYSIS_DIR" \
        --dino-model "$DINO_MODEL" --clip-model "$CLIP_MODEL" \
        --batch-size "$BATCH_SIZE" --bootstrap-samples "$BOOTSTRAP_SAMPLES" \
        --permutation-samples "$PERMUTATION_SAMPLES" --random-seed "$RANDOM_SEED"
    cp "${ANALYSIS_DIR}/summary.json" "$COMPLETE_FILE"
fi

echo "P5 experiment complete: ${META_ROOT}"
echo "Analysis: ${ANALYSIS_DIR}/summary.json"
echo "Plot: ${ANALYSIS_DIR}/p5_continuous_guidance.png"
