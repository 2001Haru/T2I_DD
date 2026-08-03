#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export TOKENIZERS_PARALLELISM=false

SPECS="${SPECS:-imageA imageB imageC}"
GENERATION_SEEDS="${GENERATION_SEEDS:-0 1}"
P1_RUN_ID="${P1_RUN_ID:-p1_cluster_recoverability_v0}"
P2P3_RUN_ID="${P2P3_RUN_ID:-p2p3_language_cluster_v0}"
RUN_ID="${P4_RUN_ID:-p4_text_execution_v0}"
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
PROTOTYPE_INIT_STRENGTH="${PROTOTYPE_INIT_STRENGTH:-0.7}"
SHUFFLE_SHIFT="${SHUFFLE_SHIFT:-1}"
RIDGE_ALPHA="${RIDGE_ALPHA:-1.0}"
BATCH_SIZE="${BATCH_SIZE:-64}"
BOOTSTRAP_SAMPLES="${BOOTSTRAP_SAMPLES:-5000}"
RANDOM_SEED="${RANDOM_SEED:-20260803}"
RUN_GENERATION="${RUN_GENERATION:-true}"
RUN_EVALUATION="${RUN_EVALUATION:-true}"
RESUME="${RESUME:-false}"
ARCHIVE_INCOMPLETE_GENERATION="${ARCHIVE_INCOMPLETE_GENERATION:-false}"
PROMPT_TEMPLATE="${P4_DCS_PROMPT_TEMPLATE:-An natural photo of a {class_name}, {caption}, centered object.}"

P1_RUN_DIR="./results/p1_cluster_recoverability_runs/${P1_RUN_ID}"
P2P3_RUN_DIR="./results/p2p3_language_cluster_runs/${P2P3_RUN_ID}"
META_ROOT="./results/p4_text_execution_runs/${RUN_ID}"
PREPARED_DIR="${META_ROOT}/prepared"
ANALYSIS_DIR="${META_ROOT}/analysis"
MANIFEST_FILE="${META_ROOT}/generation_manifest.json"
CONFIG_FILE="${META_ROOT}/run_config.txt"
COMPLETE_FILE="${META_ROOT}/complete.json"

CONFIG_CONTENT="SPECS=${SPECS}
GENERATION_SEEDS=${GENERATION_SEEDS}
P1_RUN_ID=${P1_RUN_ID}
P2P3_RUN_ID=${P2P3_RUN_ID}
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
PROTOTYPE_INIT_STRENGTH=${PROTOTYPE_INIT_STRENGTH}
SHUFFLE_SHIFT=${SHUFFLE_SHIFT}
RIDGE_ALPHA=${RIDGE_ALPHA}
PROMPT_TEMPLATE=${PROMPT_TEMPLATE}
BATCH_SIZE=${BATCH_SIZE}
BOOTSTRAP_SAMPLES=${BOOTSTRAP_SAMPLES}
RANDOM_SEED=${RANDOM_SEED}"

for path in \
    "${P1_RUN_DIR}/assignments.csv" \
    "${P1_RUN_DIR}/feature_cache/dino.npz" \
    "${P1_RUN_DIR}/feature_cache/clip.npz" \
    "${P2P3_RUN_DIR}/replayed_dcs_summaries.json"; do
    if [[ ! -f "$path" ]]; then
        echo "Missing required P4 input: ${path}" >&2
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
        echo "P4 run exists; set RESUME=true or choose another P4_RUN_ID: ${RUN_ID}" >&2
        exit 1
    fi
    if [[ ! -f "$CONFIG_FILE" || "$(<"$CONFIG_FILE")" != "$CONFIG_CONTENT" ]]; then
        echo "Resume configuration differs from ${CONFIG_FILE}" >&2
        exit 1
    fi
    if [[ -f "$COMPLETE_FILE" ]]; then
        echo "P4 run already complete: ${META_ROOT}"
        cat "${ANALYSIS_DIR}/summary.json"
        exit 0
    fi
else
    mkdir -p "$META_ROOT"
    printf '%s\n' "$CONFIG_CONTENT" > "$CONFIG_FILE"
fi

read -r -a SPEC_ARRAY <<< "$SPECS"
if [[ ! -f "${PREPARED_DIR}/preparation_summary.json" ]]; then
    mkdir -p "$PREPARED_DIR"
    python prepare_p4_text_execution.py \
        --p1-run-dir "$P1_RUN_DIR" \
        --p2p3-run-dir "$P2P3_RUN_DIR" \
        --output-dir "$PREPARED_DIR" \
        --specs "${SPEC_ARRAY[@]}" \
        --shuffle-shift "$SHUFFLE_SHIFT" \
        --ridge-alpha "$RIDGE_ALPHA" \
        --dino-model "$DINO_MODEL" --clip-model "$CLIP_MODEL"
else
    echo "==> Reusing frozen P4 prompt pairs and real-image probes"
    [[ "$(sha256sum "${P1_RUN_DIR}/assignments.csv" | awk '{print $1}')" == \
        "$(jq -r '.assignments_sha256' "${PREPARED_DIR}/preparation_summary.json")" ]] || {
        echo "P1 assignments changed after P4 preparation; choose a new P4_RUN_ID." >&2
        exit 1
    }
    [[ "$(sha256sum "${P2P3_RUN_DIR}/replayed_dcs_summaries.json" | awk '{print $1}')" == \
        "$(jq -r '.dcs_summaries_sha256' "${PREPARED_DIR}/preparation_summary.json")" ]] || {
        echo "P2/P3 summaries changed after P4 preparation; choose a new P4_RUN_ID." >&2
        exit 1
    }
    [[ "$(sha256sum "${P1_RUN_DIR}/feature_cache/dino.npz" | awk '{print $1}')" == \
        "$(jq -r '.dino_feature_cache_sha256' "${PREPARED_DIR}/preparation_summary.json")" ]] || {
        echo "P1 DINO cache changed after P4 preparation; choose a new P4_RUN_ID." >&2
        exit 1
    }
    [[ "$(sha256sum "${P1_RUN_DIR}/feature_cache/clip.npz" | awk '{print $1}')" == \
        "$(jq -r '.clip_feature_cache_sha256' "${PREPARED_DIR}/preparation_summary.json")" ]] || {
        echo "P1 CLIP cache changed after P4 preparation; choose a new P4_RUN_ID." >&2
        exit 1
    }
fi

experiment_root() {
    local spec=$1
    echo "./results/${spec}/Step-${SAMPLE_STEP}/IPC-${IPC}/DF-${DF}-GTP-${GTP}-gamma-0.0/n_${N_NEIGHBORS}_s_${MIN_CLUSTER_SIZE}"
}

validate_dataset() {
    local spec=$1
    local dataset_dir=$2
    local expected actual
    expected="$(jq '[.[] | length] | add' "${PREPARED_DIR}/${spec}_cluster_indices.json")"
    actual="$(find "$dataset_dir" -mindepth 2 -maxdepth 2 -type f -name '*.png' | wc -l)"
    [[ "$actual" -eq "$expected" ]] || return 1
    compgen -G "${dataset_dir}/prompt_records_gpu*.json" > /dev/null
}

generate_condition() {
    local spec=$1
    local seed=$2
    local visual_mode=$3
    local prompt_condition=$4
    local condition="${visual_mode}_${prompt_condition}"
    local output_dirname="p4_text_execution_runs/${RUN_ID}/seed_${seed}/generated_images_${condition}"
    local output_dir="$(experiment_root "$spec")/${output_dirname}"
    if [[ -e "$output_dir" ]]; then
        if validate_dataset "$spec" "$output_dir"; then
            echo "==> Reusing ${spec}/${condition}, generation seed ${seed}"
            return
        fi
        if [[ "$ARCHIVE_INCOMPLETE_GENERATION" != "true" ]]; then
            echo "Incomplete P4 generation exists: ${output_dir}" >&2
            echo "Set ARCHIVE_INCOMPLETE_GENERATION=true to archive and rerun it." >&2
            exit 1
        fi
        local archive="${META_ROOT}/incomplete_archives/${spec}/seed_${seed}/${condition}/$(date -u +%Y%m%dT%H%M%SZ)"
        mkdir -p "$(dirname "$archive")"
        mv -- "$output_dir" "$archive"
    fi
    if [[ "$RUN_GENERATION" != "true" ]]; then
        echo "Missing P4 dataset with RUN_GENERATION=false: ${output_dir}" >&2
        exit 1
    fi

    local args=(
        --local_model_path "$MODEL_FOLDER"
        --spec "$spec" --IPC "$IPC"
        --n_neighbors "$N_NEIGHBORS" --min_cluster_size "$MIN_CLUSTER_SIZE"
        --sample_step "$SAMPLE_STEP" --denoising_factor "$DF"
        --guideTPercent "$GTP" --cfg_guidance_scale "$CFG"
        --CoDA_guidance_scale 0.0 --seed "$seed" --generate_images
        --experiment_method "$condition" --generated_images_dirname "$output_dirname"
        --generation_cluster_indices_file "${PREPARED_DIR}/${spec}_cluster_indices.json"
        --base_prompt_template '{class_name}'
    )
    if [[ "$visual_mode" == "i1g0" ]]; then
        args+=(--prototype_initialization_strength "$PROTOTYPE_INIT_STRENGTH")
    fi
    if [[ "$prompt_condition" != "label" ]]; then
        args+=(
            --use_cluster_captions
            --cluster_caption_file "${PREPARED_DIR}/${spec}_dcs_captions.json"
            --cluster_caption_prompt_template "$PROMPT_TEMPLATE"
        )
    fi
    if [[ "$prompt_condition" == "shuffled" ]]; then
        args+=(--cluster_caption_source_map "${PREPARED_DIR}/${spec}_shuffled_source_map.json")
    fi
    echo "==> Generating ${spec}/${condition}, generation seed ${seed}"
    python CoDA_main.py "${args[@]}"
    if ! validate_dataset "$spec" "$output_dir"; then
        echo "P4 generation did not produce its complete expected subset: ${output_dir}" >&2
        exit 1
    fi
}

manifest='[]'
for spec in $SPECS; do
    for seed in $GENERATION_SEEDS; do
        for visual_mode in i0g0 i1g0; do
            for prompt_condition in label correct shuffled; do
                generate_condition "$spec" "$seed" "$visual_mode" "$prompt_condition"
                dataset_dir="$(experiment_root "$spec")/p4_text_execution_runs/${RUN_ID}/seed_${seed}/generated_images_${visual_mode}_${prompt_condition}"
                manifest="$(jq \
                    --arg spec "$spec" --argjson seed "$seed" \
                    --arg visual "$visual_mode" --arg prompt "$prompt_condition" \
                    --arg dataset "$dataset_dir" \
                    '. + [{spec:$spec,generation_seed:$seed,visual_mode:$visual,prompt_condition:$prompt,dataset_dir:$dataset}]' \
                    <<< "$manifest")"
            done
        done
    done
done
printf '%s\n' "$manifest" > "$MANIFEST_FILE"

if [[ "$RUN_EVALUATION" == "true" ]]; then
    EVAL_ARGS=(evaluate_p4_text_execution.py \
        --prepared-dir "$PREPARED_DIR" \
        --generation-manifest "$MANIFEST_FILE" \
        --output-dir "$ANALYSIS_DIR" \
        --dino-model "$DINO_MODEL" --clip-model "$CLIP_MODEL" \
        --batch-size "$BATCH_SIZE" --bootstrap-samples "$BOOTSTRAP_SAMPLES" \
        --random-seed "$RANDOM_SEED")
    if [[ "$RESUME" == "true" ]]; then
        EVAL_ARGS+=(--resume)
    fi
    python "${EVAL_ARGS[@]}"
    cp "${ANALYSIS_DIR}/summary.json" "$COMPLETE_FILE"
fi

echo "P4 experiment complete: ${META_ROOT}"
echo "Analysis: ${ANALYSIS_DIR}/summary.json"
echo "Plot: ${ANALYSIS_DIR}/p4_text_execution.png"
