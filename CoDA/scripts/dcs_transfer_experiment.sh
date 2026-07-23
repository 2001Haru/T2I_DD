#!/usr/bin/env bash
set -euo pipefail
shopt -s inherit_errexit

export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION="${PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION:-python}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"

MODEL_FOLDER="${MODEL_FOLDER:-/linxi/models/CoDA/SDXL-Refiner}"
VLM_MODEL="${VLM_MODEL:-/linxi/models/CoDA/llava-1.5-7b-hf}"
IMAGENET_TRAIN_FOLDER="${IMAGENET_TRAIN_FOLDER:-/zhangchi/imagenet_512/images}"
IMAGENET_VAL_FOLDER="${IMAGENET_VAL_FOLDER:-/linxi/dataset/imagenet/validation/val}"
SPECS="${SPECS:-imageA imageB imageC}"
GENERATION_SEEDS="${GENERATION_SEEDS:-0 1}"
METHODS="${METHODS:-dcs}"
EVAL_SEED="${EVAL_SEED:-0}"
IPC="${IPC:-10}"
N_NEIGHBORS="${N_NEIGHBORS:-85}"
MIN_CLUSTER_SIZE="${MIN_CLUSTER_SIZE:-55}"
SAMPLE_STEP="${SAMPLE_STEP:-25}"
DF="${DF:-1.0}"
GTP="${GTP:-0.9}"
GAMMA="${GAMMA:-0.05}"
DCS_THRESHOLD="${DCS_THRESHOLD:-0.7}"
DCS_TOP_K="${DCS_TOP_K:-30}"
DCS_MAX_NEW_TOKENS="${DCS_MAX_NEW_TOKENS:-128}"
DCS_MAX_CAPTION_WORDS="${DCS_MAX_CAPTION_WORDS:-0}"
DCS_MAX_IMAGES_PER_CLUSTER="${DCS_MAX_IMAGES_PER_CLUSTER:-0}"
DCS_CAPTION_BATCH_SIZE="${DCS_CAPTION_BATCH_SIZE:-1}"
DCS_CAPTION_GPU_COUNT="${DCS_CAPTION_GPU_COUNT:-1}"
DCS_CAPTION_VISIBLE_DEVICES="${DCS_CAPTION_VISIBLE_DEVICES:-0}"
DCS_PROMPT_TEMPLATE="${DCS_PROMPT_TEMPLATE:-}"
DCS_INSTRUCTION="${DCS_INSTRUCTION:-}"
if [[ -z "$DCS_PROMPT_TEMPLATE" ]]; then
    DCS_PROMPT_TEMPLATE='An natural photo of a {class_name}, {caption}, centered object.'
fi
if [[ -z "$DCS_INSTRUCTION" ]]; then
    DCS_INSTRUCTION='Describe the physical appearance of the {class_name} in the image. Include details about its shape, posture, color, and any distinct features.'
fi
RUN_ID="${DCS_TRANSFER_RUN_ID:-dcs_transfer_$(date -u +%Y%m%dT%H%M%SZ)}"
RESUME_RUN="${RESUME_RUN:-false}"
PREPARE_MISSING_CLUSTERS="${PREPARE_MISSING_CLUSTERS:-true}"
RUN_DCS_CAPTIONING="${RUN_DCS_CAPTIONING:-true}"
RUN_GENERATION="${RUN_GENERATION:-true}"
RUN_DOWNSTREAM_TRAINING="${RUN_DOWNSTREAM_TRAINING:-true}"
ARCHIVE_INCOMPLETE_CLASSIFIERS="${ARCHIVE_INCOMPLETE_CLASSIFIERS:-$RESUME_RUN}"

META_ROOT="./results/dcs_transfer_runs/${RUN_ID}"
SAVE_ROOT="./trained_results/dcs_transfer_runs/${RUN_ID}"

experiment_dir() {
    local spec=$1
    echo "./results/${spec}/Step-${SAMPLE_STEP}/IPC-${IPC}/DF-${DF}-GTP-${GTP}-gamma-${GAMMA}/n_${N_NEIGHBORS}_s_${MIN_CLUSTER_SIZE}"
}

spec_run_dir() {
    local spec=$1
    echo "$(experiment_dir "$spec")/dcs_transfer_runs/${RUN_ID}"
}

require_completed_dataset() {
    local path=$1
    python - "$path" "$IPC" <<'PY'
import os
import sys

root, ipc = sys.argv[1], int(sys.argv[2])
classes = sorted(
    name for name in os.listdir(root)
    if len(name) == 9 and name.startswith("n") and name[1:].isdigit()
    and os.path.isdir(os.path.join(root, name))
)
if len(classes) != 10:
    raise SystemExit(f"Expected 10 class directories in {root}, found {len(classes)}")
for class_id in classes:
    for index in range(ipc):
        path = os.path.join(root, class_id, f"{index}.png")
        if not os.path.isfile(path):
            raise SystemExit(f"Missing generated image: {path}")
PY
}

require_completed_result() {
    local path=$1
    python - "$path" "$EVAL_SEED" <<'PY'
import json
import os
import sys

path, seed_start = sys.argv[1], int(sys.argv[2])
if not os.path.isfile(path):
    raise SystemExit(f"Missing classifier result: {path}")
with open(path, "r", encoding="utf-8") as file:
    payload = json.load(file)
seeds = sorted(int(run["training_seed"]) for run in payload.get("runs", []))
expected = [seed_start, seed_start + 1]
if seeds != expected:
    raise SystemExit(f"Expected classifier seeds {expected} in {path}, found {seeds}")
PY
}

if [[ -e "$META_ROOT" || -e "$SAVE_ROOT" ]]; then
    if [[ "$RESUME_RUN" != "true" ]]; then
        echo "DCS transfer run already exists; set RESUME_RUN=true: ${RUN_ID}" >&2
        exit 1
    fi
fi
mkdir -p "$META_ROOT" "$SAVE_ROOT"

VISIBLE_GPU_COUNT="$(python -c 'import torch; print(torch.cuda.device_count())')"
if [[ "$VISIBLE_GPU_COUNT" != "2" ]]; then
    echo "This protocol requires exactly two visible GPUs; found ${VISIBLE_GPU_COUNT}." >&2
    exit 1
fi
if [[ "$DCS_CAPTION_GPU_COUNT" -lt 1 || "$DCS_CAPTION_GPU_COUNT" -gt "$VISIBLE_GPU_COUNT" ]]; then
    echo "DCS_CAPTION_GPU_COUNT must be between 1 and ${VISIBLE_GPU_COUNT}." >&2
    exit 1
fi
python -c 'from nltk.corpus import stopwords; stopwords.words("english")' >/dev/null

CONFIG_FILE="${META_ROOT}/run_config.txt"
CONFIG_CONTENT="SPECS=${SPECS}
GENERATION_SEEDS=${GENERATION_SEEDS}
METHODS=${METHODS}
MODEL_FOLDER=${MODEL_FOLDER}
VLM_MODEL=${VLM_MODEL}
IMAGENET_TRAIN_FOLDER=${IMAGENET_TRAIN_FOLDER}
IMAGENET_VAL_FOLDER=${IMAGENET_VAL_FOLDER}
IPC=${IPC}
N_NEIGHBORS=${N_NEIGHBORS}
MIN_CLUSTER_SIZE=${MIN_CLUSTER_SIZE}
SAMPLE_STEP=${SAMPLE_STEP}
DF=${DF}
GTP=${GTP}
GAMMA=${GAMMA}
DCS_THRESHOLD=${DCS_THRESHOLD}
DCS_TOP_K=${DCS_TOP_K}
DCS_MAX_NEW_TOKENS=${DCS_MAX_NEW_TOKENS}
DCS_MAX_CAPTION_WORDS=${DCS_MAX_CAPTION_WORDS}
DCS_MAX_IMAGES_PER_CLUSTER=${DCS_MAX_IMAGES_PER_CLUSTER}
DCS_CAPTION_BATCH_SIZE=${DCS_CAPTION_BATCH_SIZE}
DCS_PROMPT_TEMPLATE=${DCS_PROMPT_TEMPLATE}
DCS_INSTRUCTION=${DCS_INSTRUCTION}"
if [[ -f "$CONFIG_FILE" ]]; then
    if [[ "$(<"$CONFIG_FILE")" != "$CONFIG_CONTENT" ]]; then
        echo "Resume configuration differs from ${CONFIG_FILE}" >&2
        exit 1
    fi
else
    printf '%s\n' "$CONFIG_CONTENT" > "$CONFIG_FILE"
fi

ensure_clusters() {
    local spec=$1
    local feature_file="./results/clusterfile/${spec}/original_features_cache.pkl_0"
    local center_file="./results/clusterfile/${spec}/${IPC}_n_${N_NEIGHBORS}_s_${MIN_CLUSTER_SIZE}_saved_clusters_0.pkl"
    if [[ -f "$feature_file" && -f "$center_file" ]]; then
        echo "==> Reusing ${spec} feature and cluster artifacts"
        return
    fi
    if [[ "$PREPARE_MISSING_CLUSTERS" != "true" ]]; then
        echo "Missing feature or cluster artifacts for ${spec}" >&2
        exit 1
    fi
    echo "==> Preparing ${spec} features and clusters"
    python CoDA_main.py \
        --dataset_dir "$IMAGENET_TRAIN_FOLDER" --local_model_path "$MODEL_FOLDER" \
        --spec "$spec" --IPC "$IPC" \
        --n_neighbors "$N_NEIGHBORS" --min_cluster_size "$MIN_CLUSTER_SIZE" \
        --sample_step "$SAMPLE_STEP" --denoising_factor "$DF" \
        --guideTPercent "$GTP" --CoDA_guidance_scale "$GAMMA" \
        --calcu_features --calcu_cluster --cluster_detial --cluster_logger \
        --experiment_method "dcs_preparation" \
        --timing_file "$(spec_run_dir "$spec")/timings/preparation.json"
}

build_dcs_file() {
    local spec=$1
    local run_dir cache_dir output_file
    run_dir="$(spec_run_dir "$spec")"
    cache_dir="./results/dcs_caption_cache/${spec}/vlcp_dcs_class_aware"
    output_file="${run_dir}/captions_dcs_t${DCS_THRESHOLD}_k${DCS_TOP_K}_m${DCS_MAX_IMAGES_PER_CLUSTER}.json"
    if [[ -f "$output_file" ]]; then
        echo "==> Reusing ${spec} DCS manifest" >&2
        echo "$output_file"
        return
    fi
    if [[ "$RUN_DCS_CAPTIONING" == "true" ]]; then
        echo "==> Captioning ${spec} real images on ${DCS_CAPTION_GPU_COUNT} GPU(s)" >&2
        local caption_args=(
            dcs_caption.py caption
            --spec "$spec" --misc-dir ./misc
            --features-cache-path "./results/clusterfile/${spec}/original_features_cache.pkl"
            --caption-cache-dir "$cache_dir"
            --specific-cluster-dir "./results/clusterfile/${spec}"
            --saved-clusters-base-name "${IPC}_n_${N_NEIGHBORS}_s_${MIN_CLUSTER_SIZE}_saved_clusters.pkl"
            --ipc "$IPC" --max-images-per-cluster "$DCS_MAX_IMAGES_PER_CLUSTER"
            --model "$VLM_MODEL" --instruction "$DCS_INSTRUCTION"
            --max-new-tokens "$DCS_MAX_NEW_TOKENS" --batch-size "$DCS_CAPTION_BATCH_SIZE"
        )
        if [[ "$DCS_CAPTION_GPU_COUNT" == "1" ]]; then
            if ! CUDA_VISIBLE_DEVICES="$DCS_CAPTION_VISIBLE_DEVICES" \
                    python "${caption_args[@]}" 1>&2; then
                echo "DCS image captioning failed for ${spec}; refusing to build a partial manifest." >&2
                return 1
            fi
        else
            if ! CUDA_VISIBLE_DEVICES="$DCS_CAPTION_VISIBLE_DEVICES" \
                    torchrun --standalone --nproc_per_node="$DCS_CAPTION_GPU_COUNT" \
                    "${caption_args[@]}" 1>&2; then
                echo "DCS image captioning failed for ${spec}; refusing to build a partial manifest." >&2
                return 1
            fi
        fi
    fi
    echo "==> Building ${spec} VLCP-style DCS captions" >&2
    if ! python dcs_caption.py build \
            --spec "$spec" --misc-dir ./misc \
            --features-cache-path "./results/clusterfile/${spec}/original_features_cache.pkl" \
            --caption-cache-dir "$cache_dir" \
            --specific-cluster-dir "./results/clusterfile/${spec}" \
            --saved-clusters-base-name "${IPC}_n_${N_NEIGHBORS}_s_${MIN_CLUSTER_SIZE}_saved_clusters.pkl" \
            --ipc "$IPC" --max-images-per-cluster "$DCS_MAX_IMAGES_PER_CLUSTER" \
            --threshold "$DCS_THRESHOLD" --top-k "$DCS_TOP_K" \
            --max-caption-words "$DCS_MAX_CAPTION_WORDS" \
            --output "$output_file" 1>&2; then
        echo "DCS manifest construction failed for ${spec}." >&2
        return 1
    fi
    echo "$output_file"
}

generate_variant() {
    local spec=$1
    local seed=$2
    local method=$3
    local dcs_file=$4
    local run_dir output_dir output_dirname
    run_dir="$(spec_run_dir "$spec")"
    output_dir="${run_dir}/seed_${seed}/generated_images_${method}"
    output_dirname="dcs_transfer_runs/${RUN_ID}/seed_${seed}/generated_images_${method}"
    if [[ -e "$output_dir" ]]; then
        require_completed_dataset "$output_dir"
        echo "==> Reusing ${spec}/${method}, generation seed ${seed}"
        return
    fi
    if [[ "$RUN_GENERATION" != "true" ]]; then
        echo "Missing generated dataset: ${output_dir}" >&2
        exit 1
    fi

    local method_args=()
    case "$method" in
        coda_baseline)
            ;;
        dcs)
            method_args+=(--use_cluster_captions --cluster_caption_file "$dcs_file")
            method_args+=(--cluster_caption_prompt_template "$DCS_PROMPT_TEMPLATE")
            ;;
        *)
            echo "Unknown DCS transfer method: ${method}" >&2
            exit 1
            ;;
    esac
    echo "==> Generating ${spec}/${method}, generation seed ${seed}"
    python CoDA_main.py \
        --local_model_path "$MODEL_FOLDER" \
        --spec "$spec" --IPC "$IPC" \
        --n_neighbors "$N_NEIGHBORS" --min_cluster_size "$MIN_CLUSTER_SIZE" \
        --sample_step "$SAMPLE_STEP" --denoising_factor "$DF" \
        --guideTPercent "$GTP" --CoDA_guidance_scale "$GAMMA" \
        --seed "$seed" --generate_images \
        --experiment_method "$method" --generated_images_dirname "$output_dirname" \
        --timing_file "${run_dir}/seed_${seed}/timings/${method}.json" \
        "${method_args[@]}"
}

train_variant() {
    local spec=$1
    local seed=$2
    local method=$3
    local run_dir dataset_dir save_dir result_file
    run_dir="$(spec_run_dir "$spec")"
    dataset_dir="${run_dir}/seed_${seed}/generated_images_${method}"
    save_dir="${SAVE_ROOT}/${spec}/seed_${seed}/${method}-resnet_ap"
    result_file="${save_dir}/per_class_accuracy_all_seeds.json"
    if [[ -f "$result_file" ]]; then
        require_completed_result "$result_file"
        echo "==> Reusing completed classifier ${spec}/${method}, generation seed ${seed}"
        return
    fi

    local partial_artifacts=(
        "$save_dir"
        "${save_dir}_gpu${EVAL_SEED}"
        "${save_dir}_gpu$((EVAL_SEED + 1))"
    )
    local has_partial=false
    local artifact
    for artifact in "${partial_artifacts[@]}"; do
        if [[ -e "$artifact" ]]; then
            has_partial=true
        fi
    done
    if [[ "$has_partial" == "true" ]]; then
        if [[ "$ARCHIVE_INCOMPLETE_CLASSIFIERS" != "true" ]]; then
            echo "Incomplete classifier output exists: ${save_dir}" >&2
            exit 1
        fi
        local archive_dir
        archive_dir="${SAVE_ROOT}/incomplete_classifier_archives/${spec}/seed_${seed}/${method}/$(date -u +%Y%m%dT%H%M%SZ)"
        mkdir -p "$archive_dir"
        for artifact in "${partial_artifacts[@]}"; do
            if [[ -e "$artifact" ]]; then
                mv -- "$artifact" "$archive_dir/"
            fi
        done
        echo "==> Archived incomplete classifier to ${archive_dir}"
    fi
    if [[ "$RUN_DOWNSTREAM_TRAINING" != "true" ]]; then
        echo "Missing classifier result: ${result_file}" >&2
        exit 1
    fi
    echo "==> Training ${spec}/${method}, generation seed ${seed}"
    python ./test/train.py \
        --dataset_dir "$dataset_dir" "$IMAGENET_VAL_FOLDER" \
        -d imagenet --spec "$spec" --nclass 10 --size 256 --ipc "$IPC" \
        -n resnet_ap --depth 10 --save-dir "$save_dir" \
        --seed "$EVAL_SEED" --workers 12 \
        --n_neighbors "$N_NEIGHBORS" --min_cluster_size "$MIN_CLUSTER_SIZE" \
        --timing_file "${run_dir}/seed_${seed}/timings/${method}.json" \
        --experiment_method "$method" --tag "dcs_${spec}_seed_${seed}"
}

for spec in $SPECS; do
    run_dir="$(spec_run_dir "$spec")"
    mkdir -p "$run_dir"
    if [[ ! -f "${run_dir}/dataset_validation.json" ]]; then
        python validate_imagenet_subset.py \
            --train_dir "$IMAGENET_TRAIN_FOLDER" --val_dir "$IMAGENET_VAL_FOLDER" \
            --spec "$spec" --nclass 10 --min_train_per_class "$IPC" \
            --output "${run_dir}/dataset_validation.json"
    fi
    ensure_clusters "$spec"
    if ! dcs_file="$(build_dcs_file "$spec")"; then
        echo "DCS preparation failed for ${spec}; stopping the experiment." >&2
        exit 1
    fi
    if [[ ! -f "$dcs_file" ]]; then
        echo "Missing DCS manifest for ${spec}: ${dcs_file}" >&2
        exit 1
    fi
    for seed in $GENERATION_SEEDS; do
        for method in $METHODS; do
            generate_variant "$spec" "$seed" "$method" "$dcs_file"
            train_variant "$spec" "$seed" "$method"
        done
    done
done

SUMMARY_DIR="${SAVE_ROOT}/summary"
if [[ ! -f "${SUMMARY_DIR}/experiment_summary.json" ]]; then
    python summarize_dcs_transfer.py \
        --trained-root "$SAVE_ROOT" --output-dir "$SUMMARY_DIR" \
        --specs $SPECS --generation-seeds $GENERATION_SEEDS --methods $METHODS
else
    echo "==> Reusing DCS transfer summary: ${SUMMARY_DIR}"
fi

echo "DCS transfer experiment completed: ${RUN_ID}"
echo "DCS metadata root: ${META_ROOT}"
echo "Training and summary root: ${SAVE_ROOT}"
