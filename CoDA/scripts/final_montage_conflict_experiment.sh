#!/usr/bin/env bash
set -euo pipefail

export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION="${PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION:-python}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"

MODEL_FOLDER="${MODEL_FOLDER:-/linxi/models/CoDA/SDXL-Refiner}"
VLM_MODEL="${VLM_MODEL:-/linxi/models/CoDA/llava-1.5-7b-hf}"
IMAGENET_TRAIN_FOLDER="${IMAGENET_TRAIN_FOLDER:-/zhangchi/imagenet_512/images}"
IMAGENET_VAL_FOLDER="${IMAGENET_VAL_FOLDER:-/linxi/dataset/imagenet/validation/val}"
SPECS="${SPECS:-imageA imageB imageC}"
GENERATION_SEEDS="${GENERATION_SEEDS:-0 1}"
EVAL_SEED="${EVAL_SEED:-0}"
IPC="${IPC:-10}"
N_NEIGHBORS="${N_NEIGHBORS:-85}"
MIN_CLUSTER_SIZE="${MIN_CLUSTER_SIZE:-55}"
SAMPLE_STEP="${SAMPLE_STEP:-25}"
DF="${DF:-1.0}"
GTP="${GTP:-0.9}"
GAMMA="${GAMMA:-0.05}"
SOFT_ALPHA="${SOFT_ALPHA:-0.5}"
KAPPA_CAP="${KAPPA_CAP:-0.3}"
RUN_ID="${FINAL_MONTAGE_RUN_ID:-final_montage_conflict_$(date -u +%Y%m%dT%H%M%SZ)}"
RESUME_RUN="${RESUME_RUN:-false}"
REUSE_EXISTING="${REUSE_EXISTING:-true}"
PREPARE_MISSING_CLUSTERS="${PREPARE_MISSING_CLUSTERS:-true}"
RUN_GENERATION="${RUN_GENERATION:-true}"
RUN_DOWNSTREAM_TRAINING="${RUN_DOWNSTREAM_TRAINING:-true}"

IMAGEA_REFERENCE_RUN_ID="${IMAGEA_REFERENCE_RUN_ID:-imageA_multiview_v0}"
IMAGEB_REFERENCE_RUN_ID="${IMAGEB_REFERENCE_RUN_ID:-final_prompt_controls_v0}"
IMAGEB_REFERENCE_SEED="${IMAGEB_REFERENCE_SEED:-1}"

MONTAGE_INSTRUCTION='Identify recurring physical attributes of the {class_name} across the provided visual examples. Write one concise sentence of at most 35 words for a single-image generation prompt. Mention only shared shape, parts, posture, colors, textures, and distinctive features. Do not mention images, examples, panels, tiles, titles, layout, backgrounds, people, other objects, or one-off details.'
SDXL_PROMPT_TEMPLATE='An natural photo of a {class_name}, {caption}, centered object.'
METHODS=(coda_baseline montage_common_mode montage_soft_alpha_0p5 montage_kappa_cap_0p3)

META_ROOT="./results/final_montage_conflict_runs/${RUN_ID}"
SAVE_ROOT="./trained_results/final_montage_conflict_runs/${RUN_ID}"

experiment_dir() {
    local spec=$1
    echo "./results/${spec}/Step-${SAMPLE_STEP}/IPC-${IPC}/DF-${DF}-GTP-${GTP}-gamma-${GAMMA}/n_${N_NEIGHBORS}_s_${MIN_CLUSTER_SIZE}"
}

spec_run_dir() {
    local spec=$1
    echo "$(experiment_dir "$spec")/final_montage_conflict_runs/${RUN_ID}"
}

upper_key() {
    printf '%s' "$1" | tr '[:lower:]' '[:upper:]' | tr '.-' '__'
}

explicit_override() {
    local spec=$1
    local seed=$2
    local method=$3
    local kind=$4
    local variable
    variable="$(upper_key "${spec}_seed${seed}_${method}_${kind}")"
    printf '%s' "${!variable:-}"
}

default_reference_dataset() {
    local spec=$1
    local seed=$2
    local method=$3
    local base
    base="$(experiment_dir "$spec")"
    if [[ "$spec" == "imageA" ]]; then
        base="${base}/multiview_caption_runs/${IMAGEA_REFERENCE_RUN_ID}/seed_${seed}"
        case "$method" in
            coda_baseline) echo "${base}/generated_images_coda_baseline" ;;
            montage_common_mode) echo "${base}/generated_images_montage_common_mode" ;;
        esac
    elif [[ "$spec" == "imageB" && "$seed" == "$IMAGEB_REFERENCE_SEED" ]]; then
        base="${base}/final_prompt_controls/${IMAGEB_REFERENCE_RUN_ID}/seed_${seed}"
        case "$method" in
            coda_baseline) echo "${base}/generated_images_class_prompt" ;;
            montage_common_mode) echo "${base}/generated_images_montage_caption" ;;
        esac
    fi
    return 0
}

default_reference_result() {
    local spec=$1
    local seed=$2
    local method=$3
    if [[ "$spec" == "imageA" ]]; then
        local base="./trained_results/multiview_caption_runs/imageA/${IMAGEA_REFERENCE_RUN_ID}/seed_${seed}"
        case "$method" in
            coda_baseline) echo "${base}/coda_baseline-resnet_ap" ;;
            montage_common_mode) echo "${base}/montage_common_mode-resnet_ap" ;;
        esac
    elif [[ "$spec" == "imageB" && "$seed" == "$IMAGEB_REFERENCE_SEED" ]]; then
        local base="./trained_results/final_prompt_controls/${IMAGEB_REFERENCE_RUN_ID}/imageB/seed_${seed}"
        case "$method" in
            coda_baseline) echo "${base}/class_prompt-resnet_ap" ;;
            montage_common_mode) echo "${base}/montage_caption-resnet_ap" ;;
        esac
    fi
    return 0
}

require_completed_dataset() {
    local path=$1
    python - "$path" "$IPC" <<'PY'
import os
import sys

root, ipc = sys.argv[1], int(sys.argv[2])
classes = [
    name for name in os.listdir(root)
    if len(name) == 9 and name.startswith("n") and name[1:].isdigit()
    and os.path.isdir(os.path.join(root, name))
]
if len(classes) != 10:
    raise SystemExit(f"Expected 10 class directories in {root}, found {len(classes)}")
for class_id in classes:
    for index in range(ipc):
        path = os.path.join(root, class_id, f"{index}.png")
        if not os.path.isfile(path):
            raise SystemExit(f"Missing generated image: {path}")
PY
}

link_directory() {
    local source=$1
    local destination=$2
    if [[ -e "$destination" || -L "$destination" ]]; then
        if [[ "$RESUME_RUN" == "true" && "$(realpath "$destination")" == "$(realpath "$source")" ]]; then
            return
        fi
        echo "Refusing to replace an existing experiment artifact: ${destination}" >&2
        exit 1
    fi
    mkdir -p "$(dirname "$destination")"
    ln -s "$(realpath "$source")" "$destination"
}

VISIBLE_GPU_COUNT="$(python -c 'import torch; print(torch.cuda.device_count())')"
if [[ "$VISIBLE_GPU_COUNT" != "2" ]]; then
    echo "This paired protocol requires exactly 2 visible GPUs; found ${VISIBLE_GPU_COUNT}." >&2
    exit 1
fi

if [[ -e "$META_ROOT" || -e "$SAVE_ROOT" ]]; then
    if [[ "$RESUME_RUN" != "true" ]]; then
        echo "Final montage run already exists; set RESUME_RUN=true to continue: ${RUN_ID}" >&2
        exit 1
    fi
else
    mkdir -p "$META_ROOT" "$SAVE_ROOT"
fi
mkdir -p "$META_ROOT" "$SAVE_ROOT"

CONFIG_FILE="${META_ROOT}/run_config.txt"
CONFIG_CONTENT="SPECS=${SPECS}
GENERATION_SEEDS=${GENERATION_SEEDS}
EVAL_SEED=${EVAL_SEED}
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
SOFT_ALPHA=${SOFT_ALPHA}
KAPPA_CAP=${KAPPA_CAP}
MONTAGE_INSTRUCTION=${MONTAGE_INSTRUCTION}
SDXL_PROMPT_TEMPLATE=${SDXL_PROMPT_TEMPLATE}
IMAGEA_REFERENCE_RUN_ID=${IMAGEA_REFERENCE_RUN_ID}
IMAGEB_REFERENCE_RUN_ID=${IMAGEB_REFERENCE_RUN_ID}
IMAGEB_REFERENCE_SEED=${IMAGEB_REFERENCE_SEED}"
if [[ -f "$CONFIG_FILE" ]]; then
    if [[ "$(<"$CONFIG_FILE")" != "$CONFIG_CONTENT" ]]; then
        echo "Resume configuration differs from ${CONFIG_FILE}" >&2
        exit 1
    fi
else
    printf '%s\n' "$CONFIG_CONTENT" > "$CONFIG_FILE"
fi

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
    raise SystemExit(
        f"Expected classifier seeds {expected} in {path}, found {seeds}"
    )
PY
}

ensure_clusters() {
    local spec=$1
    local cluster_pattern="./results/clusterfile/${spec}/${IPC}_n_${N_NEIGHBORS}_s_${MIN_CLUSTER_SIZE}_saved_clusters_*.pkl"
    if compgen -G "$cluster_pattern" > /dev/null; then
        echo "==> Reusing ${spec} cluster artifacts"
        return
    fi
    if [[ "$PREPARE_MISSING_CLUSTERS" != "true" ]]; then
        echo "Missing cluster artifacts for ${spec}: ${cluster_pattern}" >&2
        exit 1
    fi
    local run_dir
    run_dir="$(spec_run_dir "$spec")"
    echo "==> Preparing missing features and clusters for ${spec}"
    python CoDA_main.py \
        --dataset_dir "$IMAGENET_TRAIN_FOLDER" --local_model_path "$MODEL_FOLDER" \
        --spec "$spec" --IPC "$IPC" \
        --n_neighbors "$N_NEIGHBORS" --min_cluster_size "$MIN_CLUSTER_SIZE" \
        --sample_step "$SAMPLE_STEP" --denoising_factor "$DF" \
        --guideTPercent "$GTP" --CoDA_guidance_scale "$GAMMA" \
        --calcu_features --calcu_cluster --cluster_detial --cluster_logger \
        --experiment_method "final_montage_preparation" \
        --timing_file "${run_dir}/timings/preparation.json"
}

caption_file_for_spec() {
    local spec=$1
    local experiment_dir
    experiment_dir="$(experiment_dir "$spec")"
    if [[ "$REUSE_EXISTING" == "true" && "$spec" == "imageA" ]]; then
        local candidate="${experiment_dir}/multiview_caption_runs/${IMAGEA_REFERENCE_RUN_ID}/captions_montage_common_mode.json"
        if [[ -f "$candidate" ]]; then
            echo "$candidate"
            return
        fi
    fi
    if [[ "$REUSE_EXISTING" == "true" && "$spec" == "imageB" ]]; then
        local candidate="${experiment_dir}/final_prompt_controls/${IMAGEB_REFERENCE_RUN_ID}/seed_${IMAGEB_REFERENCE_SEED}/captions_montage_common_mode.json"
        if [[ -f "$candidate" ]]; then
            echo "$candidate"
            return
        fi
    fi

    local run_dir caption_file
    run_dir="$(spec_run_dir "$spec")"
    caption_file="${run_dir}/captions_montage_common_mode.json"
    if [[ ! -f "$caption_file" ]]; then
        echo "==> Generating montage captions for ${spec}" >&2
        # Keep command output out of the command substitution that receives
        # the final caption path.
        python CoDA_main.py \
            --dataset_dir "$IMAGENET_TRAIN_FOLDER" \
            --spec "$spec" --IPC "$IPC" \
            --n_neighbors "$N_NEIGHBORS" --min_cluster_size "$MIN_CLUSTER_SIZE" \
            --sample_step "$SAMPLE_STEP" --denoising_factor "$DF" \
            --guideTPercent "$GTP" --CoDA_guidance_scale "$GAMMA" \
            --generate_cluster_captions \
            --cluster_caption_model_path "$VLM_MODEL" \
            --cluster_caption_file "$caption_file" \
            --cluster_caption_image_mode montage_neighbors \
            --cluster_caption_neighbor_count 4 --cluster_caption_max_words 35 \
            --cluster_caption_montage_dir "${run_dir}/caption_montages" \
            --cluster_caption_instruction "$MONTAGE_INSTRUCTION" \
            --experiment_method "final_montage_caption_generation" \
            --timing_file "${run_dir}/timings/caption_generation.json" 1>&2
    fi
    echo "$caption_file"
}

generate_variant() {
    local spec=$1
    local seed=$2
    local method=$3
    local caption_file=$4
    local run_dir output_dir output_dirname timing_file
    run_dir="$(spec_run_dir "$spec")"
    output_dir="${run_dir}/seed_${seed}/generated_images_${method}"
    output_dirname="final_montage_conflict_runs/${RUN_ID}/seed_${seed}/generated_images_${method}"
    timing_file="${run_dir}/seed_${seed}/timings/${method}.json"

    if [[ -e "$output_dir" || -L "$output_dir" ]]; then
        require_completed_dataset "$output_dir"
        echo "==> Reusing ${spec}/${method}, generation seed ${seed}"
        return
    fi

    if [[ "$REUSE_EXISTING" == "true" ]]; then
        local reference
        reference="$(explicit_override "$spec" "$seed" "$method" DATA_DIR)"
        if [[ -z "$reference" ]]; then
            reference="$(default_reference_dataset "$spec" "$seed" "$method")"
        fi
        if [[ -n "$reference" && -d "$reference" ]]; then
            require_completed_dataset "$reference"
            link_directory "$reference" "$output_dir"
            echo "==> Linked existing ${spec}/${method}, generation seed ${seed}"
            return
        fi
    fi

    if [[ "$RUN_GENERATION" != "true" ]]; then
        echo "Missing generated dataset with RUN_GENERATION=false: ${output_dir}" >&2
        exit 1
    fi

    local method_args=()
    case "$method" in
        coda_baseline)
            ;;
        montage_common_mode)
            method_args+=(--use_cluster_captions --cluster_caption_file "$caption_file")
            method_args+=(--cluster_caption_prompt_template "$SDXL_PROMPT_TEMPLATE")
            ;;
        montage_soft_alpha_0p5)
            method_args+=(--use_cluster_captions --cluster_caption_file "$caption_file")
            method_args+=(--cluster_caption_prompt_template "$SDXL_PROMPT_TEMPLATE")
            method_args+=(--conflict_projection_alpha "$SOFT_ALPHA")
            ;;
        montage_kappa_cap_0p3)
            method_args+=(--use_cluster_captions --cluster_caption_file "$caption_file")
            method_args+=(--cluster_caption_prompt_template "$SDXL_PROMPT_TEMPLATE")
            method_args+=(--conflict_projection_kappa_cap "$KAPPA_CAP")
            ;;
        *)
            echo "Unknown final montage method: ${method}" >&2
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
        --seed "$seed" --generate_images --measure_guidance_conflict \
        --experiment_method "$method" --generated_images_dirname "$output_dirname" \
        --timing_file "$timing_file" \
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
    if [[ "$REUSE_EXISTING" == "true" ]]; then
        local reference
        reference="$(explicit_override "$spec" "$seed" "$method" RESULT_DIR)"
        if [[ -z "$reference" ]]; then
            reference="$(default_reference_result "$spec" "$seed" "$method")"
        fi
        if [[ -n "$reference" && -f "${reference}/per_class_accuracy_all_seeds.json" ]]; then
            require_completed_result "${reference}/per_class_accuracy_all_seeds.json"
            link_directory "$reference" "$save_dir"
            echo "==> Linked existing classifier ${spec}/${method}, generation seed ${seed}"
            return
        fi
    fi
    if [[ -e "$save_dir" || -e "${save_dir}_gpu${EVAL_SEED}" || -e "${save_dir}_gpu$((EVAL_SEED + 1))" ]]; then
        echo "Incomplete classifier output exists; refusing to mix runs: ${save_dir}" >&2
        exit 1
    fi
    if [[ "$RUN_DOWNSTREAM_TRAINING" != "true" ]]; then
        echo "Missing classifier result with RUN_DOWNSTREAM_TRAINING=false: ${result_file}" >&2
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
        --experiment_method "$method" --tag "final_montage_${spec}_seed_${seed}"
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
    caption_file="$(caption_file_for_spec "$spec")"
    if [[ ! -f "$caption_file" ]]; then
        echo "Missing montage caption file for ${spec}: ${caption_file}" >&2
        exit 1
    fi

    for seed in $GENERATION_SEEDS; do
        for method in "${METHODS[@]}"; do
            generate_variant "$spec" "$seed" "$method" "$caption_file"
        done

        comparison_dir="${run_dir}/seed_${seed}/guidance_comparison"
        if [[ ! -e "$comparison_dir" ]]; then
            python compare_guidance_metrics.py \
                --input "baseline=${run_dir}/seed_${seed}/generated_images_coda_baseline/guidance_metrics/guidance_metrics_summary.json" \
                --input "montage=${run_dir}/seed_${seed}/generated_images_montage_common_mode/guidance_metrics/guidance_metrics_summary.json" \
                --input "montage_soft=${run_dir}/seed_${seed}/generated_images_montage_soft_alpha_0p5/guidance_metrics/guidance_metrics_summary.json" \
                --input "montage_kappa=${run_dir}/seed_${seed}/generated_images_montage_kappa_cap_0p3/guidance_metrics/guidance_metrics_summary.json" \
                --output_dir "$comparison_dir"
        fi

        for method in "${METHODS[@]}"; do
            train_variant "$spec" "$seed" "$method"
        done
    done
done

SUMMARY_DIR="${SAVE_ROOT}/summary"
if [[ ! -f "${SUMMARY_DIR}/experiment_summary.json" ]]; then
    if [[ -e "$SUMMARY_DIR" ]]; then
        echo "Incomplete summary directory exists; refusing to treat it as complete: ${SUMMARY_DIR}" >&2
        exit 1
    fi
    python summarize_final_montage_conflict.py \
        --output_dir "$SUMMARY_DIR" --trained_root "$SAVE_ROOT" \
        --specs $SPECS --generation_seeds $GENERATION_SEEDS
else
    echo "==> Reusing final montage summary: ${SUMMARY_DIR}"
fi

echo "Final montage conflict experiment completed: ${RUN_ID}"
echo "Generated-data metadata: ${META_ROOT}"
echo "Training and summary root: ${SAVE_ROOT}"
