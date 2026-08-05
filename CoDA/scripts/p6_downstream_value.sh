#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION="${PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION:-python}"
export TOKENIZERS_PARALLELISM=false

SPECS="${SPECS:-imageA imageB imageC}"
GENERATION_SEEDS="${GENERATION_SEEDS:-0 1}"
P4_RUN_ID="${P4_RUN_ID:-p4_text_execution_v0}"
P5_RUN_ID="${P5_RUN_ID:-p5_continuous_guidance_v0}"
RUN_ID="${P6_RUN_ID:-p6_downstream_value_v0}"
MODEL_FOLDER="${MODEL_FOLDER:-/linxi/models/CoDA/SDXL-Refiner}"
IMAGENET_VAL_FOLDER="${IMAGENET_VAL_FOLDER:-/linxi/dataset/imagenet/validation/val}"
IPC="${IPC:-10}"
N_NEIGHBORS="${N_NEIGHBORS:-85}"
MIN_CLUSTER_SIZE="${MIN_CLUSTER_SIZE:-55}"
SAMPLE_STEP="${SAMPLE_STEP:-25}"
DF="${DF:-1.0}"
GTP="${GTP:-0.9}"
CFG="${CFG:-5.0}"
GUIDANCE_GAMMA="${P6_GUIDANCE_GAMMA:-0.05}"
PROTOTYPE_INIT_STRENGTH="${PROTOTYPE_INIT_STRENGTH:-0.7}"
EVAL_SEED="${EVAL_SEED:-0}"
TRAIN_GPU_GROUPS="${TRAIN_GPU_GROUPS:-0,1 2,3}"
FILLER_CUDA_VISIBLE_DEVICES="${FILLER_CUDA_VISIBLE_DEVICES:-0}"
TRAIN_WORKERS="${P6_TRAIN_WORKERS:-4}"
VAL_WORKERS="${P6_VAL_WORKERS:-2}"
RUN_FILLER_GENERATION="${RUN_FILLER_GENERATION:-true}"
RUN_MATCHED_GENERATION="${RUN_MATCHED_GENERATION:-true}"
RUN_DATASET_ASSEMBLY="${RUN_DATASET_ASSEMBLY:-true}"
RUN_DOWNSTREAM_TRAINING="${RUN_DOWNSTREAM_TRAINING:-true}"
RUN_SUMMARY="${RUN_SUMMARY:-true}"
RESUME="${RESUME:-false}"
ARCHIVE_INCOMPLETE_FILLERS="${ARCHIVE_INCOMPLETE_FILLERS:-false}"
ARCHIVE_INCOMPLETE_MATCHED_GENERATION="${ARCHIVE_INCOMPLETE_MATCHED_GENERATION:-false}"
ARCHIVE_INCOMPLETE_DATASETS="${ARCHIVE_INCOMPLETE_DATASETS:-false}"
ARCHIVE_INCOMPLETE_CLASSIFIERS="${ARCHIVE_INCOMPLETE_CLASSIFIERS:-false}"
MATCHED_LABEL_TEMPLATE="${P6_MATCHED_LABEL_TEMPLATE:-}"
if [[ -z "$MATCHED_LABEL_TEMPLATE" ]]; then
    MATCHED_LABEL_TEMPLATE='An natural photo of a {class_name}, centered object.'
fi

P4_ROOT="./results/p4_text_execution_runs/${P4_RUN_ID}"
P5_ROOT="./results/p5_continuous_guidance_runs/${P5_RUN_ID}"
SOURCE_MANIFEST="${P5_ROOT}/generation_manifest.json"
PREPARED_DIR="${P4_ROOT}/prepared"
META_ROOT="./results/p6_downstream_value_runs/${RUN_ID}"
MATCHED_SOURCE_MANIFEST="${META_ROOT}/matched_source_manifest.json"
FILLER_MANIFEST="${META_ROOT}/filler_manifest.json"
DATASET_ROOT="${META_ROOT}/datasets"
DATASET_MANIFEST="${DATASET_ROOT}/dataset_manifest.json"
MATCHED_DATASET_MANIFEST="${DATASET_ROOT}/matched_dataset_manifest.json"
SAVE_ROOT="./trained_results/p6_downstream_value_runs/${RUN_ID}"
SUMMARY_DIR="${SAVE_ROOT}/summary"
CONFIG_FILE="${META_ROOT}/run_config.txt"
COMPLETE_FILE="${META_ROOT}/complete.json"

BASE_CONFIG_CONTENT="SPECS=${SPECS}
GENERATION_SEEDS=${GENERATION_SEEDS}
P4_RUN_ID=${P4_RUN_ID}
P5_RUN_ID=${P5_RUN_ID}
MODEL_FOLDER=${MODEL_FOLDER}
IMAGENET_VAL_FOLDER=${IMAGENET_VAL_FOLDER}
IPC=${IPC}
N_NEIGHBORS=${N_NEIGHBORS}
MIN_CLUSTER_SIZE=${MIN_CLUSTER_SIZE}
SAMPLE_STEP=${SAMPLE_STEP}
DF=${DF}
GTP=${GTP}
CFG=${CFG}
GUIDANCE_GAMMA=${GUIDANCE_GAMMA}
PROTOTYPE_INIT_STRENGTH=${PROTOTYPE_INIT_STRENGTH}
EVAL_SEED=${EVAL_SEED}"
CONFIG_CONTENT="${BASE_CONFIG_CONTENT}
MATCHED_LABEL_TEMPLATE=${MATCHED_LABEL_TEMPLATE}"

for required in \
    "${P4_ROOT}/complete.json" \
    "${P5_ROOT}/complete.json" \
    "$SOURCE_MANIFEST"; do
    [[ -f "$required" ]] || { echo "Missing completed P4/P5 input: ${required}" >&2; exit 1; }
done
for required in "$MODEL_FOLDER" "$IMAGENET_VAL_FOLDER"; do
    [[ -d "$required" ]] || { echo "Missing required directory: ${required}" >&2; exit 1; }
done

if [[ -e "$META_ROOT" ]]; then
    if [[ "$RESUME" != "true" ]]; then
        echo "P6 run exists; set RESUME=true or choose another P6_RUN_ID: ${RUN_ID}" >&2
        exit 1
    fi
    if [[ ! -f "$CONFIG_FILE" ]]; then
        echo "P6 resume configuration is missing: ${CONFIG_FILE}" >&2
        exit 1
    fi
    if [[ "$(<"$CONFIG_FILE")" == "$BASE_CONFIG_CONTENT" ]]; then
        printf '%s\n' "$CONFIG_CONTENT" > "$CONFIG_FILE"
        echo "==> Migrated P6 configuration to include the matched-label control"
    elif [[ "$(<"$CONFIG_FILE")" != "$CONFIG_CONTENT" ]]; then
        echo "P6 resume configuration differs from ${CONFIG_FILE}" >&2
        exit 1
    fi
else
    mkdir -p "$META_ROOT"
    printf '%s\n' "$CONFIG_CONTENT" > "$CONFIG_FILE"
fi

experiment_root() {
    local spec=$1 gamma=$2
    echo "./results/${spec}/Step-${SAMPLE_STEP}/IPC-${IPC}/DF-${DF}-GTP-${GTP}-gamma-${gamma}/n_${N_NEIGHBORS}_s_${MIN_CLUSTER_SIZE}"
}

missing_indices_file() {
    local spec=$1
    echo "${META_ROOT}/${spec}_missing_cluster_indices.json"
}

prepare_missing_indices() {
    local spec=$1 source="${PREPARED_DIR}/${spec}_cluster_indices.json"
    local output
    output="$(missing_indices_file "$spec")"
    [[ -f "$source" ]] || { echo "Missing P4 cluster indices: ${source}" >&2; exit 1; }
    jq --argjson ipc "$IPC" 'with_entries(.value = ([range(0; $ipc)] - .value))' \
        "$source" > "${output}.tmp"
    mv "${output}.tmp" "$output"
}

expected_missing() {
    jq '[.[] | length] | add' "$(missing_indices_file "$1")"
}

validate_filler() {
    local spec=$1 directory=$2 expected actual
    expected="$(expected_missing "$spec")"
    actual="$(find "$directory" -mindepth 2 -maxdepth 2 -type f \
        -path '*/n[0-9][0-9][0-9][0-9][0-9][0-9][0-9][0-9]/*.png' | wc -l)"
    [[ "$actual" -eq "$expected" ]]
}

generate_filler() {
    local spec=$1 seed=$2 regime=$3 gamma=0.0
    [[ "$regime" == *g1 ]] && gamma="$GUIDANCE_GAMMA"
    local dirname="p6_downstream_value_runs/${RUN_ID}/fillers/seed_${seed}/${regime}"
    local output_dir="$(experiment_root "$spec" "$gamma")/${dirname}"
    local expected
    expected="$(expected_missing "$spec")"
    if [[ "$expected" -eq 0 ]]; then
        mkdir -p "$output_dir"
        printf '%s\n' "$output_dir"
        return
    fi
    if [[ -e "$output_dir" ]]; then
        if validate_filler "$spec" "$output_dir"; then
            echo "==> Reusing neutral filler ${spec}/${regime}, generation seed ${seed}" >&2
            printf '%s\n' "$output_dir"
            return
        fi
        if [[ "$ARCHIVE_INCOMPLETE_FILLERS" != "true" ]]; then
            echo "Incomplete neutral filler exists: ${output_dir}" >&2
            echo "Set ARCHIVE_INCOMPLETE_FILLERS=true to archive and regenerate it." >&2
            exit 1
        fi
        local archive="${META_ROOT}/incomplete_filler_archives/${spec}/seed_${seed}/${regime}/$(date -u +%Y%m%dT%H%M%SZ)"
        mkdir -p "$(dirname "$archive")"
        mv -- "$output_dir" "$archive"
    fi
    if [[ "$RUN_FILLER_GENERATION" != "true" ]]; then
        echo "Missing P6 neutral filler with RUN_FILLER_GENERATION=false: ${output_dir}" >&2
        exit 1
    fi
    local args=(
        --local_model_path "$MODEL_FOLDER" --spec "$spec" --IPC "$IPC"
        --n_neighbors "$N_NEIGHBORS" --min_cluster_size "$MIN_CLUSTER_SIZE"
        --sample_step "$SAMPLE_STEP" --denoising_factor "$DF"
        --guideTPercent "$GTP" --cfg_guidance_scale "$CFG"
        --CoDA_guidance_scale "$gamma" --seed "$seed" --generate_images
        --experiment_method "p6_${regime}_neutral_filler"
        --generated_images_dirname "$dirname"
        --generation_cluster_indices_file "$(missing_indices_file "$spec")"
        --base_prompt_template '{class_name}'
    )
    [[ "$regime" == i1* ]] && args+=(--prototype_initialization_strength "$PROTOTYPE_INIT_STRENGTH")
    echo "==> Generating neutral filler ${spec}/${regime}, generation seed ${seed}" >&2
    CUDA_VISIBLE_DEVICES="$FILLER_CUDA_VISIBLE_DEVICES" python CoDA_main.py "${args[@]}" >&2
    validate_filler "$spec" "$output_dir" || {
        echo "Neutral filler generation is incomplete: ${output_dir}" >&2
        exit 1
    }
    printf '%s\n' "$output_dir"
}

expected_eligible() {
    jq '[.[] | length] | add' "${PREPARED_DIR}/${1}_cluster_indices.json"
}

validate_matched_source() {
    local spec=$1 directory=$2 expected actual
    expected="$(expected_eligible "$spec")"
    actual="$(find "$directory" -mindepth 2 -maxdepth 2 -type f \
        -path '*/n[0-9][0-9][0-9][0-9][0-9][0-9][0-9][0-9]/*.png' | wc -l)"
    [[ "$actual" -eq "$expected" ]] && \
        compgen -G "${directory}/prompt_records_gpu*.json" > /dev/null
}

generate_matched_source() {
    local spec=$1 seed=$2 regime=$3 gamma=0.0
    [[ "$regime" == *g1 ]] && gamma="$GUIDANCE_GAMMA"
    local dirname="p6_downstream_value_runs/${RUN_ID}/matched_sources/seed_${seed}/${regime}"
    local output_dir="$(experiment_root "$spec" "$gamma")/${dirname}"
    if [[ -e "$output_dir" ]]; then
        if validate_matched_source "$spec" "$output_dir"; then
            echo "==> Reusing matched-label source ${spec}/${regime}, generation seed ${seed}" >&2
            printf '%s\n' "$output_dir"
            return
        fi
        if [[ "$ARCHIVE_INCOMPLETE_MATCHED_GENERATION" != "true" ]]; then
            echo "Incomplete matched-label generation exists: ${output_dir}" >&2
            echo "Set ARCHIVE_INCOMPLETE_MATCHED_GENERATION=true to archive and regenerate it." >&2
            exit 1
        fi
        local archive="${META_ROOT}/incomplete_matched_archives/${spec}/seed_${seed}/${regime}/$(date -u +%Y%m%dT%H%M%SZ)"
        mkdir -p "$(dirname "$archive")"
        mv -- "$output_dir" "$archive"
    fi
    if [[ "$RUN_MATCHED_GENERATION" != "true" ]]; then
        echo "Missing matched-label source with RUN_MATCHED_GENERATION=false: ${output_dir}" >&2
        exit 1
    fi
    local args=(
        --local_model_path "$MODEL_FOLDER" --spec "$spec" --IPC "$IPC"
        --n_neighbors "$N_NEIGHBORS" --min_cluster_size "$MIN_CLUSTER_SIZE"
        --sample_step "$SAMPLE_STEP" --denoising_factor "$DF"
        --guideTPercent "$GTP" --cfg_guidance_scale "$CFG"
        --CoDA_guidance_scale "$gamma" --seed "$seed" --generate_images
        --experiment_method "p6_${regime}_matched_label"
        --generated_images_dirname "$dirname"
        --generation_cluster_indices_file "${PREPARED_DIR}/${spec}_cluster_indices.json"
        --base_prompt_template "$MATCHED_LABEL_TEMPLATE"
    )
    [[ "$regime" == i1* ]] && args+=(--prototype_initialization_strength "$PROTOTYPE_INIT_STRENGTH")
    echo "==> Generating matched-label source ${spec}/${regime}, generation seed ${seed}" >&2
    python CoDA_main.py "${args[@]}" >&2
    validate_matched_source "$spec" "$output_dir" || {
        echo "Matched-label generation is incomplete: ${output_dir}" >&2
        exit 1
    }
    printf '%s\n' "$output_dir"
}

validate_complete_dataset() {
    local directory=$1
    local classes images invalid
    classes="$(find "$directory" -mindepth 1 -maxdepth 1 -type d -name 'n????????' | wc -l)"
    images="$(find "$directory" -mindepth 2 -maxdepth 2 -type f -name '*.png' | wc -l)"
    invalid="$(find "$directory" -mindepth 1 -maxdepth 1 -type d -name 'n????????' \
        -exec bash -c '[[ "$(find "$1" -maxdepth 1 -type f -name "*.png" | wc -l)" -eq "$2" ]]' _ {} "$IPC" \; \
        -print | wc -l)"
    [[ "$classes" -eq 10 && "$images" -eq $((10 * IPC)) && "$invalid" -eq 10 ]]
}

for spec in $SPECS; do
    prepare_missing_indices "$spec"
done

filler_manifest='[]'
for spec in $SPECS; do
    for seed in $GENERATION_SEEDS; do
        for regime in i0g0 i1g0 i0g1 i1g1; do
            dataset="$(generate_filler "$spec" "$seed" "$regime")"
            filler_manifest="$(jq --arg spec "$spec" --argjson seed "$seed" \
                --arg regime "$regime" --arg dataset "$dataset" \
                '. + [{spec:$spec,generation_seed:$seed,visual_mode:$regime,dataset_dir:$dataset}]' \
                <<< "$filler_manifest")"
        done
    done
done
printf '%s\n' "$filler_manifest" > "$FILLER_MANIFEST"

matched_source_manifest='[]'
for spec in $SPECS; do
    for seed in $GENERATION_SEEDS; do
        for regime in i0g0 i1g0 i0g1 i1g1; do
            dataset="$(generate_matched_source "$spec" "$seed" "$regime")"
            matched_source_manifest="$(jq --arg spec "$spec" --argjson seed "$seed" \
                --arg regime "$regime" --arg dataset "$dataset" \
                '. + [{spec:$spec,generation_seed:$seed,visual_mode:$regime,prompt_condition:"matched_label",dataset_dir:$dataset}]' \
                <<< "$matched_source_manifest")"
        done
    done
done
printf '%s\n' "$matched_source_manifest" > "$MATCHED_SOURCE_MANIFEST"

if [[ ! -f "$DATASET_MANIFEST" ]]; then
    if [[ -e "$DATASET_ROOT" ]]; then
        if [[ "$ARCHIVE_INCOMPLETE_DATASETS" != "true" ]]; then
            echo "Incomplete P6 dataset assembly exists: ${DATASET_ROOT}" >&2
            echo "Set ARCHIVE_INCOMPLETE_DATASETS=true to archive and rebuild it." >&2
            exit 1
        fi
        archive="${META_ROOT}/incomplete_dataset_archives/$(date -u +%Y%m%dT%H%M%SZ)"
        mkdir -p "$(dirname "$archive")"
        mv -- "$DATASET_ROOT" "$archive"
    fi
    if [[ "$RUN_DATASET_ASSEMBLY" != "true" ]]; then
        echo "P6 complete datasets are missing and RUN_DATASET_ASSEMBLY=false." >&2
        exit 1
    fi
    read -r -a spec_array <<< "$SPECS"
    read -r -a seed_array <<< "$GENERATION_SEEDS"
    python prepare_p6_datasets.py \
        --source-manifest "$SOURCE_MANIFEST" --filler-manifest "$FILLER_MANIFEST" \
        --output-root "$DATASET_ROOT" --specs "${spec_array[@]}" \
        --generation-seeds "${seed_array[@]}" --prompts label correct shuffled --ipc "$IPC"
fi

if [[ ! -f "$MATCHED_DATASET_MANIFEST" ]]; then
    if [[ "$RUN_DATASET_ASSEMBLY" != "true" ]]; then
        echo "Matched-label datasets are missing and RUN_DATASET_ASSEMBLY=false." >&2
        exit 1
    fi
    read -r -a spec_array <<< "$SPECS"
    read -r -a seed_array <<< "$GENERATION_SEEDS"
    python prepare_p6_datasets.py \
        --source-manifest "$MATCHED_SOURCE_MANIFEST" --filler-manifest "$FILLER_MANIFEST" \
        --output-root "$DATASET_ROOT" --specs "${spec_array[@]}" \
        --generation-seeds "${seed_array[@]}" --prompts matched_label --ipc "$IPC" \
        --output-manifest "$MATCHED_DATASET_MANIFEST" \
        --audit-output "${DATASET_ROOT}/matched_assembly_audit.json"
fi

for spec in $SPECS; do
    for seed in $GENERATION_SEEDS; do
        for regime in i0g0 i1g0 i0g1 i1g1; do
            for prompt in label matched_label correct shuffled; do
                dataset_dir="${DATASET_ROOT}/${spec}/seed_${seed}/${regime}_${prompt}"
                validate_complete_dataset "$dataset_dir" || {
                    echo "P6 dataset validation failed: ${dataset_dir}" >&2
                    exit 1
                }
            done
        done
    done
done

require_completed_result() {
    local result=$1
    jq -e '.overall_top1 | length == 2' "$result" > /dev/null && \
        jq -e '.class_summary | length == 10' "$result" > /dev/null
}

train_cell() {
    local spec=$1 seed=$2 regime=$3 prompt=$4 gpu_group=$5
    local condition="${regime}_${prompt}"
    local dataset_dir="${DATASET_ROOT}/${spec}/seed_${seed}/${condition}"
    local save_dir="${SAVE_ROOT}/${spec}/seed_${seed}/${condition}-resnet_ap"
    local result="${save_dir}/per_class_accuracy_all_seeds.json"
    if [[ -f "$result" ]]; then
        require_completed_result "$result" || { echo "Malformed completed result: ${result}" >&2; exit 1; }
        echo "==> Reusing completed P6 classifier ${spec}/${condition}, generation seed ${seed}"
        return
    fi
    local artifacts=("$save_dir" "${save_dir}_gpu${EVAL_SEED}" "${save_dir}_gpu$((EVAL_SEED + 1))")
    local partial=false artifact
    for artifact in "${artifacts[@]}"; do [[ -e "$artifact" ]] && partial=true; done
    if [[ "$partial" == "true" ]]; then
        if [[ "$ARCHIVE_INCOMPLETE_CLASSIFIERS" != "true" ]]; then
            echo "Incomplete classifier output exists: ${save_dir}" >&2
            echo "Set ARCHIVE_INCOMPLETE_CLASSIFIERS=true to archive and rerun it." >&2
            exit 1
        fi
        local archive="${SAVE_ROOT}/incomplete_classifier_archives/${spec}/seed_${seed}/${condition}/$(date -u +%Y%m%dT%H%M%SZ)"
        mkdir -p "$archive"
        for artifact in "${artifacts[@]}"; do [[ -e "$artifact" ]] && mv -- "$artifact" "$archive/"; done
    fi
    if [[ "$RUN_DOWNSTREAM_TRAINING" != "true" ]]; then
        echo "Missing classifier result with RUN_DOWNSTREAM_TRAINING=false: ${result}" >&2
        exit 1
    fi
    echo "==> Training P6 ${spec}/${condition}, generation seed ${seed} on GPUs ${gpu_group}"
    CUDA_VISIBLE_DEVICES="$gpu_group" python ./test/train.py \
        --dataset_dir "$dataset_dir" "$IMAGENET_VAL_FOLDER" \
        -d imagenet --spec "$spec" --nclass 10 --size 256 --ipc "$IPC" \
        -n resnet_ap --depth 10 --save-dir "$save_dir" \
        --seed "$EVAL_SEED" --workers "$TRAIN_WORKERS" --val-workers "$VAL_WORKERS" \
        --n_neighbors "$N_NEIGHBORS" --min_cluster_size "$MIN_CLUSTER_SIZE" \
        --experiment_method "$condition" --tag "p6_${spec}_generation_seed_${seed}"
    require_completed_result "$result" || { echo "P6 classifier did not complete: ${result}" >&2; exit 1; }
}

read -r -a gpu_groups <<< "$TRAIN_GPU_GROUPS"
if [[ "${#gpu_groups[@]}" -eq 0 ]]; then
    echo "TRAIN_GPU_GROUPS selects no GPU groups." >&2
    exit 1
fi
declare -A scheduled_gpu_ids=()
for group in "${gpu_groups[@]}"; do
    if [[ ! "$group" =~ ^[0-9]+,[0-9]+$ ]]; then
        echo "Each TRAIN_GPU_GROUPS entry must contain two GPU IDs, for example 0,1: ${group}" >&2
        exit 1
    fi
    IFS=',' read -r first_gpu second_gpu <<< "$group"
    if [[ "$first_gpu" == "$second_gpu" ]]; then
        echo "A training GPU group cannot repeat the same GPU: ${group}" >&2
        exit 1
    fi
    for gpu_id in "$first_gpu" "$second_gpu"; do
        if [[ -n "${scheduled_gpu_ids[$gpu_id]:-}" ]]; then
            echo "Training GPU ${gpu_id} occurs in more than one concurrent group." >&2
            exit 1
        fi
        scheduled_gpu_ids[$gpu_id]=true
    done
done

cells=()
for spec in $SPECS; do
    for seed in $GENERATION_SEEDS; do
        for regime in i0g0 i1g0 i0g1 i1g1; do
            for prompt in label matched_label correct shuffled; do
                cells+=("${spec}|${seed}|${regime}|${prompt}")
            done
        done
    done
done

cell_index=0
while [[ "$cell_index" -lt "${#cells[@]}" ]]; do
    pids=()
    labels=()
    for gpu_group in "${gpu_groups[@]}"; do
        [[ "$cell_index" -lt "${#cells[@]}" ]] || break
        IFS='|' read -r spec seed regime prompt <<< "${cells[$cell_index]}"
        train_cell "$spec" "$seed" "$regime" "$prompt" "$gpu_group" &
        pids+=("$!")
        labels+=("${spec}/seed_${seed}/${regime}_${prompt} on ${gpu_group}")
        cell_index=$((cell_index + 1))
    done

    batch_failed=false
    for index in "${!pids[@]}"; do
        if ! wait "${pids[$index]}"; then
            echo "P6 classifier cell failed: ${labels[$index]}" >&2
            batch_failed=true
        fi
    done
    if [[ "$batch_failed" == "true" ]]; then
        echo "P6 stopped after the current concurrent batch. Completed sibling cells are reusable." >&2
        exit 1
    fi
done

if [[ "$RUN_SUMMARY" == "true" ]]; then
    read -r -a spec_array <<< "$SPECS"
    read -r -a seed_array <<< "$GENERATION_SEEDS"
    python summarize_p6_downstream_value.py \
        --trained-root "$SAVE_ROOT" --output-dir "$SUMMARY_DIR" \
        --specs "${spec_array[@]}" --generation-seeds "${seed_array[@]}"
    cp "${SUMMARY_DIR}/summary.json" "$COMPLETE_FILE"
fi

echo "P6 experiment complete: ${RUN_ID}"
echo "Datasets: ${DATASET_ROOT}"
echo "Training: ${SAVE_ROOT}"
echo "Summary: ${SUMMARY_DIR}/summary.json"
