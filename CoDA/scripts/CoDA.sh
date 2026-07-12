#nvidia-smi
set -euo pipefail

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
export PYTHONPATH="$SCRIPT_DIR:${PYTHONPATH:-}"


#IMAGENET_TRAIN_FOLDER="/root/autodl-tmp/datasets/ImageNet"
#IMAGENET_VAL_FOLDER="/root/autodl-tmp/datasets/ImageNet/validation"
#MODEL_FOLDER="/root/autodl-tmp/model/SDXL-Refiner"
#VLM_MODEL="llava-hf/llava-1.5-7b-hf"


MODEL_FOLDER="/linxi/models/CoDA/SDXL-Refiner"
VLM_MODEL="/linxi/models/CoDA/llava-1.5-7b-hf"

IMAGENET_TRAIN_FOLDER="/zhangchi/imagenet_512/images"
IMAGENET_VAL_FOLDER="/linxi/dataset/imagenet/validation/val"

# Keep the original CoDA pipeline as the default baseline. External environment
# variables take precedence, so stages can be safely reused after an interruption.
CALCULATE_FEATURES="${CALCULATE_FEATURES:-true}"
CALCULATE_CLUSTER="${CALCULATE_CLUSTER:-true}"
GENERATE_IMAGES="${GENERATE_IMAGES:-true}"
GENERATE_CLUSTER_CAPTIONS="${GENERATE_CLUSTER_CAPTIONS:-false}"
USE_CLUSTER_CAPTIONS="${USE_CLUSTER_CAPTIONS:-false}"
OVERWRITE_CLUSTER_CAPTIONS="${OVERWRITE_CLUSTER_CAPTIONS:-false}"
VLM_METHOD_TAG="${VLM_METHOD_TAG:-vlm_caption_class_focused}"
if [[ -z "${VLM_CAPTION_INSTRUCTION:-}" ]]; then
    VLM_CAPTION_INSTRUCTION='Describe the physical appearance of the {class_name} in the image. Include details about its shape, posture, color, and any distinct features.'
fi
if [[ -z "${SDXL_CAPTION_PROMPT_TEMPLATE:-}" ]]; then
    SDXL_CAPTION_PROMPT_TEMPLATE='An natural photo of a {class_name}, {caption}, centered object.'
fi
RUN_GENERATION_PIPELINE="${RUN_GENERATION_PIPELINE:-true}"
RUN_DOWNSTREAM_TRAINING="${RUN_DOWNSTREAM_TRAINING:-true}"

run_experiment() {
    local run_step1=${1:-true}
    local flag_features=${2:-false}
    local flag_cluster=${3:-false}
    local flag_generate=${4:-false}
    local flag_caption=${5:-false}
    local use_cluster_captions=${6:-false}
    local run_step2=${7:-true}
    local method_tag="coda_baseline"
    local generated_images_dirname="generated_images"
    if [[ "$use_cluster_captions" == "true" ]]; then
        method_tag="$VLM_METHOD_TAG"
        generated_images_dirname="generated_images_${method_tag}"
    fi

    local run_stages=""
    if [[ "$flag_features" == "true" ]]; then
        run_stages="$run_stages --calcu_features"
    fi
    if [[ "$flag_cluster" == "true" ]]; then
        run_stages="$run_stages --calcu_cluster"
    fi
    if [[ "$flag_generate" == "true" ]]; then
        run_stages="$run_stages --generate_images"
    fi
    if [[ "$flag_caption" == "true" ]]; then
        run_stages="$run_stages --generate_cluster_captions"
    fi
    if [[ "$use_cluster_captions" == "true" ]]; then
        run_stages="$run_stages --use_cluster_captions"
    fi
    if [[ "$OVERWRITE_CLUSTER_CAPTIONS" == "true" ]]; then
        run_stages="$run_stages --overwrite_cluster_captions"
    fi

    if [[ "$run_step1" == "true" ]]; then

        local experiment_dir="./results/${SPEC}/Step-${timestep}/IPC-${ipc}/DF-${DF}-GTP-${GTP}-gamma-${gamma}/n_${n_neighbors}_s_${size_min}"
        local timing_file="$experiment_dir/timings/${method_tag}.json"
        local caption_file="$experiment_dir/cluster_captions_${method_tag}.json"

        python CoDA_main.py \
            --dataset_dir "$IMAGENET_TRAIN_FOLDER" --local_model_path "$MODEL_FOLDER" \
            --spec "$SPEC" \
            --IPC "$ipc" \
            --n_neighbors "$n_neighbors" --min_cluster_size "$size_min" \
            --cluster_detial --cluster_logger \
            --sample_step "$timestep" --denoising_factor "$DF" --guideTPercent "$GTP" --CoDA_guidance_scale "$gamma" \
            --cluster_caption_model_path "$VLM_MODEL" \
            --cluster_caption_file "$caption_file" \
            --cluster_caption_instruction "$VLM_CAPTION_INSTRUCTION" \
            --cluster_caption_prompt_template "$SDXL_CAPTION_PROMPT_TEMPLATE" \
            --generated_images_dirname "$generated_images_dirname" \
            --timing_file "$timing_file" --experiment_method "$method_tag" \
            $run_stages

    fi

    if [[ "$run_step2" == "true" ]]; then

        local train_data_path="./results/${SPEC}/Step-${timestep}/IPC-${ipc}/DF-${DF}-GTP-${GTP}-gamma-${gamma}/n_${n_neighbors}_s_${size_min}"
        local timing_file="$train_data_path/timings/${method_tag}.json"
        local val_data_path="$IMAGENET_VAL_FOLDER"

        local use_real_images=${8:-true}
        if [[ "$use_real_images" == "true" ]]; then
            train_data_path+="/real_images"
        else
            train_data_path+="/$generated_images_dirname"
        fi

        local train_save_dir="./trained_results/ipc${ipc}/n_${n_neighbors}_s_${size_min}/step-$timestep-DF-$DF/GTP-$GTP-gamma-$gamma/${method_tag}"

        echo "==> Testing with ResNet-AP 10..."
        python ./test/train.py --dataset_dir "$train_data_path" "$val_data_path" \
            -d imagenet --spec "$SPEC" --nclass 10  --size 256 --ipc "$ipc" \
            -n resnet_ap --depth 10  --save-dir "$train_save_dir-resnet_ap"  \
            --timing_file "$timing_file" --experiment_method "$method_tag" \
            --workers 12 \
            --n_neighbors "$n_neighbors" --min_cluster_size "$size_min" --tag test

        local baseline_timing="$train_data_path/timings/coda_baseline.json"
        local caption_timing="$train_data_path/timings/${VLM_METHOD_TAG}.json"
        if [[ -f "$baseline_timing" && -f "$caption_timing" ]]; then
            python summarize_timings.py --baseline "$baseline_timing" --caption "$caption_timing"
        fi
    fi
}

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"

ipc=10

n_neighbors=85
size_min=55

timestep=25
DF=1.0
GTP=0.9
gamma=0.05

#SPEC_LIST="imageA imageB imageC imageD imageE IDC nette"
SPEC_LIST="imageA"
for SPEC in $SPEC_LIST; do
    #                Step1  cal_features cal_cluster generate captions use_captions Step2 use_real_images
    run_experiment   "$RUN_GENERATION_PIPELINE" "$CALCULATE_FEATURES" "$CALCULATE_CLUSTER" "$GENERATE_IMAGES" "$GENERATE_CLUSTER_CAPTIONS" "$USE_CLUSTER_CAPTIONS" "$RUN_DOWNSTREAM_TRAINING" false
done

# cd /root/autodl-tmp/CoDA
# conda activate MG
# scripts/CoDA.sh
