#!/usr/bin/env bash
set -euo pipefail

export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION="${PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION:-python}"

MODEL_FOLDER="${MODEL_FOLDER:-/linxi/models/CoDA/SDXL-Refiner}"
FINAL_CONTROL_RUN_ID="${FINAL_CONTROL_RUN_ID:-final_prompt_controls_v0}"
PCS_RUN_ID="${PCS_RUN_ID:-pcs_$(date -u +%Y%m%dT%H%M%SZ)}"
PCS_GPU_A="${PCS_GPU_A:-0}"
PCS_GPU_B="${PCS_GPU_B:-1}"
PCS_BATCH_SIZE="${PCS_BATCH_SIZE:-1}"
PCS_NUM_TIMESTEPS="${PCS_NUM_TIMESTEPS:-8}"
PCS_SEED="${PCS_SEED:-20260716}"
IPC="${IPC:-10}"
N_NEIGHBORS="${N_NEIGHBORS:-85}"
MIN_CLUSTER_SIZE="${MIN_CLUSTER_SIZE:-55}"

ACCURACY_CSV="./trained_results/final_prompt_controls/${FINAL_CONTROL_RUN_ID}/summary/per_class_comparison.csv"
RUN_DIR="./results/pcs_diagnostics/${PCS_RUN_ID}"

if [[ ! -f "$ACCURACY_CSV" ]]; then
    echo "Final-control per-class accuracy CSV was not found: ${ACCURACY_CSV}" >&2
    exit 1
fi
if [[ -e "$RUN_DIR" ]]; then
    echo "Refusing to overwrite PCS diagnostic run: ${RUN_DIR}" >&2
    exit 1
fi
mkdir -p "$RUN_DIR"

echo "PCS run directory: ${RUN_DIR}"
echo "Live logs: ${RUN_DIR}/imageA.log and ${RUN_DIR}/imageB.log"

compute_spec() {
    local spec=$1
    local gpu=$2
    local seed_offset=$3
    CUDA_VISIBLE_DEVICES="$gpu" python compute_prior_compatibility.py \
        --local_model_path "$MODEL_FOLDER" \
        --output_dir "${RUN_DIR}/${spec}" \
        --spec "$spec" --ipc "$IPC" \
        --n_neighbors "$N_NEIGHBORS" --min_cluster_size "$MIN_CLUSTER_SIZE" \
        --num_timesteps "$PCS_NUM_TIMESTEPS" --batch_size "$PCS_BATCH_SIZE" \
        --seed "$((PCS_SEED + seed_offset))" --device cuda:0
}

echo "==> Computing ImageA PCS on GPU ${PCS_GPU_A}"
compute_spec imageA "$PCS_GPU_A" 0 >"${RUN_DIR}/imageA.log" 2>&1 &
pid_a=$!
echo "==> Computing ImageB PCS on GPU ${PCS_GPU_B}"
compute_spec imageB "$PCS_GPU_B" 100000 >"${RUN_DIR}/imageB.log" 2>&1 &
pid_b=$!

status_a=0
status_b=0
wait "$pid_a" || status_a=$?
wait "$pid_b" || status_b=$?
if [[ "$status_a" -ne 0 || "$status_b" -ne 0 ]]; then
    echo "PCS computation failed (ImageA=${status_a}, ImageB=${status_b})." >&2
    echo "Inspect ${RUN_DIR}/imageA.log and ${RUN_DIR}/imageB.log." >&2
    exit 1
fi

python analyze_prior_compatibility.py \
    --accuracy_csv "$ACCURACY_CSV" \
    --pcs imageA "${RUN_DIR}/imageA/pcs_per_class.csv" \
    --pcs imageB "${RUN_DIR}/imageB/pcs_per_class.csv" \
    --output_dir "${RUN_DIR}/analysis"

echo "PCS diagnostic completed: ${RUN_DIR}"
