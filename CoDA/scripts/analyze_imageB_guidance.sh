#!/usr/bin/env bash
set -euo pipefail

RUN_ID="${RUN_ID:?Set RUN_ID to the completed ImageB run directory name.}"
IPC="${IPC:-10}"
N_NEIGHBORS="${N_NEIGHBORS:-85}"
MIN_CLUSTER_SIZE="${MIN_CLUSTER_SIZE:-55}"
SAMPLE_STEP="${SAMPLE_STEP:-25}"
DF="${DF:-1.0}"
GTP="${GTP:-0.9}"
GAMMA="${GAMMA:-0.05}"
GENERATION_SEEDS="${GENERATION_SEEDS:-0 1}"

EXPERIMENT_DIR="./results/imageB/Step-${SAMPLE_STEP}/IPC-${IPC}/DF-${DF}-GTP-${GTP}-gamma-${GAMMA}/n_${N_NEIGHBORS}_s_${MIN_CLUSTER_SIZE}"
RUN_DIR="${EXPERIMENT_DIR}/imageB_runs/${RUN_ID}"
TRAINED_ROOT="./trained_results/imageB_runs/${RUN_ID}"

python analyze_guidance_by_class.py \
    --run_dir "$RUN_DIR" \
    --trained_root "$TRAINED_ROOT" \
    --generation_seeds $GENERATION_SEEDS

echo "ImageB class-level diagnostics: ${RUN_DIR}/class_guidance_diagnostics"
