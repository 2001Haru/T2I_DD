#!/usr/bin/env bash
set -euo pipefail

BASE_MODEL="${BASE_MODEL:-/linxi/models/VLCP/stable-diffusion-v1-5}"
NETTE_DATA_ROOT="${NETTE_DATA_ROOT:-/linxi/dataset/VLCP/ImageNette}"
WOOF_DATA_ROOT="${WOOF_DATA_ROOT:-/linxi/dataset/VLCP/ImageWoof}"
ASSIGNMENT_ROOT="${ASSIGNMENT_ROOT:-./schedule_matched_followup_runs/prototype_checkpoint_covariance_v0/retention_inputs}"
OUTPUT_DIR="${OUTPUT_DIR:-./caption_interface_audit_runs/max_cluster_caption_info_v0}"
FEATURE_CACHE_DIR="${FEATURE_CACHE_DIR:-./caption_interface_audit_runs/caption_interface_audit_v0/feature_cache}"
DEVICE="${DEVICE:-cuda}"

python - <<'PY'
import sklearn
import torch
import transformers
print(f"Dependencies: sklearn={sklearn.__version__} torch={torch.__version__} transformers={transformers.__version__}")
PY

python experiments/text_supervision_factorial/measure_max_cluster_caption_info.py \
  --base-model "${BASE_MODEL}" \
  --dataset "nette=${NETTE_DATA_ROOT}/train/nette.jsonl" \
  --dataset "woof=${WOOF_DATA_ROOT}/train/woof.jsonl" \
  --assignment "nette=${ASSIGNMENT_ROOT}/nette/latent_assignments.csv" \
  --assignment "woof=${ASSIGNMENT_ROOT}/woof/latent_assignments.csv" \
  --output-dir "${OUTPUT_DIR}" \
  --feature-cache-dir "${FEATURE_CACHE_DIR}" \
  --device "${DEVICE}" \
  --batch-size 64 \
  --split-seed 20260819 \
  --inner-folds 2 \
  --candidate-k 1 2 4 8 16 \
  --candidate-c 0.1 1.0 10.0 \
  --minimum-cluster-size 4 \
  --local-files-only
