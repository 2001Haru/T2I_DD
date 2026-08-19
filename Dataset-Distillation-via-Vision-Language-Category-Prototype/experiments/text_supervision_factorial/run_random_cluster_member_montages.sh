#!/usr/bin/env bash
set -euo pipefail

ASSIGNMENT_ROOT="${ASSIGNMENT_ROOT:-./schedule_matched_followup_runs/prototype_checkpoint_covariance_v0/retention_inputs}"
NETTE_DATA_ROOT="${NETTE_DATA_ROOT:-/linxi/dataset/VLCP/ImageNette}"
WOOF_DATA_ROOT="${WOOF_DATA_ROOT:-/linxi/dataset/VLCP/ImageWoof}"
OUTPUT_DIR="${OUTPUT_DIR:-./caption_interface_audit_runs/random_cluster_member_montages_v0}"
SAMPLE_SEED="${SAMPLE_SEED:-20260819}"

python experiments/text_supervision_factorial/make_random_cluster_member_montages.py \
  --assignment "nette=${ASSIGNMENT_ROOT}/nette/latent_assignments.csv" \
  --assignment "woof=${ASSIGNMENT_ROOT}/woof/latent_assignments.csv" \
  --data-root "nette=${NETTE_DATA_ROOT}" \
  --data-root "woof=${WOOF_DATA_ROOT}" \
  --output-dir "${OUTPUT_DIR}" \
  --seed "${SAMPLE_SEED}" \
  --clusters-per-class 5 \
  --images-per-cluster 5 \
  --tile-size 224
