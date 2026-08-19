#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ASSIGNMENT_ROOT="${ASSIGNMENT_ROOT:-${REPO_ROOT}/caption_interface_audit_runs/ipc10_cluster_assignments_v0}"
NETTE_DATA_ROOT="${NETTE_DATA_ROOT:-/linxi/dataset/VLCP/ImageNette}"
WOOF_DATA_ROOT="${WOOF_DATA_ROOT:-/linxi/dataset/VLCP/ImageWoof}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/caption_interface_audit_runs/nearest_cluster_member_montages_ipc10_v0}"

if [[ ! -f "$ASSIGNMENT_ROOT/nette/latent_assignments.csv" || \
      ! -f "$ASSIGNMENT_ROOT/woof/latent_assignments.csv" ]]; then
  echo "IPC10 assignments are incomplete; building them before rendering nearest members."
  ASSIGNMENT_ROOT="$ASSIGNMENT_ROOT" \
  NETTE_DATA_ROOT="$NETTE_DATA_ROOT" \
  WOOF_DATA_ROOT="$WOOF_DATA_ROOT" \
    bash "$REPO_ROOT/experiments/text_supervision_factorial/run_random_cluster_member_montages_ipc10.sh"
fi

python "$REPO_ROOT/experiments/text_supervision_factorial/make_random_cluster_member_montages.py" \
  --assignment "nette=$ASSIGNMENT_ROOT/nette/latent_assignments.csv" \
  --assignment "woof=$ASSIGNMENT_ROOT/woof/latent_assignments.csv" \
  --data-root "nette=$NETTE_DATA_ROOT" \
  --data-root "woof=$WOOF_DATA_ROOT" \
  --output-dir "$OUTPUT_DIR" \
  --clusters-per-class 10 \
  --images-per-cluster 5 \
  --cluster-selection all \
  --member-selection nearest \
  --tile-size 224

echo "IPC10 nearest-member audit complete: $OUTPUT_DIR/index.html"
