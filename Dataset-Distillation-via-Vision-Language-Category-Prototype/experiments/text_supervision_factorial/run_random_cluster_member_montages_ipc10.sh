#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
NETTE_DATA_ROOT="${NETTE_DATA_ROOT:-/linxi/dataset/VLCP/ImageNette}"
WOOF_DATA_ROOT="${WOOF_DATA_ROOT:-/linxi/dataset/VLCP/ImageWoof}"
BASE_MODEL="${BASE_MODEL:-/linxi/models/VLCP/stable-diffusion-v1-5}"
BASE_RUN_ROOT="${BASE_RUN_ROOT:-${REPO_ROOT}/text_supervision_factorial_runs/text_supervision_factorial_2xa100_v0}"
WOOF_RUN_ROOT="${WOOF_RUN_ROOT:-${REPO_ROOT}/conditioning_interface_matrix_runs/conditioning_interface_generality_v0}"
NETTE_PROTOTYPE="${NETTE_PROTOTYPE:-${BASE_RUN_ROOT}/prototypes/text_supervision-ipc10-0.7-30-kmexpand1.json}"
WOOF_PROTOTYPE="${WOOF_PROTOTYPE:-${WOOF_RUN_ROOT}/artifacts/woof/ipc10/woof-ipc10-0.7-30-kmexpand1.json}"
ASSIGNMENT_ROOT="${ASSIGNMENT_ROOT:-${REPO_ROOT}/caption_interface_audit_runs/ipc10_cluster_assignments_v0}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/caption_interface_audit_runs/random_cluster_member_montages_ipc10_v0}"
SAMPLE_SEED="${SAMPLE_SEED:-20260819}"
DIFFUSERS_SRC="${DIFFUSERS_SRC:-/linxi/packages/VLCP/diffusers/src}"
export PYTHONPATH="${DIFFUSERS_SRC}:${PYTHONPATH:-}"

python - "$NETTE_PROTOTYPE" "$WOOF_PROTOTYPE" <<'PY'
import json
import sys
from pathlib import Path

for dataset, raw_path in zip(("nette", "woof"), sys.argv[1:]):
    path = Path(raw_path)
    if not path.is_file():
        raise FileNotFoundError(f"Missing {dataset} IPC10 prototype: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    counts = {key: len(value) for key, value in payload.items()}
    invalid = {key: value for key, value in counts.items() if value != 10}
    if invalid:
        raise RuntimeError(f"{dataset} is not a K=10 partition: {invalid}")
    print(f"{dataset}: verified {len(payload)} classes x 10 clusters: {path}")
PY

mkdir -p "$ASSIGNMENT_ROOT/nette" "$ASSIGNMENT_ROOT/woof"

CUDA_VISIBLE_DEVICES=0 python "$REPO_ROOT/experiments/visual_text_factorial/diagnose_cluster_members.py" \
  --data-root "$NETTE_DATA_ROOT" \
  --prototype "$NETTE_PROTOTYPE" \
  --base-model "$BASE_MODEL" \
  --output-dir "$ASSIGNMENT_ROOT/nette" \
  --device cuda --batch-size 32 --num-workers 4 --posterior-mode sample --resume &
nette_pid=$!

CUDA_VISIBLE_DEVICES=1 python "$REPO_ROOT/experiments/visual_text_factorial/diagnose_cluster_members.py" \
  --data-root "$WOOF_DATA_ROOT" \
  --prototype "$WOOF_PROTOTYPE" \
  --base-model "$BASE_MODEL" \
  --output-dir "$ASSIGNMENT_ROOT/woof" \
  --device cuda --batch-size 32 --num-workers 4 --posterior-mode sample --resume &
woof_pid=$!

wait "$nette_pid"
wait "$woof_pid"

python "$REPO_ROOT/experiments/text_supervision_factorial/make_random_cluster_member_montages.py" \
  --assignment "nette=$ASSIGNMENT_ROOT/nette/latent_assignments.csv" \
  --assignment "woof=$ASSIGNMENT_ROOT/woof/latent_assignments.csv" \
  --data-root "nette=$NETTE_DATA_ROOT" \
  --data-root "woof=$WOOF_DATA_ROOT" \
  --output-dir "$OUTPUT_DIR" \
  --seed "$SAMPLE_SEED" \
  --clusters-per-class 5 \
  --images-per-cluster 5 \
  --tile-size 224

echo "IPC10 random-member audit complete: $OUTPUT_DIR/index.html"
