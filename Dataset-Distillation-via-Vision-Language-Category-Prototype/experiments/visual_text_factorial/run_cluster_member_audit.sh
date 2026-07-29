#!/usr/bin/env bash
set -euo pipefail
shopt -s inherit_errexit

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
EXPERIMENT_DIR="$REPO_ROOT/experiments/visual_text_factorial"

: "${DATA_ROOT:?Set DATA_ROOT to the prepared ImageNette root containing train/ and val/}"

BASE_RUN_ROOT="${BASE_RUN_ROOT:-$REPO_ROOT/../vlcp_factorial_runs/visual_text_factorial_v0}"
BASE_MODEL="${BASE_MODEL:-/linxi/models/VLCP/stable-diffusion-v1-5}"
RANDOMIZATION_RUN_ID="${RANDOMIZATION_RUN_ID:-visual_text_shuffle_randomization_v0}"
RANDOMIZATION_ROOT="${RANDOMIZATION_ROOT:-$REPO_ROOT/../vlcp_shuffle_runs/$RANDOMIZATION_RUN_ID}"
DIAGNOSTICS_ID="${DIAGNOSTICS_ID:-semantic_coverage_v0}"
DIAGNOSTICS_ROOT="${DIAGNOSTICS_ROOT:-$RANDOMIZATION_ROOT/diagnostics/$DIAGNOSTICS_ID}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$DIAGNOSTICS_ROOT/cluster_member_audit}"
DECODED_PROTOTYPE_ROOT="${DECODED_PROTOTYPE_ROOT:-$DIAGNOSTICS_ROOT/cross_modal_recombination/decoded_prototypes}"
BATCH_SIZE="${BATCH_SIZE:-10}"
NUM_WORKERS="${NUM_WORKERS:-4}"
IMAGE_SIZE="${IMAGE_SIZE:-512}"
NEAREST_COUNT="${NEAREST_COUNT:-9}"
SEED="${SEED:-0}"
POSTERIOR_MODE="${POSTERIOR_MODE:-sample}"
DEVICE="${DEVICE:-cuda}"
RESUME="${RESUME:-false}"

BASE_MANIFEST="$BASE_RUN_ROOT/synthetic/seed_0/prototype_dcs/manifest.json"
if [[ ! -f "$BASE_MANIFEST" ]]; then
  echo "Missing base prototype-DCS manifest: $BASE_MANIFEST" >&2
  exit 1
fi

if [[ -z "${PROTOTYPE_PATH:-}" ]]; then
  PROTOTYPE_PATH="$(
    python - "$BASE_MANIFEST" <<'PY'
import json
import sys

with open(sys.argv[1], "r", encoding="utf-8") as handle:
    print(json.load(handle)["prototype_path"])
PY
  )"
fi

for required in "$DATA_ROOT/train" "$PROTOTYPE_PATH" "$BASE_MODEL"; do
  if [[ ! -e "$required" ]]; then
    echo "Missing required path: $required" >&2
    exit 1
  fi
done

decoded_args=()
if [[ -d "$DECODED_PROTOTYPE_ROOT" ]]; then
  decoded_args=(--decoded-prototype-root "$DECODED_PROTOTYPE_ROOT")
else
  echo "Decoded prototype directory not found; montages will contain real images only."
fi

mkdir -p "$OUTPUT_ROOT"
python - <<'PY'
import importlib

for package in ("diffusers", "matplotlib", "numpy", "PIL", "torch", "torchvision"):
    importlib.import_module(package)
print("Cluster-member audit dependencies are available.")
PY

resume_args=()
if [[ "$RESUME" == "true" ]]; then
  resume_args=(--resume)
fi

python "$EXPERIMENT_DIR/diagnose_cluster_members.py" \
  --data-root "$DATA_ROOT" \
  --prototype "$PROTOTYPE_PATH" \
  --base-model "$BASE_MODEL" \
  --output-dir "$OUTPUT_ROOT" \
  "${decoded_args[@]}" \
  --device "$DEVICE" \
  --batch-size "$BATCH_SIZE" \
  --num-workers "$NUM_WORKERS" \
  --image-size "$IMAGE_SIZE" \
  --nearest-count "$NEAREST_COUNT" \
  --seed "$SEED" \
  --posterior-mode "$POSTERIOR_MODE" \
  "${resume_args[@]}"

echo "Cluster-member audit complete: $OUTPUT_ROOT"
