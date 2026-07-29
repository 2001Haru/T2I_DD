#!/usr/bin/env bash
set -euo pipefail
shopt -s inherit_errexit

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
EXPERIMENT_DIR="$REPO_ROOT/experiments/visual_text_factorial"

BASE_RUN_ROOT="${BASE_RUN_ROOT:-$REPO_ROOT/../vlcp_factorial_runs/visual_text_factorial_v0}"
RANDOMIZATION_RUN_ID="${RANDOMIZATION_RUN_ID:-visual_text_shuffle_randomization_v0}"
RANDOMIZATION_ROOT="${RANDOMIZATION_ROOT:-$REPO_ROOT/../vlcp_shuffle_runs/$RANDOMIZATION_RUN_ID}"
BASE_MODEL="${BASE_MODEL:-/linxi/models/VLCP/stable-diffusion-v1-5}"
DINO_MODEL="${DINO_MODEL:-/models/DINOv2/dinov2-base}"
SHUFFLE_SHIFTS="${SHUFFLE_SHIFTS:-2 4 7}"
DIAGNOSTICS_ID="${DIAGNOSTICS_ID:-semantic_coverage_v0}"
DIAGNOSTICS_ROOT="${DIAGNOSTICS_ROOT:-$RANDOMIZATION_ROOT/diagnostics/$DIAGNOSTICS_ID}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$DIAGNOSTICS_ROOT/cross_modal_recombination}"
FEATURE_CACHE_DIR="${FEATURE_CACHE_DIR:-$DIAGNOSTICS_ROOT/dino/feature_cache}"
DOWNSTREAM_CSV="${DOWNSTREAM_CSV:-$DIAGNOSTICS_ROOT/downstream_per_class/summary/downstream_dino_per_class.csv}"
BATCH_SIZE="${BATCH_SIZE:-64}"
VAE_BATCH_SIZE="${VAE_BATCH_SIZE:-16}"
DEVICE="${DEVICE:-cuda}"
RESUME="${RESUME:-false}"

for required in "$BASE_RUN_ROOT" "$BASE_MODEL" "$DINO_MODEL"; do
  if [[ ! -e "$required" ]]; then
    echo "Missing required path: $required" >&2
    exit 1
  fi
done

shuffle_run_args=()
for shift in $SHUFFLE_SHIFTS; do
  shift_root="$RANDOMIZATION_ROOT/shift_$shift"
  if [[ ! -d "$shift_root/synthetic" ]]; then
    echo "Missing shuffled run for shift $shift: $shift_root" >&2
    exit 1
  fi
  shuffle_run_args+=(--shuffle-run "$shift=$shift_root")
done

downstream_args=()
if [[ -f "$DOWNSTREAM_CSV" ]]; then
  downstream_args=(--downstream-csv "$DOWNSTREAM_CSV")
else
  echo "Downstream per-class CSV not found; geometry will run without H3: $DOWNSTREAM_CSV"
fi

mkdir -p "$OUTPUT_ROOT"
CONFIG_FILE="$OUTPUT_ROOT/recombination_config.txt"
CONFIG_CONTENT="BASE_RUN_ROOT=$(realpath "$BASE_RUN_ROOT")
RANDOMIZATION_ROOT=$(realpath "$RANDOMIZATION_ROOT")
BASE_MODEL=$(realpath "$BASE_MODEL")
DINO_MODEL=$(realpath "$DINO_MODEL")
SHUFFLE_SHIFTS=$SHUFFLE_SHIFTS
FEATURE_CACHE_DIR=$(realpath -m "$FEATURE_CACHE_DIR")
DOWNSTREAM_CSV=$(realpath -m "$DOWNSTREAM_CSV")
BATCH_SIZE=$BATCH_SIZE
VAE_BATCH_SIZE=$VAE_BATCH_SIZE
DEVICE=$DEVICE"
if [[ -f "$CONFIG_FILE" ]]; then
  if [[ "$(<"$CONFIG_FILE")" != "$CONFIG_CONTENT" ]]; then
    echo "Resume configuration differs from $CONFIG_FILE" >&2
    exit 1
  fi
  if [[ "$RESUME" != "true" ]]; then
    echo "Recombination diagnostics already exist; set RESUME=true: $OUTPUT_ROOT" >&2
    exit 1
  fi
else
  printf '%s\n' "$CONFIG_CONTENT" > "$CONFIG_FILE.tmp"
  mv "$CONFIG_FILE.tmp" "$CONFIG_FILE"
fi

python - <<'PY'
import importlib

for package in ("diffusers", "matplotlib", "numpy", "PIL", "torch", "transformers"):
    importlib.import_module(package)
print("Cross-modal recombination diagnostic dependencies are available.")
PY

resume_args=()
if [[ "$RESUME" == "true" ]]; then
  resume_args=(--resume)
fi

python "$EXPERIMENT_DIR/diagnose_cross_modal_recombination.py" \
  --base-run-root "$BASE_RUN_ROOT" \
  "${shuffle_run_args[@]}" \
  --base-model "$BASE_MODEL" \
  --dino-model "$DINO_MODEL" \
  --output-dir "$OUTPUT_ROOT" \
  --feature-cache-dir "$FEATURE_CACHE_DIR" \
  "${downstream_args[@]}" \
  --device "$DEVICE" \
  --batch-size "$BATCH_SIZE" \
  --vae-batch-size "$VAE_BATCH_SIZE" \
  "${resume_args[@]}"

echo "Cross-modal recombination diagnostics complete: $OUTPUT_ROOT"
