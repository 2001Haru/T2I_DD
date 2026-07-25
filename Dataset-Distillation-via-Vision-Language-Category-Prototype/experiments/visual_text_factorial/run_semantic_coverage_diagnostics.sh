#!/usr/bin/env bash
set -euo pipefail
shopt -s inherit_errexit

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
EXPERIMENT_DIR="$REPO_ROOT/experiments/visual_text_factorial"

: "${DATA_ROOT:?Set DATA_ROOT to the prepared ImageNette root containing train/ and val/}"

BASE_RUN_ROOT="${BASE_RUN_ROOT:-$REPO_ROOT/../vlcp_factorial_runs/visual_text_factorial_v0}"
RANDOMIZATION_RUN_ID="${RANDOMIZATION_RUN_ID:-visual_text_shuffle_randomization_v0}"
RANDOMIZATION_ROOT="${RANDOMIZATION_ROOT:-$REPO_ROOT/../vlcp_shuffle_runs/$RANDOMIZATION_RUN_ID}"
BASE_MODEL="${BASE_MODEL:-/linxi/models/VLCP/stable-diffusion-v1-5}"
DINO_MODEL="${DINO_MODEL:-/models/DINOv2/dinov2-base}"
SHUFFLE_SHIFTS="${SHUFFLE_SHIFTS:-2 4 7}"
DIAGNOSTICS_ID="${DIAGNOSTICS_ID:-semantic_coverage_v0}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$RANDOMIZATION_ROOT/diagnostics/$DIAGNOSTICS_ID}"
BATCH_SIZE="${BATCH_SIZE:-64}"
NN_BLOCK_SIZE="${NN_BLOCK_SIZE:-1024}"
DEVICE="${DEVICE:-cuda}"
RESUME="${RESUME:-false}"

shuffle_run_args=()
for shift in $SHUFFLE_SHIFTS; do
  shift_root="$RANDOMIZATION_ROOT/shift_$shift"
  if [[ ! -d "$shift_root/synthetic" ]]; then
    echo "Missing shuffled run for shift $shift: $shift_root" >&2
    exit 1
  fi
  shuffle_run_args+=(--shuffle-run "$shift=$shift_root")
done

mkdir -p "$OUTPUT_ROOT"
CONFIG_FILE="$OUTPUT_ROOT/diagnostics_config.txt"
CONFIG_CONTENT="DATA_ROOT=$(realpath "$DATA_ROOT")
BASE_RUN_ROOT=$(realpath "$BASE_RUN_ROOT")
RANDOMIZATION_ROOT=$(realpath "$RANDOMIZATION_ROOT")
BASE_MODEL=$(realpath "$BASE_MODEL")
DINO_MODEL=$(realpath "$DINO_MODEL")
SHUFFLE_SHIFTS=$SHUFFLE_SHIFTS
BATCH_SIZE=$BATCH_SIZE
NN_BLOCK_SIZE=$NN_BLOCK_SIZE
DEVICE=$DEVICE"
if [[ -f "$CONFIG_FILE" ]]; then
  if [[ "$(<"$CONFIG_FILE")" != "$CONFIG_CONTENT" ]]; then
    echo "Resume configuration differs from $CONFIG_FILE" >&2
    exit 1
  fi
  if [[ "$RESUME" != "true" ]]; then
    echo "Diagnostics already exist; set RESUME=true: $OUTPUT_ROOT" >&2
    exit 1
  fi
else
  printf '%s\n' "$CONFIG_CONTENT" > "$CONFIG_FILE.tmp"
  mv "$CONFIG_FILE.tmp" "$CONFIG_FILE"
fi

python - <<'PY'
import importlib

for package in ("matplotlib", "numpy", "PIL", "torch", "transformers"):
    importlib.import_module(package)
print("Semantic coverage diagnostic dependencies are available.")
PY

python "$EXPERIMENT_DIR/diagnose_text_conditioning.py" \
  --base-run-root "$BASE_RUN_ROOT" \
  "${shuffle_run_args[@]}" \
  --base-model "$BASE_MODEL" \
  --output-dir "$OUTPUT_ROOT/text" \
  --device "$DEVICE"

resume_arg=()
if [[ "$RESUME" == "true" ]]; then
  resume_arg=(--resume)
fi
python "$EXPERIMENT_DIR/diagnose_dino_coverage.py" \
  --data-root "$DATA_ROOT" \
  --base-run-root "$BASE_RUN_ROOT" \
  "${shuffle_run_args[@]}" \
  --dino-model "$DINO_MODEL" \
  --output-dir "$OUTPUT_ROOT/dino" \
  --device "$DEVICE" \
  --batch-size "$BATCH_SIZE" \
  --nn-block-size "$NN_BLOCK_SIZE" \
  "${resume_arg[@]}"

python "$EXPERIMENT_DIR/summarize_semantic_coverage.py" \
  --text-dir "$OUTPUT_ROOT/text" \
  --dino-dir "$OUTPUT_ROOT/dino" \
  --output-dir "$OUTPUT_ROOT/summary"

echo "Semantic coverage diagnostics complete: $OUTPUT_ROOT"
