#!/usr/bin/env bash
set -euo pipefail
shopt -s inherit_errexit

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
EXPERIMENT_DIR="$REPO_ROOT/experiments/visual_text_factorial"

: "${DATA_ROOT:?Set DATA_ROOT to the prepared ImageNette root containing train/ and val/}"

BASE_RUN_ROOT="${BASE_RUN_ROOT:-$REPO_ROOT/../vlcp_factorial_runs/visual_text_factorial_v0}"
SOURCE_RUN_ROOT="${SOURCE_RUN_ROOT:-$REPO_ROOT/../vlcp_ablation_runs/author_checkpoint_pilot_v0}"
BASE_MODEL="${BASE_MODEL:-/linxi/models/VLCP/stable-diffusion-v1-5}"
PROTOTYPE_PATH="${PROTOTYPE_PATH:-$SOURCE_RUN_ROOT/prototypes/prior_alignment-ipc10-0.7-30-kmexpand1.json}"
DCS_PATH="${DCS_PATH:-$SOURCE_RUN_ROOT/prototypes/dcs.json}"
RANDOMIZATION_RUN_ID="${RANDOMIZATION_RUN_ID:-visual_text_shuffle_randomization_v0}"
RANDOMIZATION_ROOT="${RANDOMIZATION_ROOT:-$REPO_ROOT/../vlcp_shuffle_runs/$RANDOMIZATION_RUN_ID}"
SHUFFLE_SHIFTS="${SHUFFLE_SHIFTS:-2 4 7}"
GENERATION_SEEDS="${GENERATION_SEEDS:-0 1}"
RESUME="${RESUME:-false}"

mkdir -p "$RANDOMIZATION_ROOT"
CONFIG_FILE="$RANDOMIZATION_ROOT/randomization_config.txt"
CONFIG_CONTENT="BASE_RUN_ROOT=$(realpath "$BASE_RUN_ROOT")
SOURCE_RUN_ROOT=$(realpath "$SOURCE_RUN_ROOT")
BASE_MODEL=$BASE_MODEL
PROTOTYPE_PATH=$(realpath "$PROTOTYPE_PATH")
DCS_PATH=$(realpath "$DCS_PATH")
SHUFFLE_SHIFTS=$SHUFFLE_SHIFTS
GENERATION_SEEDS=$GENERATION_SEEDS"
if [[ -f "$CONFIG_FILE" ]]; then
  if [[ "$(<"$CONFIG_FILE")" != "$CONFIG_CONTENT" ]]; then
    echo "Resume configuration differs from $CONFIG_FILE" >&2
    exit 1
  fi
  if [[ "$RESUME" != "true" ]]; then
    echo "Randomization run already exists; set RESUME=true: $RANDOMIZATION_ROOT" >&2
    exit 1
  fi
else
  printf '%s\n' "$CONFIG_CONTENT" > "$CONFIG_FILE.tmp"
  mv "$CONFIG_FILE.tmp" "$CONFIG_FILE"
fi

shuffle_run_args=()
for shift in $SHUFFLE_SHIFTS; do
  if (( shift < 2 || shift > 9 )); then
    echo "Additional SHUFFLE_SHIFTS must be integers in [2, 9], got $shift" >&2
    exit 1
  fi
  shift_root="$RANDOMIZATION_ROOT/shift_$shift"
  echo "==> Running prespecified shuffle shift $shift"
  DATA_ROOT="$DATA_ROOT" \
  SOURCE_RUN_ROOT="$SOURCE_RUN_ROOT" \
  BASE_MODEL="$BASE_MODEL" \
  PROTOTYPE_PATH="$PROTOTYPE_PATH" \
  DCS_PATH="$DCS_PATH" \
  RUN_ID="${RANDOMIZATION_RUN_ID}_shift${shift}" \
  RUN_ROOT="$shift_root" \
  GENERATION_SEEDS="$GENERATION_SEEDS" \
  CONDITIONS="no_visual_dcs_shuffled prototype_dcs_shuffled" \
  SHUFFLE_SHIFT="$shift" \
  RESUME="$RESUME" \
  SUMMARIZE=false \
  bash "$EXPERIMENT_DIR/run_experiment.sh"
  shuffle_run_args+=(--shuffle-run "$shift=$shift_root")
done

python "$EXPERIMENT_DIR/summarize_shuffle_randomization.py" \
  --base-run-root "$BASE_RUN_ROOT" \
  "${shuffle_run_args[@]}" \
  --output-dir "$RANDOMIZATION_ROOT/summary"

echo "Shuffle randomization complete: $RANDOMIZATION_ROOT"
