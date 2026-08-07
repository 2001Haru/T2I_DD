#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
EXPERIMENT_DIR="$REPO_ROOT/experiments/text_supervision_factorial"

: "${NETTE_DATA_ROOT:?Set NETTE_DATA_ROOT to prepared ImageNette}"
: "${NETTE_CAPTION_FILE:?Set NETTE_CAPTION_FILE to complete ImageNette metadata JSONL}"
: "${BASE_MODEL:?Set BASE_MODEL to local SD1.5}"
: "${BASE_RUN_ROOT:?Set BASE_RUN_ROOT to the original seed-0 factorial}"
: "${CAUSAL_RUN_ROOT:?Set CAUSAL_RUN_ROOT to the completed causal-ladder extension}"

RUN_ID="${RUN_ID:-text_supervision_generality_v0}"
RUN_ROOT="${RUN_ROOT:-$REPO_ROOT/text_supervision_generality_runs/$RUN_ID}"
GPU_IDS="${GPU_IDS:-0,1}"
GENERATION_SEEDS="${GENERATION_SEEDS:-0 1}"
NEW_TRAINING_SEEDS="${NEW_TRAINING_SEEDS:-2 3}"
IPC_TRAINING_SEEDS="${IPC_TRAINING_SEEDS:-0 1}"
WOOF_TRAINING_SEEDS="${WOOF_TRAINING_SEEDS:-0 1}"
IPC_VALUES="${IPC_VALUES:-20 50}"
CLASSIFIER_REPEATS="${CLASSIFIER_REPEATS:-2}"
NUM_WORKERS="${NUM_WORKERS:-2}"
DIFFUSERS_SRC="${DIFFUSERS_SRC:-}"

if [[ -n "${WOOF_DATA_ROOT:-}" || -n "${WOOF_CAPTION_FILE:-}" ]]; then
  : "${WOOF_DATA_ROOT:?Set both WOOF_DATA_ROOT and WOOF_CAPTION_FILE}"
  : "${WOOF_CAPTION_FILE:?Set both WOOF_DATA_ROOT and WOOF_CAPTION_FILE}"
  PHASES="${PHASES:-nette_seeds nette_ipc woof}"
else
  PHASES="${PHASES:-nette_seeds nette_ipc}"
fi

read -r -a phase_args <<< "$PHASES"
read -r -a generation_args <<< "$GENERATION_SEEDS"
read -r -a new_seed_args <<< "$NEW_TRAINING_SEEDS"
read -r -a ipc_seed_args <<< "$IPC_TRAINING_SEEDS"
read -r -a woof_seed_args <<< "$WOOF_TRAINING_SEEDS"
read -r -a ipc_args <<< "$IPC_VALUES"

args=(
  --nette-data-root "$NETTE_DATA_ROOT" --nette-caption-file "$NETTE_CAPTION_FILE"
  --base-model "$BASE_MODEL" --base-run-root "$BASE_RUN_ROOT" --causal-run-root "$CAUSAL_RUN_ROOT"
  --run-root "$RUN_ROOT" --gpus "$GPU_IDS" --phases "${phase_args[@]}"
  --generation-seeds "${generation_args[@]}" --new-training-seeds "${new_seed_args[@]}"
  --ipc-training-seeds "${ipc_seed_args[@]}" --woof-training-seeds "${woof_seed_args[@]}"
  --ipc-values "${ipc_args[@]}" --classifier-repeats "$CLASSIFIER_REPEATS"
  --train-batch-size 4 --gradient-accumulation-steps 8 --num-workers "$NUM_WORKERS"
  --mixed-precision fp16 --max-parallel-evals 1
)
[[ -n "$DIFFUSERS_SRC" ]] && args+=(--diffusers-src "$DIFFUSERS_SRC")
[[ -n "${NETTE_PROTOTYPE_PATH:-}" ]] && args+=(--nette-prototype "$NETTE_PROTOTYPE_PATH")
[[ -n "${NETTE_DCS_PATH:-}" ]] && args+=(--nette-dcs "$NETTE_DCS_PATH")
[[ -n "${WOOF_DATA_ROOT:-}" ]] && args+=(--woof-data-root "$WOOF_DATA_ROOT" --woof-caption-file "$WOOF_CAPTION_FILE")

echo "Persistent generality scheduler: $RUN_ROOT"
echo "Phases: $PHASES; GPUs: $GPU_IDS"
exec python "$EXPERIMENT_DIR/run_generality.py" "${args[@]}"
