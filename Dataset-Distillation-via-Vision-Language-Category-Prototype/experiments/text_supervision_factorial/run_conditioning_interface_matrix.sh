#!/usr/bin/env bash
set -euo pipefail

EXPERIMENT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$EXPERIMENT_DIR/../.." && pwd)"

: "${NETTE_DATA_ROOT:?Set NETTE_DATA_ROOT}"
: "${NETTE_CAPTION_FILE:?Set NETTE_CAPTION_FILE}"
: "${BASE_MODEL:?Set BASE_MODEL}"
: "${BASE_RUN_ROOT:?Set BASE_RUN_ROOT}"
: "${CAUSAL_RUN_ROOT:?Set CAUSAL_RUN_ROOT}"
: "${GENERALITY_RUN_ROOT:?Set GENERALITY_RUN_ROOT}"

RUN_ID="${RUN_ID:-conditioning_interface_abc_v0}"
RUN_ROOT="${RUN_ROOT:-$REPO_ROOT/conditioning_interface_matrix_runs/$RUN_ID}"
GPU_IDS="${GPU_IDS:-0,1,2,3}"
MATRICES="${MATRICES:-A B C}"
TRAINING_SEEDS="${TRAINING_SEEDS:-0 1}"
GENERATION_SEEDS="${GENERATION_SEEDS:-0 1}"
CLASSIFIER_REPEATS="${CLASSIFIER_REPEATS:-2}"
MAX_PARALLEL_EVALS="${MAX_PARALLEL_EVALS:-2}"
NUM_WORKERS="${NUM_WORKERS:-2}"
DIFFUSERS_SRC="${DIFFUSERS_SRC:-}"
REUSE_INDEXES="${REUSE_INDEXES:-}"

read -r -a matrix_args <<< "$MATRICES"
read -r -a training_seed_args <<< "$TRAINING_SEEDS"
read -r -a generation_seed_args <<< "$GENERATION_SEEDS"
read -r -a reuse_index_args <<< "$REUSE_INDEXES"

if [[ " $MATRICES " == *" C "* ]]; then
  : "${WOOF_DATA_ROOT:?Set WOOF_DATA_ROOT when Matrix C is enabled}"
  : "${WOOF_CAPTION_FILE:?Set WOOF_CAPTION_FILE when Matrix C is enabled}"
fi

args=(
  --nette-data-root "$NETTE_DATA_ROOT" --nette-caption-file "$NETTE_CAPTION_FILE"
  --base-model "$BASE_MODEL" --base-run-root "$BASE_RUN_ROOT"
  --causal-run-root "$CAUSAL_RUN_ROOT" --generality-run-root "$GENERALITY_RUN_ROOT"
  --run-root "$RUN_ROOT" --gpus "$GPU_IDS" --matrices "${matrix_args[@]}"
  --training-seeds "${training_seed_args[@]}" --generation-seeds "${generation_seed_args[@]}"
  --classifier-repeats "$CLASSIFIER_REPEATS" --max-parallel-evals "$MAX_PARALLEL_EVALS"
  --train-batch-size 4 --gradient-accumulation-steps 8 --num-workers "$NUM_WORKERS"
  --mixed-precision fp16
)
if [[ " $MATRICES " == *" C "* ]]; then
  args+=(--woof-data-root "$WOOF_DATA_ROOT" --woof-caption-file "$WOOF_CAPTION_FILE")
fi
[[ -n "$DIFFUSERS_SRC" ]] && args+=(--diffusers-src "$DIFFUSERS_SRC")
for index_path in "${reuse_index_args[@]}"; do
  args+=(--reuse-index "$index_path")
done

echo "Persistent A/B/C conditioning-interface matrix: $RUN_ROOT"
echo "Matrices: $MATRICES; GPUs: $GPU_IDS; G condition: removed"
exec python "$EXPERIMENT_DIR/run_conditioning_interface_matrix.py" "${args[@]}"
