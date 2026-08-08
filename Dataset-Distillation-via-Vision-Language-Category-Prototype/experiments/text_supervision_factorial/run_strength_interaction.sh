#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
EXPERIMENT_DIR="$REPO_ROOT/experiments/text_supervision_factorial"

: "${NETTE_DATA_ROOT:?Set NETTE_DATA_ROOT to prepared ImageNette}"
: "${BASE_MODEL:?Set BASE_MODEL to local SD1.5}"
: "${BASE_RUN_ROOT:?Set BASE_RUN_ROOT to the original seed-0 factorial}"
: "${CAUSAL_RUN_ROOT:?Set CAUSAL_RUN_ROOT to the causal-ladder extension}"
: "${GENERALITY_RUN_ROOT:?Set GENERALITY_RUN_ROOT to the completed IPC generality run}"

RUN_ID="${RUN_ID:-strength_prompt_interaction_v0}"
RUN_ROOT="${RUN_ROOT:-$REPO_ROOT/strength_prompt_interaction_runs/$RUN_ID}"
GPU_IDS="${GPU_IDS:-0,1}"
STRENGTHS="${STRENGTHS:-0.7 0.8 0.9 1.0}"
IPC_VALUES="${IPC_VALUES:-10 50}"
TRAINING_SEEDS="${TRAINING_SEEDS:-0 1}"
GENERATION_SEEDS="${GENERATION_SEEDS:-0 1}"
PROMPTS="${PROMPTS:-label correct}"
GUIDANCE_SCALE="${GUIDANCE_SCALE:-10}"
CLASSIFIER_REPEATS="${CLASSIFIER_REPEATS:-2}"
MAX_PARALLEL_EVALS="${MAX_PARALLEL_EVALS:-2}"
DIFFUSERS_SRC="${DIFFUSERS_SRC:-}"

read -r -a strength_args <<< "$STRENGTHS"
read -r -a ipc_args <<< "$IPC_VALUES"
read -r -a training_args <<< "$TRAINING_SEEDS"
read -r -a generation_args <<< "$GENERATION_SEEDS"
read -r -a prompt_args <<< "$PROMPTS"

args=(
  --nette-data-root "$NETTE_DATA_ROOT" --base-model "$BASE_MODEL"
  --base-run-root "$BASE_RUN_ROOT" --causal-run-root "$CAUSAL_RUN_ROOT"
  --generality-run-root "$GENERALITY_RUN_ROOT" --run-root "$RUN_ROOT" --gpus "$GPU_IDS"
  --strengths "${strength_args[@]}" --ipc-values "${ipc_args[@]}"
  --training-seeds "${training_args[@]}" --generation-seeds "${generation_args[@]}"
  --prompts "${prompt_args[@]}" --guidance-scale "$GUIDANCE_SCALE"
  --classifier-repeats "$CLASSIFIER_REPEATS" --max-parallel-evals "$MAX_PARALLEL_EVALS"
)
[[ -n "$DIFFUSERS_SRC" ]] && args+=(--diffusers-src "$DIFFUSERS_SRC")
[[ "${DISABLE_REUSE_0P7:-false}" == "true" ]] && args+=(--disable-reuse-0p7)

echo "Persistent strength scheduler: $RUN_ROOT"
echo "strength={$STRENGTHS}, IPC={$IPC_VALUES}, prompts={$PROMPTS}, CFG=$GUIDANCE_SCALE"
exec python "$EXPERIMENT_DIR/run_strength_interaction.py" "${args[@]}"
