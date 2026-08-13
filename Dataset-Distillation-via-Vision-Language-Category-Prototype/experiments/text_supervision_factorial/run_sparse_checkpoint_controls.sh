#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
NETTE_DATA_ROOT="${NETTE_DATA_ROOT:-/linxi/dataset/VLCP/ImageNette}"
BASE_MODEL="${BASE_MODEL:-/linxi/models/VLCP/stable-diffusion-v1-5}"
BASE_RUN_ROOT="${BASE_RUN_ROOT:-${REPO_ROOT}/text_supervision_factorial_runs/text_supervision_factorial_2xa100_v0}"
CAUSAL_RUN_ROOT="${CAUSAL_RUN_ROOT:-${REPO_ROOT}/text_supervision_factorial_runs/text_supervision_causal_ladder_v0}"
GENERALITY_RUN_ROOT="${GENERALITY_RUN_ROOT:-${REPO_ROOT}/text_supervision_generality_runs/text_supervision_generality_v0}"
SPARSE_RUN_ROOT="${SPARSE_RUN_ROOT:-${REPO_ROOT}/sparse_prompt_search_runs/random_sparse_marginal_v0}"
RUN_ROOT="${RUN_ROOT:-${REPO_ROOT}/sparse_prompt_search_runs/random_sparse_marginal_controls_v0}"
PROTOTYPE="${PROTOTYPE:-${GENERALITY_RUN_ROOT}/artifacts/nette/ipc50/nette-ipc50-0.7-30-kmexpand1.json}"
DCS="${DCS:-${GENERALITY_RUN_ROOT}/artifacts/nette/ipc50/dcs.json}"
GPU_IDS="${GPU_IDS:-0,1,2,3}"
TRAINING_SEEDS="${TRAINING_SEEDS:-0}"
GENERATION_SEEDS="${GENERATION_SEEDS:-0 1}"
DIFFUSERS_SRC="${DIFFUSERS_SRC:-/linxi/packages/VLCP/diffusers/src}"

cd "${REPO_ROOT}"
python experiments/text_supervision_factorial/run_sparse_checkpoint_controls.py \
  --data-root "${NETTE_DATA_ROOT}" \
  --base-model "${BASE_MODEL}" \
  --base-run-root "${BASE_RUN_ROOT}" \
  --causal-run-root "${CAUSAL_RUN_ROOT}" \
  --sparse-run-root "${SPARSE_RUN_ROOT}" \
  --prototype "${PROTOTYPE}" \
  --dcs "${DCS}" \
  --run-root "${RUN_ROOT}" \
  --gpus "${GPU_IDS}" \
  --training-seeds ${TRAINING_SEEDS} \
  --generation-seeds ${GENERATION_SEEDS} \
  --ipc 50 \
  --strength 0.8 \
  --classifier-repeats 2 \
  --max-retries 0 \
  --diffusers-src "${DIFFUSERS_SRC}"
