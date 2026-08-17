#!/usr/bin/env bash
set -euo pipefail

NETTE_DATA_ROOT="${NETTE_DATA_ROOT:-/linxi/dataset/VLCP/ImageNette}"
BASE_MODEL="${BASE_MODEL:-/linxi/models/VLCP/stable-diffusion-v1-5}"
GENERALITY_RUN_ROOT="${GENERALITY_RUN_ROOT:-./text_supervision_generality_runs/text_supervision_generality_v0}"
RUN_ROOT="${RUN_ROOT:-./sparse_prompt_search_runs/sparse_t77_refit_seed01_v0}"
CAPTION_FILE="${CAPTION_FILE:-${NETTE_DATA_ROOT}/train/nette.jsonl}"
PROTOTYPE="${PROTOTYPE:-${GENERALITY_RUN_ROOT}/artifacts/nette/ipc50/nette-ipc50-0.7-30-kmexpand1.json}"
DCS="${DCS:-${GENERALITY_RUN_ROOT}/artifacts/nette/ipc50/dcs.json}"
GPU_IDS="${GPU_IDS:-0,1}"
DIFFUSERS_SRC="${DIFFUSERS_SRC:-/linxi/packages/VLCP/diffusers/src}"

python experiments/text_supervision_factorial/run_sparse_t77_completion.py \
  --data-root "${NETTE_DATA_ROOT}" \
  --caption-file "${CAPTION_FILE}" \
  --base-model "${BASE_MODEL}" \
  --prototype "${PROTOTYPE}" \
  --dcs "${DCS}" \
  --run-root "${RUN_ROOT}" \
  --gpus "${GPU_IDS}" \
  --generation-seeds 0 1 \
  --dense-training-seeds 0 1 \
  --ipc 50 \
  --strength 0.8 \
  --classifier-repeats 2 \
  --diffusers-src "${DIFFUSERS_SRC}"
