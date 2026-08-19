#!/usr/bin/env bash
set -euo pipefail

NETTE_DATA_ROOT="${NETTE_DATA_ROOT:-/linxi/dataset/VLCP/ImageNette}"
BASE_MODEL="${BASE_MODEL:-/linxi/models/VLCP/stable-diffusion-v1-5}"
GENERALITY_RUN_ROOT="${GENERALITY_RUN_ROOT:-./text_supervision_generality_runs/text_supervision_generality_v0}"
BASE_RUN_ROOT="${BASE_RUN_ROOT:-./low_variance_t77_runs/low_variance_t77_g6_c3_v0}"
REFIT_RUN_ROOT="${REFIT_RUN_ROOT:-./sparse_prompt_search_runs/sparse_t77_refit_seed01_v0}"
RUN_ROOT="${RUN_ROOT:-./low_variance_t77_runs/low_variance_t77_label_completion_v0}"
GPU_IDS="${GPU_IDS:-0,1,2,3}"
DIFFUSERS_SRC="${DIFFUSERS_SRC:-/linxi/packages/VLCP/diffusers/src}"
EXTRA_ARGS=()
if [[ "${AUDIT_ONLY:-false}" == "true" ]]; then EXTRA_ARGS+=(--audit-only); fi

python experiments/text_supervision_factorial/run_low_variance_label_completion.py \
  --data-root "${NETTE_DATA_ROOT}" \
  --base-model "${BASE_MODEL}" \
  --prototype "${GENERALITY_RUN_ROOT}/artifacts/nette/ipc50/nette-ipc50-0.7-30-kmexpand1.json" \
  --dcs "${GENERALITY_RUN_ROOT}/artifacts/nette/ipc50/dcs.json" \
  --matched-model "${REFIT_RUN_ROOT}/fixed/models/train_seed_0/matched_ft" \
  --unpaired-model "${REFIT_RUN_ROOT}/fixed/models/train_seed_0/unpaired_ft" \
  --bank-m4-model "${REFIT_RUN_ROOT}/sparse/models/bank_seed_0/m_4/sparse_ft" \
  --base-run-root "${BASE_RUN_ROOT}" \
  --run-root "${RUN_ROOT}" \
  --gpus "${GPU_IDS}" \
  --generation-seeds 0 1 2 3 4 5 \
  --classifier-repeats 3 \
  --classifier-seed 0 \
  --tail-k 10 \
  --ipc 50 \
  --strength 0.8 \
  --diffusers-src "${DIFFUSERS_SRC}" \
  "${EXTRA_ARGS[@]}"
