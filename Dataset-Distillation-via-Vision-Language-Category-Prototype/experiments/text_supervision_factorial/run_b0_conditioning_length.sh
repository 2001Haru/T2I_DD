#!/usr/bin/env bash
set -euo pipefail

NETTE_DATA_ROOT="${NETTE_DATA_ROOT:-/linxi/dataset/VLCP/ImageNette}"
BASE_MODEL="${BASE_MODEL:-/linxi/models/VLCP/stable-diffusion-v1-5}"
DIFFUSERS_SRC="${DIFFUSERS_SRC:-/linxi/packages/VLCP/diffusers}"
BASE_RUN_ROOT="${BASE_RUN_ROOT:-./text_supervision_factorial_runs/text_supervision_factorial_2xa100_v0}"
SPARSE_RUN_ROOT="${SPARSE_RUN_ROOT:-./sparse_prompt_search_runs/random_sparse_marginal_v0}"
B0_RUN_ROOT="${B0_RUN_ROOT:-./conditioning_interface_matrix_runs/b0_conditioning_length_v0}"

PROTOTYPE="${PROTOTYPE:-${BASE_RUN_ROOT}/prototypes/text_supervision-ipc10-0.7-30-kmexpand1.json}"
DCS="${DCS:-${BASE_RUN_ROOT}/prototypes/dcs.json}"
LABEL_MODEL="${LABEL_MODEL:-${BASE_RUN_ROOT}/models/label_ft}"
MATCHED_MODEL="${MATCHED_MODEL:-${BASE_RUN_ROOT}/models/matched_ft}"
SPARSE_MODEL="${SPARSE_MODEL:-${SPARSE_RUN_ROOT}/models/bank_seed_0/m_4/sparse_ft}"

python experiments/text_supervision_factorial/run_b0_conditioning_length.py \
  --data-root "${NETTE_DATA_ROOT}" \
  --base-model "${BASE_MODEL}" \
  --prototype "${PROTOTYPE}" \
  --dcs "${DCS}" \
  --label-model "${LABEL_MODEL}" \
  --matched-model "${MATCHED_MODEL}" \
  --sparse-model "${SPARSE_MODEL}" \
  --run-root "${B0_RUN_ROOT}" \
  --gpus "${GPU_IDS:-0,1,2,3}" \
  --generation-seeds ${GENERATION_SEEDS:-0 1 2} \
  --ipc 10 \
  --strength 0.7 \
  --classifier-repeats "${CLASSIFIER_REPEATS:-2}" \
  --diffusers-src "${DIFFUSERS_SRC}"
