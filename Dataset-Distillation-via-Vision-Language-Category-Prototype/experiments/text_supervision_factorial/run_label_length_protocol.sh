#!/usr/bin/env bash
set -euo pipefail

NETTE_DATA_ROOT="${NETTE_DATA_ROOT:-/linxi/dataset/VLCP/ImageNette}"
BASE_MODEL="${BASE_MODEL:-/linxi/models/VLCP/stable-diffusion-v1-5}"
DIFFUSERS_SRC="${DIFFUSERS_SRC:-/linxi/packages/VLCP/diffusers}"
GENERALITY_RUN_ROOT="${GENERALITY_RUN_ROOT:-./text_supervision_generality_runs/text_supervision_generality_v0}"
BASE_RUN_ROOT="${BASE_RUN_ROOT:-./text_supervision_factorial_runs/text_supervision_factorial_2xa100_v0}"
RUN_ROOT="${RUN_ROOT:-./conditioning_interface_control_runs/label_length_protocol_v0}"
GPUS="${GPUS:-0,1,2,3}"

python experiments/text_supervision_factorial/run_label_length_protocol.py \
  --data-root "$NETTE_DATA_ROOT" \
  --base-model "$BASE_MODEL" \
  --prototype "$GENERALITY_RUN_ROOT/artifacts/nette/ipc50/nette-ipc50-0.7-30-kmexpand1.json" \
  --dcs "$GENERALITY_RUN_ROOT/artifacts/nette/ipc50/dcs.json" \
  --matched-model "$BASE_RUN_ROOT/models/matched_ft" \
  --run-root "$RUN_ROOT" \
  --gpus "$GPUS" \
  --generation-seeds 0 1 2 \
  --ipc 50 --strength 0.8 --classifier-repeats 2 \
  --guidance-scale 10.0 --num-inference-steps 50 \
  --diffusers-src "$DIFFUSERS_SRC"
