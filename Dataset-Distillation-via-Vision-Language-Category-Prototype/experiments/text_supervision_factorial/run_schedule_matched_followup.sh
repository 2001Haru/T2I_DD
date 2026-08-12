#!/usr/bin/env bash
set -euo pipefail

EXPERIMENT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$EXPERIMENT_DIR/../.." && pwd)"

NETTE_DATA_ROOT="${NETTE_DATA_ROOT:-/linxi/dataset/VLCP/ImageNette}"
WOOF_DATA_ROOT="${WOOF_DATA_ROOT:-/linxi/dataset/VLCP/ImageWoof}"
NETTE_CAPTION_FILE="${NETTE_CAPTION_FILE:-$NETTE_DATA_ROOT/train/nette.jsonl}"
if [[ -z "${WOOF_CAPTION_FILE:-}" ]]; then
  for candidate in "$WOOF_DATA_ROOT/train/woof.jsonl" "$WOOF_DATA_ROOT/woof.jsonl"; do
    if [[ -f "$candidate" ]]; then
      WOOF_CAPTION_FILE="$candidate"
      break
    fi
  done
  WOOF_CAPTION_FILE="${WOOF_CAPTION_FILE:-$WOOF_DATA_ROOT/woof.jsonl}"
fi
BASE_MODEL="${BASE_MODEL:-/linxi/models/VLCP/stable-diffusion-v1-5}"
DIFFUSERS_SRC="${DIFFUSERS_SRC:-/linxi/packages/VLCP/diffusers}"
export PYTHONPATH="$DIFFUSERS_SRC${PYTHONPATH:+:$PYTHONPATH}"
BASE_RUN_ROOT="${BASE_RUN_ROOT:-$REPO_ROOT/text_supervision_factorial_runs/text_supervision_factorial_2xa100_v0}"
CAUSAL_RUN_ROOT="${CAUSAL_RUN_ROOT:-$REPO_ROOT/text_supervision_factorial_runs/text_supervision_causal_ladder_v0}"
GENERALITY_RUN_ROOT="${GENERALITY_RUN_ROOT:-$REPO_ROOT/text_supervision_generality_runs/text_supervision_generality_v0}"
INTERFACE_RUN_ROOT="${INTERFACE_RUN_ROOT:-$REPO_ROOT/conditioning_interface_matrix_runs/conditioning_interface_abc_v0}"
WOOF_MODEL_ROOT="${WOOF_MODEL_ROOT:-$REPO_ROOT/conditioning_interface_matrix_runs/conditioning_interface_generality_v0}"
RUN_ID="${RUN_ID:-prototype_checkpoint_covariance_v0}"
RUN_ROOT="${RUN_ROOT:-$REPO_ROOT/schedule_matched_followup_runs/$RUN_ID}"
PRIOR_FOLLOWUP_RUN_ROOT="${PRIOR_FOLLOWUP_RUN_ROOT:-$REPO_ROOT/schedule_matched_followup_runs/schedule_matched_followup_v0}"
GPU_IDS="${GPU_IDS:-0,1,2,3}"

reuse=()
for candidate in \
  "$BASE_RUN_ROOT/evaluation_index.json" \
  "$CAUSAL_RUN_ROOT/evaluation_index.json" \
  "$GENERALITY_RUN_ROOT/evaluation_index.json" \
  "$INTERFACE_RUN_ROOT/evaluation_index.json" \
  "$PRIOR_FOLLOWUP_RUN_ROOT/evaluation_index.json"; do
  [[ -f "$candidate" ]] && reuse+=(--reuse-index "$candidate")
done
for candidate in ${REUSE_INDEXES:-}; do
  reuse+=(--reuse-index "$candidate")
done

python "$EXPERIMENT_DIR/run_schedule_matched_followup.py" \
  --nette-data-root "$NETTE_DATA_ROOT" --nette-caption-file "$NETTE_CAPTION_FILE" \
  --woof-data-root "$WOOF_DATA_ROOT" --woof-caption-file "$WOOF_CAPTION_FILE" \
  --base-model "$BASE_MODEL" --diffusers-src "$DIFFUSERS_SRC" \
  --base-run-root "$BASE_RUN_ROOT" \
  --causal-run-root "$CAUSAL_RUN_ROOT" --generality-run-root "$GENERALITY_RUN_ROOT" \
  --interface-run-root "$INTERFACE_RUN_ROOT" --woof-model-root "$WOOF_MODEL_ROOT" \
  --run-root "$RUN_ROOT" \
  --matrices ${MATRICES:-E F} \
  --supervisions ${SUPERVISIONS:-label_ft matched_ft} \
  --gpus "$GPU_IDS" --classifier-repeats "${CLASSIFIER_REPEATS:-2}" \
  --max-parallel-evals "${MAX_PARALLEL_EVALS:-4}" \
  --max-walltime-hours "${MAX_WALLTIME_HOURS:-0}" "${reuse[@]}"

if [[ "${RUN_RETENTION:-true}" != "true" || ! -f "$RUN_ROOT/COMPLETE" ]]; then
  exit 0
fi

DINO_MODEL="${DINO_MODEL:-/linxi/models/DINOv2/dinov2-base}"
NETTE_PROTOTYPE="$GENERALITY_RUN_ROOT/artifacts/nette/ipc50/nette-ipc50-0.7-30-kmexpand1.json"
WOOF_PROTOTYPE="$INTERFACE_RUN_ROOT/artifacts/woof/ipc50/woof-ipc50-0.7-30-kmexpand1.json"
AUDIT_ROOT="$RUN_ROOT/retention_inputs"
mkdir -p "$AUDIT_ROOT"

CUDA_VISIBLE_DEVICES=0 python "$REPO_ROOT/experiments/visual_text_factorial/diagnose_cluster_members.py" \
  --data-root "$NETTE_DATA_ROOT" --prototype "$NETTE_PROTOTYPE" --base-model "$BASE_MODEL" \
  --output-dir "$AUDIT_ROOT/nette" --device cuda --batch-size 32 --num-workers 2 \
  --posterior-mode sample --resume &
audit_nette=$!
CUDA_VISIBLE_DEVICES=1 python "$REPO_ROOT/experiments/visual_text_factorial/diagnose_cluster_members.py" \
  --data-root "$WOOF_DATA_ROOT" --prototype "$WOOF_PROTOTYPE" --base-model "$BASE_MODEL" \
  --output-dir "$AUDIT_ROOT/woof" --device cuda --batch-size 32 --num-workers 2 \
  --posterior-mode sample --resume &
audit_woof=$!
wait "$audit_nette"
wait "$audit_woof"

CUDA_VISIBLE_DEVICES=0 python "$EXPERIMENT_DIR/analyze_visual_retention.py" \
  --evaluation-index "$RUN_ROOT/evaluation_index.json" \
  --assignment "nette=$AUDIT_ROOT/nette/latent_assignments.csv" \
  --assignment "woof=$AUDIT_ROOT/woof/latent_assignments.csv" \
  --dino-model "$DINO_MODEL" --output-dir "$RUN_ROOT/visual_retention" \
  --device cuda --batch-size "${DINO_BATCH_SIZE:-128}"
