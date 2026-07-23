#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
EXPERIMENT_DIR="$REPO_ROOT/experiments/prior_alignment_ablation"
DISTILLATION_DIR="$REPO_ROOT/03_distiilation"
EVALUATION_DIR="$REPO_ROOT/04_evaluation/Minimax"

: "${DATA_ROOT:?Set DATA_ROOT to the prepared ImageNette root containing train/ and val/}"

BASE_MODEL="${BASE_MODEL:-benjamin-paine/stable-diffusion-v1-5}"
RUN_ID="${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
RUN_ROOT="${RUN_ROOT:-$REPO_ROOT/ablation_runs/$RUN_ID}"
FINETUNED_MODEL="${FINETUNED_MODEL:-$RUN_ROOT/models/sd15_finetuned_seed0}"
GENERATION_SEEDS="${GENERATION_SEEDS:-0 1}"
NUM_PROCESSES="${NUM_PROCESSES:-2}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-4}"
if (( 32 % (NUM_PROCESSES * GRADIENT_ACCUMULATION_STEPS) != 0 )); then
  echo "NUM_PROCESSES * GRADIENT_ACCUMULATION_STEPS must divide VLCP's effective batch 32." >&2
  exit 1
fi
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-$((32 / (NUM_PROCESSES * GRADIENT_ACCUMULATION_STEPS)))}"
FINETUNE_SEED="${FINETUNE_SEED:-0}"
CLASSIFIER_SEED="${CLASSIFIER_SEED:-0}"
RESUME="${RESUME:-false}"
BUILD_PROTOTYPES="${BUILD_PROTOTYPES:-true}"
FINETUNE="${FINETUNE:-false}"
GENERATE="${GENERATE:-true}"
EVALUATE="${EVALUATE:-true}"
SUMMARIZE="${SUMMARIZE:-$EVALUATE}"

if [[ "$FINETUNE" == "true" ]]; then
  : "${TRAIN_SCRIPT:?Set TRAIN_SCRIPT to Diffusers examples/text_to_image/train_text_to_image.py}"
fi

PROTOTYPE_DIR="$RUN_ROOT/prototypes"
PROTOTYPE_PATH="$PROTOTYPE_DIR/prior_alignment-ipc10-0.7-30-kmexpand1.json"
DCS_PATH="$PROTOTYPE_DIR/dcs.json"
SYNTHETIC_ROOT="$RUN_ROOT/synthetic"
EVALUATION_ROOT="$RUN_ROOT/evaluation"
LABEL_FILE="$DISTILLATION_DIR/label-prompt/class_nette.txt"

mkdir -p "$RUN_ROOT" "$PROTOTYPE_DIR" "$EVALUATION_ROOT"

python "$EXPERIMENT_DIR/validate_setup.py" \
  --data-root "$DATA_ROOT" \
  --base-model "$BASE_MODEL" \
  --finetuned-model "$FINETUNED_MODEL"

if [[ ! -f "$DATA_ROOT/train/metadata.jsonl" ]]; then
  echo "Missing $DATA_ROOT/train/metadata.jsonl; run prepare_imagenette.py and merge_llava_answers.py first." >&2
  exit 1
fi

if [[ "$BUILD_PROTOTYPES" == "true" && ! -f "$PROTOTYPE_PATH" ]]; then
  (
    cd "$DISTILLATION_DIR"
    python gen_prototype.py \
      --batch_size 10 \
      --spec prior_alignment \
      --contamination 0.1 \
      --data_dir "$DATA_ROOT" \
      --dataset imagenet \
      --diffusion_checkpoints_path "$BASE_MODEL" \
      --ipc 10 \
      --km_expand 1 \
      --label_file_path "$LABEL_FILE" \
      --save_prototype_path "$PROTOTYPE_DIR" \
      --save_text_prototype_path "$DCS_PATH" \
      --seed 0 \
      --metajson_file "$DATA_ROOT/train/metadata.jsonl" \
      --threshold 0.7 \
      --tpk 30
  )
fi

if [[ ! -f "$PROTOTYPE_PATH" || ! -f "$DCS_PATH" ]]; then
  echo "Missing fixed prototype artifacts under $PROTOTYPE_DIR" >&2
  exit 1
fi

if [[ "$FINETUNE" == "true" && ! -f "$FINETUNED_MODEL/model_index.json" ]]; then
  if [[ -d "$FINETUNED_MODEL" && -n "$(find "$FINETUNED_MODEL" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
    echo "Incomplete fine-tune output exists at $FINETUNED_MODEL; refusing to mix runs." >&2
    echo "Move it aside, or resume that Diffusers training explicitly before rerunning this script." >&2
    exit 1
  fi
  accelerate launch --num_processes "$NUM_PROCESSES" "$TRAIN_SCRIPT" \
    --pretrained_model_name_or_path="$BASE_MODEL" \
    --train_data_dir="$DATA_ROOT/train" \
    --use_ema \
    --resolution=512 \
    --center_crop \
    --random_flip \
    --train_batch_size="$TRAIN_BATCH_SIZE" \
    --gradient_accumulation_steps="$GRADIENT_ACCUMULATION_STEPS" \
    --gradient_checkpointing \
    --mixed_precision=fp16 \
    --report_to=none \
    --learning_rate=1e-5 \
    --max_grad_norm=1 \
    --lr_scheduler=constant \
    --lr_warmup_steps=0 \
    --output_dir="$FINETUNED_MODEL" \
    --num_train_epochs=8 \
    --validation_epochs=2 \
    --seed="$FINETUNE_SEED" \
    --checkpoints_total_limit=2 \
    --checkpointing_steps=500
fi

if [[ ! -f "$FINETUNED_MODEL/model_index.json" ]]; then
  echo "Fine-tuned pipeline not found at $FINETUNED_MODEL" >&2
  exit 1
fi

resume_arg=()
if [[ "$RESUME" == "true" ]]; then
  resume_arg=(--resume)
fi

if [[ "$GENERATE" == "true" ]]; then
  read -r -a generation_seed_args <<< "$GENERATION_SEEDS"
  python "$EXPERIMENT_DIR/generate_conditions.py" \
    --prototype "$PROTOTYPE_PATH" \
    --dcs "$DCS_PATH" \
    --base-model "$BASE_MODEL" \
    --finetuned-model "$FINETUNED_MODEL" \
    --output-root "$SYNTHETIC_ROOT" \
    --generation-seeds "${generation_seed_args[@]}" \
    "${resume_arg[@]}"
fi

conditions=(frozen_label frozen_dcs finetuned_label finetuned_dcs)
if [[ "$EVALUATE" == "true" ]]; then
  for generation_seed in $GENERATION_SEEDS; do
    seed_eval_dir="$EVALUATION_ROOT/seed_$generation_seed"
    mkdir -p "$seed_eval_dir"
    for condition in "${conditions[@]}"; do
      synthetic_dir="$SYNTHETIC_ROOT/seed_$generation_seed/$condition"
      log_path="$seed_eval_dir/$condition.log"
      if [[ ! -f "$synthetic_dir/complete.json" ]]; then
        echo "Incomplete synthetic condition: $synthetic_dir" >&2
        exit 1
      fi
      if grep -q "Best, last acc" "$log_path" 2>/dev/null; then
        echo "==> Reusing completed evaluation seed=$generation_seed condition=$condition"
        continue
      fi
      if [[ -s "$log_path" ]]; then
        echo "Incomplete evaluation log exists; refusing to append: $log_path" >&2
        exit 1
      fi
      echo "==> Evaluating seed=$generation_seed condition=$condition"
      (
        cd "$EVALUATION_DIR"
        python train.py \
          -d imagenet \
          --imagenet_dir "$synthetic_dir" "$DATA_ROOT" \
          -n resnet_ap \
          --nclass 10 \
          --norm_type instance \
          --ipc 10 \
          --tag "prior_alignment_${RUN_ID}_g${generation_seed}_${condition}" \
          --slct_type random \
          --repeat 3 \
          --spec nette \
          --seed "$CLASSIFIER_SEED"
      ) 2>&1 | tee "$log_path"
    done
  done
fi

if [[ "$SUMMARIZE" == "true" ]]; then
  python "$EXPERIMENT_DIR/summarize_results.py" \
    --evaluation-root "$EVALUATION_ROOT" \
    --output-dir "$RUN_ROOT/summary"
fi

echo "Ablation complete: $RUN_ROOT"
