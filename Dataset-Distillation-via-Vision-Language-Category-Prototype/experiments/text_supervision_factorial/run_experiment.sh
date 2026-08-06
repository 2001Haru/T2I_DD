#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
EXPERIMENT_DIR="$REPO_ROOT/experiments/text_supervision_factorial"
DISTILLATION_DIR="$REPO_ROOT/03_distiilation"
EVALUATION_DIR="$REPO_ROOT/04_evaluation/Minimax"

: "${DATA_ROOT:?Set DATA_ROOT to ImageNette containing train/ and val/}"
: "${BASE_MODEL:?Set BASE_MODEL to the local SD1.5 Diffusers pipeline}"

RUN_ID="${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
RUN_ROOT="${RUN_ROOT:-$REPO_ROOT/text_supervision_factorial_runs/$RUN_ID}"
CAPTION_FILE="${CAPTION_FILE:-}"
if [[ -z "$CAPTION_FILE" ]]; then
  for candidate in "$DATA_ROOT/train/metadata.jsonl" "$DATA_ROOT/train/nette.jsonl" "$DATA_ROOT/nette.jsonl"; do
    if [[ -f "$candidate" ]]; then CAPTION_FILE="$candidate"; break; fi
  done
fi
: "${CAPTION_FILE:?No ImageNette caption JSONL found; set CAPTION_FILE}"

TRAIN_GPU_IDS="${TRAIN_GPU_IDS:-0,1,2,3}"
WORKER_GPU_IDS="${WORKER_GPU_IDS:-$TRAIN_GPU_IDS}"
NUM_PROCESSES="${NUM_PROCESSES:-4}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-4}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-2}"
NUM_WORKERS="${NUM_WORKERS:-8}"
GENERATION_SEEDS="${GENERATION_SEEDS:-0 1}"
CLASSIFIER_REPEATS="${CLASSIFIER_REPEATS:-2}"
MAX_PARALLEL_EVALS="${MAX_PARALLEL_EVALS:-2}"
CLASSIFIER_SEED="${CLASSIFIER_SEED:-0}"
FINETUNE_SEED="${FINETUNE_SEED:-0}"
MIXED_PRECISION="${MIXED_PRECISION:-fp16}"
DIFFUSERS_SRC="${DIFFUSERS_SRC:-}"
MAIN_PROCESS_PORT="${MAIN_PROCESS_PORT:-29500}"
RESUME="${RESUME:-false}"
BUILD_PROTOTYPES="${BUILD_PROTOTYPES:-true}"
TRAIN="${TRAIN:-true}"
TRAIN_ROWS="${TRAIN_ROWS:-label_ft unpaired_ft matched_ft}"
TRAIN_ONLY="${TRAIN_ONLY:-false}"
GENERATE="${GENERATE:-true}"
EVALUATE="${EVALUATE:-true}"
SUMMARIZE="${SUMMARIZE:-$EVALUATE}"

if (( NUM_PROCESSES * TRAIN_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS != 32 )); then
  echo "Effective batch must remain 32; got $((NUM_PROCESSES * TRAIN_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS))." >&2
  exit 1
fi
IFS=',' read -r -a WORKER_GPUS <<< "$WORKER_GPU_IDS"
if (( ${#WORKER_GPUS[@]} == 0 )); then
  echo "WORKER_GPU_IDS must contain at least one GPU id." >&2
  exit 1
fi

PROTOTYPE_DIR="${PROTOTYPE_DIR:-$RUN_ROOT/prototypes}"
PROTOTYPE_PATH="${PROTOTYPE_PATH:-$PROTOTYPE_DIR/text_supervision-ipc10-0.7-30-kmexpand1.json}"
DCS_PATH="${DCS_PATH:-$PROTOTYPE_DIR/dcs.json}"
LABEL_FILE="$DISTILLATION_DIR/label-prompt/class_nette.txt"
MODEL_ROOT="$RUN_ROOT/models"
SYNTHETIC_ROOT="$RUN_ROOT/synthetic"
EVALUATION_ROOT="$RUN_ROOT/evaluation"
SUMMARY_ROOT="$RUN_ROOT/summary"
mkdir -p "$RUN_ROOT" "$PROTOTYPE_DIR" "$MODEL_ROOT" "$SYNTHETIC_ROOT" "$EVALUATION_ROOT"
if [[ -n "$DIFFUSERS_SRC" ]]; then
  export PYTHONPATH="$DIFFUSERS_SRC${PYTHONPATH:+:$PYTHONPATH}"
fi

CONFIG="$RUN_ROOT/run_config.txt"
cat > "$CONFIG.new" <<EOF
DATA_ROOT=$DATA_ROOT
BASE_MODEL=$BASE_MODEL
CAPTION_FILE=$CAPTION_FILE
DIFFUSERS_SRC=$DIFFUSERS_SRC
TRAIN_GPU_IDS=$TRAIN_GPU_IDS
WORKER_GPU_IDS=$WORKER_GPU_IDS
NUM_PROCESSES=$NUM_PROCESSES
TRAIN_BATCH_SIZE=$TRAIN_BATCH_SIZE
GRADIENT_ACCUMULATION_STEPS=$GRADIENT_ACCUMULATION_STEPS
FINETUNE_SEED=$FINETUNE_SEED
GENERATION_SEEDS=$GENERATION_SEEDS
CLASSIFIER_REPEATS=$CLASSIFIER_REPEATS
PROTOTYPE_PATH=$PROTOTYPE_PATH
DCS_PATH=$DCS_PATH
EOF
if [[ -f "$CONFIG" ]]; then
  if ! cmp -s "$CONFIG" "$CONFIG.new"; then
    echo "Resume configuration differs from $CONFIG" >&2
    diff -u "$CONFIG" "$CONFIG.new" || true
    exit 1
  fi
  rm "$CONFIG.new"
  if [[ "$RESUME" != "true" ]]; then
    echo "Run exists; set RESUME=true: $RUN_ID" >&2
    exit 1
  fi
else
  mv "$CONFIG.new" "$CONFIG"
fi

if [[ "$TRAIN_ONLY" != "true" && "$BUILD_PROTOTYPES" == "true" && ! -f "$PROTOTYPE_PATH" ]]; then
  (
    cd "$DISTILLATION_DIR"
    python gen_prototype.py \
      --batch_size 10 --spec text_supervision --contamination 0.1 \
      --data_dir "$DATA_ROOT" --dataset imagenet \
      --diffusion_checkpoints_path "$BASE_MODEL" --ipc 10 --km_expand 1 \
      --label_file_path "$LABEL_FILE" --save_prototype_path "$PROTOTYPE_DIR" \
      --save_text_prototype_path "$DCS_PATH" --seed 0 \
      --metajson_file "$CAPTION_FILE" --threshold 0.7 --tpk 30
  )
fi
if [[ "$TRAIN_ONLY" != "true" ]]; then
  [[ -f "$PROTOTYPE_PATH" && -f "$DCS_PATH" ]] || { echo "Missing prototype/DCS artifacts" >&2; exit 1; }
fi

train_mode() {
  local row="$1"
  local supervision="$2"
  local output="$MODEL_ROOT/$row"
  if [[ -f "$output/model_index.json" && -f "$output/training_summary.json" ]]; then
    echo "==> Reusing complete $row"
    return
  fi
  local resume_args=()
  if [[ "$RESUME" == "true" && -d "$output" ]]; then
    if compgen -G "$output/checkpoint-*" > /dev/null; then
      resume_args=(--resume-from-checkpoint latest)
    elif [[ -n "$(find "$output" -mindepth 1 -maxdepth 1 -print -quit 2>/dev/null)" ]]; then
      echo "Incomplete training output has no resumable checkpoint: $output" >&2
      echo "Use a new RUN_ID or move this directory aside." >&2
      return 1
    fi
  fi
  echo "==> Training $row on GPUs $TRAIN_GPU_IDS"
  CUDA_VISIBLE_DEVICES="$TRAIN_GPU_IDS" accelerate launch \
    --num_processes "$NUM_PROCESSES" --num_machines 1 \
    --mixed_precision "$MIXED_PRECISION" --dynamo_backend no \
    --main_process_port "$MAIN_PROCESS_PORT" \
    "$EXPERIMENT_DIR/train_text_to_image_supervision.py" \
    --pretrained-model "$BASE_MODEL" --train-root "$DATA_ROOT/train" \
    --caption-file "$CAPTION_FILE" --output-dir "$output" --supervision "$supervision" \
    --resolution 512 --train-batch-size "$TRAIN_BATCH_SIZE" \
    --gradient-accumulation-steps "$GRADIENT_ACCUMULATION_STEPS" \
    --num-train-epochs 8 --learning-rate 1e-5 --lr-scheduler constant \
    --lr-warmup-steps 0 --max-grad-norm 1 --mixed-precision "$MIXED_PRECISION" \
    --seed "$FINETUNE_SEED" --num-workers "$NUM_WORKERS" \
    --checkpointing-steps 500 --checkpoints-total-limit 2 \
    --loss-log-steps 50 --timestep-bins 10 --random-flip --gradient-checkpointing --use-ema \
    "${resume_args[@]}"
}

if [[ "$TRAIN" == "true" ]]; then
  for row in $TRAIN_ROWS; do
    case "$row" in
      label_ft) train_mode label_ft label ;;
      unpaired_ft) train_mode unpaired_ft unpaired ;;
      matched_ft) train_mode matched_ft matched ;;
      *) echo "Unknown TRAIN_ROWS entry: $row" >&2; exit 1 ;;
    esac
  done
fi
if [[ "$TRAIN_ONLY" == "true" ]]; then
  echo "Requested checkpoint training complete: $TRAIN_ROWS"
  exit 0
fi
for mode in label_ft unpaired_ft matched_ft; do
  [[ -f "$MODEL_ROOT/$mode/model_index.json" ]] || { echo "Missing model: $MODEL_ROOT/$mode" >&2; exit 1; }
done

resume_args=()
[[ "$RESUME" == "true" ]] && resume_args=(--resume)
if [[ "$GENERATE" == "true" ]]; then
  read -r -a gen_seeds <<< "$GENERATION_SEEDS"
  modes=(frozen label_ft unpaired_ft matched_ft)
  pids=()
  for index in "${!modes[@]}"; do
    mode="${modes[$index]}"
    model_args=()
    [[ "$mode" != "frozen" ]] && model_args=(--model "$mode=$MODEL_ROOT/$mode")
    worker_gpu="${WORKER_GPUS[$((index % ${#WORKER_GPUS[@]}))]}"
    echo "==> Generating row $mode on GPU $worker_gpu"
    CUDA_VISIBLE_DEVICES="$worker_gpu" python "$EXPERIMENT_DIR/generate_factorial.py" \
      --prototype "$PROTOTYPE_PATH" --dcs "$DCS_PATH" --base-model "$BASE_MODEL" \
      "${model_args[@]}" --supervisions "$mode" --prompts label correct shuffled \
      --output-root "$SYNTHETIC_ROOT" --generation-seeds "${gen_seeds[@]}" \
      --ipc 10 --strength 0.7 --guidance-scale 10 --num-inference-steps 50 \
      --shuffle-shift 1 --size 256 "${resume_args[@]}" &
    pids+=("$!")
    if (( ${#pids[@]} >= ${#WORKER_GPUS[@]} )); then
      for pid in "${pids[@]}"; do wait "$pid"; done
      pids=()
    fi
  done
  for pid in "${pids[@]}"; do wait "$pid"; done
fi

run_eval() {
  local gpu="$1" generation_seed="$2" condition="$3"
  local synthetic="$SYNTHETIC_ROOT/seed_$generation_seed/$condition"
  local seed_dir="$EVALUATION_ROOT/seed_$generation_seed"
  local log="$seed_dir/$condition.log"
  mkdir -p "$seed_dir"
  [[ -f "$synthetic/complete.json" ]] || { echo "Incomplete synthetic cell: $synthetic" >&2; return 1; }
  if grep -q "Best, last acc" "$log" 2>/dev/null; then
    echo "==> Reusing evaluation seed=$generation_seed $condition"
    return
  fi
  if [[ -s "$log" ]]; then
    if [[ "$RESUME" == "true" ]]; then mv "$log" "$log.interrupted_$(date -u +%Y%m%dT%H%M%SZ)"; else return 1; fi
  fi
  echo "==> Evaluating seed=$generation_seed $condition on GPU $gpu"
  (
    cd "$EVALUATION_DIR"
    CUDA_VISIBLE_DEVICES="$gpu" python train.py -d imagenet \
      --imagenet_dir "$synthetic" "$DATA_ROOT" -n resnet_ap --nclass 10 \
      --norm_type instance --ipc 10 --tag "text_supervision_${RUN_ID}_g${generation_seed}_${condition}" \
      --slct_type random --repeat "$CLASSIFIER_REPEATS" --spec nette --seed "$CLASSIFIER_SEED"
  ) > "$log" 2>&1
}

if [[ "$EVALUATE" == "true" ]]; then
  active=0 task=0 pids=()
  for generation_seed in $GENERATION_SEEDS; do
    for supervision in frozen label_ft unpaired_ft matched_ft; do
      for prompt in label correct shuffled; do
        condition="${supervision}_${prompt}"
        gpu="${WORKER_GPUS[$((task % ${#WORKER_GPUS[@]}))]}"; task=$((task + 1))
        run_eval "$gpu" "$generation_seed" "$condition" & pids+=("$!"); active=$((active + 1))
        if (( active >= MAX_PARALLEL_EVALS )); then
          wait "${pids[0]}"
          pids=("${pids[@]:1}"); active=$((active - 1))
        fi
      done
    done
  done
  for pid in "${pids[@]}"; do wait "$pid"; done
fi

if [[ "$SUMMARIZE" == "true" ]]; then
  python "$EXPERIMENT_DIR/summarize_results.py" --evaluation-root "$EVALUATION_ROOT" --output-dir "$SUMMARY_ROOT"
fi
if [[ -f "$MODEL_ROOT/label_ft/timestep_loss_epochs.csv" && -f "$MODEL_ROOT/unpaired_ft/timestep_loss_epochs.csv" && -f "$MODEL_ROOT/matched_ft/timestep_loss_epochs.csv" ]]; then
  python "$EXPERIMENT_DIR/plot_timestep_loss.py" --model-root "$MODEL_ROOT" --output "$SUMMARY_ROOT/timestep_loss.png"
fi
echo "Text-supervision factorial complete: $RUN_ROOT"
