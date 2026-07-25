#!/usr/bin/env bash
set -euo pipefail
shopt -s inherit_errexit

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
EXPERIMENT_DIR="$REPO_ROOT/experiments/visual_text_factorial"
EVALUATION_DIR="$REPO_ROOT/04_evaluation/Minimax"

: "${DATA_ROOT:?Set DATA_ROOT to the prepared ImageNette root containing train/ and val/}"

SOURCE_RUN_ROOT="${SOURCE_RUN_ROOT:-$REPO_ROOT/../vlcp_ablation_runs/author_checkpoint_pilot_v0}"
BASE_MODEL="${BASE_MODEL:-/linxi/models/VLCP/stable-diffusion-v1-5}"
PROTOTYPE_PATH="${PROTOTYPE_PATH:-$SOURCE_RUN_ROOT/prototypes/prior_alignment-ipc10-0.7-30-kmexpand1.json}"
DCS_PATH="${DCS_PATH:-$SOURCE_RUN_ROOT/prototypes/dcs.json}"
RUN_ID="${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
RUN_ROOT="${RUN_ROOT:-$REPO_ROOT/../vlcp_factorial_runs/$RUN_ID}"
GENERATION_SEEDS="${GENERATION_SEEDS:-0 1}"
CONDITIONS="${CONDITIONS:-no_visual_label no_visual_dcs no_visual_dcs_shuffled prototype_label prototype_dcs prototype_dcs_shuffled}"
STRENGTH="${STRENGTH:-0.7}"
GUIDANCE_SCALE="${GUIDANCE_SCALE:-10.0}"
NUM_INFERENCE_STEPS="${NUM_INFERENCE_STEPS:-50}"
SHUFFLE_SHIFT="${SHUFFLE_SHIFT:-1}"
CLASSIFIER_SEED="${CLASSIFIER_SEED:-0}"
RESUME="${RESUME:-false}"
GENERATE="${GENERATE:-true}"
EVALUATE="${EVALUATE:-true}"
SUMMARIZE="${SUMMARIZE:-$EVALUATE}"

SYNTHETIC_ROOT="$RUN_ROOT/synthetic"
EVALUATION_ROOT="$RUN_ROOT/evaluation"
mkdir -p "$RUN_ROOT" "$EVALUATION_ROOT"

CONFIG_FILE="$RUN_ROOT/run_config.txt"
CONFIG_CONTENT="DATA_ROOT=$(realpath "$DATA_ROOT")
BASE_MODEL=$BASE_MODEL
PROTOTYPE_PATH=$(realpath "$PROTOTYPE_PATH")
DCS_PATH=$(realpath "$DCS_PATH")
GENERATION_SEEDS=$GENERATION_SEEDS
CONDITIONS=$CONDITIONS
STRENGTH=$STRENGTH
GUIDANCE_SCALE=$GUIDANCE_SCALE
NUM_INFERENCE_STEPS=$NUM_INFERENCE_STEPS
SHUFFLE_SHIFT=$SHUFFLE_SHIFT
CLASSIFIER_SEED=$CLASSIFIER_SEED"
if [[ -f "$CONFIG_FILE" ]]; then
  if [[ "$(<"$CONFIG_FILE")" != "$CONFIG_CONTENT" ]]; then
    echo "Resume configuration differs from $CONFIG_FILE" >&2
    exit 1
  fi
  if [[ "$RESUME" != "true" ]]; then
    echo "Run already exists; set RESUME=true: $RUN_ROOT" >&2
    exit 1
  fi
else
  printf '%s\n' "$CONFIG_CONTENT" > "$CONFIG_FILE.tmp"
  mv "$CONFIG_FILE.tmp" "$CONFIG_FILE"
fi

RESUME_ENV="$RUN_ROOT/resume.env"
{
  printf 'export DATA_ROOT=%q\n' "$DATA_ROOT"
  printf 'export SOURCE_RUN_ROOT=%q\n' "$SOURCE_RUN_ROOT"
  printf 'export BASE_MODEL=%q\n' "$BASE_MODEL"
  printf 'export PROTOTYPE_PATH=%q\n' "$PROTOTYPE_PATH"
  printf 'export DCS_PATH=%q\n' "$DCS_PATH"
  printf 'export RUN_ID=%q\n' "$RUN_ID"
  printf 'export RUN_ROOT=%q\n' "$RUN_ROOT"
  printf 'export GENERATION_SEEDS=%q\n' "$GENERATION_SEEDS"
  printf 'export CONDITIONS=%q\n' "$CONDITIONS"
  printf 'export STRENGTH=%q\n' "$STRENGTH"
  printf 'export GUIDANCE_SCALE=%q\n' "$GUIDANCE_SCALE"
  printf 'export NUM_INFERENCE_STEPS=%q\n' "$NUM_INFERENCE_STEPS"
  printf 'export SHUFFLE_SHIFT=%q\n' "$SHUFFLE_SHIFT"
  printf 'export CLASSIFIER_SEED=%q\n' "$CLASSIFIER_SEED"
  printf 'export RESUME=true\n'
} > "$RESUME_ENV.tmp"
mv "$RESUME_ENV.tmp" "$RESUME_ENV"

python "$EXPERIMENT_DIR/check_dependencies.py"
python "$EXPERIMENT_DIR/validate_setup.py" \
  --data-root "$DATA_ROOT" \
  --base-model "$BASE_MODEL" \
  --prototype "$PROTOTYPE_PATH" \
  --dcs "$DCS_PATH"

resume_arg=()
if [[ "$RESUME" == "true" ]]; then
  resume_arg=(--resume)
fi

if [[ "$GENERATE" == "true" ]]; then
  read -r -a generation_seed_args <<< "$GENERATION_SEEDS"
  read -r -a condition_args <<< "$CONDITIONS"
  python "$EXPERIMENT_DIR/generate_factorial.py" \
    --prototype "$PROTOTYPE_PATH" \
    --dcs "$DCS_PATH" \
    --base-model "$BASE_MODEL" \
    --output-root "$SYNTHETIC_ROOT" \
    --generation-seeds "${generation_seed_args[@]}" \
    --conditions "${condition_args[@]}" \
    --strength "$STRENGTH" \
    --guidance-scale "$GUIDANCE_SCALE" \
    --num-inference-steps "$NUM_INFERENCE_STEPS" \
    --shuffle-shift "$SHUFFLE_SHIFT" \
    "${resume_arg[@]}"
fi

read -r -a condition_args <<< "$CONDITIONS"
if [[ "$EVALUATE" == "true" ]]; then
  for generation_seed in $GENERATION_SEEDS; do
    seed_eval_dir="$EVALUATION_ROOT/seed_$generation_seed"
    mkdir -p "$seed_eval_dir"
    for condition in "${condition_args[@]}"; do
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
        if [[ "$RESUME" == "true" ]]; then
          interrupted_log="$log_path.interrupted_$(date -u +%Y%m%dT%H%M%SZ)"
          mv "$log_path" "$interrupted_log"
          echo "==> Archived interrupted evaluation log: $interrupted_log"
        else
          echo "Incomplete evaluation log exists; refusing to append: $log_path" >&2
          echo "Rerun with RESUME=true to archive and restart this condition." >&2
          exit 1
        fi
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
          --tag "visual_text_${RUN_ID}_g${generation_seed}_${condition}" \
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

echo "Visual x text factorial complete: $RUN_ROOT"
