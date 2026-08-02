#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export TOKENIZERS_PARALLELISM=false

SPECS="${SPECS:-imageA imageB imageC}"
DINO_MODEL="${DINO_MODEL:-/linxi/models/DINOv2/dinov2-base}"
CLIP_MODEL="${CLIP_MODEL:-/linxi/models/CLIP/clip-vit-large-patch14}"
IPC="${IPC:-10}"
N_NEIGHBORS="${N_NEIGHBORS:-85}"
MIN_CLUSTER_SIZE="${MIN_CLUSTER_SIZE:-55}"
FOLDS="${FOLDS:-5}"
NULL_PARTITIONS="${NULL_PARTITIONS:-100}"
LINEAR_NULL_PARTITIONS="${LINEAR_NULL_PARTITIONS:-20}"
RANDOM_SEED="${RANDOM_SEED:-20260802}"
RIDGE_ALPHA="${RIDGE_ALPHA:-1.0}"
BATCH_SIZE="${BATCH_SIZE:-64}"
CPU_JOBS="${CPU_JOBS:-4}"
DEVICE="${DEVICE:-cuda}"
RUN_ID="${P1_RUN_ID:-p1_cluster_recoverability_v0}"
RESUME="${RESUME:-false}"

OUTPUT_DIR="./results/p1_cluster_recoverability_runs/${RUN_ID}"
CONFIG_FILE="${OUTPUT_DIR}/run_config.txt"
COMPLETE_FILE="${OUTPUT_DIR}/complete.json"
SAVED_CLUSTERS="${IPC}_n_${N_NEIGHBORS}_s_${MIN_CLUSTER_SIZE}_saved_clusters.pkl"

python - <<'PY'
import joblib
import matplotlib
import sklearn
import torch
import transformers

print(
    "P1 dependencies:",
    f"torch={torch.__version__}",
    f"transformers={transformers.__version__}",
    f"sklearn={sklearn.__version__}",
    f"joblib={joblib.__version__}",
)
PY

for model_path in "$DINO_MODEL" "$CLIP_MODEL"; do
    if [[ ! -d "$model_path" ]]; then
        echo "Missing local encoder model: ${model_path}" >&2
        exit 1
    fi
done

for spec in $SPECS; do
    feature_file="./results/clusterfile/${spec}/original_features_cache.pkl_0"
    center_file="./results/clusterfile/${spec}/${IPC}_n_${N_NEIGHBORS}_s_${MIN_CLUSTER_SIZE}_saved_clusters_0.pkl"
    if [[ ! -f "$feature_file" || ! -f "$center_file" ]]; then
        echo "Missing fixed CoDA artifacts for ${spec}:" >&2
        echo "  ${feature_file}" >&2
        echo "  ${center_file}" >&2
        exit 1
    fi
done

CONFIG_CONTENT="SPECS=${SPECS}
DINO_MODEL=${DINO_MODEL}
CLIP_MODEL=${CLIP_MODEL}
IPC=${IPC}
N_NEIGHBORS=${N_NEIGHBORS}
MIN_CLUSTER_SIZE=${MIN_CLUSTER_SIZE}
FOLDS=${FOLDS}
NULL_PARTITIONS=${NULL_PARTITIONS}
LINEAR_NULL_PARTITIONS=${LINEAR_NULL_PARTITIONS}
RANDOM_SEED=${RANDOM_SEED}
RIDGE_ALPHA=${RIDGE_ALPHA}
BATCH_SIZE=${BATCH_SIZE}
CPU_JOBS=${CPU_JOBS}
DEVICE=${DEVICE}"

if [[ -e "$OUTPUT_DIR" ]]; then
    if [[ "$RESUME" != "true" ]]; then
        echo "P1 run already exists; set RESUME=true or choose a new P1_RUN_ID: ${RUN_ID}" >&2
        exit 1
    fi
    if [[ ! -f "$CONFIG_FILE" || "$(<"$CONFIG_FILE")" != "$CONFIG_CONTENT" ]]; then
        echo "Resume configuration differs from ${CONFIG_FILE}" >&2
        exit 1
    fi
    if [[ -f "$COMPLETE_FILE" ]]; then
        echo "P1 run is already complete: ${OUTPUT_DIR}"
        cat "${OUTPUT_DIR}/summary.json"
        exit 0
    fi
else
    mkdir -p "$OUTPUT_DIR"
    printf '%s\n' "$CONFIG_CONTENT" > "$CONFIG_FILE"
fi

read -r -a SPEC_ARRAY <<< "$SPECS"
ARGS=(
    diagnose_cluster_recoverability.py
    --specs "${SPEC_ARRAY[@]}"
    --misc-dir ./misc
    --cluster-root ./results/clusterfile
    --saved-clusters-base-name "$SAVED_CLUSTERS"
    --ipc "$IPC"
    --dino-model "$DINO_MODEL"
    --clip-model "$CLIP_MODEL"
    --output-dir "$OUTPUT_DIR"
    --device "$DEVICE"
    --batch-size "$BATCH_SIZE"
    --folds "$FOLDS"
    --null-partitions "$NULL_PARTITIONS"
    --linear-null-partitions "$LINEAR_NULL_PARTITIONS"
    --random-seed "$RANDOM_SEED"
    --ridge-alpha "$RIDGE_ALPHA"
    --jobs "$CPU_JOBS"
)
if [[ "$RESUME" == "true" ]]; then
    ARGS+=(--resume)
fi

python "${ARGS[@]}"

echo "P1 diagnostics complete: ${OUTPUT_DIR}"
echo "Primary decision: ${OUTPUT_DIR}/summary.json"
echo "Plot: ${OUTPUT_DIR}/cluster_recoverability.png"
