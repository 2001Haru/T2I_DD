#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export TOKENIZERS_PARALLELISM=false

SPECS="${SPECS:-imageA imageB imageC}"
P1_RUN_ID="${P1_RUN_ID:-p1_cluster_recoverability_v0}"
RUN_ID="${P2P3_RUN_ID:-p2p3_language_cluster_v0}"
P1_RUN_DIR="./results/p1_cluster_recoverability_runs/${P1_RUN_ID}"
OUTPUT_DIR="./results/p2p3_language_cluster_runs/${RUN_ID}"
CAPTION_CACHE_ROOT="${CAPTION_CACHE_ROOT:-./results/dcs_caption_cache}"
CAPTION_CACHE_NAME="${CAPTION_CACHE_NAME:-vlcp_dcs_class_aware}"
CLIP_MODEL="${CLIP_MODEL:-/linxi/models/CLIP/clip-vit-large-patch14}"
BATCH_SIZE="${BATCH_SIZE:-256}"
FOLDS="${FOLDS:-5}"
NULL_PARTITIONS="${NULL_PARTITIONS:-100}"
LINEAR_NULL_PARTITIONS="${LINEAR_NULL_PARTITIONS:-20}"
RETRIEVAL_NULL_PARTITIONS="${RETRIEVAL_NULL_PARTITIONS:-1000}"
RANDOM_SEED="${RANDOM_SEED:-20260803}"
RIDGE_ALPHA="${RIDGE_ALPHA:-1.0}"
DCS_THRESHOLD="${DCS_THRESHOLD:-0.7}"
DCS_TOP_K="${DCS_TOP_K:-30}"
CPU_JOBS="${CPU_JOBS:-4}"
DEVICE="${DEVICE:-cuda}"
RESUME="${RESUME:-false}"

CONFIG_FILE="${OUTPUT_DIR}/run_config.txt"
COMPLETE_FILE="${OUTPUT_DIR}/complete.json"
CONFIG_CONTENT="SPECS=${SPECS}
P1_RUN_ID=${P1_RUN_ID}
CAPTION_CACHE_ROOT=${CAPTION_CACHE_ROOT}
CAPTION_CACHE_NAME=${CAPTION_CACHE_NAME}
CLIP_MODEL=${CLIP_MODEL}
BATCH_SIZE=${BATCH_SIZE}
FOLDS=${FOLDS}
NULL_PARTITIONS=${NULL_PARTITIONS}
LINEAR_NULL_PARTITIONS=${LINEAR_NULL_PARTITIONS}
RETRIEVAL_NULL_PARTITIONS=${RETRIEVAL_NULL_PARTITIONS}
RANDOM_SEED=${RANDOM_SEED}
RIDGE_ALPHA=${RIDGE_ALPHA}
DCS_THRESHOLD=${DCS_THRESHOLD}
DCS_TOP_K=${DCS_TOP_K}
CPU_JOBS=${CPU_JOBS}
DEVICE=${DEVICE}"

for path in \
    "${P1_RUN_DIR}/assignments.csv" \
    "${P1_RUN_DIR}/feature_cache/clip.npz"; do
    if [[ ! -f "$path" ]]; then
        echo "Missing required P1 artifact: ${path}" >&2
        exit 1
    fi
done
if [[ ! -d "$CLIP_MODEL" ]]; then
    echo "Missing CLIP model: ${CLIP_MODEL}" >&2
    exit 1
fi
for spec in $SPECS; do
    cache_dir="${CAPTION_CACHE_ROOT}/${spec}/${CAPTION_CACHE_NAME}"
    if ! compgen -G "${cache_dir}/captions.rank*.jsonl" > /dev/null; then
        echo "Missing caption shards: ${cache_dir}" >&2
        exit 1
    fi
done

if [[ -e "$OUTPUT_DIR" ]]; then
    if [[ "$RESUME" != "true" ]]; then
        echo "P2/P3 run exists; set RESUME=true or choose another P2P3_RUN_ID: ${RUN_ID}" >&2
        exit 1
    fi
    if [[ ! -f "$CONFIG_FILE" || "$(<"$CONFIG_FILE")" != "$CONFIG_CONTENT" ]]; then
        echo "Resume configuration differs from ${CONFIG_FILE}" >&2
        exit 1
    fi
    if [[ -f "$COMPLETE_FILE" ]]; then
        echo "P2/P3 diagnostic already complete: ${OUTPUT_DIR}"
        cat "${OUTPUT_DIR}/summary.json"
        exit 0
    fi
else
    mkdir -p "$OUTPUT_DIR"
    printf '%s\n' "$CONFIG_CONTENT" > "$CONFIG_FILE"
fi

read -r -a SPEC_ARRAY <<< "$SPECS"
ARGS=(
    diagnose_language_cluster_alignment.py
    --specs "${SPEC_ARRAY[@]}"
    --p1-run-dir "$P1_RUN_DIR"
    --caption-cache-root "$CAPTION_CACHE_ROOT"
    --caption-cache-name "$CAPTION_CACHE_NAME"
    --clip-model "$CLIP_MODEL"
    --output-dir "$OUTPUT_DIR"
    --device "$DEVICE"
    --batch-size "$BATCH_SIZE"
    --folds "$FOLDS"
    --null-partitions "$NULL_PARTITIONS"
    --linear-null-partitions "$LINEAR_NULL_PARTITIONS"
    --retrieval-null-partitions "$RETRIEVAL_NULL_PARTITIONS"
    --random-seed "$RANDOM_SEED"
    --ridge-alpha "$RIDGE_ALPHA"
    --threshold "$DCS_THRESHOLD"
    --top-k "$DCS_TOP_K"
    --jobs "$CPU_JOBS"
)
if [[ "$RESUME" == "true" ]]; then
    ARGS+=(--resume)
fi

python "${ARGS[@]}"

echo "P2/P3 diagnostic complete: ${OUTPUT_DIR}"
echo "Summary: ${OUTPUT_DIR}/summary.json"
echo "Plot: ${OUTPUT_DIR}/language_cluster_alignment.png"
