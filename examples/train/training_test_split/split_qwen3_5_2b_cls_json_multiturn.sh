#!/usr/bin/env bash
set -euo pipefail

# Prepare the train/test split for Qwen/Qwen3.5-2B on the CLS-JSON-FIRST multi-turn dataset
# (see data_qure/Data.md §1c). This tokenizes the data ONCE and splits it into train/val,
# saving to disk with HF save_to_disk under $OUTPUT_DIR/{train,val}.
#
# No augmentation is used with this split, so pre-tokenizing/caching is fine (the frozen
# pixel_values are exactly what every epoch should see).
#
# Error handling: --strict false (also the default) means any row that throws during loading
# or tokenization is DROPPED (with a logged traceback, up to traceback_limit=10) and the run
# continues instead of crashing. See swift/dataset/preprocessor/core.py:192-203.
# Images: with SKIP_BROKEN_IMAGES=1 (set below), swift/template/vision_utils.py load_image
# re-raises on a read failure so the bad datapoint is DROPPED (not kept as a noise placeholder).
# Unsetting it restores the placeholder fallback used for inference/serving.
#
# Packing is intentionally NOT passed here: `swift export --to_cached_dataset` rejects
# --packing ("Packing will be handled during training"). packing / padding_free are TRAIN-time
# flags consumed by `swift sft --cached_dataset` (see footer). Note that packing auto-enables
# padding_free internally and makes group_by_length irrelevant, so group_by_length is unused.
#
# ⚠️ Qwen3.5 uses GatedDeltaNet. Per examples/train/liger/qure_full_4b.sh, with the
# transformers/swift backend `packing`/`padding_free` may NOT be supported for Qwen3.5. If the
# training run errors on them, drop to `--packing false --padding_free false`; the cached split
# below is unaffected either way (it only tokenizes).
#
# The cache is tokenizer/template-specific, so it is tied to Qwen3.5-2B + these template args
# (new_special_tokens, add_non_thinking_prefix, loss_scale, system, max_length, IMAGE_MAX_TOKEN_NUM).
# Keep them identical in the training run.

export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
# Cap threads PER worker so dataset_num_proc processes don't each spawn dozens of
# tokenizer/BLAS/torch threads and oversubscribe the box (load was 2500+ otherwise).
# Single-threaded workers => load ~= dataset_num_proc, and the encode actually flies.
export OMP_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
# Skip (drop) any datapoint whose image can't be read, instead of substituting a noise
# placeholder. load_image re-raises when this is set, and --strict false drops the row.
export SKIP_BROKEN_IMAGES=1
export IMAGE_MAX_TOKEN_NUM=2500
export VIDEO_MAX_TOKEN_NUM=128
export FPS_MAX_FRAMES=12
export CUDA_VISIBLE_DEVICES=0

SPECIAL_TOKENS_PATH="/localstorage/sahil_work/VLM_qure/vlm/swift/examples/train/liger/token.txt"
OUTPUT_DIR="/fastdata4/sahil_work/VLM_qure/cached_datasets/Qwen3.5-2B_cls_json_multiturn"
SPLIT_RATIO=0.01   # fraction of train reserved as the test/val split

# CLS-JSON-FIRST multi-turn data only. Every conversation opens with the detailed
# classification JSON turn (full ontology, presence true|false), then ASP / hierarchical graph
# / report branches in mixed combinations.
# Built by: vlm_inference_analysis/scripts/build_cls_json_multiturn_data.py
DATASET_PATHS=(
    "/fastdata4/sahil_work/VLM_qure/data_preparation/training_data/06_Aug_2026_cls_json_multiturn_asp_graph_report.json"
)

# --- resolve special tokens path (/localstorage <-> /fastdata4 fallback) ---
if [[ -f "$SPECIAL_TOKENS_PATH" ]]; then
    RESOLVED_SPECIAL_TOKENS_PATH="$SPECIAL_TOKENS_PATH"
elif [[ -f "${SPECIAL_TOKENS_PATH//\/localstorage/\/fastdata4}" ]]; then
    RESOLVED_SPECIAL_TOKENS_PATH="${SPECIAL_TOKENS_PATH//\/localstorage/\/fastdata4}"
    echo "Using alternate special tokens path: $RESOLVED_SPECIAL_TOKENS_PATH" >&2
else
    echo "Special tokens file not found: $SPECIAL_TOKENS_PATH" >&2
    exit 1
fi

# --- resolve each dataset path (/localstorage <-> /fastdata4 fallback, both directions) ---
RESOLVED_DATASET_PATHS=()
for DATASET_PATH in "${DATASET_PATHS[@]}"; do
    if [[ -f "$DATASET_PATH" ]]; then
        RESOLVED_DATASET_PATHS+=("$DATASET_PATH")
    elif [[ -f "${DATASET_PATH//\/localstorage/\/fastdata4}" ]]; then
        RESOLVED_DATASET_PATHS+=("${DATASET_PATH//\/localstorage/\/fastdata4}")
        echo "Using alternate dataset path: ${DATASET_PATH//\/localstorage/\/fastdata4}" >&2
    elif [[ -f "${DATASET_PATH//\/fastdata4/\/localstorage}" ]]; then
        RESOLVED_DATASET_PATHS+=("${DATASET_PATH//\/fastdata4/\/localstorage}")
        echo "Using alternate dataset path: ${DATASET_PATH//\/fastdata4/\/localstorage}" >&2
    else
        echo "Training dataset not found: $DATASET_PATH" >&2
        exit 1
    fi
done

swift export \
    --model Qwen/Qwen3.5-2B \
    --to_cached_dataset true \
    --dataset "${RESOLVED_DATASET_PATHS[@]}" \
    --split_dataset_ratio "$SPLIT_RATIO" \
    --dataset_shuffle true \
    --data_seed 42 \
    --new_special_tokens "$RESOLVED_SPECIAL_TOKENS_PATH" \
    --add_non_thinking_prefix true \
    --loss_scale ignore_empty_think \
    --max_length 28000 \
    --max_new_tokens 18000 \
    --torch_dtype bfloat16 \
    --system 'You are a helpful medical assistant.' \
    --dataset_num_proc 64 \
    --strict false \
    --use_hf true \
    --output_dir "$OUTPUT_DIR"

echo "Cached split written to: $OUTPUT_DIR/{train,val}"
echo "Train with:  swift sft --cached_dataset $OUTPUT_DIR \\"
echo "                 --packing true --padding_free true ...   # group_by_length intentionally unused"
