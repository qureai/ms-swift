#!/usr/bin/env bash
set -euo pipefail

# Full SFT of OpenGVLab/InternVL3_5-2B-Instruct on the PRE-TOKENIZED cached split produced by
# examples/train/training_test_split/split_internvl_3_5_2b.sh.
#
# Mirror of qure_full_2b_cached_packing.sh (the Qwen3.5-2B run) with packing enabled, adapted to
# InternVL. The cache stores already-encoded rows (token ids + processed pixel_values), so there
# is NO tokenization or image loading at train time; packing is applied HERE over the cached rows.
#
# PREREQUISITE: build the cache first ->  bash examples/train/training_test_split/split_internvl_3_5_2b.sh
#   That writes train/ and val/ as SEPARATE datasets (no top-level dataset_dict.json), and swift's
#   get_cached_dataset() calls load_from_disk() on each path directly, so point:
#       --cached_dataset      <DIR>/train
#       --cached_val_dataset  <DIR>/val
#   (NOT the bare parent dir.)
#
# Template/tokenizer args MUST match the split run (INPUT_SIZE, new_special_tokens,
# add_non_thinking_prefix, loss_scale, system, max_length). --new_special_tokens is required so the
# model's embedding table is resized to the token ids baked into the cache.
#
# packing=true auto-enables padding_free and makes group_by_length irrelevant (so it is unused).
# Both require flash attention. Unlike Qwen3.5 (GatedDeltaNet), InternVL3.5 uses standard attention,
# so packing + padding_free are cleanly supported here.

export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export INPUT_SIZE=448
export NPROC_PER_NODE=8
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
# Keep ALL regenerable caches/temp OFF the (shared, often-full) root disk -> on a big scratch fs.
# (A ModelScope/datasets/torchinductor cache on root is what crashed the export earlier.)
# Resolve a writable scratch base the same way the dataset/token paths below do: prefer
# /fastdata4, fall back to /localstorage (whichever exists and is writable on THIS machine).
SCRATCH_BASE=""
for base in /fastdata4 /localstorage; do
    if mkdir -p "$base/sahil_work/cache" 2>/dev/null; then
        SCRATCH_BASE="$base"
        break
    fi
done
if [[ -z "$SCRATCH_BASE" ]]; then
    echo "No writable scratch base found (tried /fastdata4 /localstorage)" >&2
    exit 1
fi
echo "Using scratch base: $SCRATCH_BASE" >&2
export MODELSCOPE_CACHE=$SCRATCH_BASE/sahil_work/cache/modelscope
export HF_DATASETS_CACHE=$SCRATCH_BASE/sahil_work/cache/hf_datasets
export TMPDIR=$SCRATCH_BASE/sahil_work/tmp
export TRITON_CACHE_DIR=$SCRATCH_BASE/sahil_work/cache/triton
export TORCHINDUCTOR_CACHE_DIR=$SCRATCH_BASE/sahil_work/cache/torchinductor
mkdir -p "$MODELSCOPE_CACHE" "$HF_DATASETS_CACHE" "$TMPDIR" "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR"

SPECIAL_TOKENS_PATH="/localstorage/sahil_work/VLM_qure/vlm/swift/examples/train/liger/token.txt"
CACHED_DIR="/fastdata4/sahil_work/VLM_qure/cached_datasets/InternVL3_5-2B_correct_data_split"

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

# --- resolve cached dir (/localstorage <-> /fastdata4 fallback, both directions) ---
if [[ -d "$CACHED_DIR" ]]; then
    RESOLVED_CACHED_DIR="$CACHED_DIR"
elif [[ -d "${CACHED_DIR//\/localstorage/\/fastdata4}" ]]; then
    RESOLVED_CACHED_DIR="${CACHED_DIR//\/localstorage/\/fastdata4}"
    echo "Using alternate cached dir: $RESOLVED_CACHED_DIR" >&2
elif [[ -d "${CACHED_DIR//\/fastdata4/\/localstorage}" ]]; then
    RESOLVED_CACHED_DIR="${CACHED_DIR//\/fastdata4/\/localstorage}"
    echo "Using alternate cached dir: $RESOLVED_CACHED_DIR" >&2
else
    echo "Cached dataset dir not found: $CACHED_DIR" >&2
    echo "Build it first: bash examples/train/training_test_split/split_internvl_3_5_2b.sh" >&2
    exit 1
fi

if [[ ! -d "$RESOLVED_CACHED_DIR/train" || ! -d "$RESOLVED_CACHED_DIR/val" ]]; then
    echo "Expected $RESOLVED_CACHED_DIR/{train,val} to exist (from the split step)." >&2
    echo "Build it first: bash examples/train/training_test_split/split_internvl_3_5_2b.sh" >&2
    exit 1
fi

# per_device_train_batch_size counts PACKED sequences (each up to max_length=28000 tokens), so it
# is small. Global tokens/step ~= per_device * NPROC * grad_accum * 28000. Tune per_device up if a
# packed 28k sequence fits comfortably on your GPUs.
swift sft \
    --model OpenGVLab/InternVL3_5-2B-Instruct \
    --run_name 'InternVL3_5-2B_cached_packing_correct_data_split' \
    --tuner_type full \
    --cached_dataset "$RESOLVED_CACHED_DIR/train" \
    --cached_val_dataset "$RESOLVED_CACHED_DIR/val" \
    --data_seed 42 \
    --new_special_tokens "$RESOLVED_SPECIAL_TOKENS_PATH" \
    --add_non_thinking_prefix true \
    --loss_scale ignore_empty_think \
    --max_new_tokens 25000 \
    --torch_dtype bfloat16 \
    --model_kwargs '{"low_cpu_mem_usage": false}' \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 1 \
    --learning_rate 1e-5 \
    --vit_lr 1e-5 \
    --aligner_lr 1e-5 \
    --freeze_vit false \
    --freeze_aligner false \
    --num_train_epochs 1 \
    --gradient_accumulation_steps 8 \
    --gradient_checkpointing true \
    --eval_steps 800 \
    --save_steps 800 \
    --logging_steps 5 \
    --max_length 28000 \
    --warmup_ratio 0.05 \
    --dataloader_num_workers 8 \
    --save_total_limit 5 \
    --save_only_model false \
    --output_dir output/InternVL3_5-2B_cached_packing_correct_data_split \
    --system 'You are a helpful medical assistant.' \
    --deepspeed zero3 \
    --attn_impl flash_attention_2 \
    --use_hf true \
    --use_liger_kernel true \
    --report_to tensorboard wandb \
    --create_checkpoint_symlink true \
    --packing true \
    --padding_free true
