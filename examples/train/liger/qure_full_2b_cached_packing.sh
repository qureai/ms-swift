#!/usr/bin/env bash
set -euo pipefail

# Full SFT of Qwen/Qwen3.5-2B on the PRE-TOKENIZED cached split produced by
# examples/train/training_test_split/split_qwen3_5_2b.sh.
#
# The cache stores already-encoded rows (token ids + processed pixel_values), so there is NO
# tokenization or image loading at train time. Packing is applied HERE, over the cached rows.
#
# IMPORTANT — cached_dataset layout:
#   The split saved train/ and val/ as SEPARATE datasets (no top-level dataset_dict.json), and
#   swift's get_cached_dataset() calls load_from_disk() on each path directly. So point:
#       --cached_dataset      <DIR>/train
#       --cached_val_dataset  <DIR>/val
#   (NOT the bare parent dir — load_from_disk on the parent would fail.)
#
# Template/tokenizer args MUST match the split run. In particular --new_special_tokens is
# required so the model's embedding table is resized to the token ids baked into the cache.
# (add_non_thinking_prefix/loss_scale/system were already applied at encode time; kept here for
# consistency of the collator/template.)
#
# packing=true auto-enables padding_free (they are one knob) and makes group_by_length
# irrelevant, so group_by_length is unused. Both require flash attention.
# ⚠️ Qwen3.5 uses GatedDeltaNet; per qure_full_4b.sh packing/padding_free may NOT be supported
# for Qwen3.5 on the transformers/swift backend. If training errors on them, set
# `--packing false --padding_free false` (the cache is still valid; you just lose packing).

export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export IMAGE_MAX_TOKEN_NUM=2500
export VIDEO_MAX_TOKEN_NUM=128
export FPS_MAX_FRAMES=12
export NPROC_PER_NODE=8
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

SPECIAL_TOKENS_PATH="/localstorage/sahil_work/VLM_qure/vlm/swift/examples/train/liger/token.txt"
CACHED_DIR="/fastdata4/sahil_work/VLM_qure/cached_datasets/Qwen3.5-2B_correct_data_split"

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
    exit 1
fi

if [[ ! -d "$RESOLVED_CACHED_DIR/train" || ! -d "$RESOLVED_CACHED_DIR/val" ]]; then
    echo "Expected $RESOLVED_CACHED_DIR/{train,val} to exist (from the split step)." >&2
    exit 1
fi

# per_device_train_batch_size counts PACKED sequences (each up to max_length=28000 tokens),
# so it is small. Global tokens/step ~= per_device * NPROC * grad_accum * 28000.
# Tune per_device up if a packed 28k sequence fits comfortably on your GPUs.
swift sft \
    --model Qwen/Qwen3.5-2B \
    --run_name 'Qwen3.5-2B_cached_packing_correct_data_split' \
    --tuner_type full \
    --cached_dataset "$RESOLVED_CACHED_DIR/train" \
    --cached_val_dataset "$RESOLVED_CACHED_DIR/val" \
    --data_seed 42 \
    --new_special_tokens "$RESOLVED_SPECIAL_TOKENS_PATH" \
    --add_non_thinking_prefix true \
    --loss_scale ignore_empty_think \
    --max_new_tokens 18000 \
    --torch_dtype bfloat16 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 1 \
    --learning_rate 1e-5 \
    --vit_lr 1e-5 \
    --aligner_lr 1e-5 \
    --freeze_vit false \
    --freeze_aligner false \
    --num_train_epochs 1 \
    --gradient_accumulation_steps 4 \
    --gradient_checkpointing true \
    --eval_steps 800 \
    --save_steps 800 \
    --logging_steps 5 \
    --max_length 28000 \
    --warmup_ratio 0.05 \
    --dataloader_num_workers 8 \
    --save_total_limit 5 \
    --save_only_model false \
    --output_dir output/Qwen3.5-2B_cached_packing_correct_data_split \
    --system 'You are a helpful medical assistant.' \
    --deepspeed zero2 \
    --attn_impl flash_attention_2 \
    --use_hf true \
    --use_liger_kernel true \
    --report_to tensorboard wandb \
    --create_checkpoint_symlink true \
    --packing true \
    --padding_free true
