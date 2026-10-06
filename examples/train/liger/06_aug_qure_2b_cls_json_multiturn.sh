#!/usr/bin/env bash
set -euo pipefail

# Full SFT of Qwen/Qwen3.5-2B on the CLS-JSON-FIRST multi-turn data.
#
# Data: 06_Aug_2026_cls_json_multiturn_asp_graph_report.json, built by
#   vlm_inference_analysis/scripts/build_cls_json_multiturn_data.py
# Every conversation opens with the detailed classification JSON turn; the ASP / hierarchical
# graph / report turns follow in mixed combinations, so the model learns that the
# classification always comes first and any branch may follow it.
#
# WHY THE FULL ONTOLOGY MATTERS: turn-1 targets enumerate ~112 findings each with
# `presence: true|false` (~6 true). The previous JSON-classification data was filtered to
# positives only -- 0 `false` in 2,609,050 presence fields -- which made `presence` a
# syntactic constant with ~0.00002 nats of entropy and its logprob worthless as confidence.
# With both classes present, presence becomes a real per-tag decision, which is what makes
# (a) token-weighted loss on the true/false tokens meaningful and (b) per-tag eval possible.
#
# TWO-STAGE PIPELINE — this script is stage 2. Stage 1 encodes + caches:
#   bash examples/train/training_test_split/split_qwen3_5_2b.sh     # edit its dataset path
# then point this script at <CACHED_DIR>/{train,val}. Template/tokenizer args here MUST match
# the split run, especially --new_special_tokens (the cache bakes in token ids, so the
# embedding table has to be resized identically).
#
# NOTE ON LOSS WEIGHTING: --loss_scale is still `ignore_empty_think`, i.e. all response tokens
# are weighted equally. Up-weighting the `true`/`false` value tokens inside the classification
# JSON needs a custom swift loss_scale plugin; that is a separate change and this run does not
# include it.
#
# packing=true auto-enables padding_free (one knob) and both require flash attention.
# ⚠️ Qwen3.5 uses GatedDeltaNet; if packing/padding_free error out on this
# transformers/swift build, set `--packing false --padding_free false` — the cache stays valid.

export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export IMAGE_MAX_TOKEN_NUM=2500
export VIDEO_MAX_TOKEN_NUM=128
export FPS_MAX_FRAMES=12
export NPROC_PER_NODE=8
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

SPECIAL_TOKENS_PATH="/localstorage/sahil_work/VLM_qure/vlm/swift/examples/train/liger/token.txt"
CACHED_DIR="/fastdata4/sahil_work/VLM_qure/cached_datasets/Qwen3.5-2B_cls_json_multiturn"
RUN_NAME="Qwen3.5-2B_cls_json_multiturn"

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
    echo "Run the split/encode step first (examples/train/training_test_split/split_qwen3_5_2b.sh)." >&2
    exit 1
fi

if [[ ! -d "$RESOLVED_CACHED_DIR/train" || ! -d "$RESOLVED_CACHED_DIR/val" ]]; then
    echo "Expected $RESOLVED_CACHED_DIR/{train,val} to exist (from the split step)." >&2
    exit 1
fi

# per_device_train_batch_size counts PACKED sequences (each up to max_length=28000 tokens).
# Turn-1 targets are ~12k characters (~4k tokens) and the ASP branch adds ~18k characters, so
# these conversations are long -- keep max_length at 28000.
swift sft \
    --model Qwen/Qwen3.5-2B \
    --run_name "$RUN_NAME" \
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
    --output_dir "output/$RUN_NAME" \
    --system 'You are a helpful medical assistant.' \
    --deepspeed zero2 \
    --attn_impl flash_attention_2 \
    --use_hf true \
    --use_liger_kernel true \
    --report_to tensorboard wandb \
    --create_checkpoint_symlink true \
    --packing true \
    --padding_free true
