#!/bin/bash
# Re-predict the nodule/mass/granuloma FALSE-NEGATIVES (final FN = 5,663) with the
# bbox + hierarchical-graph prompts, using the SAME Qwen3.5-2B ckpt5191 and the SAME
# swift-infer recipe that produced the original 140k eval (see infer.log of
# Qwen3.5-2B_ckpt5191_140k_high_score_eval).
set -euo pipefail

CKPT=/fastdata4/sahil_work/VLM_qure/vlm/swift/output/Qwen3.5-2B_cached_packing_correct_data_split/v0-20260709-102020/checkpoint-5191
# Output dir is named from the data (nodule-FN, bbox + hierarchical) plus a run
# date-time stamp; pass an existing prepared dir as $1, else a fresh stamped one.
OUTDIR=${1:-/fastdata4/sahil_work/VLM_qure/eval_outputs/Qwen3.5-2B_ckpt5191_nodule_fn_bbox_hierarchical_$(date +%Y%m%d_%H%M%S)}

IMAGE_MAX_TOKEN_NUM=2500 \
VIDEO_MAX_TOKEN_NUM=128 \
FPS_MAX_FRAMES=12 \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
swift infer \
    --model "$CKPT" \
    --infer_backend sglang \
    --sglang_dp_size 8 \
    --sglang_tp_size 1 \
    --sglang_mem_fraction_static 0.9 \
    --val_dataset "$OUTDIR/infer_input.jsonl" \
    --result_path "$OUTDIR/result.jsonl" \
    --write_batch_size 1000 \
    --max_new_tokens 8192 \
    --max_length 28000 \
    --temperature 0 \
    --top_p 1.0 \
    --top_k 1 \
    --repetition_penalty 1.0 \
    --enable_thinking false \
    --system "You are a helpful medical assistant." \
    --use_hf true \
    --val_dataset_shuffle false
