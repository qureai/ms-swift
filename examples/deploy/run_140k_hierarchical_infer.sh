#!/usr/bin/env bash
# Hierarchical positive-findings-graph prompt over the FULL high-quality test set
# (all 145,735 images from segmed_prod_lc_140k_high_quality_eval_tags_report.parquet).
#
# Decode settings are identical to the earlier hierarchical run
# (Qwen3.5-2B_ckpt5191_nodule_fn_bbox_hierarchical_20260726_064236) so outputs are comparable.
#
# NOTE: logprobs are deliberately OFF here -- this run is for the graph/triples text itself.
# Turning them on would inflate the output ~100x (to several hundred GB). Add
# `--logprobs true --top_logprobs 5` only if a scored analysis is actually needed.
set -euo pipefail

CONDA_ENV=/home/ubuntu/miniconda3/envs/ms-swift-may-s4
export PATH="$CONDA_ENV/bin:$PATH"          # sglang JIT kernels need `ninja` on PATH
export IMAGE_MAX_TOKEN_NUM=2500
export FPS_MAX_FRAMES=12

CKPT=/fastdata4/sahil_work/VLM_qure/vlm/swift/output/Qwen3.5-2B_cached_packing_correct_data_split/v0-20260709-102020/checkpoint-5191
OUT=/fastdata4/sahil_work/VLM_qure/eval_outputs/Qwen3.5-2B_ckpt5191_140k_hierarchical

cd /fastdata4/sahil_work/VLM_qure/vlm/swift

python3.12 swift/cli/infer.py \
    --model "$CKPT" \
    --infer_backend sglang \
    --sglang_dp_size 8 \
    --sglang_tp_size 1 \
    --sglang_mem_fraction_static 0.9 \
    --val_dataset "$OUT/infer_input.jsonl" \
    --result_path "$OUT/result.jsonl" \
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
    --val_dataset_shuffle false \
    2>&1 | tee "$OUT/infer.log"
