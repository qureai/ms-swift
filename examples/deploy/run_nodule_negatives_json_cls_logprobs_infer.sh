#!/usr/bin/env bash
# Run the JSON-classification prompt over the 14,583 GT nodule-NEGATIVE studies sampled by
# build_nodule_auc_cohort.py -- the negative half needed for a nodule presence AUC.
#
# Same model / decode settings as Qwen3.5-2B_ckpt5191_nodule_positives_json_cls_20260729_115821,
# so responses should reproduce; the only additions are --logprobs / --top_logprobs.
# Needs sglang logprob support in swift/infer_engine/sglang_engine.py.
set -euo pipefail

CONDA_ENV=/home/ubuntu/miniconda3/envs/ms-swift-may-s4
export PATH="$CONDA_ENV/bin:$PATH"          # sglang JIT kernels need `ninja` on PATH
export IMAGE_MAX_TOKEN_NUM=2500
export FPS_MAX_FRAMES=12

CKPT=/fastdata4/sahil_work/VLM_qure/vlm/swift/output/Qwen3.5-2B_cached_packing_correct_data_split/v0-20260709-102020/checkpoint-5191
OUT=/fastdata4/sahil_work/VLM_qure/eval_outputs/Qwen3.5-2B_ckpt5191_nodule_negatives_json_cls_logprobs

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
    --logprobs true \
    --top_logprobs 5 \
    --token_ids_logprob 804 867 819 1802 3721 2827 2912 3439 2434 3913 \
    2>&1 | tee "$OUT/infer.log"

# --token_ids_logprob ids are every single-token spelling of the presence value, so
# log p(false) is recorded at every position even when it is far outside the top-5:
#   804=' true'  867=' false'  819=' null'  1802='true' 3721='false' 2827='null'
#   2912=' True' 3439=' False' 2434='True'  3913='False'
