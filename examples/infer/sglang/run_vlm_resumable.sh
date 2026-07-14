#!/usr/bin/env bash
# Resumable batched VLM inference over sglang (ms-swift in-process SglangEngine).
# Outputs go to OUTPUT_DIR/task_{TASK}_temperature_{TEMPERATURE}_dataset_{DATASET_NAME}
# (created if missing); a rerun resumes by skipping image paths already there.
# NOTE: keep --batch-size modest (<=512). sglang's real concurrency is set by
# --max-running-requests (32); a huge per-call batch (2000+) starves the sglang
# scheduler (GPU goes idle) and trips its watchdog. Control parquet file size via
# --save-every-n-batches instead (batch 256 * 40 ~= 10k rows/file).
set -euo pipefail

# --- Edit these for your run -------------------------------------------------
SCRIPT=/cache/fast_data_nas8/janhavi/VLM_qure/vlm/swift/examples/infer/sglang/run_sglang_inference_resumable.py
MODEL=/cache/fast_data_nas71/janhavi/vlm_ckpts/SAHIL_Qwen_2_5_vl_8b_2611_corrected_pretraining_cleaned_bbox_segmed_prod/v5-20251201-063754/checkpoint-14422
DATASET=/cache/fast_data_nas8/vlm_team_data/janhavi/22_Nov_testing_master_updated_new_tag_corrected.csv
OUTPUT_DIR=/cache/fast_data_nas71/janhavi/vlm_results/SAHIL_Qwen_2_5_vl_8b_2611_corrected_pretraining_cleaned_bbox_segmed_prod
TASK=autoreporting
DATASET_NAME=internal
TEMPERATURE=0.2
PROMPT="Generate a detailed ontology report of the given X-ray with Findings, Summary, Complete Reasoning, Impression, Recommendations, and Prominence Score."
# ----------------------------------------------------------------------------

export MAX_PIXELS=$((1500 * 1500))
export CUDA_VISIBLE_DEVICES=4,5,6,7

python "$SCRIPT" \
    --model "$MODEL" \
    --dataset "$DATASET" \
    --prompt "$PROMPT" \
    --output-dir "$OUTPUT_DIR" \
    --task "$TASK" \
    --dataset-name "$DATASET_NAME" \
    --temperature "$TEMPERATURE" \
    --tp-size 4 \
    --batch-size 256 \
    --save-every-n-batches 40 \
    --context-length 14000 \
    --mem-fraction-static 0.8 \
    --max-running-requests 32 \
    --max-tokens 8192
