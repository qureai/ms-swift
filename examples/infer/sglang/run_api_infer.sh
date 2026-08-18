#!/usr/bin/env bash
# Resumable async VLM inference against a HOSTED sglang OpenAI API.
# Start the server first with ../deploy/sglang.sh (must point --model at the
# config-fixed symlink checkpoint, else every request fails with qwen2_5_vl_text).
# Wait until it prints "Uvicorn running on http://0.0.0.0:8000", then run this.
# Outputs go to OUTPUT_DIR/task_{TASK}_temperature_{TEMPERATURE}_dataset_{DATASET_NAME}
# (created if missing); a rerun resumes by skipping image paths already there.
set -euo pipefail

# --- Edit these for your run -------------------------------------------------
SCRIPT=/cache/fast_data_nas71/janhavi/VLM_qure/vlm/swift/examples/infer/sglang/sglang_api_infer.py
DATASET=/cache/fast_data_nas71/janhavi/data/22_Nov_testing_master_updated_new_tag_corrected.csv
# DATASET=/cache/fast_data_nas71/janhavi/data/vlm_eval/13_July_2026_prod_filtered_high_score_reports_140k_eval.parquet
OUTPUT_DIR=/cache/fast_data_nas71/janhavi/vlm_results/SAHIL_Qwen_2_5_vl_8b_2611_corrected_pretraining_cleaned_bbox_segmed_prod
# OUTPUT_DIR=/cache/fast_data_nas71/janhavi/vlm_results/SAHIL_Qwen_2_5_vl_3b_2611_corrected_pretraining_cleaned_bbox_segmed_prod
# TASK=autoreporting
# TASK=long_asp
TASK=bbox
# DATASET_NAME=prod_140k
DATASET_NAME=internal
TEMPERATURE=0.5
# PROMPT="Generate a detailed ontology report of the given X-ray with Findings, Summary, Complete Reasoning, Impression, Recommendations, and Prominence Score."
# PROMPT="Generate Answer Set Programming (ASP)-based reasoning for the provided medical image. Include multi-level diagnostic analysis and structured, step-wise clinical reasoning. Present the logic in a transparent, explainable format appropriate for computational reasoning frameworks."
PROMPT="Generate the bbox of diseases and medical devices present in the image."
# server / client knobs
BASE_URL=http://localhost:8000/v1
MODEL=Qwen2.5_8B          # must match --served_model_name in ../deploy/sglang.sh
# MODEL=Qwen2.5_8B
CONCURRENCY=128            # in-flight requests; throughput saturates ~32 decodes on 4 GPUs
MAX_PIXELS=$((1500 * 1500))
MAX_TOKENS=4000           # <= server --max_new_tokens
SAVE_EVERY=30000          # successful rows per output parquet
# ----------------------------------------------------------------------------

python "$SCRIPT" \
    --dataset "$DATASET" \
    --prompt "$PROMPT" \
    --output-dir "$OUTPUT_DIR" \
    --task "$TASK" \
    --dataset-name "$DATASET_NAME" \
    --temperature "$TEMPERATURE" \
    --base-url "$BASE_URL" \
    --model "$MODEL" \
    --concurrency "$CONCURRENCY" \
    --max-pixels "$MAX_PIXELS" \
    --max-tokens "$MAX_TOKENS" \
    --save-every "$SAVE_EVERY"
