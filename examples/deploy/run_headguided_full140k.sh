#!/usr/bin/env bash
set -euo pipefail
# FULL head-guided 3-stage inference over the 136k eval set, unattended.
#   1. score the ML-Decoder head on all 136k  (8-GPU sharded)
#   2. build sens==spec (EER) positive lists
#   3. 3 dependent sglang passes: cls JSON -> ASP -> report  (head positives injected in stage 1)

SWIFT_ROOT="/fastdata4/sahil_work/VLM_qure/vlm/swift"
PY="/home/ubuntu/miniconda3/envs/ms-swift-may-s4/bin/python"
CKPT="$SWIFT_ROOT/output/Qwen3.5-2B_cls_json_mldecoder_joint/v13-20260809-033855/checkpoint-1800"
E="$SWIFT_ROOT/vlm_inference_analysis/results/joint_run/eval140k"
INPUT="$E/input.jsonl"
HEADDIR="$E/head_full"
THR="$E/head_eer_thresholds.json"
POS="$E/head_positives_eer_FULL.json"
OUT="$SWIFT_ROOT/vlm_inference_analysis/results/joint_run/headguided_full140k"
mkdir -p "$OUT"

echo "===== [1/3] head scoring on all 136k  $(date) ====="
bash "$SWIFT_ROOT/examples/deploy/run_head_scoring_140k.sh" "$INPUT" "$HEADDIR" "$CKPT"

echo "===== [2/3] build EER positives  $(date) ====="
$PY "$SWIFT_ROOT/vlm_inference_analysis/scripts/build_head_positives.py" "$HEADDIR" "$THR" "$POS"

echo "===== [3/3] 3-stage generation  $(date) ====="
$PY "$SWIFT_ROOT/examples/deploy/infer_headguided_3stage.py" \
    --ckpt "$CKPT" --input "$INPUT" --positives "$POS" --out_dir "$OUT" --dp 8 --only-scored

echo "===== ALL DONE  $(date)  ->  $OUT/final.jsonl ====="
