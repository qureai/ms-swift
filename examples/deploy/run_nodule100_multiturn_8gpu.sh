#!/usr/bin/env bash
# Fan the 100-case multi-turn run across all 8 GPUs, one shard per GPU.
# pt/sdpa is ~20 tok/s single-stream and each case is 3 long turns, so one GPU would take
# hours; 8 disjoint shards bring it to ~1h. Each shard writes its own result file; merge after.
set -euo pipefail
CONDA=/home/ubuntu/miniconda3/envs/ms-swift-may-s4
export PATH="$CONDA/bin:$PATH"
ROOT=/fastdata4/sahil_work/VLM_qure/vlm/swift
export PYTHONPATH="$ROOT:/fastdata4/sahil_work/VLM_qure/vision_foundation_2d/foundation_model_accelerate"

CKPT="${CKPT:-$ROOT/output/Qwen3.5-2B_cls_json_mldecoder_joint/v9-20260807-004153/checkpoint-900}"
INPUT="${1:-$ROOT/vlm_inference_analysis/results/joint_run/nodule100_input.jsonl}"
OUTDIR="${2:-$ROOT/vlm_inference_analysis/results/joint_run/nodule100_multiturn}"
N="${NUM_SHARDS:-8}"
mkdir -p "$OUTDIR"

for s in $(seq 0 $((N-1))); do
    CUDA_VISIBLE_DEVICES=$s nohup python "$ROOT/examples/deploy/multiturn_cls_asp_report.py" \
        --ckpt "$CKPT" --input "$INPUT" \
        --out "$OUTDIR/result_shard${s}.jsonl" \
        --shard "$s" --num-shards "$N" \
        --batch "${BATCH:-16}" \
        --max-new-tokens 28000 --max-len 28000 \
        > "$OUTDIR/shard${s}.log" 2>&1 &
    echo "shard $s -> GPU $s (pid $!)"
done
echo "all $N shards launched. merge with: cat $OUTDIR/result_shard*.jsonl > $OUTDIR/result.jsonl"
