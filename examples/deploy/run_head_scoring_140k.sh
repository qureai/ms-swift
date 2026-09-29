#!/usr/bin/env bash
set -euo pipefail
# Score the ML-Decoder head over the FULL 140k eval set, sharded 1 shard / GPU.
#   bash run_head_scoring_140k.sh <input.jsonl> <out_dir> [ckpt]
# Output: <out_dir>/shard_<i>.jsonl  ({key,image,probs[132]})  +  <out_dir>/shard_0.jsonl.tags.json

CONDA_ENV="${CONDA_ENV:-/home/ubuntu/miniconda3/envs/ms-swift-may-s4}"
SWIFT_ROOT="/fastdata4/sahil_work/VLM_qure/vlm/swift"
FM_REPO="${FM_ACCELERATE_REPO:-/fastdata4/sahil_work/VLM_qure/vision_foundation_2d/foundation_model_accelerate}"
export PATH="$CONDA_ENV/bin:$PATH"
export PYTHONPATH="$SWIFT_ROOT:$FM_REPO:${PYTHONPATH:-}"
export IMAGE_MAX_TOKEN_NUM="${IMAGE_MAX_TOKEN_NUM:-2500}"

INPUT="${1:?usage: $0 <input.jsonl> <out_dir> [ckpt]}"
OUT_DIR="${2:?usage: $0 <input.jsonl> <out_dir> [ckpt]}"
CKPT="${3:-$SWIFT_ROOT/output/Qwen3.5-2B_cls_json_mldecoder_joint/v13-20260809-033855/checkpoint-1800}"
NGPU="${NGPU:-8}"
mkdir -p "$OUT_DIR"

pids=()
for i in $(seq 0 $((NGPU-1))); do
    CUDA_VISIBLE_DEVICES=$i python "$SWIFT_ROOT/examples/deploy/score_head_batch.py" \
        --ckpt "$CKPT" --input "$INPUT" --out "$OUT_DIR/shard_${i}.jsonl" \
        --shard "$i" --num-shards "$NGPU" --batch 128 --io-workers 24 \
        > "$OUT_DIR/shard_${i}.log" 2>&1 &
    pids+=($!)
    echo "launched shard $i on GPU $i (pid ${pids[-1]})"
done
echo "waiting for ${#pids[@]} shards ..."
for p in "${pids[@]}"; do wait "$p"; done
echo "--- all shards done. rows: $(cat "$OUT_DIR"/shard_*.jsonl | wc -l)"
