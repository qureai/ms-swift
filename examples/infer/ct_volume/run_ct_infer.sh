#!/usr/bin/env bash
# Resumable IN-PROCESS inference for the 3D CT-volume checkpoint over a parquet/CSV.
# No server to start (CT volumes ride the <video> channel into model.visual_3d, which
# sglang/vLLM don't serve) -- this loads the checkpoint locally like notebooks/model_loading.ipynb.
# Outputs go to OUTPUT_DIR/task_{TASK}_temperature_{TEMPERATURE}_dataset_{DATASET_NAME}
# (created if missing); a rerun resumes by skipping volume paths already there.
#
# SPEED: this launches ONE PROCESS PER GPU (data-parallel), each in bfloat16 on a single GPU
# and owning a disjoint shard of the dataset. That is ~Nx faster than one model spread across
# N GPUs (which only runs one forward at a time). bfloat16 (~9GB weights) makes the 4B model
# fit comfortably on one GPU with room for the KV cache. All shards share OUTPUT_DIR and resume
# independently. See the knobs below to also raise MAX_BATCH_SIZE or try flash-attn.
#
# ENV: run with the `swift_infer` conda env (the notebook's kernel) -- the qwen3_5 checkpoint
# needs transformers>=5.x, which swift_july's 4.57.1 does NOT have (fails with KeyError: 'qwen3_5').
#   conda activate swift_infer   # or set PY below to that env's python
set -euo pipefail

# --- Edit these for your run -------------------------------------------------
PY=python                  # e.g. /cache/fast_data_nas71/janhavi/miniconda3/envs/swift_infer/bin/python
SCRIPT=/cache/fast_data_nas71/janhavi/VLM_qure/vlm/swift/examples/infer/ct_volume/ct_volume_infer.py
DATASET=/cache/fast_data_nas71/janhavi/data/ct_eval/ct_rate_eval.parquet
IMAGE_COLUMN=volume_uri     # column in DATASET holding the CT volume path (s3:// / local / .nii/.npy/DICOM-dir)
MODEL=/cache/fast_data_nas71/janhavi/VLM_qure/vlm/swift/vlm_ckpts/2907_qwen3_5_4b_atlas_3d_ct/v6-20260804-010105/checkpoint-26000
OUTPUT_DIR=/cache/fast_data_nas71/janhavi/vlm_results/2907_qwen3_5_4b_atlas_3d_ct
TASK=autoreport
DATASET_NAME=ctrate
TEMPERATURE=0.4
PROMPT="Generate the report for the given CT volume."
# PROMPT="Classify the abnormalities in the given CT volume in JSON format with presence, size, location and very short characterization."
# PROMPT="Classify the abnormalities from the given CT volume."
SYSTEM="You are a helpful medical assistant."
# engine knobs
GPUS=(0 1 2 3)             # ONE process is launched per entry -> data-parallel across these GPUs
TORCH_DTYPE=bfloat16      # bfloat16: ~2x faster than the float32 checkpoint + fits on one GPU. '' = float32
ATTN_IMPL=sdpa            # sdpa (safe) | flash_attn (faster but may error on the CT path) | eager
MAX_BATCH_SIZE=1          # volumes per forward pass PER process; try 2-4 in bf16 for more throughput
MAX_TOKENS=4096
TOP_P=0.8
TOP_K=20
SAVE_EVERY=10000          # successful rows per output parquet (per process)
# ----------------------------------------------------------------------------

NUM_SHARDS=${#GPUS[@]}
DTYPE_ARG=(); [[ -n "$TORCH_DTYPE" ]] && DTYPE_ARG=(--torch-dtype "$TORCH_DTYPE")
ATTN_ARG=();  [[ -n "$ATTN_IMPL"   ]] && ATTN_ARG=(--attn-impl "$ATTN_IMPL")

echo "Launching $NUM_SHARDS data-parallel workers on GPUs: ${GPUS[*]}"
pids=()
for i in "${!GPUS[@]}"; do
    gpu=${GPUS[$i]}
    echo "  -> shard $i/$NUM_SHARDS on GPU $gpu"
    "$PY" "$SCRIPT" \
        --dataset "$DATASET" \
        --image-column "$IMAGE_COLUMN" \
        --prompt "$PROMPT" \
        --model "$MODEL" \
        --output-dir "$OUTPUT_DIR" \
        --task "$TASK" \
        --dataset-name "$DATASET_NAME" \
        --system "$SYSTEM" \
        --temperature "$TEMPERATURE" \
        --gpus "$gpu" \
        --num-shards "$NUM_SHARDS" \
        --shard-id "$i" \
        --max-batch-size "$MAX_BATCH_SIZE" \
        --max-tokens "$MAX_TOKENS" \
        --top-p "$TOP_P" \
        --top-k "$TOP_K" \
        --save-every "$SAVE_EVERY" \
        "${DTYPE_ARG[@]}" "${ATTN_ARG[@]}" \
        > "${OUTPUT_DIR%/}_shard${i}.log" 2>&1 &
    pids+=($!)
done

echo "PIDs: ${pids[*]}   (logs: ${OUTPUT_DIR%/}_shard*.log)"
# Wait for all shards; exit non-zero if any shard fails.
rc=0
for pid in "${pids[@]}"; do wait "$pid" || rc=$?; done
echo "All shards finished (exit $rc)."
exit $rc
