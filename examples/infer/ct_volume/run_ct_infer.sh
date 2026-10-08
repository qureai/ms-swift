#!/usr/bin/env bash
# Resumable IN-PROCESS inference for the 3D CT-volume checkpoint over a test set.
# No server (CT volumes ride the <video> channel into model.visual_3d, which sglang/vLLM don't serve) --
# this loads the checkpoint locally like notebooks/model_loading.ipynb.
#
# STREAMLINED USAGE: pick a prompt from the PROMPTS registry (PROMPT_KEY) and a test set (DATASET); the
# TASK is auto-derived from the prompt and the DATASET_NAME from the filename (override either if you want).
# Outputs are auto-routed to  OUTPUT_ROOT/<model_name>/v<N>_checkpoint_<ckpt>/task_..._dataset_.../ , so you
# never edit the output path per model -- it's parsed from the checkpoint path.
#
# DATA: .json/.jsonl ShareGPT test sets (volume read from videos[0]; the record's messages are saved
# alongside the output for reference) OR .parquet/.csv (must contain the --image-column volume path).
#
# SPEED: one process per GPU (data-parallel), each in bf16 on a single GPU owning a disjoint dataset shard
# (~Nx faster than one model spread over N GPUs). max_batch_size batches volumes per forward (the Atlas
# encoder is now batched too), and max_tokens is auto-sized per task so short tasks don't over-generate.
#
# ENV: run with the env that matches the checkpoint (the qwen3_5 checkpoint needs transformers>=5.x, i.e.
# `swift_infer`, NOT swift_july's 4.57.1). Set PY below or `conda activate` first.
set -euo pipefail

# ================================ PROMPT / TASK REGISTRY ========================================
# One entry per task. PROMPTS[key]=the exact instruction; PROMPT_TASK[key]=task label used in the output
# folder name; PROMPT_MAXTOK[key]=auto max_new_tokens (short tasks generate less -> faster). Add rows here.
# PROMPT2[key] (optional) makes a task TWO-TURN: turn1=PROMPTS[key], turn2=PROMPT2[key] (volume + turn-1
# answer stay in context). Leave a key out of PROMPT2 for a normal single-turn task.
declare -A PROMPTS PROMPT_TASK PROMPT_MAXTOK PROMPT2
PROMPTS[cls_tokens]="Classify the abnormalities from the given CT volume."
PROMPT_TASK[cls_tokens]="cls_tokens";               PROMPT_MAXTOK[cls_tokens]=256

PROMPTS[tag_json]="Classify the abnormalities in the given CT volume in JSON format with presence, size, location and very short characterization."
PROMPT_TASK[tag_json]="tag_json";                   PROMPT_MAXTOK[tag_json]=1024

PROMPTS[mini_report]="Generate a structured report in JSON for the given CT volume with the following fields: findings, summary, complete_reasoning, impression, recommendations, and prominence_score."
PROMPT_TASK[mini_report]="mini_report";             PROMPT_MAXTOK[mini_report]=2048

PROMPTS[full_report]="Generate a complete structured radiology report in JSON for the given CT volume, covering every anatomical system as well as the findings, summary, complete_reasoning, impression, recommendations, and prominence_score."
PROMPT_TASK[full_report]="full_report";             PROMPT_MAXTOK[full_report]=4096

PROMPTS[cls_mini_combined]="For the given CT volume, first classify the abnormalities as tags, then generate a structured report in JSON with the following fields: findings, summary, complete_reasoning, impression, recommendations, and prominence_score."
PROMPT_TASK[cls_mini_combined]="cls_mini_combined"; PROMPT_MAXTOK[cls_mini_combined]=4096

# TWO-TURN task (cls -> report): turn1 classifies, turn2 generates the report with the volume + turn-1
# answer in context. Matches 13_Aug_2026_ct_cls_plus_mini_report_multiturn_testing.json.
PROMPTS[cls_mini_multiturn]="Classify the abnormalities from the given CT volume."
PROMPT2[cls_mini_multiturn]="Now generate a structured report in JSON with the following fields: findings, summary, complete_reasoning, impression, recommendations, and prominence_score."
PROMPT_TASK[cls_mini_multiturn]="cls_mini_multiturn"; PROMPT_MAXTOK[cls_mini_multiturn]=4096

# ================================ EDIT THESE FOR YOUR RUN =======================================
PROMPT_KEY=mini_report     # <-- which prompt/task to run (a key from PROMPTS above)
DATASET=/cache/fast_data_nas71/janhavi/data/ct_training_data/testing/13_Aug_2026_ct_mini_structured_report_testing.json

MODEL=/cache/fast_data_nas71/janhavi/VLM_qure/vlm/swift/vlm_ckpts/1308_qwen3_5_4b_atlas_3d_ct/v1-20260814-115348/checkpoint-26500
OUTPUT_ROOT=/cache/fast_data_nas71/janhavi/vlm_results   # results root; per-model/checkpoint subdirs are auto-created

# Optional overrides (leave EMPTY to auto-derive) --------------------------------------------------
TASK=""                    # empty -> PROMPT_TASK[$PROMPT_KEY]
MAX_TOKENS=""              # empty -> PROMPT_MAXTOK[$PROMPT_KEY]
DATASET_NAME=""            # empty -> derived from DATASET filename (date prefix / _testing suffix stripped)
PROMPT=""                  # empty -> PROMPTS[$PROMPT_KEY]  (set this to run a fully custom prompt)

PY=python                  # e.g. /cache/fast_data_nas71/janhavi/miniconda3/envs/swift_infer/bin/python
SCRIPT=/cache/fast_data_nas71/janhavi/VLM_qure/vlm/swift/examples/infer/ct_volume/ct_volume_infer.py
IMAGE_COLUMN=volume_uri    # .json: volume path (videos[0]) is loaded into this column. .parquet/.csv: must already have it
SYSTEM="You are a helpful medical assistant."
TEMPERATURE=0.4
# engine knobs
GPUS=(0 1 2 3)             # ONE process launched per entry -> data-parallel across these GPUs
TORCH_DTYPE=bfloat16       # bf16: ~2x faster than fp32 + fits one GPU. '' = float32
ATTN_IMPL=flash_attn       # flash attention (requested). Fall back to sdpa if it errors on the CT path
MAX_BATCH_SIZE=6           # volumes per forward pass PER process (bumped from 4; if a shard OOMs, drop to 4 -- resume continues)
TOP_P=0.8
TOP_K=20
SAVE_EVERY=2000            # flush a parquet every 2k successes -> partial progress persists (resumable)
# ===============================================================================================

# ---- resolve prompt/task/max_tokens from the registry (explicit values win) --------------------
if [[ -z "${PROMPTS[$PROMPT_KEY]+x}" ]]; then
    echo "Unknown PROMPT_KEY '$PROMPT_KEY'. Available: ${!PROMPTS[*]}" >&2; exit 1
fi
[[ -z "$PROMPT"     ]] && PROMPT="${PROMPTS[$PROMPT_KEY]}"
[[ -z "$TASK"       ]] && TASK="${PROMPT_TASK[$PROMPT_KEY]}"
[[ -z "$MAX_TOKENS" ]] && MAX_TOKENS="${PROMPT_MAXTOK[$PROMPT_KEY]:-4096}"
# TWO-TURN if this prompt key has a PROMPT2 entry (empty otherwise -> single-turn)
PROMPT2_TEXT=""; [[ -n "${PROMPT2[$PROMPT_KEY]+x}" ]] && PROMPT2_TEXT="${PROMPT2[$PROMPT_KEY]}"

# ---- auto dataset name from the filename (strip a leading DD_Mon_YYYY_ and a trailing _testing) --
if [[ -z "$DATASET_NAME" ]]; then
    DATASET_NAME=$(basename "$DATASET"); DATASET_NAME=${DATASET_NAME%.*}
    DATASET_NAME=$(printf '%s' "$DATASET_NAME" | sed -E 's/^[0-9]{1,2}_[A-Za-z]+_[0-9]{4}_//; s/_(testing|test|eval)$//')
fi

# ---- auto output dir: OUTPUT_ROOT/<model_name>/v<N>_checkpoint_<ckpt> ---------------------------
# MODEL = .../<model_name>/v<N>-<timestamp>/checkpoint-<ckpt>  ->  <model_name> / v<N>_checkpoint_<ckpt>
CKPT_NUM=$(basename "$MODEL"); CKPT_NUM=${CKPT_NUM#checkpoint-}
VERSION_DIR=$(basename "$(dirname "$MODEL")"); VERSION=${VERSION_DIR%%-*}   # v6-20260804-010105 -> v6
MODEL_NAME=$(basename "$(dirname "$(dirname "$MODEL")")")
if [[ "$VERSION" == v* && "$CKPT_NUM" =~ ^[0-9]+$ ]]; then
    CKPT_TAG="${VERSION}_checkpoint_${CKPT_NUM}"
else                                   # non-standard path (e.g. a flat dir) -> fall back to the basename
    CKPT_TAG=$(basename "$MODEL")
    MODEL_NAME=$(basename "$(dirname "$MODEL")")
fi
OUTPUT_DIR="${OUTPUT_ROOT%/}/${MODEL_NAME}/${CKPT_TAG}"
mkdir -p "$OUTPUT_DIR"
# shard logs go in a logs/ subdir inside the model output dir (task+dataset prefixed so concurrent
# tasks in the same checkpoint dir don't clash) -- not loose files in the checkpoint dir.
LOG_DIR="${OUTPUT_DIR%/}/logs"
mkdir -p "$LOG_DIR"

echo "PROMPT_KEY=$PROMPT_KEY  TASK=$TASK  MAX_TOKENS=$MAX_TOKENS"
echo "DATASET=$DATASET  (name=$DATASET_NAME)"
echo "MODEL=$MODEL"
echo "OUTPUT_DIR=$OUTPUT_DIR"
echo "PROMPT=\"$PROMPT\""

DTYPE_ARG=(); [[ -n "$TORCH_DTYPE" ]] && DTYPE_ARG=(--torch-dtype "$TORCH_DTYPE")
ATTN_ARG=();  [[ -n "$ATTN_IMPL"   ]] && ATTN_ARG=(--attn-impl "$ATTN_IMPL")
PROMPT2_ARG=(); [[ -n "$PROMPT2_TEXT" ]] && { PROMPT2_ARG=(--prompt2 "$PROMPT2_TEXT"); echo "TWO-TURN turn2=\"$PROMPT2_TEXT\""; }

NUM_SHARDS=${#GPUS[@]}
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
        "${DTYPE_ARG[@]}" "${ATTN_ARG[@]}" "${PROMPT2_ARG[@]}" \
        > "${LOG_DIR}/${TASK}_${DATASET_NAME}_shard${i}.log" 2>&1 &
    pids+=($!)
done

echo "PIDs: ${pids[*]}   (logs: ${LOG_DIR}/${TASK}_${DATASET_NAME}_shard*.log)"
rc=0
for pid in "${pids[@]}"; do wait "$pid" || rc=$?; done
echo "All shards finished (exit $rc). Results under: $OUTPUT_DIR"
exit $rc
