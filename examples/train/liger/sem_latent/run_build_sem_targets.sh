#!/usr/bin/env bash
# SEM LATENT (stage 7): build the offline TEACHER SIDECAR that the training plugin reads via
# SEM_TARGETS_DIR (swift_plugins/sem_latent_plugin.py). Wraps build_sem_targets.py (header there
# documents the teacher recipe, resumability and the sidecar layout; the target rules live in
# swift_plugins/sem_targets.py).
#
#   1. COLLECT  once, on CPU (multiprocessing over byte-range chunks; seconds when the files are in
#               the page cache): unique canonical report texts of the mixture -> $OUT/collect/
#   2. ENCODE   one shard per GPU listed in GPUS, each launched `setsid nohup ... & disown` into
#               $LOG_DIR/shard_<i>.log, so the shards survive this shell / the Claude session
#   3. MERGE    once every shard's parts/part_<i>.npz exists -> sidecar files at $OUT/ (meta.json last)
#
# RE-RUNNING IS SAFE: the collection is cached (invalidated when a dataset file's size/mtime, the
# rule version or LIMIT changes), finished shards are skipped, a killed shard resumes from its chunk
# files, and a shard that is STILL RUNNING is waited for (per-part flock) instead of duplicated.
# The wait loop itself is only a convenience: if this shell dies, re-run the script (or just
# `build_sem_targets.py ... --merge`) once the shard logs say "wrote".
#
# Detach the whole thing (recommended for the full build):
#   setsid nohup bash examples/train/liger/sem_latent/run_build_sem_targets.sh > /tmp/sem_build.log 2>&1 < /dev/null & disown
#
# ENV
#   MIX           mcr (default) | 6c   picks the defaults of OUT / DATASETS / VAL_DATASETS, exactly as the stage-7
#                                       launcher's MIX does (so each MIX's default sidecar covers its default files)
#   OUT           sidecar dir          (MIX=mcr: $DATA_DIR/sem_targets/stage7_mcr_rule_v1;
#                                       MIX=6c:  $DATA_DIR/sem_targets/stage7_rule_v1 -- the existing 6c sidecar;
#                                       a complete sidecar is never overwritten, see below)
#   DATASETS      train files          (MIX=mcr: $DATA_DIR/stage7_mcr/{A_TRAIN_120k,B3_TRAIN,15_stage5_headguided_TRAIN,
#                                       17_refresher_multiturn_TRAIN,RET_diverse60k_TRAIN}.mcr.jsonl + REX_TRAIN.jsonl,
#                                       built by build_stage7_mcr_data.py; MIX=6c: the stage-6c mixture A_TRAIN_120k
#                                       B3_TRAIN 15_stage5_headguided 17_refresher RET_diverse60k)
#   VAL_DATASETS  val files            (MIX=mcr: A_VAL.mcr B3_VAL.mcr REX_VAL; MIX=6c: A_VAL B3_VAL). Keys seen in
#                                       DATASETS get in_train=1.
#   DEVICE        cuda (default) | cpu
#   GPUS          "0 1 2 3 4 5 6 7"    one shard per entry. With DEVICE=cuda each entry is that shard's
#                                       CUDA_VISIBLE_DEVICES. With DEVICE=cpu: unset -> ONE foreground CPU
#                                       shard (smoke); set -> that many detached CPU shards (tests the
#                                       orchestration without GPUs).
#   DTYPE         bf16 (cuda default) | fp32 (cpu default)
#   MAX_LENGTH    1024   BATCH_TOKENS 32768 (cpu 4096)   MAX_BATCH 256   CHUNK_TEXTS 4096
#   LIMIT         cap on unique texts (smoke; family-balanced, split over files)
#   WRITE_TEXTS   1 (default) -> texts.jsonl in the sidecar (eval dumps / debugging; ~350 MB full)
#   TEACHER, PROBE_HEAD, HEADS_JSON    override the teacher / probe / fingerprint reference
#   WORKERS       collect processes (default 96)      THREADS  torch threads per CPU shard (default 64)
#   LOG_DIR       default $OUT/logs     POLL  seconds between shard checks (default 30)
#   PYTHON        interpreter (default $CONDA_ENV/bin/python)
set -euo pipefail

# repo root = four levels up from this script (examples/train/liger/sem_latent/); override with SWIFT_ROOT=...
SWIFT_ROOT="${SWIFT_ROOT:-$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/../../../.." && pwd)}"
CONDA_ENV="${CONDA_ENV:-/home/ubuntu/miniconda3/envs/ms-swift-may-s4}"
PYTHON="${PYTHON:-$CONDA_ENV/bin/python}"
export PYTHONPATH="$SWIFT_ROOT:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false
export HF_HOME="${HF_HOME:-/fastdata4/sahil_work/VLM_qure/cache_stage6/hf}"   # keep caches off the root disk
BUILDER="$SWIFT_ROOT/examples/train/liger/sem_latent/build_sem_targets.py"

DATA_DIR="/fastdata4/sahil_work/VLM_qure/data_preparation/training_data"
MIX="${MIX:-mcr}"
MCR_DIR="$DATA_DIR/stage7_mcr"
case "$MIX" in
    mcr)
        DEFAULT_OUT="$DATA_DIR/sem_targets/stage7_mcr_rule_v1"
        DEFAULT_TRAIN="$MCR_DIR/A_TRAIN_120k.mcr.jsonl $MCR_DIR/B3_TRAIN.mcr.jsonl $MCR_DIR/15_stage5_headguided_TRAIN.mcr.jsonl $MCR_DIR/17_refresher_multiturn_TRAIN.mcr.jsonl $MCR_DIR/RET_diverse60k_TRAIN.mcr.jsonl $MCR_DIR/REX_TRAIN.jsonl"
        DEFAULT_VAL="$MCR_DIR/A_VAL.mcr.jsonl $MCR_DIR/B3_VAL.mcr.jsonl $MCR_DIR/REX_VAL.jsonl" ;;
    6c)
        DEFAULT_OUT="$DATA_DIR/sem_targets/stage7_rule_v1"
        DEFAULT_TRAIN="$DATA_DIR/stage6_toolcalling/A_TRAIN_120k.jsonl $DATA_DIR/stage6_toolcalling/B3_TRAIN.jsonl $DATA_DIR/15_stage5_headguided_TRAIN.json $DATA_DIR/17_refresher_multiturn_TRAIN.json $DATA_DIR/stage6_toolcalling/RET_diverse60k_TRAIN.jsonl"
        DEFAULT_VAL="$DATA_DIR/stage6_toolcalling/A_VAL.jsonl $DATA_DIR/stage6_toolcalling/B3_VAL.jsonl" ;;
    *) echo "[sem-build] MIX must be mcr|6c, got '$MIX'" >&2; exit 1 ;;
esac
OUT="${OUT:-$DEFAULT_OUT}"
read -r -a TRAIN_FILES <<< "${DATASETS:-$DEFAULT_TRAIN}"
read -r -a VAL_FILES <<< "${VAL_DATASETS:-$DEFAULT_VAL}"
DEVICE="${DEVICE:-cuda}"
if [[ "$DEVICE" == cpu ]]; then DTYPE="${DTYPE:-fp32}"; BATCH_TOKENS="${BATCH_TOKENS:-4096}"; else DTYPE="${DTYPE:-bf16}"; BATCH_TOKENS="${BATCH_TOKENS:-32768}"; fi
LOG_DIR="${LOG_DIR:-$OUT/logs}"
POLL="${POLL:-30}"
mkdir -p "$OUT" "$LOG_DIR"

for f in "${TRAIN_FILES[@]}" "${VAL_FILES[@]}"; do
    [[ -f "${f%#*}" ]] || { echo "[sem-build] dataset missing: $f" >&2; exit 1; }
done

COMMON=(--train_datasets "${TRAIN_FILES[@]}" --out "$OUT" --max_length "${MAX_LENGTH:-1024}")
[[ ${#VAL_FILES[@]} -gt 0 ]] && COMMON+=(--val_datasets "${VAL_FILES[@]}")
[[ -n "${LIMIT:-}" ]] && COMMON+=(--limit "$LIMIT")
[[ -n "${TEACHER:-}" ]] && COMMON+=(--teacher "$TEACHER")
[[ -n "${PROBE_HEAD:-}" ]] && COMMON+=(--probe_head "$PROBE_HEAD")
[[ -n "${HEADS_JSON:-}" ]] && COMMON+=(--heads_json "$HEADS_JSON")
ENC=(--device "$DEVICE" --dtype "$DTYPE" --batch_tokens "$BATCH_TOKENS" --max_batch "${MAX_BATCH:-256}"
     --chunk_texts "${CHUNK_TEXTS:-4096}" --threads "${THREADS:-64}")
MERGE=(--merge)
[[ "${WRITE_TEXTS:-1}" == 1 ]] && MERGE+=(--write_texts)

if [[ -f "$OUT/meta.json" ]]; then
    echo "[sem-build] sidecar already complete: $OUT/meta.json (delete it, or run the builder with --merge --overwrite)" >&2
    exit 0
fi

# ---- 1. collect (foreground, once) ------------------------------------------------------------
echo "[sem-build] collect -> $OUT/collect  (train ${#TRAIN_FILES[@]} files, val ${#VAL_FILES[@]} files)" >&2
CUDA_VISIBLE_DEVICES="" "$PYTHON" "$BUILDER" "${COMMON[@]}" --collect_only --workers "${WORKERS:-96}" \
    > "$LOG_DIR/collect.log" 2>&1 || { tail -30 "$LOG_DIR/collect.log" >&2; exit 1; }
grep -E '"n_unique"|"n_unique_in_train"' "$LOG_DIR/collect.log" >&2 || true

# ---- 2. encode ----------------------------------------------------------------------------------
if [[ "$DEVICE" == cpu && -z "${GPUS:-}" ]]; then
    echo "[sem-build] encode: 1 foreground CPU shard ($DTYPE)" >&2
    CUDA_VISIBLE_DEVICES="" "$PYTHON" "$BUILDER" "${COMMON[@]}" "${ENC[@]}" --num_shards 1 --shard_id 0 \
        > "$LOG_DIR/shard_0.log" 2>&1 || { tail -30 "$LOG_DIR/shard_0.log" >&2; exit 1; }
    NSH=1
else
    read -r -a GPU_LIST <<< "${GPUS:-0 1 2 3 4 5 6 7}"
    NSH=${#GPU_LIST[@]}
    PIDS=()
    for i in "${!GPU_LIST[@]}"; do
        if [[ -f "$OUT/parts/part_$i.npz" && -f "$OUT/parts/part_$i.json" ]] && \
           "$PYTHON" -c "import json,sys; sys.exit(0 if json.load(open('$OUT/parts/part_$i.json'))['num_shards']==$NSH else 1)"; then
            echo "[sem-build] shard $i/$NSH already done" >&2; PIDS+=(0); continue
        fi
        if [[ "$DEVICE" == cpu ]]; then CVD=""; else CVD="${GPU_LIST[$i]}"; fi
        CUDA_VISIBLE_DEVICES="$CVD" setsid nohup "$PYTHON" "$BUILDER" "${COMMON[@]}" "${ENC[@]}" \
            --num_shards "$NSH" --shard_id "$i" > "$LOG_DIR/shard_$i.log" 2>&1 < /dev/null &
        PIDS+=($!)
        disown
        echo "[sem-build] shard $i/$NSH launched (pid ${PIDS[-1]}, CUDA_VISIBLE_DEVICES='$CVD', log $LOG_DIR/shard_$i.log)" >&2
    done
    # wait: a shard is done when its part exists; a dead pid without its part is a failure
    while :; do
        done_n=0; failed=()
        for i in "${!GPU_LIST[@]}"; do
            if [[ -f "$OUT/parts/part_$i.npz" && -f "$OUT/parts/part_$i.json" ]]; then done_n=$((done_n + 1)); continue; fi
            pid=${PIDS[$i]}
            if [[ "$pid" == 0 ]] || ! kill -0 "$pid" 2>/dev/null; then failed+=("$i"); fi
        done
        if [[ ${#failed[@]} -gt 0 ]]; then
            sleep 2   # a shard that just finished may still be renaming its part
            still=()
            for i in "${failed[@]}"; do [[ -f "$OUT/parts/part_$i.npz" && -f "$OUT/parts/part_$i.json" ]] || still+=("$i"); done
            if [[ ${#still[@]} -gt 0 ]]; then
                for i in "${still[@]}"; do echo "[sem-build] shard $i FAILED -- tail of $LOG_DIR/shard_$i.log:" >&2; tail -20 "$LOG_DIR/shard_$i.log" >&2; done
                echo "[sem-build] fix the error and re-run this script: finished shards/chunks are reused" >&2
                exit 1
            fi
            continue
        fi
        [[ $done_n -eq $NSH ]] && break
        echo "[sem-build] $(date +%H:%M:%S) shards done $done_n/$NSH" >&2
        sleep "$POLL"
    done
fi

# ---- 3. merge -----------------------------------------------------------------------------------
CUDA_VISIBLE_DEVICES="" "$PYTHON" "$BUILDER" "${COMMON[@]}" --num_shards "$NSH" "${MERGE[@]}" \
    > "$LOG_DIR/merge.log" 2>&1 || { tail -30 "$LOG_DIR/merge.log" >&2; exit 1; }
tail -2 "$LOG_DIR/merge.log" >&2
echo "[sem-build] done: SEM_TARGETS_DIR=$OUT" >&2
