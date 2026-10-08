#!/usr/bin/env bash
#SBATCH --job-name=bench2b
#SBATCH --partition=h100
#SBATCH --exclude=qrnd4                 # avoid the running 4B training node; any 2 free H100 GPUs elsewhere
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=24
#SBATCH --mem=200G
#SBATCH --time=03:00:00
#SBATCH --output=/cache/fast_data_nas71/janhavi/infer_eval/logs/bench2b-%j.out
#SBATCH --error=/cache/fast_data_nas71/janhavi/infer_eval/logs/bench2b-%j.err
#
# Throughput + max-batch benchmark for the 2B slow-fast CT model on 2xH100 (ZeRO-2), FAKE np.random volumes
# (real size; page-cached -> model-throughput, not data IO). Sweeps batch over checkpoint on/off, finds the
# max that fits + its throughput, then runs the best config ~55 min and reports total images.
set -uo pipefail
set +u
source /cache/fast_data_nas71/janhavi/miniconda3/etc/profile.d/conda.sh
conda activate swift_july
set -u
cd /cache/fast_data_nas71/janhavi/VLM_qure/vlm/swift

export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export CT_DATASET_CHANNELS=1
export NPROC_PER_NODE=2
export CUDA_VISIBLE_DEVICES=0,1
export USE_HF=1
export MAX_PIXELS=$((3500 * 3500))
export VLLM_FORCE_FA2=1
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=7200

MODEL=Qwen/Qwen3.5-2B
CT_WINDOWS="lung,mediastinum,abdomen,liver,bone,brain,subdural,stroke,temporal_bone,soft_tissue"
DS_CONFIG=/cache/fast_data_nas71/janhavi/VLM_qure/vlm/swift/examples/train/liger/ds_zero2_overlap.json
FAKE=/mnt/scratch/janhavi/_bench_fake
OUT=/mnt/scratch/janhavi/vlm_ckpts/_bench
SUMMARY=/cache/fast_data_nas71/janhavi/infer_eval/logs/bench2b_summary.txt
mkdir -p "$OUT"; : > "$SUMMARY"

echo "### generating fake volumes ###"
python /cache/fast_data_nas71/janhavi/infer_eval/gen_fake_ct_data.py --out "$FAKE" --n 16 --answer_tokens 8000

# one benchmark run: $1=batch $2=ckpt(true/false) $3=max_steps $4=tag ; appends a RESULT line to $SUMMARY
run_bench () {
  local BS=$1 CKPT=$2 STEPS=$3 TAG=$4
  local LOG="$OUT/run_${TAG}.log"
  echo "### RUN tag=$TAG bs=$BS ckpt=$CKPT steps=$STEPS ###"
  rm -rf "$OUT/ckpt_$TAG"; mkdir -p "$OUT/ckpt_$TAG"
  timeout "${5:-900}" swift sft \
    --model "$MODEL" --run_name "bench_$TAG" --vision_3d_model "$MODEL" \
    --vision_3d_trust_remote_code false --vision_3d_max_tokens 12000 --vision_3d_resampler none \
    --vision_3d_patch_init tile --vision_3d_sta true --vision_3d_sta_window 5 --vision_3d_slowfast true \
    --vision_3d_fast_grid 8,8,8 --vision_3d_fast_pool avgmax --vision_3d_crossattn_every_n 2 \
    --vision_3d_slow_max_tokens 12000 --vision_3d_rankreg_lambda 1.0 --vision_3d_rankreg_max_tokens 2048 \
    --vision_3d_proj_zero_init true --vision_3d_split_lr_groups true --vision_3d_vit_freeze_steps 300 \
    --presence_ce_w_true 6.0 --presence_ce_w_false 1.0 --presence_ce_channel cls_json \
    --log_collapse_metrics false --log_train_acc false --eval_cls_auc false \
    --template qwen2_5_vl_ct --ct_windows "$CT_WINDOWS" --ct_window_base full_range --ct_augment false \
    --ct_load_dtype bfloat16 --ct_normalize standardize --tuner_type full \
    --dataset "$FAKE/fake_train.jsonl" --dataset_shuffle true --load_from_cache_file false \
    --split_dataset_ratio 0.0 --torch_dtype bfloat16 --fp16 false --bf16 true \
    --per_device_train_batch_size "$BS" --gradient_accumulation_steps 1 \
    --gradient_checkpointing "$CKPT" --gradient_checkpointing_kwargs '{"use_reentrant": false}' \
    --learning_rate 2e-5 --vit_lr 2e-5 --aligner_lr 2e-4 \
    --freeze_llm false --freeze_vit true --freeze_aligner true --trainable_parameters visual_3d \
    --max_steps "$STEPS" --max_length 18000 --warmup_ratio 0.0 --logging_steps 1 --save_strategy no \
    --dataloader_num_workers 8 --dataset_num_proc 8 --group_by_length false \
    --system 'You are a helpful medical assistant.' --deepspeed "$DS_CONFIG" \
    --attn_impl flash_attention_2 --use_hf true --use_liger_kernel true --report_to none \
    --padding_free false --packing false --ddp_find_unused_parameters true --output_dir "$OUT/ckpt_$TAG" \
    > "$LOG" 2>&1
  local rc=$?
  local sit mem gstep
  sit=$(grep -oE "train_speed\(s/it\)': '[0-9.]+'" "$LOG" | tail -1 | grep -oE "[0-9.]+")
  mem=$(grep -oE "memory\(GiB\)': '[0-9.]+'" "$LOG" | tail -1 | grep -oE "[0-9.]+")
  gstep=$(grep -oE "global_step/max_steps': '[0-9]+" "$LOG" | tail -1 | grep -oE "[0-9]+$")
  if [ "$rc" -ne 0 ] && grep -qiE "out of memory|OutOfMemory" "$LOG"; then
    echo "RESULT tag=$TAG bs=$BS ckpt=$CKPT status=OOM sit=NA mem=NA imgps=0 gstep=${gstep:-0}" | tee -a "$SUMMARY"
  elif [ -z "$sit" ]; then
    echo "RESULT tag=$TAG bs=$BS ckpt=$CKPT status=ERR(rc=$rc) sit=NA mem=NA imgps=0 gstep=${gstep:-0}" | tee -a "$SUMMARY"
  else
    local imgps; imgps=$(python3 -c "print(f'{$BS*2/float($sit):.3f}')")
    echo "RESULT tag=$TAG bs=$BS ckpt=$CKPT status=OK sit=$sit mem=$mem imgps=$imgps gstep=${gstep:-0}" | tee -a "$SUMMARY"
  fi
}

echo "################# BATCH SWEEP (max_steps 6 each) #################"
# gradient checkpointing OFF (the aggressive config) -- likely small max batch at 12k tokens
for BS in 4 2 1; do run_bench "$BS" false 6 "nock_bs$BS"; done
# gradient checkpointing ON -- allows larger batch
for BS in 16 12 8 6 4; do run_bench "$BS" true 6 "ck_bs$BS"; done

echo "################# SWEEP SUMMARY (by throughput) #################"
sort -t= -k7 -n -r "$SUMMARY" | tee "$SUMMARY.sorted"
# pick the best OK config (highest imgps)
BEST=$(grep "status=OK" "$SUMMARY" | sort -t= -k7 -n -r | head -1)
echo "### BEST: $BEST ###"
BBS=$(echo "$BEST" | grep -oE "bs=[0-9]+" | grep -oE "[0-9]+")
BCK=$(echo "$BEST" | grep -oE "ckpt=(true|false)" | cut -d= -f2)

if [ -n "${BBS:-}" ]; then
  echo "################# 1-HOUR RUN at bs=$BBS ckpt=$BCK #################"
  # ~55 min wall clock; big max_steps, stopped by timeout. images = gstep * bs * 2 GPUs.
  run_bench "$BBS" "$BCK" 100000 "final_1hr" 3300
  G=$(grep "tag=final_1hr" "$SUMMARY" | grep -oE "gstep=[0-9]+" | cut -d= -f2)
  echo "################# 1-HOUR RESULT #################"
  echo "best_batch=$BBS ckpt=$BCK steps_in_~55min=$G images=$(python3 -c "print(${G:-0}*$BBS*2)")  (2 GPUs)"
  echo "(8-GPU extrapolation ~x4 throughput; ZeRO-2 on 8 GPUs also allows a LARGER max batch than this 2-GPU sweep)"
else
  echo "### No config fit even at batch 1 -> 12k tokens may need gradient_checkpointing + smaller batch. See $SUMMARY ###"
fi
echo "################# BENCH DONE #################"
