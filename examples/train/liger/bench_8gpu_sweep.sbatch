#!/usr/bin/env bash
#SBATCH --job-name=bench8b
#SBATCH --partition=h100
#SBATCH --reservation=vlm          # qrnd4 (freed by killing the 4B run)
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=96
#SBATCH --mem=512G
#SBATCH --time=03:30:00
#SBATCH --output=/cache/fast_data_nas71/janhavi/infer_eval/logs/bench8b-%j.out
#SBATCH --error=/cache/fast_data_nas71/janhavi/infer_eval/logs/bench8b-%j.err
#
# AUTHORITATIVE 8xH100 batch sweep (= the real-run parallelism). Same worst-case inputs as the improved
# 2-GPU bench (max-resolution volumes + ~16k-token sequences + monitoring ON), but on 8 GPUs where ZeRO-2
# shards optimizer+grads /8 -> ~9-12GB more headroom/GPU than the 2-GPU run, so the real max batch shows
# here. Sweeps ckpt-off (may now fit) and ckpt-on, then a ~55min run at the best config for true images/hr.
# This REPLACES the single-batch smoke (which only tested bs6).
set -uo pipefail; set +u
source /cache/fast_data_nas71/janhavi/miniconda3/etc/profile.d/conda.sh
conda activate swift_july; set -u
cd /cache/fast_data_nas71/janhavi/VLM_qure/vlm/swift

export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export CT_DATASET_CHANNELS=1
export NPROC_PER_NODE=8
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export USE_HF=1
export MAX_PIXELS=$((3500 * 3500))
export VLLM_FORCE_FA2=1
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=7200

MODEL=Qwen/Qwen3.5-2B
CT_WINDOWS="lung,mediastinum,abdomen,liver,bone,brain,subdural,stroke,temporal_bone,soft_tissue"
DS_CONFIG=/cache/fast_data_nas71/janhavi/VLM_qure/vlm/swift/examples/train/liger/ds_zero2_overlap.json
FAKE=/mnt/scratch/janhavi/_bench_fake_v2
OUT=/mnt/scratch/janhavi/vlm_ckpts/_bench_8g
SUMMARY=/cache/fast_data_nas71/janhavi/infer_eval/logs/bench8b_summary.txt
mkdir -p "$OUT"; : > "$SUMMARY"

echo "### generating WORST-CASE fake data (max-size volumes + ~16k-token answers) ###"
python /cache/fast_data_nas71/janhavi/infer_eval/gen_fake_ct_data.py --out "$FAKE" --n 64 \
  --d 128 --h 512 --w 512 --answer_tokens 12000

NGPU=8
run_bench () {   # $1=batch $2=ckpt $3=max_steps $4=tag [$5=timeout_s]
  local BS=$1 CKPT=$2 STEPS=$3 TAG=$4
  local LOG="$OUT/run_${TAG}.log"
  echo "### RUN tag=$TAG bs=$BS ckpt=$CKPT steps=$STEPS ###"
  rm -rf "$OUT/ckpt_$TAG"; mkdir -p "$OUT/ckpt_$TAG"
  timeout "${5:-1200}" swift sft \
    --model "$MODEL" --run_name "bench8_$TAG" --vision_3d_model "$MODEL" \
    --vision_3d_trust_remote_code false --vision_3d_max_tokens 12000 --vision_3d_resampler none \
    --vision_3d_patch_init tile --vision_3d_sta true --vision_3d_sta_window 5 --vision_3d_slowfast true \
    --vision_3d_fast_grid 16,8,8 --vision_3d_fast_pool avgmax --vision_3d_crossattn_every_n 2 \
    --vision_3d_slow_max_tokens 12000 --vision_3d_rankreg_lambda 1.0 --vision_3d_rankreg_max_tokens 2048 --vision_3d_rankreg_fast_lambda 1.0 --vision_3d_proj_ortho_lambda 1.0 --vision_3d_proj_fast_wd 1e-5 --vision_3d_proj_fast_init orthogonal \
    --vision_3d_proj_zero_init true --vision_3d_split_lr_groups true --vision_3d_vit_freeze_steps 300 \
    --presence_ce_w_true 6.0 --presence_ce_w_false 1.0 --presence_ce_channel cls_json \
    --log_collapse_metrics true --log_train_acc true --eval_cls_auc false \
    --template qwen2_5_vl_ct --ct_windows "$CT_WINDOWS" --ct_window_base full_range --ct_augment false \
    --ct_load_dtype bfloat16 --ct_normalize standardize --tuner_type full \
    --dataset "$FAKE/fake_train.jsonl" --dataset_shuffle true --load_from_cache_file false \
    --split_dataset_ratio 0.0 --torch_dtype bfloat16 --fp16 false --bf16 true \
    --per_device_train_batch_size "$BS" --gradient_accumulation_steps 1 \
    --gradient_checkpointing "$CKPT" --gradient_checkpointing_kwargs '{"use_reentrant": false}' \
    --learning_rate 2e-5 --vit_lr 2e-5 --aligner_lr 2e-4 \
    --freeze_llm false --freeze_vit true --freeze_aligner true --trainable_parameters visual_3d \
    --max_steps "$STEPS" --max_length 18000 --warmup_ratio 0.0 --logging_steps 1 --save_strategy no \
    --dataloader_num_workers 8 --dataset_num_proc 16 --group_by_length false \
    --system 'You are a helpful medical assistant.' --deepspeed "$DS_CONFIG" \
    --attn_impl flash_attention_2 --use_hf true --use_liger_kernel true --report_to none \
    --padding_free false --packing false --ddp_find_unused_parameters true --output_dir "$OUT/ckpt_$TAG" \
    > "$LOG" 2>&1
  local rc=$? sit mem gstep imgps
  sit=$(grep -oE "train_speed\(s/it\)': '[0-9.]+'" "$LOG" | tail -1 | grep -oE "[0-9.]+")
  mem=$(grep -oE "memory\(GiB\)': '[0-9.]+'" "$LOG" | tail -1 | grep -oE "[0-9.]+")
  gstep=$(grep -oE "global_step/max_steps': '[0-9]+" "$LOG" | tail -1 | grep -oE "[0-9]+$")
  if [ "$rc" -ne 0 ] && grep -qiE "out of memory|OutOfMemory" "$LOG"; then
    echo "RESULT tag=$TAG bs=$BS ckpt=$CKPT status=OOM sit=NA mem=NA imgps=0 gstep=${gstep:-0}" | tee -a "$SUMMARY"
  elif [ -z "$sit" ]; then
    echo "RESULT tag=$TAG bs=$BS ckpt=$CKPT status=ERR(rc=$rc) sit=NA mem=NA imgps=0 gstep=${gstep:-0}" | tee -a "$SUMMARY"
  else
    imgps=$(python3 -c "print(f'{$BS*$NGPU/float($sit):.3f}')")
    echo "RESULT tag=$TAG bs=$BS ckpt=$CKPT status=OK sit=$sit mem=$mem imgps=$imgps gstep=${gstep:-0}" | tee -a "$SUMMARY"
  fi
}

echo "################# 8-GPU BATCH SWEEP (worst-case inputs, monitoring ON) #################"
# ckpt OFF: 8-GPU ZeRO-2 frees ~9-12GB/GPU vs 2-GPU, so small batches MAY now fit (2-GPU OOM'd even bs1)
for BS in 2 1; do run_bench "$BS" false 6 "nock_bs$BS"; done
# ckpt ON: expect a larger max batch than the 2-GPU result (bs6)
for BS in 16 12 10 8 6 4; do run_bench "$BS" true 6 "ck_bs$BS"; done

echo "################# SWEEP SUMMARY (by throughput) #################"
sort -t= -k7 -n -r "$SUMMARY" | tee "$SUMMARY.sorted"
BEST=$(grep "status=OK" "$SUMMARY" | sort -t= -k7 -n -r | head -1)
echo "### BEST (8-GPU, worst-case): $BEST ###"
BBS=$(echo "$BEST" | grep -oE "bs=[0-9]+" | grep -oE "[0-9]+"); BCK=$(echo "$BEST" | grep -oE "ckpt=(true|false)" | cut -d= -f2)
if [ -n "${BBS:-}" ]; then
  echo "################# 1-HOUR RUN at bs=$BBS ckpt=$BCK (8 GPUs) #################"
  run_bench "$BBS" "$BCK" 100000 "final_1hr" 3300
  G=$(grep "tag=final_1hr" "$SUMMARY" | grep -oE "gstep=[0-9]+" | cut -d= -f2)
  echo "FINAL(8GPU): best_batch=$BBS ckpt=$BCK steps_~55min=$G images=$(python3 -c "print(${G:-0}*$BBS*$NGPU)")"
else
  echo "### No config fit -> 12k tokens exceeds 80GB even on 8-GPU ZeRO-2. Would need ZeRO-3 or fewer tokens. ###"
fi
echo "################# BENCH 8GPU DONE #################"
