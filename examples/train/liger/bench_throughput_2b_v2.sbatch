#!/usr/bin/env bash
#SBATCH --job-name=bench2bv2
#SBATCH --partition=h100
#SBATCH --exclude=qrnd4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=24
#SBATCH --mem=220G
#SBATCH --time=03:00:00
#SBATCH --output=/cache/fast_data_nas71/janhavi/infer_eval/logs/bench2bv2-%j.out
#SBATCH --error=/cache/fast_data_nas71/janhavi/infer_eval/logs/bench2bv2-%j.err
#
# IMPROVED 2xH100 throughput/max-batch benchmark. vs v1, this captures the real-training memory peak:
#   * fixed-at-MAX-resolution volumes (128x512x512 -> smart-resize caps at vision_3d_max_tokens=12000):
#     worst-case per-volume, so the chosen batch survives any real variable-resolution batch composition.
#   * NEAR-MAX-LENGTH sequences (~16k-token fake answers) toward max_length 18000: worst-case LLM activation.
#   * MONITORING ON (--log_train_acc true materialises logits on logging steps = the main periodic spike;
#     --log_collapse_metrics true) -- these were OFF in v1 and are ON in the real config.
# NOTE: 2 GPUs is a LOWER BOUND on the real 8-GPU max batch (ZeRO-2 shards optimizer+grads /8 vs /2 ->
# ~9-12GB more headroom/GPU on 8). The 8-GPU sweep (bench_8gpu_sweep.sbatch) is the authoritative one.
set -uo pipefail; set +u
source /cache/fast_data_nas71/janhavi/miniconda3/etc/profile.d/conda.sh
conda activate swift_july; set -u
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
FAKE=/mnt/scratch/janhavi/_bench_fake_v2
OUT=/mnt/scratch/janhavi/vlm_ckpts/_bench_v2
SUMMARY=/cache/fast_data_nas71/janhavi/infer_eval/logs/bench2bv2_summary.txt
mkdir -p "$OUT"; : > "$SUMMARY"

echo "### generating WORST-CASE fake data (max-size volumes + ~16k-token answers) ###"
python /cache/fast_data_nas71/janhavi/infer_eval/gen_fake_ct_data.py --out "$FAKE" --n 32 \
  --d 128 --h 512 --w 512 --answer_tokens 12000

NGPU=2
run_bench () {   # $1=batch $2=ckpt $3=max_steps $4=tag [$5=timeout_s]
  local BS=$1 CKPT=$2 STEPS=$3 TAG=$4
  local LOG="$OUT/run_${TAG}.log"
  echo "### RUN tag=$TAG bs=$BS ckpt=$CKPT steps=$STEPS ###"
  rm -rf "$OUT/ckpt_$TAG"; mkdir -p "$OUT/ckpt_$TAG"
  timeout "${5:-1200}" swift sft \
    --model "$MODEL" --run_name "benchv2_$TAG" --vision_3d_model "$MODEL" \
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
    --dataloader_num_workers 8 --dataset_num_proc 8 --group_by_length false \
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

echo "################# BATCH SWEEP (worst-case inputs, monitoring ON, max_steps 6) #################"
run_bench 1 false 6 "nock_bs1"                 # confirm no-ckpt feasibility at bs1
for BS in 8 6 4 3 2; do run_bench "$BS" true 6 "ck_bs$BS"; done

echo "################# SWEEP SUMMARY (by throughput) #################"
sort -t= -k7 -n -r "$SUMMARY" | tee "$SUMMARY.sorted"
BEST=$(grep "status=OK" "$SUMMARY" | sort -t= -k7 -n -r | head -1)
echo "### BEST (2-GPU, worst-case): $BEST ###"
BBS=$(echo "$BEST" | grep -oE "bs=[0-9]+" | grep -oE "[0-9]+"); BCK=$(echo "$BEST" | grep -oE "ckpt=(true|false)" | cut -d= -f2)
if [ -n "${BBS:-}" ]; then
  echo "################# 1-HOUR RUN at bs=$BBS ckpt=$BCK #################"
  run_bench "$BBS" "$BCK" 100000 "final_1hr" 3300
  G=$(grep "tag=final_1hr" "$SUMMARY" | grep -oE "gstep=[0-9]+" | cut -d= -f2)
  echo "FINAL: best_batch=$BBS ckpt=$BCK steps_~55min=$G images=$(python3 -c "print(${G:-0}*$BBS*$NGPU)")  (2 GPUs, worst-case inputs)"
  echo "(8-GPU: throughput ~x4; max batch LARGER due to ZeRO-2 /8 optimizer sharding -- see bench_8gpu_sweep.sbatch)"
else
  echo "### No config fit even at bs1+ckpt -> real memory peak exceeds 80GB at 12k tokens on 2 GPUs. ###"
fi
echo "################# BENCH v2 DONE #################"
