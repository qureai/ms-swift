#!/usr/bin/env bash
#SBATCH --job-name=bench8es
#SBATCH --partition=h100
#SBATCH --reservation=vlm
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=96
#SBATCH --mem=512G
#SBATCH --time=01:30:00
#SBATCH --output=/cache/fast_data_nas71/janhavi/infer_eval/logs/bench8es-%j.out
#SBATCH --error=/cache/fast_data_nas71/janhavi/infer_eval/logs/bench8es-%j.err
# EVAL+SAVE memory test: real 2B config but max_steps 5 with eval_steps 3 + save_steps 3 (save_only_model
# false -> optimizer saved, like the real run) so an eval AND a checkpoint-save both fire -> measures the TRUE
# peak incl eval + ZeRO-2 save. Sweeps bs 6/5/4. "OK (reached step>=4)" => survives eval+save at that batch.
set -uo pipefail; set +u
source /cache/fast_data_nas71/janhavi/miniconda3/etc/profile.d/conda.sh
conda activate swift_july; set -u
cd /cache/fast_data_nas71/janhavi/VLM_qure/vlm/swift
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export CT_DATASET_CHANNELS=1; export NPROC_PER_NODE=8; export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export USE_HF=1; export MAX_PIXELS=$((3500*3500)); export VLLM_FORCE_FA2=1; export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=7200
MODEL=Qwen/Qwen3.5-2B
CT_WINDOWS="lung,mediastinum,abdomen,liver,bone,brain,subdural,stroke,temporal_bone,soft_tissue"
DS_CONFIG=/cache/fast_data_nas71/janhavi/VLM_qure/vlm/swift/examples/train/liger/ds_zero2_overlap.json
FAKE=/mnt/scratch/janhavi/_bench_fake_v2
OUT=/mnt/scratch/janhavi/vlm_ckpts/_bench_evalsave
SUMMARY=/cache/fast_data_nas71/janhavi/infer_eval/logs/bench8es_summary.txt
mkdir -p "$OUT"; : > "$SUMMARY"
python /cache/fast_data_nas71/janhavi/infer_eval/gen_fake_ct_data.py --out "$FAKE" --n 64 --d 128 --h 512 --w 512 --answer_tokens 12000
run_es () {
  local BS=$1 TAG=$2
  local LOG="$OUT/run_${TAG}.log"
  echo "### EVAL+SAVE test bs=$BS ###"
  rm -rf "$OUT/ckpt_$TAG"; mkdir -p "$OUT/ckpt_$TAG"
  timeout 1500 swift sft \
    --model "$MODEL" --run_name "es_$TAG" --vision_3d_model "$MODEL" \
    --vision_3d_trust_remote_code false --vision_3d_max_tokens 12000 --vision_3d_resampler none \
    --vision_3d_patch_init tile --vision_3d_sta true --vision_3d_sta_window 5 --vision_3d_slowfast true \
    --vision_3d_fast_grid 16,8,8 --vision_3d_fast_pool avgmax --vision_3d_crossattn_every_n 2 \
    --vision_3d_slow_max_tokens 12000 --vision_3d_rankreg_lambda 1.0 --vision_3d_rankreg_max_tokens 2048 \
    --vision_3d_rankreg_fast_lambda 1.0 --vision_3d_proj_ortho_lambda 1.0 --vision_3d_proj_fast_wd 1e-5 \
    --vision_3d_proj_fast_init orthogonal --vision_3d_proj_zero_init true --vision_3d_split_lr_groups true \
    --vision_3d_vit_freeze_steps 300 --presence_ce_w_true 6.0 --presence_ce_w_false 1.0 --presence_ce_channel cls_json \
    --log_collapse_metrics true --log_train_acc true --eval_cls_auc true \
    --template qwen2_5_vl_ct --ct_windows "$CT_WINDOWS" --ct_window_base full_range --ct_augment false \
    --ct_load_dtype bfloat16 --ct_normalize standardize --tuner_type full \
    --dataset "$FAKE/fake_train.jsonl" --dataset_shuffle true --load_from_cache_file false \
    --split_dataset_ratio 0.05 --torch_dtype bfloat16 --fp16 false --bf16 true \
    --per_device_train_batch_size "$BS" --per_device_eval_batch_size 2 --gradient_accumulation_steps 1 \
    --gradient_checkpointing true --gradient_checkpointing_kwargs '{"use_reentrant": false}' \
    --learning_rate 2e-5 --vit_lr 2e-5 --aligner_lr 2e-4 --freeze_llm false --freeze_vit true \
    --freeze_aligner true --trainable_parameters visual_3d --max_steps 5 --max_length 18000 --warmup_ratio 0.0 \
    --eval_strategy steps --eval_steps 3 --save_strategy steps --save_steps 3 --save_only_model false \
    --save_total_limit 1 --logging_steps 1 --dataloader_num_workers 8 --dataset_num_proc 16 --group_by_length false \
    --system 'You are a helpful medical assistant.' --deepspeed "$DS_CONFIG" --attn_impl flash_attention_2 \
    --use_hf true --use_liger_kernel true --report_to none --padding_free false --packing false \
    --ddp_find_unused_parameters true --output_dir "$OUT/ckpt_$TAG" > "$LOG" 2>&1
  local rc=$? mem gstep saved
  mem=$(grep -oE "memory\(GiB\)': '[0-9.]+'" "$LOG" | tail -1 | grep -oE "[0-9.]+")
  gstep=$(grep -oE "global_step/max_steps': '[0-9]+" "$LOG" | tail -1 | grep -oE "[0-9]+$")
  saved=$(ls -d "$OUT/ckpt_$TAG"/*/checkpoint-* 2>/dev/null | head -1)
  if grep -qiE "out of memory|OutOfMemory" "$LOG"; then
    echo "RESULT bs=$BS status=OOM mem=NA gstep=${gstep:-0} saved=no" | tee -a "$SUMMARY"
  elif [ "${gstep:-0}" -ge 4 ] && [ -n "$saved" ]; then
    echo "RESULT bs=$BS status=OK(survived eval+save) mem=${mem:-NA} gstep=${gstep:-0} saved=yes" | tee -a "$SUMMARY"
  else
    echo "RESULT bs=$BS status=ERR(rc=$rc) mem=${mem:-NA} gstep=${gstep:-0} saved=${saved:+yes}" | tee -a "$SUMMARY"
  fi
}
for BS in 6 5 4; do run_es "$BS" "bs$BS"; done
echo "===== EVAL+SAVE SUMMARY ====="; cat "$SUMMARY"
echo "BENCH EVALSAVE DONE"
