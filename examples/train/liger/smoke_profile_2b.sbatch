#!/usr/bin/env bash
#SBATCH --job-name=smoke2b
#SBATCH --partition=h100
#SBATCH --exclude=qrnd4                 # qrnd4 is the running 4B training; any OTHER free h100 node
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=96
#SBATCH --mem=512G
#SBATCH --time=02:00:00
#SBATCH --output=/cache/fast_data_nas71/janhavi/VLM_qure/vlm/swift/examples/train/liger/logs/%x-%j.out
#SBATCH --error=/cache/fast_data_nas71/janhavi/VLM_qure/vlm/swift/examples/train/liger/logs/%x-%j.err
#
# Smoke + profile for the aggressive 2B config (batch 6, 12k encoder+slow tokens, NO grad checkpointing,
# group_by_length, max_length 18000). Part A = real 2B config @ max_steps 4 on 8xH100 ZeRO-2 -> faithful
# MEMORY test (does it OOM? -> if so, re-enable gradient_checkpointing / drop batch). Part B = ct_profile.py
# (1 GPU, batch 1) -> WHERE the time goes (encoder vs LLM+cross-attn) + op table. Part A `|| true` so an
# expected OOM still lets the profiler run.
set -uo pipefail
set +u
source /cache/fast_data_nas71/janhavi/miniconda3/etc/profile.d/conda.sh
conda activate swift_july
set -u
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
SMOKE_OUT=/mnt/scratch/janhavi/vlm_ckpts/_smoke_2b
mkdir -p "$SMOKE_OUT"
DATA=/cache/fast_data_nas71/janhavi/data/ct_final_data/train
DATASET_PATHS=(
    "$DATA/30_sep_2026_vlm_ct_tag_extraction_complete_cls_json_training.json"
    "$DATA/cac_json_train_30sep2026.json"
    "$DATA/13_Aug_2026_ct_mini_structured_report_training.json"
    "$DATA/nodule_fpr_combined_train_30sep2026.json"
)

echo "################# PART A: 8xH100 MEMORY SMOKE (max_steps 4) #################"
swift sft \
    --model "$MODEL" --run_name 'smoke_2b' --vision_3d_model "$MODEL" \
    --vision_3d_trust_remote_code false --vision_3d_max_tokens 12000 --vision_3d_resampler none \
    --vision_3d_patch_init tile --vision_3d_sta true --vision_3d_sta_window 5 --vision_3d_slowfast true \
    --vision_3d_fast_grid 8,8,8 --vision_3d_fast_pool avgmax --vision_3d_crossattn_every_n 2 \
    --vision_3d_slow_max_tokens 12000 --vision_3d_rankreg_lambda 1.0 --vision_3d_rankreg_var_w 1.0 \
    --vision_3d_rankreg_cov_w 1.0 --vision_3d_rankreg_gamma 1.0 --vision_3d_rankreg_max_tokens 2048 \
    --vision_3d_proj_zero_init true --vision_3d_split_lr_groups true --vision_3d_vit_freeze_steps 300 \
    --presence_ce_w_true 6.0 --presence_ce_w_false 1.0 --presence_ce_channel cls_json \
    --log_collapse_metrics true --log_train_acc true --eval_cls_auc false \
    --template qwen2_5_vl_ct --ct_windows "$CT_WINDOWS" --ct_window_base full_range --ct_augment false \
    --ct_load_dtype bfloat16 --ct_normalize standardize --tuner_type full \
    --dataset "${DATASET_PATHS[@]}" --dataset_shuffle true --load_from_cache_file true \
    --split_dataset_ratio 0.0 --torch_dtype bfloat16 --fp16 false --bf16 true \
    --per_device_train_batch_size 6 --gradient_accumulation_steps 1 \
    --gradient_checkpointing false --learning_rate 2e-5 --vit_lr 2e-5 --aligner_lr 2e-4 \
    --freeze_llm false --freeze_vit true --freeze_aligner true --trainable_parameters visual_3d \
    --max_steps 4 --max_length 18000 --warmup_ratio 0.0 --logging_steps 1 --save_strategy no \
    --dataloader_num_workers 8 --dataset_num_proc 16 --group_by_length true \
    --system 'You are a helpful medical assistant.' --deepspeed "$DS_CONFIG" \
    --attn_impl flash_attention_2 --use_hf true --use_liger_kernel true --report_to none \
    --padding_free false --packing false --ddp_find_unused_parameters true --output_dir "$SMOKE_OUT" \
    || echo "### PART A exited non-zero (likely OOM -> re-enable gradient_checkpointing / drop batch). Continuing to profiler. ###"

echo "################# PART B: PROFILER (1 GPU, batch 1) #################"
CUDA_VISIBLE_DEVICES=0 python /cache/fast_data_nas71/janhavi/infer_eval/ct_profile.py \
    --model "$MODEL" --max_tokens 12000 --batch 1 --gpus 0 \
    || echo "### PART B profiler failed (if OOM at 12k/batch1, rerun with --max_tokens 6000). ###"
echo "################# SMOKE+PROFILE DONE #################"
