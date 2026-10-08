#!/usr/bin/env bash
#SBATCH --job-name=qwen35_ct_30sep_slowfast
#SBATCH --partition=h100
#SBATCH --reservation=vlm          # qrnd4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=96
#SBATCH --mem=512G
#SBATCH --time=14-00:00:00
#SBATCH --output=/cache/fast_data_nas71/janhavi/VLM_qure/vlm/swift/examples/train/liger/logs/%x-%j.out
#SBATCH --error=/cache/fast_data_nas71/janhavi/VLM_qure/vlm/swift/examples/train/liger/logs/%x-%j.err
#
# 30-Sep Qwen3.5-4B CT full-FT with the anti-collapse recipe + slow-fast. Built from the Phase-0 diagnosis:
#   post-proj centered eff_rank collapsed 11.9 (step-0) -> 1.07 (ckpt-1000), a CORRELATION collapse, essentially
#   complete by step 500. So we RESTART FRESH (no resume; ckpt-500/1000 are already collapsed) with:
#     * differential LRs        : pretrained encoder blocks LOW (--vit_lr), new params (patch-embed / projector
#                                 / cross-attn) HIGH (--aligner_lr) via --vision_3d_split_lr_groups.
#     * VICReg rank-reg         : --vision_3d_rankreg_lambda (cov term = mean off-diagonal correlation^2, the
#                                 operative lever against the correlation collapse).
#     * ViT warmup-freeze       : freeze pretrained blocks the first N steps so the adapter learns first.
#     * zero-init projector out : no gradient into the tower through the projector at step 0.
#     * ct_augment OFF          : the aggressive shuffle/blackout aug taught volume-independence (Phase-0).
#   Slow-fast: 512 FAST tokens (avg+max 3D pool) ride <video>; up to 4096 SLOW tokens feed gated cross-attention.
#   Monitoring: post-proj eff_rank/sv0 (collapse detector, no extra forward), per-tag AUC on cls_json val, and
#   TRAIN token accuracy (Liger has no train logits -> computed from hidden+lm_head on logging steps).
#
# NOTE: launch only AFTER the pre-launch smoke passes. Tune --vision_3d_rankreg_lambda from the smoke's
#       loss/rankreg ratio (cov term is in [0,1]; start 1.0). Optional: --eval_cls_auc_tags <ordered tag list>
#       to name the AUC slots instead of slot0..slotK.
set -euo pipefail
set +u
source /cache/fast_data_nas71/janhavi/miniconda3/etc/profile.d/conda.sh
conda activate swift_july
set -u
cd /cache/fast_data_nas71/janhavi/VLM_qure/vlm/swift

export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export CT_DATASET_CHANNELS=1                 # tag each dataset with a `channel` (per-channel loss + presence/AUC gate)
export NPROC_PER_NODE=8
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export USE_HF=1
export MAX_PIXELS=$((3500 * 3500))
export VLLM_FORCE_FA2=1                       # H100 FA3 PTX crash guard
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=7200      # tolerate slow first-time tokenization of 17 datasets
export TORCH_NCCL_ENABLE_MONITORING=1             # keep the monitor, but with the long heartbeat above

CT_WINDOWS="lung,mediastinum,abdomen,liver,bone,brain,subdural,stroke,temporal_bone,soft_tissue"
DS_CONFIG=/cache/fast_data_nas71/janhavi/VLM_qure/vlm/swift/examples/train/liger/ds_zero2_overlap.json

# Output on /mnt/scratch (created if missing).
OUTPUT_DIR=/mnt/scratch/janhavi/vlm_ckpts/30sep_qwen3_5_4b_ct_slowfast
mkdir -p "$OUTPUT_DIR"

# (1) All training jsons from ct_final_data/train.
DATA=/cache/fast_data_nas71/janhavi/data/ct_final_data/train
DATASET_PATHS=(
    "$DATA/30_sep_2026_vlm_ct_tag_extraction_complete_cls_json_training.json"
    "$DATA/30_sep_2026_ct_tag_extraction_cls_tokens_training.json"
    "$DATA/30_sep_2026_ct_cls_plus_mini_report_combined_training.json"
    "$DATA/30_sep_2026_ct_cls_plus_mini_report_multiturn_training.json"
    "$DATA/13_Aug_2026_ct_full_structured_report_training.json"
    "$DATA/13_Aug_2026_ct_mini_structured_report_training.json"
    "$DATA/04_Sep_2026_ct_knowledge_graph_training.json"
    "$DATA/cac_json_train_30sep2026.json"
    "$DATA/nodule_fpr_combined_train_30sep2026.json"
    "$DATA/pe_multiturn_train_29sep2026.json"
    "$DATA/segmed_nodule_bbox_train_04sep2026_clean.json"
    "$DATA/segmed_nodule_qa_train_04sep2026_clean.json"
    "$DATA/segmed_anatomy_grounded_abnormality_train_08sep2026.json"
    "$DATA/omnict_label_bbox_multiturn_train_29sep2026.json"
    "$DATA/omnict_region_findings_qa_train_29sep2026.json"
    "$DATA/omnict_anatomy_localization_train_29sep2026_clean.json"
    "$DATA/report_comparison_train_09sep2026_clean.json"
)

swift sft \
    --model Qwen/Qwen3.5-4B \
    --run_name '30sep_qwen3_5_4b_ct_slowfast' \
    --vision_3d_model Qwen/Qwen3.5-4B \
    --vision_3d_trust_remote_code false \
    --vision_3d_max_tokens 3072 \
    --vision_3d_resampler none \
    --vision_3d_patch_init tile \
    --vision_3d_sta true \
    --vision_3d_sta_window 5 \
    --vision_3d_slowfast true \
    --vision_3d_fast_grid 8,8,8 \
    --vision_3d_fast_pool avgmax \
    --vision_3d_crossattn_every_n 2 \
    --vision_3d_slow_max_tokens 4096 \
    --vision_3d_rankreg_lambda 1.0 \
    --vision_3d_rankreg_var_w 1.0 \
    --vision_3d_rankreg_cov_w 1.0 \
    --vision_3d_rankreg_gamma 1.0 \
    --vision_3d_rankreg_max_tokens 2048 \
    --vision_3d_proj_zero_init true \
    --vision_3d_split_lr_groups true \
    --vision_3d_vit_freeze_steps 300 \
    --presence_ce_w_true 6.0 \
    --presence_ce_w_false 1.0 \
    --presence_ce_channel cls_json \
    --log_collapse_metrics true \
    --log_train_acc true \
    --eval_cls_auc true \
    --template qwen2_5_vl_ct \
    --ct_windows "$CT_WINDOWS" \
    --ct_window_base full_range \
    --ct_augment false \
    --ct_load_dtype bfloat16 \
    --ct_normalize standardize \
    --tuner_type full \
    --dataset "${DATASET_PATHS[@]}" \
    --dataset_shuffle true \
    --load_from_cache_file true \
    --split_dataset_ratio 0.005 \
    --torch_dtype bfloat16 \
    --fp16 false --bf16 true \
    --per_device_train_batch_size 3 \
    --per_device_eval_batch_size 2 \
    --gradient_accumulation_steps 3 \
    --gradient_checkpointing true \
    --gradient_checkpointing_kwargs '{"use_reentrant": false}' \
    --learning_rate 2e-5 \
    --vit_lr 2e-5 \
    --aligner_lr 2e-4 \
    --freeze_llm false \
    --freeze_vit true \
    --freeze_aligner true \
    --trainable_parameters visual_3d \
    --num_train_epochs 1 \
    --max_length 32000 \
    --warmup_ratio 0.05 \
    --eval_steps 600 \
    --save_steps 200 \
    --logging_steps 5 \
    --save_total_limit 6 \
    --save_only_model false \
    --dataloader_num_workers 8 \
    --dataloader_persistent_workers true \
    --dataloader_prefetch_factor 2 \
    --dataloader_pin_memory true \
    --dataset_num_proc 16 \
    --system 'You are a helpful medical assistant.' \
    --deepspeed "$DS_CONFIG" \
    --attn_impl flash_attention_2 \
    --use_hf true \
    --use_liger_kernel true \
    --report_to wandb \
    --group_by_length false \
    --padding_free false \
    --packing false \
    --ddp_find_unused_parameters true \
    --output_dir "$OUTPUT_DIR"
