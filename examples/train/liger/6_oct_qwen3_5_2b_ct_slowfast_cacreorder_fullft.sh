#!/usr/bin/env bash
#SBATCH --job-name=qwen35_2b_cacreorder
#SBATCH --partition=h100
#SBATCH --reservation=vlm          # qrnd4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=48
#SBATCH --mem=850G
#SBATCH --time=14-00:00:00
#SBATCH --output=/cache/fast_data_nas71/janhavi/VLM_qure/vlm/swift/examples/train/liger/logs/%x-%j.out
#SBATCH --error=/cache/fast_data_nas71/janhavi/VLM_qure/vlm/swift/examples/train/liger/logs/%x-%j.err
#
# 6-Oct cac-REORDER WARM-START run. Continues the 1_oct 2B slow-fast model FROM an existing checkpoint
# (--model = a trained checkpoint dir; host + trained visual_3d weights are overlaid via args.model_dir)
# to fix the coronary-calcium generation collapse: the cac task is swapped to the REORDERED schema
# (cac_json_reordered_*_06oct2026 -> risk_category / presence / bbox BEFORE the exact agatston_total), so
# the model stops defaulting to the modal-0 cascade. WARM-START, NOT a resume: fresh optimizer / scheduler
# / step 0 (resume would assume the identical dataset; we changed the cac file + subsampled the rest).
#
# Data: cls_json (complete), cls_plus_mini_report_multiturn, pe_multiturn and the reordered cac are used
# FULL; every other dataset is subsampled to ~75% via `path#N` (fixed counts from wc -l, 06-Oct).
# vit_freeze_steps 0 (encoder is already trained). NOT launched -- config only. Before launch, set WARM to
# the desired checkpoint (copy it out of the live run dir first -- live checkpoints rotate at save_total_limit 6).
set -euo pipefail
set +u
source /cache/fast_data_nas71/janhavi/miniconda3/etc/profile.d/conda.sh
conda activate swift_july
set -u
cd /cache/fast_data_nas71/janhavi/VLM_qure/vlm/swift

MODEL=Qwen/Qwen3.5-2B                         # base skeleton for --vision_3d_model (the 3D ViT source)
# Warm-start weights: host VLM + trained visual_3d (overlaid from this dir via args.model_dir). Repoint to
# a later checkpoint before launch; copy it out of the live run dir first (those rotate every 6 saves).
WARM=/mnt/scratch/janhavi/vlm_ckpts/1_oct_qwen3_5_2b_ct_slowfast/v1-20261005-122323/checkpoint-4600
# RESUME (not warm-start): continue step count + optimizer + scheduler + RNG from a 6_oct checkpoint.
RESUME=/mnt/scratch/janhavi/vlm_ckpts/6_oct_qwen3_5_2b_ct_slowfast_cacreorder/v3-20261006-214945/checkpoint-100

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
OUTPUT_DIR=/mnt/scratch/janhavi/vlm_ckpts/6_oct_qwen3_5_2b_ct_slowfast_cacreorder
mkdir -p "$OUTPUT_DIR"

# (1) Training jsons from ct_final_data/train.
#     FULL: cls_json (complete), cls_plus_mini_report_multiturn, pe_multiturn, reordered cac.
#     ~75% (via #N, counts from wc -l on 06-Oct): everything else.
DATA=/cache/fast_data_nas71/janhavi/data/ct_final_data/train
DATASET_PATHS=(
    "$DATA/30_sep_2026_vlm_ct_tag_extraction_complete_cls_json_training.json"              # full  (cls_json)
    "$DATA/30_sep_2026_ct_tag_extraction_cls_tokens_training.json#180379"                  # 75% of 240506
    "$DATA/30_sep_2026_ct_cls_plus_mini_report_combined_training.json#62304"               # 75% of 83073
    "$DATA/30_sep_2026_ct_cls_plus_mini_report_multiturn_training.json"                    # full  (mini report multiturn)
    "$DATA/13_Aug_2026_ct_mini_structured_report_training.json#225099"                     # 75% of 300133
    "$DATA/04_Sep_2026_ct_knowledge_graph_training.json#142023"                            # 75% of 189365
    "$DATA/cac_json_reordered_train_06oct2026.json"                                        # full  (REORDERED cac)
    "$DATA/nodule_fpr_combined_train_30sep2026.json#268518"                                # 75% of 358025
    "$DATA/pe_multiturn_train_29sep2026.json"                                              # full  (pe)
    "$DATA/segmed_nodule_bbox_train_04sep2026_clean.json#204530"                           # 75% of 272707
    "$DATA/segmed_nodule_qa_train_04sep2026_clean.json#204530"                             # 75% of 272707
    "$DATA/segmed_anatomy_grounded_abnormality_train_08sep2026.json#47633"                 # 75% of 63511
    "$DATA/omnict_label_bbox_multiturn_train_29sep2026.json#140630"                        # 75% of 187507
    "$DATA/omnict_region_findings_qa_train_29sep2026.json#140630"                          # 75% of 187507
    "$DATA/omnict_anatomy_localization_train_29sep2026_clean.json#140607"                  # 75% of 187476
    "$DATA/report_comparison_train_09sep2026_clean.json#110097"                            # 75% of 146796
)

swift sft \
    --model "$MODEL" \
    --resume_from_checkpoint "$RESUME" \
    --run_name '6_oct_qwen3_5_2b_ct_slowfast_cacreorder' \
    --vision_3d_model "$MODEL" \
    --vision_3d_trust_remote_code false \
    --vision_3d_max_tokens 12000 \
    --vision_3d_resampler none \
    --vision_3d_patch_init tile \
    --vision_3d_sta true \
    --vision_3d_sta_window 5 \
    --vision_3d_slowfast true \
    --vision_3d_fast_grid 16,8,8 \
    --vision_3d_fast_pool avgmax \
    --vision_3d_crossattn_every_n 2 \
    --vision_3d_slow_max_tokens 12000 \
    --vision_3d_rankreg_lambda 1.0 \
    --vision_3d_rankreg_var_w 1.0 \
    --vision_3d_rankreg_cov_w 1.0 \
    --vision_3d_rankreg_gamma 1.0 \
    --vision_3d_rankreg_max_tokens 2048 \
    --vision_3d_rankreg_fast_lambda 1.0 \
    --vision_3d_proj_ortho_lambda 1.0 \
    --vision_3d_proj_fast_wd 1e-5 \
    --vision_3d_proj_fast_init orthogonal \
    --vision_3d_proj_zero_init true \
    --vision_3d_split_lr_groups true \
    --vision_3d_vit_freeze_steps 0 \
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
    --split_dataset_ratio 0.0005 \
    --torch_dtype bfloat16 \
    --fp16 false --bf16 true \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 2 \
    --gradient_accumulation_steps 4 \
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
    --max_length 18000 \
    --warmup_ratio 0.05 \
    --eval_steps 300 \
    --save_steps 100 \
    --logging_steps 5 \
    --save_total_limit 6 \
    --save_only_model false \
    --dataloader_num_workers 4 \
    --dataloader_persistent_workers true \
    --dataloader_prefetch_factor 1 \
    --dataloader_pin_memory false \
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
