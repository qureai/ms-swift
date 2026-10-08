#!/usr/bin/env bash
#SBATCH --job-name=qwen35_ct_25sep_aux
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

set -euo pipefail
set +u
source /cache/fast_data_nas71/janhavi/miniconda3/etc/profile.d/conda.sh
conda activate swift_july
set -u

# ==============================================================================================
# 25 Sep 2026 -- FULL FINE-TUNE + AUXILIARY ML-DECODER GROUNDING HEAD.
#
# Why: three prior runs ALL collapsed grounding to a constant, input-independent encoder feature
# (verified: encoder reldiff ~0.004, grounding lift ~0) -- 09_sep full-FT, 22_sep high-LR full-FT, and the
# 23_sep frozen-LLM stage-1 (flat loss ~0.65, no recovery through checkpoint-5000). Plain next-token LM
# loss is satisfiable by the report marginal, so the encoder always takes the constant-marginal shortcut.
#
# Fix: an auxiliary ML-Decoder multi-label head on the PRE-projector encoder tokens, trained with
# Asymmetric Loss over the 256 CT cls-token findings. A constant feature CANNOT minimise ASL against
# per-volume labels -> it forces the encoder to encode each volume's findings. Full-FT (not frozen) so
# the trainable LLM can then learn to READ those now-discriminative features for report generation (a
# frozen LLM can't use them). The encoder-path fixes are kept (they made the encoder input-sensitive at
# init; the aux head keeps it that way through training):
#   --vision_3d_patch_init tile     : renorm-to-base-energy warm-start (fresh reldiff ~0.80)
#   --ct_normalize standardize      : per-channel zero-mean/unit-std input
#   --learning_rate 2e-5 + --vit_lr 3e-4 : LLM gentle, encoder+projector fast (attach.py routes visual_3d
#                                          into the vit_lr group; the aux head is part of visual_3d too)
#
# Aux head (opt-in via lambda>0; masked when a volume's series_uid is absent from the label table):
#   --vision_3d_aux_mldecoder_lambda 0.5   lambda * ASL added to the LM loss (backprops into the encoder)
#   --vision_3d_aux_tags        24_sep_ct_cls_tags.txt          (256 ordered classes == head num_classes)
#   --vision_3d_aux_label_table 24_sep_ct_cls_tag_labels.parquet (series_uid -> present-tag indices)
#
# batch 6 (full-FT proven size; batch 8 OOM'd). save_steps 1000 for an early grounding gate.
#
# BEFORE the long launch, GPU smoke-test: startup log must show `aux ML-Decoder head ON (num_classes=256,
# lambda=0.5, label table N volumes)`, and the trainer must log `aux_mldecoder_loss` > 0 with finite
# grad_norm/loss. GATE at checkpoint-1000/2000: fresh_encoder_check.py reldiff must be >>0 (not ~0.004)
# and grounding_loss_probe.py lift must be >0 -- the first time any run breaks the collapse.
# ==============================================================================================

export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export NPROC_PER_NODE=8
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export USE_HF=1
export MAX_PIXELS=$((3500 * 3500))

DS_CONFIG=/cache/fast_data_nas71/janhavi/VLM_qure/vlm/swift/examples/train/liger/ds_zero2_overlap.json

DATASET_PATHS=(
    "/cache/fast_data_nas71/janhavi/data/ct_training_data/training/13_Aug_2026_vlm_ct_tag_extraction_json_presence_true_training.json"
    "/cache/fast_data_nas71/janhavi/data/ct_training_data/training/13_Aug_2026_ct_tag_extraction_cls_tokens_training.json"
    "/cache/fast_data_nas71/janhavi/data/ct_training_data/training/13_Aug_2026_ct_full_structured_report_training.json"
    "/cache/fast_data_nas71/janhavi/data/ct_training_data/training/13_Aug_2026_ct_mini_structured_report_training.json"
    "/cache/fast_data_nas71/janhavi/data/ct_training_data/training/13_Aug_2026_ct_cls_plus_mini_report_multiturn_training.json"
    "/cache/fast_data_nas71/janhavi/data/ct_training_data/training/13_Aug_2026_ct_cls_plus_mini_report_combined_training.json"
    "/cache/fast_data_nas71/janhavi/data/ct_training_data/training/04_Sep_2026_ct_knowledge_graph_training.json"
    "/cache/fast_data_nas71/janhavi/data/ct_vlm_grounding/pe/pe_multiturn_train_28aug2026.json"
    "/cache/fast_data_nas71/janhavi/data/ct_vlm_grounding/omnict/omnict_anatomy_localization_train_28aug2026.json"
    "/cache/fast_data_nas71/janhavi/data/ct_vlm_grounding/omnict/omnict_abnormality_train_28aug2026.json"
    "/cache/fast_data_nas71/janhavi/data/ct_vlm_grounding/cac/cac_multiturn_train_28aug2026.json"
    "/cache/fast_data_nas71/janhavi/data/ct_vlm_grounding/nodule_fpr/nodule_fpr_multiturn_train_28aug2026.json#600000"
    "/cache/fast_data_nas71/janhavi/data/ct_vlm_grounding/segmed_nodule_matching/segmed_nodule_bbox_train_04sep2026.json"
    "/cache/fast_data_nas71/janhavi/data/ct_vlm_grounding/segmed_nodule_matching/segmed_nodule_qa_train_04sep2026.json"
    "/cache/fast_data_nas71/janhavi/data/ct_vlm_grounding/omnict/omnict_anatomy_grounded_abnormality_train_08sep2026.json"
    "/cache/fast_data_nas71/janhavi/data/ct_vlm_grounding/segmed_nodule_matching/segmed_anatomy_grounded_abnormality_train_08sep2026.json"
)
for DATASET_PATH in "${DATASET_PATHS[@]}"; do
    if [[ ! -f "${DATASET_PATH%%#*}" ]]; then
        echo "Training dataset not found: $DATASET_PATH" >&2; exit 1
    fi
done

CT_WINDOWS="lung,mediastinum,abdomen,liver,bone,brain,subdural,stroke,temporal_bone,soft_tissue"
AUX_TAGS=/cache/fast_data_nas71/janhavi/VLM_qure/vlm/swift/examples/train/liger/24_sep_ct_cls_tags.txt
AUX_TABLE=/cache/fast_data_nas71/janhavi/data/ct_data_misc/24_sep_ct_cls_tag_labels.parquet

swift sft \
    --model Qwen/Qwen3.5-4B \
    --run_name '2509_qwen3_5_4b_ct_fullft_aux' \
    --vision_3d_model Qwen/Qwen3.5-4B \
    --vision_3d_trust_remote_code false \
    --vision_3d_max_tokens 10000 \
    --vision_3d_resampler none \
    --vision_3d_sta true \
    --vision_3d_patch_init tile \
    --vision_3d_aux_mldecoder_lambda 0.5 \
    --vision_3d_aux_tags "$AUX_TAGS" \
    --vision_3d_aux_label_table "$AUX_TABLE" \
    --template qwen2_5_vl_ct \
    --ct_windows "$CT_WINDOWS" \
    --ct_window_base full_range \
    --ct_augment true \
    --ct_augment_prob 0.15 \
    --ct_load_dtype bfloat16 \
    --ct_normalize standardize \
    --tuner_type full \
    --dataset "${DATASET_PATHS[@]}" \
    --dataset_shuffle true \
    --load_from_cache_file true \
    --split_dataset_ratio 0.01 \
    --torch_dtype bfloat16 \
    --fp16 false --bf16 true \
    --per_device_train_batch_size 6 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 2 \
    --gradient_checkpointing true \
    --learning_rate 2e-5 \
    --vit_lr 3e-4 \
    --freeze_llm false \
    --freeze_vit true \
    --freeze_aligner true \
    --trainable_parameters visual_3d \
    --num_train_epochs 1 \
    --max_length 32000 \
    --warmup_ratio 0.05 \
    --eval_steps 4000 \
    --save_steps 1000 \
    --logging_steps 5 \
    --save_total_limit 10 \
    --save_only_model false \
    --dataloader_num_workers 4 \
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
    --output_dir /cache/fast_data_nas71/janhavi/VLM_qure/vlm/swift/vlm_ckpts/2509_qwen3_5_4b_ct_fullft_aux
