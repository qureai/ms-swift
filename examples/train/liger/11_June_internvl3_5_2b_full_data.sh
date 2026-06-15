#!/usr/bin/env bash
set -euo pipefail

# Qwen3.5 uses GatedDeltaNet. With the transformers/swift backend, packing and
# padding_free are not supported. group_by_length can reduce padding if enabled,
# but it adds preprocessing and can make the loss curve noisier.
#
# Current batch size:
#   MICRO_BATCH_SIZE = per-device train batch size
#   GLOBAL_BATCH_SIZE = MICRO_BATCH_SIZE * data parallel size * gradient accumulation steps
#   720 = 6 * 8 * 15
# Increase MICRO_BATCH_SIZE and keep GLOBAL_BATCH_SIZE fixed if the 28k context
# fits comfortably on your GPUs.

export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export VIDEO_MAX_TOKEN_NUM=128
export FPS_MAX_FRAMES=12
export NPROC_PER_NODE=8
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export INPUT_SIZE=448

SPECIAL_TOKENS_PATH="/localstorage/sahil_work/VLM_qure/vlm/swift/examples/train/liger/token.txt"

DATASET_PATHS=(
    "/localstorage/sahil_work/VLM_qure/data_preparation/training_data/05_June_2026_vlm_bbox_comparison_data_550k_neysa.json"
    "/localstorage/sahil_work/VLM_qure/data_preparation/training_data/08_June_2026_final_report_comparison_data_deepseek_490k_neysa.json"
    "/localstorage/sahil_work/VLM_qure/data_preparation/training_data/11_Nov_pretraining_data_without_bbox_neysa.json"
    "/localstorage/sahil_work/VLM_qure/data_preparation/training_data/13_Nov_data_with_ASP_and_multilevel_reasoning_neysa.json"
    "/localstorage/sahil_work/VLM_qure/data_preparation/training_data/14_Nov_corrected_internvl_1000res_cxr_nodule_bbox_data_with_conversation_and_target_res.json"
    "/localstorage/sahil_work/VLM_qure/data_preparation/training_data/15_July_final_segmed_chexpert_mimic_vlm_conv_gemini_response_neysa.json"
    "/localstorage/sahil_work/VLM_qure/data_preparation/training_data/19_Nov_training_data_internvl_1000res_bbox_data_monochrome2_segmed_with_png_res_neysa.json"
)
# Intentionally not included:
# /localstorage/sahil_work/VLM_qure/data_preparation/training_data/1_Aug_final_data_for_pretraining_cleaned_neysa.json

MICRO_BATCH_SIZE=20
GLOBAL_BATCH_SIZE=720
USE_LIGER_KERNEL=true

for DATASET_PATH in "${DATASET_PATHS[@]}"; do
    if [[ ! -f "$DATASET_PATH" ]]; then
        echo "Training dataset not found: $DATASET_PATH" >&2
        exit 1
    fi
done

# Non-thinking SFT data: Qwen3.5 is hybrid-thinking, so add the empty
# non-thinking marker and mask its loss.
swift sft \
    --model OpenGVLab/InternVL3_5-2B-Instruct \
    --run_name '11_june_InternVL3_5-2B-Instruct_corrected_pretraining_cleaned_bbox_segmed_prod_comparison' \
    --tuner_type full \
    --dataset "${DATASET_PATHS[@]}" \
    --dataset_shuffle true \
    --load_from_cache_file true \
    --new_special_tokens "$SPECIAL_TOKENS_PATH" \
    --add_non_thinking_prefix true \
    --loss_scale ignore_empty_think \
    --max_new_tokens 25000 \
    --split_dataset_ratio 0.01 \
    --torch_dtype bfloat16 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 1 \
    --learning_rate 1e-5 \
    --vit_lr 1e-5 \
    --aligner_lr 1e-5 \
    --freeze_vit false \
    --freeze_aligner false \
    --num_train_epochs 1 \
    --gradient_accumulation_steps 20 \
    --eval_steps 800 \
    --save_steps 800 \
    --logging_steps 5 \
    --max_length 28000 \
    --warmup_ratio 0.05 \
    --dataloader_num_workers 8 \
    --dataset_num_proc 8 \
    --save_total_limit 4 \
    --save_only_model false \
    --output_dir output/11_june_InternVL3_5-2B-Instruct_pretraining_cleaned_bbox_segmed_prod \
    --resume_from_checkpoint output/11_june_InternVL3_5-2B-Instruct_pretraining_cleaned_bbox_segmed_prod/v13-20260611-083225/checkpoint-4800 \
    --system 'You are a helpful medical assistant.' \
    --deepspeed zero3 \
    --attn_impl flash_attention_2 \
    --use_hf true \
    --use_liger_kernel true \
    --report_to wandb \
    --create_checkpoint_symlink true \
    --group_by_length false \
    --padding_free false \
    --packing false
