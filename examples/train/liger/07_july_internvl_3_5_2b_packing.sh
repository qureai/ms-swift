#!/usr/bin/env bash
set -euo pipefail

# InternVL3_5-2B full SFT on the verified-correct dataset set (see data_qure/Data.md §1c).
# InternVL supports packing + padding_free with a flash-attention backend, so both are ON here
# (unlike the Qwen3.5 GatedDeltaNet runs, where packing/padding_free are unsupported).
#
# Packing builds an in-memory index that bin-packs samples up to --max_length (28000) per sequence;
# it is not persisted to disk. per_device_train_batch_size counts PACKED sequences.
#   GLOBAL_BATCH_SIZE = per_device_train_batch_size * NPROC_PER_NODE * gradient_accumulation_steps
#   128 = 2 * 8 * 8

export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export VIDEO_MAX_TOKEN_NUM=128
export FPS_MAX_FRAMES=12
export NPROC_PER_NODE=8
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export INPUT_SIZE=448

SPECIAL_TOKENS_PATH="/localstorage/sahil_work/VLM_qure/vlm/swift/examples/train/liger/token.txt"

# Verified-correct datasets only (bad/broken sets excluded; common duplicate removed).
DATASET_PATHS=(
    "/fastdata4/sahil_work/VLM_qure/data_preparation/training_data/05_June_2026_vlm_bbox_comparison_data_550k_neysa.json"
    "/fastdata4/sahil_work/VLM_qure/data_preparation/training_data/11_Nov_pretraining_data_without_bbox_neysa.json"
    "/fastdata4/sahil_work/VLM_qure/data_preparation/training_data/14_Nov_corrected_internvl_1000res_cxr_nodule_bbox_data_with_conversation_and_target_res.json"
    "/fastdata4/sahil_work/VLM_qure/data_preparation/training_data/15_July_final_segmed_chexpert_mimic_vlm_conv_gemini_response_neysa.json"
    "/fastdata4/sahil_work/VLM_qure/data_preparation/training_data/19_Nov_training_data_internvl_1000res_bbox_data_monochrome2_segmed_with_png_res_neysa.json"
    "/localstorage/vlm_data_parquets/updated_image_path_datas/05_June_2026_vlm_bbox_label_matching_data_617k.json"
    "/localstorage/vlm_data_parquets/updated_image_path_datas/18_June_2026_cls_token_data_mimic_chex_segmed.json"
    "/localstorage/vlm_data_parquets/updated_image_path_datas/18_June_2026_combined_cls_asp_report_prompt_data_mimic_chex_segmed.json"
    "/localstorage/vlm_data_parquets/updated_image_path_datas/18_June_2026_multiturn_cls_asp_report_mimic_chex_segmed.json"
    "/localstorage/vlm_data_parquets/updated_image_path_datas/18_June_2026_prod_cls_token_data_monochrome2_mapped.json"
    "/localstorage/vlm_data_parquets/updated_image_path_datas/22_June_2026_hierarchical_graph_data_internal_mimic_segmed_chexpert.json"
    "/localstorage/vlm_data_parquets/updated_image_path_datas/23_June_2026_corrected_final_report_comparison_data_deepseek_490k.json"
    "/localstorage/vlm_data_parquets/updated_image_path_datas/23_June_2026_corrected_full_tag_extraction_vlm_mimic_chex_segmed.json"
)

if [[ -f "$SPECIAL_TOKENS_PATH" ]]; then
    RESOLVED_SPECIAL_TOKENS_PATH="$SPECIAL_TOKENS_PATH"
elif [[ -f "${SPECIAL_TOKENS_PATH//\/localstorage/\/fastdata4}" ]]; then
    RESOLVED_SPECIAL_TOKENS_PATH="${SPECIAL_TOKENS_PATH//\/localstorage/\/fastdata4}"
    echo "Using alternate special tokens path: $RESOLVED_SPECIAL_TOKENS_PATH" >&2
else
    echo "Special tokens file not found: $SPECIAL_TOKENS_PATH" >&2
    echo "Also tried: ${SPECIAL_TOKENS_PATH//\/localstorage/\/fastdata4}" >&2
    exit 1
fi

# Resolve each dataset path, falling back between /localstorage and /fastdata4.
RESOLVED_DATASET_PATHS=()
for DATASET_PATH in "${DATASET_PATHS[@]}"; do
    if [[ -f "$DATASET_PATH" ]]; then
        RESOLVED_DATASET_PATHS+=("$DATASET_PATH")
    elif [[ -f "${DATASET_PATH//\/localstorage/\/fastdata4}" ]]; then
        ALT_DATASET_PATH="${DATASET_PATH//\/localstorage/\/fastdata4}"
        echo "Using alternate dataset path: $ALT_DATASET_PATH" >&2
        RESOLVED_DATASET_PATHS+=("$ALT_DATASET_PATH")
    elif [[ -f "${DATASET_PATH//\/fastdata4/\/localstorage}" ]]; then
        ALT_DATASET_PATH="${DATASET_PATH//\/fastdata4/\/localstorage}"
        echo "Using alternate dataset path: $ALT_DATASET_PATH" >&2
        RESOLVED_DATASET_PATHS+=("$ALT_DATASET_PATH")
    elif [[ -f "${DATASET_PATH//\/localstorage/\/fastdata3}" ]]; then
        ALT_DATASET_PATH="${DATASET_PATH//\/localstorage/\/fastdata3}"
        echo "Using alternate dataset path: $ALT_DATASET_PATH" >&2
        RESOLVED_DATASET_PATHS+=("$ALT_DATASET_PATH")
    else
        echo "Training dataset not found: $DATASET_PATH" >&2
        exit 1
    fi
done

# Non-thinking SFT data: add the empty non-thinking marker and mask its loss.
swift sft \
    --model OpenGVLab/InternVL3_5-2B-Instruct \
    --run_name '07_july_InternVL3_5-2B-Instruct_correct_data_packing' \
    --tuner_type full \
    --dataset "${RESOLVED_DATASET_PATHS[@]}" \
    --dataset_shuffle true \
    --load_from_cache_file true \
    --new_special_tokens "$RESOLVED_SPECIAL_TOKENS_PATH" \
    --add_non_thinking_prefix true \
    --loss_scale ignore_empty_think \
    --max_new_tokens 25000 \
    --split_dataset_ratio 0.01 \
    --torch_dtype bfloat16 \
    --per_device_train_batch_size 3 \
    --per_device_eval_batch_size 1 \
    --learning_rate 1e-5 \
    --vit_lr 1e-5 \
    --aligner_lr 1e-5 \
    --freeze_vit false \
    --freeze_aligner false \
    --num_train_epochs 1 \
    --gradient_accumulation_steps 4 \
    --eval_steps 800 \
    --save_steps 800 \
    --logging_steps 5 \
    --max_length 28000 \
    --warmup_ratio 0.05 \
    --dataloader_num_workers 32 \
    --dataset_num_proc 32 \
    --save_total_limit 4 \
    --save_only_model false \
    --output_dir output/07_july_InternVL3_5-2B-Instruct_correct_data_packing \
    --system 'You are a helpful medical assistant.' \
    --deepspeed zero3 \
    --attn_impl flash_attn \
    --use_hf true \
    --use_liger_kernel true \
    --report_to tensorboard wandb \
    --create_checkpoint_symlink true \
    --group_by_length false \
    --padding_free false \