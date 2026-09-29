#!/usr/bin/env bash
set -euo pipefail

# Live-training script for Qwen/Qwen3.5-2B that trains DIRECTLY from --dataset (NO cached
# dataset). This is the augmentation-friendly path:
#
#   --lazy_tokenize true  => the full _encode pipeline (text tokenization + image processing)
#   runs per-sample at __getitem__ each epoch, instead of being run once and frozen to disk.
#   Inject your image augmentation into the image-load / transform path and it will be applied
#   fresh every epoch. (A `--to_cached_dataset` split would bake pixel_values once and defeat
#   stochastic augmentation.)
#
# ⚠️ packing MUST stay false here. Swift raises `Packing and lazy_tokenize are incompatible.`
# (base_args.py:130-132), because packing pre-encodes the whole dataset to bin-pack by length —
# which freezes pixel_values once and kills per-epoch augmentation, defeating this script.
# (packing=true also auto-sets padding_free=true internally; they are not independent knobs.)
#
# padding_free is collation-time only, needs no pre-encoding, and is compatible with
# lazy_tokenize + augmentation, so it is set true below. group_by_length is redundant under
# padding_free (no padding to group away) and is left false.
#
# ⚠️ Qwen3.5 uses GatedDeltaNet. Per the header of qure_full_4b.sh, with the transformers/swift
# backend `padding_free` may NOT be supported for Qwen3.5. If training errors on padding_free at
# runtime, set `--padding_free false` and `--group_by_length true` instead (both keep lazy
# tokenize + augmentation intact).
#
# Batch sizing:
#   GLOBAL_BATCH_SIZE = per_device_train_batch_size * data_parallel_size * gradient_accumulation
# Tune per_device_train_batch_size up (2B leaves more headroom than 4B) and keep the global
# batch fixed by lowering gradient_accumulation_steps.

export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export IMAGE_MAX_TOKEN_NUM=2500
export VIDEO_MAX_TOKEN_NUM=128
export FPS_MAX_FRAMES=12
export NPROC_PER_NODE=8
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

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

# --- resolve special tokens path (/localstorage <-> /fastdata4 fallback) ---
if [[ -f "$SPECIAL_TOKENS_PATH" ]]; then
    RESOLVED_SPECIAL_TOKENS_PATH="$SPECIAL_TOKENS_PATH"
elif [[ -f "${SPECIAL_TOKENS_PATH//\/localstorage/\/fastdata4}" ]]; then
    RESOLVED_SPECIAL_TOKENS_PATH="${SPECIAL_TOKENS_PATH//\/localstorage/\/fastdata4}"
    echo "Using alternate special tokens path: $RESOLVED_SPECIAL_TOKENS_PATH" >&2
else
    echo "Special tokens file not found: $SPECIAL_TOKENS_PATH" >&2
    exit 1
fi

# --- resolve each dataset path (/localstorage <-> /fastdata4 fallback) ---
RESOLVED_DATASET_PATHS=()
for DATASET_PATH in "${DATASET_PATHS[@]}"; do
    if [[ -f "$DATASET_PATH" ]]; then
        RESOLVED_DATASET_PATHS+=("$DATASET_PATH")
    elif [[ -f "${DATASET_PATH//\/localstorage/\/fastdata4}" ]]; then
        ALT_DATASET_PATH="${DATASET_PATH//\/localstorage/\/fastdata4}"
        echo "Using alternate dataset path: $ALT_DATASET_PATH" >&2
        RESOLVED_DATASET_PATHS+=("$ALT_DATASET_PATH")
    else
        echo "Training dataset not found: $DATASET_PATH" >&2
        echo "Also tried: ${DATASET_PATH//\/localstorage/\/fastdata4}" >&2
        exit 1
    fi
done

# Non-thinking SFT data: Qwen3.5 is hybrid-thinking, so add the empty
# non-thinking marker and mask its loss.
swift sft \
    --model Qwen/Qwen3.5-2B \
    --run_name 'Qwen3.5-2B_lazy_augment_pretraining_cleaned_bbox_segmed' \
    --tuner_type full \
    --dataset "${RESOLVED_DATASET_PATHS[@]}" \
    --dataset_shuffle true \
    --lazy_tokenize true \
    --load_from_cache_file false \
    --new_special_tokens "$RESOLVED_SPECIAL_TOKENS_PATH" \
    --add_non_thinking_prefix true \
    --loss_scale ignore_empty_think \
    --max_new_tokens 18000 \
    --split_dataset_ratio 0.01 \
    --torch_dtype bfloat16 \
    --per_device_train_batch_size 20 \
    --per_device_eval_batch_size 1 \
    --learning_rate 1e-5 \
    --vit_lr 1e-5 \
    --aligner_lr 1e-5 \
    --freeze_vit false \
    --freeze_aligner false \
    --num_train_epochs 1 \
    --gradient_accumulation_steps 3 \
    --eval_steps 800 \
    --save_steps 800 \
    --logging_steps 5 \
    --max_length 28000 \
    --warmup_ratio 0.05 \
    --dataloader_num_workers 8 \
    --dataset_num_proc 4 \
    --save_total_limit 5 \
    --save_only_model false \
    --output_dir output/Qwen3.5-2B_lazy_augment_pretraining_cleaned_bbox_segmed \
    --system 'You are a helpful medical assistant.' \
    --deepspeed zero2 \
    --attn_impl flash_attention_2 \
    --use_hf true \
    --use_liger_kernel true \
    --report_to tensorboard wandb \
    --create_checkpoint_symlink true \
    --group_by_length false \
    --padding_free true \
    --packing false
