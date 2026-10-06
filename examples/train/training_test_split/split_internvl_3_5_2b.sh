#!/usr/bin/env bash
set -euo pipefail

# Prepare the train/test split for InternVL3_5-2B, using the verified-correct dataset set
# (see data_qure/Data.md §1c). This step tokenizes the data ONCE and splits it into train/val,
# saving to disk with HF save_to_disk under $OUTPUT_DIR/{train,val}.
#
# Packing is intentionally NOT set here: `swift export --to_cached_dataset` rejects --packing,
# because packing is applied at TRAIN time over this cached dataset. Consume it with:
#     swift sft --cached_dataset $OUTPUT_DIR --packing true ...   (no --dataset needed)
#
# The cache is tokenizer/template-specific, so it is tied to InternVL3_5-2B + these template args
# (new_special_tokens, add_non_thinking_prefix, loss_scale, system, max_length). Keep them identical
# in the training run.

export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export INPUT_SIZE=448
export CUDA_VISIBLE_DEVICES=0
# Keep all regenerable caches/temp OFF the (shared, often-full) root disk -> on /fastdata4.
# The ModelScope dataset cache is what filled root and crashed the earlier run.
export MODELSCOPE_CACHE=/fastdata4/sahil_work/cache/modelscope
export HF_DATASETS_CACHE=/fastdata4/sahil_work/cache/hf_datasets
export TMPDIR=/fastdata4/sahil_work/tmp
mkdir -p "$MODELSCOPE_CACHE" "$HF_DATASETS_CACHE" "$TMPDIR"
# Cap threads PER worker so dataset_num_proc processes don't each spawn dozens of
# tokenizer/BLAS/torch threads and oversubscribe the box. Single-threaded workers => load ~= num_proc.
export OMP_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
# Skip (drop) any datapoint whose image can't be read, instead of substituting a noise placeholder.
# load_image re-raises when this is set, and --strict false drops the row.
export SKIP_BROKEN_IMAGES=1

SPECIAL_TOKENS_PATH="/localstorage/sahil_work/VLM_qure/vlm/swift/examples/train/liger/token.txt"
OUTPUT_DIR="/fastdata4/sahil_work/VLM_qure/cached_datasets/InternVL3_5-2B_correct_data_split"
SPLIT_RATIO=0.01   # fraction of train reserved as the test/val split

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

# --- resolve each dataset path (/localstorage <-> /fastdata4 fallback, both directions) ---
RESOLVED_DATASET_PATHS=()
for DATASET_PATH in "${DATASET_PATHS[@]}"; do
    if [[ -f "$DATASET_PATH" ]]; then
        RESOLVED_DATASET_PATHS+=("$DATASET_PATH")
    elif [[ -f "${DATASET_PATH//\/localstorage/\/fastdata4}" ]]; then
        RESOLVED_DATASET_PATHS+=("${DATASET_PATH//\/localstorage/\/fastdata4}")
        echo "Using alternate dataset path: ${DATASET_PATH//\/localstorage/\/fastdata4}" >&2
    elif [[ -f "${DATASET_PATH//\/fastdata4/\/localstorage}" ]]; then
        RESOLVED_DATASET_PATHS+=("${DATASET_PATH//\/fastdata4/\/localstorage}")
        echo "Using alternate dataset path: ${DATASET_PATH//\/fastdata4/\/localstorage}" >&2
    else
        echo "Training dataset not found: $DATASET_PATH" >&2
        exit 1
    fi
done

swift export \
    --model OpenGVLab/InternVL3_5-2B-Instruct \
    --to_cached_dataset true \
    --dataset "${RESOLVED_DATASET_PATHS[@]}" \
    --split_dataset_ratio "$SPLIT_RATIO" \
    --dataset_shuffle true \
    --data_seed 42 \
    --new_special_tokens "$RESOLVED_SPECIAL_TOKENS_PATH" \
    --add_non_thinking_prefix true \
    --loss_scale ignore_empty_think \
    --max_length 28000 \
    --max_new_tokens 25000 \
    --torch_dtype bfloat16 \
    --system 'You are a helpful medical assistant.' \
    --dataset_num_proc 64 \
    --strict false \
    --use_hf true \
    --output_dir "$OUTPUT_DIR"

echo "Cached split written to: $OUTPUT_DIR/{train,val}"
echo "Train with:  swift sft --cached_dataset $OUTPUT_DIR --packing true ..."
