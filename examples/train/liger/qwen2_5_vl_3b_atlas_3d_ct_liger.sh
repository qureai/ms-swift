#!/usr/bin/env bash
set -euo pipefail

# ==============================================================================================
# 2D + 3D SFT: Qwen2.5-VL-3B-Instruct (host VLM + 2D image encoder) with a secondary 3D CT-volume
# encoder (Pillar0/Atlas), Liger kernels on. FULL fine-tune, but the 2D pathway is FROZEN:
# only the LLM + the 3D encoder (Atlas) + its projector are trained.
#
#   --model            Qwen/Qwen2.5-VL-3B-Instruct   -> LLM + 2D vision tower (images)
#   --vision_3d_model  YalaLab/Pillar0-ChestCT        -> 3D encoder for CT volumes (<video> .nii.gz)
#   --template         qwen2_5_vl_ct                  -> routes images->visual, volumes->visual_3d
#
# Trainable: LLM (model.language_model) + visual_3d.encoder + visual_3d.proj.
# Frozen:    model.visual (2D ViT incl. every block's MLP)  AND  model.visual.merger (2D aligner MLP).
#            (X-rays therefore ride a frozen 2D tower; only the LLM adapts to their features.)
#
# See ../../../README_VISION_3D.md for the full argument reference.
# ==============================================================================================

export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export NPROC_PER_NODE=8
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
# HuggingFace hub + cache (Pillar0 is gated; make sure you are logged in / HF_HOME points at your cache).
export USE_HF=1

# Image (X-ray) resize budget -- APPLIES to <image> inputs going through the 2D tower. Higher max ->
# higher-resolution X-rays -> more image tokens -> finer detail (and more memory / longer sequences).
export MAX_PIXELS=$((3500 * 3500))   # 12,250,000 px cap per image

# VIDEO_* settings are INERT for CT: the CT template bypasses Qwen's native video processor. CT
# resolution is set by --ct_volume_size and CT token count by --vision_3d_max_tokens (below). These
# two only affect Qwen's native video path (which CT never uses); set here as requested.
# (512/32 are token-scale; VIDEO_MAX_PIXELS=512 would be < one 28x28 patch, so we use the *_TOKEN_NUM vars.)
export VIDEO_MAX_TOKEN_NUM=512
export VIDEO_MIN_TOKEN_NUM=32

# ---- data --------------------------------------------------------------------------------------
# Replace with your CT dataset(s): swift chat JSONL where CT volumes are referenced via <video>, e.g.
#   {"messages": [{"role":"user","content":"<video>Report this chest CT."},
#                 {"role":"assistant","content":"..."}],
#    "videos": ["/abs/path/scan.nii.gz"]}
# (For a quick smoke, point this at test_cases/dummy_mixed_data/ct_mixed.jsonl after running
#  `python test_cases/make_dummy_mixed_data.py`.)
DATASET_PATHS=(
    "/cache/fast_data_nas71/janhavi/data/ct_training_data/15_July_2026_preliminary_ct_autoreporting_data.json"
)
for DATASET_PATH in "${DATASET_PATHS[@]}"; do
    if [[ ! -f "$DATASET_PATH" ]]; then
        echo "Training dataset not found: $DATASET_PATH (edit DATASET_PATHS in this script)" >&2
        exit 1
    fi
done

# 11 windows (full_range base + 10 anatomical) == Pillar0-ChestCT native in_channels (11), so the
# pretrained patch embed is kept as-is. --ct_volume_size >=128^3 yields Atlas's 3-scale layout.
CT_WINDOWS="lung,mediastinum,abdomen,liver,bone,brain,subdural,stroke,temporal_bone,soft_tissue"

swift sft \
    --model Qwen/Qwen2.5-VL-3B-Instruct \
    --run_name 'qwen2_5_vl_3b_atlas_3d_ct' \
    --vision_3d_model YalaLab/Pillar0-ChestCT \
    --vision_3d_trust_remote_code true \
    --vision_3d_max_tokens 256 \
    --template qwen2_5_vl_ct \
    --ct_windows "$CT_WINDOWS" \
    --ct_window_base full_range \
    --ct_volume_size 128,128,128 \
    --ct_augment true \
    --ct_augment_prob 0.15 \
    --tuner_type full \
    --dataset "${DATASET_PATHS[@]}" \
    --dataset_shuffle true \
    --load_from_cache_file true \
    --split_dataset_ratio 0.01 \
    --torch_dtype float32 \
    --fp16 false --bf16 false \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 20 \
    --gradient_checkpointing true \
    --learning_rate 1e-5 \
    --freeze_llm false \
    --freeze_vit true \
    --freeze_aligner true \
    --trainable_parameters visual_3d \
    --num_train_epochs 1 \
    --max_length 28000 \
    --warmup_ratio 0.05 \
    --eval_steps 800 \
    --save_steps 800 \
    --logging_steps 5 \
    --save_total_limit 4 \
    --save_only_model false \
    --dataloader_num_workers 8 \
    --dataset_num_proc 8 \
    --system 'You are a helpful medical assistant.' \
    --deepspeed zero3 \
    --attn_impl sdpa \
    --use_hf true \
    --use_liger_kernel true \
    --report_to tensorboard \
    --group_by_length false \
    --padding_free false \
    --packing false \
    --ddp_find_unused_parameters true \
    --output_dir /cache/fast_data_nas71/janhavi/VLM_qure/vlm/swift/vlm_ckpts

# Notes:
# * Trainable set = LLM + visual_3d (encoder + proj). --freeze_vit/--freeze_aligner true freeze the
#   whole 2D pathway; --trainable_parameters visual_3d guarantees the 3D encoder+projector stay
#   trainable even if a transformers version uses the bare `visual` freeze-prefix (which would else
#   also match `visual_3d`). To instead train the projector only: --freeze_parameters visual_3d.encoder.
# * flash_attention_2 may raise "flash_attention_forward() got multiple values for 'cu_seq_lens_q'"
#   with some transformers+flash-attn combos (unrelated to the 3D code). If so, switch to --attn_impl sdpa
#   (verified working with the CT path).
# * --ddp_find_unused_parameters true is kept as a safety net: on a mixed image+CT dataset, image-only
#   batches run visual_3d only as a zeroed dummy forward. It can be set false (all trainable params --
#   LLM + visual_3d -- are touched every step) for a small speedup; keep true if unsure.
