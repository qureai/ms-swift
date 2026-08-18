#!/usr/bin/env bash
#SBATCH --job-name=qwen35_atlas_3d_ct
#SBATCH --partition=h100
#SBATCH --reservation=vlm          # dedicated H100 nodes qrnd[2,4]; drop this to use the general h100 pool
#SBATCH --nodes=1
#SBATCH --ntasks=1                  # one task; swift/torchrun spawns 8 procs via NPROC_PER_NODE=8
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=96         # 12 cores/GPU: feeds the 3D CT-volume dataloader (256^3 vols are heavy to decode/window/resample)
#SBATCH --mem=512G                 # CT-volume prefetch buffers are large; high real usage keeps this well above the reaper's 10% floor
#SBATCH --time=6-00:00:00          # generous so a 1-epoch run finishes without a timeout/resume
#SBATCH --output=/cache/fast_data_nas71/janhavi/VLM_qure/vlm/swift/examples/train/liger/logs/%x-%j.out
#SBATCH --error=/cache/fast_data_nas71/janhavi/VLM_qure/vlm/swift/examples/train/liger/logs/%x-%j.err

set -euo pipefail

# --- conda env (swift_july): activate under relaxed `set -u` (conda's cuda-nvcc activate
# script references unset vars and would abort under nounset) ---
set +u
source /cache/fast_data_nas71/janhavi/miniconda3/etc/profile.d/conda.sh
conda activate swift_july
set -u

# ==============================================================================================
# 13 Aug 2026 experiment (NEW run, not resumed). 2D + 3D SFT: Qwen3.5-4B (host VLM + 2D image encoder)
# with a secondary 3D CT-volume encoder (Pillar0/Atlas), Liger kernels on. FULL fine-tune, but the 2D
# pathway is FROZEN: only the LLM + the 3D encoder (Atlas) + its resampler are trained.
#
#   --model            Qwen/Qwen3.5-4B                -> LLM + 2D vision tower (images)
#   --vision_3d_model  YalaLab/Pillar0-ChestCT        -> 3D encoder for CT volumes (<video> .nii.gz)
#   --template         qwen2_5_vl_ct                  -> routes images->visual, volumes->visual_3d
#
# This run vs the 29-Jul one:
#   * bf16 + flash_attention_2 (Atlas runs in bf16 via the autocast fix; ~half the encoder memory,
#     faster/leaner LLM attention).
#   * Perceiver resampler (learned token aggregation) instead of mean-pool + MLP projector.
#   * 2048 CT tokens per volume (up from 1024).
#   * fresh experiment -- no --resume_from_checkpoint.
#
# Trainable: LLM (model.language_model) + visual_3d.encoder + visual_3d.resampler.
# Frozen:    model.visual (2D ViT incl. every block's MLP)  AND  model.visual.merger (2D aligner MLP).
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
# only affect Qwen's native video path (which CT never uses); set here for consistency only.
export VIDEO_MAX_TOKEN_NUM=2048
export VIDEO_MIN_TOKEN_NUM=32

# ---- data --------------------------------------------------------------------------------------
# Replace with your CT dataset(s): swift chat JSONL where CT volumes are referenced via <video>, e.g.
#   {"messages": [{"role":"user","content":"<video>Report this chest CT."},
#                 {"role":"assistant","content":"..."}],
#    "videos": ["/abs/path/scan.nii.gz"]}
# Train split only (train+val; test series held out, CT-RATE excluded, unassigned Segmed dropped).
DATASET_PATHS=(
    "/cache/fast_data_nas71/janhavi/data/ct_training_data/training/13_Aug_2026_vlm_ct_tag_extraction_json_presence_true_training.json"
    "/cache/fast_data_nas71/janhavi/data/ct_training_data/training/13_Aug_2026_ct_tag_extraction_cls_tokens_training.json"
    "/cache/fast_data_nas71/janhavi/data/ct_training_data/training/13_Aug_2026_ct_full_structured_report_training.json"
    "/cache/fast_data_nas71/janhavi/data/ct_training_data/training/13_Aug_2026_ct_mini_structured_report_training.json"
    "/cache/fast_data_nas71/janhavi/data/ct_training_data/training/13_Aug_2026_ct_cls_plus_mini_report_multiturn_training.json"
    "/cache/fast_data_nas71/janhavi/data/ct_training_data/training/13_Aug_2026_ct_cls_plus_mini_report_combined_training.json"
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
    --model Qwen/Qwen3.5-4B \
    --run_name '1308_qwen3_5_4b_atlas_3d_ct' \
    --vision_3d_model YalaLab/Pillar0-ChestCT \
    --vision_3d_trust_remote_code true \
    --vision_3d_max_tokens 2048 \
    --vision_3d_resampler perceiver \
    --vision_3d_resampler_depth 2 \
    --vision_3d_resampler_heads 8 \
    --template qwen2_5_vl_ct \
    --ct_windows "$CT_WINDOWS" \
    --ct_window_base full_range \
    --ct_volume_size 256,256,256 \
    --ct_augment true \
    --ct_augment_prob 0.15 \
    --tuner_type full \
    --dataset "${DATASET_PATHS[@]}" \
    --dataset_shuffle true \
    --load_from_cache_file true \
    --split_dataset_ratio 0.01 \
    --torch_dtype bfloat16 \
    --fp16 false --bf16 true \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 3 \
    --gradient_checkpointing true \
    --learning_rate 1e-5 \
    --freeze_llm false \
    --freeze_vit true \
    --freeze_aligner true \
    --trainable_parameters visual_3d \
    --num_train_epochs 1 \
    --max_length 32000 \
    --warmup_ratio 0.05 \
    --eval_steps 2000 \
    --save_steps 500 \
    --logging_steps 5 \
    --save_total_limit 5 \
    --save_only_model false \
    --dataloader_num_workers 6 \
    --dataset_num_proc 16 \
    --system 'You are a helpful medical assistant.' \
    --deepspeed zero3 \
    --attn_impl flash_attention_2 \
    --use_hf true \
    --use_liger_kernel true \
    --report_to wandb \
    --group_by_length false \
    --padding_free false \
    --packing false \
    --ddp_find_unused_parameters true \
    --output_dir /cache/fast_data_nas71/janhavi/VLM_qure/vlm/swift/vlm_ckpts/1308_qwen3_5_4b_atlas_3d_ct

# Notes:
# * bf16 + flash_attention_2: the Atlas encoder runs in bf16 (its one fp32-hardcoded op is harmonized by
#   an autocast wrap in AtlasAdapter). Needs flash-attn installed in the env; if you hit the
#   "flash_attention_forward() got multiple values for 'cu_seq_lens_q'" error, fall back to --attn_impl sdpa.
# * --vision_3d_resampler perceiver: learned latent queries cross-attend to the Atlas tokens and emit
#   exactly --vision_3d_max_tokens (2048) tokens; replaces the mean-pool + MLP projector. Trained as part
#   of visual_3d and saved into the checkpoint (vision_3d_config records the resampler settings for reload).
# * Memory watch: 2048 CT tokens (2x) + resampler + 256^3 volumes. bf16 + flash-attn give headroom, but if
#   you OOM, first drop --per_device_train_batch_size to 1 (raise --gradient_accumulation_steps to keep the
#   effective batch), then consider fewer tokens or a smaller --ct_volume_size.
# * Trainable set = LLM + visual_3d (encoder + resampler). --trainable_parameters visual_3d keeps the whole
#   3D module trainable even if a transformers version uses the bare `visual` freeze-prefix.
