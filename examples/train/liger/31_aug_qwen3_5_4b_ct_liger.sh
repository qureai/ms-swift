#!/usr/bin/env bash
#SBATCH --job-name=qwen35_self3d_ct
#SBATCH --partition=h100
#SBATCH --reservation=vlm          # dedicated H100 nodes qrnd[2,4]; drop this to use the general h100 pool
#SBATCH --nodes=1
#SBATCH --ntasks=1                  # one task; swift/torchrun spawns 8 procs via NPROC_PER_NODE=8
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=96
#SBATCH --mem=600G
#SBATCH --time=9-00:00:00
#SBATCH --output=/cache/fast_data_nas71/janhavi/VLM_qure/vlm/swift/examples/train/liger/logs/%x-%j.out
#SBATCH --error=/cache/fast_data_nas71/janhavi/VLM_qure/vlm/swift/examples/train/liger/logs/%x-%j.err

set -euo pipefail

# --- conda env (swift_july) ---
set +u
source /cache/fast_data_nas71/janhavi/miniconda3/etc/profile.d/conda.sh
conda activate swift_july
set -u

# ==============================================================================================
# 24 Aug 2026 experiment. 2D + 3D SFT: Qwen3.5-4B host, and the 3D CT-volume encoder is the HOST
# MODEL'S OWN vision tower (Qwen3.5-4B) used Photon-style -- i.e. the same vision encoder as the
# main VLM, loaded as a separate 3D instance. FULL fine-tune with the 2D pathway frozen: only the
# LLM + the 3D encoder + its resampler are trained.
#
#   --model            Qwen/Qwen3.5-4B   -> LLM + 2D vision tower (images)
#   --vision_3d_model  Qwen/Qwen3.5-4B   -> SAME model; its vision tower is extracted as the 3D encoder
#   --template         qwen2_5_vl_ct     -> routes images->visual, volumes->visual_3d
#
# Note: `--vision_3d_model == --model` loads Qwen3.5-4B a second time at startup only to extract its
# vision tower into `model.visual_3d` (the LLM half of that second load is freed). So visual_3d is a
# SEPARATE, trainable copy initialised from Qwen3.5-4B's pretrained vision tower -- it is NOT weight-shared
# with the frozen 2D tower (model.visual).
#
# Qwen3.5-4B vision: patch_size=16, spatial_merge=2, temporal_patch=2, in_channels=3. Therefore:
#   * H,W must be divisible by patch*merge = 32 and T by 2 -> 384 % 32 == 0, 384 % 2 == 0  -> 384^3 OK.
#     (512^3 also divides, but is much heavier -- see below; 384^3 chosen as the lighter option.)
#   * CHANNELS ARE NOT LIMITED TO 3. The channel count = len(ct_windows) + 1 (the full_range base). If it
#     equals the tower's native 3, the pretrained RGB patch-embed is temporally INFLATED (kept). If it is
#     anything else (e.g. 11 here), ensure_conv3d_patch_embed() REINITIALISES just the patch-embed conv to
#     that many channels and it trains from scratch -- the rest of the tower stays pretrained. We use the
#     full 10-window CT set (same as the Atlas run) => 11 channels => patch-embed reinitialised & trained.
#
# !!! MEMORY WARNING !!! A 384^3 volume patchifies to (T/2)*(H/16)*(W/16) = 192*24*24 = 110,592 patches
#   PER VOLUME through the Qwen ViT (~110 images' worth). Even bf16 + flash-attn + grad-checkpointing, a
#   large micro-batch will OOM. per_device_train_batch_size=4 is set as requested, but expect to drop it to
#   1-2; if it OOMs at 1, reduce --ct_volume_size or --vision_3d_max_tokens.
# ==============================================================================================

export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export NPROC_PER_NODE=8
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export USE_HF=1

export MAX_PIXELS=$((3500 * 3500))          # <image> (X-ray) resize budget for the 2D tower

# VIDEO_* are INERT for CT (the CT template bypasses Qwen's native video processor); set for consistency only.
export VIDEO_MAX_TOKEN_NUM=2048
export VIDEO_MIN_TOKEN_NUM=32

# ---- data --------------------------------------------------------------------------------------
DATASET_PATHS=(
    "/cache/fast_data_nas71/janhavi/data/ct_training_data/training/13_Aug_2026_vlm_ct_tag_extraction_json_presence_true_training.json"
    "/cache/fast_data_nas71/janhavi/data/ct_training_data/training/13_Aug_2026_ct_tag_extraction_cls_tokens_training.json"
    "/cache/fast_data_nas71/janhavi/data/ct_training_data/training/13_Aug_2026_ct_full_structured_report_training.json"
    "/cache/fast_data_nas71/janhavi/data/ct_training_data/training/13_Aug_2026_ct_mini_structured_report_training.json"
    "/cache/fast_data_nas71/janhavi/data/ct_training_data/training/13_Aug_2026_ct_cls_plus_mini_report_multiturn_training.json"
    "/cache/fast_data_nas71/janhavi/data/ct_training_data/training/13_Aug_2026_ct_cls_plus_mini_report_combined_training.json"
    # --- grounding datasets (bbox z-xy 0-1000; added 31 Aug 2026) ---
    "/cache/fast_data_nas71/janhavi/data/ct_vlm_grounding/pe/pe_multiturn_train_28aug2026.json"
    "/cache/fast_data_nas71/janhavi/data/ct_vlm_grounding/omnict/omnict_anatomy_localization_train_28aug2026.json"
    "/cache/fast_data_nas71/janhavi/data/ct_vlm_grounding/omnict/omnict_abnormality_train_28aug2026.json"
    "/cache/fast_data_nas71/janhavi/data/ct_vlm_grounding/cac/cac_multiturn_train_28aug2026.json"
    "/cache/fast_data_nas71/janhavi/data/ct_vlm_grounding/nodule_fpr/nodule_fpr_multiturn_train_28aug2026.json#600000"
)
for DATASET_PATH in "${DATASET_PATHS[@]}"; do
    if [[ ! -f "${DATASET_PATH%%#*}" ]]; then
        echo "Training dataset not found: $DATASET_PATH (edit DATASET_PATHS in this script)" >&2
        exit 1
    fi
done

# Full CT window set (same as the Atlas run): 10 windows + full_range base == 11 channels. This exceeds
# Qwen's native in_channels=3, so the patch-embed conv is reinitialised to 11ch and trained from scratch
# (rest of the tower stays pretrained). Drop to "lung,mediastinum" (==3ch) if you'd rather KEEP the
# pretrained RGB patch-embed instead.
CT_WINDOWS="lung,mediastinum,abdomen,liver,bone,brain,subdural,stroke,temporal_bone,soft_tissue"

swift sft \
    --model Qwen/Qwen3.5-4B \
    --run_name '3108_qwen3_5_4b_ct' \
    --vision_3d_model Qwen/Qwen3.5-4B \
    --vision_3d_trust_remote_code false \
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
    --per_device_train_batch_size 6 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 2 \
    --gradient_checkpointing true \
    --learning_rate 1e-5 \
    --freeze_llm false \
    --freeze_vit true \
    --freeze_aligner true \
    --trainable_parameters visual_3d \
    --num_train_epochs 1 \
    --max_length 32000 \
    --warmup_ratio 0.05 \
    --eval_steps 4000 \
    --save_steps 2000 \
    --logging_steps 5 \
    --save_total_limit 5 \
    --save_only_model false \
    --dataloader_num_workers 2 \
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
    --output_dir /cache/fast_data_nas71/janhavi/VLM_qure/vlm/swift/vlm_ckpts/3108_qwen3_5_4b_ct

# Notes:
# * 3D encoder = Qwen3.5-4B's OWN vision tower (QwenVLVideoAdapter): the volume is patchified as a "video"
#   and run through the tower; each volume is encoded independently (batching volumes into one tower forward
#   is NOT equivalent for a Qwen tower, so the adapter loops per volume).
# * bf16 + flash_attention_2: Qwen tower is bf16-native. Fall back to --attn_impl sdpa on the "cu_seq_lens_q" error.
# * --vision_3d_resampler perceiver: learns 2048 tokens from the tower's ~27,648 merged tokens (at 384^3).
#   Its cross-attn runs at the tower's hidden dim (large), so it's memory-heavy -- use --vision_3d_resampler
#   none (mean-pool + MLP) to cut memory.
# * MEMORY: see the header warning. If you OOM, drop --per_device_train_batch_size to 1 (raise
#   --gradient_accumulation_steps to keep the effective batch), then reduce --ct_volume_size / --vision_3d_max_tokens.
