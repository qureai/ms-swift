#!/usr/bin/env bash
#SBATCH --job-name=qwen35_ct_zero2
#SBATCH --partition=h100
#SBATCH --reservation=vlm          # qrnd4 only; queue behind the current run so this auto-starts when it frees
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=96         # 12 cores/GPU: feeds the 3D CT-volume dataloader (heavy decode/window/resample)
#SBATCH --mem=512G                 # 13_aug-proven footprint; bf16 volumes are smaller than that run's fp32, so ample headroom
#SBATCH --time=14-00:00:00
#SBATCH --output=/cache/fast_data_nas71/janhavi/VLM_qure/vlm/swift/examples/train/liger/logs/%x-%j.out
#SBATCH --error=/cache/fast_data_nas71/janhavi/VLM_qure/vlm/swift/examples/train/liger/logs/%x-%j.err

set -euo pipefail
set +u
source /cache/fast_data_nas71/janhavi/miniconda3/etc/profile.d/conda.sh
conda activate swift_july
set -u

# ==============================================================================================
# 07 Sep 2026. Same model as 31_aug (Qwen3.5-4B host + its OWN vision tower as the 3D CT encoder,
# 11 CT windows), VARIABLE resolution (max 10000 tokens/volume), plus the profiler-driven fixes:
#
#   (A) ZeRO-3 -> ZeRO-2 + overlap_comm=true  (ds_zero2_overlap.json). The profile showed ZeRO-3's
#       per-step param all-gather was the single largest GPU-time sink (~30% on L40S, un-overlapped).
#       A 4B model fits comfortably on 80GB H100 with ZeRO-2 (params replicated, optimizer sharded),
#       which ELIMINATES the param all-gather; overlap_comm then hides the grad reduce-scatter behind
#       backward. NOTE: ZeRO-2 uses ~16GB/GPU more than ZeRO-3 (replicated params+grads). On 8xH100-80GB
#       batch 6 fits; if you OOM, drop --per_device_train_batch_size, or fall back to --deepspeed zero3
#       (overlap can't be flipped on swift's zero3 preset without a custom json).
#
#   (B) Dataloader: --cpus-per-task 96 (12 cores/GPU) + --mem 512G -- 13_aug's proven footprint (CT
#       dataloader is CPU-heavy so it clears the reaper CPU-eff rule; RAM eff high). prefetch_factor 2 (do
#       NOT reduce -- shallower buffers => more GPU stall), + --ct_load_dtype bfloat16 (bf16 ~=0.43 GB/vol,
#       half of fp32; encoder consumes bf16 anyway -> identical values). num_workers 4: the host-RAM buffer
#       is PER-RANK x8 = 8 * workers * prefetch * batch vols, PLUS ~1.5 GB base RSS per worker proc. 13_aug
#       ran fine at batch 2; we are batch 6 (3x the per-worker buffer), so match its footprint with ~3x
#       fewer workers. At workers 4: 8*4*2*6 ~= 384 vols ~= 165 GB + ~48 GB worker base ~= 250-300 GB (fits
#       512G). LESSON: workers 8 OOM-killed at 512G in 11 min (64 worker procs -> ~330 GB buffer + ~96 GB
#       base > 512G); workers 3 OOM'd 200G at step 190. Do NOT raise workers for variable+10k+batch6.
#       Bigger throughput win: pre-cache resized bf16 volumes to node-local disk (separate data-prep step).
#
#   (C) Code fixes already in the repo (no flag needed, active automatically):
#       - 2D tower dummy forward is now SKIPPED when frozen+no-images (freeze_vit=true here) -> saves the
#         ~166 ms/step wasted param-gather+forward through model.visual on CT-only batches.
#       - The Qwen 3D encoder now BATCHES all volumes of a micro-batch into ONE tower forward (windowless
#         Qwen3.5 tower -> verified bit-equivalent to the per-volume loop), improving GPU utilisation.
#
#   (D) Spatial-Temporal Attention (--vision_3d_sta true): injects temporal/z-axis self-attention into
#       every Qwen vision block (4 temporal heads vs 16 spatial, 1D temporal RoPE, zero-init output proj
#       -> identity at step 0, trains up from zero; spatial blocks NOT frozen). Gives the encoder the
#       z-axis modelling its 2D-only RoPE/pos-embed otherwise lacks. Compatible with the perceiver
#       mean-pool projector (resampler removed) + fixed 256^3 here. Adds ~some encoder compute; drop
#       batch if you OOM. STA config is saved in vision_3d_config, so a reloaded checkpoint (inference)
#       rebuilds it automatically.
#
#   Token compression: --vision_3d_resampler none. In VARIABLE mode (below) there is NO pooling -- each
#   volume passes its native token count straight through a 2-layer MLP projector (the resampler is
#   required to be off in variable mode anyway).
#
#   (E) VARIABLE resolution: --ct_volume_size is UNSET -> each volume keeps a near-native, grid-divisible
#       size, capped so it yields <= --vision_3d_max_tokens (=10000) tokens (the per-volume ceiling, NOT a
#       fixed budget). Larger volumes are isotropically downscaled to fit; small ones keep native detail.
#       So per-volume token counts VARY. Requires --vision_3d_resampler none (asserted). Qwen-VL only.
#       NOTE: variable + 10k cap + batch 6 is HEAVY (up to ~10k video tokens/sample through the LLM, big
#       encoder inputs); drop --per_device_train_batch_size if you OOM.
#
# Resolution is VARIABLE (no --ct_volume_size), max 10000 tokens/volume. Everything else matches 31_aug.
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
    # --- new grounding datasets (added 08 Sep 2026) ---
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

swift sft \
    --model Qwen/Qwen3.5-4B \
    --run_name '0909_qwen3_5_4b_ct_zero2' \
    --vision_3d_model Qwen/Qwen3.5-4B \
    --vision_3d_trust_remote_code false \
    --vision_3d_max_tokens 10000 \
    --vision_3d_resampler none \
    --vision_3d_sta true \
    --template qwen2_5_vl_ct \
    --ct_windows "$CT_WINDOWS" \
    --ct_window_base full_range \
    --ct_augment true \
    --ct_augment_prob 0.15 \
    --ct_load_dtype bfloat16 \
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
    --output_dir /cache/fast_data_nas71/janhavi/VLM_qure/vlm/swift/vlm_ckpts/0909_qwen3_5_4b_ct_zero2

# Follow-ups NOT in this config (need code/data work, not CLI flags):
#  * Pre-cache windowed+resized volumes to local fp16 mmap  -> the real fix for the 31% dataloader stall.
#  * Coarser CT patch (temporal_patch/patch_size) + partial-freeze of lower ViT blocks -> cut the encoder,
#    the dominant compute module (~1.6x the LLM forward).
