#!/usr/bin/env bash
#SBATCH --job-name=qwen35_ct_28sep_fullft
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
# 28 Sep 2026 -- FULL FINE-TUNE, PLAIN CONTROL (no aux head). Fresh from base Qwen/Qwen3.5-4B.
#
# THIS IS THE FIRST RUN AFTER THE REAL ROOT CAUSE WAS FOUND. Every prior run (09_sep, 16_sep/22_sep
# full-FT, the frozen-LLM stage-1, and the 25_sep aux run job 30685) collapsed grounding IDENTICALLY
# for ONE reason, proven by infer_eval/arch_diagnostic.py:
#
#   video_token_id MISMATCH. --template qwen2_5_vl_ct carries video_token_id=151656 (inherited from
#   Qwen2.5-VL); Qwen3.5-4B emits 248057. _encode did findall(input_ids, 151656) -> matched NOTHING ->
#   the <video> placeholder was never expanded -> masked_scatter filled 1 of ~1421 encoder tokens.
#   The LLM literally never received the volume. Diagnostic: #video_tokens=1 vs visual_3d_tokens=1421,
#   dL/d(video)~1e-5 vs 2e-3 text, ablation ~0. The encoder/LR/freeze/aux work was all treating symptoms.
#
# FIX (in code, applies automatically to this config): Volume3DTemplateMixin._init_placeholder_tokens
# now resolves video_token_id from processor/model config (logs `video_token_id 151656 -> 248057`).
# Post-fix diagnostic: SPLICE #video_tokens == visual_3d_tokens (MATCH). A fresh process picks it up.
#
# WHY NO AUX HEAD: the aux ML-Decoder head existed to force the encoder off a constant-marginal
# shortcut -- but that shortcut was a CONSEQUENCE of the LLM getting no volume. Now that the volume
# actually reaches the LLM, plain next-token LM loss has real gradient into the encoder. This control
# tells us whether grounding was ONLY ever the plumbing bug. If it grounds, the aux head is unnecessary
# complexity to drop. If it does NOT ground, we add the aux head back -- now as a real signal, not a
# patch over a broken pipe.
#
# Kept from the diagnosis (these are hygiene, not the fix -- they make the encoder input-sensitive
# from step 0; harmless if unnecessary):
#   --vision_3d_patch_init tile : warm-start the 11-channel CT patch embed (renorm-to-base-energy).
#   --ct_normalize standardize  : per-channel zero-mean/unit-std input.
#   --learning_rate 2e-5 (LLM) + --vit_lr 3e-4 (visual_3d encoder+projector). attach.py registers
#       visual_3d in model_arch.vision_tower so --vit_lr targets it (else the multimodal optimizer
#       DROPS every visual_3d param). Two LRs on purpose: one global LR can't serve both.
#
# MEMORY (batch 2 x grad_accum 6, NOT 6 x 2): the prior "batch 6 proven size" was a LIE -- it was only
# proven in the BROKEN regime, where the video_token_id bug left <video> unexpanded so the LLM saw ~1
# volume token instead of the full encoder token set. With the fix, sequences are now FULL length for the
# first time and batch 6 OOM'd at step 0 in the GatedDeltaNet backward (job 31765, GPU4 78.6/79.2 GiB).
# batch 4 x accum 3 = effective batch 96 (== old 6 x 2 x 8), so the LR stays valid. batch 4 (not 2): at
# max_tokens=4096, batch 2 used only ~30-35 GB / 81.6 (job 31867) -- ~46 GB free -- so batch was doubled to
# use the headroom (worst-case ~16k video tokens/microbatch, vs the 60k that OOM'd at 10000xbatch6, so safe).
#
# SPEED / max_tokens 4096 (was 10000): job 31810 ran clean & healthy at max_tokens=10000 (loss 1.07 and
# moving, fix confirmed) but steady-state 136 s/it -> ~52 DAYS for one 34571-step epoch (14-day limit).
# The cost is the 10000-token/volume budget: the trainable vision encoder does a full fwd+bwd over up to
# 10000 patches AND the LLM carries them as sequence length. max_tokens is a smart-resize CEILING (input
# downsampled so it yields <= N tokens; NOT truncation, NOT pooling), so 4096 ~2x faster (~26 days/epoch);
# at the 14-day wall that is ~50% of an epoch on shuffled data. Trade: coarser voxels -> some fidelity loss
# for small findings (nodules). Memory at 4096 is well under budget, so batch could be raised later to use it.
# eval every 2000, save every 500, keep top 6.
#
# SMOKE-TEST at startup (first ~50 steps): the log MUST show `video_token_id 151656 -> 248057` (the
# fix firing), `registered visual_3d in model_arch.vision_tower`, and an optimizer line with BOTH a
# vit group (3e-4) and an llm group (2e-5). save_steps 500 (keep top 6) for an EARLY grounding gate.
#
# GATE at checkpoint-1000/2000 (~10-19h) -- the first time any run should break the collapse:
#   fresh_encoder_check.py reldiff must be >>0 (not ~0.004) AND grounding_loss_probe.py lift must be >0.
# WATCH (secondary): video-token embedding norm was ~151x text on the blind checkpoint; should
# self-regulate once the projector gets real gradient. If it persists and hurts, add a projector
# output-norm (one line). Not blocking.
# ==============================================================================================

# DATA: the 13_Aug presence json is replaced by the 28_Sep COMPLETE cls-json roll-call (every tag listed with
# explicit "presence": true/false; records carry "channel":"cls_json"). This is plain SFT (cross-entropy) over
# all datasets including the new cls-json.
# PRESENCE-BOOLEAN CE WEIGHTING (optional, OFF by default -> plain unweighted SFT). Up-weights the LM loss
# at the presence value-token (true/false) positions of the cls-json roll-call, so getting a finding's
# presence right matters more; w_true > w_false makes predicting `false` for a real (true) finding the
# hardest loss. Scoped to the cls_json channel only (other datasets untouched). Token ids are derived from
# the tokenizer at runtime (no hardcoding); scored on hidden_states[-1] sliced to the logits_to_keep window
# (Liger stays on). To ENABLE, add these three args to the `swift sft` block below (uncommented):
#     --presence_ce_w_true 6.0 --presence_ce_w_false 2.0 --presence_ce_channel cls_json
# Omit them (or leave both weights at 1.0) => feature fully off, identical to before. Logs `presence_ce_extra`.

export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
# Per-dataset loss logging: tag each dataset with a short `channel` (swift/dataset/ct_channels.py) and log
# per-channel mean CE to wandb as loss/<channel> (Seq2SeqTrainer._log_channel_losses, no-grad, Liger kept).
export CT_DATASET_CHANNELS=1
export NPROC_PER_NODE=8
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export USE_HF=1
export MAX_PIXELS=$((3500 * 3500))

DS_CONFIG=/cache/fast_data_nas71/janhavi/VLM_qure/vlm/swift/examples/train/liger/ds_zero2_overlap.json

DATASET_PATHS=(
    "/cache/fast_data_nas71/janhavi/data/ct_training_data/training/28_Sep_2026_vlm_ct_tag_extraction_complete_cls_json_training.json"
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

swift sft \
    --model Qwen/Qwen3.5-4B \
    --run_name '2809_qwen3_5_4b_ct_fullft' \
    --vision_3d_model Qwen/Qwen3.5-4B \
    --vision_3d_trust_remote_code false \
    --vision_3d_max_tokens 4096 \
    --vision_3d_resampler none \
    --vision_3d_sta true \
    --vision_3d_patch_init tile \
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
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 2 \
    --gradient_accumulation_steps 3 \
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
    --eval_steps 2000 \
    --save_steps 500 \
    --logging_steps 5 \
    --save_total_limit 6 \
    --save_only_model false \
    --dataloader_num_workers 8 \
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
    --resume_from_checkpoint "${RESUME_CKPT:-/cache/fast_data_nas72/janhavi/vlm_ckpts/2809_qwen3_5_4b_ct_fullft/v3-20260929-160759/checkpoint-1000}" \
    --ignore_data_skip true \
    --output_dir /cache/fast_data_nas71/janhavi/vlm_ckpts/2809_qwen3_5_4b_ct_fullft
