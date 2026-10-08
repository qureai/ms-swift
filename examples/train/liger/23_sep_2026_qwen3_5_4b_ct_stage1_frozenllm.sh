#!/usr/bin/env bash
#SBATCH --job-name=qwen35_ct_23sep_s1
#SBATCH --partition=h100
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
# 23 Sep 2026 -- STAGE 1 (VISION ALIGNMENT, LLM FROZEN) with the encoder-path fixes.
#
# WHY frozen-LLM: three joint full-FT runs ALL collapsed grounding identically -- 09_sep (random init,
# 1e-5), and the 22_sep full-FT (renorm patch-init + 3e-4 encoder LR + standardized input). The last one
# started with a strongly input-sensitive encoder (fresh reldiff ~0.80) but DECAYED to a constant
# (checkpoint-2000 reldiff 0.011, grounding lift +0.0000) within 2000 steps: a trainable LLM lowers loss
# faster by fitting the label marginal in its own weights, so the vision path is optional -> its gradient
# goes weak -> the patch-embed decays -> blind. Init/LR/normalization were necessary but NOT sufficient.
#
# Freezing the LLM removes the escape hatch: with the LLM frozen, the ONLY way to reduce next-token loss
# is through the vision tokens, so gradient MUST keep improving the encoder -- it cannot decay without loss
# rising. This structurally forces grounding. It's also the go/no-go gate: if grounding does not appear
# even here, the problem is deeper than the LLM-collapse.
#
#   * --freeze_llm true : ONLY visual_3d (encoder + STA + projector) trains; LLM + 2D tower frozen.
#   * --learning_rate 2e-4 : this is the ENCODER lr (the only trainable group), so a single --learning_rate
#       suffices -- NO --vit_lr / multimodal-optimizer split needed while the LLM is frozen.
#   * --vision_3d_patch_init tile : renormalized-to-base-energy warm-start (the fixed init; fresh reldiff ~0.80).
#   * --ct_normalize standardize : per-channel zero-mean/unit-std input (matches the tower's expected stats).
#   * batch 6 : an earlier frozen-LLM batch-8 attempt OOM'd (activation memory of the ~9k-token forward+
#       backward through the frozen 4B LLM is unchanged by freezing). 6 is the proven-safe size. Do NOT raise.
#   * --save_steps 1000 : frequent checkpoints so the grounding gate can be read early.
#
# GATE (checkpoint-1000/2000): infer_eval/fresh_encoder_check.py --model <ckpt> (encoder reldiff must stay
#   >>0, unlike full-FT's collapse to ~0.01) + grounding_loss_probe.py (lift must be >0). If grounding
#   holds, a STAGE 2 (unfreeze LLM at low lr + vit_lr for the encoder) continues from the best checkpoint.
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

swift sft \
    --model Qwen/Qwen3.5-4B \
    --run_name '2309_qwen3_5_4b_ct_stage1_frozenllm' \
    --vision_3d_model Qwen/Qwen3.5-4B \
    --vision_3d_trust_remote_code false \
    --vision_3d_max_tokens 10000 \
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
    --per_device_train_batch_size 6 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 2 \
    --gradient_checkpointing true \
    --learning_rate 2e-4 \
    --freeze_llm true \
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
    --output_dir /cache/fast_data_nas71/janhavi/VLM_qure/vlm/swift/vlm_ckpts/2309_qwen3_5_4b_ct_stage1_frozenllm
