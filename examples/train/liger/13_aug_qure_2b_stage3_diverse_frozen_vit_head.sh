#!/usr/bin/env bash
set -euo pipefail

# STAGE 3: diverse multiturn SFT on 13_Aug_2026_multiturn_diverse (~404k conversations,
# 19 templates: cls-JSON -> {bbox, ASP, graph, ontology/structured report, sections,
# region-identify, comparisons}), starting from the stage-2 joint checkpoint.
#
# WHAT IS TRAINED:   LLM + aligner (the language side learns the new task chains)
# WHAT IS FROZEN:    ViT (--freeze_vit true)  +  ML-Decoder head (MLD_FREEZE_HEAD=1)
#   The vision features and the classifier that reached macro AUC 0.885 on the 140k eval are
#   final; stage 3 must not disturb them. With both frozen the ASL term has no trainable path,
#   so ASL_WEIGHT=0 skips the classification loss entirely. The head weights are still loaded
#   from the checkpoint (attach_head restores ml_decoder_head.* explicitly) and are carried
#   through to every saved checkpoint, so downstream inference keeps working unchanged.
#
# TOKEN WEIGHTING (qure_ce_weight_plugin, via Liger fused-CE ce_weight — zero cost):
#   ` true`=6.0  ` false`=1.0  (6:1, as specified — biases the cls turn's presence decisions
#   toward recall; stage 2 ran these 1:1)   JSON boilerplate=0.5   everything else=1.0
#
# VAL: 13_Aug_2026_multiturn_diverse_VAL.json — stratified, 100 rows from EACH of the 19
#   templates (split_diverse_train_val.py), so eval covers every capability mix.
#
# PACKING STAYS OFF (plugin invariant). Lazy tokenize; augmentation on as in stage 2.

SWIFT_ROOT="/fastdata4/sahil_work/VLM_qure/vlm/swift"
CONDA_ENV="${CONDA_ENV:-/home/ubuntu/miniconda3/envs/ms-swift-may-s4}"
export PATH="$CONDA_ENV/bin:$PATH"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export IMAGE_MAX_TOKEN_NUM=2500
export VIDEO_MAX_TOKEN_NUM=128
export FPS_MAX_FRAMES=12

# --- stage-3 loss configuration ------------------------------------------------------------
# All stage-3-specific behavior (head restore from ckpt, head freeze, bbox-safe augmentation)
# lives in qure_stage3_plugin.py, which ONLY this script loads. The shared plugins are stock.
export AR_WEIGHT="${AR_WEIGHT:-1.0}"
export ASL_WEIGHT="${ASL_WEIGHT:-0.0}"            # head+ViT frozen -> no classification loss
export MLD_RESTORE_HEAD=1                         # reload ml_decoder_head.* from BASE_MODEL
export MLD_FREEZE_HEAD=1                          # freeze the ML-Decoder head
export QURE_AUG_SKIP_BBOX=1                       # no augmentation on bbox conversations
# 6:1 true/false token weighting for the cls-JSON turns
export QURE_W_TRUE="${QURE_W_TRUE:-6.0}"
export QURE_W_FALSE="${QURE_W_FALSE:-1.0}"
export QURE_W_FORMAT="${QURE_W_FORMAT:-0.5}"
export QURE_W_DEFAULT="${QURE_W_DEFAULT:-1.0}"
# augmentation identical to stage 2
export QURE_AUG_PROB="${QURE_AUG_PROB:-1.0}"
export QURE_AUG_CROP_MIN="${QURE_AUG_CROP_MIN:-0.10}"
export QURE_AUG_CROP_MAX="${QURE_AUG_CROP_MAX:-0.15}"
export FM_ACCELERATE_REPO="${FM_ACCELERATE_REPO:-/fastdata4/sahil_work/VLM_qure/vision_foundation_2d/foundation_model_accelerate}"
export PYTHONPATH="$SWIFT_ROOT:$FM_ACCELERATE_REPO:${PYTHONPATH:-}"
echo "stage3: AR only (ASL_WEIGHT=0, head FROZEN, ViT FROZEN) · true:false = ${QURE_W_TRUE}:${QURE_W_FALSE}" >&2

export NPROC_PER_NODE=8
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

BASE_MODEL="${BASE_MODEL:-$SWIFT_ROOT/output/Qwen3.5-2B_cls_json_mldecoder_joint/v13-20260809-033855/checkpoint-1800}"
DATA_DIR="/fastdata4/sahil_work/VLM_qure/data_preparation/training_data"
TRAIN_SET="$DATA_DIR/13_Aug_2026_multiturn_diverse_TRAIN.json"
VAL_SET="$DATA_DIR/13_Aug_2026_multiturn_diverse_VAL.json"
SPECIAL_TOKENS_PATH="$SWIFT_ROOT/examples/train/liger/token.txt"
RUN_NAME="Qwen3.5-2B_stage3_diverse_frozen_vit_head"

[[ -d "$BASE_MODEL" ]] || { echo "base ckpt missing: $BASE_MODEL" >&2; exit 1; }
[[ -f "$TRAIN_SET" ]] || { echo "train set missing: $TRAIN_SET (run split_diverse_train_val.py)" >&2; exit 1; }
[[ -f "$VAL_SET" ]] || { echo "val set missing: $VAL_SET" >&2; exit 1; }
[[ -f "$SPECIAL_TOKENS_PATH" ]] || { echo "token.txt missing: $SPECIAL_TOKENS_PATH" >&2; exit 1; }
[[ -d "$DATA_DIR/cls_label_sidecar" ]] || { echo "cls label sidecar missing" >&2; exit 1; }

# BATCH: same memory envelope as stage 2 (28 x 5 = 1120 effective). ~360 optimizer steps/epoch
# over 402k conversations — short run; eval/save every 50 steps.
# LR: LLM 4e-6 (2x stage 2 — it is the only real learner now and the new region-identify task
# is genuinely new), aligner 5e-5 (halved: ViT is frozen so there is no encoder drift for the
# projector to absorb; it only fine-tunes alignment for the new task mix).
RESUME_ARGS=()
if [[ -n "${RESUME:-}" ]]; then
    RESUME_ARGS=(--resume_from_checkpoint "$RESUME")
    echo "RESUMING from $RESUME" >&2
fi

swift sft \
    --model "$BASE_MODEL" \
    "${RESUME_ARGS[@]}" \
    --run_name "$RUN_NAME" \
    --tuner_type full \
    --dataset "$TRAIN_SET" \
    --val_dataset "$VAL_SET" \
    --split_dataset_ratio 0 \
    --lazy_tokenize true \
    --dataset_shuffle true \
    --data_seed 42 \
    --new_special_tokens "$SPECIAL_TOKENS_PATH" \
    --add_non_thinking_prefix true \
    --loss_scale ignore_empty_think \
    --max_new_tokens 18000 \
    --torch_dtype bfloat16 \
    --per_device_train_batch_size "${BS:-8}" \
    --per_device_eval_batch_size 2 \
    --learning_rate 4e-6 \
    --aligner_lr 5e-5 \
    --freeze_vit true \
    --freeze_aligner false \
    --num_train_epochs "${EPOCHS:-1}" \
    --gradient_accumulation_steps "${GAS:-16}" \
    --gradient_checkpointing true \
    --eval_strategy steps \
    --eval_steps 50 \
    --save_steps 50 \
    --logging_steps 5 \
    --max_length 28000 \
    --warmup_ratio 0.05 \
    --dataloader_num_workers 8 \
    --save_total_limit 5 \
    --save_only_model false \
    --output_dir "output/$RUN_NAME" \
    --system 'You are a helpful medical assistant.' \
    --deepspeed zero2 \
    --attn_impl flash_attention_2 \
    --use_hf true \
    --use_liger_kernel true \
    --report_to tensorboard wandb \
    --create_checkpoint_symlink true \
    --external_plugins "$SWIFT_ROOT/swift_plugins/mldecoder_joint_plugin.py" \
                       "$SWIFT_ROOT/swift_plugins/qure_augment_plugin.py" \
                       "$SWIFT_ROOT/swift_plugins/qure_ce_weight_plugin.py" \
                       "$SWIFT_ROOT/swift_plugins/qure_stage3_plugin.py" \
    --packing false \
    --padding_free false
