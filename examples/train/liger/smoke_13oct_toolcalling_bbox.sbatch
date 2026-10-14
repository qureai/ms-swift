#!/usr/bin/env bash
#SBATCH --job-name=smoke_tc_bbox
#SBATCH --partition=l40s
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=320G
#SBATCH --time=01:30:00
#SBATCH --exclude=qrnd6,qrnd12,qrnd15,qrnd16,qrnd20,qrnd22,qrnd25
#SBATCH --output=/cache/fast_data_nas71/janhavi/VLM_qure/vlm/swift/examples/train/liger/logs/%x-%j.out
#SBATCH --error=/cache/fast_data_nas71/janhavi/VLM_qure/vlm/swift/examples/train/liger/logs/%x-%j.err
#
# SMOKE TEST for the 13-Oct tool-calling + bbox warm-start run. Exercises the full pipeline on L40S:
# warm-start load of ckpt-8500, +291 special tokens (embedding/lm_head resize 248320->248448), data load,
# a few train steps, and one checkpoint save of the resized model. NO wandb. Tiny data subset. Not a real run.
set -euo pipefail
set +u
source /cache/fast_data_nas71/janhavi/miniconda3/etc/profile.d/conda.sh
conda activate swift_july
set -u
cd /cache/fast_data_nas71/janhavi/VLM_qure/vlm/swift

MODEL=Qwen/Qwen3.5-2B
WARM=/mnt/scratch/janhavi/vlm_ckpts/warm_starts/6oct_slowfast_cacreorder_ckpt8500
TOKENS=/cache/fast_data_nas71/janhavi/VLM_qure/vlm/swift/examples/train/liger/tools_tokens.txt

export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export CT_DATASET_CHANNELS=1
export NPROC_PER_NODE=4
export CUDA_VISIBLE_DEVICES=0,1,2,3
export USE_HF=1
export MAX_PIXELS=$((3500 * 3500))
export VLLM_FORCE_FA2=1
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=7200
export TORCH_NCCL_ENABLE_MONITORING=1

CT_WINDOWS="lung,mediastinum,abdomen,liver,bone,brain,subdural,stroke,temporal_bone,soft_tissue"
DS_CONFIG=/cache/fast_data_nas71/janhavi/VLM_qure/vlm/swift/examples/train/liger/ds_zero2_overlap.json
OUTPUT_DIR=/mnt/scratch/janhavi/vlm_ckpts/smoke_13oct_toolcalling_bbox
rm -rf "$OUTPUT_DIR" 2>/dev/null || true
mkdir -p "$OUTPUT_DIR"

# Tiny subset of each dataset (#N) so tokenization + s3 volume fetch are fast.
TC=/cache/fast_data_nas71/janhavi/data/ct_training_data/tool_calling/training
DATA=/cache/fast_data_nas71/janhavi/data/ct_final_data/train
DATASET_PATHS=(
    "$TC/12_oct_2026_ct_tool_calling_report_training.json#1500"
    "$DATA/segmed_nodule_bbox_train_04sep2026_clean.json#1500"
    "$DATA/omnict_anatomy_localization_train_29sep2026_clean.json#1500"
    "$DATA/omnict_label_bbox_multiturn_train_29sep2026.json#1500"
    "$DATA/segmed_anatomy_grounded_abnormality_train_08sep2026.json#1500"
)

swift sft \
    --model "$WARM" \
    --new_special_tokens "$TOKENS" \
    --run_name 'smoke_13oct_toolcalling_bbox' \
    --vision_3d_model "$MODEL" \
    --vision_3d_trust_remote_code false \
    --vision_3d_max_tokens 12000 \
    --vision_3d_resampler none \
    --vision_3d_patch_init tile \
    --vision_3d_sta true \
    --vision_3d_sta_window 5 \
    --vision_3d_slowfast true \
    --vision_3d_fast_grid 16,8,8 \
    --vision_3d_fast_pool avgmax \
    --vision_3d_crossattn_every_n 2 \
    --vision_3d_slow_max_tokens 12000 \
    --vision_3d_rankreg_lambda 1.0 \
    --vision_3d_rankreg_var_w 1.0 \
    --vision_3d_rankreg_cov_w 1.0 \
    --vision_3d_rankreg_gamma 1.0 \
    --vision_3d_rankreg_max_tokens 2048 \
    --vision_3d_rankreg_fast_lambda 1.0 \
    --vision_3d_proj_ortho_lambda 1.0 \
    --vision_3d_proj_fast_wd 1e-5 \
    --vision_3d_proj_fast_init orthogonal \
    --vision_3d_proj_zero_init true \
    --vision_3d_split_lr_groups true \
    --vision_3d_vit_freeze_steps 0 \
    --presence_ce_w_true 6.0 \
    --presence_ce_w_false 1.0 \
    --presence_ce_channel cls_json \
    --log_collapse_metrics true \
    --log_train_acc true \
    --eval_cls_auc true \
    --template qwen2_5_vl_ct \
    --ct_windows "$CT_WINDOWS" \
    --ct_window_base full_range \
    --ct_augment false \
    --ct_load_dtype bfloat16 \
    --ct_normalize standardize \
    --tuner_type full \
    --dataset "${DATASET_PATHS[@]}" \
    --dataset_shuffle true \
    --load_from_cache_file false \
    --split_dataset_ratio 0.0 \
    --torch_dtype bfloat16 \
    --fp16 false --bf16 true \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --gradient_checkpointing true \
    --gradient_checkpointing_kwargs '{"use_reentrant": false}' \
    --learning_rate 2e-5 \
    --vit_lr 2e-5 \
    --aligner_lr 2e-4 \
    --freeze_llm false \
    --freeze_vit true \
    --freeze_aligner true \
    --trainable_parameters visual_3d \
    --max_steps 6 \
    --max_length 12000 \
    --warmup_ratio 0.0 \
    --save_steps 5 \
    --save_total_limit 1 \
    --logging_steps 1 \
    --save_only_model true \
    --dataloader_num_workers 4 \
    --dataloader_persistent_workers false \
    --dataloader_prefetch_factor 1 \
    --dataloader_pin_memory false \
    --dataset_num_proc 8 \
    --system 'You are a helpful medical assistant.' \
    --deepspeed "$DS_CONFIG" \
    --attn_impl flash_attention_2 \
    --use_hf true \
    --use_liger_kernel true \
    --report_to none \
    --group_by_length false \
    --padding_free false \
    --packing false \
    --ddp_find_unused_parameters true \
    --output_dir "$OUTPUT_DIR"

echo "=== SMOKE TRAIN DONE (rc=$?) ==="
# Post-checks: resized vocab in the saved checkpoint + the new tokens are registered there.
python - <<'PY'
import json, glob, os
out="/mnt/scratch/janhavi/vlm_ckpts/smoke_13oct_toolcalling_bbox"
ck=sorted(glob.glob(os.path.join(out,"**","checkpoint-*"), recursive=True))
print("saved checkpoints:", ck)
if ck:
    c=json.load(open(os.path.join(ck[-1],"config.json")))
    v=c.get("vocab_size", (c.get("text_config") or {}).get("vocab_size"))
    print("saved vocab_size:", v, "(expected 248448)")
    tj=os.path.join(ck[-1],"tokenizer.json")
    if os.path.exists(tj):
        tk=json.load(open(tj)); added=len(tk.get("added_tokens",[]))
        print("tokenizer added_tokens:", added, "(base 33 + 291 new = 324 expected)")
print("=== SMOKE POSTCHECK DONE ===")
PY
