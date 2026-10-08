#!/usr/bin/env bash
#SBATCH --job-name=qwen35_2b_ct_slowfast
#SBATCH --partition=h100
#SBATCH --reservation=vlm          # qrnd4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=48
#SBATCH --mem=768G
#SBATCH --time=14-00:00:00
#SBATCH --output=/cache/fast_data_nas71/janhavi/VLM_qure/vlm/swift/examples/train/liger/logs/%x-%j.out
#SBATCH --error=/cache/fast_data_nas71/janhavi/VLM_qure/vlm/swift/examples/train/liger/logs/%x-%j.err
#
# Qwen3.5-2B CT full-FT, slow-fast anti-collapse recipe. SAME hyperparameters as the 30-Sep 4B run
# (30_sep_2026_qwen3_5_4b_ct_slowfast_fullft.sh) -- only the host model + the 3D ViT source are the 2B
# variant, plus a new run_name / output dir. The 2B model roughly halves per-step compute vs 4B (the
# biggest single speedup lever); Flash-Attention is already wired (3D encoder via --attn_impl, STA +
# slow-fast cross-attn via SDPA) so no extra flags are needed for it.
#
# NOTE: confirm the exact HF id for the 2B VLM before launch -- using `Qwen/Qwen3.5-2B` to mirror
# `Qwen/Qwen3.5-4B`; if the published id differs (e.g. an -Instruct / -VL suffix), set MODEL below.
# The slow-fast / STA / cross-attn dims are all derived from the model config at attach time, so nothing
# here needs manual resizing for the smaller hidden size. NOT launched -- config only.
#
# CHANGES vs the 4B config (per request):
#  * --per_device_train_batch_size 3 -> 6.
#  * --vision_3d_slow_max_tokens 4096 -> 12000  AND  --vision_3d_max_tokens 3072 -> 12000 : the slow branch
#    now carries up to 12000 encoder tokens per volume (full-resolution context via cross-attn, while <video>
#    stays at 512 fast tokens). This is a LARGE compute+memory increase in the 3D encoder (12000 tokens
#    through STA) -- deliberately chosen for more volume context.
#  * --gradient_checkpointing true -> false : recompute was ~20-30% of step time; disabling it is a direct
#    speedup IF it fits in memory.
#
# >>> MEMORY WARNING / SMOKE-THEN-ADJUST (do NOT launch blindly) <<<
# batch 6 + 12000 encoder tokens + NO gradient checkpointing compound memory hard and will very likely OOM
# on an H100 (80GB) as-is. The PLAN (run only when instructed): run a 1-step smoke, read peak memory, and if
# it OOMs, re-enable --gradient_checkpointing true first, then if still tight drop --per_device_train_batch_size
# (6 -> 3/2/1). The 12000-token encoder is the dominant new cost. These values are the aggressive target to
# be validated by the smoke, not a known-good config.
set -euo pipefail
set +u
source /cache/fast_data_nas71/janhavi/miniconda3/etc/profile.d/conda.sh
conda activate swift_july
set -u
cd /cache/fast_data_nas71/janhavi/VLM_qure/vlm/swift

MODEL=Qwen/Qwen3.5-2B                         # host VLM (2B); 3D ViT is taken from the same 2B model

export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export CT_DATASET_CHANNELS=1                 # tag each dataset with a `channel` (per-channel loss + presence/AUC gate)
export NPROC_PER_NODE=8
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export USE_HF=1
export MAX_PIXELS=$((3500 * 3500))
export VLLM_FORCE_FA2=1                       # H100 FA3 PTX crash guard
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=7200      # tolerate slow first-time tokenization of 17 datasets
export TORCH_NCCL_ENABLE_MONITORING=1             # keep the monitor, but with the long heartbeat above

CT_WINDOWS="lung,mediastinum,abdomen,liver,bone,brain,subdural,stroke,temporal_bone,soft_tissue"
DS_CONFIG=/cache/fast_data_nas71/janhavi/VLM_qure/vlm/swift/examples/train/liger/ds_zero2_overlap.json

# Output on /mnt/scratch (created if missing).
OUTPUT_DIR=/mnt/scratch/janhavi/vlm_ckpts/1_oct_qwen3_5_2b_ct_slowfast
mkdir -p "$OUTPUT_DIR"

# (1) All training jsons from ct_final_data/train.
DATA=/cache/fast_data_nas71/janhavi/data/ct_final_data/train
DATASET_PATHS=(
    "$DATA/30_sep_2026_vlm_ct_tag_extraction_complete_cls_json_training.json"
    "$DATA/30_sep_2026_ct_tag_extraction_cls_tokens_training.json"
    "$DATA/30_sep_2026_ct_cls_plus_mini_report_combined_training.json"
    "$DATA/30_sep_2026_ct_cls_plus_mini_report_multiturn_training.json"
    "$DATA/13_Aug_2026_ct_mini_structured_report_training.json"
    "$DATA/04_Sep_2026_ct_knowledge_graph_training.json"
    "$DATA/cac_json_train_30sep2026.json"
    "$DATA/nodule_fpr_combined_train_30sep2026.json"
    "$DATA/pe_multiturn_train_29sep2026.json"
    "$DATA/segmed_nodule_bbox_train_04sep2026_clean.json"
    "$DATA/segmed_nodule_qa_train_04sep2026_clean.json"
    "$DATA/segmed_anatomy_grounded_abnormality_train_08sep2026.json"
    "$DATA/omnict_label_bbox_multiturn_train_29sep2026.json"
    "$DATA/omnict_region_findings_qa_train_29sep2026.json"
    "$DATA/omnict_anatomy_localization_train_29sep2026_clean.json"
    "$DATA/report_comparison_train_09sep2026_clean.json"
)

swift sft \
    --model "$MODEL" \
    --run_name '1_oct_qwen3_5_2b_ct_slowfast' \
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
    --vision_3d_vit_freeze_steps 300 \
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
    --load_from_cache_file true \
    --split_dataset_ratio 0.0005 \
    --torch_dtype bfloat16 \
    --fp16 false --bf16 true \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 2 \
    --gradient_accumulation_steps 4 \
    --gradient_checkpointing true \
    --gradient_checkpointing_kwargs '{"use_reentrant": false}' \
    --learning_rate 2e-5 \
    --vit_lr 2e-5 \
    --aligner_lr 2e-4 \
    --freeze_llm false \
    --freeze_vit true \
    --freeze_aligner true \
    --trainable_parameters visual_3d \
    --num_train_epochs 1 \
    --max_length 18000 \
    --warmup_ratio 0.05 \
    --eval_steps 300 \
    --save_steps 100 \
    --logging_steps 5 \
    --save_total_limit 6 \
    --save_only_model false \
    --dataloader_num_workers 4 \
    --dataloader_persistent_workers true \
    --dataloader_prefetch_factor 1 \
    --dataloader_pin_memory false \
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
    --resume_from_checkpoint /mnt/scratch/janhavi/vlm_ckpts/1_oct_qwen3_5_2b_ct_slowfast/v0-20261002-192752/checkpoint-3200 \
    --output_dir "$OUTPUT_DIR"
