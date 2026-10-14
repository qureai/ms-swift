#!/usr/bin/env bash
#SBATCH --job-name=qwen35_2b_toolcall_bbox
#SBATCH --partition=h100
#SBATCH --reservation=vlm          # qrnd4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=48
#SBATCH --mem=850G
#SBATCH --time=14-00:00:00
#SBATCH --output=/cache/fast_data_nas71/janhavi/VLM_qure/vlm/swift/examples/train/liger/logs/%x-%j.out
#SBATCH --error=/cache/fast_data_nas71/janhavi/VLM_qure/vlm/swift/examples/train/liger/logs/%x-%j.err
#
# 19-Oct TOOL-CALLING + BBOX WARM-START run. Continues the 6_oct 2B slow-fast CT model from an existing
# checkpoint (--model = the trained checkpoint dir; host LLM + trained visual_3d tower + lm_head are all in
# that checkpoint, 470 visual_3d tensors verified). Goals:
#   (1) teach the implicit tool-calling report format (12_oct, 684,300 train records), and
#   (2) reinforce 3D grounding (bbox datasets from ct_final_data) -- anatomy localization is left/right-biased
#       and nodule/abnormality bbox is near-floor at ckpt-7500 (grounding IoU eval), so more bbox exposure.
#
# WARM-START, NOT a resume. The tool-calling format introduces 291 new special tokens (tools_tokens.txt:
# finding/characterization tags + <*_TOOL> wrappers + <ANSWER> + |<ASP>|). Adding tokens resizes the input
# embedding + tied lm_head (248320 -> 248448, new rows mean-initialised by transformers), which changes the
# embedding/lm_head tensor shapes. A --resume_from_checkpoint would therefore shape-mismatch the DeepSpeed
# optimizer state, so we warm-start the WEIGHTS only (--model <ckpt>, no --resume_from_checkpoint): fresh
# optimizer / scheduler / step 0. All visual_3d flags are kept identical to the 6_oct run so the tower that
# gets rebuilt matches the saved vision_3d_config before its weights are overlaid.
#
# NOTE on AUROC: --eval_cls_auc is ON and the mix now INCLUDES the standalone cls_json roll-call (channel
# cls_json), so the trainer's teacher-forced per-tag AUROC fires on the val split: pooled-across-ranks
# macro/micro/prevw + per-tag CSV/W&B table (seq2seq_trainer._cls_auc_compute_and_log). Per-channel train
# loss (loss/tool, loss/cls_json, loss/kg, loss/nod_bbox, ...) and val loss are logged too.
#
# NOT launched -- config only. Before launch: copy WARM out of the live 6_oct run dir first (live checkpoints
# rotate at save_total_limit 6) and repoint WARM at the copy.
set -euo pipefail
set +u
source /cache/fast_data_nas71/janhavi/miniconda3/etc/profile.d/conda.sh
conda activate swift_july
set -u
cd /cache/fast_data_nas71/janhavi/VLM_qure/vlm/swift

MODEL=Qwen/Qwen3.5-2B                         # base skeleton for --vision_3d_model (the 3D ViT source)
# Warm-start weights = the LATEST 6_oct checkpoint, resolved AT JOB-START (not submit time). This runs when the
# job actually launches, so when queued with `--dependency=afterany:<6_oct_job>` it picks up whatever checkpoint
# the (completed) 6_oct run left behind -- no manual repointing needed. Scans ALL version dirs (v*/) and sorts by
# the global step number after `checkpoint-` (monotonic across resumes), requiring config.json so a half-written
# final checkpoint is skipped. Fails loudly if none found. Warm-start uses only config+safetensors+tokenizer.
CKPT_ROOT=/mnt/scratch/janhavi/vlm_ckpts/6_oct_qwen3_5_2b_ct_slowfast_cacreorder
WARM=$(for d in "$CKPT_ROOT"/v*/checkpoint-*; do [ -f "$d/config.json" ] && echo "${d##*checkpoint-} $d"; done \
       | sort -n | tail -1 | cut -d' ' -f2-)
[ -z "$WARM" ] && { echo "FATAL: no complete checkpoint found under $CKPT_ROOT" >&2; exit 1; }
echo "WARM resolved to: $WARM"
TOKENS=/cache/fast_data_nas71/janhavi/VLM_qure/vlm/swift/examples/train/liger/tools_tokens.txt

export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export CT_DATASET_CHANNELS=1                 # tag each dataset with a `channel` (per-channel loss)
export NPROC_PER_NODE=8
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export USE_HF=1
export MAX_PIXELS=$((3500 * 3500))
export VLLM_FORCE_FA2=1                       # H100 FA3 PTX crash guard
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=7200      # tolerate slow first-time tokenization
export TORCH_NCCL_ENABLE_MONITORING=1

CT_WINDOWS="lung,mediastinum,abdomen,liver,bone,brain,subdural,stroke,temporal_bone,soft_tissue"
DS_CONFIG=/cache/fast_data_nas71/janhavi/VLM_qure/vlm/swift/examples/train/liger/ds_zero2_overlap.json

OUTPUT_DIR=/mnt/scratch/janhavi/vlm_ckpts/19_oct_qwen3_5_2b_ct_toolcalling_bbox
mkdir -p "$OUTPUT_DIR"

# Data: implicit tool-calling report (full) + bbox/grounding datasets (full, to reinforce grounding) +
# standalone cls_json roll-call and knowledge-graph. The cls_json dataset carries the `cls_json` channel, so
# eval_cls_auc NOW fires (teacher-forced per-tag AUROC over the val split). All share the ShareGPT schema
# {videos:<s3 str>, messages:[{from,value}]}.
TC=/cache/fast_data_nas71/janhavi/data/ct_training_data/tool_calling/training
DATA=/cache/fast_data_nas71/janhavi/data/ct_final_data/train
DATASET_PATHS=(
    "$TC/12_oct_2026_ct_tool_calling_report_training.json"          # 684,300  tool-calling report    (tool)
    "$DATA/9_oct_2026_vlm_ct_tag_extraction_complete_cls_json_training.json"  # 342,795 cls_json roll-call (cls_json -> AUROC)
    "$DATA/12_oct_2026_ct_knowledge_graph_training.json"            # 688,698  knowledge graph        (kg)
    "$DATA/segmed_nodule_bbox_train_04sep2026_clean.json"           # 272,707  nodule 3D bbox         (nod_bbox)
    "$DATA/omnict_anatomy_localization_train_29sep2026_clean.json"  # 187,476  anatomy localization   (anat_loc)
    "$DATA/omnict_label_bbox_multiturn_train_29sep2026.json"        # 187,507  label + bbox multiturn (omnict_bbox)
    "$DATA/segmed_anatomy_grounded_abnormality_train_08sep2026.json" # 63,511  abn grounding          (seg_anat_abn)
)

swift sft \
    --model "$WARM" \
    --new_special_tokens "$TOKENS" \
    --run_name '19_oct_qwen3_5_2b_ct_toolcalling_bbox' \
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
    --output_dir "$OUTPUT_DIR"
