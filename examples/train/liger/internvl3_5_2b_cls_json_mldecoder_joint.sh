#!/usr/bin/env bash
set -euo pipefail

# JOINT training: InternVL3.5-2B + ML-Decoder classification head on the vision encoder.
# Port of 06_aug_qure_2b_cls_json_mldecoder_joint.sh (Qwen3.5) to InternVL3.5.
#   loss = AR_WEIGHT*autoregressive_CE + ASL_WEIGHT*ASL(cls_logits, 132 tags)
#
# BASE MODEL: the FIRST-STAGE SFT checkpoint (InternVL analog of Qwen ckpt-5191), i.e. the
# completed output/InternVL3_5-2B_cached_packing_correct_data_split/.../checkpoint-12571.
# We continue from it as a fresh run (new joint objective), NOT --resume.
#
# HEAD TAP (see mldecoder_joint_internvl_plugin.py): InternVL has no Qwen-style pre/post merger
# pair. The head taps model.mlp1's output (post-projector vit_embeds, the tokens the LLM sees),
# regroups the flat tile stream into images via Internvl2Template's num_patches, and runs
# MLDecoder over each image's (n_tiles*256) tokens -> one 132-logit vector per image.
#
# PACKING OFF (required): packing concatenates conversations and destroys the image->row label
# correspondence the ASL term relies on. The plugin refuses to compute ASL if it detects packing.
#
# SCOPE NOTE: this first InternVL run mirrors the CORE Qwen recipe (MLD head + same data + the
# LR spread + ASL weight). The two auxiliary Qwen plugins are NOT loaded here because both patch
# the *Qwen* template and need their own InternVL port:
#   qure_augment_plugin.py     (RandAugment on the vision input)
#   qure_ce_weight_plugin.py   (down-weight JSON boilerplate tokens in the AR CE)
# Port them next if the run looks good.

SWIFT_ROOT="/localstorage/sahil_work/VLM_qure/vlm/swift"
CONDA_ENV="${CONDA_ENV:-/home/ubuntu/.conda/envs/ms-swift-internvl}"
export PATH="$CONDA_ENV/bin:$PATH"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export IMAGE_MAX_TOKEN_NUM=2500
export VIDEO_MAX_TOKEN_NUM=128
export FPS_MAX_FRAMES=12
export INPUT_SIZE=448
export MAX_NUM="${MAX_NUM:-12}"          # max dynamic-resolution tiles per image

# --- loss weights (mirror the Qwen joint run) ---
export AR_WEIGHT="${AR_WEIGHT:-1.0}"
export ASL_WEIGHT="${ASL_WEIGHT:-0.03}"      # pinned constant, parity point
export MLD_HEAD_LR="${MLD_HEAD_LR:-5e-4}"

# --- vision augmentation (foundation RandAugment via qure_augment_internvl_plugin) ---
# elastic/grid distortion (p=0.3), random-crop 10-15% OR rotate +/-15 (p=0.5), gamma/brightness/
# contrast. Applied to each CXR before InternVL's dynamic tiling. AUG=0 disables the plugin.
export AUG="${AUG:-1}"
export QURE_AUG_PROB="${QURE_AUG_PROB:-1.0}"
export QURE_AUG_CROP_MIN="${QURE_AUG_CROP_MIN:-0.10}"
export QURE_AUG_CROP_MAX="${QURE_AUG_CROP_MAX:-0.15}"

export FM_ACCELERATE_REPO="${FM_ACCELERATE_REPO:-/localstorage/sahil_work/VLM_qure/vision_foundation_2d/foundation_model_accelerate}"
export CLS_LABEL_SIDECAR="${CLS_LABEL_SIDECAR:-/localstorage/sahil_work/VLM_qure/data_preparation/training_data/cls_label_sidecar}"
export PYTHONPATH="$SWIFT_ROOT:$FM_ACCELERATE_REPO:${PYTHONPATH:-}"

# HF datasets/Arrow cache + temp MUST live on a big disk. The 80GB JSON -> Arrow conversion needs
# far more than the ~66GB free on root (/), which caused OSError 28 (No space left on device).
# /localstorage has TBs free. (Mirrors internvl3_5_2b_cached_packing.sh's scratch redirection.)
SCRATCH="${SCRATCH:-/localstorage/sahil_work/cache}"
export HF_DATASETS_CACHE="$SCRATCH/hf_datasets"
export HF_HOME="${HF_HOME:-/localstorage/huggingface/hf_home}"
export TMPDIR="$SCRATCH/tmp"
export MODELSCOPE_CACHE="$SCRATCH/modelscope"
export TRITON_CACHE_DIR="$SCRATCH/triton"
export TORCHINDUCTOR_CACHE_DIR="$SCRATCH/torchinductor"
mkdir -p "$HF_DATASETS_CACHE" "$TMPDIR" "$MODELSCOPE_CACHE" "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR"

# --- Weights & Biases (already logged in as sahil1 via ~/.netrc; --report_to below enables it) ---
# Keep wandb's run/cache dirs off root (/) — same ENOSPC reasoning as the HF cache above.
export WANDB_PROJECT="${WANDB_PROJECT:-internvl3_5-mldecoder-joint}"
export WANDB_DIR="${WANDB_DIR:-$SCRATCH/wandb}"
export WANDB_CACHE_DIR="${WANDB_CACHE_DIR:-$SCRATCH/wandb_cache}"
export WANDB_ARTIFACT_DIR="${WANDB_ARTIFACT_DIR:-$SCRATCH/wandb_artifacts}"
mkdir -p "$WANDB_DIR" "$WANDB_CACHE_DIR" "$WANDB_ARTIFACT_DIR"

export NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"

SPECIAL_TOKENS_PATH="$SWIFT_ROOT/examples/train/liger/token.txt"
BASE_MODEL="${BASE_MODEL:-$SWIFT_ROOT/output/InternVL3_5-2B_cached_packing_correct_data_split/v10-20260722-112118/checkpoint-12571}"
DATASET="${DATASET:-/localstorage/sahil_work/VLM_qure/data_preparation/training_data/06_Aug_2026_cls_json_multiturn_asp_graph_report.json}"
RUN_NAME="InternVL3_5-2B_cls_json_mldecoder_joint"

[[ -f "$SPECIAL_TOKENS_PATH" ]] || { echo "special tokens not found: $SPECIAL_TOKENS_PATH" >&2; exit 1; }
[[ -e "$BASE_MODEL" ]]         || { echo "base model not found: $BASE_MODEL" >&2; exit 1; }
[[ -f "$DATASET" ]]           || { echo "dataset not found: $DATASET" >&2; exit 1; }
[[ -d "$CLS_LABEL_SIDECAR" ]] || { echo "cls label sidecar missing: $CLS_LABEL_SIDECAR" >&2; exit 1; }

# Smoke-test knobs: SMOKE=1 runs a tiny few-step CPU-cheap validation (1 GPU, max_steps=SMOKE_STEPS).
SMOKE="${SMOKE:-0}"
EXTRA_ARGS=()
if [[ "$SMOKE" == "1" ]]; then
    export NPROC_PER_NODE=1
    export CUDA_VISIBLE_DEVICES="${SMOKE_GPU:-0}"
    export MLD_DEBUG=1
    EXTRA_ARGS=(--max_steps "${SMOKE_STEPS:-4}" --split_dataset_ratio 0.0 --eval_strategy no
                --save_strategy no --per_device_train_batch_size 2 --gradient_accumulation_steps 1
                --dataset_num_proc 1 --logging_steps 1)
    echo "SMOKE run: 1 GPU, ${SMOKE_STEPS:-4} steps, debug on" >&2
else
    EXTRA_ARGS=(--per_device_train_batch_size "${PDBS:-3}"
                --gradient_accumulation_steps "${GAS:-8}"
                --split_dataset_ratio 0.002 --eval_strategy steps --eval_steps "${EVAL_STEPS:-500}"
                --save_steps 300 --logging_steps 5)
fi

# TUNER: full (default) or lora. LoRA adapters on ALL linear layers (LLM + ViT + aligner) via
# target_modules=all-linear with nothing frozen; the ML-Decoder head still full-trains (plugin).
TUNER="${TUNER:-full}"
TUNER_ARGS=(--tuner_type "$TUNER" --freeze_vit false --freeze_aligner false)
if [[ "$TUNER" == "lora" ]]; then
    TUNER_ARGS+=(--lora_rank "${LORA_RANK:-256}" --lora_alpha "${LORA_ALPHA:-512}"
                 --lora_dropout "${LORA_DROPOUT:-0.05}" --target_modules all-linear --freeze_llm false)
    echo "LoRA: rank=${LORA_RANK:-256} alpha=${LORA_ALPHA:-512} target=all-linear (LLM+ViT+aligner)" >&2
fi

"$CONDA_ENV/bin/python" -m swift.cli.main sft \
    --model "$BASE_MODEL" \
    --model_type internvl_chat \
    --template internvl3_5 \
    --run_name "$RUN_NAME" \
    "${TUNER_ARGS[@]}" \
    --dataset "$DATASET" \
    --lazy_tokenize true \
    --dataset_shuffle true \
    --data_seed 42 \
    --new_special_tokens "$SPECIAL_TOKENS_PATH" \
    --add_non_thinking_prefix true \
    --loss_scale ignore_empty_think \
    --max_new_tokens 18000 \
    --torch_dtype bfloat16 \
    --per_device_eval_batch_size 1 \
    --learning_rate "${LR:-2e-6}" \
    --vit_lr "${VIT_LR:-2e-5}" \
    --aligner_lr "${ALIGNER_LR:-1e-4}" \
    --num_train_epochs 1 \
    --gradient_checkpointing "${GRAD_CKPT:-true}" \
    --max_length 28000 \
    --warmup_ratio 0.05 \
    --dataloader_num_workers "${DL_WORKERS:-48}" \
    --save_total_limit 5 \
    --save_only_model false \
    --output_dir "output/$RUN_NAME" \
    --system 'You are a helpful medical assistant.' \
    --deepspeed "${ZERO:-zero2}" \
    --attn_impl flash_attention_2 \
    --use_hf true \
    --use_liger_kernel true \
    --report_to tensorboard wandb \
    --create_checkpoint_symlink true \
    --external_plugins "$SWIFT_ROOT/swift_plugins/mldecoder_joint_internvl_plugin.py" \
                       $( [[ "${AUG:-1}" == "1" ]] && echo "$SWIFT_ROOT/swift_plugins/qure_augment_internvl_plugin.py" ) \
    --packing false \
    --padding_free false \
    "${EXTRA_ARGS[@]}"
