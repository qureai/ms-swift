#!/usr/bin/env bash
# STAGE 6c: REDO of the tool-calling stage with POSITIVE + NEGATIVE tool cases (B3 = B2 positives + N negatives), head TRAINED.
#   init      stage-5 ckpt-116 (the stage-6 starting point; override with BASE_MODEL=... e.g. the 6b ckpt-608 for a continuation)
#   negatives N1-N7: negative screen->normal_screen, empty detect/count/locate on absent findings, verdict->report with NO
#             further call (chain termination), tools offered but not needed, verifiers that find nothing, specialist block
#             with every presence false, knowledge chain for an absent finding. Built by build_component_b3_negatives.py.
#   data      A = 80% of stage 4 (all 19 templates, bbox turns re-serialised, cls verbatim)
#             B = tool-calling set (W1-W9; self-executor: tool results are assistant turns with loss;
#                 the NOISED screening_head prior carries loss:false; image-returning tools use role tool)
#   loss      AR (+ per-message loss flags) + ASL on labelled rows only (row-mask in mldecoder_joint_plugin)
#   frozen    ViT only. Head + LLM + aligner train. ASL weight cosine 0.3 -> 0.05 over 800 steps.
#   tools     `tools` field per row -> qwen3_5 agent template renders <tool_call>/<tool_response>.
set -euo pipefail

# STAGE 5: add HEAD-CONDITIONED generation on top of stage 4, same frozen substrate.
#
# Starts from the stage-4 checkpoint (ckpt-300) and trains on a FRESH refresher of the stage-4
# capabilities (re-composed, not replayed) + the new head-guided/direct-bbox additions, so ALL
# stage-4 prompts/capabilities are retained without re-showing the model its exact stage-4 data.
#
# NEW capabilities (context = frozen ML-Decoder head output, injected as a prior turn):
#   V  head prob scores      -> cls-JSON -> ontology report
#   W  head youden positives -> cls-JSON -> ontology report
#   X  head positives        -> knowledge graph -> ontology report
#   Y  head positives        -> knowledge graph -> Findings+Impression+CT-recommendations
#   Z  head positives        -> ASP -> Findings+Impression+CT-recommendations
#   DB direct bbox           -> single-turn bbox, NO head context   (capability #6)
#   HB head positives        -> bbox for flagged findings            (if built with --bbox)
#
# DATA (per your directive — do NOT replay the exact stage-4 conversations):
#   * REFRESHER set (17_refresher_*): stage-4's 19 templates RE-COMPOSED with DIVERSE_SEED=15,
#     so images get different template chains / section combos -> fresh data, same capabilities,
#     no verbatim replay. Sized as an anti-forgetting replay buffer, not a full retrain.
#   * head-guided + direct-bbox additions (15_stage5_headguided_*).
# The exact 13_Aug stage-4 TRAIN/VAL files are intentionally NOT used here.
#
# DISTRIBUTION-BASED knobs (replacing crude tail-upsampling):
#   * sampling: build_headguided_s5.py S5_ALPHA (target freq ∝ prevalence**ALPHA; 0.5 default).
#   * true/false token weight: 6:1 is the DISTRIBUTION-DERIVED global value for this corpus
#     (2.84% per-slot true prior -> a 1.8-nat boundary shift = exp(1.8)=6.05, boundary p*=0.143
#     in the Youden region). Kept on the fast Liger ce_weight path; per-class movement comes
#     from S5_ALPHA sampling, not from a positional plugin.
#
# WHAT IS TRAINED:   LLM + aligner        (same as stage 4)
# WHAT IS FROZEN:    ViT + ML-Decoder head (MUST stay frozen — the head GENERATES the injected
#   context; unfreezing it makes the youden thresholds stale and creates train/serve skew, and
#   breaks the fixed-J* target the whole eval depends on). ASL_WEIGHT=0 (frozen head, no grad).
#
# TRAIN/SERVE CONTRACT: the prior text is produced by head_context.serialize_context — the SAME
#   function mld_head_service.py / infer_headguided_3stage.py must use at inference. Do not
#   re-implement the format anywhere.
#
# PACKING STAYS OFF (plugin invariant; image->row correspondence). Lazy tokenize; augmentation
# on for non-bbox rows.

SWIFT_ROOT="/fastdata4/sahil_work/VLM_qure/vlm/swift"
CONDA_ENV="${CONDA_ENV:-/home/ubuntu/miniconda3/envs/ms-swift-may-s4}"
export PATH="$CONDA_ENV/bin:$PATH"
# --- caches OFF the 439 GB root disk (it hit 100% on the first launch: HF datasets Arrow cache) ---
export HF_HOME="/fastdata4/sahil_work/VLM_qure/cache_stage6/hf"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export TMPDIR="/dev/shm/cache_stage6_tmp"   # tmpfs: NFS TMPDIR gave Errno 16 on pymp-* semaphore cleanup
export TRITON_CACHE_DIR="/fastdata4/sahil_work/VLM_qure/cache_stage6/triton"
export TORCHINDUCTOR_CACHE_DIR="/fastdata4/sahil_work/VLM_qure/cache_stage6/inductor"
export MODELSCOPE_CACHE="/fastdata4/sahil_work/VLM_qure/cache_stage6/modelscope"   # swift routes dataset cache here even with --use_hf
export XDG_CACHE_HOME="/fastdata4/sahil_work/VLM_qure/cache_stage6/xdg"
mkdir -p "$HF_DATASETS_CACHE" "$TMPDIR" "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR" "$MODELSCOPE_CACHE" "$XDG_CACHE_HOME"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export IMAGE_MAX_TOKEN_NUM=2500
export VIDEO_MAX_TOKEN_NUM=128
export FPS_MAX_FRAMES=12
# Robustness: if a rank ever desyncs on a collective (e.g. a corrupt image gives one rank a bad
# batch), ABORT on the watchdog timeout instead of spinning at 100% util forever. The v0 run
# hung this way at step 85; combined with a fresh DATA_SEED on resume it should not recur.
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_ASYNC_ERROR_HANDLING=1

# --- loss configuration (identical to stage 4) ---------------------------------------------
export AR_WEIGHT="${AR_WEIGHT:-1.0}"
export ASL_WEIGHT_START="${ASL_WEIGHT_START:-0.30}"; export ASL_WEIGHT_END="${ASL_WEIGHT_END:-0.05}"; export ASL_DECAY_STEPS="${ASL_DECAY_STEPS:-800}"   # head TRAINED: cosine-decayed ASL
export MLD_RESTORE_HEAD=1                         # reload ml_decoder_head.* from BASE_MODEL
export MLD_FREEZE_HEAD=0                          # stage 6: train the ML-Decoder head
export QURE_AUG_SKIP_BBOX=1                       # install the bbox-safe augmentation guard
# bbox rows now get PHOTOMETRIC-only aug (brightness/contrast/gamma/CLAHE/noise/blur/sharpen/
# jpeg/downscale) — intensity changes only, so box coordinates stay valid. GEOMETRIC ops
# (elastic/grid/crop-zoom) remain suppressed on bbox rows. Non-bbox rows keep full aug.
export QURE_AUG_BBOX_MODE="${QURE_AUG_BBOX_MODE:-photometric}"
export QURE_AUG_PHOTO_PROB="${QURE_AUG_PHOTO_PROB:-1.0}"
# true/false token weighting: 6.0 is the DISTRIBUTION-DERIVED global ratio for this corpus
# (2.84% per-slot true prior -> 1.8-nat boundary shift -> exp(1.8)=6.05). Same value as stage 4,
# now justified by the distribution rather than chosen; per-class movement is via S5_ALPHA sampling.
export QURE_W_TRUE="${QURE_W_TRUE:-6.0}"
export QURE_W_FALSE="${QURE_W_FALSE:-1.0}"
export QURE_W_FORMAT="${QURE_W_FORMAT:-0.5}"
export QURE_W_DEFAULT="${QURE_W_DEFAULT:-1.0}"
# bbox IoU lever: up-weight coordinate digit tokens (crude, global — see qure_ce_weight_plugin).
# Pairs with the COORD_STEP=50 (5%) quantization baked into the bbox data at build time.
export QURE_W_COORD="${QURE_W_COORD:-2.0}"
# augmentation identical to stage 4
export QURE_AUG_PROB="${QURE_AUG_PROB:-1.0}"
export QURE_AUG_CROP_MIN="${QURE_AUG_CROP_MIN:-0.10}"
export QURE_AUG_CROP_MAX="${QURE_AUG_CROP_MAX:-0.15}"
export FM_ACCELERATE_REPO="${FM_ACCELERATE_REPO:-/fastdata4/sahil_work/VLM_qure/vision_foundation_2d/foundation_model_accelerate}"
export PYTHONPATH="$SWIFT_ROOT:$FM_ACCELERATE_REPO:${PYTHONPATH:-}"
echo "stage6c: AR + ASL · base=${BASE_MODEL:-stage5 ckpt-116} · A 120k + B3 (B2 positives + N negatives) + retention (head-guided 71k, refresher 47k, diverse 60k) · tool-calling self-executor" >&2

export NPROC_PER_NODE=8
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# stage-4 best generative operating point
BASE_MODEL="${BASE_MODEL:-/fastdata4/sahil_work/VLM_qure/data_preparation/training_data/output/Qwen3.5-2B_stage5_headguided_frozen/v1-20260904-104531/checkpoint-116}"
DATA_DIR="/fastdata4/sahil_work/VLM_qure/data_preparation/training_data"
# FRESH refresher of stage-4 capabilities (re-composed, seed 15) + stage-5 head-guided/bbox additions
RF_TRAIN="$DATA_DIR/stage6_toolcalling/A_TRAIN_120k.jsonl"
RF_VAL="$DATA_DIR/stage6_toolcalling/A_VAL.jsonl"
S5_TRAIN="$DATA_DIR/stage6_toolcalling/B3_TRAIN.jsonl"
S5_VAL="$DATA_DIR/stage6_toolcalling/B3_VAL.jsonl"
# RETENTION from the earlier stages, so the tool redo does not narrow the model (generalisation):
#   head-guided generation (the train/serve context contract), the stage-5 capability refresher, and a
#   60k sample of the stage-3/4 composition templates.  All are plain messages+images rows (no tools field).
RET_HG="$DATA_DIR/15_stage5_headguided_TRAIN.json"
RET_RF="$DATA_DIR/17_refresher_multiturn_TRAIN.json"
RET_DV="$DATA_DIR/stage6_toolcalling/RET_diverse60k_TRAIN.jsonl"
RET_HG_VAL="$DATA_DIR/15_stage5_headguided_VAL.json"
RET_RF_VAL="$DATA_DIR/17_refresher_multiturn_VAL.json"
SPECIAL_TOKENS_PATH="$SWIFT_ROOT/examples/train/liger/token.txt"
RUN_NAME="${RUN_NAME:-Qwen3.5-2B_stage6c_posneg}"

[[ -d "$BASE_MODEL" ]] || { echo "base ckpt missing: $BASE_MODEL" >&2; exit 1; }
for f in "$RF_TRAIN" "$RF_VAL" "$S5_TRAIN" "$S5_VAL" "$RET_HG" "$RET_RF" "$RET_DV" "$RET_HG_VAL" "$RET_RF_VAL"; do
    [[ -f "$f" ]] || { echo "dataset missing: $f (build the refresher + stage-5 additions first — see runbook)" >&2; exit 1; }
done
[[ -f "$SPECIAL_TOKENS_PATH" ]] || { echo "token.txt missing: $SPECIAL_TOKENS_PATH" >&2; exit 1; }
[[ -d "$DATA_DIR/cls_label_sidecar" ]] || { echo "cls label sidecar missing" >&2; exit 1; }

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
    --dataset "$RF_TRAIN" "$S5_TRAIN" "$RET_HG" "$RET_RF" "$RET_DV" \
    --val_dataset "$RF_VAL" "$S5_VAL" \
    --split_dataset_ratio 0 \
    --lazy_tokenize true \
    --dataset_shuffle true \
    --data_seed "${DATA_SEED:-42}" \
    --new_special_tokens "$SPECIAL_TOKENS_PATH" \
    --add_non_thinking_prefix true \
    --loss_scale ignore_empty_think \
    --max_new_tokens 18000 \
    --torch_dtype bfloat16 \
    --per_device_train_batch_size "${BS:-8}" \
    --per_device_eval_batch_size 2 \
    --learning_rate 4e-6 \
    --aligner_lr 5e-5 \
    --freeze_vit false \
    --vit_lr 2e-6 \
    --freeze_aligner false \
    --num_train_epochs "${EPOCHS:-1}" \
    --gradient_accumulation_steps "${GAS:-16}" \
    --gradient_checkpointing true \
    --eval_strategy steps \
    --eval_steps 50 \
    --save_steps 50 \
    --logging_steps 5 \
    --max_length 28000 \
    --warmup_ratio 0.03 \
    --dataloader_num_workers "${NUM_WORKERS:-12}" \
    --save_total_limit 3 \
    --save_only_model false \
    --output_dir "$DATA_DIR/output/$RUN_NAME" \
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
