#!/usr/bin/env bash
set -euo pipefail

# JOINT training: Qwen/Qwen3.5-2B + ML-Decoder classification head on the vision encoder.
#   loss = autoregressive CE  +  ASL_WEIGHT * ASL(cls_logits, 132 binary tags)
# The MLP projector and the LLM are unchanged; only the head is new (12.8M params).
# Head input is the concatenation of BOTH vision taps: pre-merger ViT tokens grouped by
# spatial_merge_size**2 (4*1024) plus the post-merger embeds (2048) = 6144 features.
# LOSS = AR_WEIGHT * autoregressive_CE  +  ASL_WEIGHT * ASL
# Both weights are explicit because the two terms are on different scales: AR is a per-token
# mean (0.1-0.4), ASL is a SUM over 132 classes (6-11), so ASL is 20-40x larger for purely
# structural reasons. Measured at ASL_WEIGHT=1.0: loss is ~97% classification and grad_norm
# hits 260-510 vs a usual 4-8. The defaults below put the weighted terms at parity.
# NOTE: no --loss_type. The ASL term is folded into the model's loss inside forward(). Using
# swift's --loss_type hook would make transformers pop `labels`, materialize the full logits
# (~10 GB for a 20k-token sequence) and bypass Liger's fused CE — it OOMs.
#
# Data: 06_Aug_2026_cls_json_multiturn_asp_graph_report.json, built by
#   vlm_inference_analysis/scripts/build_cls_json_multiturn_data.py
# Every conversation opens with the detailed classification JSON turn; the ASP / hierarchical
# graph / report turns follow in mixed combinations, so the model learns that the
# classification always comes first and any branch may follow it.
#
# WHY THE FULL ONTOLOGY MATTERS: turn-1 targets enumerate ~112 findings each with
# `presence: true|false` (~6 true). The previous JSON-classification data was filtered to
# positives only -- 0 `false` in 2,609,050 presence fields -- which made `presence` a
# syntactic constant with ~0.00002 nats of entropy and its logprob worthless as confidence.
# With both classes present, presence becomes a real per-tag decision, which is what makes
# (a) token-weighted loss on the true/false tokens meaningful and (b) per-tag eval possible.
#
# TWO-STAGE PIPELINE — this script is stage 2. Stage 1 encodes + caches:
#   bash examples/train/training_test_split/split_qwen3_5_2b.sh     # edit its dataset path
# then point this script at <CACHED_DIR>/{train,val}. Template/tokenizer args here MUST match
# the split run, especially --new_special_tokens (the cache bakes in token ids, so the
# embedding table has to be resized identically).
#
# TOKEN WEIGHTING: --loss_scale stays `ignore_empty_think` (BINARY, so Liger's fused CE stays
# active). The true/false and JSON-format weighting is applied instead through Liger's
# `ce_weight` (qure_ce_weight_plugin.py): a vocab-sized vector gathered by target token id, so
# it costs nothing -- no logits tensor, no custom kernel, no batch-size cut.
#   ` true` = 6.0   ` false` = 1.0   JSON braces/quotes/null/attribute-keys = 0.1
# NOT covered: ASP / graph findings emphasis, which is positional rather than lexical and so
# cannot be expressed as a class weight. See qure_graded_loss_scale.py for that piece.
#
# PACKING IS OFF AND MUST STAY OFF: packing concatenates several conversations into one
# sequence, which destroys the image->row correspondence the per-row label vector relies on.
# The plugin refuses to compute ASL rather than mislabel. Expect lower throughput as a result.

SWIFT_ROOT="/fastdata4/sahil_work/VLM_qure/vlm/swift"
# the `swift` entrypoint lives in the conda env; the original scripts assumed it was already
# activated, which fails under nohup/cron
CONDA_ENV="${CONDA_ENV:-/home/ubuntu/miniconda3/envs/ms-swift-may-s4}"
export PATH="$CONDA_ENV/bin:$PATH"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export IMAGE_MAX_TOKEN_NUM=2500
export VIDEO_MAX_TOKEN_NUM=128
export FPS_MAX_FRAMES=12
# loss weights (see the plugin header for the measurements behind these)
export AR_WEIGHT="${AR_WEIGHT:-1.0}"
# ASL weight is now CONSTANT at 0.03 (set ASL_WEIGHT to pin it and bypass the schedule in the
# plugin). At 0.03 the early ASL (~12) contributes ~0.36 vs AR ~0.5 -- balanced, no early
# blow-up -- and it is the steady state the previous (decayed) run spent most of its time at
# while reaching macro AUC 0.90. The decay knobs below are left for reference but are ignored
# whenever ASL_WEIGHT is set.
export ASL_WEIGHT="${ASL_WEIGHT:-0.03}"
export ASL_WEIGHT_START="${ASL_WEIGHT_START:-1.0}"
export ASL_WEIGHT_END="${ASL_WEIGHT_END:-0.03}"
export ASL_DECAY_STEPS="${ASL_DECAY_STEPS:-800}"
export ASL_DECAY_SHAPE="${ASL_DECAY_SHAPE:-cosine}"
# head LR: the ML-Decoder is randomly initialized, so it gets its own param group far above the
# backbone (no swift flag exists for it -- see patch_head_param_group in the plugin)
export MLD_HEAD_LR="${MLD_HEAD_LR:-5e-4}"
# AR token weighting (qure_ce_weight_plugin). NOT 6:1 -- true and false are equal (1.0); only
# the JSON boilerplate (braces/keys/nulls, ~97% of the cls turn) is down-weighted to 0.5, to
# concentrate the AR gradient on the informative true/false + content tokens (~1.9x). Symmetric
# between true/false, so it does not shift the decision boundary (AUC-relevant, not a threshold nudge).
export QURE_W_TRUE="${QURE_W_TRUE:-1.0}"
export QURE_W_FALSE="${QURE_W_FALSE:-1.0}"
export QURE_W_FORMAT="${QURE_W_FORMAT:-0.5}"
export QURE_W_DEFAULT="${QURE_W_DEFAULT:-1.0}"
# augmentation (foundation_model_accelerate RandAugment); crop is 10-15% of the real H,W
export QURE_AUG_PROB="${QURE_AUG_PROB:-1.0}"
export QURE_AUG_CROP_MIN="${QURE_AUG_CROP_MIN:-0.10}"
export QURE_AUG_CROP_MAX="${QURE_AUG_CROP_MAX:-0.15}"
export FM_ACCELERATE_REPO="${FM_ACCELERATE_REPO:-/fastdata4/sahil_work/VLM_qure/vision_foundation_2d/foundation_model_accelerate}"
export PYTHONPATH="$SWIFT_ROOT:$FM_ACCELERATE_REPO:${PYTHONPATH:-}"
echo "loss = ${AR_WEIGHT}*ar + ${ASL_WEIGHT}*asl (ASL weight CONSTANT) · token weighting OFF (uniform)" >&2

export NPROC_PER_NODE=8
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

SPECIAL_TOKENS_PATH="/localstorage/sahil_work/VLM_qure/vlm/swift/examples/train/liger/token.txt"
CACHED_DIR="/fastdata4/sahil_work/VLM_qure/cached_datasets/Qwen3.5-2B_cls_json_multiturn_mldecoder"
RUN_NAME="Qwen3.5-2B_cls_json_mldecoder_joint"

# --- resolve special tokens path (/localstorage <-> /fastdata4 fallback) ---
if [[ -f "$SPECIAL_TOKENS_PATH" ]]; then
    RESOLVED_SPECIAL_TOKENS_PATH="$SPECIAL_TOKENS_PATH"
elif [[ -f "${SPECIAL_TOKENS_PATH//\/localstorage/\/fastdata4}" ]]; then
    RESOLVED_SPECIAL_TOKENS_PATH="${SPECIAL_TOKENS_PATH//\/localstorage/\/fastdata4}"
    echo "Using alternate special tokens path: $RESOLVED_SPECIAL_TOKENS_PATH" >&2
else
    echo "Special tokens file not found: $SPECIAL_TOKENS_PATH" >&2
    exit 1
fi

DATASET="/fastdata4/sahil_work/VLM_qure/data_preparation/training_data/06_Aug_2026_cls_json_multiturn_asp_graph_report.json"
if [[ ! -f "$DATASET" ]]; then
    echo "Training dataset not found: $DATASET" >&2
    exit 1
fi
if [[ ! -d "/fastdata4/sahil_work/VLM_qure/data_preparation/training_data/cls_label_sidecar" ]]; then
    echo "cls label sidecar missing — run build_cls_label_sidecar.py first" >&2
    exit 1
fi

# BATCH SIZE: with --packing false, per_device_train_batch_size counts REAL sequences, not
# packed ones. 6 x 10 = 60 seq/device/step, 480 across 8 ranks.
# Measured: 1 x 64 -> 117 s/step, 21.7 GB, 8.1-day ETA (launch-bound: 64 tiny forwards).
#            6 x 10 ->  44 s/step, 37-41 GB, 3.3-day ETA, GPU util already 100%.
#           16 x  8 ->  96 s/step, 77-84 GB, 3.3-day ETA -- NO gain over 6 x 10 (0.751 vs
#                        0.733 s/sample): the GPU was already saturated at bs=6, so extra batch
#                        only adds padding waste across the 1k-28k token spread.
# Now 28 x 5 = 140 seq/device/step, 1120 across 8 ranks. (1.75x the batch; 8/1.75 = 4.57 is not
# an integer, and 5 preserves the effective batch better than 4 would.)
# ⚠️ Measured 117-128 GB of 140 GB at 28 x 5 -- only 12-23 GB of headroom, so a micro-batch of
# 28 unusually long ASP conversations can still OOM. Two mitigations are on:
#   --group_by_length is NOT usable here: it needs a precomputed `lengths` column, which only
#       exists when the dataset is encoded up front. With --lazy_tokenize true it dies with
#       "ValueError: Column 'lengths' doesn't exist." Getting length bucketing would mean
#       running the full 3.09M-record encode first (and would then freeze augmentation).
#   --save_steps 300 (was 800) caps an OOM's cost at ~8 h of work instead of ~21 h.
# If it OOMs anyway, 16 x 8 ran clean at 77-84 GB with the same throughput.
# ⚠️ MEMORY: extrapolating 6 -> 16 puts this near 100-110 GB of 140 GB, and a micro-batch of 16
# unusually long ASP conversations can spike well past the average. If it OOMs, 8 x 16 keeps a
# similar effective batch with real margin.
# ⚠️ EFFECTIVE BATCH doubled (480 -> 1024) so the step count halves to ~3.0k. --learning_rate
# is still 1e-5, unchanged from the 480-batch setting; a 2x larger batch usually wants a larger
# LR (sqrt or linear scaling), so this run is effectively more conservative per sample.
# VAL SPLIT: 0.002 of 3.09M is ~6.2k studies, sharded across 8 ranks for eval. Larger splits
# make the every-500-step eval dominate wall-clock.
# LEARNING RATES -- deliberately spread ~250x, because the modules are in different states:
#   2e-6  LLM      converged in ckpt-5191; kept low to preserve the ASP / report / cls-token
#                  abilities the new data reuses. Measured: `ar` fell 0.74->0.53 in 20 steps at
#                  an LR of 7e-08, so the LLM is nowhere near LR-starved.
#   2e-5  ViT      needs to surface tag-relevant features; only 10x the LLM because it is SHARED
#                  and ASL is ~96% of the loss early, so a big ViT LR lets classification
#                  pressure rewrite the features the LLM depends on.
#   1e-4  aligner  the projector absorbs the encoder drift so the near-frozen LLM does not fall
#                  out of sync with shifting visual features. This is the safety valve for the
#                  high-ViT / low-LLM combination.
#   5e-4  head     randomly initialized, 12.8M params, nothing pretrained to damage.
# ASL_DECAY_STEPS is 800 (not 250) so the high-w_asl window overlaps FULL LR: warmup_ratio 0.05
# means LR only reaches 1e-5-scale at step ~138, and at 250 steps w_asl was already 0.03 by then
# -- the head would have had its strongest loss weight while the LR was ~0.
# per_device_train_batch_size counts PACKED sequences (each up to max_length=28000 tokens).
# Turn-1 targets are ~12k characters (~4k tokens) and the ASP branch adds ~18k characters, so
# these conversations are long -- keep max_length at 28000.
# RESUME=<checkpoint dir> continues an existing run (restores weights, optimizer, scheduler,
# step, RNG). Used to apply the boilerplate down-weight mid-run without losing progress. When
# resuming, --model is ignored by swift in favor of the checkpoint's weights.
RESUME_ARGS=()
if [[ -n "${RESUME:-}" ]]; then
    RESUME_ARGS=(--resume_from_checkpoint "$RESUME")
    echo "RESUMING from $RESUME" >&2
fi

swift sft \
    --model "${BASE_MODEL:-Qwen/Qwen3.5-2B}" \
    "${RESUME_ARGS[@]}" \
    --run_name "$RUN_NAME" \
    --tuner_type full \
    --dataset "$DATASET" \
    --split_dataset_ratio 0.002 \
    --lazy_tokenize true \
    --dataset_shuffle true \
    --data_seed 42 \
    --new_special_tokens "$RESOLVED_SPECIAL_TOKENS_PATH" \
    --add_non_thinking_prefix true \
    --loss_scale ignore_empty_think \
    --max_new_tokens 18000 \
    --torch_dtype bfloat16 \
    --per_device_train_batch_size 28 \
    --per_device_eval_batch_size 2 \
    --learning_rate 2e-6 \
    --vit_lr 2e-5 \
    --aligner_lr 1e-4 \
    --freeze_vit false \
    --freeze_aligner false \
    --num_train_epochs 1 \
    --gradient_accumulation_steps 5 \
    --gradient_checkpointing true \
    --eval_strategy steps \
    --eval_steps 100 \
    --save_steps 300 \
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
    --packing false \
    --padding_free false
