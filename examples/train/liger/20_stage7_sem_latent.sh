#!/usr/bin/env bash
# STAGE 7: SEM LATENT-TOKEN EMBEDDING SUPERVISION on top of stage 6b, everything else as stage 6c.
#   init      stage-6b ckpt-608 (all 87 tools; full ckpt incl. the 23 ml_decoder_head.* tensors). Override with
#             BASE_MODEL=... (e.g. the 6c ckpt-945: config/tokenizer/jinja are byte-identical, so every check holds).
#   what      K=8 reserved tokens <|sem_0|>..<|sem_7|> (examples/train/liger/token_sem.txt) sit RIGHT AFTER every
#             '<|im_start|>assistant\n' -- in training AND at inference, inserted by swift core
#             (swift/template/sem_latent.py, hooked in Template._encode_context_list, gated on the tokenizer alone).
#             Their final hidden states (they see image + prompt only: causal) are pooled by a small head
#             (sem_head.*) and regressed onto the FROZEN teacher embedding (bb_23sep, 2560-d) of that turn's
#             ground-truth report: centred cosine + soft InfoNCE against a queue of cached teacher targets.
#             Labels on SEM tokens are -100: nothing is ever generated from them; the CE loss is unchanged.
#   targets   T1 report turns only (onto / fict / medrep / template-G Findings|Summary|Impression), canonical
#             "Findings: ...\nSummary: ...\nImpression: ..." text (swift_plugins/sem_targets.py, RULE_VERSION).
#             The teacher runs OFFLINE: SEM_TARGETS_DIR is a sidecar built ONCE by
#             examples/train/liger/sem_latent/run_build_sem_targets.sh. This script REFUSES to start without a
#             complete sidecar of the right rule_version and prints the exact precompute command.
#   data      MIX picks the default DATASETS / VAL_DATASETS (+ the matching default SEM_TARGETS_DIR); explicit
#             DATASETS / VAL_DATASETS / SEM_TARGETS_DIR always win.
#             MIX=mcr (DEFAULT; the user's stage-7 rule "use only mimic chexpert and rex gradient data"):
#               $DATA_DIR/stage7_mcr/ = the 5+2 stage-6c files filtered to rows whose images ALL come from MIMIC-CXR
#               or CheXpert (same raw lines, <stem>.mcr.jsonl) + REX_TRAIN (ReXGradient-160K official TRAIN
#               split, one frontal png_hires image per study, stage-6c fict / medrep / sec_* / cls-JSON formats)
#               + REX_VAL (VALID split, <= 2,000 studies); the ReX TEST split is never used (asserted). Built by
#               examples/train/liger/sem_latent/build_stage7_mcr_data.py; counts, predicates and leakage checks in
#               stage7_mcr/MANIFEST.json + README.md. Sidecar: $DATA_DIR/sem_targets/stage7_mcr_rule_v1.
#               351,548 rows (229,476 MIMIC/CheXpert + 122,072 ReX) = 343 steps at 1024; ~737 SEM-supervised
#               (non-leaked) T1 turns per step. No positive-box rows survive (every box row used internal films):
#               the 5,018 kept B3/N7 rows only ever show `detect` returning {"boxes": []} -- see README §8 (D1).
#               Also exports SKIP_BROKEN_IMAGES=1 (unless set): every image of these files exists (checked at build
#               time), so an unreadable image is a real error -- the row is dropped and swift fetches another,
#               instead of training on swift's 512x512 random-noise placeholder.
#             MIX=6c (the previous default, unchanged): A_TRAIN_120k + B3_TRAIN (B2 positives + N negatives, i.e.
#               the tool-calling data of 6b retained) + retention 15_stage5_headguided + 17_refresher_multiturn +
#               RET_diverse60k; val A_VAL + B3_VAL. ~330 supervised report turns per 1024-row step (map_data.md:
#               326,777 T1 turns in 967,467 rows). Sidecar: $DATA_DIR/sem_targets/stage7_rule_v1. On THIS host ~75%
#               of its rows point at segmed / internal images that are not mounted (swift substitutes noise).
#   loss      AR (CE, per-message loss flags) + ASL (ML-Decoder head TRAINED, CONSTANT weight 0.05 -- the head
#             is already converged from 6b, no decay needed) + SEM (w_cos * L_cos + w_nce * L_nce, 0.05 each).
#   trained   LLM + aligner + ViT (2e-6, as 6c's actual flags) + ML-Decoder head + sem_head (own lr 2e-4).
#             The SEM embedding ROWS are part of embed_tokens and train at the LLM lr.
#   README    examples/train/liger/sem_latent/README.md (method, runbook, every knob, limitations).
#
# PLUGINS (order matters): mldecoder_joint -> qure_augment -> qure_ce_weight -> qure_stage3 -> [EXTRA_PLUGINS]
#   -> sem_latent_plugin LAST (its loss hook must run after mldecoder's; its encode/collator/post_encode
#   wrappers chain on everything installed before it).
#
# BUG FIXES NOW ACTIVE IN EVERY PLUGIN (each has an escape hatch; see the README):
#   F1 one module object per plugin (QURE_PLUGIN_DEDUP=0 = legacy double import). NOTE: with the fix, non-bbox
#      images are augmented ONCE and bbox rows get photometric-only aug. The 6b/6c runs effectively augmented
#      non-bbox images TWICE and applied GEOMETRIC aug to bbox rows. QURE_PLUGIN_DEDUP=0 reproduces 6c exactly.
#   F2 plugin mode from argv[0] (QURE_PLUGIN_MODE=train = legacy). This script refuses QURE_PLUGIN_MODE=infer.
#   F3 saved configs say tie_word_embeddings: false (SEM_UNTIE_CONFIG=0 = legacy tied config). Serving a
#      stage-7 checkpoint with sglang/vLLM therefore uses the TRAINED lm_head (6b/6c served lm_head=embed_tokens).
#
# PRECHECKS (all run before anything heavy; DRY_RUN=1 runs them and prints the swift command, launching nothing)
#   bash   : base ckpt, every dataset file (a '#N' swift sampling suffix is allowed), token files, plugin files,
#            cls label sidecar, SEM sidecar meta.json, RESUME dir, root disk >= ROOT_FREE_MIN_GB, output disk >=
#            OUT_FREE_MIN_GB, selected GPUs idle (a real launch refuses GPUs with > GPU_BUSY_MIB in use; DRY_RUN
#            only reports it), SWIFT_SEM_LATENT != 0, QURE_PLUGIN_MODE unset|train.
#   python (PRECHECK=1, ~20 s, CPU only, CUDA_VISIBLE_DEVICES=""):
#            sidecar opens with the plugin's own SidecarReader; rule_version == sem_targets.RULE_VERSION; teacher
#            fingerprint recorded (fingerprint_ok reported); every train/val file was part of the build with the
#            same size+mtime and the sidecar is not a --limit smoke build (SEM_SIDECAR_CHECK=warn downgrades these
#            two to warnings); every SEM_* knob parses (the plugin's own SemConfig); the base tokenizer + BOTH token
#            files through swift's get_processor: K contiguous <|sem_i|>, K <= 109, len(tokenizer) <= config
#            vocab_size (no embedding resize -> no lm_head re-tie); a real train-mode and sglang-mode encode of a
#            2-turn row through the working-tree swift gives exactly one SEM block per assistant header (proves the
#            core hook is live); the SEM-aware chat template converts; the checkpoint's ml_decoder_head.* /
#            sem_head.* / lm_head keys and tie flags; the FINAL swift sft argv (incl. EXTRA_ARGS) parsed by swift's
#            HfArgumentParser(SftArguments) and passed to sem_latent_plugin._check_train_args (template_backend
#            jinja, loss_scale base 'all' / non-binary, --loss_type, label smoothing, dft / channel loss, packing
#            ... are refused); the encode check uses that argv's --loss_scale.
#
# ENV (every value is ${VAR:-default}; nothing is echoed before it is assigned -- a `set -u` + echo-before-
# assignment bug aborted a past launch)
#   BASE_MODEL        stage-6b checkpoint-608 (below)          RUN_NAME   Qwen3.5-2B_stage7_sem_latent
#   MIX               mcr (default) | 6c    -> the defaults of the next three (see "data" above)
#   DATASETS          space-separated train files (default MIX=mcr: stage7_mcr/{A_TRAIN_120k,B3_TRAIN,
#                     15_stage5_headguided_TRAIN,17_refresher_multiturn_TRAIN,RET_diverse60k_TRAIN}.mcr.jsonl +
#                     REX_TRAIN.jsonl; MIX=6c: the stage-6c mixture)
#   VAL_DATASETS      space-separated val files   (MIX=mcr: A_VAL.mcr B3_VAL.mcr REX_VAL; MIX=6c: A_VAL B3_VAL)
#   SEM_TARGETS_DIR   MIX=mcr: $DATA_DIR/sem_targets/stage7_mcr_rule_v1; MIX=6c: $DATA_DIR/sem_targets/stage7_rule_v1
#                     (== run_build_sem_targets.sh's default OUT for the same MIX)
#   SKIP_BROKEN_IMAGES MIX=mcr: 1 unless set (drop a row whose image fails to load); MIX=6c: left as is (legacy)
#   SEM_SIDECAR_CHECK strict (default) | warn    coverage / --limit checks: refuse | warn
#   RESUME            checkpoint dir -> --resume_from_checkpoint     IGNORE_DATA_SKIP=1 -> --ignore_data_skip true
#                     (lazy_tokenize resumes REPLAY the skipped rows through the loader: slow; prefer
#                     IGNORE_DATA_SKIP=1 + a new DATA_SEED)
#   DATA_SEED 42  BS 8  GAS 16  EPOCHS 1  NUM_WORKERS 12  LR 4e-6  ALIGNER_LR 5e-5  VIT_LR 2e-6
#   MAX_LENGTH 28000  EVAL_STEPS 50  SAVE_STEPS 50  SAVE_TOTAL_LIMIT 3  LOGGING_STEPS 5  REPORT_TO "tensorboard wandb"
#   GPUS              0,1,2,3,4,5,6,7 (-> CUDA_VISIBLE_DEVICES; NPROC_PER_NODE defaults to its count)
#                     global batch = NPROC_PER_NODE x BS x GAS (8 x 8 x 16 = 1024, as 6b/6c)
#   OUTPUT_DIR        $DATA_DIR/output/$RUN_NAME
#   EXTRA_PLUGINS     extra --external_plugins inserted BEFORE sem_latent_plugin (it must stay last)
#   EXTRA_ARGS        appended verbatim to `swift sft` (e.g. "--max_steps 20"); it comes LAST, so it overrides the
#                     defaults above -- the python precheck parses the FINAL argv with swift's own parser and runs the
#                     SEM plugin's training refusals on it (e.g. EXTRA_ARGS="--loss_scale all" or
#                     "--template_backend jinja" is refused before anything heavy)
#   DDP_TIMEOUT       10800 s -> --ddp_timeout: the NCCL watchdog timeout that turns a rank desync into an abort
#                     (swift's default 18,000,000 s would hang ~208 days). It must exceed the longest LEGITIMATE
#                     single wait: a rank-0 checkpoint save (~41 GB with optimizer), a full eval pass, a first-time
#                     (cold Arrow cache) dataset build behind swift's rank barrier -- raise it for new DATASETS
#                     --remove_unused_columns false is added automatically when the sidecar holds sem_targets /
#                     sem_target override texts (meta n_texts_per_family.override > 0; else they would be dropped)
#   DRY_RUN=1         prechecks + print the command, launch nothing (exit 0)
#   DETACH=1          after the prechecks pass, re-run this script under `setsid nohup ... & disown` into
#                     $OUTPUT_DIR/train_<date>.log (survives the shell / Claude session) and return
#   PRECHECK=0        skip the python precheck (NOT recommended; the bash checks still run)
#   ROOT_FREE_MIN_GB 3   OUT_FREE_MIN_GB 150 (3 kept checkpoints x ~41 GB with optimizer + margin)
#   GPU_BUSY_MIB 2048    ALLOW_BUSY_GPUS=1 skips the idle-GPU check      OOM_SCORE_ADJ 1000 ('' = leave alone)
#   CACHE_ROOT        /fastdata4/sahil_work/VLM_qure/cache_stage6 (HF / datasets / triton / inductor / modelscope /
#                     xdg; the SAME cache as 6c, so the Arrow cache of these files is reused)
#   CACHE_TMPDIR      /dev/shm/cache_stage6_tmp (tmpfs: NFS TMPDIR gave Errno 16 on pymp-* semaphore cleanup)
#   CONDA_ENV         /home/ubuntu/miniconda3/envs/ms-swift-may-s4
#   loss / aug knobs  AR_WEIGHT 1.0, ASL_WEIGHT 0.05 (constant: set => the schedule is bypassed), MLD_RESTORE_HEAD 1,
#                     MLD_FREEZE_HEAD 0, QURE_* exactly as 6c
#   SEM knobs         SEM_W_COS 0.05  SEM_W_NCE 0.05  SEM_TAU 0.05  (SEM_TAU_T = SEM_TAU unless set)
#                     SEM_NCE_MODE soft  SEM_FN_THRESH 0.9  SEM_QUEUE 16384  SEM_NCE_GATHER 0  SEM_POOL attn
#                     SEM_HEAD_HIDDEN 2048  SEM_HEAD_LR 2e-4  SEM_DETACH_STEPS 20  SEM_RESTORE_HEAD 1
#                     SEM_FREEZE_HEAD 0  SEM_INIT_EMBED 1  SEM_ONLINE_TEACHER 0  SEM_SKIP_LEAKED 1
#                     SEM_SKIP_LOSS_FALSE 1  SEM_UNTIE_CONFIG 1  SEM_CHAT_TEMPLATE 1  SEM_LOG_EVERY 20
#                     SEM_PROBE_HEAD lp23sep/head_xray_bce.pt; every other SEM_* the plugin reads passes through
#                     (SEM_W_*_START/_END/_DECAY_STEPS/_DECAY_SHAPE, SEM_INIT_TOL, SEM_SEED, SEM_TEACHER_PATH, ...)
# Paths may be given under any storage root: /localstorage/... is resolved to /fastdata3/... or /fastdata4/...
# when only that one exists (the user's /localstorage/sahil_work/tag_extractor/lp23sep is /fastdata4/... here),
# exactly like swift/template/vision_utils.py and sem_targets.resolve_storage_path.
set -euo pipefail

# repo root = three levels up from this script (examples/train/liger/); override with SWIFT_ROOT=...
SWIFT_ROOT="${SWIFT_ROOT:-$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/../../.." && pwd)}"
CONDA_ENV="${CONDA_ENV:-/home/ubuntu/miniconda3/envs/ms-swift-may-s4}"
PYTHON="${PYTHON:-$CONDA_ENV/bin/python}"
export PATH="$CONDA_ENV/bin:$PATH"
SELF="$(readlink -f "${BASH_SOURCE[0]}")"
TAG='[stage7]'
DRY_RUN="${DRY_RUN:-0}"
die() { echo "$TAG ABORT: $*" >&2; exit 1; }
warn() { echo "$TAG WARNING: $*" >&2; }
info() { echo "$TAG $*" >&2; }
sq() { printf "'%s'" "${1//\'/\'\\\'\'}"; }   # single-quote for a copy-pastable command line

# ---- storage-root resolution (/localstorage <-> /fastdata3 <-> /fastdata4) ----------------------------
STORAGE_ROOTS=(/localstorage /fastdata3 /fastdata4)
resolve_path() {   # $1 -> $1 if it exists, else the same suffix under another root that exists, else $1
    local p="$1" r a suf
    if [[ -z "$p" || -e "$p" ]]; then printf '%s' "$p"; return 0; fi
    for r in "${STORAGE_ROOTS[@]}"; do
        if [[ "$p" == "$r"/* ]]; then
            suf="${p#"$r"}"
            for a in "${STORAGE_ROOTS[@]}"; do
                if [[ "$a" != "$r" && -e "$a$suf" ]]; then printf '%s' "$a$suf"; return 0; fi
            done
            break
        fi
    done
    printf '%s' "$p"
}
resolve_spec() {   # a swift dataset spec 'path[#N]' -> resolved path with the suffix kept
    local s="$1" base suffix=""
    base="${s%%#*}"
    [[ "$base" != "$s" ]] && suffix="#${s#*#}"
    printf '%s%s' "$(resolve_path "$base")" "$suffix"
}

# ---- caches OFF the 439 GB root disk (it hit 100% on the 6c launch: HF datasets Arrow cache) -----------
CACHE_ROOT="${CACHE_ROOT:-/fastdata4/sahil_work/VLM_qure/cache_stage6}"
export HF_HOME="$CACHE_ROOT/hf"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export TMPDIR="${CACHE_TMPDIR:-/dev/shm/cache_stage6_tmp}"   # tmpfs: NFS TMPDIR gave Errno 16 on pymp-* semaphore cleanup
export TRITON_CACHE_DIR="$CACHE_ROOT/triton"
export TORCHINDUCTOR_CACHE_DIR="$CACHE_ROOT/inductor"
export MODELSCOPE_CACHE="$CACHE_ROOT/modelscope"   # swift routes dataset cache here even with --use_hf
export XDG_CACHE_HOME="$CACHE_ROOT/xdg"
mkdir -p "$HF_DATASETS_CACHE" "$TMPDIR" "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR" "$MODELSCOPE_CACHE" "$XDG_CACHE_HOME"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export IMAGE_MAX_TOKEN_NUM=2500
export VIDEO_MAX_TOKEN_NUM=128
export FPS_MAX_FRAMES=12
export TOKENIZERS_PARALLELISM=false
# Robustness: if a rank ever desyncs on a collective (e.g. a corrupt image gives one rank a bad batch), ABORT on
# the watchdog timeout instead of spinning at 100% util forever (the 6c v0 run hung this way at step 85). The
# timeout itself is --ddp_timeout (DDP_TIMEOUT, below): swift's default is 18,000,000 s (~208 days), under which
# this abort never fires.
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_ASYNC_ERROR_HANDLING=1

# ---- loss configuration ----------------------------------------------------------------------------------
export AR_WEIGHT="${AR_WEIGHT:-1.0}"
export ASL_WEIGHT="${ASL_WEIGHT:-0.05}"           # CONSTANT (mldecoder: ASL_WEIGHT set => START/END schedule bypassed)
export MLD_RESTORE_HEAD="${MLD_RESTORE_HEAD:-1}"  # reload ml_decoder_head.* from BASE_MODEL
export MLD_FREEZE_HEAD="${MLD_FREEZE_HEAD:-0}"    # the ML-Decoder head keeps training (as 6b/6c)
# augmentation / CE weighting: identical to 6c (see 18_stage6c_posneg.sh for the derivations)
export QURE_AUG_SKIP_BBOX=1                       # install the bbox-safe augmentation guard
export QURE_AUG_BBOX_MODE="${QURE_AUG_BBOX_MODE:-photometric}"
export QURE_AUG_PHOTO_PROB="${QURE_AUG_PHOTO_PROB:-1.0}"
export QURE_W_TRUE="${QURE_W_TRUE:-6.0}"
export QURE_W_FALSE="${QURE_W_FALSE:-1.0}"
export QURE_W_FORMAT="${QURE_W_FORMAT:-0.5}"
export QURE_W_DEFAULT="${QURE_W_DEFAULT:-1.0}"
export QURE_W_COORD="${QURE_W_COORD:-2.0}"
export QURE_AUG_PROB="${QURE_AUG_PROB:-1.0}"
export QURE_AUG_CROP_MIN="${QURE_AUG_CROP_MIN:-0.10}"
export QURE_AUG_CROP_MAX="${QURE_AUG_CROP_MAX:-0.15}"
export FM_ACCELERATE_REPO="${FM_ACCELERATE_REPO:-/fastdata4/sahil_work/VLM_qure/vision_foundation_2d/foundation_model_accelerate}"
export PYTHONPATH="$SWIFT_ROOT:$FM_ACCELERATE_REPO:${PYTHONPATH:-}"

# ---- paths -----------------------------------------------------------------------------------------------
DATA_DIR="/fastdata4/sahil_work/VLM_qure/data_preparation/training_data"
BASE_MODEL="$(resolve_path "${BASE_MODEL:-/fastdata4/sahil_work/VLM_qure/data_preparation/training_data/output/Qwen3.5-2B_stage6b_alltools/v0-20260906-173237/checkpoint-608}")"
RUN_NAME="${RUN_NAME:-Qwen3.5-2B_stage7_sem_latent}"
OUTPUT_DIR="${OUTPUT_DIR:-$DATA_DIR/output/$RUN_NAME}"
# the data mixture (== run_build_sem_targets.sh's DEFAULT_TRAIN / DEFAULT_VAL / OUT for the same MIX: the default
# sidecar of each MIX covers exactly its default files)
MIX="${MIX:-mcr}"
MCR_DIR="$DATA_DIR/stage7_mcr"
case "$MIX" in
    mcr)
        DEFAULT_TRAIN="$MCR_DIR/A_TRAIN_120k.mcr.jsonl $MCR_DIR/B3_TRAIN.mcr.jsonl $MCR_DIR/15_stage5_headguided_TRAIN.mcr.jsonl $MCR_DIR/17_refresher_multiturn_TRAIN.mcr.jsonl $MCR_DIR/RET_diverse60k_TRAIN.mcr.jsonl $MCR_DIR/REX_TRAIN.jsonl"
        DEFAULT_VAL="$MCR_DIR/A_VAL.mcr.jsonl $MCR_DIR/B3_VAL.mcr.jsonl $MCR_DIR/REX_VAL.jsonl"
        DEFAULT_SEM_TARGETS_DIR="$DATA_DIR/sem_targets/stage7_mcr_rule_v1" ;;
    6c)
        DEFAULT_TRAIN="$DATA_DIR/stage6_toolcalling/A_TRAIN_120k.jsonl $DATA_DIR/stage6_toolcalling/B3_TRAIN.jsonl $DATA_DIR/15_stage5_headguided_TRAIN.json $DATA_DIR/17_refresher_multiturn_TRAIN.json $DATA_DIR/stage6_toolcalling/RET_diverse60k_TRAIN.jsonl"
        DEFAULT_VAL="$DATA_DIR/stage6_toolcalling/A_VAL.jsonl $DATA_DIR/stage6_toolcalling/B3_VAL.jsonl"
        DEFAULT_SEM_TARGETS_DIR="$DATA_DIR/sem_targets/stage7_rule_v1" ;;
    *) die "MIX must be mcr|6c, got '$MIX'" ;;
esac
if [[ "$MIX" == mcr && -z "${SKIP_BROKEN_IMAGES+x}" ]]; then
    export SKIP_BROKEN_IMAGES=1   # swift/template/vision_utils.py load_image: re-raise -> LazyLLMDataset drops the row
elif [[ -n "${SKIP_BROKEN_IMAGES+x}" ]]; then
    export SKIP_BROKEN_IMAGES
fi
read -r -a _TRAIN_IN <<< "${DATASETS:-$DEFAULT_TRAIN}"
read -r -a _VAL_IN <<< "${VAL_DATASETS:-$DEFAULT_VAL}"
TRAIN_FILES=(); VAL_FILES=()
for f in "${_TRAIN_IN[@]}"; do TRAIN_FILES+=("$(resolve_spec "$f")"); done
for f in "${_VAL_IN[@]}"; do VAL_FILES+=("$(resolve_spec "$f")"); done
SPECIAL_TOKENS_PATH="$SWIFT_ROOT/examples/train/liger/token.txt"
SEM_TOKENS_PATH="$SWIFT_ROOT/examples/train/liger/token_sem.txt"
PLUGIN_DIR="$SWIFT_ROOT/swift_plugins"
PLUGINS=("$PLUGIN_DIR/mldecoder_joint_plugin.py" "$PLUGIN_DIR/qure_augment_plugin.py"
         "$PLUGIN_DIR/qure_ce_weight_plugin.py" "$PLUGIN_DIR/qure_stage3_plugin.py")
read -r -a _EXTRA_PLUGINS <<< "${EXTRA_PLUGINS:-}"
for p in "${_EXTRA_PLUGINS[@]}"; do PLUGINS+=("$(resolve_path "$p")"); done
PLUGINS+=("$PLUGIN_DIR/sem_latent_plugin.py")    # LAST (loss hook after mldecoder's; wrappers chain on all others)
BUILD_SH="$SWIFT_ROOT/examples/train/liger/sem_latent/run_build_sem_targets.sh"

# ---- SEM knobs (swift_plugins/sem_latent_plugin.py header documents each) -----------------------------------
export SEM_TARGETS_DIR="$(resolve_path "${SEM_TARGETS_DIR:-$DEFAULT_SEM_TARGETS_DIR}")"
export SEM_W_COS="${SEM_W_COS:-0.05}"
export SEM_W_NCE="${SEM_W_NCE:-0.05}"
export SEM_TAU="${SEM_TAU:-0.05}"                 # SEM_TAU_T (teacher temperature) defaults to SEM_TAU in the plugin
export SEM_NCE_MODE="${SEM_NCE_MODE:-soft}"
export SEM_FN_THRESH="${SEM_FN_THRESH:-0.9}"
export SEM_QUEUE="${SEM_QUEUE:-16384}"
export SEM_NCE_GATHER="${SEM_NCE_GATHER:-0}"
export SEM_POOL="${SEM_POOL:-attn}"
export SEM_HEAD_HIDDEN="${SEM_HEAD_HIDDEN:-2048}"
export SEM_HEAD_LR="${SEM_HEAD_LR:-2e-4}"
export SEM_DETACH_STEPS="${SEM_DETACH_STEPS:-20}"
export SEM_RESTORE_HEAD="${SEM_RESTORE_HEAD:-1}"
export SEM_FREEZE_HEAD="${SEM_FREEZE_HEAD:-0}"
export SEM_INIT_EMBED="${SEM_INIT_EMBED:-1}"
export SEM_ONLINE_TEACHER="${SEM_ONLINE_TEACHER:-0}"
export SEM_SKIP_LEAKED="${SEM_SKIP_LEAKED:-1}"
export SEM_SKIP_LOSS_FALSE="${SEM_SKIP_LOSS_FALSE:-1}"
export SEM_UNTIE_CONFIG="${SEM_UNTIE_CONFIG:-1}"
export SEM_CHAT_TEMPLATE="${SEM_CHAT_TEMPLATE:-1}"
export SEM_LOG_EVERY="${SEM_LOG_EVERY:-20}"
export SEM_PROBE_HEAD="$(resolve_path "${SEM_PROBE_HEAD:-/fastdata4/sahil_work/tag_extractor/lp23sep/head_xray_bce.pt}")"
[[ -n "${SEM_TEACHER_PATH:-}" ]] && export SEM_TEACHER_PATH="$(resolve_path "$SEM_TEACHER_PATH")"
SEM_SIDECAR_CHECK="${SEM_SIDECAR_CHECK:-strict}"

# ---- GPUs / batch ----------------------------------------------------------------------------------------
GPUS="${GPUS:-0,1,2,3,4,5,6,7}"
IFS=',' read -r -a _GPU_LIST <<< "$GPUS"
export NPROC_PER_NODE="${NPROC_PER_NODE:-${#_GPU_LIST[@]}}"
BS="${BS:-8}"; GAS="${GAS:-16}"

info "stage7: AR + ASL (const ${ASL_WEIGHT}) + SEM (w_cos ${SEM_W_COS}, w_nce ${SEM_W_NCE}, ${SEM_NCE_MODE}) · base=${BASE_MODEL} · run=${RUN_NAME} · global batch $((NPROC_PER_NODE * BS * GAS)) (${NPROC_PER_NODE} GPUs x ${BS} x GAS ${GAS}) · data MIX=${MIX}${DATASETS:+ (DATASETS overridden)} (${#TRAIN_FILES[@]} train / ${#VAL_FILES[@]} val files)"

# ---- the exact precompute command for this data selection ---------------------------------------------------
precompute_cmd() {   # $1 = sidecar dir to build
    printf '%s' "mkdir -p $(sq "$1") && GPUS=$(sq "${GPUS//,/ }") OUT=$(sq "$1") DATASETS=$(sq "${TRAIN_FILES[*]}") VAL_DATASETS=$(sq "${VAL_FILES[*]}") setsid nohup bash $(sq "$BUILD_SH") > $(sq "$1/build.log") 2>&1 < /dev/null & disown"
}
PRECOMPUTE_CMD="$(precompute_cmd "$SEM_TARGETS_DIR")"
# a COMPLETE but unsuitable sidecar (wrong rule_version / other data / --limit smoke) is never overwritten by the
# builder (it exits "already complete"), so the fix is a build into a NEW directory
NEW_SIDECAR_DIR="${SEM_TARGETS_DIR%/}_rebuild_$(date +%Y%m%d)"
PRECOMPUTE_CMD_NEW="$(precompute_cmd "$NEW_SIDECAR_DIR")   # then re-run with SEM_TARGETS_DIR=$(sq "$NEW_SIDECAR_DIR")"
need_sidecar() {
    echo "$TAG ABORT: $1" >&2
    echo "$TAG build the teacher sidecar first (8 GPUs; collect -> encode shards -> merge; resumable, re-run is safe):" >&2
    echo "    $PRECOMPUTE_CMD" >&2
    echo "$TAG then re-run this script (SEM_TARGETS_DIR=$SEM_TARGETS_DIR). Progress: $SEM_TARGETS_DIR/logs/, done when $SEM_TARGETS_DIR/meta.json exists." >&2
    exit 1
}

# ---- bash prechecks ---------------------------------------------------------------------------------------
[[ "${SWIFT_SEM_LATENT:-1}" != 0 ]] || die "SWIFT_SEM_LATENT=0 disables SEM insertion in swift core; stage 7 needs it (unset it)"
case "${QURE_PLUGIN_MODE:-}" in
    ''|train) ;;
    *) die "QURE_PLUGIN_MODE=${QURE_PLUGIN_MODE} would skip every train-only plugin patch; unset it (or =train)" ;;
esac
if [[ "${QURE_PLUGIN_DEDUP:-1}" == 0 ]]; then
    warn "QURE_PLUGIN_DEDUP=0: LEGACY double plugin import (non-bbox images augmented twice, geometric aug on bbox rows -- exactly 6c)"
fi
case "$SEM_SIDECAR_CHECK" in strict|warn) ;; *) die "SEM_SIDECAR_CHECK must be strict|warn, got '$SEM_SIDECAR_CHECK'" ;; esac
[[ -d "$BASE_MODEL" ]] || die "base ckpt missing: $BASE_MODEL"
[[ -f "$BASE_MODEL/config.json" ]] || die "base ckpt has no config.json: $BASE_MODEL"
[[ ${#TRAIN_FILES[@]} -gt 0 ]] || die "DATASETS is empty"
for f in "${TRAIN_FILES[@]}" "${VAL_FILES[@]}"; do
    [[ -f "${f%%#*}" ]] || die "dataset missing: ${f%%#*}"
done
for f in "$SPECIAL_TOKENS_PATH" "$SEM_TOKENS_PATH"; do [[ -f "$f" ]] || die "token file missing: $f"; done
for p in "${PLUGINS[@]}"; do [[ -f "$p" ]] || die "plugin missing: $p"; done
[[ -f "$BUILD_SH" ]] || die "sidecar builder missing: $BUILD_SH"
[[ -d "$DATA_DIR/cls_label_sidecar" ]] || die "cls label sidecar missing: $DATA_DIR/cls_label_sidecar (mldecoder_joint_plugin's ASL labels)"
[[ -d "$SEM_TARGETS_DIR" ]] || need_sidecar "SEM_TARGETS_DIR does not exist: $SEM_TARGETS_DIR"
[[ -f "$SEM_TARGETS_DIR/meta.json" ]] || need_sidecar "no complete sidecar at $SEM_TARGETS_DIR (meta.json missing: never built, or still building -- see $SEM_TARGETS_DIR/logs/)"
if [[ "$SEM_ONLINE_TEACHER" != 0 ]]; then
    _teacher="${SEM_TEACHER_PATH:-$(resolve_path /fastdata4/sahil_work/tag_extractor/bb_23sep)}"
    [[ -d "$_teacher" ]] || die "SEM_ONLINE_TEACHER=1 but the teacher dir is missing: $_teacher (set SEM_TEACHER_PATH)"
fi
[[ -f "$SEM_PROBE_HEAD" ]] || warn "probe head missing ($SEM_PROBE_HEAD): the val_sem_probe/* diagnostic will be skipped"

RESUME_ARGS=()
if [[ -n "${RESUME:-}" ]]; then
    RESUME="$(resolve_path "$RESUME")"
    [[ -d "$RESUME" ]] || die "RESUME checkpoint missing: $RESUME"
    [[ -f "$RESUME/trainer_state.json" ]] || warn "RESUME=$RESUME has no trainer_state.json (not a full training checkpoint?)"
    RESUME_ARGS=(--resume_from_checkpoint "$RESUME")
    info "RESUMING from $RESUME"
    if [[ "${IGNORE_DATA_SKIP:-0}" == 1 ]]; then RESUME_ARGS+=(--ignore_data_skip true); fi
fi

ROOT_FREE_MIN_GB="${ROOT_FREE_MIN_GB:-3}"   # every cache and TMPDIR are redirected off root above
ROOT_FREE_GB="$(df -BG --output=avail / | tail -1 | tr -dc '0-9')"
[[ "${ROOT_FREE_GB:-0}" -ge "$ROOT_FREE_MIN_GB" ]] || die "root disk has only ${ROOT_FREE_GB} GB free (min ${ROOT_FREE_MIN_GB}); set ROOT_FREE_MIN_GB to override"
OUT_FREE_MIN_GB="${OUT_FREE_MIN_GB:-150}"
_out_probe="$OUTPUT_DIR"; while [[ ! -e "$_out_probe" ]]; do _out_probe="$(dirname "$_out_probe")"; done
OUT_FREE_GB="$(df -BG --output=avail "$_out_probe" | tail -1 | tr -dc '0-9')"
[[ "${OUT_FREE_GB:-0}" -ge "$OUT_FREE_MIN_GB" ]] || die "output disk ($_out_probe) has only ${OUT_FREE_GB} GB free (min ${OUT_FREE_MIN_GB}: ${SAVE_TOTAL_LIMIT:-3} checkpoints x ~41 GB); set OUT_FREE_MIN_GB to override"
info "disk: / ${ROOT_FREE_GB} GB free (min ${ROOT_FREE_MIN_GB}); output ${OUT_FREE_GB} GB free (min ${OUT_FREE_MIN_GB})"

if [[ "${ALLOW_BUSY_GPUS:-0}" != 1 ]]; then
    if command -v nvidia-smi > /dev/null 2>&1; then
        _busy="$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits -i "$GPUS" 2>/dev/null \
                 | awk -F', *' -v lim="${GPU_BUSY_MIB:-2048}" '$2 + 0 > lim {printf "%s(%s MiB) ", $1, $2}')" || _busy=""
        if [[ -n "$_busy" ]]; then
            if [[ "$DRY_RUN" == 1 ]]; then
                warn "GPUs in use: ${_busy}-- a real launch would ABORT here (ALLOW_BUSY_GPUS=1 to override)"
            else
                die "GPUs in use: ${_busy}(> ${GPU_BUSY_MIB:-2048} MiB). Another job is running; this script never kills anything. ALLOW_BUSY_GPUS=1 to launch anyway."
            fi
        fi
    else
        warn "nvidia-smi not found: idle-GPU check skipped"
    fi
fi

# ---- the command (built BEFORE the python precheck, which parses this exact argv) ---------------------------
read -r -a _EXTRA_ARGS <<< "${EXTRA_ARGS:-}"
read -r -a _REPORT_TO <<< "${REPORT_TO:-tensorboard wandb}"
VAL_ARGS=()
[[ ${#VAL_FILES[@]} -gt 0 ]] && VAL_ARGS=(--val_dataset "${VAL_FILES[@]}")
# sem_targets / sem_target override columns reach encode() only with --remove_unused_columns false (swift's loader
# drops non-standard columns by default). The builder embedded every override text it saw, so when the sidecar holds
# any, pass the flag (unless EXTRA_ARGS sets it); a sidecar without overrides (the default mixture) changes nothing.
RUC_ARGS=()
N_OVERRIDE="$("$PYTHON" -c 'import json, sys; m = json.load(open(sys.argv[1])); print(int((m.get("n_texts_per_family") or {}).get("override", 0) or 0))' "$SEM_TARGETS_DIR/meta.json" 2>/dev/null || echo 0)"
if [[ "${N_OVERRIDE:-0}" -gt 0 ]] && ! printf '%s\n' "${_EXTRA_ARGS[@]}" | grep -qE -- '^--remove[-_]unused[-_]columns'; then
    RUC_ARGS=(--remove_unused_columns false)
    info "sidecar holds ${N_OVERRIDE} sem_targets/sem_target override texts -> --remove_unused_columns false (else they would be ignored)"
fi
CMD=(swift sft
    --model "$BASE_MODEL"
    "${RESUME_ARGS[@]}"
    --run_name "$RUN_NAME"
    --tuner_type full
    --dataset "${TRAIN_FILES[@]}"
    "${VAL_ARGS[@]}"
    --split_dataset_ratio 0
    --lazy_tokenize true
    --dataset_shuffle true
    --data_seed "${DATA_SEED:-42}"
    --new_special_tokens "$SPECIAL_TOKENS_PATH" "$SEM_TOKENS_PATH"
    --add_non_thinking_prefix true
    --loss_scale ignore_empty_think
    --max_new_tokens 18000
    --torch_dtype bfloat16
    --per_device_train_batch_size "$BS"
    --per_device_eval_batch_size 2
    --learning_rate "${LR:-4e-6}"
    --aligner_lr "${ALIGNER_LR:-5e-5}"
    --freeze_vit false
    --vit_lr "${VIT_LR:-2e-6}"
    --freeze_aligner false
    --num_train_epochs "${EPOCHS:-1}"
    --gradient_accumulation_steps "$GAS"
    --gradient_checkpointing true
    --eval_strategy steps
    --eval_steps "${EVAL_STEPS:-50}"
    --save_steps "${SAVE_STEPS:-50}"
    --logging_steps "${LOGGING_STEPS:-5}"
    --max_length "${MAX_LENGTH:-28000}"
    --warmup_ratio 0.03
    --dataloader_num_workers "${NUM_WORKERS:-12}"
    --save_total_limit "${SAVE_TOTAL_LIMIT:-3}"
    --save_only_model false
    --output_dir "$OUTPUT_DIR"
    --system 'You are a helpful medical assistant.'
    --deepspeed zero2
    --attn_impl flash_attention_2
    --use_hf true
    --use_liger_kernel true
    --report_to "${_REPORT_TO[@]}"
    --create_checkpoint_symlink true
    --external_plugins "${PLUGINS[@]}"
    --packing false
    --padding_free false
    --ddp_timeout "${DDP_TIMEOUT:-10800}"
    "${RUC_ARGS[@]}"
    "${_EXTRA_ARGS[@]}")

# ---- python precheck (CPU only; the plugin's own reader/config + swift's own processor/template) -----------
if [[ "${PRECHECK:-1}" != 0 ]]; then
    info "python precheck (sidecar, SEM config, tokenizer, core SEM hook) ..."
    if ! CUDA_VISIBLE_DEVICES="" QURE_PLUGIN_MODE=infer \
         P_BASE="$BASE_MODEL" P_SIDECAR="$SEM_TARGETS_DIR" P_CHECK="$SEM_SIDECAR_CHECK" \
         P_TRAIN="$(printf '%s\n' "${TRAIN_FILES[@]}")" P_VAL="$(printf '%s\n' "${VAL_FILES[@]}")" \
         P_TOKFILES="$(printf '%s\n' "$SPECIAL_TOKENS_PATH" "$SEM_TOKENS_PATH")" \
         P_MAX_LENGTH="${MAX_LENGTH:-28000}" P_PRECOMPUTE="$PRECOMPUTE_CMD_NEW" \
         P_ARGV="$(printf '%s\n' "${CMD[@]:2}")" P_PLUGINS="$(printf '%s\n' "${PLUGINS[@]}")" \
         "$PYTHON" -P - <<'PY'   # -P: no cwd on sys.path -- import swift exactly as the real launch does (PYTHONPATH)
import json, os, sys
TAG = '[stage7-precheck]'
fails, warns = [], []
def info(m): print(f'{TAG} {m}', file=sys.stderr, flush=True)
def warn(m): warns.append(m); print(f'{TAG} WARNING: {m}', file=sys.stderr, flush=True)
def fail(m): fails.append(m); print(f'{TAG} FAIL: {m}', file=sys.stderr, flush=True)
lines = lambda k: [x for x in os.environ.get(k, '').split('\n') if x.strip()]
base, sdir, mode = os.environ['P_BASE'], os.environ['P_SIDECAR'], os.environ['P_CHECK']
soft = warn if mode == 'warn' else fail
sidecar_bad = False
r = None

import swift_plugins.sem_targets as st
# 1. sidecar: exactly what the plugin's get_reader() does, then the launcher-only coverage checks
try:
    r = st.SidecarReader(st.resolve_storage_path(sdir), mmap=True)
    r.check_rule_version(st.RULE_VERSION)
    m = r.meta
    if not m.get('teacher_fingerprint'):
        raise ValueError('sidecar records no teacher_fingerprint (the plugin refuses it): rebuild with build_sem_targets.py')
    info(f'sidecar {r.dir}: n={r.n:,} dim={r.dim} rule={r.rule_version} in_train={int(m.get("n_in_train", -1)):,} '
         f'teacher={m.get("teacher_path")} fp={str(m["teacher_fingerprint"])[:16]} max_length={m.get("max_length")} '
         f'created={m.get("created")} per_family={m.get("n_texts_per_family")}')
    if m.get('fingerprint_ok') is False:
        warn('sidecar was built with --allow_fingerprint_mismatch (teacher != lp23sep backbone): the probe diagnostic is invalid')
    if m.get('n_missing'):
        warn(f'sidecar merge reported n_missing={m["n_missing"]} collected texts without an embedding (training misses)')
    if m.get('limit') is not None:
        sidecar_bad = True
        soft(f'sidecar is a --limit {m["limit"]} SMOKE build (n={r.n}): almost every training lookup would miss')
    ds = m.get('datasets') or {}
    if isinstance(ds, str):
        import ast
        ds = ast.literal_eval(ds)
    def norm(p):
        p = os.path.realpath(p)
        for root in st.STORAGE_ROOTS:
            if p.startswith(root + '/'):
                return p[len(root):]
        return p
    rec = {}
    for role in ('train', 'val'):
        for d in ds.get(role) or []:
            rec.setdefault(norm(d['path']), []).append((role, d))
    for role, key in (('train', 'P_TRAIN'), ('val', 'P_VAL')):
        for spec in lines(key):
            p = spec.split('#', 1)[0]
            hits = rec.get(norm(p), [])
            ok_roles = ('train', ) if role == 'train' else ('train', 'val')
            hit = [d for rl, d in hits if rl in ok_roles]
            if not hit:
                sidecar_bad = True
                soft(f'{role} file {p} was not part of the sidecar build ({"as " + hits[0][0] if hits else "absent"}): '
                     'its report turns would miss the cache' + (' / in_train flags wrong' if hits else ''))
                continue
            s = os.stat(p)
            if not any(int(d.get('size', -1)) == s.st_size and int(d.get('mtime', -1)) == int(s.st_mtime) for d in hit):
                sidecar_bad = True
                soft(f'{role} file {p} changed since the sidecar build (size/mtime differ): rebuild the sidecar')
except Exception as e:  # noqa: BLE001
    sidecar_bad = True
    fail(f'sidecar {sdir}: {type(e).__name__}: {e}')

# 2. every SEM_* knob parses (the plugin's own SemConfig; QURE_PLUGIN_MODE=infer -> it installs nothing)
try:
    import swift_plugins.sem_latent_plugin as sp
    info(f'SEM config: {sp.CFG.summary()}')
except Exception as e:  # noqa: BLE001
    fail(f'sem_latent_plugin import / SEM_* config: {type(e).__name__}: {e}')

# 2b. the FINAL `swift sft` argv (EXTRA_ARGS comes last, so it wins) through swift's own parser, then the plugin's
#     own training refusals on it (template_backend jinja, loss_scale 'all' / non-binary, --loss_type, label
#     smoothing, dft / channel loss, packing, ...): fail here instead of after the model load
train_ls, train_backend = 'ignore_empty_think', 'swift'
try:
    from swift.utils import import_external_file
    for p in lines('P_PLUGINS'):
        if os.path.basename(p).endswith('loss_scale.py'):      # custom loss scales register on import
            import_external_file(p)
    from transformers import HfArgumentParser
    from swift.arguments import SftArguments
    from swift.utils.utils import _patch_get_type_hints
    with _patch_get_type_hints():
        parser = HfArgumentParser([SftArguments])
    try:
        ns, _unknown = parser.parse_known_args(lines('P_ARGV'))
    except SystemExit as e:
        raise ValueError(f'swift\'s argument parser rejected the command line (exit {e.code}; see the usage above)')
    train_ls, train_backend = ns.loss_scale, getattr(ns, 'template_backend', 'swift')
    try:
        from swift.loss_scale import get_loss_scale
        get_loss_scale(train_ls)
    except Exception as e:  # noqa: BLE001
        warn(f'--loss_scale {train_ls!r} is not known to swift here ({type(e).__name__}: {e}); the SEM plugin cannot '
             'check it before the run (is the plugin that registers it listed?)')
    try:
        sp._check_train_args(ns, reader=r)
        info(f'training args: loss_scale={train_ls} template_backend={train_backend} ddp_timeout={ns.ddp_timeout} '
             f'remove_unused_columns={ns.remove_unused_columns}: accepted by the SEM plugin')
    except sp.SemConfigError as e:
        fail(str(e))
except Exception as e:  # noqa: BLE001
    fail(f'final argv check: {type(e).__name__}: {e}')

# 3. tokenizer + both token files through swift's own get_processor (what `swift sft --new_special_tokens` does)
try:
    from swift.model.register import get_processor
    from swift.template import get_template
    from swift.template.sem_latent import (MAX_SEM, SEM_TOKEN_FMT, count_sem_blocks, get_sem_ids,
                                           make_sem_chat_template, get_sem_str)
    toks = []
    for p in lines('P_TOKFILES'):
        with open(p, encoding='utf-8') as f:
            toks += f.read().split()
    proc = get_processor(base, new_special_tokens=toks)
    tok = getattr(proc, 'tokenizer', proc)
    ids = get_sem_ids(tok)
    with open(os.path.join(base, 'config.json')) as f:
        cfg = json.load(f)
    tc = cfg.get('text_config') or {}
    vocab = int(tc.get('vocab_size') or cfg.get('vocab_size') or 0)
    n_tok = len(tok)
    sem_in_file = [t for t in toks if t.startswith('<|sem_')]
    if not ids:
        fail('the tokenizer has no <|sem_0|> after adding the token files: no SEM block would ever be inserted')
    else:
        K = len(ids)
        if [SEM_TOKEN_FMT.format(i) for i in range(K)] != sem_in_file[:K] or len(sem_in_file) != K:
            fail(f'token_sem.txt must list <|sem_0|>..<|sem_{{K-1}}|> in order with no gap (tokenizer K={K}, '
                 f'file has {len(sem_in_file)} sem tokens)')
        if K > MAX_SEM:
            fail(f'K={K} > {MAX_SEM} spare embedding rows: the embeddings would be resized and lm_head re-tied')
        if list(ids) != list(range(ids[0], ids[0] + K)):
            fail(f'SEM ids are not contiguous: {ids}')
        info(f'tokenizer: K={K} SEM ids {ids[0]}..{ids[-1]} | len(tokenizer)={n_tok} config vocab_size={vocab}')
    if vocab and n_tok > vocab:
        fail(f'len(tokenizer)={n_tok} > config vocab_size={vocab}: swift would RESIZE the embeddings -> HF tie_weights() '
             're-ties lm_head to embed_tokens and discards the trained lm_head')
    # 4. the core hook is live: a real encode through the working tree gives one SEM block per assistant header
    if ids:
        sys_prompt = 'You are a helpful medical assistant.'
        row = {'messages': [{'role': 'user', 'content': 'Generate a medical report of the given chest X-ray.'},
                            {'role': 'assistant', 'content': '{"Findings": "* Lungs are clear.", "Impression": "Normal."}'},
                            {'role': 'user', 'content': 'Generate the Impression of the given chest X-ray.'},
                            {'role': 'assistant', 'content': 'Normal.'}]}
        t = get_template(proc, default_system=sys_prompt, template_type='qwen3_5', max_length=int(os.environ['P_MAX_LENGTH']),
                         loss_scale=train_ls, add_non_thinking_prefix=True)   # the loss_scale the run will use
        t.set_mode('train')
        enc = t.encode(row)
        nb = count_sem_blocks(enc['input_ids'], ids)
        lab_sem = [l for i, l in zip(enc['input_ids'], enc['labels']) if i in set(ids)]
        if nb != 2 or any(l != -100 for l in lab_sem):
            fail(f'train-mode encode: {nb} SEM blocks (want 2), SEM labels {set(lab_sem)} (want -100): the core hook in '
                 'swift/template/base.py is not active -- is swift imported from this working tree?')
        ti = get_template(proc, default_system=sys_prompt, template_type='qwen3_5', enable_thinking=False)
        ti.set_mode('sglang')
        enc2 = ti.encode({'messages': row['messages'][:3]})
        nb2 = count_sem_blocks(enc2['input_ids'], ids)
        if nb2 != 2 or enc2['input_ids'][-(len(ids) + 4):-4] != list(ids):
            fail(f'sglang-mode encode: {nb2} SEM blocks (want 2: history + generation prompt, SEM before the empty think)')
        else:
            info(f'core SEM hook live: train encode 2/2 blocks (labels -100), sglang prompt 2/2 blocks ending header+SEM+empty-think')
        src = getattr(tok, 'chat_template', None)
        if isinstance(src, str) and make_sem_chat_template(src, get_sem_str(tok)) is None:
            warn('the checkpoint chat template does not match the Qwen3.5 layout: saved checkpoints will NOT carry a '
                 'SEM-aware jinja (native sglang/vLLM servers would serve without SEM)')
except Exception as e:  # noqa: BLE001
    import traceback
    traceback.print_exc()
    fail(f'tokenizer / encode check: {type(e).__name__}: {e}')

# 5. checkpoint keys and tie flags (info; the plugins act on them)
try:
    idx = os.path.join(base, 'model.safetensors.index.json')
    keys = list(json.load(open(idx))['weight_map']) if os.path.isfile(idx) else []
    n_mld = sum(k.startswith('ml_decoder_head.') for k in keys)
    n_sem = sum(k.startswith('sem_head.') for k in keys)
    has_head = any(k in ('lm_head.weight', 'language_model.lm_head.weight') for k in keys)
    tie = (cfg.get('tie_word_embeddings'), (cfg.get('text_config') or {}).get('tie_word_embeddings'))
    info(f'checkpoint: {len(keys)} keys | ml_decoder_head.* {n_mld} | sem_head.* {n_sem} '
         f'({"continuing a SEM stage: head restored" if n_sem else "first SEM stage: fresh head"}) | '
         f'lm_head.weight {"present" if has_head else "ABSENT"} | tie_word_embeddings top/text {tie[0]}/{tie[1]}'
         f'{" (saved configs will be untied: SEM_UNTIE_CONFIG=1)" if has_head and any(tie) and os.environ.get("SEM_UNTIE_CONFIG", "1") != "0" else ""}')
    if os.environ.get('MLD_RESTORE_HEAD', '1') == '1' and keys and n_mld == 0:
        warn('MLD_RESTORE_HEAD=1 but the base checkpoint has no ml_decoder_head.* tensors: the ML-Decoder head starts fresh')
    if os.environ.get('SEM_FREEZE_HEAD', '0') not in ('0', '') and n_sem == 0:
        warn('SEM_FREEZE_HEAD=1 but the base checkpoint has no sem_head.*: the LM would chase a frozen RANDOM head')
except Exception as e:  # noqa: BLE001
    warn(f'checkpoint key scan failed: {type(e).__name__}: {e}')

if sidecar_bad and fails:
    print(f'{TAG} sidecar problem. {sdir} is a COMPLETE sidecar, which the builder never overwrites; build one for THIS '
          f'data selection into a NEW directory:\n    {os.environ["P_PRECOMPUTE"]}', file=sys.stderr)
if fails:
    print(f'{TAG} {len(fails)} check(s) FAILED, {len(warns)} warning(s)', file=sys.stderr)
    sys.exit(1)
print(f'{TAG} OK ({len(warns)} warning(s))', file=sys.stderr)
PY
    then
        die "python precheck failed (see [stage7-precheck] lines above)"
    fi
else
    warn "PRECHECK=0: python precheck skipped (sidecar rule_version / coverage, tokenizer K, core hook NOT verified here)"
fi

info "env: CUDA_VISIBLE_DEVICES=$GPUS NPROC_PER_NODE=$NPROC_PER_NODE AR_WEIGHT=$AR_WEIGHT ASL_WEIGHT=$ASL_WEIGHT MLD_RESTORE_HEAD=$MLD_RESTORE_HEAD MLD_FREEZE_HEAD=$MLD_FREEZE_HEAD QURE_PLUGIN_DEDUP=${QURE_PLUGIN_DEDUP:-1} SKIP_BROKEN_IMAGES=${SKIP_BROKEN_IMAGES:-<unset>}"
info "env: $(env | grep -E '^SEM_' | sort | tr '\n' ' ')"
if [[ "$DRY_RUN" == 1 ]]; then
    info "DRY_RUN=1: every precheck passed; NOT launching. The command (run with the env above):"
    printf '%s CMD:' "$TAG"; printf ' %q' "${CMD[@]}"; printf '\n'
    exit 0
fi

if [[ "${DETACH:-0}" == 1 ]]; then
    mkdir -p "$OUTPUT_DIR"
    LOG="$OUTPUT_DIR/train_$(date +%Y%m%d_%H%M%S).log"
    DETACH=0 PRECHECK=0 ALLOW_BUSY_GPUS=1 setsid nohup bash "$SELF" > "$LOG" 2>&1 < /dev/null &
    _pid=$!
    disown
    info "detached: pid $_pid, log $LOG (prechecks passed here; the child skips the python precheck)"
    exit 0
fi

export CUDA_VISIBLE_DEVICES="$GPUS"
# the OOM killer should take training before the box / the session (inherited by torchrun, ranks, workers)
if [[ -n "${OOM_SCORE_ADJ-1000}" ]]; then echo "${OOM_SCORE_ADJ-1000}" > "/proc/$$/oom_score_adj" 2>/dev/null || true; fi
printf '%s CMD:' "$TAG" >&2; printf ' %q' "${CMD[@]}" >&2; printf '\n' >&2
"${CMD[@]}"
