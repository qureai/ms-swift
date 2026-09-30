#!/usr/bin/env python
"""SEM latent-token embedding supervision (stage 7): the K ``<|sem_i|>`` hidden states that swift core
puts right after every assistant header are pooled by a small head and regressed onto the frozen
teacher embedding (bb_23sep, 2560-d) of that turn's ground-truth report.

Load with (LAST in the list, after qure_stage3_plugin):
    --new_special_tokens examples/train/liger/token.txt examples/train/liger/token_sem.txt
    --external_plugins .../mldecoder_joint_plugin.py .../qure_augment_plugin.py
                       .../qure_ce_weight_plugin.py .../qure_stage3_plugin.py
                       .../sem_latent_plugin.py
    --packing false --padding_free false          # REQUIRED (per-row block <-> target alignment)
    SEM_TARGETS_DIR=<sidecar built by examples/train/liger/sem_latent/run_build_sem_targets.sh>

WHAT IS ALREADY DONE ELSEWHERE (no plugin needed)
  * swift/template/sem_latent.py + the Template._encode_context_list hook insert the SEM block
    after every '<|im_start|>assistant\\n' in EVERY mode, gated on the tokenizer alone. This plugin
    never inserts tokens; it only supervises them. Inference needs nothing from here.
  * swift_plugins/sem_targets.py is THE rule for "which turn is a report and what text does the
    teacher embed" (shared byte-for-byte with the offline sidecar builder, RULE_VERSION-checked).

DATA PATH (train only; every piece chained on whatever earlier plugins installed)
  Qwen3_5Template.encode (wrapper, *args/**kwargs)   raw row dict -> thread-local pending state:
        image identity key (full normalised path of the first image, sem_targets.image_key) and the
        explicit override columns `sem_targets` (list aligned with the merged assistant turns,
        str|null) / `sem_target` (str -> last turn). Cleared in `finally` (never leaks to the next
        row, retry-safe, thread-safe).
  Template._encode (BASE class wrapper; Qwen3VLTemplate._encode calls Template._encode(self, ..)
        dynamically, so this runs under mldecoder's _encode wrapper too) -- AFTER the original ran,
        `inputs.messages` are swift's MERGED messages (tool rounds merged into one list-content
        assistant turn = one header = one SEM block). Per assistant message j (= SEM block j):
            target = override[j] or sem_targets.turn_target(last user prompt, content)
            unsupervised when: not a T1 report | final element loss:false (SEM_SKIP_LOSS_FALSE) |
            leaked (SEM_SKIP_LEAKED: Findings already verbatim in system + earlier messages; the
            turn's own tool trace comes AFTER the SEM block and is not "earlier") | key not in the
            sidecar (cache miss) | #SEM blocks in input_ids != #assistant messages (a literal header
            inside user/tool text, ...).
        Lookup: a lazily-constructed memory-mapped sem_targets.SidecarReader per process (fork
        shares the pages; np.searchsorted on keys; never a per-row file open).
        Emits (no key ends in 'length'; fresh tensors -- encode runs under inference_mode):
            sem_tgt  f16  [n_blk, D]  teacher target (zeros when not valid)
            sem_ok   bool [n_blk]     st == VALID
            sem_st   i8   [n_blk]     0 none | 1 valid | 2 miss | 3 leaked | 4 loss:false |
                                      5 block/turn mismatch | 6 target error
            sem_key  i64  [n_blk]     uint64 target key viewed as int64 (queue / FN masks / eval)
            sem_img  i64  [n_blk]     uint64 image key viewed as int64 (hard-mode FN mask; 0 = none)
            sem_txt  list[str]        canonical text of cache misses (only with SEM_ONLINE_TEACHER=1)
  Qwen3_5Template._data_collator (after mldecoder's): concatenates rows in row order and adds
        sem_nblk [B]. It RECOUNTS complete blocks per row in the final (possibly truncated)
        input_ids; a row whose count disagrees keeps its blocks but loses supervision (st=5), so the
        batch is always self-consistent.
  Qwen3_5Template._post_encode (after mldecoder's): pops every sem_* key, finds complete blocks in
        the padded inputs['input_ids'] (input_ids are dropped before model.forward), verifies the per
        row counts against sem_nblk (mismatch -> the whole batch unsupervised, warned once), fills
        cache misses with the online teacher (optional), and stashes ONE `_sem_batch` object on the
        model. Every _post_encode replaces it; the loss hook pops it -> stale state can never reach
        another micro-batch or eval.

MODEL (TunerMixin.prepare_model wrapper, getattr(..., '__func__', ...) pattern; before the optimizer)
  attach(model, template): startup checks, model.sem_head = SemanticHead, restore/freeze, SEM row
  init, F3 untie, SEM chat template, and two hooks:
    * forward hook on model.model.language_model (Qwen3_5TextModel): gathers
      last_hidden_state[_sem_pos] -> [n_blk, K, 2048] (post final norm; fires once per micro-step
      and carries grad under reentrant checkpointing -- decoder-layer hooks would not);
    * top-level forward hook (with_kwargs), registered AFTER mldecoder's (this wrapper calls the
      original first), adds the SEM term to out.loss when out.loss is not None. mldecoder therefore
      logs `ar` WITHOUT the SEM term. swift then rescales out.loss by the micro-batch's label share
      exactly as it does for ASL.
  SemanticHead: LayerNorm(2048) -> pool over K (attn: learned query, zero-init = mean | mean |
  concat) -> Linear(., SEM_HEAD_HIDDEN) -> SiLU -> Linear(., 2560) -> L2 norm (fp32). Params in
  SEM_HEAD_DTYPE (bf16 under ZeRO-2); loss math in fp32. Param names are `sem_head.{norm,pool_query,
  fc1,fc2}.*`: none contains a substring sglang renames (q_proj/k_proj/v_proj/gate_proj/up_proj/
  in_proj_/visual/language_model/embed_tokens), so servers skip them as unknown weights.

LOSS (fp32)
  z^ = head output (unit norm, UNcentred teacher space); z* = teacher target; mu = sidecar mean
  (fixed, |mu|~0.7: the space is anisotropic, a constant predictor already gets raw cos 0.72).
  c(v) = normalize(v - mu).
      L_cos = mean_valid(1 - <c(z^), c(z*)>)
      C     = [c(z*) of this micro-batch's valid blocks] ++ queue (per-rank FIFO of past c(z*) with
              their target/image keys, no grad) ++ other ranks' c(z*) (SEM_NCE_GATHER=1 only)
      soft (default): p = softmax(c(z*) C^T / tau_t) (no grad);  CE = -sum p log_softmax(c(z^) C^T / tau)
                      L_nce = KL = CE - H(p): the same gradient as CE (H(p) has none) but 0 at the optimum,
                      whereas CE carries H(p), which grows with the candidate count (queue fill) and
                      would drift loss / eval_loss / sem_share upward on its own. CE is logged as
                      train/sem_nce_ce. (normal films collapse: raw normal-normal cos 0.857)
      hard          : InfoNCE, positive = own column; negatives masked when same target key, same
                      image key (!= 0) or teacher centred cos > SEM_FN_THRESH
      total = out.loss + w_cos * L_cos + w_nce * L_nce       (weights scheduled on the optimizer step)
  SEM_DETACH_STEPS: while step < N the gathered hidden states are .detach()-ed (head-only warm-up).
  ZeRO-2 hygiene: a micro-batch with SEM blocks but 0 valid targets adds 0 * z^.sum(); one with no
  blocks (or no stash) adds 0 * head(zeros[1, K, H]).sum() -> head grads always defined AND ready in the
  same hook order as a real batch (ZeRO-2 reduces its buckets position-wise across ranks; a plain
  0 * sum(p.sum()) fires them in another order and mixes up sem_head grads between ranks).
  Eval forwards add the same terms WITHOUT the training queue (candidates = the eval micro-batch only,
  so eval_loss does not depend on the training-queue fill; no push, no gather) and record (z^, z*) for
  on_evaluate (whole-eval-set retrieval: val_sem/*).

OPTIMIZER  SwiftMixin.create_optimizer wrapper (chained after mldecoder's): identity-guarded
  add_param_group for sem_head at SEM_HEAD_LR, weight_decay 0, initial_lr set, tagged sem_head=True
  (create_optimizer runs twice). With HF's default optimizer (no --vit_lr / --aligner_lr) the head is
  already in the decay / no-decay groups: its params are MOVED into the SEM group (warned), else
  SEM_HEAD_LR / wd 0 would be silently ignored. A frozen head (SEM_FREEZE_HEAD=1) -> no group.

EMBEDDING ROWS (SEM_INIT_EMBED=1)  The 109 spare rows of the Qwen3.5-2B checkpoints are ONE
  untrained vector up to bf16 rounding (ckpt-608: 6 distinct bf16 patterns, max |delta| 1.2e-4, i.e.
  0.009 x the median per-dim std of the trained rows -- NOT bit-identical). If the K SEM rows (+ the
  remaining spare rows) of embed_tokens spread by <= SEM_INIT_TOL x that std, row i := mean(trained
  rows) + N(0, 1) * std(trained rows) * SEM_INIT_NOISE (per dim; float64 on CPU, torch.Generator
  seeded with SEM_SEED -> identical on every rank, no collective). lm_head rows are left unchanged.
  Already-distinct rows (a SEM checkpoint: ~SEM_INIT_NOISE x std apart) are never touched.

RESTORE (SEM_RESTORE_HEAD=1)  HF drops `sem_head.*` as unexpected keys on load, so after attach the
  tensors are read back from model.model_dir with safetensors.safe_open (only shards the index names
  for sem_head.*, only those keys -- never the 5 GB of LM weights). None found -> info (first SEM
  stage). A shape/key mismatch (pool/K/hidden changed) raises instead of silently training a random
  head; SEM_RESTORE_HEAD=0 starts fresh.

STARTUP CHECKS (train, at prepare_model)  tokenizer has <|sem_0|> (else raise with the exact
  --new_special_tokens fix); K <= 109 and every SEM id < embedding rows; embedding rows == the
  checkpoint's config vocab_size (a resize re-ties lm_head to embed_tokens and discards the trained
  lm_head); sidecar meta rule_version == sem_targets.RULE_VERSION and a teacher fingerprint recorded.
  _check_train_args REFUSES: --packing / --padding_free / sequence parallel; --template_backend jinja
  (whole text at loss scale 1: no SEM inserted, or SEM ids labelled); a --loss_scale with base
  strategy 'all' (headers trained -> no SEM in training, SEM at inference); a non-binary --loss_scale
  (qure_graded, hermes, ...), --loss_type, --label_smoothing_factor > 0, --enable_dft_loss,
  --enable_channel_loss (swift pops `labels`, out.loss is None, the SEM term has nowhere to go -- the
  loss hook also raises if a TRAINING micro-batch with valid blocks ever returns loss=None). WARNS when
  the sidecar holds override texts but --remove_unused_columns is true. Logs K, ids, sidecar size,
  lm_head tie state.

SAVE HYGIENE  the SEM-aware chat_template is installed on template.processor (+ its tokenizer) at
  prepare_model and again at trainer init (SEM_CHAT_TEMPLATE), so every checkpoint ships a jinja that
  native sglang/vLLM servers render with SEM; SEM_UNTIE_CONFIG applies F3 (qure_untie_config_plugin.
  untie_tied_config) to the in-memory config so saved configs say tie_word_embeddings: false. The
  function is found as swift_plugins.qure_untie_config_plugin, else the already-loaded plugin, else the
  sibling file under a private module name (no patch installed); if none works, attach RAISES.

LOGGING  (trainer.log in on_step_end every logging_steps, control.should_log restored like mldecoder;
  rank-0-local running means over training micro-batches)
    train/sem_cos_loss train/sem_nce_loss (soft: == KL) train/sem_kl train/sem_nce_ce (soft: the CE)
    train/sem_cos_centred train/sem_cos_raw
    train/sem_valid_frac (valid / SEM blocks) train/sem_miss_rate (misses / (valid + misses), before
    any online fill) train/sem_leak_frac train/sem_mismatch_frac train/sem_online_frac
    train/sem_w_cos train/sem_w_nce train/sem_share (% of the total loss) train/sem_queue_fill
    train/sem_detached
  EVAL (on_evaluate; all_gather_object across ranks, every rank participates even when empty):
    val_sem/{cos_centred, cos_raw, cos_raw_meanpred (constant mu predictor), r1, r5, r10,
    median_rank, margin (raw, RL-style: cos to own - mean cos to the other eval targets), margin_centred,
    n, n_gallery} over all eval targets, and the same under val_sem_unseen/ for targets whose text is
    not in a --train_datasets file (sidecar in_train == 0). R@k is duplicate-aware: the gallery is the
    unique eval targets and a hit is any target with the same key or teacher centred cos >
    SEM_DUP_THRESH. Probe diagnostic: lp23sep head_xray_bce.pt on UNcentred z^ vs labels
    probe(z*) >= thr -> val_sem_probe/macro_auc over tags with n_pos >= SEM_PROBE_MIN_POS,
    val_sem_probe/n_tags_scored. Rank 0 dumps eval_predictions/sem_step{N:06d}.npz (guarded).

ENV (prefix SEM_)
    SEM_TARGETS_DIR        required for training: the teacher-target sidecar directory
    SEM_W_COS / SEM_W_NCE  0.05 / 0.05 constant weights
    SEM_W_COS_START|_END|_DECAY_STEPS|_DECAY_SHAPE  optional schedule (like ASL): with _START set,
    SEM_W_NCE_START|_END|_DECAY_STEPS|_DECAY_SHAPE  w = END + (START-END)*f(step/DECAY_STEPS), END
                           afterwards; END defaults to the constant, DECAY_STEPS 1000, SHAPE cosine
                           (cosine | linear | exp)
    SEM_TAU (0.05)  SEM_TAU_T (= SEM_TAU)  SEM_NCE_MODE (soft | hard)  SEM_FN_THRESH (0.9, hard mode)
    SEM_QUEUE (16384; 0 = in-batch candidates only)   SEM_NCE_GATHER (0: no collectives in the loss;
                           1: all_gather other ranks' c(z*) as extra candidates every train micro-step)
    SEM_POOL (attn | mean | concat)   SEM_HEAD_HIDDEN (2048)   SEM_HEAD_LR (2e-4)
    SEM_HEAD_DTYPE (bfloat16)         SEM_DETACH_STEPS (20)
    SEM_RESTORE_HEAD (1)  SEM_FREEZE_HEAD (0)  SEM_INIT_EMBED (1)  SEM_INIT_NOISE (0.5)  SEM_SEED (1234)
    SEM_INIT_TOL (0.05: "fresh spare rows" spread threshold, x median trained-row std)
    SEM_ONLINE_TEACHER (0: misses are masked + counted; 1: encode misses with the teacher on the rank
                           device inside _post_encode -- the teacher is kept OFF the module tree, loaded
                           with the builder's own Teacher class, fingerprint-checked against the sidecar)
    SEM_ONLINE_CACHE (50000 embeddings kept per rank)
    SEM_PROBE_HEAD (lp23sep/head_xray_bce.pt)  SEM_PROBE_MIN_POS (50)  SEM_DUP_THRESH (0.95)
    SEM_SKIP_LEAKED (1)   SEM_SKIP_LOSS_FALSE (1: a turn whose final answer is loss:false is unsupervised)
    SEM_LOG_EVERY (20 micro-steps, logger line)   SEM_EVAL_PRED_DIR (default <output_dir>/eval_predictions)
    SEM_UNTIE_CONFIG (1)  SEM_CHAT_TEMPLATE (1)
  Also read: SEM_TEACHER_PATH (online teacher; default = sidecar meta teacher_path),
  QURE_PLUGIN_MODE / QURE_PLUGIN_DEDUP (qure_plugin_mode).

MODE  train-only patches, skipped when qure_plugin_mode.is_inference() (swift infer/deploy/...). The
  core insertion needs no plugin, so a SEM checkpoint serves correctly without this file.
ONE COPY PER PROCESS  dedup_module first (F1), every patch once()-guarded ('sem_latent_plugin.<fn>').

KNOWN LIMITATIONS
  * Qwen3.5 template only for the encode wrapper (image keys); another template class still gets
    targets through the base _encode wrapper but image keys are 0 (hard-mode image masking off).
  * packing / padding_free / sequence parallel are refused; --cached_dataset must be re-exported
    with this plugin loaded (targets are computed in encode).
  * A multi-image row uses its FIRST image as the identity (all T1 turns are single-image rows).
  * The queue is per rank and is not checkpointed (a resume starts it empty).
  * The `sem_targets` / `sem_target` override columns reach encode() only with
    --remove_unused_columns false: swift's loader keeps only its standard columns by default
    (swift/dataset/loader.py:66 -> RowPreprocessor.remove_useless_columns). Without it the rule applies
    (warned at prepare_model when the sidecar holds override texts; the stage-7 launcher then passes
    --remove_unused_columns false itself).
"""
# F1 -- must stay the first statements of the module (qure_plugin_mode.dedup_module)
try:
    from swift_plugins.qure_plugin_mode import dedup_module as _dedup_module, once as _once
except ImportError:  # repo root not on sys.path; swift put this plugin's own dir there
    from qure_plugin_mode import dedup_module as _dedup_module, once as _once
_dedup_module(__name__, 'sem_latent_plugin')

import collections  # noqa: E402
import glob  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import os  # noqa: E402
import sys  # noqa: E402
import threading  # noqa: E402
from types import SimpleNamespace  # noqa: E402
from typing import Any, Dict, List, Optional, Sequence, Tuple  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from swift.template.sem_latent import (MAX_SEM, count_sem_blocks, get_sem_ids,  # noqa: E402
                                       install_sem_chat_template)
from swift.utils import get_logger  # noqa: E402

try:
    import swift_plugins.sem_targets as st  # noqa: E402
except ImportError:  # pragma: no cover - plugin dir on sys.path, repo root not
    import sem_targets as st  # noqa: E402

logger = get_logger()

# =========================================================================================
# config
# =========================================================================================
_FALSE = ('0', 'false', 'no', 'off')


def _env(name: str, default: str) -> str:
    v = os.environ.get(name)
    return default if v is None or not v.strip() else v.strip()


def _flag(name: str, default: str) -> bool:
    return _env(name, default).lower() not in _FALSE


class _Sched:
    """Constant weight SEM_W_X, or (SEM_W_X_START set) END + (START-END)*f(step/DECAY_STEPS)."""

    def __init__(self, prefix: str, const_default: str):
        self.prefix = prefix
        self.const = float(_env(prefix, const_default))
        s = os.environ.get(prefix + '_START')
        self.start = float(s) if s is not None and s.strip() else None
        self.end = float(_env(prefix + '_END', repr(self.const)))
        self.steps = int(_env(prefix + '_DECAY_STEPS', '1000'))
        self.shape = _env(prefix + '_DECAY_SHAPE', 'cosine').lower()

    def __call__(self, step: int) -> float:
        if self.start is None:
            return self.const
        if self.steps <= 0 or step >= self.steps or self.start == self.end:
            return self.end
        t = max(step, 0) / self.steps
        if self.shape == 'exp' and self.start > 0 and self.end > 0:
            return self.start * (self.end / self.start)**t
        f = (1.0 - t) if self.shape in ('linear', 'exp') else 0.5 * (1.0 + math.cos(math.pi * t))
        return self.end + (self.start - self.end) * f

    def describe(self) -> str:
        if self.start is None:
            return f'{self.const:g}'
        return f'[{self.start:g}->{self.end:g} over {self.steps} steps, {self.shape}]'


class SemConfig:
    """Every SEM_* knob (see the module header). Re-read with reload_config() (tests)."""

    def __init__(self):
        self.targets_dir = os.environ.get('SEM_TARGETS_DIR') or ''
        self.w_cos = _Sched('SEM_W_COS', '0.05')
        self.w_nce = _Sched('SEM_W_NCE', '0.05')
        self.tau = float(_env('SEM_TAU', '0.05'))
        self.tau_t = float(_env('SEM_TAU_T', repr(self.tau)))
        self.nce_mode = _env('SEM_NCE_MODE', 'soft').lower()
        self.fn_thresh = float(_env('SEM_FN_THRESH', '0.9'))
        self.queue = int(_env('SEM_QUEUE', '16384'))
        self.nce_gather = _flag('SEM_NCE_GATHER', '0')
        self.pool = _env('SEM_POOL', 'attn').lower()
        self.head_hidden = int(_env('SEM_HEAD_HIDDEN', '2048'))
        self.head_lr = float(_env('SEM_HEAD_LR', '2e-4'))
        self.head_dtype = getattr(torch, _env('SEM_HEAD_DTYPE', 'bfloat16'))
        self.detach_steps = int(_env('SEM_DETACH_STEPS', '20'))
        self.restore_head = _flag('SEM_RESTORE_HEAD', '1')
        self.freeze_head = _flag('SEM_FREEZE_HEAD', '0')
        self.init_embed = _flag('SEM_INIT_EMBED', '1')
        self.init_noise = float(_env('SEM_INIT_NOISE', '0.5'))
        self.init_tol = float(_env('SEM_INIT_TOL', '0.05'))
        self.seed = int(_env('SEM_SEED', '1234'))
        self.online_teacher = _flag('SEM_ONLINE_TEACHER', '0')
        self.online_cache = int(_env('SEM_ONLINE_CACHE', '50000'))
        self.probe_head = os.environ.get('SEM_PROBE_HEAD') or ''
        self.probe_min_pos = int(_env('SEM_PROBE_MIN_POS', '50'))
        self.dup_thresh = float(_env('SEM_DUP_THRESH', '0.95'))
        self.skip_leaked = _flag('SEM_SKIP_LEAKED', '1')
        self.skip_loss_false = _flag('SEM_SKIP_LOSS_FALSE', '1')
        self.log_every = max(int(_env('SEM_LOG_EVERY', '20')), 1)
        self.eval_pred_dir = os.environ.get('SEM_EVAL_PRED_DIR') or ''
        self.untie_config = _flag('SEM_UNTIE_CONFIG', '1')
        self.chat_template = _flag('SEM_CHAT_TEMPLATE', '1')
        if self.nce_mode not in ('soft', 'hard'):
            raise ValueError(f'[sem-latent] SEM_NCE_MODE must be soft|hard, got {self.nce_mode!r}')
        if self.pool not in ('attn', 'mean', 'concat'):
            raise ValueError(f'[sem-latent] SEM_POOL must be attn|mean|concat, got {self.pool!r}')
        if self.tau <= 0 or self.tau_t <= 0:
            raise ValueError('[sem-latent] SEM_TAU / SEM_TAU_T must be > 0')

    def summary(self) -> str:
        return (f'w_cos={self.w_cos.describe()} w_nce={self.w_nce.describe()} mode={self.nce_mode} '
                f'tau={self.tau:g} tau_t={self.tau_t:g} fn_thresh={self.fn_thresh:g} queue={self.queue} '
                f'gather={int(self.nce_gather)} pool={self.pool} head_hidden={self.head_hidden} '
                f'head_lr={self.head_lr:g} detach_steps={self.detach_steps} online={int(self.online_teacher)} '
                f'skip_leaked={int(self.skip_leaked)} skip_loss_false={int(self.skip_loss_false)}')


CFG = SemConfig()


def reload_config() -> SemConfig:
    """Re-read every SEM_* env var (the plugin reads them once at import; tests change them)."""
    global CFG
    CFG = SemConfig()
    _STATE['queue'] = None
    return CFG


# =========================================================================================
# process state
# =========================================================================================
ST_NONE, ST_VALID, ST_MISS, ST_LEAK, ST_LOSSFALSE, ST_MISMATCH, ST_ERROR = range(7)
ST_NAMES = ('none', 'valid', 'miss', 'leaked', 'loss_false', 'mismatch', 'error')
SEM_KEYS = ('sem_tgt', 'sem_ok', 'sem_st', 'sem_key', 'sem_img', 'sem_nblk', 'sem_txt')

# optimizer step, published by the callback (a forward counter would count micro-batches)
_SCHED = {'step': 0, 'total': None}
_STATE: Dict[str, Any] = {
    'enabled': True,          # test hook: False -> both hooks pass through (stash still popped)
    'keep_term': False,       # test hook: keep the last SEM term WITH its graph in _STATE['last_term']
    'last_term': None,
    'last': {},               # float stats of the last micro-batch (always kept, cheap)
    'reader': None, 'reader_dir': None,
    'mu': {},                 # device -> fp32 mu (kept OFF the module tree: DeepSpeed would cast it)
    'queue': None,
    'teacher': None,
    'online_cache': collections.OrderedDict(),
    'probe': None, 'probe_path': None,
    'warned': set(),
    'n_micro': 0,
    'model': None,            # the attached model (for the callback)
}
_TLS = threading.local()
_RUN = collections.defaultdict(float)
_EVAL: Dict[str, list] = {'zh': [], 'zt': [], 'key': [], 'img': [], 'st': []}


def set_enabled(flag: bool) -> None:
    """TEST HOOK: disable the SEM loss without removing hooks (the stash is still consumed)."""
    _STATE['enabled'] = bool(flag)


def _warn_once(key: str, msg: str) -> None:
    if key in _STATE['warned']:
        return
    _STATE['warned'].add(key)
    logger.warning(msg)


_MISSING_DIR_MSG = (
    '[sem-latent] SEM_TARGETS_DIR is not set / has no complete sidecar. Build it first:\n'
    '    GPUS="0 1 2 3 4 5 6 7" OUT=<dir> bash examples/train/liger/sem_latent/run_build_sem_targets.sh\n'
    '  then export SEM_TARGETS_DIR=<dir>.')


class SemConfigError(RuntimeError):
    """A configuration problem that must stop training (never swallowed by the encode wrapper)."""


def get_reader(required: bool = True) -> Optional['st.SidecarReader']:
    """The process-wide memory-mapped sidecar (lazy; fork shares the pages with the workers)."""
    d = CFG.targets_dir
    r = _STATE['reader']
    if r is not None and _STATE['reader_dir'] == d:
        return r
    if not d:
        if required:
            raise SemConfigError(_MISSING_DIR_MSG)
        return None
    path = st.resolve_storage_path(d)
    try:
        r = st.SidecarReader(path, mmap=True)
    except FileNotFoundError as e:
        raise SemConfigError(f'{_MISSING_DIR_MSG}\n  ({e})') from e
    try:
        r.check_rule_version(st.RULE_VERSION)
    except ValueError as e:
        raise SemConfigError(str(e)) from e
    if not r.meta.get('teacher_fingerprint'):
        raise SemConfigError(f'[sem-latent] sidecar {path} records no teacher_fingerprint: rebuild it with '
                             'build_sem_targets.py (the fingerprint ties the targets to one teacher)')
    _STATE['reader'] = r
    _STATE['reader_dir'] = d
    _STATE['mu'] = {}                       # mu / online targets belong to ONE sidecar
    _STATE['online_cache'].clear()
    return r


def _mu(device) -> torch.Tensor:
    key = str(device)
    m = _STATE['mu'].get(key)
    if m is None:
        m = torch.from_numpy(np.asarray(get_reader().mu, dtype=np.float32).copy()).to(device)
        _STATE['mu'][key] = m
    return m


def _u64_to_i64(a) -> np.ndarray:
    return np.ascontiguousarray(np.asarray(a, dtype=np.uint64)).view(np.int64)


def _i64_to_u64(a) -> np.ndarray:
    return np.ascontiguousarray(np.asarray(a, dtype=np.int64)).view(np.uint64)


# =========================================================================================
# encode side: per-row targets (pure, dataloader workers)
# =========================================================================================
def _path_of(x):
    if isinstance(x, str):
        return x
    if isinstance(x, dict):
        return x.get('path')
    return getattr(x, 'filename', None)


def pending_from_raw(row: Dict[str, Any]) -> Dict[str, Any]:
    """What the encode wrapper takes from the RAW row: image identity + explicit overrides."""
    v = row.get('images')
    if v is None:
        v = row.get('image')
    cand = [v] if v is not None and not isinstance(v, (list, tuple)) else list(v or [])
    paths = [p for p in (_path_of(x) for x in cand) if isinstance(p, str) and p]
    return {'img': st.image_key(paths[0]) if paths else 0,
            'sem_targets': row.get('sem_targets'), 'sem_target': row.get('sem_target')}


def _final_loss_false(m: Dict[str, Any]) -> bool:
    loss = m.get('loss')
    if isinstance(loss, (list, tuple)):
        loss = loss[-1] if loss else None
    return loss is False


def _overrides(n: int, sem_targets: Any, sem_target: Any) -> List[Optional[str]]:
    ov: List[Optional[str]] = [None] * n
    if isinstance(sem_targets, (list, tuple)):
        for j, t in enumerate(sem_targets[:n]):
            ov[j] = st.normalise_override(t)
    one = st.normalise_override(sem_target)
    if one is not None and n:
        ov[-1] = one
    return ov


def row_block_targets(messages: Sequence[Dict[str, Any]], system: Optional[str], n_blocks: int,
                      img_key: int = 0, sem_targets: Any = None, sem_target: Any = None,
                      reader=None, cfg: Optional[SemConfig] = None, dim: Optional[int] = None) -> Dict[str, Any]:
    """Per SEM block of one row (block j = merged assistant message j):
    {'tgt' f16[n,D], 'st' i8[n], 'key' u64[n], 'img' u64[n], 'txt' [str], 'fam' [str|None]}."""
    cfg = cfg or CFG
    reader = reader if reader is not None else get_reader()
    dim = dim or reader.dim
    n = int(n_blocks)
    out = {'tgt': np.zeros((n, dim), np.float16), 'st': np.zeros(n, np.int8), 'key': np.zeros(n, np.uint64),
           'img': np.full(n, np.uint64(img_key or 0), np.uint64), 'txt': [''] * n, 'fam': [None] * n}
    asst = [m for m in messages if m.get('role') == 'assistant']
    if len(asst) != n:
        out['st'][:] = ST_MISMATCH
        return out
    ov = _overrides(n, sem_targets, sem_target)
    prior: List[Any] = [system] if system else []
    last_user = None
    cand: List[Tuple[int, str]] = []
    j = 0
    for m in messages:
        role, content = m.get('role'), m.get('content')
        if role == 'assistant':
            if ov[j] is not None:
                out['fam'][j] = st.OVERRIDE_FAMILY
                cand.append((j, ov[j]))
            else:
                tt = st.turn_target(last_user if last_user is not None else '', content)
                if tt is not None:
                    fam, can = tt
                    out['fam'][j] = fam
                    if cfg.skip_loss_false and _final_loss_false(m):
                        out['st'][j] = ST_LOSSFALSE
                    elif cfg.skip_leaked and st.is_leaked(can, prior):
                        out['st'][j] = ST_LEAK
                    else:
                        cand.append((j, can))
            j += 1
        elif role == 'user':
            last_user = content
        prior.append(content)
    if cand:
        idx = [c[0] for c in cand]
        keys = st.target_keys([c[1] for c in cand])
        out['key'][idx] = keys
        rows = reader.lookup(keys)
        hit = rows >= 0
        for (jj, txt), h in zip(cand, hit):
            out['st'][jj] = ST_VALID if h else ST_MISS
            if not h:
                out['txt'][jj] = txt
        if hit.any():
            out['tgt'][np.asarray(idx)[hit]] = reader.rows(rows[hit])
    return out


def _emit(encoded: Dict[str, Any], res: Dict[str, Any], online: bool) -> None:
    """Fresh tensors into the encoded dict (encode runs under inference_mode: never mutated later)."""
    encoded['sem_tgt'] = torch.from_numpy(np.ascontiguousarray(res['tgt']))
    encoded['sem_st'] = torch.from_numpy(res['st'].astype(np.int8))
    encoded['sem_ok'] = torch.from_numpy(res['st'] == ST_VALID)
    encoded['sem_key'] = torch.from_numpy(_u64_to_i64(res['key']).copy())
    encoded['sem_img'] = torch.from_numpy(_u64_to_i64(res['img']).copy())
    if online:
        encoded['sem_txt'] = list(res['txt'])


def attach_row_targets(template, inputs, encoded: Dict[str, Any], sem_ids: Sequence[int]) -> None:
    """Called by the Template._encode wrapper after the original: compute and emit the row's targets."""
    n_blk = count_sem_blocks(encoded['input_ids'], sem_ids)
    if n_blk == 0 and any(m.get('role') == 'assistant' for m in (getattr(inputs, 'messages', None) or [])):
        # nothing to supervise AND nothing that would show up as a mismatch: say it (once per process)
        _warn_once('no_blocks', '[sem-latent] a training row with assistant turns has NO SEM block (headers in a '
                   'loss-scale > 0 context, e.g. --loss_scale all?) -- serving inserts one after every header')
    pend = getattr(_TLS, 'pending', None) or {}
    extra = getattr(inputs, 'extra_kwargs', None) or {}
    sem_targets = pend.get('sem_targets') if pend.get('sem_targets') is not None else extra.get('sem_targets')
    sem_target = pend.get('sem_target') if pend.get('sem_target') is not None else extra.get('sem_target')
    reader = get_reader()
    try:
        res = row_block_targets(inputs.messages, getattr(inputs, 'system', None), n_blk,
                                img_key=pend.get('img', 0) or 0, sem_targets=sem_targets, sem_target=sem_target,
                                reader=reader)
    except SemConfigError:
        raise
    except Exception as e:  # noqa: BLE001 - a bad row must not kill a multi-day run: unsupervised
        _STATE['n_err'] = _STATE.get('n_err', 0) + 1
        if _STATE['n_err'] <= 5:
            logger.warning(f'[sem-latent] target extraction failed ({type(e).__name__}: {e}); row unsupervised')
        res = {'tgt': np.zeros((n_blk, reader.dim), np.float16), 'st': np.full(n_blk, ST_ERROR, np.int8),
               'key': np.zeros(n_blk, np.uint64), 'img': np.zeros(n_blk, np.uint64), 'txt': [''] * n_blk}
    _emit(encoded, res, CFG.online_teacher)


# =========================================================================================
# collator / post-encode
# =========================================================================================
def _as_t(x, dtype, shape_tail=()):
    if x is None:
        return None
    if isinstance(x, torch.Tensor):
        return x.to(dtype)
    return torch.as_tensor(np.asarray(x), dtype=dtype).reshape(-1, *shape_tail) if shape_tail else \
        torch.as_tensor(np.asarray(x), dtype=dtype).reshape(-1)


def collate_sem(batch: Sequence[Dict[str, Any]], sem_ids: Sequence[int], dim: int,
                online: bool = False) -> Dict[str, Any]:
    """Concatenate per-row sem_* in row order; recount complete blocks in each row's final input_ids."""
    tg, stt, key, img, nblk, txt = [], [], [], [], [], []
    n_fix = 0
    for b in batch:
        ids = b.get('input_ids')
        n_act = count_sem_blocks(ids, sem_ids) if ids is not None else 0
        r_st = _as_t(b.get('sem_st'), torch.int8)
        if r_st is not None and r_st.numel() == n_act:
            tg.append(_as_t(b['sem_tgt'], torch.float16, (dim, )).reshape(n_act, dim))
            stt.append(r_st)
            key.append(_as_t(b.get('sem_key'), torch.int64))
            img.append(_as_t(b.get('sem_img'), torch.int64))
            t = b.get('sem_txt')
            txt.extend(list(t) if t is not None else [''] * n_act)
        else:
            if r_st is not None:
                n_fix += 1          # truncation cut a block, or the row was encoded elsewhere
            tg.append(torch.zeros(n_act, dim, dtype=torch.float16))
            stt.append(torch.full((n_act, ), ST_MISMATCH if r_st is not None else ST_NONE, dtype=torch.int8))
            key.append(torch.zeros(n_act, dtype=torch.int64))
            img.append(torch.zeros(n_act, dtype=torch.int64))
            txt.extend([''] * n_act)
        nblk.append(n_act)
    if n_fix:
        _warn_once('collate_fix', f'[sem-latent] {n_fix} row(s) in a batch had a SEM block count different from '
                   'their targets (truncation?) -> those rows are unsupervised')
    s = torch.cat(stt) if stt else torch.zeros(0, dtype=torch.int8)
    res = {'sem_tgt': torch.cat(tg) if tg else torch.zeros(0, dim, dtype=torch.float16), 'sem_st': s,
           'sem_ok': s == ST_VALID, 'sem_key': torch.cat(key) if key else torch.zeros(0, dtype=torch.int64),
           'sem_img': torch.cat(img) if img else torch.zeros(0, dtype=torch.int64),
           'sem_nblk': torch.tensor(nblk, dtype=torch.int64)}
    if online:
        res['sem_txt'] = txt
    return res


def sem_block_positions(input_ids: torch.Tensor, sem_ids: Sequence[int]):
    """Complete SEM blocks in a padded [B, L] id tensor -> (b_idx [n], pos [n, K], counts [B]).
    Blocks come out row-major (row, then position) -- the collator's concatenation order."""
    K = len(sem_ids)
    if input_ids.dim() == 1:
        input_ids = input_ids[None]
    B, L = input_ids.shape
    dev = input_ids.device
    if K == 0 or L < K:
        e = torch.zeros(0, dtype=torch.long, device=dev)
        return e, e.reshape(0, max(K, 1))[:, :K], torch.zeros(B, dtype=torch.long, device=dev)
    sem = torch.as_tensor(list(sem_ids), dtype=input_ids.dtype, device=dev)
    match = (input_ids.unfold(1, K, 1) == sem).all(-1)          # [B, L-K+1]
    b_idx, start = match.nonzero(as_tuple=True)
    pos = start[:, None] + torch.arange(K, device=dev)[None]
    return b_idx, pos, match.sum(1)


class SemBatch:
    """One micro-batch's SEM stash (set by _post_encode, filled by the LM hook, popped by the loss hook)."""

    __slots__ = ('b_idx', 'pos', 'tgt', 'st', 'key', 'img', 'shape', 'h', 'n', 'n_online')

    def __init__(self, b_idx, pos, tgt, st_, key, img, shape, n_online=0):
        self.b_idx, self.pos, self.tgt, self.st, self.key, self.img = b_idx, pos, tgt, st_, key, img
        self.shape, self.h, self.n, self.n_online = tuple(shape), None, int(pos.shape[0]), n_online

    @property
    def ok(self):
        return self.st == ST_VALID


def make_batch(input_ids: torch.Tensor, sem_ids: Sequence[int], popped: Dict[str, Any],
               dim: int) -> SemBatch:
    b_idx, pos, counts = sem_block_positions(input_ids, sem_ids)
    n = int(pos.shape[0])
    dev = input_ids.device
    tgt, st_, key, img = (popped.get('sem_tgt'), popped.get('sem_st'), popped.get('sem_key'), popped.get('sem_img'))
    nblk = popped.get('sem_nblk')
    consistent = (st_ is not None and tgt is not None and int(st_.numel()) == n and nblk is not None
                  and torch.equal(torch.as_tensor(nblk).to(counts.device).long(), counts.long()))
    if not consistent:
        if st_ is not None or nblk is not None:
            _warn_once('post_mismatch', f'[sem-latent] SEM blocks in the padded batch ({n}, per row '
                       f'{counts.tolist()}) != collated targets (per row '
                       f'{None if nblk is None else torch.as_tensor(nblk).tolist()}) -> batch unsupervised '
                       '(is packing / padding_free on?)')
        tgt = torch.zeros(n, dim, dtype=torch.float16, device=dev)
        st_ = torch.full((n, ), ST_MISMATCH if st_ is not None else ST_NONE, dtype=torch.int8, device=dev)
        key = torch.zeros(n, dtype=torch.int64, device=dev)
        img = torch.zeros(n, dtype=torch.int64, device=dev)
    else:
        tgt, st_, key, img = (x.to(dev) for x in (tgt, st_, key, img))
    n_online = 0
    if consistent and CFG.online_teacher and n and bool((st_ == ST_MISS).any()):
        tgt, st_, n_online = _online_fill(tgt, st_, key, popped.get('sem_txt'))
    return SemBatch(b_idx, pos, tgt, st_, key, img, input_ids.shape, n_online=n_online)


# =========================================================================================
# online teacher (optional, main process)
# =========================================================================================
def _builder_module():
    import importlib.util
    here = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(os.path.dirname(here), 'examples', 'train', 'liger', 'sem_latent', 'build_sem_targets.py')
    spec = importlib.util.spec_from_file_location('_sem_build_sem_targets', path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _get_online_teacher(device):
    """The builder's own Teacher (identical recipe to the sidecar), OFF the module tree, fingerprinted."""
    t = _STATE['teacher']
    if t is not None:
        return t
    meta = get_reader().meta
    path = st.resolve_storage_path(os.environ.get('SEM_TEACHER_PATH') or meta.get('teacher_path')
                                   or st.default_teacher_path())
    b = _builder_module()
    fp = b.backbone_fingerprint(path)
    if fp != meta.get('teacher_fingerprint'):
        raise SemConfigError(f'[sem-latent] online teacher {path} fingerprint {fp} != sidecar '
                             f'{meta.get("teacher_fingerprint")}: online targets would live in another space')
    dev = torch.device(device)
    args = SimpleNamespace(max_length=int(meta.get('max_length') or st.TEACHER_MAX_LENGTH),
                           device=str(dev), dtype='bf16' if dev.type == 'cuda' else 'fp32',
                           teacher=path, probe_head=None, threads=8)
    t = b.Teacher(args)
    _STATE['teacher'] = t
    logger.info(f'[sem-latent] online teacher loaded on {dev} ({path}, fingerprint ok)')
    return t


def _online_fill(tgt, st_, key, texts):
    miss = (st_ == ST_MISS).nonzero().flatten().tolist()
    if not miss or not texts:
        return tgt, st_, 0
    cache = _STATE['online_cache']
    tgt, st_ = tgt.clone(), st_.clone()
    todo = {}
    for i in miss:
        k = int(key[i])
        if k in cache:
            cache.move_to_end(k)
            tgt[i] = torch.from_numpy(cache[k]).to(tgt)
            st_[i] = ST_VALID
        elif texts[i]:
            todo.setdefault(texts[i], []).append(i)
    if todo:
        tl = list(todo)
        z, _ = _get_online_teacher(tgt.device).encode(tl)
        for txt, zz in zip(tl, np.asarray(z, dtype=np.float16)):
            for i in todo[txt]:
                tgt[i] = torch.from_numpy(zz).to(tgt)
                st_[i] = ST_VALID
                cache[int(key[i])] = zz
        while len(cache) > CFG.online_cache:
            cache.popitem(last=False)
    return tgt, st_, sum(1 for i in miss if int(st_[i]) == ST_VALID)


# =========================================================================================
# the head, the queue, the loss
# =========================================================================================
class SemanticHead(nn.Module):
    """[n, K, in_dim] SEM hidden states -> [n, out_dim] unit vectors (fp32) in the teacher space."""

    def __init__(self, in_dim: int, out_dim: int, k: int, pool: str = 'attn', hidden: int = 2048, eps: float = 1e-6):
        super().__init__()
        if pool not in ('attn', 'mean', 'concat'):
            raise ValueError(f'pool must be attn|mean|concat, got {pool!r}')
        self.in_dim, self.out_dim, self.k, self.pool = in_dim, out_dim, k, pool
        self.norm = nn.LayerNorm(in_dim, eps=eps)
        if pool == 'attn':
            # zero-init query = uniform attention = mean pooling at step 0
            self.pool_query = nn.Parameter(torch.zeros(in_dim))
        feat = in_dim * k if pool == 'concat' else in_dim
        self.fc1 = nn.Linear(feat, hidden)
        self.act = nn.SiLU()
        self.fc2 = nn.Linear(hidden, out_dim)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        if h.dim() != 3 or h.shape[-1] != self.in_dim:
            raise ValueError(f'SemanticHead expects [n, K, {self.in_dim}], got {tuple(h.shape)}')
        dt = self.fc1.weight.dtype
        x = self.norm(h.to(dt))
        if self.pool == 'attn':
            s = (x.float() @ self.pool_query.float()) / math.sqrt(self.in_dim)       # [n, K]
            a = torch.softmax(s, dim=1).to(dt)
            x = (a.unsqueeze(-1) * x).sum(1)
        elif self.pool == 'mean':
            x = x.mean(1)
        else:
            if h.shape[1] != self.k:
                raise ValueError(f'concat pooling was built for K={self.k}, got K={h.shape[1]}')
            x = x.reshape(x.shape[0], self.k * self.in_dim)       # explicit: n may be 0
        y = self.fc2(self.act(self.fc1(x)))
        return F.normalize(y.float(), dim=-1)


class SemQueue:
    """Per-rank FIFO of past centred teacher targets + their target / image keys (no grad)."""

    def __init__(self, size: int):
        self.size = max(int(size), 0)
        self.vecs = self.keys = self.imgs = None
        self.ptr = 0
        self.count = 0

    def __len__(self):
        return self.count

    @torch.no_grad()
    def push(self, vecs: torch.Tensor, keys: torch.Tensor, imgs: torch.Tensor) -> None:
        if self.size == 0 or vecs.shape[0] == 0:
            return
        vecs, keys, imgs = vecs.detach().float(), keys.detach().long(), imgs.detach().long()
        if self.vecs is None or self.vecs.device != vecs.device or self.vecs.shape[1] != vecs.shape[1]:
            self.vecs = torch.zeros(self.size, vecs.shape[1], device=vecs.device)
            self.keys = torch.zeros(self.size, dtype=torch.long, device=vecs.device)
            self.imgs = torch.zeros(self.size, dtype=torch.long, device=vecs.device)
            self.ptr = self.count = 0
        n = vecs.shape[0]
        if n >= self.size:
            vecs, keys, imgs, n = vecs[-self.size:], keys[-self.size:], imgs[-self.size:], self.size
        idx = (self.ptr + torch.arange(n, device=vecs.device)) % self.size
        self.vecs[idx], self.keys[idx], self.imgs[idx] = vecs, keys, imgs
        self.ptr = int((self.ptr + n) % self.size)
        self.count = min(self.count + n, self.size)

    def get(self):
        """(vecs, keys, imgs) of the filled slots, oldest first."""
        if self.count == 0:
            return None
        if self.count < self.size:
            sl = slice(0, self.count)
            return self.vecs[sl], self.keys[sl], self.imgs[sl]
        order = (self.ptr + torch.arange(self.size, device=self.vecs.device)) % self.size
        return self.vecs[order], self.keys[order], self.imgs[order]


def _queue() -> SemQueue:
    q = _STATE['queue']
    if q is None or q.size != max(CFG.queue, 0):
        q = _STATE['queue'] = SemQueue(CFG.queue)
    return q


def centre(v: torch.Tensor, mu: torch.Tensor) -> torch.Tensor:
    """c(v) = normalize(v - mu), fp32."""
    return F.normalize(v.float() - mu.float(), dim=-1)


def sem_losses(z_hat: torch.Tensor, z_tgt: torch.Tensor, mu: torch.Tensor, keys: Optional[torch.Tensor] = None,
               imgs: Optional[torch.Tensor] = None, extra=None, mode: str = 'soft', tau: float = 0.05,
               tau_t: float = 0.05, fn_thresh: float = 0.9) -> Dict[str, Any]:
    """Loss terms over n VALID blocks (fp32). extra = (vecs [M, D] ALREADY centred, keys [M], imgs [M])
    or None -- the queue / gathered candidates. Returns tensors cos_loss, nce_loss (with graph; soft mode:
    the KL = CE - H(p), hard mode: the one-hot CE), kl (soft, == nce_loss), nce_ce (soft: the CE, detached)
    and detached diagnostics cos_c / cos_raw (per block) and ct (centred targets)."""
    zh, zt = z_hat.float(), z_tgt.float()
    n = zh.shape[0]
    ch = centre(zh, mu)
    with torch.no_grad():
        ct = centre(zt, mu)
    cos_c = (ch * ct).sum(-1)
    cos_loss = (1.0 - cos_c).mean()
    with torch.no_grad():
        cos_raw = (F.normalize(zh, dim=-1) * F.normalize(zt, dim=-1)).sum(-1)
    dev = zh.device
    keys = keys.to(dev).long() if keys is not None else torch.zeros(n, dtype=torch.long, device=dev)
    imgs = imgs.to(dev).long() if imgs is not None else torch.zeros(n, dtype=torch.long, device=dev)
    cand, ck, ci = ct, keys, imgs
    if extra is not None and extra[0] is not None and extra[0].shape[0]:
        cand = torch.cat([ct, extra[0].to(dev).float()])
        ck = torch.cat([keys, extra[1].to(dev).long()])
        ci = torch.cat([imgs, extra[2].to(dev).long()])
    logits = ch @ cand.T / tau
    kl = None
    nce_ce = None
    if mode == 'soft':
        with torch.no_grad():
            tl = ct @ cand.T / tau_t
            p = torch.softmax(tl, dim=-1)
            ent = -(p * torch.log_softmax(tl, dim=-1)).sum(-1)
        ce = -(p * torch.log_softmax(logits, dim=-1)).sum(-1)
        # OPTIMISE the KL, not the cross-entropy: same gradient (H(p) is a no-grad constant), but the CE
        # carries H(p) > 0 even for a perfect student and H(p) grows with the candidate count (queue fill),
        # which would drift loss / eval_loss / sem_share upward on its own. The CE is still reported.
        kl = (ce - ent).mean()
        nce_loss = kl
        nce_ce = ce.mean().detach()
    elif mode == 'hard':
        with torch.no_grad():
            fn = (ct @ cand.T) > fn_thresh
            fn |= ck[None, :] == keys[:, None]
            fn |= (ci[None, :] == imgs[:, None]) & (imgs[:, None] != 0)
            eye = torch.arange(n, device=dev)
            fn[eye, eye] = False
        nce_loss = F.cross_entropy(logits.masked_fill(fn, float('-inf')), torch.arange(n, device=dev))
    else:
        raise ValueError(f'mode must be soft|hard, got {mode!r}')
    return {'cos_loss': cos_loss, 'nce_loss': nce_loss, 'kl': kl, 'nce_ce': nce_ce, 'cos_c': cos_c.detach(),
            'cos_raw': cos_raw, 'ct': ct, 'n_cand': int(cand.shape[0])}


def _gather_candidates(ct: torch.Tensor, keys: torch.Tensor, imgs: torch.Tensor):
    """SEM_NCE_GATHER=1: other ranks' centred targets (no grad). EVERY rank must call this on every
    training micro-batch (including ones with 0 valid blocks), which the loss hook guarantees."""
    import torch.distributed as dist
    if not (dist.is_available() and dist.is_initialized()) or dist.get_world_size() == 1:
        return None
    ws, rank = dist.get_world_size(), dist.get_rank()
    dev, dim = ct.device, ct.shape[1]
    n = torch.tensor([ct.shape[0]], device=dev, dtype=torch.long)
    sizes = [torch.zeros_like(n) for _ in range(ws)]
    dist.all_gather(sizes, n)
    sizes = [int(s) for s in sizes]
    m = max(sizes)
    if m == 0:
        return None
    k = ct.shape[0]
    pad = torch.zeros(m, dim, device=dev, dtype=torch.float32)
    if k:
        pad[:k] = ct.detach().float()
    # keys/imgs travel as int64 in a second buffer (exact)
    kb = torch.zeros(m, 2, device=dev, dtype=torch.long)
    if k:
        kb[:k, 0], kb[:k, 1] = keys.long(), imgs.long()
    vs = [torch.zeros_like(pad) for _ in range(ws)]
    ks = [torch.zeros_like(kb) for _ in range(ws)]
    dist.all_gather(vs, pad)
    dist.all_gather(ks, kb)
    parts = [(vs[r][:sizes[r]], ks[r][:sizes[r]]) for r in range(ws) if r != rank and sizes[r]]
    if not parts:
        return None
    return (torch.cat([p[0] for p in parts]), torch.cat([p[1][:, 0] for p in parts]),
            torch.cat([p[1][:, 1] for p in parts]))


def current_weights(step: Optional[int] = None) -> Tuple[float, float]:
    s = _SCHED['step'] if step is None else step
    return CFG.w_cos(s), CFG.w_nce(s)


def _head_params(head) -> List[torch.Tensor]:
    return [p for p in head.parameters() if p.requires_grad] if head is not None else []


def _zero_head_term(head, ref: torch.Tensor, zero: torch.Tensor, training: bool) -> torch.Tensor:
    """0 * (the head's output on a zero input): every trainable head param gets a defined, zero gradient.

    Built THROUGH the head's own forward (not sum(p.sum())) so the params become grad-ready in the same
    order as on a real batch: DeepSpeed ZeRO-2 fills its reduce buckets in hook order and all-reduces
    them position-wise, so a rank on this path while another rank has valid blocks would otherwise mix
    up the sem_head gradients (review D2). No graph into the LM (the input is a constant).
    """
    if not (training and _head_params(head)):
        return zero
    w = head.fc1.weight
    z = head(torch.zeros(1, head.k, head.in_dim, device=w.device, dtype=w.dtype))
    return z.sum().to(zero.device) * 0.0


def compute_sem_term(owner: nn.Module, batch: Optional[SemBatch], ref: torch.Tensor,
                     training: bool) -> Tuple[torch.Tensor, Dict[str, float]]:
    """The tensor added to out.loss for one micro-batch (+ float stats). `ref` gives device/dtype."""
    head = getattr(owner, 'sem_head', None)
    w_cos, w_nce = current_weights()
    stats: Dict[str, float] = {'w_cos': w_cos, 'w_nce': w_nce, 'n_blk': 0, 'n_valid': 0}
    zero = ref.new_zeros(()).float()
    gather = training and CFG.nce_gather
    if batch is None or batch.n == 0 or batch.h is None:
        if batch is not None and batch.n and batch.h is None:
            _warn_once('no_hidden', '[sem-latent] SEM blocks stashed but the language-model hook captured no hidden '
                       'states (model.model.language_model not called?) -> no SEM loss')
        if gather:
            _gather_candidates(torch.zeros(0, get_reader().dim, device=ref.device), torch.zeros(0, dtype=torch.long,
                               device=ref.device), torch.zeros(0, dtype=torch.long, device=ref.device))
        return _zero_head_term(head, ref, zero, training), stats
    stc = batch.st.long()
    for code, name in enumerate(ST_NAMES):
        stats[f'n_{name}'] = int((stc == code).sum())
    stats['n_blk'] = batch.n
    stats['n_online'] = batch.n_online
    h = batch.h
    detached = training and _SCHED['step'] < CFG.detach_steps
    if detached:
        h = h.detach()
    stats['detached'] = float(detached)
    z_hat = head(h)
    valid = batch.ok
    nv = int(valid.sum())
    stats['n_valid'] = nv
    dev = z_hat.device
    mu = _mu(dev)
    if nv == 0:
        if gather:
            _gather_candidates(torch.zeros(0, mu.shape[0], device=dev), torch.zeros(0, dtype=torch.long, device=dev),
                               torch.zeros(0, dtype=torch.long, device=dev))
        return z_hat.sum() * 0.0, stats
    zv, tv = z_hat[valid], batch.tgt[valid].to(dev)
    kv, iv = batch.key[valid].to(dev), batch.img[valid].to(dev)
    # the queue holds TRAINING targets: eval scores against its own micro-batch only, so eval_loss does not
    # depend on how full the training queue happens to be (retrieval over the whole eval set is val_sem/*)
    q = _queue().get() if (training and CFG.queue > 0) else None
    extra = None if q is None else (q[0], q[1], q[2])
    if gather:
        with torch.no_grad():
            g = _gather_candidates(centre(tv, mu), kv, iv)
        if g is not None:
            extra = g if extra is None else tuple(torch.cat([a, b]) for a, b in zip(extra, g))
    L = sem_losses(zv, tv, mu, kv, iv, extra=extra, mode=CFG.nce_mode, tau=CFG.tau, tau_t=CFG.tau_t,
                   fn_thresh=CFG.fn_thresh)
    term = w_cos * L['cos_loss'] + w_nce * L['nce_loss']
    stats.update(cos_loss=float(L['cos_loss'].detach()), nce_loss=float(L['nce_loss'].detach()),
                 kl=float(L['kl'].detach()) if L['kl'] is not None else float('nan'),
                 nce_ce=float(L['nce_ce']) if L.get('nce_ce') is not None else float('nan'),
                 cos_centred=float(L['cos_c'].mean()), cos_raw=float(L['cos_raw'].mean()), n_cand=L['n_cand'],
                 term=float(term.detach()))
    if training:
        _queue().push(L['ct'], kv, iv)
    else:
        _EVAL['zh'].append(zv.detach().to('cpu', torch.float16).numpy())
        _EVAL['zt'].append(tv.detach().to('cpu', torch.float16).numpy())
        _EVAL['key'].append(kv.detach().cpu().numpy())
        _EVAL['img'].append(iv.detach().cpu().numpy())
    return term, stats


def _record_run(stats: Dict[str, float], total: float) -> None:
    _RUN['micro'] += 1
    for k in ('n_blk', 'n_valid', 'n_miss', 'n_leaked', 'n_mismatch', 'n_loss_false', 'n_error', 'n_online'):
        _RUN[k] += stats.get(k, 0)
    if stats.get('n_valid', 0):
        _RUN['n'] += 1
        for k in ('cos_loss', 'nce_loss', 'cos_centred', 'cos_raw'):
            _RUN[k] += stats[k]
        if not math.isnan(stats.get('kl', float('nan'))):
            _RUN['kl'] += stats['kl']
            _RUN['n_kl'] += 1
        if not math.isnan(stats.get('nce_ce', float('nan'))):
            _RUN['nce_ce'] += stats['nce_ce']
            _RUN['n_ce'] += 1
        _RUN['share'] += 100.0 * stats['term'] / max(abs(total), 1e-12)


# =========================================================================================
# attach: head, hooks, restore, init, checks
# =========================================================================================
def _unwrap(model):
    m = model
    for _ in range(4):
        nxt = getattr(m, 'module', None)
        if nxt is None or nxt is m:
            break
        m = nxt
    gb = getattr(m, 'get_base_model', None)
    if callable(gb):
        try:
            m = gb()
        except Exception:  # noqa: BLE001
            pass
    return m


def _owner(model):
    """The module attach() ran on (it carries the head + hooks), from whatever wraps it."""
    m = model
    for _ in range(6):
        if m is None:
            break
        if isinstance(m, nn.Module) and m.__dict__.get('_sem_attached', False):
            return m
        nxt = getattr(m, 'module', None)
        if nxt is None or nxt is m:
            gb = getattr(m, 'get_base_model', None)
            nxt = gb() if callable(gb) else None
        if nxt is None or nxt is m:
            break
        m = nxt
    return _unwrap(model)


def _language_model(base):
    inner = getattr(base, 'model', None)
    lm = getattr(inner, 'language_model', None)
    if lm is None:
        lm = getattr(base, 'language_model', None)
    if lm is None:
        raise RuntimeError('[sem-latent] cannot find model.model.language_model (expected Qwen3.5-VL)')
    return lm


def _model_dir(base) -> Optional[str]:
    for src in (getattr(base, 'model_dir', None), getattr(base, 'name_or_path', None),
                getattr(getattr(base, 'config', None), '_name_or_path', None)):
        if src and os.path.isdir(src):
            return src
    return None


def _ckpt_vocab(model_dir: Optional[str]) -> Optional[int]:
    if not model_dir:
        return None
    try:
        with open(os.path.join(model_dir, 'config.json')) as f:
            cfg = json.load(f)
    except Exception:  # noqa: BLE001
        return None
    for sub in ('text_config', 'llm_config', 'language_config'):
        if isinstance(cfg.get(sub), dict) and cfg[sub].get('vocab_size'):
            return int(cfg[sub]['vocab_size'])
    return int(cfg['vocab_size']) if cfg.get('vocab_size') else None


def startup_checks(base, tokenizer) -> Tuple[int, ...]:
    """Raise on every condition that would make SEM training silently wrong; returns the SEM ids."""
    sem_ids = get_sem_ids(tokenizer)
    if not sem_ids:
        raise SemConfigError(
            '[sem-latent] the tokenizer has no <|sem_0|> token, so no SEM block is ever inserted. Add\n'
            '    --new_special_tokens examples/train/liger/token.txt examples/train/liger/token_sem.txt\n'
            '  (and do not set SWIFT_SEM_LATENT=0).')
    K = len(sem_ids)
    if K > MAX_SEM:
        raise SemConfigError(f'[sem-latent] K={K} SEM tokens > the {MAX_SEM} spare embedding rows: adding them '
                             'resized the embeddings, which re-ties lm_head to embed_tokens. Use K <= 109.')
    emb = base.get_input_embeddings().weight
    rows = int(emb.shape[0])
    if rows and max(sem_ids) >= rows:
        raise SemConfigError(f'[sem-latent] SEM id {max(sem_ids)} >= embedding rows {rows}')
    ck = _ckpt_vocab(_model_dir(base))
    if ck is not None and rows and ck != rows:
        raise SemConfigError(f'[sem-latent] embedding rows {rows} != checkpoint vocab_size {ck}: the embeddings '
                             'were RESIZED (HF tie_weights() then re-ties lm_head to embed_tokens and discards the '
                             'trained lm_head). Refusing; keep K within the spare rows.')
    if ck is None:
        _warn_once('no_ckpt_vocab', '[sem-latent] could not read the checkpoint vocab_size: resize check skipped')
    out = base.get_output_embeddings()
    tied = out is not None and getattr(out, 'weight', None) is emb
    reader = get_reader()
    logger.info(f'[sem-latent] K={K} SEM tokens, ids {sem_ids[0]}..{sem_ids[-1]} '
                f'(contiguous={list(sem_ids) == list(range(sem_ids[0], sem_ids[0] + K))}) | embedding rows {rows} '
                f'(ckpt vocab {ck}; no resize) | lm_head tied to embed_tokens: {tied} | sidecar {reader.dir}: '
                f'n={reader.n:,} dim={reader.dim} rule={reader.rule_version} fingerprint='
                f'{str(reader.meta.get("teacher_fingerprint"))[:16]} probe_logits={reader.meta.get("has_probe_logits")} '
                f'|mu|={float(np.linalg.norm(reader.mu)):.3f}')
    return tuple(sem_ids)


def restore_head(head: nn.Module, src_dir: Optional[str], prefix: str = 'sem_head.') -> int:
    """Load `prefix*` tensors from the safetensors of src_dir (safe_open + key filter). -> #tensors."""
    if not src_dir:
        logger.info('[sem-latent] no model dir to restore sem_head from -- fresh head')
        return 0
    files = []
    idx_path = os.path.join(src_dir, 'model.safetensors.index.json')
    if os.path.isfile(idx_path):
        with open(idx_path) as f:
            wm = json.load(f).get('weight_map', {})
        files = sorted({os.path.join(src_dir, v) for k, v in wm.items() if k.startswith(prefix)})
    else:
        files = sorted(glob.glob(os.path.join(src_dir, '*.safetensors')))
    from safetensors import safe_open
    sd = {}
    for fp in files:
        with safe_open(fp, framework='pt', device='cpu') as sf:
            for k in sf.keys():
                if k.startswith(prefix):
                    sd[k[len(prefix):]] = sf.get_tensor(k)
    if not sd:
        logger.info(f'[sem-latent] no {prefix}* tensors in {src_dir} -- fresh head (first SEM stage)')
        return 0
    try:
        missing, unexpected = head.load_state_dict(sd, strict=False)
    except RuntimeError as e:
        raise SemConfigError(f'[sem-latent] {prefix}* in {src_dir} does not fit this head (SEM_POOL / K / '
                             f'SEM_HEAD_HIDDEN changed?): {e}. SEM_RESTORE_HEAD=0 starts a fresh head.') from e
    if missing or unexpected:
        raise SemConfigError(f'[sem-latent] partial {prefix}* restore from {src_dir}: missing={missing} '
                             f'unexpected={unexpected}. SEM_RESTORE_HEAD=0 starts a fresh head.')
    logger.info(f'[sem-latent] restored {len(sd)} {prefix}* tensors from {src_dir}')
    return len(sd)


def init_sem_embeddings(base, sem_ids: Sequence[int], n_vocab: Optional[int] = None, seed: int = 1234,
                        noise: float = 0.5, tol: float = 0.05) -> bool:
    """Re-init the SEM rows of embed_tokens iff they (+ the other spare rows) are ONE vector.

    "One vector" is tolerance-based: the Qwen3.5-2B spare rows are the same untrained vector up to bf16
    rounding (checkpoint-608: 6 distinct bf16 patterns among the 109 rows, max |row_i - row_0| 1.2e-4 =
    0.009 x the median per-dim std of the trained rows), so bit-identity would never fire. Fresh iff
    max_i,d |row_i[d] - row_0[d]| <= tol * median_d std_d(trained rows). Initialised SEM rows differ by
    ~noise*std per dim (~50x the threshold), so a SEM checkpoint is never re-initialised.
    """
    W = base.get_input_embeddings().weight
    if W.numel() == 0:
        _warn_once('zero3', '[sem-latent] embed_tokens is partitioned (ZeRO-3?) -- SEM row init skipped')
        return False
    rows = W.shape[0]
    ids = sorted(set(int(i) for i in sem_ids))
    spare = list(range(max(n_vocab or 0, max(ids) + 1), rows))
    chk = ids + spare
    if len(chk) < 2:
        logger.info('[sem-latent] cannot tell whether the single SEM row is fresh -- left unchanged')
        return False
    n_tr = min(ids)
    H = W.shape[1]
    s1 = torch.zeros(H, dtype=torch.float64)
    s2 = torch.zeros(H, dtype=torch.float64)
    with torch.no_grad():
        for a in range(0, n_tr, 16384):          # float64 on CPU, fixed order -> identical on every rank
            x = W.data[a:min(a + 16384, n_tr)].detach().to('cpu', torch.float64)
            s1 += x.sum(0)
            s2 += (x * x).sum(0)
        blk = W.data[torch.as_tensor(chk, device=W.device)].detach().to('cpu', torch.float64)
    mean = s1 / n_tr
    std = (s2 / n_tr - mean * mean).clamp_min(0).sqrt()
    spread = float((blk - blk[0:1]).abs().max())
    thresh = float(tol) * float(std.median())
    if spread > thresh:
        logger.info(f'[sem-latent] SEM embedding rows are already distinct (spread {spread:.2e} > {thresh:.2e} = '
                    f'{tol} x median trained-row std) -- left unchanged (SEM_INIT_EMBED)')
        return False
    g = torch.Generator(device='cpu').manual_seed(int(seed))
    z = torch.randn(len(ids), H, generator=g, dtype=torch.float64)
    new = mean[None] + z * std[None] * float(noise)
    with torch.no_grad():
        W.data[torch.as_tensor(ids, device=W.device)] = new.to(device=W.device, dtype=W.dtype)
    logger.info(f'[sem-latent] SEM embedding rows {ids[0]}..{ids[-1]} re-initialised (spare-row spread {spread:.2e} <= '
                f'{thresh:.2e}): mean(trained rows 0..{n_tr - 1}) + N(0,1)*std*{noise} (seed {seed}); '
                f'|mean|={float(mean.norm()):.4f} mean std={float(std.mean()):.5f}; lm_head rows unchanged')
    return True


def _untie_fn():
    """qure_untie_config_plugin.untie_tied_config WITHOUT installing that plugin's own patch.

    `swift_plugins.qure_untie_config_plugin` when the repo root is on sys.path; else the copy already
    loaded by --external_plugins; else the sibling file loaded under a private module name (a top-level
    `import qure_untie_config_plugin` would count as the plugin load and wrap prepare_model). Untie is
    what makes sglang/vLLM serve the trained lm_head (F3), so failing to find it RAISES
    (SEM_UNTIE_CONFIG=0 skips it deliberately).
    """
    try:
        from swift_plugins.qure_untie_config_plugin import untie_tied_config
        return untie_tied_config
    except ImportError as e:
        if not (isinstance(e, ModuleNotFoundError) and (e.name or '').startswith('swift_plugins')):
            raise
    loaded = sys.modules.get('qure_untie_config_plugin')
    if loaded is not None and callable(getattr(loaded, 'untie_tied_config', None)):
        return loaded.untie_tied_config
    import importlib.util
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'qure_untie_config_plugin.py')
    if not os.path.isfile(path):
        raise SemConfigError(f'[sem-latent] SEM_UNTIE_CONFIG=1 but qure_untie_config_plugin is not importable (no '
                             f'swift_plugins package on sys.path, no {path}): saved checkpoints would stay '
                             'tie_word_embeddings=true and sglang/vLLM would serve embed_tokens as lm_head. Put the repo '
                             'root on PYTHONPATH (or SEM_UNTIE_CONFIG=0 to skip the untie deliberately).')
    spec = importlib.util.spec_from_file_location('_sem_qure_untie_config', path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.untie_tied_config


def attach(model, template=None, tokenizer=None):
    """Add sem_head + hooks (idempotent). Called from the prepare_model wrapper; usable directly."""
    base = _unwrap(model)
    if base.__dict__.get('_sem_attached', False):
        return model
    tok = tokenizer if tokenizer is not None else getattr(template, 'tokenizer', None)
    if tok is None:
        raise SemConfigError('[sem-latent] attach() needs the template (or tokenizer) to find the SEM ids')
    sem_ids = startup_checks(base, tok)
    K = len(sem_ids)
    reader = get_reader()
    emb = base.get_input_embeddings().weight
    in_dim = int(emb.shape[1])
    head = SemanticHead(in_dim, reader.dim, K, pool=CFG.pool, hidden=CFG.head_hidden)
    head.to(device=emb.device, dtype=CFG.head_dtype)
    base.sem_head = head
    if CFG.restore_head:
        restore_head(head, _model_dir(base))
    if CFG.freeze_head:
        head.requires_grad_(False)
        head.eval()
        logger.info('[sem-latent] sem_head FROZEN (no optimizer group; still saved; loss still trains the LM)')
    if CFG.init_embed:
        n_vocab = None
        try:
            n_vocab = len(tok)
        except Exception:  # noqa: BLE001
            pass
        init_sem_embeddings(base, sem_ids, n_vocab=n_vocab, seed=CFG.seed, noise=CFG.init_noise, tol=CFG.init_tol)
    if CFG.untie_config:
        _untie_fn()(base)
    if CFG.chat_template and template is not None:
        install_sem_chat_template(template.processor)

    lm = _language_model(base)

    def lm_hook(module, args, output):
        b = base.__dict__.get('_sem_batch')
        if b is None or b.h is not None or b.n == 0 or not _STATE['enabled']:
            return output
        h = getattr(output, 'last_hidden_state', None)
        if h is None:
            h = output[0] if isinstance(output, (tuple, list)) else output
        if tuple(h.shape[:2]) != b.shape:
            _warn_once('hshape', f'[sem-latent] hidden states {tuple(h.shape)} vs ids {b.shape} -> no SEM loss')
            return output
        b.h = h[b.b_idx[:, None], b.pos]              # [n_blk, K, H]
        return output

    def loss_hook(module, args, kwargs, out):
        b = module.__dict__.pop('_sem_batch', None)
        if not _STATE['enabled']:
            return out
        loss = getattr(out, 'loss', None)
        if loss is None:
            if b is not None and b.n and bool(b.ok.any()):
                msg = ('[sem-latent] the model returned loss=None on a micro-batch with supervised SEM blocks: swift '
                       'computes the loss OUTSIDE the model (it pops `labels` for a non-binary --loss_scale such as '
                       'qure_graded, --loss_type, label smoothing, --enable_dft_loss / --enable_channel_loss, sequence '
                       'parallel), so the SEM term cannot be added to out.loss and SEM would silently train nothing.')
                if module.training:
                    raise SemConfigError(msg + ' Refusing: use a binary loss_scale (e.g. ignore_empty_think) and none '
                                         'of those flags.')
                _warn_once('loss_none_eval', msg + ' (eval: no SEM term and no val_sem/* record for such batches)')
            return out
        term, stats = compute_sem_term(module, b, loss, module.training)
        total = loss + term.to(loss.dtype)
        out.loss = total
        _STATE['last'] = dict(stats, ce=float(loss.detach()), total=float(total.detach()))
        _STATE['last_term'] = term if _STATE['keep_term'] else None
        if module.training:
            _STATE['n_micro'] += 1
            _record_run(stats, float(total.detach()))
            if (_STATE['n_micro'] - 1) % CFG.log_every == 0:
                logger.info(
                    f'[sem-latent] step={_SCHED["step"]} w_cos={stats["w_cos"]:.4f} w_nce={stats["w_nce"]:.4f} | '
                    f'valid {stats.get("n_valid", 0)}/{stats.get("n_blk", 0)} blocks (miss {stats.get("n_miss", 0)}, '
                    f'leak {stats.get("n_leaked", 0)}) | cos_loss={stats.get("cos_loss", float("nan")):.4f} '
                    f'nce={stats.get("nce_loss", float("nan")):.4f} kl={stats.get("kl", float("nan")):.4f} '
                    f'cos_c={stats.get("cos_centred", float("nan")):.3f} raw={stats.get("cos_raw", float("nan")):.3f} | '
                    f'sem term {stats.get("term", 0.0):.4f} of total {float(total.detach()):.4f} '
                    f'| detached={int(stats.get("detached", 0))} queue={len(_queue())}')
        return out

    handles = [lm.register_forward_hook(lm_hook), base.register_forward_hook(loss_hook, with_kwargs=True)]
    base.__dict__['_sem_hook_handles'] = handles
    base.__dict__['_sem_ids'] = tuple(sem_ids)
    base._sem_attached = True
    _STATE['model'] = base
    n = sum(p.numel() for p in head.parameters())
    logger.info(f'[sem-latent] sem_head attached: {in_dim}x{K} -> {CFG.pool} -> {CFG.head_hidden} -> {reader.dim} '
                f'({n:,} params, {CFG.head_dtype}) | hooks on language_model + model | {CFG.summary()}')
    return model


def add_sem_param_group(opt, model) -> int:
    """Identity-guarded add_param_group for sem_head (create_optimizer runs twice). -> #params added.

    The group is tagged `sem_head: True`. swift's name-prefix ('multimodal') optimizer, used when --vit_lr /
    --aligner_lr is set (the stage-7 launcher), never contains the head; HF's default create_optimizer
    (neither flag set) puts EVERY trainable param into its decay / no-decay groups, sem_head included --
    there the head params are MOVED out of those groups into the SEM group, so SEM_HEAD_LR and weight
    decay 0 apply (runs inside create_optimizer: before any optimizer state and before the scheduler).
    """
    if opt is None:
        return 0
    head = getattr(_owner(model), 'sem_head', None)
    if head is None:
        return 0
    want = [p for p in head.parameters() if p.requires_grad]
    if not want:
        return 0
    head_ids = {id(p) for p in want}
    moved = 0
    for g in list(opt.param_groups):
        if g.get('sem_head'):
            continue
        keep = [p for p in g['params'] if id(p) not in head_ids]
        if len(keep) != len(g['params']):
            moved += len(g['params']) - len(keep)
            logger.warning(f'[sem-latent] {len(g["params"]) - len(keep)} sem_head tensor(s) were in an optimizer group '
                           f'at lr={g.get("lr")} wd={g.get("weight_decay")} (the default optimizer groups every '
                           f'trainable param) -> moved to the SEM group (lr={CFG.head_lr}, wd 0)')
            g['params'] = keep
    if moved:
        opt.param_groups[:] = [g for g in opt.param_groups if g['params'] or g.get('sem_head')]
    in_opt = {id(p) for g in opt.param_groups for p in g['params']}
    ps = [p for p in want if id(p) not in in_opt]
    if not ps:
        logger.info('[sem-latent] sem_head params already in the optimizer (identity guard)')
        return 0
    n = sum(p.numel() for p in ps)
    opt.add_param_group({'params': ps, 'lr': CFG.head_lr, 'initial_lr': CFG.head_lr, 'weight_decay': 0.0,
                         'sem_head': True})
    logger.info(f'[sem-latent] sem_head param group ADDED: {len(ps)} tensors ({n:,} params) at lr={CFG.head_lr}')
    return n


# =========================================================================================
# eval metrics (pure numpy)
# =========================================================================================
def _np_centre(v, mu):
    x = np.asarray(v, np.float32) - np.asarray(mu, np.float32)[None]
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)


def _np_unit(v):
    x = np.asarray(v, np.float32)
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)


def auroc(score, y) -> float:
    """Mann-Whitney AUROC with average ranks for ties (nan if one class is empty)."""
    s = np.asarray(score, np.float64)
    y = np.asarray(y, bool)
    npos, nneg = int(y.sum()), int((~y).sum())
    if npos == 0 or nneg == 0:
        return float('nan')
    _, inv, cnt = np.unique(s, return_inverse=True, return_counts=True)
    avg = np.cumsum(cnt) - (cnt - 1) / 2.0
    r = avg[inv]
    return float((r[y].sum() - npos * (npos + 1) / 2.0) / (npos * nneg))


def retrieval_metrics(zh, zt, keys, mu, query_mask=None, dup_thresh: float = 0.95) -> Dict[str, float]:
    """Duplicate-aware retrieval of each prediction among the UNIQUE eval targets."""
    keys = np.asarray(keys)
    ch, ct = _np_centre(zh, mu), _np_centre(zt, mu)
    zhn, ztn = _np_unit(zh), _np_unit(zt)
    cos_c = (ch * ct).sum(1)
    cos_raw = (zhn * ztn).sum(1)
    muh = np.asarray(mu, np.float32) / max(float(np.linalg.norm(mu)), 1e-12)
    base_raw = ztn @ muh
    uk, first = np.unique(keys, return_index=True)
    gc, graw = ct[first], ztn[first]
    S = ch @ gc.T
    pos = (keys[:, None] == uk[None, :]) | ((ct @ gc.T) > dup_thresh)
    best = np.where(pos, S, -np.inf).max(1)
    rank = 1 + ((S > best[:, None]) & ~pos).sum(1)
    Sraw = zhn @ graw.T
    oth = ~pos
    n_oth = np.maximum(oth.sum(1), 1)
    margin_raw = cos_raw - (Sraw * oth).sum(1) / n_oth
    margin_c = cos_c - (S * oth).sum(1) / n_oth
    qm = np.ones(len(keys), bool) if query_mask is None else np.asarray(query_mask, bool)
    if not qm.any():
        return {'n': 0}
    out = {'n': int(qm.sum()), 'n_gallery': int(len(uk)), 'cos_centred': float(cos_c[qm].mean()),
           'cos_raw': float(cos_raw[qm].mean()), 'cos_raw_meanpred': float(base_raw[qm].mean()),
           'median_rank': float(np.median(rank[qm])), 'margin': float(margin_raw[qm].mean()),
           'margin_centred': float(margin_c[qm].mean())}
    for k in (1, 5, 10):
        out[f'r{k}'] = float((rank[qm] <= k).mean())
    out['_rank'] = rank
    out['_cos_c'] = cos_c
    out['_cos_raw'] = cos_raw
    return out


def load_probe(path: Optional[str] = None):
    path = st.resolve_storage_path(path or CFG.probe_head or st.default_probe_head_path())
    if _STATE['probe'] is not None and _STATE['probe_path'] == path:
        return _STATE['probe']
    if not path or not os.path.isfile(path):
        return None
    try:
        ck = torch.load(path, map_location='cpu', weights_only=True)
    except Exception:  # noqa: BLE001 - trusted local bundle (numpy arrays inside)
        ck = torch.load(path, map_location='cpu', weights_only=False)
    sd = ck['state_dict']
    std_ = bool(ck.get('standardize', False))
    pr = {'W': np.asarray(sd['fc.weight'].float().numpy()), 'b': np.asarray(sd['fc.bias'].float().numpy()),
          'mean': np.asarray(ck['mean'], np.float32).reshape(1, -1) if std_ else None,
          'std': np.asarray(ck['std'], np.float32).reshape(1, -1) if std_ else None,
          'tags': list(ck.get('tags') or []), 'thr': {k: float(v) for k, v in (ck.get('thresholds') or {}).items()}}
    _STATE['probe'], _STATE['probe_path'] = pr, path
    return pr


def probe_logits(pr, z):
    x = np.asarray(z, np.float32)
    if pr['mean'] is not None:
        x = (x - pr['mean']) / pr['std']
    return x @ pr['W'].T + pr['b']


def probe_metrics(zh, zt, pr, min_pos: int = 50) -> Dict[str, float]:
    """lp23sep head on UNcentred z^ vs labels probe(z*) >= thr: per-tag AUROC, macro over n_pos >= min_pos."""
    lh, lt = probe_logits(pr, zh), probe_logits(pr, zt)
    thr = np.array([pr['thr'].get(t, 0.5) for t in pr['tags']] if pr['tags'] else [0.5] * lh.shape[1], np.float64)
    thr_logit = np.log(np.clip(thr, 1e-12, 1 - 1e-12) / np.clip(1 - thr, 1e-12, 1))
    y = lt >= thr_logit[None]
    aucs = {}
    for j in range(lh.shape[1]):
        npos = int(y[:, j].sum())
        if npos >= min_pos and npos < y.shape[0]:
            aucs[pr['tags'][j] if pr['tags'] else str(j)] = auroc(lh[:, j], y[:, j])
    out = {'n_tags_scored': len(aucs), 'n_tags': int(lh.shape[1])}
    if aucs:
        out['macro_auc'] = float(np.mean(list(aucs.values())))
    return out


def _drain_eval():
    if _EVAL['zh']:
        local = tuple(np.concatenate(_EVAL[k]) for k in ('zh', 'zt', 'key', 'img'))
    else:
        d = get_reader(required=False)
        dim = d.dim if d is not None else st.EMB_DIM
        local = (np.zeros((0, dim), np.float16), np.zeros((0, dim), np.float16), np.zeros(0, np.int64),
                 np.zeros(0, np.int64))
    for v in _EVAL.values():
        v.clear()
    return local


def evaluate_payload(step: int, out_dir: Optional[str] = None, is_rank0: bool = True,
                     gathered=None) -> Dict[str, float]:
    """val_sem/* payload from the recorded eval blocks (gathered = list of per-rank tuples)."""
    parts = gathered if gathered is not None else [_drain_eval()]
    zh = np.concatenate([p[0] for p in parts]).astype(np.float32)
    zt = np.concatenate([p[1] for p in parts]).astype(np.float32)
    keys = np.concatenate([p[2] for p in parts]).astype(np.int64)
    imgs = np.concatenate([p[3] for p in parts]).astype(np.int64)
    if zh.shape[0] == 0:
        logger.warning('[sem-latent] eval produced no valid SEM targets -- no val_sem metrics')
        return {}
    reader = get_reader()
    mu = reader.mu
    rows = reader.lookup(_i64_to_u64(keys))
    in_train = reader.in_train(rows)
    payload: Dict[str, float] = {}
    m_all = retrieval_metrics(zh, zt, keys, mu, dup_thresh=CFG.dup_thresh)
    for k, v in m_all.items():
        if not k.startswith('_'):
            payload[f'val_sem/{k}'] = v
    m_un = retrieval_metrics(zh, zt, keys, mu, query_mask=~in_train, dup_thresh=CFG.dup_thresh)
    for k, v in m_un.items():
        if not k.startswith('_'):
            payload[f'val_sem_unseen/{k}'] = v
    pr = load_probe()
    if pr is not None and pr['W'].shape[1] == zh.shape[1]:
        pm = probe_metrics(zh, zt, pr, min_pos=CFG.probe_min_pos)
        payload.update({f'val_sem_probe/{k}': v for k, v in pm.items()})
    if is_rank0:
        try:
            d = out_dir or CFG.eval_pred_dir or 'eval_predictions'
            os.makedirs(d, exist_ok=True)
            fn = os.path.join(d, f'sem_step{int(step):06d}.npz')
            np.savez_compressed(fn, z_hat=zh.astype(np.float16), keys=_i64_to_u64(keys), img_keys=_i64_to_u64(imgs),
                                in_train=in_train.astype(np.uint8), rank=m_all['_rank'].astype(np.int32),
                                cos_centred=m_all['_cos_c'].astype(np.float32),
                                cos_raw=m_all['_cos_raw'].astype(np.float32), step=int(step))
            logger.info(f'[sem-latent] saved eval predictions -> {fn} ({zh.shape[0]} blocks)')
        except Exception as e:  # noqa: BLE001 - a dump failure must never take down a run
            logger.warning(f'[sem-latent] eval-prediction dump failed ({type(e).__name__}: {e})')
    logger.info(f'[sem-latent] VAL n={m_all["n"]} cos_c={m_all["cos_centred"]:.4f} cos_raw={m_all["cos_raw"]:.4f} '
                f'(mean-predictor {m_all["cos_raw_meanpred"]:.4f}) R@1/5/10={m_all["r1"]:.3f}/{m_all["r5"]:.3f}/'
                f'{m_all["r10"]:.3f} median rank {m_all["median_rank"]:.0f}/{m_all["n_gallery"]} margin '
                f'{m_all["margin"]:.4f} | unseen n={m_un.get("n", 0)}'
                + (f' | probe macro AUC {payload["val_sem_probe/macro_auc"]:.4f} over '
                   f'{payload["val_sem_probe/n_tags_scored"]} tags' if 'val_sem_probe/macro_auc' in payload else ''))
    return payload


def train_payload() -> Dict[str, float]:
    """train/sem_* means since the last call (rank-local), then reset."""
    r = dict(_RUN)
    _RUN.clear()
    if not r.get('micro'):
        return {}
    n = max(r.get('n', 0), 1)
    blk = max(r.get('n_blk', 0), 1)
    # eligible T1 turns = valid (incl. online-filled) + residual misses; n_online were misses before the fill
    t1 = max(r.get('n_valid', 0) + r.get('n_miss', 0), 1)
    w_cos, w_nce = current_weights()
    p = {'train/sem_w_cos': w_cos, 'train/sem_w_nce': w_nce,
         'train/sem_valid_frac': r.get('n_valid', 0) / blk,
         'train/sem_miss_rate': (r.get('n_miss', 0) + r.get('n_online', 0)) / t1,
         'train/sem_leak_frac': r.get('n_leaked', 0) / blk,
         'train/sem_mismatch_frac': (r.get('n_mismatch', 0) + r.get('n_error', 0)) / blk,
         'train/sem_queue_fill': float(len(_queue())),
         'train/sem_detached': float(_SCHED['step'] < CFG.detach_steps)}
    if CFG.online_teacher:
        p['train/sem_online_frac'] = r.get('n_online', 0) / t1
    if r.get('n'):
        p.update({'train/sem_cos_loss': r['cos_loss'] / n, 'train/sem_nce_loss': r['nce_loss'] / n,
                  'train/sem_cos_centred': r['cos_centred'] / n, 'train/sem_cos_raw': r['cos_raw'] / n,
                  'train/sem_share': r['share'] / n})
        if r.get('n_kl'):
            p['train/sem_kl'] = r['kl'] / r['n_kl']
        if r.get('n_ce'):
            p['train/sem_nce_ce'] = r['nce_ce'] / r['n_ce']
    return p


# =========================================================================================
# patches
# =========================================================================================
def patch_template():
    if not _once('sem_latent_plugin.patch_template'):
        return  # F1: already patched in this process
    from swift.template.base import Template
    from swift.template.templates.qwen import Qwen3_5Template

    orig_encode_public = Qwen3_5Template.encode
    orig_base_encode = Template._encode
    orig_collator = Qwen3_5Template._data_collator
    orig_post_encode = Qwen3_5Template._post_encode

    def encode(self, inputs, *args, **kwargs):
        """Raw row -> thread-local pending (image key, overrides). *args: serving engines call
        encode(inputs, return_template_inputs) positionally."""
        prev = getattr(_TLS, 'pending', None)
        _TLS.pending = pending_from_raw(inputs) if isinstance(inputs, dict) else None
        try:
            return orig_encode_public(self, inputs, *args, **kwargs)
        finally:
            _TLS.pending = prev

    def _encode(self, inputs):
        encoded = orig_base_encode(self, inputs)
        if self.is_training and getattr(self, 'task_type', 'causal_lm') == 'causal_lm' \
                and isinstance(encoded.get('input_ids'), list):
            sem_ids = get_sem_ids(self.tokenizer)
            if sem_ids:
                attach_row_targets(self, inputs, encoded, sem_ids)
        return encoded

    def _data_collator(self, batch, *, padding_to=None):
        res = orig_collator(self, batch, padding_to=padding_to)
        if self.is_training and batch and isinstance(batch[0], dict):
            sem_ids = get_sem_ids(self.tokenizer)
            if sem_ids:
                r = get_reader(required=False)
                dim = r.dim if r is not None else st.EMB_DIM
                res.update(collate_sem(batch, sem_ids, dim, online=CFG.online_teacher))
        return res

    def _post_encode(self, model, inputs):
        popped = {k: inputs.pop(k, None) for k in SEM_KEYS if k in inputs}
        owner = _owner(model)
        owner.__dict__.pop('_sem_batch', None)          # never let a previous batch's stash survive
        if self.is_training and owner.__dict__.get('_sem_attached', False):
            ids = inputs.get('input_ids')
            sem_ids = owner.__dict__.get('_sem_ids') or get_sem_ids(self.tokenizer)
            if isinstance(ids, torch.Tensor) and sem_ids:
                owner.__dict__['_sem_batch'] = make_batch(ids, sem_ids, popped, get_reader().dim)
        return orig_post_encode(self, model, inputs)

    Qwen3_5Template.encode = encode
    Template._encode = _encode
    Qwen3_5Template._data_collator = _data_collator
    Qwen3_5Template._post_encode = _post_encode
    logger.info('[sem-latent] templates patched (Qwen3_5Template.encode/_data_collator/_post_encode, '
                'Template._encode)')


def _check_train_args(args, reader=None) -> None:
    """Refuse every training configuration under which SEM would silently train wrong or not at all.

    (the launcher precheck runs this same function on the FINAL `swift sft` argv)
      packing / padding_free / sequence parallel   per-row block <-> target alignment breaks
      --template_backend != swift                  jinja encodes the whole text at loss scale 1: the core
                                                   hook inserts nothing (0 blocks while serving inserts
                                                   them) or the SEM jinja makes every SEM id a CE label
      --loss_scale with base strategy 'all'        headers get loss scale > 0: no SEM block in training,
                                                   one after every header at inference
      non-binary --loss_scale (qure_graded ...), --loss_type, label_smoothing_factor > 0,
      --enable_dft_loss, --enable_channel_loss     swift pops `labels` and computes the loss OUTSIDE the
                                                   model: out.loss is None and the SEM term has nowhere to go
    Also WARNS when the sidecar holds `sem_targets`/`sem_target` override texts but
    --remove_unused_columns is true (swift's loader then drops those columns before encode).
    """
    if args is None:
        return
    for k in ('packing', 'padding_free'):
        if getattr(args, k, False):
            raise SemConfigError(f'[sem-latent] --{k} true breaks the per-row SEM block <-> target alignment; '
                                 f'use --{k} false')
    if int(getattr(args, 'sequence_parallel_size', 1) or 1) > 1:
        raise SemConfigError('[sem-latent] sequence parallel is not supported (SEM positions are per full row)')
    backend = getattr(args, 'template_backend', 'swift') or 'swift'
    if backend != 'swift':
        raise SemConfigError(f'[sem-latent] --template_backend {backend} is not supported for training: the jinja '
                             'backend encodes the whole text at loss scale 1, so the core hook inserts no SEM block '
                             '(or, with the SEM-aware jinja installed, every SEM id becomes a CE label). Use '
                             '--template_backend swift (inference on the jinja backend is fine).')
    ls = getattr(args, 'loss_scale', None)
    if isinstance(ls, str) and ls:
        obj = None
        try:
            from swift.loss_scale import get_loss_scale
            obj = get_loss_scale(ls)
        except Exception:  # noqa: BLE001 - an unknown loss_scale fails in swift itself
            obj = None
        if obj is not None and getattr(obj, 'base_strategy', None) == 'all':
            raise SemConfigError(f'[sem-latent] --loss_scale {ls}: base strategy "all" gives the assistant headers a '
                                 'loss scale > 0, so NO SEM block is inserted in training while every inference path '
                                 'inserts one after every header. Use a default/last_round strategy (e.g. '
                                 'ignore_empty_think).')
        if obj is not None and getattr(obj, 'is_binary_loss_scale', True) is False:
            raise SemConfigError(f'[sem-latent] --loss_scale {ls} is non-binary: swift then pops `labels` and computes '
                                 'the loss outside the model, so out.loss is None and the SEM term cannot be added '
                                 '(SEM would train nothing). Use a binary loss_scale (e.g. ignore_empty_think).')
    popped = []
    if getattr(args, 'loss_type', None):
        popped.append(f'--loss_type {args.loss_type}')
    for flag in ('enable_dft_loss', 'enable_channel_loss'):
        if getattr(args, flag, False):
            popped.append(f'--{flag}')
    if float(getattr(args, 'label_smoothing_factor', 0) or 0) > 0:
        popped.append(f'--label_smoothing_factor {args.label_smoothing_factor}')
    if popped:
        raise SemConfigError(f'[sem-latent] {", ".join(popped)}: swift then pops `labels` and computes the loss outside '
                             'the model, so out.loss is None and the SEM term cannot be added (SEM would train '
                             'nothing). Drop these flags for a SEM run.')
    if getattr(args, 'remove_unused_columns', True):
        try:
            r = reader if reader is not None else get_reader()
            fam = (r.meta.get('n_texts_per_family') or {}) if r is not None else {}
            n_ov = int(fam.get(st.OVERRIDE_FAMILY, 0) or 0)
        except SemConfigError:
            raise
        except Exception:  # noqa: BLE001 - meta without per-family counts
            n_ov = 0
        if n_ov > 0:
            _warn_once('override_cols', f'[sem-latent] the sidecar holds {n_ov} `sem_targets`/`sem_target` override '
                       'texts, but --remove_unused_columns is true: swift\'s loader drops those columns before encode, '
                       'so the overrides are IGNORED (the T1 rule applies instead). Pass --remove_unused_columns false '
                       '(the stage-7 launcher adds it automatically when the sidecar has overrides).')


def patch_prepare_model():
    if not _once('sem_latent_plugin.patch_prepare_model'):
        return
    from swift.pipelines.train.tuner import TunerMixin
    orig = getattr(TunerMixin.prepare_model, '__func__', TunerMixin.prepare_model)

    def prepare_model(cls, args, model, **kwargs):
        model = orig(cls, args, model, **kwargs)      # mldecoder & co attach (and register hooks) first
        _check_train_args(args)
        attach(model, template=kwargs.get('template'))
        return model

    TunerMixin.prepare_model = prepare_model
    logger.info('[sem-latent] prepare_model wrapped (sem_head lands before the optimizer; hooks after mldecoder)')


def patch_optimizer():
    if not _once('sem_latent_plugin.patch_optimizer'):
        return
    from swift.trainers.mixin import SwiftMixin
    orig = SwiftMixin.create_optimizer

    def create_optimizer(self, model=None):
        opt = orig(self, model=model)
        add_sem_param_group(opt, self.model)
        return opt

    SwiftMixin.create_optimizer = create_optimizer
    logger.info(f'[sem-latent] create_optimizer patched (sem_head lr={CFG.head_lr})')


def make_probe_callback(trainer=None):
    """The TrainerCallback (step probe + train/sem_* logging + val_sem/* eval). `trainer` is the
    SwiftMixin instance (trainer.log / trainer.template); None is fine for tests."""
    from transformers import TrainerCallback

    class _SemProbe(TrainerCallback):

        def on_train_begin(self, args, state, control, **kwargs):
            _SCHED['total'] = state.max_steps
            tr = getattr(self, '_trainer', None)
            tpl = getattr(tr, 'template', None)
            if CFG.chat_template and tpl is not None:
                install_sem_chat_template(tpl.processor)
            logger.info(f'[sem-latent] train begin: {CFG.summary()} | max_steps={state.max_steps}')

        def on_step_begin(self, args, state, control, **kwargs):
            _SCHED['step'] = state.global_step

        def on_step_end(self, args, state, control, **kwargs):
            every = max(int(getattr(args, 'logging_steps', 1) or 1), 1)
            if state.global_step % every != 0 or not _RUN.get('micro'):
                return
            payload = train_payload()
            trainer = getattr(self, '_trainer', None)
            if trainer is not None and payload:
                # trainer.log -> on_log sets control.should_log=False; restore it so the trainer's own
                # {loss, grad_norm, learning_rate} row for this step is not skipped (mldecoder pattern)
                should_log = getattr(control, 'should_log', None)
                trainer.log(payload)
                if should_log is not None:
                    control.should_log = should_log
            return control

        def on_evaluate(self, args, state, control, metrics=None, **kwargs):
            local = _drain_eval()
            parts = [local]
            is_rank0 = True
            try:
                import torch.distributed as dist
                if dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1:
                    is_rank0 = dist.get_rank() == 0
                    parts = [None] * dist.get_world_size()
                    dist.all_gather_object(parts, local)      # every rank, even with 0 blocks
            except Exception as e:  # noqa: BLE001
                logger.warning(f'[sem-latent] eval gather failed ({e}); scoring this rank\'s shard only')
                parts = [local]
            step = int(getattr(state, 'global_step', 0))
            out_dir = CFG.eval_pred_dir or os.path.join(getattr(args, 'output_dir', '.'), 'eval_predictions')
            try:
                payload = evaluate_payload(step, out_dir=out_dir, is_rank0=is_rank0, gathered=parts)
            except Exception as e:  # noqa: BLE001 - diagnostics must never kill training
                logger.warning(f'[sem-latent] eval metrics failed ({type(e).__name__}: {e})')
                payload = {}
            trainer = getattr(self, '_trainer', None)
            if trainer is not None and payload:
                trainer.log(payload)

    cb = _SemProbe()
    cb._trainer = trainer
    return cb


def patch_trainer():
    if not _once('sem_latent_plugin.patch_trainer'):
        return
    from swift.trainers.mixin import SwiftMixin

    orig_init = SwiftMixin.__init__

    def __init__(self, *a, **kw):
        orig_init(self, *a, **kw)
        if CFG.chat_template and getattr(self, 'template', None) is not None:
            install_sem_chat_template(self.template.processor)
        self.add_callback(make_probe_callback(self))

    SwiftMixin.__init__ = __init__
    logger.info('[sem-latent] trainer callback registered (train/sem_*, val_sem/*)')


def install() -> None:
    patch_template()
    patch_prepare_model()
    patch_optimizer()
    patch_trainer()
    logger.info(f'[sem-latent] plugin loaded (train) -- list it LAST; --packing false; {CFG.summary()}')


try:
    from swift_plugins.qure_plugin_mode import is_inference  # noqa: E402
except ImportError:  # pragma: no cover
    from qure_plugin_mode import is_inference  # noqa: E402

if is_inference():
    # SEM insertion lives in swift core (tokenizer-gated) and needs nothing from here at inference.
    logger.info('[sem-latent] inference mode -- no patches installed (SEM insertion is in swift core)')
else:
    install()
