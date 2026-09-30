#!/usr/bin/env python
"""SEM latent-token training plugin: swift_plugins/sem_latent_plugin.py (spec S4).

RUN (CPU only; standalone or under pytest)
    CUDA_VISIBLE_DEVICES="" PYTHONPATH=/fastdata4/sahil_work/VLM_qure/vlm/swift \\
        /home/ubuntu/miniconda3/envs/ms-swift-may-s4/bin/python tests/qure/test_sem_plugin.py [-k substr]
Exits non-zero on any failure; prints a PASS/FAIL table at the end.

WHAT IS PROVEN
  (1) UNIT (no model): SemanticHead shapes for attn / mean / concat (+ unit norm, n=0, sglang-safe
      param names); the loss math (L_cos, soft CE with KL, hard InfoNCE with the same-key / same-image
      / teacher-cos false-negative masks, queue candidates) against an independent float64 numpy
      reference on random anisotropic data; queue FIFO order + key alignment; centring (a constant
      mean predictor scores high raw cos but ~1 centred loss); SEM_DETACH_STEPS blocks gradients into
      the hidden states and not into the head; zero-valid and no-SEM micro-batches give a finite term
      and non-None sem_head grads; weight schedules; block finder == sem_latent.find_sem_blocks;
      collator recount (truncated block -> unsupervised row); post-encode consistency check; the
      identity-guarded optimizer group; sem_head restore via safe_open (round trip, mismatch raises);
      SEM embedding-row init (deterministic, only for identical spare rows, lm_head untouched);
      startup refusals (no SEM tokens -> the exact --new_special_tokens fix, K > 109, resized
      embeddings, rule_version / fingerprint / missing sidecar); eval metrics (duplicate-aware R@k,
      AUROC vs sklearn); the online-teacher fill with a stub teacher (+ its cache); the SEM_NCE_GATHER
      all_gather on a real 2-process gloo group; inference mode installs nothing; dedup-first.
  (1b) REAL TEMPLATE: 4 real rows (an onto report row, a B3 tool row, a template-G multi-section row,
      a row without any T1 turn) through the real ckpt-608 processor + SEM tokens in train mode with the
      plugin's patches: per-block (status, key) equals an INDEPENDENT derivation from the raw rows,
      targets equal the sidecar rows, image keys, overrides (sem_targets / sem_target), loss:false,
      leak guard, cache misses, collator recount.
  (2) TOY END-TO-END (subprocesses): a tiny random Qwen3_5ForConditionalGeneration (4 layers: 3
      linear-attention + 1 full-attention, hidden 64; 2-block ViT) saved as a checkpoint dir with the
      ckpt-608 tokenizer, loaded through swift get_model_processor(new_special_tokens=token.txt +
      token_sem.txt), the REAL template, collator, pre_forward_hook/_post_encode, forward, backward:
        * blocks found == merged assistant turns; valid mapping == the independent derivation
        * loss(plugin on) - loss(plugin off) == w_cos*L_cos + w_nce*L_nce, and == an independent
          recomputation from the captured last_hidden_state with the numpy reference
        * during SEM_DETACH_STEPS the SEM term has NO gradient into decoder layers / embed rows but
          does into sem_head; after it, gradients reach sem_head, the SEM embedding rows and >= 1
          decoder layer; eval forward -> val_sem payload + npz dump; untied config; SEM chat template;
          SEM rows re-initialised; optimizer group
        * composition with mldecoder_joint + qure_augment + qure_ce_weight + qure_stage3 loaded in the
          stage-6c order, SEM LAST: mldecoder's loss hook runs first and its logged `ar` is the CE
          without the SEM term; total = mldecoder loss + SEM term.

ENV
  SEM_TEST_CKPT   checkpoint whose tokenizer/processor is used (default stage-6b checkpoint-608)
  SEM_TEST_OUT    artefact dir (default <scratchpad>/tests_out/sem_plugin)
Nothing under the repo, the checkpoint or the datasets is written.
"""
import copy
import json
import os
import random
import shutil
import subprocess
import sys
import tempfile
import time
import traceback
from types import SimpleNamespace

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SCRATCH = '/tmp/claude-1000/-fastdata4-sahil-work-VLM-qure-vlm-swift/b2c1bd9c-54da-4ec4-b14d-2d8473057a20/scratchpad'
if not os.path.isdir(SCRATCH):  # another host / session: artefacts go to the system tmp dir
    SCRATCH = os.path.join(tempfile.gettempdir(), 'qure_sem_tests')
CKPT = os.environ.get('SEM_TEST_CKPT') or ('/fastdata4/sahil_work/VLM_qure/data_preparation/training_data/output/'
                                           'Qwen3.5-2B_stage6b_alltools/v0-20260906-173237/checkpoint-608')
OUT = os.environ.get('SEM_TEST_OUT') or os.path.join(SCRATCH, 'tests_out', 'sem_plugin')
DATA = '/fastdata4/sahil_work/VLM_qure/data_preparation/training_data'
FILES = {'A': f'{DATA}/stage6_toolcalling/A_TRAIN_120k.jsonl', 'B3': f'{DATA}/stage6_toolcalling/B3_TRAIN.jsonl'}
TOKEN_TXT = os.path.join(REPO, 'examples/train/liger/token.txt')
TOKEN_SEM_TXT = os.path.join(REPO, 'examples/train/liger/token_sem.txt')
SYS = 'You are a helpful medical assistant.'
TEMPLATE_KW = dict(default_system=SYS, max_length=28000, truncation_strategy='raise', loss_scale='ignore_empty_think',
                   add_non_thinking_prefix=True, template_type='qwen3_5')
FORBIDDEN = ('q_proj', 'k_proj', 'v_proj', 'gate_proj', 'up_proj', 'in_proj_', 'visual', 'language_model',
             'embed_tokens')



def _apply_test_env():
    """This module's process env. Applied only when the module RUNS (standalone, --scenario children) or, under
    pytest, around ITS tests (module fixture below, restored afterwards) -- never at import, so collecting tests/
    (pytest, tests/run.py) changes nothing for any other test (review BC-3). test_sem_integration_cpu reuses it."""
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    os.environ.setdefault('QURE_PLUGIN_MODE', 'train')
    os.environ.setdefault('OMP_NUM_THREADS', '16')
    os.environ['SKIP_BROKEN_IMAGES'] = '1'
    os.environ.setdefault('IMAGE_MAX_TOKEN_NUM', '64')
    os.environ.pop('SWIFT_SEM_LATENT', None)


def module_env_fixture():
    """A module-scoped autouse pytest fixture applying _apply_test_env() and restoring the env afterwards
    (None when pytest is unavailable)."""
    try:
        import pytest
    except ImportError:
        return None

    @pytest.fixture(scope='module', autouse=True)
    def _qure_module_env():
        saved = dict(os.environ)
        _apply_test_env()
        try:
            yield
        finally:
            os.environ.clear()
            os.environ.update(saved)

    return _qure_module_env


if __name__ == '__main__':
    _apply_test_env()
else:
    _qure_module_env = module_env_fixture()
if REPO not in sys.path:
    sys.path.insert(0, REPO)

_REPORT = []
_CACHE = {}


def _rep(section, check, ok, detail=''):
    _REPORT.append((section, check, 'PASS' if ok else 'FAIL', str(detail)[:300]))
    assert ok, f'[{section}] {check}: {detail}'


# ============================================================================================ shared helpers
def st_mod():
    import swift_plugins.sem_targets as st
    return st


def disable_fla():
    """fla / causal-conv1d kernels need CUDA (FusedRMSNormGated is built on cuda.current_device())."""
    import transformers.models.qwen3_5.modeling_qwen3_5 as M
    for n in ('FusedRMSNormGated', 'causal_conv1d_fn', 'causal_conv1d_update', 'chunk_gated_delta_rule',
              'fused_recurrent_gated_delta_rule'):
        setattr(M, n, None)
    M.is_fast_path_available = False


def sem_tokens():
    toks = []
    for p in (TOKEN_TXT, TOKEN_SEM_TXT):  # exactly `--new_special_tokens token.txt token_sem.txt`
        with open(p, encoding='utf-8') as f:
            toks += f.read().split()
    return toks


def to_swift_row(d):
    R = {'human': 'user', 'gpt': 'assistant', 'system': 'system'}
    msgs = []
    for m in d['messages']:
        role = m.get('role') or R.get(m.get('from'), m.get('from'))
        mm = {'role': role, 'content': m['content'] if 'content' in m else m.get('value')}
        for k in ('loss', 'loss_scale'):
            if k in m:
                mm[k] = m[k]
        msgs.append(mm)
    row = {'messages': msgs, 'images': [{'bytes': None, 'path': p} for p in (d.get('images') or [])]}
    for k in ('tools', 'sem_targets', 'sem_target'):
        if d.get(k) is not None:
            row[k] = d[k]
    return row


def _images_ok(paths):
    from PIL import Image
    from swift.template.vision_utils import _resolve_image_path
    for p in paths:
        rp = _resolve_image_path(p)
        if not rp or not os.path.isfile(rp):
            return False
        try:
            with Image.open(rp) as im:
                im.size  # noqa: B018
        except Exception:  # noqa: BLE001
            return False
    return True


def _roles(d):
    return [m.get('role') or m.get('from') for m in d['messages']]


def _nchars(d):
    return sum(len(str(m.get('content') or m.get('value') or '')) for m in d['messages']) + len(d.get('tools') or '')


def _fams(d):
    return [f for _, f, _ in st_mod().iter_row_targets(d)]


WANT = [
    # (id, file, predicate)
    ('onto_plain', 'B3', lambda d: _roles(d) == ['user', 'assistant'] and _fams(d) == ['onto'] and _nchars(d) < 6000),
    ('b3_tool', 'B3', lambda d: 'tool_call' in _roles(d) and 'tool' not in _roles(d) and _fams(d) == ['onto']
     and d.get('family') in ('N1r', 'N3', 'W9') and _nchars(d) < 7000),
    ('tplG', 'A', lambda d: d.get('template') == 'G' and sum(f.startswith('sec_') for f in _fams(d)) >= 2
     and _nchars(d) < 20000),
    ('no_t1', 'A', lambda d: not _fams(d) and 'bbox_2d' not in json.dumps(d['messages'])
     and _roles(d).count('user') >= 2 and _nchars(d) < 16000),
]


def pick_rows():
    """[(id, raw_row)] -- sampled once (seeded), cached in OUT/rows.jsonl for every later run/subprocess."""
    if 'rows' in _CACHE:
        return _CACHE['rows']
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, 'rows.jsonl')
    if os.path.isfile(path):
        rows = [(r['id'], r['raw']) for r in map(json.loads, open(path))]
    else:
        rng = random.Random(20260930)
        rows = []
        for rid, fk, pred in WANT:
            fp = FILES[fk]
            size = os.path.getsize(fp)
            found = None
            with open(fp, 'rb') as f:
                for _ in range(20000):
                    f.seek(rng.randrange(size))
                    f.readline()
                    line = f.readline()
                    if not line.strip():
                        continue
                    try:
                        d = json.loads(line)
                    except ValueError:
                        continue
                    if pred(d) and _images_ok(d.get('images') or []):
                        found = d
                        break
            assert found is not None, f'no row for {rid} in {fp}'
            rows.append((rid, found))
        with open(path, 'w') as f:
            for rid, d in rows:
                f.write(json.dumps({'id': rid, 'raw': d}, ensure_ascii=False) + '\n')
    _CACHE['rows'] = rows
    return rows


def expected_blocks(raw, sidecar_keys, skip_leaked=True, skip_loss_false=True):
    """INDEPENDENT per-block (status, key) from the RAW row: one block per assistant_turn_group; the
    group's LAST message carries the target; prior = raw messages before the group's first message."""
    st = st_mod()
    msgs = st.row_messages(raw)
    groups = st.assistant_turn_groups(msgs)
    tg = {i: (f, c) for i, f, c in st.iter_row_targets(raw) if f != st.OVERRIDE_FAMILY}
    ov = [None] * len(groups)
    if isinstance(raw.get('sem_targets'), list):
        for j, t in enumerate(raw['sem_targets'][:len(groups)]):
            ov[j] = st.normalise_override(t)
    if st.normalise_override(raw.get('sem_target')) is not None and groups:
        ov[-1] = st.normalise_override(raw['sem_target'])
    out = []
    for j, g in enumerate(groups):
        last = g[-1]
        if ov[j] is not None:
            k = st.target_key(ov[j])
            out.append((1 if k in sidecar_keys else 2, k))
            continue
        if last not in tg:
            out.append((0, 0))
            continue
        _, can = tg[last]
        k = st.target_key(can)
        prior = [m['content'] for m in msgs[:g[0]]]
        if skip_loss_false and msgs[last].get('loss') is False:
            out.append((4, k))
        elif skip_leaked and st.is_leaked(can, prior):
            out.append((3, k))
        else:
            out.append((1 if k in sidecar_keys else 2, k))
    return out


def build_tiny_sidecar(out_dir, texts, dim=2560, seed=0, unseen=(), n_filler=0):
    """Random ANISOTROPIC unit vectors (|mu| ~ 0.7 like the teacher) for exactly `texts` (+ fillers)."""
    import numpy as np
    st = st_mod()
    rng = np.random.default_rng(seed)
    texts = list(dict.fromkeys(texts))
    keys = [st.target_key(t) for t in texts]
    fill = rng.integers(1, 2**63, size=n_filler, dtype=np.int64).astype(np.uint64).tolist()
    allk = np.asarray(keys + fill, dtype=np.uint64)
    n = len(allk)
    u0 = rng.standard_normal(dim)
    u0 /= np.linalg.norm(u0)
    g = rng.standard_normal((n, dim))
    g /= np.linalg.norm(g, axis=1, keepdims=True)
    e = 0.75 * u0[None] + 0.66 * g
    e /= np.linalg.norm(e, axis=1, keepdims=True)
    in_train = np.ones(n, np.uint8)
    for t in unseen:
        in_train[keys.index(st.target_key(t))] = 0
    if os.path.isdir(out_dir):
        shutil.rmtree(out_dir)
    st.write_sidecar(out_dir, allk, e.astype(np.float32), in_train,
                     {'rule_version': st.RULE_VERSION, 'teacher_fingerprint': 'TEST-random-unit-vectors',
                      'teacher_path': 'random', 'max_length': 1024, 'dtype': 'fp32'},
                     texts=texts + [f'filler {i}' for i in range(n_filler)])
    return {int(k): e[i] for i, k in enumerate(allk)}


OVERRIDE_TEXT = 'Findings: override target for the no-T1 row.'


def rows_and_sidecar():
    """(rows, sidecar_dir, key->vec, all texts) shared by the in-process tests and the subprocesses."""
    if 'sc' in _CACHE:
        return _CACHE['sc']
    st = st_mod()
    rows = pick_rows()
    texts = [c for _, d in rows for _, _, c in st.iter_row_targets(d)] + [OVERRIDE_TEXT]
    unseen = [texts[0]]
    d = os.path.join(OUT, 'sidecar')
    vecs = build_tiny_sidecar(d, texts, unseen=unseen, n_filler=40)
    _CACHE['sc'] = (rows, d, vecs, texts)
    return _CACHE['sc']


def plugin():
    """The plugin imported IN THIS PROCESS (train mode, patches installed) on the shared tiny sidecar."""
    if 'P' in _CACHE:
        return _CACHE['P']
    _, d, _, _ = rows_and_sidecar()
    os.environ['SEM_TARGETS_DIR'] = d
    os.environ['QURE_PLUGIN_MODE'] = 'train'
    disable_fla()
    import swift_plugins.sem_latent_plugin as P
    P.reload_config()
    _CACHE['P'] = P
    return P


def processor():
    if 'proc' not in _CACHE:
        from swift.model.register import get_processor
        _CACHE['proc'] = get_processor(CKPT, new_special_tokens=sem_tokens())
    return _CACHE['proc']


def template(proc=None):
    from swift.template import get_template
    t = get_template(proc or processor(), **TEMPLATE_KW)
    t.set_mode('train')
    return t


class _EnvPatch:

    def __init__(self, **kw):
        self.kw, self.old = kw, {}

    def __enter__(self):
        for k, v in self.kw.items():
            self.old[k] = os.environ.get(k)
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = str(v)
        plugin().reload_config()
        return self

    def __exit__(self, *a):
        for k, v in self.old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        plugin().reload_config()


# ============================================================================================ reference loss
def ref_losses(zh, zt, mu, keys, imgs, extra, mode, tau, tau_t, thr):
    """Independent float64 numpy reference (loops) -> (L_cos, L_nce, KL, CE).

    soft: L_nce IS the KL (review SEM-LOSS-1: same gradient as the CE, 0 at the optimum), CE reported
    separately; hard: L_nce = the one-hot CE, KL / CE None."""
    import numpy as np
    zh, zt, mu = (np.asarray(x, np.float64) for x in (zh, zt, mu))

    def c(v):
        x = v - mu
        return x / np.linalg.norm(x)

    ch = [c(v) for v in zh]
    ct = [c(v) for v in zt]
    cand, ck, ci = list(ct), list(keys), list(imgs)
    if extra is not None:
        cand += [np.asarray(v, np.float64) for v in extra[0]]
        ck += list(extra[1])
        ci += list(extra[2])
    n = len(ch)
    cos_loss = float(np.mean([1 - ch[i] @ ct[i] for i in range(n)]))
    ces, kls = [], []
    for i in range(n):
        s = np.array([ch[i] @ v / tau for v in cand])
        if mode == 'soft':
            t = np.array([ct[i] @ v / tau_t for v in cand])
            p = np.exp(t - t.max())
            p /= p.sum()
            logq = s - (np.log(np.exp(s - s.max()).sum()) + s.max())
            logp = np.log(p)
            ce = -(p * logq).sum()
            ces.append(ce)
            kls.append(ce + (p * logp).sum())
        else:
            keep = [j for j in range(len(cand)) if j == i or not (
                (ct[i] @ cand[j]) > thr or ck[j] == keys[i] or (ci[j] == imgs[i] and imgs[i] != 0))]
            sk = s[keep]
            lse = np.log(np.exp(sk - sk.max()).sum()) + sk.max()
            ces.append(lse - s[i])
    if mode == 'soft':
        return cos_loss, float(np.mean(kls)), float(np.mean(kls)), float(np.mean(ces))
    return cos_loss, float(np.mean(ces)), None, None


def _aniso(n, dim, seed, a=0.75):
    import numpy as np
    rng = np.random.default_rng(seed)
    u0 = rng.standard_normal(dim)
    u0 /= np.linalg.norm(u0)
    g = rng.standard_normal((n, dim))
    g /= np.linalg.norm(g, axis=1, keepdims=True)
    e = a * u0[None] + 0.66 * g
    return e / np.linalg.norm(e, axis=1, keepdims=True), u0


# ============================================================================================ (1) unit tests
def test_head_shapes():
    import torch
    P = plugin()
    for pool in ('attn', 'mean', 'concat'):
        h = P.SemanticHead(64, 40, 8, pool=pool, hidden=32)
        x = torch.randn(5, 8, 64)
        y = h(x)
        _rep('unit', f'head {pool}: [5,8,64] -> [5,40] unit fp32', tuple(y.shape) == (5, 40) and y.dtype == torch.float32
             and torch.allclose(y.norm(dim=-1), torch.ones(5), atol=1e-5), tuple(y.shape))
        y0 = h(torch.randn(0, 8, 64))
        _rep('unit', f'head {pool}: n=0', tuple(y0.shape) == (0, 40))
        names = [n for n, _ in h.named_parameters()]
        _rep('unit', f'head {pool}: param names sglang-safe', not any(f in n for n in names for f in FORBIDDEN), names)
        hb = P.SemanticHead(64, 40, 8, pool=pool, hidden=32).to(torch.bfloat16)
        _rep('unit', f'head {pool}: bf16 params, fp32 input', hb(x).dtype == torch.float32)
    h = P.SemanticHead(64, 40, 8, pool='concat', hidden=32)
    try:
        h(torch.randn(2, 4, 64))
        ok = False
    except ValueError:
        ok = True
    _rep('unit', 'concat head rejects a different K', ok)
    # attn with zero query == mean pooling
    ha = P.SemanticHead(64, 40, 8, pool='attn', hidden=32)
    hm = P.SemanticHead(64, 40, 8, pool='mean', hidden=32)
    hm.load_state_dict({k: v for k, v in ha.state_dict().items() if k != 'pool_query'})
    x = torch.randn(3, 8, 64)
    _rep('unit', 'attn head zero-init query == mean pooling', torch.allclose(ha(x), hm(x), atol=1e-6))


def test_loss_math_vs_reference():
    import numpy as np
    import torch
    P = plugin()
    rng = np.random.default_rng(1)
    n, m, dim = 7, 9, 48
    zt, u0 = _aniso(n, dim, 2)
    mu = 0.75 * u0 + 0.01 * rng.standard_normal(dim)
    zh = zt + 0.3 * rng.standard_normal((n, dim))
    zh /= np.linalg.norm(zh, axis=1, keepdims=True)
    keys = np.array([11, 12, 13, 11, 14, 15, 16])          # a duplicate key (0 and 3)
    imgs = np.array([101, 0, 102, 103, 102, 0, 104])       # a shared image (2 and 4), zeros never mask
    ev, _ = _aniso(m, dim, 3)
    ev = (ev - mu) / np.linalg.norm(ev - mu, axis=1, keepdims=True)   # queue holds CENTRED targets
    ev[0] = (zt[5] - mu) / np.linalg.norm(zt[5] - mu)      # a queue entry identical to target 5 (teacher cos 1)
    ek = np.arange(200, 200 + m)
    ek[1] = 12                                             # a queue entry with target 1's key
    ei = np.zeros(m, np.int64)
    ei[2] = 104                                            # a queue entry with target 6's image
    T = lambda a, dt=torch.float32: torch.as_tensor(a, dtype=dt)  # noqa: E731
    for mode in ('soft', 'hard'):
        for use_extra in (False, True):
            extra = (T(ev), T(ek, torch.long), T(ei, torch.long)) if use_extra else None
            L = P.sem_losses(T(zh), T(zt), T(mu), T(keys, torch.long), T(imgs, torch.long), extra=extra, mode=mode,
                             tau=0.07, tau_t=0.05, fn_thresh=0.9)
            rc, rn, rk, rce = ref_losses(zh, zt, mu, keys, imgs, (ev, ek, ei) if use_extra else None, mode, 0.07, 0.05,
                                         0.9)
            ok = abs(float(L['cos_loss']) - rc) < 1e-5 and abs(float(L['nce_loss']) - rn) < 1e-4 * max(1, abs(rn))
            if mode == 'soft':
                ok = ok and abs(float(L['kl']) - rk) < 1e-4 * max(1, abs(rk)) and float(L['kl']) >= -1e-5
                ok = ok and abs(float(L['nce_ce']) - rce) < 1e-4 * max(1, abs(rce)) and rce >= rk - 1e-6
            else:
                ok = ok and L['kl'] is None and L['nce_ce'] is None
            _rep('unit', f'loss vs float64 reference: {mode} extra={use_extra}', ok,
                 f'cos {float(L["cos_loss"]):.6f}/{rc:.6f} nce {float(L["nce_loss"]):.6f}/{rn:.6f} '
                 f'kl {None if L["kl"] is None else round(float(L["kl"]), 6)}/{rk}')
    # hard-mode masks really matter: without them the loss differs
    L1 = P.sem_losses(T(zh), T(zt), T(mu), T(keys, torch.long), T(imgs, torch.long), mode='hard', tau=0.07,
                      fn_thresh=0.9)
    L2 = P.sem_losses(T(zh), T(zt), T(mu), T(np.arange(n), torch.long), T(np.zeros(n), torch.long), mode='hard',
                      tau=0.07, fn_thresh=2.0)
    _rep('unit', 'hard FN masks change the loss (dup key / shared image)', abs(float(L1['nce_loss'])
                                                                              - float(L2['nce_loss'])) > 1e-4)
    # gradient reaches z_hat only (targets detached)
    zh_t = T(zh).requires_grad_(True)
    zt_t = T(zt).requires_grad_(True)
    L = P.sem_losses(zh_t, zt_t, T(mu), mode='soft')
    (L['cos_loss'] + L['nce_loss']).backward()
    _rep('unit', 'grad flows to z_hat, not to the teacher targets', zh_t.grad is not None and zt_t.grad is None)


def test_queue_fifo():
    import torch
    P = plugin()
    q = P.SemQueue(5)
    v = lambda a, b: torch.arange(a, b, dtype=torch.float32)[:, None].repeat(1, 3)  # noqa: E731
    k = lambda a, b: torch.arange(a, b)  # noqa: E731
    _rep('unit', 'queue empty -> get() None', q.get() is None and len(q) == 0)
    q.push(v(0, 3), k(100, 103), k(0, 3))
    g = q.get()
    _rep('unit', 'queue partial fill in order', len(q) == 3 and g[1].tolist() == [100, 101, 102])
    q.push(v(3, 7), k(103, 107), k(3, 7))
    g = q.get()
    _rep('unit', 'queue FIFO after wrap (oldest dropped, oldest-first order)',
         g[1].tolist() == [102, 103, 104, 105, 106] and g[0][:, 0].tolist() == [2, 3, 4, 5, 6]
         and g[2].tolist() == [2, 3, 4, 5, 6], g[1].tolist())
    q.push(v(7, 20), k(107, 120), k(7, 20))
    g = q.get()
    _rep('unit', 'queue push larger than size keeps the newest', g[1].tolist() == [115, 116, 117, 118, 119])
    q0 = P.SemQueue(0)
    q0.push(v(0, 3), k(0, 3), k(0, 3))
    _rep('unit', 'queue size 0 is a no-op', q0.get() is None)


def test_centring():
    import numpy as np
    import torch
    P = plugin()
    zt, u0 = _aniso(64, 256, 5)
    mu = zt.mean(0)
    T = lambda a: torch.as_tensor(a, dtype=torch.float32)  # noqa: E731
    mean_pred = np.repeat((mu / np.linalg.norm(mu))[None], 64, 0)
    L = P.sem_losses(T(mean_pred), T(zt), T(mu))
    raw = float(L['cos_raw'].mean())
    _rep('unit', 'centring: constant mean predictor has high RAW cos but ~1 centred loss',
         raw > 0.6 and 0.8 < float(L['cos_loss']) < 1.2, f'raw {raw:.3f} centred loss {float(L["cos_loss"]):.3f}')
    L = P.sem_losses(T(zt), T(zt), T(mu))
    _rep('unit', 'centring: perfect prediction -> L_cos 0', float(L['cos_loss']) < 1e-5)
    c = P.centre(T(zt), T(mu))
    ref = (zt - mu) / np.linalg.norm(zt - mu, axis=1, keepdims=True)
    _rep('unit', 'centre(v) == normalize(v - mu)', np.allclose(c.numpy(), ref, atol=1e-5))


class _Owner:
    """Minimal stand-in for the attached model (owns sem_head; training flag)."""

    def __new__(cls, head):
        import torch.nn as nn
        m = nn.Module()
        m.sem_head = head
        return m


def _fake_batch(P, n_blk, K, H, valid, dim=2560, seed=0):
    import torch
    g = torch.Generator().manual_seed(seed)
    h = torch.randn(n_blk, K, H, generator=g, requires_grad=True)
    r = P.get_reader()
    rows = torch.as_tensor(r.rows(list(range(n_blk))).astype('float32'))
    stt = torch.tensor([P.ST_VALID if v else P.ST_NONE for v in valid], dtype=torch.int8)
    key = torch.as_tensor(P._u64_to_i64(r.keys[:n_blk]).copy())
    b = P.SemBatch(torch.zeros(n_blk, dtype=torch.long), torch.zeros(n_blk, K, dtype=torch.long), rows.half(), stt,
                   key, torch.zeros(n_blk, dtype=torch.long), (1, 10))
    b.h = h
    return b, h


def test_detach_blocks_hidden_grads():
    import torch
    P = plugin()
    with _EnvPatch(SEM_DETACH_STEPS='5', SEM_QUEUE='0', SEM_W_COS='1', SEM_W_NCE='1'):
        head = P.SemanticHead(32, 2560, 4, pool='attn', hidden=16)
        owner = _Owner(head)
        for step, expect_h in ((0, False), (4, False), (5, True), (9, True)):
            P._SCHED['step'] = step
            b, h = _fake_batch(P, 3, 4, 32, [True, True, False])
            term, stats = P.compute_sem_term(owner, b, torch.zeros(()), training=True)
            head.zero_grad(set_to_none=True)
            term.backward()
            got_h = h.grad is not None and float(h.grad.abs().sum()) > 0
            got_head = all(p.grad is not None for p in head.parameters()) and \
                float(head.fc2.weight.grad.abs().sum()) > 0
            _rep('unit', f'detach step {step}: hidden grad {expect_h}, head grad True',
                 got_h == expect_h and got_head and stats['detached'] == float(not expect_h),
                 f'h.grad={None if h.grad is None else float(h.grad.abs().sum())}')
        # eval never detaches-by-step and never needs grads
        P._SCHED['step'] = 0
        b, h = _fake_batch(P, 3, 4, 32, [True, True, True])
        with torch.no_grad():
            term, _ = P.compute_sem_term(owner, b, torch.zeros(()), training=False)
        P._drain_eval()
        _rep('unit', 'eval term finite, no grad', bool(torch.isfinite(term)) and not term.requires_grad)


def test_zero_valid_and_no_sem_batches():
    import torch
    P = plugin()
    with _EnvPatch(SEM_DETACH_STEPS='0'):
        P._SCHED['step'] = 100
        head = P.SemanticHead(32, 2560, 4, pool='mean', hidden=16)
        owner = _Owner(head)
        cases = {}
        b, h = _fake_batch(P, 2, 4, 32, [False, False])
        cases['blocks, 0 valid'] = b
        cases['no stash (None)'] = None
        e = torch.zeros(0, dtype=torch.long)
        b0 = P.SemBatch(e, torch.zeros(0, 4, dtype=torch.long), torch.zeros(0, 2560).half(), torch.zeros(0, dtype=torch.int8),
                        e, e, (1, 10))
        cases['stash with 0 blocks'] = b0
        for name, b in cases.items():
            head.zero_grad(set_to_none=True)
            loss = torch.tensor(1.5, requires_grad=True)
            term, stats = P.compute_sem_term(owner, b, loss, training=True)
            total = loss + term
            total.backward()
            ok = bool(torch.isfinite(total)) and float(term.detach()) == 0.0 and all(p.grad is not None for p in head.parameters())
            _rep('unit', f'{name}: finite, term 0, sem_head grads defined', ok, stats)


def test_weight_schedule():
    P = plugin()
    with _EnvPatch(SEM_W_COS='0.05', SEM_W_COS_START='1.0', SEM_W_COS_DECAY_STEPS='100', SEM_W_NCE='0.2'):
        c = P.CFG
        _rep('unit', 'schedule cosine START->END', abs(c.w_cos(0) - 1.0) < 1e-9 and abs(c.w_cos(50) - 0.525) < 1e-9
             and abs(c.w_cos(100) - 0.05) < 1e-9 and abs(c.w_cos(10**6) - 0.05) < 1e-9, [c.w_cos(s) for s in (0, 50, 100)])
        _rep('unit', 'constant weight without _START', c.w_nce(0) == c.w_nce(10**6) == 0.2)
    with _EnvPatch(SEM_W_NCE_START='0', SEM_W_NCE_END='0.1', SEM_W_NCE_DECAY_STEPS='10', SEM_W_NCE_DECAY_SHAPE='linear'):
        _rep('unit', 'linear warm-up schedule 0->0.1', abs(P.CFG.w_nce(5) - 0.05) < 1e-9 and P.CFG.w_nce(20) == 0.1)


def test_block_positions_vs_find_sem_blocks():
    import torch
    from swift.template.sem_latent import find_sem_blocks
    P = plugin()
    sem = (500, 501, 502)
    rng = random.Random(3)
    rows = []
    for r in range(4):
        ids = [rng.randrange(0, 50) for _ in range(40)]
        for s in sorted(rng.sample(range(0, 36), 2), reverse=True):
            ids[s:s] = list(sem)
        ids[5:5] = [500, 501]                               # a partial block: never counted
        rows.append(ids)
    L = max(map(len, rows)) + 3
    padded = torch.tensor([r + [0] * (L - len(r)) for r in rows])
    b_idx, pos, counts = P.sem_block_positions(padded, sem)
    exp = [(b, s) for b, r in enumerate(rows) for s in find_sem_blocks(r, sem)]
    got = list(zip(b_idx.tolist(), pos[:, 0].tolist()))
    _rep('unit', 'block finder == find_sem_blocks (row-major, partial blocks ignored)', got == exp
         and counts.tolist() == [len(find_sem_blocks(r, sem)) for r in rows]
         and all(padded[b, p].tolist() == list(sem) for b, p in zip(b_idx.tolist(), pos.tolist())), (got, exp))


def test_collate_and_post_encode_consistency():
    import torch
    P = plugin()
    sem = (500, 501)
    D = 2560
    mk = lambda n, code=1: {  # noqa: E731
        'sem_tgt': torch.ones(n, D).half(), 'sem_st': torch.full((n, ), code, dtype=torch.int8),
        'sem_ok': torch.full((n, ), code == 1), 'sem_key': torch.arange(n) + 7, 'sem_img': torch.zeros(n, dtype=torch.long)}
    rows = [dict(input_ids=[1, 500, 501, 2, 500, 501, 3], **mk(2)),          # consistent
            dict(input_ids=[1, 500, 501, 2, 500], **mk(2)),                  # truncated: 1 complete block, 2 targets
            {'input_ids': [4, 500, 501, 5]}]                                 # encoded without the plugin
    res = P.collate_sem(rows, sem, D)
    _rep('unit', 'collator recount: consistent row kept, truncated row -> mismatch, bare row -> none',
         res['sem_nblk'].tolist() == [2, 1, 1] and res['sem_st'].tolist() == [1, 1, P.ST_MISMATCH, P.ST_NONE]
         and res['sem_ok'].tolist() == [True, True, False, False] and res['sem_tgt'].shape == (4, D), res['sem_st'])
    ids = torch.tensor([[1, 500, 501, 2, 500, 501, 3], [1, 500, 501, 2, 500, 0, 0], [4, 500, 501, 5, 0, 0, 0]])
    b = P.make_batch(ids, sem, dict(res), D)
    _rep('unit', 'post-encode: consistent batch keeps statuses', b.n == 4 and b.st.tolist() == res['sem_st'].tolist())
    bad = dict(res)
    bad['sem_nblk'] = torch.tensor([2, 2, 0])
    b = P.make_batch(ids, sem, bad, D)
    _rep('unit', 'post-encode: per-row count mismatch -> whole batch unsupervised', b.n == 4 and not bool(b.ok.any()))
    b = P.make_batch(ids, sem, {}, D)
    _rep('unit', 'post-encode: no collated sem keys -> blocks unsupervised (none)',
         b.n == 4 and b.st.tolist() == [P.ST_NONE] * 4)


def test_param_group_identity_guard():
    import torch
    P = plugin()
    with _EnvPatch(SEM_HEAD_LR='3e-4'):
        m = torch.nn.Module()
        m.body = torch.nn.Linear(4, 4)
        m.sem_head = P.SemanticHead(8, 16, 2, hidden=4)
        m._sem_attached = True
        opt = torch.optim.AdamW(m.body.parameters(), lr=1e-5)
        n1 = P.add_sem_param_group(opt, m)
        n2 = P.add_sem_param_group(opt, m)
        g = opt.param_groups[-1]
        _rep('unit', 'optimizer: head group added once (identity guard), lr/initial_lr/wd',
             n1 == sum(p.numel() for p in m.sem_head.parameters()) and n2 == 0 and len(opt.param_groups) == 2
             and g['lr'] == 3e-4 and g['initial_lr'] == 3e-4 and g['weight_decay'] == 0.0)
        m.sem_head.requires_grad_(False)
        opt2 = torch.optim.AdamW(m.body.parameters(), lr=1e-5)
        _rep('unit', 'optimizer: frozen head -> no group', P.add_sem_param_group(opt2, m) == 0
             and len(opt2.param_groups) == 1)



def test_soft_nce_is_kl_zero_at_optimum():
    """review SEM-LOSS-1: soft L_nce is the KL (CE - H(p)): 0 for a perfect student however many candidates the
    queue holds (the CE grows with them), and the SAME gradient as the CE."""
    import numpy as np
    import torch
    P = plugin()
    zt, u0 = _aniso(6, 48, 5)
    mu = 0.75 * u0
    rng0 = np.random.default_rng(6)
    # queue entries that are NEAR-duplicates of the targets (normal films collapse in teacher space): p spreads
    ev = (zt - mu)[np.arange(40) % 6] / np.linalg.norm(zt - mu, axis=1, keepdims=True)[np.arange(40) % 6]
    ev = ev + 0.03 * rng0.standard_normal(ev.shape)
    ev /= np.linalg.norm(ev, axis=1, keepdims=True)
    T = lambda a, dt=torch.float32: torch.as_tensor(a, dtype=dt)  # noqa: E731
    ces = []
    for m in (0, 10, 40):
        extra = (T(ev[:m]), T(np.arange(m) + 100, torch.long), T(np.zeros(m), torch.long)) if m else None
        L = P.sem_losses(T(zt), T(zt), T(mu), extra=extra, mode='soft', tau=0.05, tau_t=0.05)
        ces.append(float(L['nce_ce']))
        _rep('unit', f'soft perfect student, {6 + m} candidates: nce_loss == KL == 0 <= CE = H(p)',
             abs(float(L['nce_loss'])) < 1e-5 and float(L['nce_loss']) == float(L['kl'])
             and float(L['nce_ce']) > (1e-2 if m else -1e-6), (float(L['nce_loss']), float(L['nce_ce'])))
    _rep('unit', 'soft CE (not optimised any more) grows with the candidate count', ces[0] < ces[1] < ces[2], ces)
    rng = np.random.default_rng(7)
    zh = T(zt + 0.2 * rng.standard_normal(zt.shape)).requires_grad_(True)
    extra = (T(ev), T(np.arange(40) + 100, torch.long), T(np.zeros(40), torch.long))
    L = P.sem_losses(zh, T(zt), T(mu), extra=extra, mode='soft', tau=0.05, tau_t=0.05)
    g_kl, = torch.autograd.grad(L['nce_loss'], zh)
    zh2 = zh.detach().clone().requires_grad_(True)
    ch, ct = P.centre(zh2, T(mu)), P.centre(T(zt), T(mu))
    cand = torch.cat([ct, T(ev)])
    p_ = torch.softmax(ct @ cand.T / 0.05, -1)
    ce = -(p_ * torch.log_softmax(ch @ cand.T / 0.05, -1)).sum(-1).mean()
    g_ce, = torch.autograd.grad(ce, zh2)
    _rep('unit', 'soft KL has the same gradient as the CE', torch.allclose(g_kl, g_ce, atol=1e-6)
         and abs(float(ce) - float(L['nce_ce'])) < 1e-5, float((g_kl - g_ce).abs().max()))


def test_eval_ignores_training_queue():
    """review SEM-LOSS-2: the eval SEM term scores against the eval micro-batch only -- the same for an empty and a
    full training queue -- and eval never pushes; training still uses the queue."""
    import torch
    P = plugin()
    with _EnvPatch(SEM_QUEUE='64', SEM_DETACH_STEPS='0', SEM_W_COS='1', SEM_W_NCE='1'):
        P._SCHED['step'] = 100
        head = P.SemanticHead(32, 2560, 4, pool='mean', hidden=16)
        owner = _Owner(head)
        b, _ = _fake_batch(P, 3, 4, 32, [True, True, True], seed=1)
        with torch.no_grad():
            t0, s0 = P.compute_sem_term(owner, b, torch.zeros(()), training=False)
        for sd in range(5):
            bt, _ = _fake_batch(P, 3, 4, 32, [True, True, True], seed=10 + sd)
            P.compute_sem_term(owner, bt, torch.zeros(()), training=True)
        nq = len(P._queue())
        with torch.no_grad():
            t1, s1 = P.compute_sem_term(owner, b, torch.zeros(()), training=False)
        _rep('unit', 'eval SEM term independent of the training-queue fill; eval does not push',
             nq == 15 and abs(float(t0) - float(t1)) < 1e-7 and s0['n_cand'] == s1['n_cand'] == 3
             and len(P._queue()) == nq, (nq, float(t0), float(t1), s0['n_cand'], s1['n_cand']))
        _, s2 = P.compute_sem_term(owner, b, torch.zeros(()), training=True)
        _rep('unit', 'training SEM term does use the queue', s2['n_cand'] == 3 + nq, s2['n_cand'])
        P._drain_eval()


def test_zero_term_same_hook_order_as_real_batch():
    """review D2: the no-SEM-block / no-hidden-state fallback makes the head params grad-ready in the SAME order
    as a real batch (ZeRO-2 reduces its buckets position-wise across ranks), with zero gradients."""
    import torch
    P = plugin()
    with _EnvPatch(SEM_DETACH_STEPS='0', SEM_QUEUE='0'):
        P._SCHED['step'] = 100
        for pool in ('attn', 'mean', 'concat'):
            head = P.SemanticHead(32, 2560, 4, pool=pool, hidden=16)
            owner = _Owner(head)
            names = {id(p): n for n, p in head.named_parameters()}
            order = []
            hooks = [p.register_post_accumulate_grad_hook(lambda p: order.append(names[id(p)])) for p in head.parameters()]

            def run(b, legacy=False):
                order.clear()
                head.zero_grad(set_to_none=True)
                if legacy:                                   # the pre-fix fallback, for the record
                    term = sum(p.sum() for p in head.parameters()) * 0.0
                else:
                    term, _ = P.compute_sem_term(owner, b, torch.zeros(()), training=True)
                term.backward()
                return list(order)

            real = run(_fake_batch(P, 3, 4, 32, [True, True, False])[0])
            zero_valid = run(_fake_batch(P, 2, 4, 32, [False, False])[0])
            none = run(None)
            zero = all(p.grad is not None and float(p.grad.abs().sum()) == 0.0 for p in head.parameters())
            legacy = run(None, legacy=True)
            for h in hooks:
                h.remove()
            _rep('unit', f'{pool}: fallback grad-ready order == real batch order, grads defined and 0',
                 real == zero_valid == none and len(real) == len(names) and zero,
                 f'real {real} | fallback {none} | pre-fix fallback {legacy}')


def test_param_group_moves_head_out_of_default_groups():
    """review D3: with HF's default optimizer (every trainable param in decay / no-decay groups, no --vit_lr /
    --aligner_lr) the sem_head params are MOVED into the SEM group, so SEM_HEAD_LR and wd 0 apply."""
    import torch
    P = plugin()
    with _EnvPatch(SEM_HEAD_LR='2e-4'):
        m = torch.nn.Module()
        m.body = torch.nn.Linear(4, 4)
        m.sem_head = P.SemanticHead(8, 16, 2, hidden=4)
        m._sem_attached = True
        nd = lambda n: 'bias' in n or 'norm' in n  # noqa: E731
        decay = [p for n, p in m.named_parameters() if not nd(n)]
        nodecay = [p for n, p in m.named_parameters() if nd(n)]
        opt = torch.optim.AdamW([{'params': decay, 'weight_decay': 0.1}, {'params': nodecay, 'weight_decay': 0.0}],
                                lr=4e-6)
        n1 = P.add_sem_param_group(opt, m)
        n2 = P.add_sem_param_group(opt, m)                         # create_optimizer runs twice
        head = {id(p) for p in m.sem_head.parameters()}
        body = {id(p) for p in m.body.parameters()}
        gs = [(g['lr'], g['weight_decay'], {id(p) for p in g['params']}) for g in opt.param_groups]
        ok = (n1 == sum(p.numel() for p in m.sem_head.parameters()) and n2 == 0 and len(gs) == 3
              and gs[2] == (2e-4, 0.0, head) and gs[0][2] | gs[1][2] == body and not (gs[0][2] | gs[1][2]) & head
              and opt.param_groups[2].get('sem_head') is True)
        m.sem_head(torch.randn(3, 2, 8)).sum().backward()
        m.body(torch.randn(2, 4)).sum().backward()
        opt.step()
        _rep('unit', 'optimizer: head moved out of the default decay/no-decay groups into its own (lr 2e-4, wd 0)',
             ok, [(g[0], g[1], len(g[2])) for g in gs])


def _train_args(**kw):
    base = dict(packing=False, padding_free=False, sequence_parallel_size=1, template_backend='swift',
                loss_scale='ignore_empty_think', loss_type=None, enable_dft_loss=False, enable_channel_loss=False,
                label_smoothing_factor=0.0, remove_unused_columns=True)
    base.update(kw)
    return SimpleNamespace(**base)


def test_check_train_args_refusals():
    """review P1 / P3 / SEM-LOSS-3 / D4 / TGT-1: configurations under which SEM would silently train wrong or not at
    all are refused at prepare_model (the launcher precheck runs the same function on the final argv)."""
    P = plugin()
    P._check_train_args(_train_args())                           # the stage-7 launcher's configuration
    for ls in ('default', 'last_round', 'default+ignore_empty_think'):
        P._check_train_args(_train_args(loss_scale=ls))
    _rep('unit', 'train args: the launcher config and default/last_round binary loss scales are accepted', True)
    refused = {'template_backend jinja (P3)': _train_args(template_backend='jinja'),
               'loss_scale all (P1)': _train_args(loss_scale='all'),
               'loss_scale all+ignore_empty_think (P1)': _train_args(loss_scale='all+ignore_empty_think'),
               'loss_scale all+hermes (P1)': _train_args(loss_scale='all+hermes'),
               'non-binary loss_scale hermes (labels popped)': _train_args(loss_scale='hermes'),
               '--loss_type (labels popped)': _train_args(loss_type='infonce'),
               '--enable_dft_loss (labels popped)': _train_args(enable_dft_loss=True),
               '--enable_channel_loss (labels popped)': _train_args(enable_channel_loss=True),
               '--label_smoothing_factor 0.1 (labels popped)': _train_args(label_smoothing_factor=0.1),
               '--packing': _train_args(packing=True), 'sequence parallel': _train_args(sequence_parallel_size=2)}
    for name, a in refused.items():
        try:
            P._check_train_args(a)
            msg = None
        except P.SemConfigError as e:
            msg = str(e)
        _rep('unit', f'train args refused: {name}', msg is not None, msg)
    reader = lambda n: SimpleNamespace(meta={'n_texts_per_family': {'override': n}})  # noqa: E731
    got = []
    for n, ruc in ((3, True), (3, False), (0, True)):
        P._STATE['warned'].discard('override_cols')
        P._check_train_args(_train_args(remove_unused_columns=ruc), reader=reader(n))
        got.append('override_cols' in P._STATE['warned'])
    _rep('unit', 'override texts in the sidecar + remove_unused_columns true -> warned (TGT-1)',
         got == [True, False, False], got)


def test_loss_none_with_valid_blocks_raises():
    """review SEM-LOSS-3 / D4: when swift computes the loss outside the model (labels popped) out.loss is None; a
    TRAINING micro-batch with valid SEM blocks then raises instead of silently training nothing (eval warns)."""
    import numpy as np
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    P = plugin()

    class Out(dict):
        __getattr__ = dict.get

        def __setattr__(self, k, v):
            self[k] = v

    class LM(nn.Module):
        def __init__(self, H):
            super().__init__()
            self.lin = nn.Linear(H, H)

        def forward(self, x):
            return SimpleNamespace(last_hidden_state=torch.tanh(self.lin(x)))

    class Fake(nn.Module):
        def __init__(self, V=248320, H=16):
            super().__init__()
            self.model = nn.Module()
            self.model.language_model = LM(H)
            self.embed = nn.Embedding(V, H)
            self.head = nn.Linear(H, 32)

        def get_input_embeddings(self):
            return self.embed

        def get_output_embeddings(self):
            return None

        def forward(self, input_ids, labels=None):
            logits = self.head(self.model.language_model(self.embed(input_ids)).last_hidden_state)
            loss = None if labels is None else F.cross_entropy(logits.reshape(-1, 32), labels.reshape(-1) % 32)
            return Out(loss=loss, logits=logits)

    tok = processor().tokenizer
    with _EnvPatch(SEM_UNTIE_CONFIG='0', SEM_CHAT_TEMPLATE='0', SEM_INIT_EMBED='0', SEM_RESTORE_HEAD='0',
                   SEM_HEAD_DTYPE='float32', SEM_HEAD_HIDDEN='8', SEM_QUEUE='0', SEM_DETACH_STEPS='0'):
        m = Fake()
        prev_model = P._STATE.get('model')
        P.attach(m, tokenizer=tok)
        try:
            sem = P.get_sem_ids(tok)
            r = P.get_reader()
            ids = torch.tensor([[1, 2] + list(sem) + [3, 4, 5]])
            popped = {'sem_tgt': torch.as_tensor(r.rows([0]).astype(np.float16)),
                      'sem_st': torch.tensor([P.ST_VALID], dtype=torch.int8),
                      'sem_key': torch.as_tensor(P._u64_to_i64(r.keys[:1]).copy()),
                      'sem_img': torch.zeros(1, dtype=torch.long), 'sem_nblk': torch.tensor([1])}
            m.train()
            P._SCHED['step'] = 100
            m.__dict__['_sem_batch'] = P.make_batch(ids, sem, dict(popped), r.dim)
            out = m(ids, labels=ids)
            added = out.loss is not None and P._STATE['last'].get('n_valid') == 1
            m.__dict__['_sem_batch'] = P.make_batch(ids, sem, dict(popped), r.dim)
            try:
                m(ids, labels=None)
                raised = None
            except P.SemConfigError as e:
                raised = str(e)
            stash_gone = '_sem_batch' not in m.__dict__
            m.eval()
            P._STATE['warned'].discard('loss_none_eval')
            m.__dict__['_sem_batch'] = P.make_batch(ids, sem, dict(popped), r.dim)
            with torch.no_grad():
                out_e = m(ids, labels=None)
            m.train()
            no_blocks = m(torch.tensor([[1, 2, 3]]), labels=None)   # no SEM block: None passes through
            _rep('unit', 'loss=None + valid SEM blocks: train raises, eval warns, no-block batches pass',
                 added and raised is not None and 'OUTSIDE the model' in raised and stash_gone and out_e.loss is None
                 and 'loss_none_eval' in P._STATE['warned'] and no_blocks.loss is None, raised)
        finally:
            for h in m.__dict__.get('_sem_hook_handles', []):
                h.remove()
            P._STATE['model'] = prev_model
            P._RUN.clear()
            P._drain_eval()


def test_restore_head_roundtrip():
    import torch
    from safetensors.torch import save_file
    P = plugin()
    d = tempfile.mkdtemp(dir=OUT)
    try:
        h1 = P.SemanticHead(32, 64, 4, pool='attn', hidden=16)
        with torch.no_grad():
            for p in h1.parameters():
                p.normal_()
        save_file({'model.language_model.embed_tokens.weight': torch.randn(10, 4)}, os.path.join(d, 'model-1.safetensors'))
        save_file({f'sem_head.{k}': v.contiguous() for k, v in h1.state_dict().items()} | {'lm_head.weight': torch.randn(3, 3)},
                  os.path.join(d, 'model-2.safetensors'))
        wm = {'model.language_model.embed_tokens.weight': 'model-1.safetensors', 'lm_head.weight': 'model-2.safetensors'}
        wm.update({f'sem_head.{k}': 'model-2.safetensors' for k in h1.state_dict()})
        json.dump({'weight_map': wm}, open(os.path.join(d, 'model.safetensors.index.json'), 'w'))
        h2 = P.SemanticHead(32, 64, 4, pool='attn', hidden=16)
        n = P.restore_head(h2, d)
        same = all(torch.equal(a, b) for a, b in zip(h1.state_dict().values(), h2.state_dict().values()))
        _rep('unit', 'restore: safe_open key filter round trip (index -> only the sem_head shard)', n == 7 and same, n)
        h3 = P.SemanticHead(32, 64, 4, pool='mean', hidden=16)
        try:
            P.restore_head(h3, d)
            ok = False
        except P.SemConfigError:
            ok = True
        _rep('unit', 'restore: pool mismatch raises SemConfigError (never a silent random head)', ok)
        h4 = P.SemanticHead(32, 64, 4, pool='attn', hidden=24)
        try:
            P.restore_head(h4, d)
            ok = False
        except P.SemConfigError:
            ok = True
        _rep('unit', 'restore: hidden-size mismatch raises SemConfigError', ok)
        e = tempfile.mkdtemp(dir=OUT)
        save_file({'x': torch.zeros(1)}, os.path.join(e, 'model.safetensors'))
        _rep('unit', 'restore: no sem_head.* -> 0 (fresh head)', P.restore_head(h3, e) == 0)
        shutil.rmtree(e)
    finally:
        shutil.rmtree(d)


def test_init_sem_embeddings():
    import torch
    P = plugin()

    def mk():
        torch.manual_seed(0)
        m = torch.nn.Module()
        emb = torch.nn.Embedding(120, 16)
        head = torch.nn.Linear(16, 120, bias=False)
        with torch.no_grad():
            emb.weight[100:] = emb.weight[:100].mean(0) * 0.9          # one shared "spare" vector
            head.weight[100:] = 0.123
        m.emb, m.head = emb, head
        m.get_input_embeddings = lambda: emb
        m.get_output_embeddings = lambda: head
        return m

    a, b = mk(), mk()
    ids = list(range(100, 108))
    ra = P.init_sem_embeddings(a, ids, n_vocab=108, seed=7, noise=0.5)
    rb = P.init_sem_embeddings(b, ids, n_vocab=108, seed=7, noise=0.5)
    Wa = a.emb.weight.detach()
    tr = Wa[:100].double()
    new = Wa[100:108].double()
    ok = (ra and rb and torch.equal(Wa, b.emb.weight.detach())
          and bool((new[1:] - new[:1]).abs().max() > 0)                   # rows now distinct
          and torch.equal(Wa[108:], mk().emb.weight.detach()[108:])      # other spare rows untouched
          and bool((a.head.weight[100:] == 0.123).all()))                 # lm_head untouched
    z = (new - tr.mean(0)) / (tr.std(0, unbiased=False) * 0.5)
    _rep('unit', 'SEM row init: deterministic, distinct, other spares + lm_head untouched', ok)
    _rep('unit', 'SEM row init: rows ~ mean + N(0,1)*std*0.5', abs(float(z.mean())) < 0.3 and 0.7 < float(z.std()) < 1.3,
         f'z mean {float(z.mean()):.3f} std {float(z.std()):.3f}')
    _rep('unit', 'SEM row init: already-distinct rows left unchanged',
         not P.init_sem_embeddings(a, ids, n_vocab=108, seed=7) and torch.equal(a.emb.weight.detach(), Wa))
    # tolerance: the real spare rows differ by bf16 rounding (max 1.2e-4 = 0.009 x median trained std)
    c = mk()
    with torch.no_grad():
        c.emb.weight[100:] += 1e-3 * torch.randn(20, 16)                # << 0.05 x std(~1)
    _rep('unit', 'SEM row init: spare rows equal up to rounding noise (<= tol x std) -> initialised',
         P.init_sem_embeddings(c, ids, n_vocab=108, seed=7, tol=0.05))
    c = mk()
    with torch.no_grad():
        c.emb.weight[100:] += 0.2 * torch.randn(20, 16)                 # > 0.05 x std: already trained
    W0 = c.emb.weight.detach().clone()
    _rep('unit', 'SEM row init: spread > tol x std -> left unchanged',
         not P.init_sem_embeddings(c, ids, n_vocab=108, seed=7, tol=0.05) and torch.equal(c.emb.weight.detach(), W0))


def test_startup_refusals():
    import numpy as np
    import torch
    P = plugin()
    from swift.model.register import get_processor
    plain = get_processor(CKPT).tokenizer
    m = torch.nn.Module()
    m.emb = torch.nn.Embedding(248320, 4)
    m.get_input_embeddings = lambda: m.emb
    m.get_output_embeddings = lambda: None
    try:
        P.startup_checks(m, plain)
        msg = ''
    except P.SemConfigError as e:
        msg = str(e)
    _rep('unit', 'no SEM tokens -> raise with the exact --new_special_tokens fix',
         'token.txt examples/train/liger/token_sem.txt' in msg and '--new_special_tokens' in msg, msg[:120])
    sem_tok = processor().tokenizer
    orig = P.get_sem_ids
    try:
        P.get_sem_ids = lambda t: tuple(range(248211, 248211 + 110))
        try:
            P.startup_checks(m, sem_tok)
            msg = ''
        except P.SemConfigError as e:
            msg = str(e)
        _rep('unit', 'K > 109 -> refused', 'K=110' in msg, msg[:100])
    finally:
        P.get_sem_ids = orig
    d = tempfile.mkdtemp(dir=OUT)
    json.dump({'text_config': {'vocab_size': 248192}}, open(os.path.join(d, 'config.json'), 'w'))
    m.model_dir = d
    try:
        P.startup_checks(m, sem_tok)
        msg = ''
    except P.SemConfigError as e:
        msg = str(e)
    _rep('unit', 'resized embeddings (rows != ckpt vocab) -> refused', 'RESIZED' in msg, msg[:100])
    json.dump({'text_config': {'vocab_size': 248320}}, open(os.path.join(d, 'config.json'), 'w'))
    _rep('unit', 'rows == ckpt vocab -> ok, returns K=8 ids', len(P.startup_checks(m, sem_tok)) == 8)
    shutil.rmtree(d)
    # sidecar checks
    st = st_mod()
    bad = os.path.join(OUT, 'sidecar_badrule')
    for rule, fp, expect in (('v0', 'x', 'rule_version'), (st.RULE_VERSION, '', 'teacher_fingerprint')):
        if os.path.isdir(bad):
            shutil.rmtree(bad)
        st.write_sidecar(bad, np.array([1, 2], np.uint64), np.eye(2, 8, dtype=np.float32), np.ones(2, np.uint8),
                         {'rule_version': rule, 'teacher_fingerprint': fp})
        with _EnvPatch(SEM_TARGETS_DIR=bad):
            P._STATE['reader'] = None
            try:
                P.get_reader()
                msg = ''
            except P.SemConfigError as e:
                msg = str(e)
        P._STATE['reader'] = None
        _rep('unit', f'sidecar refusal: {expect}', expect in msg, msg[:120])
    with _EnvPatch(SEM_TARGETS_DIR=None):
        P._STATE['reader'] = None
        try:
            P.get_reader()
            msg = ''
        except P.SemConfigError as e:
            msg = str(e)
    P._STATE['reader'] = None
    _rep('unit', 'SEM_TARGETS_DIR unset -> raise with the precompute command', 'run_build_sem_targets.sh' in msg)
    shutil.rmtree(bad)


def test_eval_metrics():
    import numpy as np
    P = plugin()
    zt, u0 = _aniso(30, 64, 9)
    mu = zt.mean(0)
    keys = np.arange(30)
    keys[29] = keys[28]                                     # a duplicate text
    zt[29] = zt[28]
    m = P.retrieval_metrics(zt, zt, keys, mu)
    _rep('unit', 'retrieval: perfect predictions -> R@1 1, median rank 1, gallery = unique keys',
         m['r1'] == 1.0 and m['median_rank'] == 1 and m['n_gallery'] == 29 and m['n'] == 30, m)
    rng = np.random.default_rng(0)
    zh = rng.standard_normal((30, 64))
    m2 = P.retrieval_metrics(zh, zt, keys, mu)
    _rep('unit', 'retrieval: random predictions ~ chance', m2['r1'] < 0.3 and m2['median_rank'] > 3, m2['r1'])
    mu_hat = mu / np.linalg.norm(mu)
    m3 = P.retrieval_metrics(np.repeat(mu_hat[None], 30, 0), zt, keys, mu)
    _rep('unit', 'retrieval: mean-predictor baseline == cos_raw of mu predictor',
         abs(m3['cos_raw'] - m3['cos_raw_meanpred']) < 1e-5)
    qm = np.zeros(30, bool)
    qm[:5] = True
    _rep('unit', 'retrieval: query subset', P.retrieval_metrics(zt, zt, keys, mu, query_mask=qm)['n'] == 5)
    s = rng.standard_normal(500)
    y = rng.random(500) < 1 / (1 + np.exp(-2 * s))
    s[::7] = np.round(s[::7], 1)                             # ties
    from sklearn.metrics import roc_auc_score
    _rep('unit', 'auroc == sklearn (with ties)', abs(P.auroc(s, y) - roc_auc_score(y, s)) < 1e-12)
    pr = {'W': rng.standard_normal((6, 64)).astype(np.float32), 'b': np.zeros(6, np.float32), 'mean': None,
          'std': None, 'tags': list('abcdef'), 'thr': {t: 0.5 for t in 'abcdef'}}
    pm = P.probe_metrics(zt, zt, pr, min_pos=3)
    _rep('unit', 'probe diagnostic: z_hat == z* -> macro AUROC 1', pm.get('macro_auc') == 1.0 and pm['n_tags_scored'] > 0, pm)


def test_online_teacher_stub():
    import numpy as np
    import torch
    P = plugin()

    class Stub:
        calls = []

        def encode(self, texts):
            Stub.calls.append(list(texts))
            return np.stack([np.full(2560, 0.02 * (i + 1), np.float32) for i in range(len(texts))]), None

    orig = P._get_online_teacher
    P._get_online_teacher = lambda device: Stub()
    try:
        with _EnvPatch(SEM_ONLINE_TEACHER='1'):
            P._STATE['online_cache'].clear()
            sem = (500, 501)
            rows = [{'input_ids': [1, 500, 501, 2, 500, 501], 'sem_tgt': torch.zeros(2, 2560).half(),
                     'sem_st': torch.tensor([P.ST_MISS, P.ST_VALID], dtype=torch.int8), 'sem_ok': torch.tensor([False, True]),
                     'sem_key': torch.tensor([77, 78]), 'sem_img': torch.zeros(2, dtype=torch.long),
                     'sem_txt': ['Findings: text A', '']}]
            res = P.collate_sem(rows, sem, 2560, online=True)
            ids = torch.tensor([[1, 500, 501, 2, 500, 501]])
            b = P.make_batch(ids, sem, dict(res), 2560)
            b2 = P.make_batch(ids, sem, dict(res), 2560)
            _rep('unit', 'online teacher: miss filled + marked valid, 2nd batch hits the cache',
                 b.st.tolist() == [1, 1] and b.n_online == 1 and abs(float(b.tgt[0, 0]) - 0.02) < 1e-3
                 and len(Stub.calls) == 1 and b2.n_online == 1 and b2.st.tolist() == [1, 1])
    finally:
        P._get_online_teacher = orig
        P._STATE['online_cache'].clear()


def test_trainer_callback():
    import numpy as np
    P = plugin()
    ctl = SimpleNamespace(should_log=True)

    class FakeTrainer:
        template = None

        def __init__(self):
            self.logs = []

        def log(self, payload):
            self.logs.append(dict(payload))
            ctl.should_log = False                      # what HF CallbackHandler.on_log does

    tr = FakeTrainer()
    cb = P.make_probe_callback(tr)
    d = tempfile.mkdtemp(dir=OUT)
    args = SimpleNamespace(logging_steps=5, output_dir=d)
    state = SimpleNamespace(global_step=10, max_steps=100)
    cb.on_train_begin(args, state, ctl)
    cb.on_step_begin(args, state, ctl)
    _rep('unit', 'callback: on_step_begin publishes the optimizer step', P._SCHED['step'] == 10)
    P._RUN.clear()
    for _ in range(3):
        P._record_run({'n_blk': 4, 'n_valid': 2, 'n_miss': 1, 'n_leaked': 1, 'cos_loss': 0.5, 'nce_loss': 2.0,
                       'kl': 0.3, 'cos_centred': 0.5, 'cos_raw': 0.8, 'term': 0.1, 'w_cos': 0.05, 'w_nce': 0.05}, 1.0)
    state.global_step = 11
    cb.on_step_end(args, state, ctl)
    _rep('unit', 'callback: no log off the logging_steps grid', not tr.logs and P._RUN.get('micro') == 3)
    state.global_step = 15
    cb.on_step_end(args, state, ctl)
    lg = tr.logs[-1] if tr.logs else {}
    _rep('unit', 'callback: train/sem_* logged, running means, control.should_log restored',
         abs(lg.get('train/sem_cos_loss', 0) - 0.5) < 1e-9 and abs(lg.get('train/sem_valid_frac', 0) - 0.5) < 1e-9
         and abs(lg.get('train/sem_miss_rate', 0) - 1 / 3) < 1e-9 and abs(lg.get('train/sem_share', 0) - 10.0) < 1e-9
         and ctl.should_log is True and not P._RUN, lg)
    r = P.get_reader()
    n = 20
    zt = r.rows(list(range(n))).astype(np.float32)
    rng = np.random.default_rng(0)
    zh = zt + 0.05 * rng.standard_normal(zt.shape).astype(np.float32)
    zh /= np.linalg.norm(zh, axis=1, keepdims=True)
    P._drain_eval()
    P._EVAL['zh'].append(zh.astype(np.float16))
    P._EVAL['zt'].append(zt.astype(np.float16))
    P._EVAL['key'].append(P._u64_to_i64(np.asarray(r.keys[:n])))
    P._EVAL['img'].append(np.zeros(n, np.int64))
    state.global_step = 50
    cb.on_evaluate(args, state, ctl)
    lg = tr.logs[-1]
    npz = os.path.join(d, 'eval_predictions', 'sem_step000050.npz')
    ok = (lg.get('val_sem/n') == n and lg.get('val_sem/r1', 0) > 0.9 and 'val_sem_unseen/n' in lg and os.path.isfile(npz)
          and not P._EVAL['zh'])
    if ok:
        z = np.load(npz, allow_pickle=False)
        ok = z['z_hat'].shape == (n, r.dim) and z['keys'].dtype == np.uint64 and int(z['step']) == 50
    _rep('unit', 'callback: on_evaluate -> val_sem/* logged, npz dumped on rank 0, eval buffers drained', ok, lg)
    cb.on_evaluate(args, state, ctl)                     # nothing recorded: no crash, no log
    _rep('unit', 'callback: empty eval -> no payload, no crash', len(tr.logs) == 2)
    shutil.rmtree(d)


_ONLINE_WORKER = r'''
import os, sys, json, numpy as np, torch
sys.path.insert(0, {repo!r})
import swift_plugins.sem_latent_plugin as P
r = P.get_reader()
idx = [0, 1, 2]
texts = r.texts(idx)
key = torch.as_tensor(P._u64_to_i64(np.asarray(r.keys[idx])))
res = {{'sem_tgt': torch.zeros(3, r.dim).half(), 'sem_st': torch.full((3, ), P.ST_MISS, dtype=torch.int8),
        'sem_key': key, 'sem_img': torch.zeros(3, dtype=torch.long), 'sem_nblk': torch.tensor([3]), 'sem_txt': texts}}
sem = (900, 901)
ids = torch.tensor([[1, 900, 901, 2, 900, 901, 3, 900, 901]])
b = P.make_batch(ids, sem, res, r.dim)
ref = r.rows(idx).astype(np.float32)
got = b.tgt.float().numpy()
cos = (got * ref).sum(1) / np.linalg.norm(got, axis=1) / np.linalg.norm(ref, axis=1)
print('ONLINE', json.dumps({{'st': b.st.tolist(), 'n_online': b.n_online, 'cos': cos.tolist(),
                             'teacher_off_module_tree': True}}), flush=True)
'''


def test_online_teacher_real():
    """OPT-IN (SEM_TEST_ONLINE=1; loads the 4B teacher in fp32 on CPU, ~16 GB, ~2 min): the plugin's
    online path (builder Teacher, fingerprint check) reproduces the teacher-built smoke sidecar rows."""
    smoke = os.path.join(SCRATCH, 'tests_out', 'sidecar_smoke')
    if os.environ.get('SEM_TEST_ONLINE') != '1' or not os.path.isfile(os.path.join(smoke, 'meta.json')):
        print('    (opt-in: SEM_TEST_ONLINE=1 and the smoke sidecar at tests_out/sidecar_smoke)')
        return
    d = tempfile.mkdtemp(dir=OUT)
    try:
        script = os.path.join(d, 'w.py')
        open(script, 'w').write(_ONLINE_WORKER.format(repo=REPO))
        p = subprocess.run([sys.executable, script], capture_output=True, text=True, timeout=3600,
                           env=dict(os.environ, PYTHONPATH=REPO, SEM_TARGETS_DIR=smoke, SEM_ONLINE_TEACHER='1',
                                    QURE_PLUGIN_MODE='train', CUDA_VISIBLE_DEVICES=''))
        res = None
        for line in p.stdout.splitlines():
            if line.startswith('ONLINE '):
                res = json.loads(line[7:])
        ok = (p.returncode == 0 and res is not None and res['st'] == [1, 1, 1] and res['n_online'] == 3
              and min(res['cos']) > 0.9999)
        _rep('unit', 'online teacher (real bb_23sep, fp32 CPU): fingerprint ok, misses filled == sidecar rows',
             ok, (p.returncode, res, p.stderr[-600:]))
    finally:
        shutil.rmtree(d, ignore_errors=True)


_GATHER_WORKER = r'''
import os, sys, torch, torch.distributed as dist
sys.path.insert(0, {repo!r})
os.environ['QURE_PLUGIN_MODE'] = 'train'
def run(rank, init):
    dist.init_process_group('gloo', init_method=init, rank=rank, world_size=2)
    import swift_plugins.sem_latent_plugin as P
    n = [3, 0][rank]
    ct = torch.arange(n * 4, dtype=torch.float32).view(n, 4) + 100 * rank
    g0 = P._gather_candidates(ct, torch.arange(n) + 10 * (rank + 1), torch.arange(n) + 1000)
    n = [1, 2][rank]
    ct = torch.full((n, 4), float(rank + 1))
    g1 = P._gather_candidates(ct, torch.arange(n) + 50 * (rank + 1), torch.arange(n))
    out = [None if g is None else [g[0].tolist(), g[1].tolist(), g[2].tolist()] for g in (g0, g1)]
    import json
    print('RESULT', rank, json.dumps(out), flush=True)
    dist.destroy_process_group()
if __name__ == '__main__':
    import torch.multiprocessing as mp
    mp.spawn(run, args=({init!r},), nprocs=2)
'''


def test_nce_gather_two_ranks():
    d = tempfile.mkdtemp(dir=OUT)
    try:
        init = 'file://' + os.path.join(d, 'pg')
        script = os.path.join(d, 'w.py')
        open(script, 'w').write(_GATHER_WORKER.format(repo=REPO, init=init))
        p = subprocess.run([sys.executable, script], capture_output=True, text=True, timeout=300,
                           env=dict(os.environ, PYTHONPATH=REPO, SEM_TARGETS_DIR=rows_and_sidecar()[1]))
        res = {}
        for line in p.stdout.splitlines():
            if line.startswith('RESULT'):
                _, r, js = line.split(' ', 2)
                res[int(r)] = json.loads(js)
        ok = (p.returncode == 0 and res[0][0] is None                       # rank1 sent 0 rows
              and res[1][0][0] == [[0., 1., 2., 3.], [4., 5., 6., 7.], [8., 9., 10., 11.]]
              and res[1][0][1] == [10, 11, 12] and res[1][0][2] == [1000, 1001, 1002]
              and res[0][1][0] == [[2.] * 4] * 2 and res[0][1][1] == [100, 101]
              and res[1][1][0] == [[1.] * 4] and res[1][1][1] == [50])
        _rep('unit', 'SEM_NCE_GATHER: gloo 2 ranks, unequal/zero sizes, own rows excluded, keys exact', ok,
             (p.returncode, res, p.stderr[-400:]))
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_inference_mode_installs_nothing():
    code = ('import sys; sys.path.insert(0, %r)\n'
            'from swift.template.base import Template\n'
            'from swift.template.templates.qwen import Qwen3_5Template as Q\n'
            'from swift.pipelines.train.tuner import TunerMixin\n'
            'from swift.trainers.mixin import SwiftMixin\n'
            'before = (Template._encode, Q.encode, Q._data_collator, Q._post_encode, TunerMixin.__dict__["prepare_model"],'
            ' SwiftMixin.create_optimizer, SwiftMixin.__init__)\n'
            'import swift_plugins.sem_latent_plugin as P\n'
            'after = (Template._encode, Q.encode, Q._data_collator, Q._post_encode, TunerMixin.__dict__["prepare_model"],'
            ' SwiftMixin.create_optimizer, SwiftMixin.__init__)\n'
            'print("SAME", all(a is b for a, b in zip(before, after)))\n') % REPO
    p = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True, timeout=300,
                       env=dict(os.environ, PYTHONPATH=REPO, QURE_PLUGIN_MODE='infer'))
    _rep('unit', 'inference mode (QURE_PLUGIN_MODE=infer): no patch installed', 'SAME True' in p.stdout,
         p.stdout[-200:] + p.stderr[-300:])
    p = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True, timeout=300,
                       env=dict(os.environ, PYTHONPATH=REPO, QURE_PLUGIN_MODE='train'))
    _rep('unit', 'train mode: every patch point replaced', 'SAME False' in p.stdout, p.stdout[-200:])


def test_dedup_first_and_single_copy():
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import test_plugin_fixes as TF
    TF.test_f1_every_plugin_dedups_first()
    _rep('unit', 'dedup_module is the first statement (test_plugin_fixes F1 static check incl. this plugin)', True)
    code = ('import sys; sys.path.insert(0, %r)\n'
            'from swift.utils import import_external_file\n'
            'a = import_external_file(%r)\n'
            'import swift_plugins.sem_latent_plugin as b\n'
            'from swift.template.base import Template\n'
            'print("ONE", a is b, Template._encode.__module__)\n') % (REPO, os.path.join(REPO, 'swift_plugins',
                                                                                     'sem_latent_plugin.py'))
    p = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True, timeout=300,
                       env=dict(os.environ, PYTHONPATH=REPO, QURE_PLUGIN_MODE='train',
                                SEM_TARGETS_DIR=rows_and_sidecar()[1]))
    n_patched = (p.stdout + p.stderr).count('templates patched')
    _rep('unit', '--external_plugins copy + swift_plugins.X import -> one module, patched once',
         'ONE True sem_latent_plugin' in p.stdout and n_patched == 1, (p.stdout[-200:], n_patched))


# ============================================================================================ (1b) real template
def _encode_rows(tpl, rows):
    encs = []
    for rid, raw in rows:
        encs.append((rid, raw, tpl.encode(to_swift_row(raw), return_length=True)))
    return encs


def test_real_template_targets():
    import numpy as np
    from swift.template.sem_latent import find_sem_blocks, get_sem_ids
    P = plugin()
    st = st_mod()
    rows, d, vecs, texts = rows_and_sidecar()
    P._STATE['reader'] = None
    tpl = template()
    sem = list(get_sem_ids(tpl.tokenizer))
    keys = set(vecs)
    encs = _encode_rows(tpl, rows)
    n_valid = 0
    for rid, raw, enc in encs:
        exp = expected_blocks(raw, keys)
        got = list(zip(enc['sem_st'].tolist(), P._i64_to_u64(enc['sem_key'].numpy()).tolist()))
        nb = len(find_sem_blocks(enc['input_ids'], sem))
        ok = nb == len(exp) == len(got) and got == [(s, k) for s, k in exp]
        rd = P.get_reader()
        tg_ok = all(np.array_equal(enc['sem_tgt'][j].numpy(), rd.rows(rd.lookup([k]))[0]) if s == 1 else
                    not enc['sem_tgt'][j].any() for j, (s, k) in enumerate(got))
        img = st.image_key(raw['images'][0]) if raw.get('images') else 0
        img_ok = set(P._i64_to_u64(enc['sem_img'].numpy()).tolist()) <= {img}
        n_valid += sum(s == 1 for s, _ in got)
        _rep('real', f'{rid}: blocks {nb} == merged turns, (status,key) == independent raw derivation, targets, image key',
             ok and tg_ok and img_ok and enc['sem_ok'].tolist() == [s == 1 for s, _ in got]
             and not any(k.endswith('length') for k in enc if k.startswith('sem')),
             f'got {[(s, k % 1000) for s, k in got]} exp {[(s, k % 1000) for s, k in exp]}')
    _rep('real', 'at least 4 valid blocks across the 4 rows', n_valid >= 4, n_valid)
    byid = {rid: raw for rid, raw in rows}
    # overrides: list column on the no-T1 row (block 0) and str column (last block)
    r = copy.deepcopy(byid['no_t1'])
    r['sem_targets'] = [OVERRIDE_TEXT]
    enc = tpl.encode(to_swift_row(r))
    k = st.target_key(OVERRIDE_TEXT)
    _rep('real', 'override sem_targets[0] -> block 0 supervised with the override key',
         enc['sem_st'][0].item() == 1 and int(P._i64_to_u64(enc['sem_key'][:1].numpy())[0]) == k
         and expected_blocks(r, keys)[0] == (1, k))
    r = copy.deepcopy(byid['no_t1'])
    r['sem_target'] = OVERRIDE_TEXT
    enc = tpl.encode(to_swift_row(r))
    _rep('real', 'override sem_target -> LAST block', enc['sem_st'][-1].item() == 1 and enc['sem_st'][0].item() == 0)
    # the pending state never leaks to the next row
    enc = tpl.encode(to_swift_row(byid['no_t1']))
    _rep('real', 'override state cleared after the row (next row unaffected)', not bool(enc['sem_ok'].any()))
    # loss:false on the final answer
    r = copy.deepcopy(byid['onto_plain'])
    r['messages'][-1]['loss'] = False
    enc = tpl.encode(to_swift_row(r))
    _rep('real', 'final answer loss:false -> status loss_false (SEM_SKIP_LOSS_FALSE=1)', enc['sem_st'].tolist() == [4])
    with _EnvPatch(SEM_SKIP_LOSS_FALSE='0'):
        enc = tpl.encode(to_swift_row(r))
        _rep('real', 'SEM_SKIP_LOSS_FALSE=0 -> supervised', enc['sem_st'].tolist() == [1])
    # leak: the findings already in an earlier turn of the same row
    r = copy.deepcopy(byid['onto_plain'])
    can = st.iter_row_targets(byid['onto_plain'])[0][2]
    body = st.leak_body(can)
    user = r['messages'][0]
    r['messages'] = [dict(user), {'role': 'assistant', 'content': 'Prior read: ' + body},
                     {'role': 'user', 'content': user['content'].replace('<image>', '')}] + r['messages'][1:]
    enc = tpl.encode(to_swift_row(r))
    _rep('real', 'leak guard: findings verbatim in an earlier turn -> status leaked', enc['sem_st'].tolist() == [0, 3]
         and expected_blocks(r, keys) == [(0, 0), (3, st.target_key(can))], enc['sem_st'].tolist())
    with _EnvPatch(SEM_SKIP_LEAKED='0'):
        enc = tpl.encode(to_swift_row(r))
        _rep('real', 'SEM_SKIP_LEAKED=0 -> supervised', enc['sem_st'].tolist() == [0, 1])
    # cache miss: a sidecar without the onto key
    miss_dir = os.path.join(OUT, 'sidecar_miss')
    drop = st.target_key(can)
    build_tiny_sidecar(miss_dir, [t for t in texts if st.target_key(t) != drop], seed=1)
    with _EnvPatch(SEM_TARGETS_DIR=miss_dir):
        P._STATE['reader'] = None
        enc = tpl.encode(to_swift_row(byid['onto_plain']))
        _rep('real', 'target not in the sidecar -> status miss, zeros', enc['sem_st'].tolist() == [2]
             and not enc['sem_tgt'].any())
    P._STATE['reader'] = None
    # collator on the real rows (+ a truncated copy)
    batch = [e for _, _, e in encs]
    res = tpl.data_collator([copy.deepcopy(e) for e in batch])
    nb = [len(find_sem_blocks(e['input_ids'], sem)) for e in batch]
    _rep('real', 'collator: sem_nblk per row, targets concatenated in row order',
         res['sem_nblk'].tolist() == nb and res['sem_st'].tolist() == sum((e['sem_st'].tolist() for e in batch), [])
         and res['sem_tgt'].shape == (sum(nb), 2560) and 'cls_labels' not in res)
    ids = res['input_ids']
    b = P.make_batch(ids, sem, {k: res[k] for k in P.SEM_KEYS if k in res}, 2560)
    _rep('real', 'post-encode on the padded real batch: consistent, positions hold SEM ids',
         b.n == sum(nb) and b.st.tolist() == res['sem_st'].tolist()
         and all(ids[bi, p].tolist() == sem for bi, p in zip(b.b_idx.tolist(), b.pos.tolist())))
    jm = int(np.argmax(nb))                                  # the row with the most blocks
    t = copy.deepcopy(batch[jm])
    j = find_sem_blocks(t['input_ids'], sem)[-1]
    t['input_ids'] = t['input_ids'][:j + 3]
    t['labels'] = t['labels'][:j + 3]
    res2 = tpl.data_collator([t])
    _rep('real', 'collator: a row truncated inside a block -> that row unsupervised',
         nb[jm] >= 2 and set(res2['sem_st'].tolist()) == {P.ST_MISMATCH} and res2['sem_nblk'].tolist() == [nb[jm] - 1])
    _CACHE['real_encs'] = encs


# ============================================================================================ (2) toy e2e
TOY_DIR = os.path.join(OUT, 'toy_ckpt')



def test_loss_scale_all_row_warned_and_decode_strips_sem():
    """review P1: a --loss_scale all training encode inserts NO SEM block (headers trained) -> the core hook and the
    plugin warn (the plugin also refuses the run: test_check_train_args_refusals). review P4: Template.decode never
    returns SEM token strings (a generated SEM id fed back as history would give a stray block)."""
    from swift.template import get_template
    from swift.template import sem_latent as S
    P = plugin()
    proc = processor()
    sem = S.get_sem_ids(proc.tokenizer)
    row = {'messages': [{'role': 'user', 'content': 'Generate a medical report of the given chest X-ray.'},
                        {'role': 'assistant', 'content': '{"Findings": "* Lungs are clear.", "Impression": "Normal."}'}]}
    S._WARNED.discard('header_trained')
    P._STATE['warned'].discard('no_blocks')
    n_def = S.count_sem_blocks(template(proc).encode(copy.deepcopy(row))['input_ids'], sem)
    quiet = 'header_trained' not in S._WARNED and 'no_blocks' not in P._STATE['warned']
    ta = get_template(proc, **dict(TEMPLATE_KW, loss_scale='all'))
    ta.set_mode('train')
    n_all = S.count_sem_blocks(ta.encode(copy.deepcopy(row))['input_ids'], sem)
    _rep('real', 'loss_scale all: 0 SEM blocks (default: 1) and BOTH the core and the plugin warn',
         n_def == 1 and quiet and n_all == 0 and 'header_trained' in S._WARNED and 'no_blocks' in P._STATE['warned'],
         (n_def, n_all, sorted(S._WARNED), sorted(P._STATE['warned'])))
    tok = proc.tokenizer
    t = template(proc)
    t.set_mode('transformers')
    body = tok.encode('{"a": 1} <TAG> text', add_special_tokens=False)
    im_end = tok.convert_tokens_to_ids('<|im_end|>')
    for name, ids in (('partial block first', list(sem[:3]) + body + [im_end]),
                      ('full block mid-text', body[:2] + list(sem) + body[2:] + [im_end]),
                      ('single id last', body + [sem[5], im_end])):
        resp = t.decode(ids)
        plain = t.decode(body + [im_end])
        _rep('real', f'decode strips SEM ({name})', '<|sem_' not in resp and resp == plain, (resp, plain))


def build_toy_ckpt(out_dir=TOY_DIR):
    """Tiny random Qwen3_5ForConditionalGeneration + ckpt-608 tokenizer/processor files; shaped like the
    real checkpoint: 248320 rows, separate lm_head, config tie=true at both levels, one shared spare row."""
    marker = os.path.join(out_dir, 'TOY_OK')
    if os.path.isfile(marker):
        return out_dir
    import torch
    disable_fla()
    from transformers import Qwen3_5Config, Qwen3_5ForConditionalGeneration
    cfg = json.load(open(os.path.join(CKPT, 'config.json')))
    tc = cfg['text_config']
    tc.update(hidden_size=64, intermediate_size=128, num_hidden_layers=4,
              layer_types=['linear_attention', 'linear_attention', 'linear_attention', 'full_attention'],
              full_attention_interval=4, num_attention_heads=4, num_key_value_heads=2, head_dim=16,
              linear_num_key_heads=2, linear_num_value_heads=2, linear_key_head_dim=16, linear_value_head_dim=16,
              mtp_num_hidden_layers=0, tie_word_embeddings=False, dtype='float32')
    tc['rope_parameters'] = dict(tc['rope_parameters'], mrope_section=[1, 1, 0])
    vc = cfg['vision_config']
    vc.update(depth=2, hidden_size=32, intermediate_size=64, num_heads=2, out_hidden_size=64, dtype='float32')
    cfg.update(hidden_size=64, tie_word_embeddings=False, dtype='float32')
    for k in ('architectures', 'transformers_version', 'model_type'):
        cfg.pop(k, None)
    torch.manual_seed(0)
    config = Qwen3_5Config(**cfg)
    model = Qwen3_5ForConditionalGeneration(config).float()
    with torch.no_grad():
        E = model.get_input_embeddings().weight
        E[248211:] = E[:248211].mean(0, keepdim=True).expand(E.shape[0] - 248211, -1).clone() * 0.9
        H = model.get_output_embeddings().weight
        assert H is not E
        H[248211:] = 0.01
    if os.path.isdir(out_dir):
        shutil.rmtree(out_dir)
    model.save_pretrained(out_dir, safe_serialization=True)
    c = json.load(open(os.path.join(out_dir, 'config.json')))
    c['tie_word_embeddings'] = True                         # like every real checkpoint since stage 2
    c['text_config']['tie_word_embeddings'] = True
    json.dump(c, open(os.path.join(out_dir, 'config.json'), 'w'), indent=1)
    for fn in ('tokenizer.json', 'tokenizer_config.json', 'chat_template.jinja', 'preprocessor_config.json',
               'processor_config.json', 'generation_config.json'):
        if os.path.isfile(os.path.join(CKPT, fn)):
            shutil.copy2(os.path.join(CKPT, fn), os.path.join(out_dir, fn))
    open(marker, 'w').write('ok\n')
    return out_dir


def _toy_env(kind):
    _, d, _, _ = rows_and_sidecar()
    return dict(os.environ, PYTHONPATH=REPO, QURE_PLUGIN_MODE='train', SEM_TARGETS_DIR=d, SEM_W_COS='0.3',
                SEM_W_NCE='0.2', SEM_DETACH_STEPS='2', SEM_QUEUE='64', SEM_HEAD_DTYPE='float32', SEM_HEAD_HIDDEN='32',
                SEM_LOG_EVERY='1', SEM_TAU='0.1', SEM_TAU_T='0.05', IMAGE_MAX_TOKEN_NUM='64', MLD_HEAD_DTYPE='float32',
                SEM_EVAL_PRED_DIR=os.path.join(OUT, f'eval_{kind}'), OMP_NUM_THREADS='16', CUDA_VISIBLE_DEVICES='')


def _run_scenario(kind, timeout=3600):
    build_toy_ckpt()
    rows_and_sidecar()
    t0 = time.time()
    log = os.path.join(OUT, f'scenario_{kind}.log')
    with open(log, 'w') as f:
        p = subprocess.run([sys.executable, os.path.abspath(__file__), '--scenario', kind], stdout=f,
                           stderr=subprocess.STDOUT, env=_toy_env(kind), timeout=timeout, cwd=OUT)
    txt = open(log).read()
    res = None
    for line in txt.splitlines():
        if line.startswith('SCENARIO_RESULT '):
            res = json.loads(line[len('SCENARIO_RESULT '):])
    assert p.returncode == 0 and res is not None, f'scenario {kind} failed (rc {p.returncode}); log {log}:\n{txt[-3000:]}'
    res['_secs'] = time.time() - t0
    return res


def _scenario_main(kind):
    """Runs INSIDE a fresh subprocess (plugins patch classes process-wide)."""
    import numpy as np
    import torch
    disable_fla()
    torch.manual_seed(0)
    from swift.utils import import_external_file
    SP = os.path.join(REPO, 'swift_plugins')
    plugins = ['sem_latent_plugin.py'] if kind == 'sem' else [
        'mldecoder_joint_plugin.py', 'qure_augment_plugin.py', 'qure_ce_weight_plugin.py', 'qure_stage3_plugin.py',
        'sem_latent_plugin.py']
    for p in plugins:                                   # exactly what --external_plugins does, in list order
        import_external_file(os.path.join(SP, p))
    P = sys.modules['sem_latent_plugin']
    from swift.model.register import get_model_processor
    from swift.template import get_template
    from swift.template.sem_latent import find_sem_blocks, get_sem_ids
    model, proc = get_model_processor(TOY_DIR, torch_dtype=torch.float32, device_map='cpu',
                                      new_special_tokens=sem_tokens())
    tok = proc.tokenizer
    E0 = model.get_input_embeddings().weight.detach().clone()
    tpl = get_template(proc, **TEMPLATE_KW)
    tpl.set_mode('train')
    tpl.model = model
    args = SimpleNamespace(use_liger_kernel=False, is_adapter=False, tuner_type='full', freeze_parameters_ratio=0.0,
                           freeze_parameters=[], freeze_parameters_regex=None, trainable_parameters=[],
                           trainable_parameters_regex=None, use_galore=False, packing=False, padding_free=False,
                           sequence_parallel_size=1, model_type='qwen3_5')
    from swift.pipelines.train.tuner import TunerMixin
    model = TunerMixin().prepare_model(args, model, template=tpl)    # sft.py: self.prepare_model(...)
    R = {'kind': kind, 'len_tok': len(tok), 'emb_rows': int(model.get_input_embeddings().weight.shape[0])}
    sem = list(get_sem_ids(tok))
    R['K'] = len(sem)
    E1 = model.get_input_embeddings().weight.detach()
    R['sem_rows_reinit'] = bool(not torch.equal(E1[sem], E0[sem]) and (E1[sem][1:] - E1[sem][:1]).abs().max() > 0
                                and torch.equal(E1[:sem[0]], E0[:sem[0]]))
    R['cfg_untied'] = [model.config.tie_word_embeddings, model.config.text_config.tie_word_embeddings]
    R['chat_template_sem'] = '<|sem_0|>' in (proc.chat_template or '') and '<|sem_0|>' in (tok.chat_template or '')
    R['forward_hooks'] = [f'{getattr(h, "__module__", "?")}:{getattr(h, "__qualname__", "?")}'
                          for h in model._forward_hooks.values()]
    # --- data (exactly the LazyLLMDataset -> collator path)
    rows = pick_rows()
    sidecar_keys = {int(k) for k in P.get_reader().keys}
    encs = [tpl.encode(to_swift_row(raw), return_length=True) for _, raw in rows]
    batch = tpl.data_collator(encs)
    exp = [x for _, raw in rows for x in expected_blocks(raw, sidecar_keys)]
    got = list(zip(batch['sem_st'].tolist(), P._i64_to_u64(batch['sem_key'].numpy()).tolist()))
    ids = batch['input_ids']
    R['n_blocks_found'] = sum(len(find_sem_blocks(ids[b].tolist(), sem)) for b in range(ids.shape[0]))
    R['n_blocks_expected'] = len(exp)
    R['mapping_ok'] = got == exp
    R['n_valid_expected'] = sum(s == 1 for s, _ in exp)
    R['seq_len'] = list(ids.shape)
    R['has_cls_labels'] = 'cls_labels' in batch
    tpl.register_post_encode_hook([model])
    model.train()
    mld_head = getattr(model, 'ml_decoder_head', None)
    if mld_head is not None:
        # MLDecoder has dropout: in train mode two forwards give different ASL values. eval() on the
        # ML head ONLY makes the on/off comparison exact (the LM and sem_head stay in train mode).
        mld_head.eval()
    # independent capture of the final hidden states (registered AFTER the plugin's LM hook)
    cap = {}
    lm = model.model.language_model
    lm.register_forward_hook(lambda m, a, o: cap.__setitem__('h', o.last_hidden_state))
    mld = sys.modules.get('mldecoder_joint_plugin')

    def fwd():
        return model(**batch)

    # (a) plugin disabled -> baseline (CE [+ mldecoder ASL])
    P.set_enabled(False)
    ar0 = mld._RUN['ar'] if mld else None
    with torch.no_grad():
        out0 = fwd()
    base_loss = float(out0.loss)
    R['mld_ar_disabled'] = (mld._RUN['ar'] - ar0) if mld else None
    # (b) plugin enabled, step 0 < SEM_DETACH_STEPS
    P.set_enabled(True)
    P._STATE['keep_term'] = True
    P._SCHED['step'] = 0
    ar1 = mld._RUN['ar'] if mld else None
    out1 = fwd()
    term = P._STATE['last_term']
    s1 = dict(P._STATE['last'])
    R['mld_ar_enabled'] = (mld._RUN['ar'] - ar1) if mld else None
    R['mld_n'] = mld._RUN['n'] if mld else None
    R['stats_step0'] = s1
    R['loss_off'] = base_loss
    R['loss_on'] = float(out1.loss)
    R['diff'] = float(out1.loss) - base_loss
    R['diff_vs_incoming'] = float(out1.loss) - s1['ce']
    R['expected_diff'] = s1['w_cos'] * s1['cos_loss'] + s1['w_nce'] * s1['nce_loss']
    # independent recomputation from the captured hidden states (queue was empty for this forward)
    h = cap['h'].detach()
    blocks = [(b, s) for b in range(ids.shape[0]) for s in find_sem_blocks(ids[b].tolist(), sem)]
    hs = torch.stack([h[b, s:s + len(sem)] for b, s in blocks])
    with torch.no_grad():
        zh = model.sem_head(hs).numpy()
    valid = [j for j, (stt, _) in enumerate(exp) if stt == 1]
    r = P.get_reader()
    tv = np.stack([r.rows(r.lookup([exp[j][1]]))[0].astype(np.float32) for j in valid])
    kv = np.array([P._u64_to_i64([exp[j][1]])[0] for j in valid])
    img = P._u64_to_i64(batch['sem_img'].numpy())[valid]
    rc, rn, _, _ = ref_losses(zh[valid], tv, r.mu, kv, img, None, 'soft', 0.1, 0.05, 0.9)
    R['ref_diff'] = 0.3 * rc + 0.2 * rn
    # detach: the SEM term has no path into the LM (only into the head)
    layer_ps = [p for n, p in model.model.language_model.layers.named_parameters() if p.requires_grad]
    E = model.get_input_embeddings().weight
    hp = list(model.sem_head.parameters())
    g = torch.autograd.grad(term, layer_ps + [E] + hp, allow_unused=True, retain_graph=True)
    gl, ge, gh = g[:len(layer_ps)], g[len(layer_ps)], g[len(layer_ps) + 1:]
    R['detach_layer_grad'] = sum(0.0 if x is None else float(x.abs().sum()) for x in gl)
    R['detach_embed_sem_grad'] = 0.0 if ge is None else float(ge[sem].abs().sum())
    R['detach_head_grad'] = sum(0.0 if x is None else float(x.abs().sum()) for x in gh)
    out1.loss.backward()
    R['head_grads_defined_after_backward'] = all(p.grad is not None for p in model.sem_head.parameters())
    model.zero_grad(set_to_none=True)
    # (c) after the warm-up: gradients reach the LM through the SEM hidden states
    P._SCHED['step'] = 2
    out2 = fwd()
    term = P._STATE['last_term']
    s2 = dict(P._STATE['last'])
    R['stats_step2'] = s2
    g = torch.autograd.grad(term, layer_ps + [E] + hp, allow_unused=True, retain_graph=True)
    gl, ge, gh = g[:len(layer_ps)], g[len(layer_ps)], g[len(layer_ps) + 1:]
    per_layer = []
    for i, layer in enumerate(model.model.language_model.layers):
        ps = [p for p in layer.parameters() if p.requires_grad]
        gg = torch.autograd.grad(term, ps, allow_unused=True, retain_graph=True)
        per_layer.append(sum(0.0 if x is None else float(x.abs().sum()) for x in gg))
    R['post_layer_grad'] = per_layer
    R['post_embed_sem_grad'] = 0.0 if ge is None else float(ge[sem].abs().sum())
    R['post_head_grad'] = sum(0.0 if x is None else float(x.abs().sum()) for x in gh)
    out2.loss.backward()
    R['queue_len'] = len(P._queue())
    # (d) optimizer group (identity guard; create_optimizer runs twice in swift)
    opt = torch.optim.AdamW([p for n, p in model.named_parameters() if not n.startswith('sem_head.')], lr=1e-5)
    R['opt_added'] = [P.add_sem_param_group(opt, model), P.add_sem_param_group(opt, model)]
    # (e) eval forward -> val_sem payload + npz
    model.eval()
    with torch.no_grad():
        out3 = fwd()
    R['eval_loss_finite'] = bool(torch.isfinite(out3.loss))
    pay = P.evaluate_payload(7, out_dir=os.environ['SEM_EVAL_PRED_DIR'])
    R['eval_payload'] = {k: v for k, v in pay.items() if not isinstance(v, np.ndarray)}
    R['eval_npz'] = os.path.isfile(os.path.join(os.environ['SEM_EVAL_PRED_DIR'], 'sem_step000007.npz'))
    R['stale_stash'] = '_sem_batch' in model.__dict__
    # (f) train payload keys
    R['train_payload'] = P.train_payload()
    print('SCENARIO_RESULT ' + json.dumps(R, default=float), flush=True)



def _check_toy(R, kind):
    pre = f'toy[{kind}]'
    _rep('toy', f'{pre}: no resize (len(tok) 248219, rows 248320), K=8', R['len_tok'] == 248219
         and R['emb_rows'] == 248320 and R['K'] == 8, (R['len_tok'], R['emb_rows'], R['K']))
    _rep('toy', f'{pre}: SEM blocks found == merged assistant turns', R['n_blocks_found'] == R['n_blocks_expected'],
         (R['n_blocks_found'], R['n_blocks_expected']))
    _rep('toy', f'{pre}: valid mapping == independent raw-row derivation', R['mapping_ok']
         and R['stats_step0']['n_valid'] == R['n_valid_expected'] >= 4, R['stats_step0'])
    d, e, r = R['diff'], R['expected_diff'], R['ref_diff']
    _rep('toy', f'{pre}: loss(on) - loss(off) == w_cos*L_cos + w_nce*L_nce', abs(d - e) < 1e-5 * max(1, abs(e)),
         f'diff {d:.7f} expected {e:.7f}')
    _rep('toy', f'{pre}: == independent recomputation from captured hidden states', abs(d - r) < 2e-5 * max(1, abs(r)),
         f'diff {d:.7f} ref {r:.7f}')
    _rep('toy', f'{pre}: out.loss == loss seen by the SEM hook + SEM term', abs(R['diff_vs_incoming'] - e) < 1e-5,
         (R['diff_vs_incoming'], e))
    _rep('toy', f'{pre}: during SEM_DETACH_STEPS: no SEM grad into decoder layers / SEM embed rows, head yes',
         R['detach_layer_grad'] == 0.0 and R['detach_embed_sem_grad'] == 0.0 and R['detach_head_grad'] > 0,
         (R['detach_layer_grad'], R['detach_embed_sem_grad'], R['detach_head_grad']))
    _rep('toy', f'{pre}: after warm-up: SEM grad reaches sem_head, SEM embed rows, >=1 decoder layer',
         R['post_head_grad'] > 0 and R['post_embed_sem_grad'] > 0 and sum(x > 0 for x in R['post_layer_grad']) >= 1,
         (R['post_head_grad'], R['post_embed_sem_grad'], R['post_layer_grad']))
    _rep('toy', f'{pre}: sem_head grads defined after backward', R['head_grads_defined_after_backward'])
    _rep('toy', f'{pre}: queue filled with past targets (used as candidates at step 2)', R['queue_len'] >= 4
         and R['stats_step2']['n_cand'] > R['stats_step2']['n_valid'], (R['queue_len'], R['stats_step2']['n_cand']))
    _rep('toy', f'{pre}: SEM rows re-initialised (spare rows were identical), config untied, SEM jinja',
         R['sem_rows_reinit'] and R['cfg_untied'] == [False, False] and R['chat_template_sem'],
         (R['sem_rows_reinit'], R['cfg_untied'], R['chat_template_sem']))
    _rep('toy', f'{pre}: optimizer group added once', R['opt_added'][0] > 0 and R['opt_added'][1] == 0, R['opt_added'])
    ep = R['eval_payload']
    _rep('toy', f'{pre}: eval payload + npz, no stale stash', R['eval_loss_finite'] and R['eval_npz']
         and not R['stale_stash'] and all(f'val_sem/{k}' in ep for k in ('cos_centred', 'cos_raw', 'r1', 'r5', 'r10',
                                                                         'median_rank', 'margin', 'cos_raw_meanpred'))
         and ep.get('val_sem_unseen/n', 0) >= 1, sorted(ep)[:8])
    tp = R['train_payload']
    _rep('toy', f'{pre}: train/sem_* payload', all(f'train/sem_{k}' in tp for k in (
        'cos_loss', 'nce_loss', 'kl', 'cos_centred', 'cos_raw', 'valid_frac', 'miss_rate', 'w_cos', 'w_nce', 'share')),
         sorted(tp))


def test_toy_e2e_sem_only():
    R = _run_scenario('sem')
    _CACHE['toy_sem'] = R
    _check_toy(R, 'sem')
    sem_hooks = [h for h in R['forward_hooks'] if h.startswith('sem_latent_plugin:')]
    _rep('toy', 'toy[sem]: exactly one SEM loss hook on the model', len(sem_hooks) == 1, R['forward_hooks'])


def test_toy_e2e_composition_stage6c_order():
    R = _run_scenario('compose')
    _CACHE['toy_compose'] = R
    _check_toy(R, 'compose')
    hooks = R['forward_hooks']
    i_m = [i for i, h in enumerate(hooks) if h.startswith('mldecoder_joint_plugin:')]
    i_s = [i for i, h in enumerate(hooks) if h.startswith('sem_latent_plugin:')]
    _rep('toy', 'compose: hook order mldecoder loss hook BEFORE the SEM loss hook', len(i_m) == 1 and len(i_s) == 1
         and i_m[0] < i_s[0], hooks)
    _rep('toy', 'compose: mldecoder computed ASL on this batch (cls labels present)', R['has_cls_labels']
         and R['mld_n'] and R['mld_n'] >= 1, (R['has_cls_labels'], R['mld_n']))
    _rep('toy', "compose: mldecoder's logged 'ar' excludes SEM (same with SEM on and off)",
         R['mld_ar_enabled'] is not None and abs(R['mld_ar_enabled'] - R['mld_ar_disabled']) < 1e-6
         and abs(R['stats_step0']['ce'] - R['loss_off']) < 1e-6,
         (R['mld_ar_disabled'], R['mld_ar_enabled'], R['stats_step0']['ce'], R['loss_off']))


# ============================================================================================ runner
def _main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--scenario', default=None)
    ap.add_argument('-k', default=None)
    a = ap.parse_args()
    if a.scenario:
        _scenario_main(a.scenario)
        return 0
    tests = [(n, f) for n, f in sorted(globals().items(), key=lambda kv: getattr(kv[1], '__code__', None) and
                                       kv[1].__code__.co_firstlineno or 0)
             if n.startswith('test_') and callable(f) and (a.k is None or a.k in n)]
    failed = []
    t0 = time.time()
    for n, f in tests:
        t = time.time()
        try:
            f()
            print(f'[PASS] {n} ({time.time() - t:.1f}s)', flush=True)
        except Exception:  # noqa: BLE001
            failed.append(n)
            print(f'[FAIL] {n}\n{traceback.format_exc()}', flush=True)
    print('\n==== SEM plugin checks ====')
    for sec, chk, status, det in _REPORT:
        print(f'{status:4s} [{sec}] {chk}' + (f'  -- {det}' if status == 'FAIL' or os.environ.get('SEM_TEST_VERBOSE')
                                              else ''))
    for k in ('toy_sem', 'toy_compose'):
        if k in _CACHE:
            R = _CACHE[k]
            print(f'{k}: seq {R["seq_len"]} diff {R["diff"]:.7f} expected {R["expected_diff"]:.7f} ref {R["ref_diff"]:.7f} '
                  f'| step0 {R["stats_step0"]} | {R["_secs"]:.0f}s')
    n_pass = sum(1 for r in _REPORT if r[2] == 'PASS')
    print(f'{n_pass}/{len(_REPORT)} checks passed, {len(tests) - len(failed)}/{len(tests)} tests '
          f'({time.time() - t0:.0f}s)' + (f'; FAILED: {failed}' if failed else ''))
    return 1 if failed else 0


if __name__ == '__main__':
    sys.exit(_main())
