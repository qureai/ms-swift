#!/usr/bin/env python
"""SEM latent-token insertion: swift/template/sem_latent.py + the Template._encode_context_list hook.

RUN (CPU only; standalone or under pytest)
    CUDA_VISIBLE_DEVICES="" PYTHONPATH=/fastdata4/sahil_work/VLM_qure/vlm/swift \\
        /home/ubuntu/miniconda3/envs/ms-swift-may-s4/bin/python tests/qure/test_sem_insertion.py
Exits non-zero on any failure and prints a compact summary table at the end.

WHAT IS PROVEN
  (a) BACKWARD COMPAT. With the ORIGINAL checkpoint tokenizer (no SEM tokens) every encode output
      (every key, exact values: input_ids, labels, pixel_values, image_grid_thw, length(s), images,
      mm_processor_kwargs) is identical to the PRE-CHANGE code, for train / transformers / sglang /
      vllm modes (+ loss_scale=default and enable_thinking None/False variants) on real rows sampled
      from A_TRAIN_120k, B3_TRAIN (tool rows with loss:false, role tool, 2 images), 15_stage5,
      17_refresher and RET_diverse60k. The pre-change side runs in a subprocess on a COPY of the
      current swift/ tree with the backed-up pre-change files restored (base.py, ...) and
      sem_latent.py removed; the new side runs the same worker on the live tree. Also: with the
      original tokenizer the insertion function is never even called.
  (b) WITH SEM (8 tokens added in memory via swift's own --new_special_tokens path, reading
      examples/train/liger/token.txt + token_sem.txt): exactly one K-block right after every
      assistant header, no stray SEM ids, labels -100 on SEM, block count == merged assistant
      messages (train) / history + generation prompt (inference), removing the SEM ids gives back
      the no-SEM encode exactly (ids, labels, pixel_values, grids, images), the generation prompt
      ends with header+SEM+empty-think (enable_thinking False) or header+SEM+'<think>\\n' with a
      single WARNING (thinking), and mrope position_ids through the collator are consecutive over
      every SEM block.
  (c) JINJA PARITY. HF apply_chat_template with make_sem_chat_template(ckpt jinja) gives token ids
      IDENTICAL to swift sglang-mode encode: single-turn, multi-turn history (plain / with think /
      real report history), history already containing SEM text (idempotency), text-only and with
      an image; control: original jinja == no-SEM swift. install_sem_chat_template round-trip.
  (d) SWIFT_SEM_LATENT=0 disables insertion (encode == original tokenizer's encode) and unsetting
      it re-enables (cache revalidated per call).
  Plus pure-function unit checks, thread-safety of concurrent encodes on one template, cache
  invalidation on a deep-copied tokenizer, per-call overhead of the no-op path.

ENV
  SEM_TEST_CKPT                 checkpoint (default stage-6b checkpoint-608)
  SEM_TEST_OUT                  artefact dir (default <scratchpad>/tests_out/sem_insertion)
  SEM_TEST_BACKUP               pre-change backup root holding swift/... (default <scratchpad>/pre_change_backup)
  SEM_TEST_IMAGE_MAX_TOKEN_NUM  IMAGE_MAX_TOKEN_NUM for the encodes (default 2500 = the stage-6/7 launchers)
Nothing under the repo, the checkpoint or the datasets is written.
"""
import copy
import hashlib
import json
import logging
import os
import pickle
import random
import shutil
import subprocess
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SCRATCH = '/tmp/claude-1000/-fastdata4-sahil-work-VLM-qure-vlm-swift/b2c1bd9c-54da-4ec4-b14d-2d8473057a20/scratchpad'
if not os.path.isdir(SCRATCH):  # another host / session: artefacts go to the system tmp dir
    import tempfile
    SCRATCH = os.path.join(tempfile.gettempdir(), 'qure_sem_tests')
CKPT = os.environ.get('SEM_TEST_CKPT') or ('/fastdata4/sahil_work/VLM_qure/data_preparation/training_data/output/'
                                           'Qwen3.5-2B_stage6b_alltools/v0-20260906-173237/checkpoint-608')
OUT = os.environ.get('SEM_TEST_OUT') or os.path.join(SCRATCH, 'tests_out', 'sem_insertion')
BACKUP = os.environ.get('SEM_TEST_BACKUP') or os.path.join(SCRATCH, 'pre_change_backup')
DATA = '/fastdata4/sahil_work/VLM_qure/data_preparation/training_data'
FILES = {
    'A': f'{DATA}/stage6_toolcalling/A_TRAIN_120k.jsonl',
    'B3': f'{DATA}/stage6_toolcalling/B3_TRAIN.jsonl',
    'HG': f'{DATA}/15_stage5_headguided_TRAIN.json',
    'RF': f'{DATA}/17_refresher_multiturn_TRAIN.json',
    'DV': f'{DATA}/stage6_toolcalling/RET_diverse60k_TRAIN.jsonl',
}
TOKEN_TXT = os.path.join(REPO, 'examples/train/liger/token.txt')
TOKEN_SEM_TXT = os.path.join(REPO, 'examples/train/liger/token_sem.txt')
SYS = 'You are a helpful medical assistant.'
TEMPLATE_KW = dict(default_system=SYS, max_length=28000, truncation_strategy='raise', loss_scale='ignore_empty_think',
                   add_non_thinking_prefix=True, template_type='qwen3_5')
# mode_key -> (template mode, extra get_template kwargs, row variant)
MODES = {
    'train': ('train', {}, 'full'),
    'train_lsdefault': ('train', {'loss_scale': 'default'}, 'full'),
    'pt_nothink': ('transformers', {'enable_thinking': False}, 'prompt'),
    'pt_think': ('transformers', {}, 'prompt'),
    'pt_full_nothink': ('transformers', {'enable_thinking': False}, 'full'),
    'sglang_nothink': ('sglang', {'enable_thinking': False}, 'prompt'),
    'sglang_think': ('sglang', {}, 'prompt'),
    'vllm_nothink': ('vllm', {'enable_thinking': False}, 'prompt'),
}
H_IDS = [248045, 74455, 198]  # <|im_start|> assistant \n
NT_IDS = [248068, 271, 248069, 271]  # <think> \n\n </think> \n\n
TH_IDS = [248068, 198]  # <think> \n
K_EXPECT = 8
SEM_ID0 = 248211  # first free id of the ckpt tokenizer (len 248211)



def _apply_test_env():
    """This module's process env. Applied only when the module RUNS (standalone, its --worker children) or, under
    pytest, around ITS tests (module fixture below, restored afterwards) -- never at import, so collecting tests/
    (pytest, tests/run.py) changes nothing for any other test (review BC-3)."""
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    os.environ['IMAGE_MAX_TOKEN_NUM'] = os.environ.get('SEM_TEST_IMAGE_MAX_TOKEN_NUM') or '2500'
    os.environ['SKIP_BROKEN_IMAGES'] = '1'  # a broken image must raise, never fall back to a RANDOM image
    os.environ.pop('SWIFT_SEM_LATENT', None)
    os.environ.setdefault('OMP_NUM_THREADS', '8')


if __name__ == '__main__':
    _apply_test_env()
else:
    try:
        import pytest
    except ImportError:  # imported by another runner: no env side effects
        pytest = None
    if pytest is not None:

        @pytest.fixture(scope='module', autouse=True)
        def _qure_module_env():
            saved = dict(os.environ)
            _apply_test_env()
            try:
                yield
            finally:
                os.environ.clear()
                os.environ.update(saved)

_REPORT = []  # (section, check, status, detail)
_CACHE = {}


def _rep(section, check, ok, detail=''):
    _REPORT.append((section, check, 'PASS' if ok else 'FAIL', str(detail)))
    assert ok, f'[{section}] {check}: {detail}'


# ----------------------------------------------------------------------------------------------- helpers (both sides)
def _canon(v):
    """Comparable, compact form of an encode value (big tensors / images as sha256 of their exact bytes)."""
    import torch
    from PIL import Image
    if isinstance(v, torch.Tensor):
        t = v.detach().cpu().contiguous()
        raw = t.view(torch.int16) if t.dtype == torch.bfloat16 else t
        return ('T', str(t.dtype), tuple(t.shape), hashlib.sha256(raw.numpy().tobytes()).hexdigest())
    if isinstance(v, Image.Image):
        return ('PIL', v.mode, v.size, hashlib.sha256(v.tobytes()).hexdigest())
    if isinstance(v, dict):
        return {k: _canon(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return type(v)(_canon(x) for x in v)
    if v is None or isinstance(v, (bool, int, float, str)):
        return v
    return ('R', type(v).__name__, repr(v))


def _to_swift_row(d):
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
    if d.get('tools'):
        row['tools'] = d['tools']
    return row


def _n_tags(msgs):
    return sum(m['content'].count('<image>') for m in msgs if isinstance(m.get('content'), str))


def _prompt_row(row):
    """Inference prompt: messages up to the last user/tool message; images trimmed to the kept tags."""
    msgs = row['messages']
    last = max(i for i, m in enumerate(msgs) if m['role'] in ('user', 'tool', 'tool_response'))
    kept = copy.deepcopy(msgs[:last + 1])
    extra = max(0, len(row['images']) - _n_tags(msgs))  # images whose tags swift auto-adds to message 0
    out = {'messages': kept, 'images': copy.deepcopy(row['images'][:_n_tags(kept) + extra])}
    if 'tools' in row:
        out['tools'] = row['tools']
    return out


def _variant(row, variant):
    return copy.deepcopy(row) if variant == 'full' else _prompt_row(row)


def _load_processor(sem: bool):
    from swift.model.register import get_processor
    if not sem:
        return get_processor(CKPT)
    toks = []
    for p in (TOKEN_TXT, TOKEN_SEM_TXT):  # exactly what `--new_special_tokens token.txt token_sem.txt` does
        with open(p, encoding='utf-8') as f:
            toks += f.read().split()
    return get_processor(CKPT, new_special_tokens=toks)


def _templates(proc):
    from swift.template import get_template
    out = {}
    for key, (mode, kw, _) in MODES.items():
        t = get_template(proc, **{**TEMPLATE_KW, **kw})
        t.set_mode(mode)
        out[key] = t
    return out


def _encode_all(proc, rows, return_raw=False):
    """{(mode_key, row_id): canon(encode)} (+ raw encodes and template_inputs when return_raw)."""
    tpls = _templates(proc)
    res, raw = {}, {}
    for key, (mode, _, variant) in MODES.items():
        for rid, row in rows:
            r = _variant(row, variant)
            try:
                enc = tpls[key].encode(r, return_template_inputs=return_raw, return_length=(mode == 'train'))
            except Exception as e:  # noqa: BLE001 - recorded and asserted absent by the caller
                res[(key, rid)] = {'__error__': repr(e)}
                continue
            ti = enc.pop('template_inputs', None)
            res[(key, rid)] = _canon(enc)
            if return_raw:
                raw[(key, rid)] = (enc, ti)
    return (res, raw, tpls) if return_raw else res


def _worker(rows_path, out_path, expect_root):
    import swift
    swift_file = os.path.realpath(swift.__file__)
    assert swift_file.startswith(os.path.realpath(expect_root) + os.sep), (swift_file, expect_root)
    import importlib.util
    has_sem = importlib.util.find_spec('swift.template.sem_latent') is not None
    import swift.template.base as base_mod
    with open(base_mod.__file__, 'rb') as f:
        base_sha = hashlib.sha256(f.read()).hexdigest()
    with open(rows_path) as f:
        rows = [(r['id'], r['row']) for r in map(json.loads, f)]
    proc = _load_processor(sem=False)
    t0 = time.time()
    res = _encode_all(proc, rows)
    meta = {'swift_file': swift_file, 'has_sem_module': has_sem, 'base_sha': base_sha, 'secs': time.time() - t0,
            'len_tok': len(proc.tokenizer)}
    with open(out_path, 'wb') as f:
        pickle.dump({'meta': meta, 'res': res}, f)


# ----------------------------------------------------------------------------------------------- rows
def _first_user(d):
    m = d['messages'][0]
    return m.get('content') or m.get('value') or ''


def _roles(d):
    return [m.get('role') or m.get('from') for m in d['messages']]


def _asst_text(d):
    return ' '.join(str(m.get('content') or m.get('value') or '') for m in d['messages']
                    if (m.get('role') or m.get('from')) in ('assistant', 'gpt'))


WANT = [
    ('A_cls', 'A', lambda d: _first_user(d).startswith('Classify')),
    ('A_bbox', 'A', lambda d: 'bbox_2d' in _asst_text(d)),
    ('A_multi', 'A', lambda d: _roles(d).count('user') >= 3),
    ('B3_lossfalse', 'B3', lambda d: any(m.get('loss') is False for m in d['messages'])),
    ('B3_toolrole', 'B3', lambda d: 'tool' in _roles(d)),
    ('B3_plain', 'B3', lambda d: 'tool_call' not in _roles(d)),
    ('HG', 'HG', lambda d: True),
    ('RF_multi', 'RF', lambda d: _roles(d).count('human') >= 2),
    ('DV', 'DV', lambda d: True),
]


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


def _rows():
    if 'rows' in _CACHE:
        return _CACHE['rows']
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, 'rows.jsonl')
    rng = random.Random(20260930)
    rows = []
    for rid, fk, pred in WANT:
        fp = FILES[fk]
        size = os.path.getsize(fp)
        found = None
        with open(fp, 'rb') as f:
            for _ in range(4000):
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
        assert found is not None, f'no row found for {rid} in {fp}'
        rows.append((rid, _to_swift_row(found)))
    with open(path, 'w') as f:
        for rid, row in rows:
            f.write(json.dumps({'id': rid, 'row': row}, ensure_ascii=False) + '\n')
    _CACHE['rows'] = rows
    _CACHE['rows_path'] = path
    return rows


def _procs():
    if 'procs' not in _CACHE:
        _CACHE['procs'] = (_load_processor(sem=False), _load_processor(sem=True))
    return _CACHE['procs']


def _sem_ids():
    from swift.template.sem_latent import get_sem_ids
    return list(get_sem_ids(_procs()[1].tokenizer))


def _block_report(ids, sem):
    """(n_headers, n_blocks, all_headers_followed_by_block, all_blocks_preceded_by_header, n_sem_ids)."""
    from swift.template.sem_latent import find_sem_blocks
    k = len(sem)
    heads = [i for i in range(len(ids) - 2) if ids[i:i + 3] == H_IDS]
    blocks = find_sem_blocks(ids, sem)
    after = all(ids[i + 3:i + 3 + k] == sem for i in heads)
    before = all(s >= 3 and ids[s - 3:s] == H_IDS for s in blocks)
    n_sem = sum(1 for x in ids if x in set(sem))
    return len(heads), len(blocks), after, before, n_sem


def _merged_assistants(msgs):
    """Independent count of swift's merged assistant turns (tool_call -> assistant; consecutive merge)."""
    roles = []
    for m in msgs:
        r = m['role']
        if r == 'system':
            continue
        r = 'assistant' if r in ('assistant', 'tool_call') else ('tool' if r in ('tool', 'tool_response') else r)
        if not (roles and roles[-1] == r == 'assistant'):
            roles.append(r)
    return roles.count('assistant')


# ----------------------------------------------------------------------------------------------- tests
def test_pure_functions():
    from swift.template import sem_latent as S
    sem = '<|sem_0|><|sem_1|>'
    H = S.HEADER
    cl = [f'<|im_start|>user\nq<|im_end|>\n{H}<think>\n\n</think>\n\n', 'ANSWER', '<|im_end|>\n']
    ls = [0., 1., 0.]
    a, b = S.insert_sem_into_contexts(cl, ls, sem)
    _rep('unit', 'insert after header, before think', a[0].endswith(H + sem + '<think>\n\n</think>\n\n')
         and a[1:] == cl[1:] and b is ls and len(a) == len(cl), a[0][-60:])
    a2, b2 = S.insert_sem_into_contexts(a, b, sem)
    _rep('unit', 'idempotent (same objects on 2nd pass)', a2 is a and b2 is b)
    x = ['no header here', [1, 2, 3], 'x']
    xl = [0., 0., 0.]
    r1, r2 = S.insert_sem_into_contexts(x, xl, sem)
    _rep('unit', 'no header -> same objects', r1 is x and r2 is xl)
    r1, r2 = S.insert_sem_into_contexts(cl, ls, '')
    _rep('unit', 'empty sem_str -> same objects', r1 is cl and r2 is ls)
    y = [f'q{H}', f'{sem}hist']
    r1, _ = S.insert_sem_into_contexts(y, None, sem)
    _rep('unit', 'cross-boundary idempotency (header at end, next starts with SEM)', r1 is y)
    y = [f'q{H}', 'hist']
    r1, r2 = S.insert_sem_into_contexts(y, None, sem)
    _rep('unit', 'header at context end, loss_scale_list None', r1 == [f'q{H}{sem}', 'hist'] and r2 is None, r1)
    z = [f'{H}resp', f'{H}tail']
    r1, _ = S.insert_sem_into_contexts(z, [1., 0.], sem)
    _rep('unit', 'loss-scale>0 contexts untouched', r1 == [f'{H}resp', f'{H}{sem}tail'], r1)
    m = f'a{H}b{H}{sem}c{H}'
    r1, _ = S.insert_sem_into_contexts([m], [0.], sem)
    _rep('unit', 'every occurrence, skip already-SEM', r1 == [f'a{H}{sem}b{H}{sem}c{H}{sem}'], r1)
    _rep('unit', 'list (pre-tokenised) contexts untouched',
         S.insert_sem_into_contexts([[248045, 74455, 198]], [0.], sem)[0] == [[248045, 74455, 198]])
    ids = [1, 5, 6, 2, 5, 6, 5, 3, 5, 6]
    _rep('unit', 'find/count_sem_blocks', S.find_sem_blocks(ids, (5, 6)) == [1, 4, 8]
         and S.count_sem_blocks(ids, (5, 6)) == 3 and S.count_sem_blocks(ids, ()) == 0)
    import torch
    t = torch.tensor([[1, 5, 6, 0], [5, 6, 9, 9]])
    mk = S.sem_position_mask(t, (5, 6))
    _rep('unit', 'sem_position_mask', mk.dtype == torch.bool and mk.tolist() == [[False, True, True, False],
                                                                               [True, True, False, False]]
         and not S.sem_position_mask(t, ()).any() and S.count_sem_blocks(torch.tensor(ids), (5, 6)) == 3)
    _rep('unit', 'make_sem_chat_template(None pattern) -> None', S.make_sem_chat_template('{{ x }}', sem) is None
         and S.make_sem_chat_template('', sem) is None)
    with open(os.path.join(CKPT, 'chat_template.jinja')) as f:
        src = f.read()
    j = S.make_sem_chat_template(src, sem)
    j2 = S.make_sem_chat_template(j, sem)
    j3 = S.make_sem_chat_template(j, '<|sem_0|>')
    _rep('unit', 'make_sem_chat_template idempotent / K update', j is not None and j2 == j and j3 is not None
         and "set sem_tokens = '<|sem_0|>' %}" in j3 and j3.count('sem_tokens') == j.count('sem_tokens'))
    _rep('unit', 'get_sem_ids on non-tokenizers', S.get_sem_ids(None) == () and S.get_sem_ids(object()) == ()
         and S.get_sem_str(None) == '')


def test_a_backward_compat():
    rows = _rows()
    rows_path = _CACHE['rows_path']
    # pre-change tree = copy of the CURRENT swift/ with every backed-up pre-change file restored
    tree = os.path.join(OUT, 'prechange_tree')
    shutil.rmtree(tree, ignore_errors=True)
    shutil.copytree(os.path.join(REPO, 'swift'), os.path.join(tree, 'swift'),
                    ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    restored = []
    bsw = os.path.join(BACKUP, 'swift')
    for dp, _, fns in os.walk(bsw):
        for fn in fns:
            src = os.path.join(dp, fn)
            rel = os.path.relpath(src, bsw)
            shutil.copy2(src, os.path.join(tree, 'swift', rel))
            restored.append(rel)
    os.remove(os.path.join(tree, 'swift', 'template', 'sem_latent.py'))
    with open(os.path.join(bsw, 'template', 'base.py'), 'rb') as f:
        backup_base_sha = hashlib.sha256(f.read()).hexdigest()
    procs = {}
    for side, root in (('new', REPO), ('old', tree)):
        env = dict(os.environ, PYTHONPATH=root, CUDA_VISIBLE_DEVICES='')
        outp = os.path.join(OUT, f'encode_{side}.pkl')
        cmd = [sys.executable, os.path.abspath(__file__), '--worker', rows_path, outp, root]
        log = open(os.path.join(OUT, f'worker_{side}.log'), 'w')
        procs[side] = (subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT, cwd=OUT), outp, log)
    results = {}
    for side, (p, outp, log) in procs.items():
        rc = p.wait()
        log.close()
        assert rc == 0, f'{side} worker failed rc={rc}; see {log.name}'
        with open(outp, 'rb') as f:
            results[side] = pickle.load(f)
    mn, mo = results['new']['meta'], results['old']['meta']
    _rep('a', 'new side imports the live tree', mn['swift_file'].startswith(os.path.realpath(REPO)) and
         mn['has_sem_module'], mn['swift_file'])
    _rep('a', 'old side imports the pre-change copy', mo['swift_file'].startswith(os.path.realpath(tree))
         and not mo['has_sem_module'] and mo['base_sha'] == backup_base_sha,
         f"{mo['swift_file']} restored={restored}")
    rn, ro = results['new']['res'], results['old']['res']
    errs = [k for k, v in rn.items() if '__error__' in v] + [k for k, v in ro.items() if '__error__' in v]
    _rep('a', 'no encode errors', not errs, [(k, rn.get(k, ro.get(k))) for k in errs][:3])
    _rep('a', 'same record keys', set(rn) == set(ro), len(rn))
    by_mode = {}
    for key in rn:
        same = rn[key] == ro[key]
        by_mode.setdefault(key[0], []).append(same)
        if not same:
            diff = [kk for kk in set(rn[key]) | set(ro[key]) if rn[key].get(kk) != ro[key].get(kk)]
            print(f'[a] DIFF {key}: keys {diff}')
    for mode, v in by_mode.items():
        keys = sorted(next(rn[k] for k in rn if k[0] == mode).keys())
        _rep('a', f'{mode}: identical to pre-change code', all(v), f'{sum(v)}/{len(v)} rows; keys={keys}')
    n_img = sum(len(r['images']) for _, r in rows)
    _rep('a', 'row coverage', len(rows) >= 8, f'{len(rows)} real rows ({", ".join(r for r, _ in rows)}), '
         f'{n_img} images; worker secs new={mn["secs"]:.0f} old={mo["secs"]:.0f}')
    # the hook must not even call the insertion with the original tokenizer
    import swift.template.base as base_mod
    orig_proc, _ = _procs()
    calls, gets = [], []
    real, real_get = base_mod.insert_sem_into_contexts, base_mod.get_sem_str
    base_mod.insert_sem_into_contexts = lambda *a, **k: calls.append(1) or real(*a, **k)
    base_mod.get_sem_str = lambda *a, **k: gets.append(1) or real_get(*a, **k)
    try:
        tp = _templates(orig_proc)
        for key in ('train', 'sglang_nothink'):
            for _, row in rows[:3]:
                tp[key].encode(_variant(row, MODES[key][2]))
    finally:
        base_mod.insert_sem_into_contexts, base_mod.get_sem_str = real, real_get
    _rep('a', 'original tokenizer: hook runs, insertion never invoked', gets and not calls,
         f'get_sem_str calls={len(gets)}, insert calls={len(calls)}')


def test_b_sem_insertion():
    from swift.template import sem_latent as S
    rows = _rows()
    orig_proc, sem_proc = _procs()
    sem = _sem_ids()
    semset = set(sem)
    _rep('b', 'K and ids from token_sem.txt', sem == list(range(SEM_ID0, SEM_ID0 + K_EXPECT))
         and len(sem_proc.tokenizer) == SEM_ID0 + K_EXPECT and len(orig_proc.tokenizer) == SEM_ID0
         and S.get_sem_ids(orig_proc.tokenizer) == (), f'ids={sem[0]}..{sem[-1]} len={len(sem_proc.tokenizer)}')
    k = len(sem)
    # capture [sem] warnings
    recs = []

    class _H(logging.Handler):

        def emit(self, record):
            recs.append(record.getMessage())

    h = _H()
    S.logger.addHandler(h)
    S._WARNED.discard('thinking')
    try:
        base_res = _encode_all(orig_proc, rows)
        sem_res, sem_raw, sem_tpls = _encode_all(sem_proc, rows, return_raw=True)
    finally:
        S.logger.removeHandler(h)
    n_think_warn = sum('thinking prefix' in r for r in recs)
    otpl = _templates(orig_proc)['train']
    _rep('b', 'sem_active / get_sem_str accept template, processor, tokenizer',
         S.sem_active(sem_tpls['train']) and S.sem_active(sem_proc) and S.sem_active(sem_proc.tokenizer)
         and not S.sem_active(otpl) and not S.sem_active(orig_proc)
         and S.get_sem_str(sem_proc) == ''.join(S.SEM_TOKEN_FMT.format(i) for i in range(k)))
    stats = {}
    for (key, rid), (enc, ti) in sem_raw.items():
        mode, _, variant = MODES[key]
        base = base_res[(key, rid)]
        assert '__error__' not in base, base
        ids = enc['input_ids']
        n_h, n_b, after, before, n_sem = _block_report(ids, sem)
        n_asst = sum(1 for m in ti.messages if m['role'] == 'assistant')
        row = dict(rows)[rid]
        ok = n_h == n_b and after and before and n_sem == k * n_b and n_b == n_asst
        exp = _merged_assistants(_variant(row, variant)['messages']) + (1 if variant == 'prompt' else 0)
        ok_indep = n_b == exp
        stripped = [x for x in ids if x not in semset]
        ok_strip = stripped == base['input_ids']
        if 'labels' in enc:
            lab = enc['labels']
            ok_lab = all(lab[i] == -100 for i, x in enumerate(ids) if x in semset)
            ok_strip &= [lb for x, lb in zip(ids, lab) if x not in semset] == base['labels']
        else:
            ok_lab = True
        c = _canon(enc)
        others = {kk for kk in set(c) | set(base) if kk not in ('input_ids', 'labels', 'length', 'lengths')}
        ok_other = all(c.get(kk) == base.get(kk) for kk in others)
        if 'length' in c:
            ok_other &= c['length'] == base['length'] + k * n_b
        if variant == 'prompt':
            tail = NT_IDS if 'nothink' in key else TH_IDS
            ok_tail = ids[-(3 + k + len(tail)):] == H_IDS + sem + tail
        else:
            ok_tail = True
        s = stats.setdefault(key, {'rows': 0, 'blocks': 0, 'fail': []})
        s['rows'] += 1
        s['blocks'] += n_b
        for name, v in (('blocks', ok), ('indep_count', ok_indep), ('strip==base', ok_strip), ('labels-100', ok_lab),
                        ('other_keys', ok_other), ('gen_tail', ok_tail)):
            if not v:
                s['fail'].append(f'{rid}:{name}(h={n_h},b={n_b},asst={n_asst},exp={exp})')
    for key, s in stats.items():
        _rep('b', f'{key}: 1 block/header, labels -100, strip==no-SEM, count==assistants', not s['fail'],
             f"{s['rows']} rows, {s['blocks']} blocks {s['fail'][:3]}")
    n_think_prompts = sum(1 for (key, _) in sem_raw if key.endswith('_think'))
    _rep('b', 'thinking-prefix WARNING logged exactly once', n_think_warn == 1,
         f'{n_think_warn} warning(s) over {n_think_prompts} thinking prompts')
    # collator / mrope
    import torch
    t = sem_tpls['train']
    encs = [sem_raw[('train', rid)][0] for rid, _ in rows[:4]]
    batch = t.data_collator([copy.deepcopy(e) for e in encs])
    ii, pos = batch['input_ids'], batch['position_ids']
    mask = S.sem_position_mask(ii, sem)
    ok = pos.shape[-2:] == ii.shape and bool((mask & (batch['attention_mask'] == 0)).sum() == 0)
    per_row = [int(mask[b].sum()) for b in range(ii.shape[0])]
    ok &= per_row == [k * S.count_sem_blocks(e['input_ids'], sem) for e in encs]
    consec = True
    tpos = batch.get('text_position_ids')
    for b in range(ii.shape[0]):
        for s0 in S.find_sem_blocks(ii[b].tolist(), sem):
            seg = pos[:, b, s0 - 1:s0 + k]  # the header '\n' + the block
            consec &= bool((seg[:, 1:] - seg[:, :-1] == 1).all())
            if tpos is not None:
                consec &= bool((tpos[b, s0 - 1:s0 + k].diff() == 1).all())
    _rep('b', 'collator: mrope position_ids consecutive over SEM, mask rows ok', ok and consec,
         f"input_ids {tuple(ii.shape)} position_ids {tuple(pos.shape)} sem/row {per_row}")
    # sanity: every image expansion unchanged (pixel_values/grid digests equal) is covered by 'other_keys'
    _CACHE['sem_tpls'] = sem_tpls
    del torch


def _swift_ids(proc, msgs, images, enable_thinking=False):
    from swift.template import get_template
    key = ('swift_tpl', id(proc), enable_thinking)
    if key not in _CACHE:
        t = get_template(proc, default_system=SYS, template_type='qwen3_5', enable_thinking=enable_thinking)
        t.set_mode('sglang')
        _CACHE[key] = t
    return _CACHE[key].encode({'messages': copy.deepcopy(msgs), 'images': list(images)})['input_ids']


def _hf_ids(tok, msgs, jinja, enable_thinking=False):
    text = tok.apply_chat_template([{'role': 'system', 'content': SYS}] + msgs, tokenize=False,
                                   add_generation_prompt=True, enable_thinking=enable_thinking, chat_template=jinja)
    return tok.encode(text, add_special_tokens=False)


def test_c_jinja_parity():
    from PIL import Image
    from swift.template import sem_latent as S
    orig_proc, sem_proc = _procs()
    sem_tok, orig_tok = sem_proc.tokenizer, orig_proc.tokenizer
    sem_str = S.get_sem_str(sem_tok)
    jpath = os.path.join(CKPT, 'chat_template.jinja')
    with open(jpath, 'rb') as f:
        jbytes = f.read()
    src = jbytes.decode()
    sem_jinja = S.make_sem_chat_template(src, sem_str)
    _rep('c', 'make_sem_chat_template(ckpt-608 jinja)', sem_jinja is not None)
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, 'chat_template_sem.jinja'), 'w') as f:
        f.write(sem_jinja)
    img = Image.new('RGB', (96, 80), (120, 130, 140))
    THINK = '<think>\n\n</think>\n\n'
    # a real multi-turn history (A_multi row: cls/ASP/report answers)
    real = dict(_rows())['A_multi']['messages']
    real_hist = [{'role': m['role'], 'content': m['content']} for m in real[:4]]
    base_cases = {
        'single': [('u', 'Describe the film.')],
        'multi': [('u', 'Q1'), ('a', 'A1 answer.'), ('u', 'Q2')],
        'multi_think_hist': [('u', 'Q1'), ('a', THINK + 'A1 answer.'), ('u', 'Q2'), ('a', THINK + 'A2.'), ('u', 'Q3')],
        'hist_sem_text': [('u', 'Q1'), ('a', sem_str + 'A1 answer.'), ('u', 'Q2')],
        'hist_think_then_sem': [('u', 'Q1'), ('a', THINK + sem_str + 'A1 answer.'), ('u', 'Q2')],
        'hist_sem_then_think': [('u', 'Q1'), ('a', sem_str + THINK + 'A1 answer.'), ('u', 'Q2')],
        'real_history': [('u' if m['role'] == 'user' else 'a', m['content'].replace('<image>', '').strip())
                         for m in real_hist] + [('u', 'Now generate the Impression.')],
        # assistant AFTER the last user query -> the jinja's last-turn-with-reasoning branch
        'tool_step': [('u', 'Q1'), ('a', '<tool_call>\n<function=screening_head>\n</function>\n</tool_call>'),
                      ('t', '{"p": 0.5}')],
    }
    fails, n = [], 0
    for name, turns in base_cases.items():
        for with_img in (False, True):
            sw, hf = [], []
            for i, (r, c) in enumerate(turns):
                role = {'u': 'user', 'a': 'assistant', 't': 'tool'}[r]
                if role == 'user' and i == 0 and with_img:
                    sw.append({'role': role, 'content': '<image>' + c})
                    hf.append({'role': role, 'content': [{'type': 'image'}, {'type': 'text', 'text': c}]})
                else:
                    sw.append({'role': role, 'content': c})
                    hf.append({'role': role, 'content': c})
            imgs = [img] if with_img else []
            for et in (False, True):
                n += 1
                a = _swift_ids(sem_proc, sw, imgs, et)
                b = _hf_ids(sem_tok, hf, sem_jinja, et)
                a0 = _swift_ids(orig_proc, sw, imgs, et)
                b0 = _hf_ids(orig_tok, hf, src, et)
                n_asst = sum(1 for r, _ in turns if r == 'a') + 1
                tag = f'{name}/{"img" if with_img else "txt"}/think={et}'
                if a0 != b0:
                    fails.append(f'{tag}: CONTROL (no SEM) differs')
                if a != b:
                    fails.append(f'{tag}: SEM swift!=jinja')
                na, nb = S.count_sem_blocks(a, _sem_ids()), S.count_sem_blocks(b, _sem_ids())
                if na != n_asst or nb != n_asst:
                    fails.append(f'{tag}: blocks {na}/{nb} != {n_asst}')
                if [x for x in a if x not in set(_sem_ids())] != a0 and 'sem' not in name:
                    fails.append(f'{tag}: strip(SEM ids) != no-SEM ids')
    _rep('c', 'HF(SEM jinja) ids == swift sglang ids (+control, block counts)', not fails,
         f'{n - len(fails)}/{n} cases; {fails[:4]}')
    # install_sem_chat_template round trip (in-memory on the SEM processor copy only)
    ok1 = S.install_sem_chat_template(sem_proc)
    ok2 = sem_proc.chat_template == sem_jinja and sem_tok.chat_template == sem_jinja
    ok3 = S.install_sem_chat_template(sem_proc) and sem_proc.chat_template == sem_jinja
    ok4 = not S.install_sem_chat_template(orig_proc) and orig_proc.chat_template == src
    msgs = [{'role': 'user', 'content': [{'type': 'image'}, {'type': 'text', 'text': 'Q1'}]}]
    ok5 = _hf_ids(sem_tok, msgs, None) == _swift_ids(sem_proc, [{'role': 'user', 'content': '<image>Q1'}], [img])
    with open(jpath, 'rb') as f:
        ok6 = f.read() == jbytes
    _rep('c', 'install_sem_chat_template: installs, idempotent, no-op w/o SEM, ckpt file untouched',
         ok1 and ok2 and ok3 and ok4 and ok5 and ok6, [ok1, ok2, ok3, ok4, ok5, ok6])
    # what a checkpoint save ships (swift saves via processor.save_pretrained) and what a server reloads
    from transformers import AutoTokenizer
    sdir = os.path.join(OUT, 'saved_processor')
    shutil.rmtree(sdir, ignore_errors=True)
    sem_proc.save_pretrained(sdir)
    with open(os.path.join(sdir, 'chat_template.jinja')) as f:
        saved = f.read()
    with open(os.path.join(sdir, 'tokenizer_config.json')) as f:
        leaked = any('sem_cache' in kk for kk in json.load(f))
    t2 = AutoTokenizer.from_pretrained(sdir)
    _rep('c', 'save_pretrained ships SEM jinja + tokens; reload active', saved == sem_jinja and not leaked
         and S.get_sem_ids(t2) == tuple(_sem_ids()) and t2.chat_template == sem_jinja, sdir)


def test_d_env_disable():
    from swift.template import sem_latent as S
    rows = _rows()
    orig_proc, sem_proc = _procs()
    sub = [r for r in rows if r[0] in ('A_multi', 'B3_toolrole', 'RF_multi')]
    try:
        os.environ['SWIFT_SEM_LATENT'] = '0'
        off_ids = S.get_sem_ids(sem_proc.tokenizer)
        off = _encode_all(sem_proc, sub)
    finally:
        os.environ.pop('SWIFT_SEM_LATENT', None)
    base = _encode_all(orig_proc, sub)
    on_ids = S.get_sem_ids(sem_proc.tokenizer)
    same = all(off[k] == base[k] for k in base)
    _rep('d', 'SWIFT_SEM_LATENT=0: get_sem_ids==() and encode == original tokenizer (all modes)',
         off_ids == () and same, f'{sum(off[k] == base[k] for k in base)}/{len(base)} records')
    _rep('d', 'unset -> active again (cache revalidated)', len(on_ids) == K_EXPECT)


def test_thread_safety_and_cache():
    from swift.template import get_template
    from swift.template import sem_latent as S
    rows = _rows()
    orig_proc, sem_proc = _procs()
    t = get_template(sem_proc, **{**TEMPLATE_KW, 'enable_thinking': False})
    t.set_mode('sglang')
    jobs = [_prompt_row(r) for _, r in rows] * 3
    serial = [_canon(t.encode(copy.deepcopy(j))) for j in jobs]
    with ThreadPoolExecutor(8) as ex:
        par = list(ex.map(lambda j: _canon(t.encode(copy.deepcopy(j))), jobs))
    _rep('thread', 'concurrent encodes on one template == serial', par == serial, f'{len(jobs)} encodes x 8 threads')
    # cache invalidation: a deep copy carries the cache; extending it must be noticed
    tok = sem_proc.tokenizer
    t0 = time.time()
    tok2 = copy.deepcopy(tok)
    dc = time.time() - t0
    stale = getattr(tok2, S._CACHE_ATTR, None)
    tok2.add_special_tokens({'additional_special_tokens': [S.SEM_TOKEN_FMT.format(K_EXPECT)]})
    k2 = len(S.get_sem_ids(tok2))
    _rep('cache', 'deep-copied tokenizer extended to K=9 -> cache invalidated', stale is not None and k2 == K_EXPECT + 1
         and len(S.get_sem_ids(tok)) == K_EXPECT, f'K(copy)={k2}, deepcopy {dc:.1f}s')
    otok = orig_proc.tokenizer
    n = 20000
    t0 = time.time()
    for _ in range(n):
        S.get_sem_str(otok)
    off_us = (time.time() - t0) / n * 1e6
    t0 = time.time()
    for _ in range(n):
        S.get_sem_str(tok)
    on_us = (time.time() - t0) / n * 1e6
    _rep('cache', 'per-call cost of get_sem_str', off_us < 50 and on_us < 50,
         f'no-SEM {off_us:.2f} us, SEM {on_us:.2f} us (called twice per encode)')




def test_e_decode_lookup_and_warnings():
    """review P4 / P1 / BC-2: decode strips SEM strings only when SEM is active; the header-trained warning; the
    SEM lookup is SILENT on transformers' MistralCommonBackend (Tekken), whose convert_tokens_to_ids warns."""
    import logging
    import tempfile
    from swift.template import get_template
    from swift.template import sem_latent as S
    sem = ''.join(S.SEM_TOKEN_FMT.format(i) for i in range(8))
    _rep('unit', 'strip_sem: whole block, single ids of the block, K-bounded, no-op without SEM',
         S.strip_sem('a' + sem + 'b<|sem_3|>c<|sem_99|>', sem) == 'abc<|sem_99|>' and S.strip_sem('plain', sem) == 'plain'
         and S.strip_sem('x<|sem_0|>', '') == 'x<|sem_0|>' and S.strip_sem('', sem) == '')
    H = S.HEADER
    tr, inf = type('Tr', (), {'is_training': True})(), type('If', (), {'is_training': False})()
    S._WARNED.discard('header_trained')
    got = (S.warn_if_sem_header_trained(tr, [f'u{H}', 'resp'], [0., 1.]), S.warn_if_sem_header_trained(inf, [f'u{H}r'], [1.]),
           S.warn_if_sem_header_trained(tr, [f'u{H}', 'r'], None), 'header_trained' in S._WARNED,
           S.warn_if_sem_header_trained(tr, [f'u{H}resp'], [1.]), 'header_trained' in S._WARNED)
    _rep('unit', 'warn_if_sem_header_trained: only a TRAINING header in a loss-scale>0 str context',
         got == (False, False, False, False, True, True), got)
    orig_proc, _ = _procs()
    t = get_template(orig_proc, **TEMPLATE_KW)
    t.set_mode('transformers')
    tok = orig_proc.tokenizer
    ids = tok.encode('literal <|sem_0|> typed as text {"a": 1}', add_special_tokens=False)
    want = tok.decode(ids, spaces_between_special_tokens=False)
    got = t.decode(ids + [tok.convert_tokens_to_ids('<|im_end|>')], first_token=False)
    _rep('compat', 'non-SEM tokenizer: Template.decode unchanged (a literal <|sem_0|> in text is kept)',
         got == want and '<|sem_0|>' in got, (got, want))
    try:
        import mistral_common
        from transformers.tokenization_mistral_common import MistralCommonBackend
    except Exception as e:  # noqa: BLE001
        _rep('compat', 'Tekken lookup silent: SKIPPED (mistral_common unavailable)', True, repr(e))
        return
    d = tempfile.mkdtemp(prefix='tekken_', dir=OUT if os.path.isdir(OUT) else None)
    shutil.copy(os.path.join(os.path.dirname(mistral_common.__file__), 'data', 'tekken_240911.json'),
                os.path.join(d, 'tekken.json'))
    ttok = MistralCommonBackend.from_pretrained(d)
    seen = []

    class _Cap(logging.Handler):
        def emit(self, rec):
            if 'sem_' in rec.getMessage():
                seen.append(rec.getMessage())

    cap = _Cap()
    lg = logging.getLogger('transformers.tokenization_mistral_common')
    lg.addHandler(cap)
    try:
        ttok.convert_tokens_to_ids('<|sem_0|>')                  # control: the public API does warn
        n_ctrl = len(seen)
        t0 = time.time()
        res = {S.get_sem_str(ttok) for _ in range(100)}
        us = (time.time() - t0) / 100 * 1e6
    finally:
        lg.removeHandler(cap)
        shutil.rmtree(d, ignore_errors=True)
    _rep('compat', 'Tekken (MistralCommonBackend): SEM lookup inactive and SILENT (0 warnings / 100 calls)',
         n_ctrl == 1 and len(seen) == 1 and res == {''} and us < 100, f'control {n_ctrl}, after {len(seen)}, {us:.1f} us/call')


TESTS = [test_pure_functions, test_a_backward_compat, test_b_sem_insertion, test_c_jinja_parity, test_d_env_disable,
         test_thread_safety_and_cache, test_e_decode_lookup_and_warnings]


def main():
    failed = []
    t_all = time.time()
    for fn in TESTS:
        t0 = time.time()
        try:
            fn()
            print(f'[sem-test] {fn.__name__} OK ({time.time() - t0:.0f}s)', flush=True)
        except Exception:  # noqa: BLE001
            failed.append(fn.__name__)
            traceback.print_exc()
            print(f'[sem-test] {fn.__name__} FAILED ({time.time() - t0:.0f}s)', flush=True)
    w = max(len(c) for _, c, _, _ in _REPORT) if _REPORT else 10
    print('\n' + '=' * 110)
    print(f'{"sec":7s} {"check":{w}s} {"res":4s}  detail')
    print('-' * 110)
    for sec, check, st, det in _REPORT:
        print(f'{sec:7s} {check:{w}s} {st:4s}  {det.replace(chr(10), "|")[:160]}')
    print('=' * 110)
    n_fail = sum(1 for r in _REPORT if r[2] == 'FAIL')
    print(f'{len(_REPORT) - n_fail}/{len(_REPORT)} checks passed; tests failed: {failed or "none"}; '
          f'{time.time() - t_all:.0f}s; artefacts in {OUT}')
    return 1 if failed else 0


if __name__ == '__main__':
    if len(sys.argv) >= 2 and sys.argv[1] == '--worker':
        _worker(sys.argv[2], sys.argv[3], sys.argv[4])
        sys.exit(0)
    sys.exit(main())
