#!/usr/bin/env python
"""Tests for swift_plugins/sem_targets.py (SEM target rules + sidecar reader) and the offline
builder examples/train/liger/sem_latent/build_sem_targets.py.

Run standalone (exit code != 0 on failure; prints before/after canonicalisation examples):
    PYTHONPATH=<repo> CUDA_VISIBLE_DEVICES="" python tests/qure/test_sem_targets.py
or with pytest (functions are test_*).

Groups
  * pure rule tests on inline examples (always run)
  * real-row tests: rows sampled at evenly spaced byte offsets from the stage-6c mixture files
    (skipped when the data directory is absent)
  * sidecar round-trip on a synthetic sidecar (always run)
  * TEACHER end-to-end (opt-in, ~1-2 min on CPU): SEM_TEST_E2E=1 builds a tiny sidecar with
    --device cpu --dtype fp32 (or checks an existing one given by SEM_TEST_SIDECAR=<dir>) and
    verifies every collected key hits, rows are unit-norm, 2 rows match an independent bs=1
    manual encode (cos > 0.99999) and the probe logits match a manual probe pass.
Artefacts go to $SEM_TEST_OUT (default: the session scratchpad tests_out/).
"""
import json
import os
import pickle
import subprocess
import sys
import tempfile

import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
if REPO not in sys.path:
    sys.path.insert(0, REPO)
from swift_plugins import sem_targets as st  # noqa: E402

DATA = '/fastdata4/sahil_work/VLM_qure/data_preparation/training_data'
FILES = {
    'A': f'{DATA}/stage6_toolcalling/A_TRAIN_120k.jsonl', 'B3': f'{DATA}/stage6_toolcalling/B3_TRAIN.jsonl',
    'HG': f'{DATA}/15_stage5_headguided_TRAIN.json', 'RF': f'{DATA}/17_refresher_multiturn_TRAIN.json',
    'DV': f'{DATA}/stage6_toolcalling/RET_diverse60k_TRAIN.jsonl', 'AV': f'{DATA}/stage6_toolcalling/A_VAL.jsonl',
    'B3V': f'{DATA}/stage6_toolcalling/B3_VAL.jsonl',
}
OUT = os.environ.get('SEM_TEST_OUT', '/tmp/claude-1000/-fastdata4-sahil-work-VLM-qure-vlm-swift/'
                     'b2c1bd9c-54da-4ec4-b14d-2d8473057a20/scratchpad/tests_out')
BUILDER = os.path.join(REPO, 'examples', 'train', 'liger', 'sem_latent', 'build_sem_targets.py')
PY = sys.executable
VERBOSE = __name__ == '__main__'


def _show(tag, before, after, n=260):
    if VERBOSE:
        print(f'--- {tag}\n  BEFORE: {before[:n]!r}\n  AFTER : {(after or "")[:n]!r}')


ONTO_JSON = json.dumps({
    'Findings': '\n* Enlarged cardiac silhouette.\n* Small left pleural effusion.  ',
    'Summary': 'The chest radiograph shows cardiomegaly and a small left effusion.',
    'Complete Reasoning': 'The heart is enlarged ...',
    'Impression': 'Cardiomegaly with a small left pleural effusion.',
    'Recommendations': 'Follow-up radiograph.',
    'Prominence_Score': {'cardiomegaly': {'prominence_score': 3, 'reasoning': 'obvious'}},
}, indent=2)
ONTO_CANON = ('Findings: * Enlarged cardiac silhouette.\n* Small left pleural effusion.\n'
              'Summary: The chest radiograph shows cardiomegaly and a small left effusion.\n'
              'Impression: Cardiomegaly with a small left pleural effusion.')
P_ONTO = ('Generate a detailed ontology report of the given X-ray with Findings, Summary, Complete Reasoning, '
          'Impression, Recommendations, and Prominence Score.')
P_FICT = 'Generate the Findings, Impression, and CT / follow-up recommendations for the given chest X-ray.'
P_MEDREP = 'Generate a medical report of the given X-ray with Findings, Impression, and Recommendations.'
P_SEC = {f: f'Generate the {n} of the given chest X-ray.' for f, n in
         (('sec_findings', 'Findings'), ('sec_summary', 'Summary'), ('sec_impression', 'Impression'))}
EXCLUDED_PROMPTS = {
    'sec_reasoning': 'Generate the Complete Reasoning of the given chest X-ray.',
    'sec_prominence': 'Generate the Prominence Score for the prominent findings of the given chest X-ray.',
    'sjson': 'Provide a structured JSON report for the X-ray.',
    'graph': 'Generate a hierarchical positive findings graph and positive findings triples for the given image.',
    'cls': 'Classify the abnormalities in the given chest x-ray as a detailed JSON report.',
    'bbox': 'Generate the bbox of diseases and medical devices present in the image.',
    'asp': 'Generate Answer Set Programming (ASP)-based reasoning for the provided medical image.',
    'followup': 'Are there any follow-up recommendations?',
    'repcmp': 'Compare the radiographic findings across the images.',
    'describe': 'Describe every finding on this film.',
}


# ============================================================================================
# pure rules
# ============================================================================================

def test_api_constants():
    assert st.RULE_VERSION == 'v1'
    assert st.T1_FAMILIES == ('onto', 'fict', 'medrep', 'sec_findings', 'sec_summary', 'sec_impression')
    for name in ('clean_user_prompt', 'classify_turn', 'canonical_target', 'target_key', 'image_key', 'is_leaked',
                 'row_messages', 'iter_row_targets', 'SidecarReader', 'write_sidecar'):
        assert hasattr(st, name), name
    # pure module: importing it must not pull torch or swift
    code = ('import sys; sys.path.insert(0, %r); import swift_plugins.sem_targets; '
            'bad=[m for m in ("torch","swift","transformers") if m in sys.modules]; print(bad); '
            'sys.exit(1 if bad else 0)' % REPO)
    r = subprocess.run([PY, '-c', code], capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr


def test_module_alias_single_object():
    code = ('import sys; sys.path.insert(0, %r); sys.path.insert(0, %r); import swift_plugins.sem_targets as a; '
            'import sem_targets as b; sys.exit(0 if a is b else 1)' % (REPO, os.path.join(REPO, 'swift_plugins')))
    assert subprocess.run([PY, '-c', code]).returncode == 0


def test_clean_user_prompt():
    prior = ('Automated screening prior (verify each against the image):\nConfident: cardiomegaly\n'
             'Uncertain (near threshold — confirm or reject): effusion (0.55 vs thr 0.60)\n\n')
    assert st.clean_user_prompt(prior + P_ONTO + '\n<image>') == P_ONTO
    empty = ('Automated screening prior: no findings clearly flagged. Read the image carefully and report '
             'findings as usual.\n\n')
    assert st.clean_user_prompt(empty + P_FICT) == P_FICT
    for cue in st.TOOL_CUES:
        assert st.clean_user_prompt(P_ONTO + '\n' + cue + '\n<image>') == P_ONTO, cue
        assert st.clean_user_prompt(cue + '\nImage 1: <image>' + P_MEDREP) == P_MEDREP, cue
        assert st.clean_user_prompt(P_ONTO + '\n' + cue) == P_ONTO
    assert st.clean_user_prompt('<image>' + P_SEC['sec_findings']) == P_SEC['sec_findings']
    assert st.clean_user_prompt(None) == ''
    assert st.clean_user_prompt([{'type': 'text', 'text': P_ONTO}]) == P_ONTO
    # a cue only counts as a whole line: text that merely CONTAINS it is kept
    assert 'Solve this step by step using tool calls.' in st.clean_user_prompt(
        'Please: Solve this step by step using tool calls. ok')


def test_prompt_family():
    assert st.prompt_family(P_ONTO) == 'onto'
    assert st.prompt_family(P_FICT) == 'fict'
    assert st.prompt_family(P_MEDREP) == 'medrep'
    for f, p in P_SEC.items():
        assert st.prompt_family(p) == f
    for f, p in EXCLUDED_PROMPTS.items():
        assert st.prompt_family(p) is None, f
    assert st.prompt_family('Generate the Findings and the graph: hierarchical positive findings graph') is None


def test_canonical_onto_medrep_fict_sections():
    _show('onto', ONTO_JSON, st.canonical_target(ONTO_JSON, P_ONTO))
    assert st.canonical_target(ONTO_JSON, P_ONTO) == ONTO_CANON
    assert st.classify_turn(P_ONTO, ONTO_JSON) == 'onto'
    medrep = json.dumps({'Findings': '\n* Lungs are clear.', 'Impression': 'No acute disease.',
                         'Recommendations': 'None.'})
    assert st.canonical_target(medrep, P_MEDREP) == 'Findings: * Lungs are clear.\nImpression: No acute disease.'
    fict = ('Findings:\n* Low lung volumes.\n* Mild bibasilar opacity.\n\nImpression:\nLow volumes with '
            'atelectasis.\n\nRecommendations:\nNo CT needed.')
    _show('fict', fict, st.canonical_target(fict, P_FICT))
    assert st.canonical_target(fict, P_FICT) == ('Findings: * Low lung volumes.\n* Mild bibasilar opacity.\n'
                                                 'Impression: Low volumes with atelectasis.')
    assert st.canonical_target('\n* Heart size normal.\n* Lungs clear.', P_SEC['sec_findings']) == \
        'Findings: * Heart size normal.\n* Lungs clear.'
    assert st.canonical_target('No acute process.', P_SEC['sec_impression']) == 'Impression: No acute process.'
    assert st.canonical_target('Summary: Normal study.', P_SEC['sec_summary']) == 'Summary: Normal study.'
    # the SAME film gives the same Findings line whichever family carries it
    fl = st.canonical_target(ONTO_JSON, P_ONTO).split('\nSummary:')[0]
    assert st.canonical_target('\n* Enlarged cardiac silhouette.\n* Small left pleural effusion.',
                               P_SEC['sec_findings']) == fl


def test_rejections():
    empty_onto = json.dumps({'Findings': '', 'Summary': '', 'Complete Reasoning': '', 'Impression': '',
                             'Recommendations': '', 'Prominence_Score': {'Normal': 'Normal scan'}})
    cases = [
        (P_ONTO, empty_onto), (P_SEC['sec_summary'], ''), (P_SEC['sec_impression'], '  \n '),
        (P_FICT, 'Findings:\n\n\nImpression:\n\n\nRecommendations:\n'),
        (P_ONTO, '{"general": {"nodules": {"presence": true}}}'),                 # cls JSON under onto prompt
        (P_ONTO, '<tool_response>{"confident": []}</tool_response>'),              # tool response message
        (P_ONTO, '<tool_call>\n{"name": "screening_head"}\n</tool_call>'),
        (P_FICT, '{"Findings": "x"}'),                                             # fict must be prose
        (P_SEC['sec_findings'], '{"a": 1}'),
        (P_SEC['sec_findings'], "[{'bbox_2d': [1, 2, 3, 4], 'label': 'opacity'}]"),
        (P_SEC['sec_impression'], 'ok.'),                                          # < MIN_BODY_CHARS
    ]
    for p, a in cases:
        assert st.classify_turn(p, a) is None and st.canonical_target(a, p) is None, (p[:40], a[:60])
    for f, p in EXCLUDED_PROMPTS.items():
        assert st.classify_turn(p, ONTO_JSON) is None, f
        assert st.canonical_target('The heart is enlarged and there is a small effusion.', p) is None, f


def test_cleaning_variants():
    think = '<think>\n\n</think>\n\n' + ONTO_JSON
    assert st.canonical_target(think, P_ONTO) == ONTO_CANON
    assert st.canonical_target('<think>\nreasoning here\n</think>\n\n' + ONTO_JSON, P_ONTO) == ONTO_CANON
    assert st.canonical_target('reasoning\n</think>\n\n' + ONTO_JSON, P_ONTO) == ONTO_CANON  # thinking-mode output
    fence = '```json\n' + ONTO_JSON + '\n```'
    assert st.canonical_target(fence, P_ONTO) == ONTO_CANON
    pyrepr = repr(json.loads(ONTO_JSON))
    assert st.canonical_target(pyrepr, P_ONTO) == ONTO_CANON
    # swift-merged list content -> the LAST element (final answer after the tool trace)
    merged = ['<tool_call>\n<function=screening_head>\n</function>\n</tool_call>',
              '<tool_response>{"confident": []}</tool_response>', ONTO_JSON]
    assert st.canonical_target(merged, P_ONTO) == ONTO_CANON
    traced = '<tool_call>x</tool_call><tool_response>y</tool_response>' + ONTO_JSON
    assert st.canonical_target(traced, P_ONTO) == ONTO_CANON
    toks = ONTO_JSON.replace('Enlarged', '<CARDIOMEGALY>Enlarged') + '<|sem_0|><|sem_7|><|im_end|>'
    assert st.canonical_target('<|sem_0|><|sem_1|>' + toks, P_ONTO) == ONTO_CANON
    assert st.canonical_target('|<ASP>|No acute process.|</ASP>|', P_SEC['sec_impression']) == \
        'Impression: No acute process.'
    # list / dict section values are flattened
    d = json.dumps({'Findings': ['* A.', '* B.'], 'Impression': {'lungs': 'clear', 'heart': 'normal'}})
    assert st.canonical_target(d, P_MEDREP) == 'Findings: * A.\n* B.\nImpression: lungs: clear\nheart: normal'
    # cap
    big = json.dumps({'Findings': '* ' + 'x' * 9000})
    assert len(st.canonical_target(big, P_MEDREP)) <= st.MAX_CHARS


def test_format_only_detection():
    assert st.classify_turn(None, ONTO_JSON) == 'onto'
    assert st.canonical_target(ONTO_JSON) == ONTO_CANON
    assert st.classify_turn(None, json.dumps({'Findings': '* a b c', 'Impression': 'x y z'})) == 'medrep'
    assert st.classify_turn(None, 'Findings:\n* A.\n\nImpression:\nB.') == 'fict'
    assert st.classify_turn(None, 'No acute process.') is None
    assert st.classify_turn(None, '{"general": {}}') is None


def test_special_tokens_match_token_txt():
    p = os.path.join(REPO, 'examples', 'train', 'liger', 'token.txt')
    if not os.path.isfile(p):
        print('SKIP token.txt missing')
        return
    toks = [ln.strip() for ln in open(p) if ln.strip()]
    assert set(toks) == set(st.VLM_SPECIAL_TOKENS), set(toks) ^ set(st.VLM_SPECIAL_TOKENS)
    for t in toks:
        assert st.clean_assistant_text('a' + t + 'b') == 'ab', t


def test_target_key():
    assert st.target_key('abc') == 0xa9993e364706816a
    k = st.target_keys(['abc', ONTO_CANON])
    assert k.dtype == np.uint64 and int(k[0]) == 0xa9993e364706816a and int(k[1]) == st.target_key(ONTO_CANON)
    assert st.target_key('Findings: é') != st.target_key('Findings: e')


def test_image_key():
    a = st.image_key('/localstorage/cxr_data/chexpert/p1/s1/view1_frontal.png')
    assert a == st.image_key('/fastdata4/cxr_data/chexpert/p1/s1/view1_frontal.png')
    assert a == st.image_key('/fastdata3/cxr_data/chexpert/p1/s1/./view1_frontal.png')
    assert a == st.image_key('/localstorage/cxr_data/chexpert/p1/s1/image_512x640_view1_frontal.png')
    assert a == st.image_key({'path': '/localstorage/cxr_data/chexpert/p1/s1/view1_frontal.png'})
    assert a != st.image_key('/localstorage/cxr_data/chexpert/p2/s1/view1_frontal.png')   # basenames collide
    assert st.image_key(None) == 0 and st.image_key(b'bytes') == 0 and st.image_key('') == 0
    assert 0 < a < 2**64


def test_is_leaked():
    can = ONTO_CANON
    prior_json = json.dumps({'Findings': '\n* Enlarged cardiac silhouette.\n* Small left pleural effusion.'})
    assert st.is_leaked(can, ['Classify ...', prior_json])                       # escaped JSON prior
    assert st.is_leaked(can, ['* Enlarged cardiac silhouette.   * Small left pleural effusion.'])
    assert not st.is_leaked(can, ['* Enlarged cardiac silhouette.'])             # only part of Findings
    assert not st.is_leaked(can, [])
    sec = 'Impression: No acute process.'
    assert st.is_leaked(sec, [json.dumps({'Impression': 'No acute process.'})])
    assert not st.is_leaked(sec, ['No acute processes are absent'])
    assert st.is_leaked(sec, [['<tool_response>', '{"Impression": "No acute process."}']])   # list content
    assert st.leak_body(can) == '* Enlarged cardiac silhouette.\n* Small left pleural effusion.'
    assert st.split_canonical(can)['Impression'] == 'Cardiomegaly with a small left pleural effusion.'


def test_row_messages_and_groups():
    sharegpt = {'messages': [{'from': 'system', 'value': 'sys'}, {'from': 'human', 'value': P_ONTO + '\n<image>'},
                             {'from': 'gpt', 'value': ONTO_JSON}]}
    ms = st.row_messages(sharegpt)
    assert [m['role'] for m in ms] == ['user', 'assistant'] and ms[1]['content'] == ONTO_JSON
    assert st.row_messages({'conversations': sharegpt['messages']}) == ms
    b3 = {'messages': [{'role': 'user', 'content': P_ONTO + '\nUse tools where helpful and finish with the final answer.\n<image>'},
                       {'role': 'tool_call', 'content': '{"name": "screening_head", "arguments": {}}'},
                       {'role': 'assistant', 'content': '<tool_response>{"confident": []}</tool_response>', 'loss': False},
                       {'role': 'tool_call', 'content': '{"name": "crop", "arguments": {}}'},
                       {'role': 'tool', 'content': '<image>crop'},
                       {'role': 'assistant', 'content': ONTO_JSON}]}
    ms = st.row_messages(b3)
    assert ms[2]['loss'] is False and ms[0]['loss'] is None
    assert st.assistant_turn_groups(ms) == [[1, 2, 3], [5]]
    assert st.iter_row_targets(b3) == [(5, 'onto', ONTO_CANON)]


def test_iter_row_targets_template_g_and_overrides():
    row = {'messages': [
        {'role': 'user', 'content': '<image>' + EXCLUDED_PROMPTS['cls']}, {'role': 'assistant', 'content': '{"general": {}}'},
        {'role': 'user', 'content': P_SEC['sec_impression']}, {'role': 'assistant', 'content': 'No acute process.'},
        {'role': 'user', 'content': P_SEC['sec_findings']}, {'role': 'assistant', 'content': '\n* A is seen.\n* B.'},
        {'role': 'user', 'content': EXCLUDED_PROMPTS['sec_reasoning']}, {'role': 'assistant', 'content': 'Because.'},
    ]}
    got = st.iter_row_targets(row)
    assert got == [(3, 'sec_impression', 'Impression: No acute process.'),
                   (5, 'sec_findings', 'Findings: * A is seen.\n* B.')], got
    row2 = dict(row, sem_targets=[None, '  explicit text  ', None, 'last'])
    got2 = st.iter_row_targets(row2)
    assert (3, 'override', 'explicit text') in got2 and (7, 'override', 'last') in got2
    row3 = dict(row, sem_target='final override')
    assert st.iter_row_targets(row3)[-1] == (7, 'override', 'final override')
    row4 = dict(row, sem_targets=[None] * 4 + ['beyond'])
    assert st.iter_row_targets(row4)[-1] == (-1, 'override', 'beyond')


# ============================================================================================
# real rows
# ============================================================================================

def _spread_rows(path, n):
    size = os.path.getsize(path)
    with open(path, 'rb') as f:
        for i in range(n):
            f.seek(max(0, size * i // n - 1))
            if i:
                f.readline()
            line = f.readline()
            if line.strip():
                yield json.loads(line)


def _raw_prompt_family(u):
    """independent labeler (the data reader's prefixes, understand/fam.py) for EXCLUDED families"""
    s = st.clean_user_prompt(u)
    if 'hierarchical positive findings graph' in s:
        return 'graph'
    for f, p in (('cls', 'Classify the abnormalities'), ('bbox', 'Generate the bbox'), ('asp', 'Generate Answer Set'),
                 ('sjson', 'Provide a structured JSON report'), ('repcmp', 'Compare the radiographic findings'),
                 ('followup', 'Are there any follow-up'), ('sec_reasoning', 'Generate the Complete Reasoning'),
                 ('sec_prominence', 'Generate the Prominence Score'), ('describe', 'Describe every finding'),
                 ('nod_char', 'Characterise the nodule'), ('nod_assess', 'Assess the pulmonary nodule'),
                 ('lines', 'Check the lines and tubes'), ('verify', 'Here is a draft classification JSON'),
                 ('audit', 'Audit this draft answer'), ('region', 'Identify the abnormality or medical device')):
        if s.startswith(p):
            return f
    return None


def test_real_rows_every_family(n_per_file=600):
    if not all(os.path.isfile(p) for p in FILES.values()):
        print('SKIP real-data tests: data files missing')
        return
    found, excluded = {}, {}
    n_turns = n_t1 = 0
    for name, path in FILES.items():
        for row in _spread_rows(path, n_per_file):
            ms = st.row_messages(row)
            tg = {i: (f, c) for i, f, c in st.iter_row_targets(row)}
            u = None
            for i, m in enumerate(ms):
                if m['role'] == 'user':
                    u = m['content']
                    continue
                if m['role'] != 'assistant':
                    continue
                n_turns += 1
                raw = m['content'] if isinstance(m['content'], str) else str(m['content'])
                if i in tg:
                    fam, can = tg[i]
                    n_t1 += 1
                    assert fam == st.classify_turn(u, m['content']) and can == st.canonical_target(m['content'], u)
                    assert can.split(': ', 1)[0] in st.SECTION_NAMES, can[:40]
                    assert len(can) <= st.MAX_CHARS and can == can.strip()
                    assert '{' not in can[:2] and '<|' not in can and '<think>' not in can and '\n\n' not in can
                    sec = st.split_canonical(can)
                    assert list(sec) == [n for n in st.SECTION_NAMES if n in sec]   # canonical order
                    if fam in ('onto', 'medrep'):
                        o = json.loads(st.clean_assistant_text(raw))
                        assert sec['Findings'] == st._norm_body(o['Findings'])
                        assert ('Summary' in sec) == (fam == 'onto' and bool(str(o.get('Summary', '')).strip()))
                    found.setdefault(fam, (name, raw, can))
                else:
                    ef = _raw_prompt_family(u)
                    if ef:
                        excluded.setdefault(ef, (name, raw))
                    t = st.clean_assistant_text(raw) if st.prompt_family(u) else ''
                    if t:
                        # a T1 prompt whose non-empty answer was rejected must be an EMPTY report
                        # (census: every rejection in the mixture is '' or all-empty sections)
                        o = st._parse_obj(t)
                        if isinstance(o, dict):
                            body = ''.join(str(v) for k, v in o.items()
                                           if st._norm_key(k) in ('findings', 'summary', 'impression'))
                        else:
                            body = st._PROSE_HDR_RE.sub('', t)
                        assert not body.strip() or len(body.strip()) < st.MIN_BODY_CHARS, (name, t[:200])
    for fam in st.T1_FAMILIES:
        assert fam in found, f'no real example of {fam}'
        name, raw, can = found[fam]
        _show(f'{fam} ({name})', raw, can)
    for fam in ('sec_reasoning', 'sjson', 'graph', 'cls', 'bbox', 'asp'):
        assert fam in excluded, f'no real example of excluded family {fam}'
    for fam, (name, raw) in sorted(excluded.items()):
        _show(f'EXCLUDED {fam} ({name})', raw, None, n=120)
    if VERBOSE:
        print(f'real rows: {n_turns} assistant turns, {n_t1} supervised; T1 families {sorted(found)}; '
              f'excluded seen {sorted(excluded)}')


def test_real_cross_family_same_film():
    """template-G rows carry several sections of ONE film: their Findings/Impression lines must
    equal the corresponding lines of the onto / medrep canonical text of the same row."""
    paths = [FILES['RF'], FILES['DV'], FILES['A']]
    if not all(os.path.isfile(p) for p in paths):
        print('SKIP real-data tests: data files missing')
        return
    checked = 0
    for path in paths:
        for row in _spread_rows(path, 3000):
            tg = st.iter_row_targets(row)
            fams = {f: c for _, f, c in tg}
            full = fams.get('onto') or fams.get('medrep')
            if not full:
                continue
            fs = st.split_canonical(full)
            for f, n in (('sec_findings', 'Findings'), ('sec_impression', 'Impression'), ('sec_summary', 'Summary')):
                if f in fams and n in fs:
                    assert st.split_canonical(fams[f])[n] == fs[n], (f, fams[f][:80], fs[n][:80])
                    checked += 1
    assert checked >= 20, checked
    if VERBOSE:
        print(f'cross-family same-film section equality: {checked} pairs checked')


# ============================================================================================
# sidecar
# ============================================================================================

def _synthetic(n=50, dim=16, seed=0):
    rng = np.random.default_rng(seed)
    texts = [f'Findings: synthetic {i}' for i in range(n)]
    keys = st.target_keys(texts)
    e = rng.standard_normal((n, dim)).astype(np.float32) + 0.5
    e /= np.linalg.norm(e, axis=1, keepdims=True)
    return texts, keys, e


def test_sidecar_roundtrip():
    texts, keys, e = _synthetic()
    d = tempfile.mkdtemp(prefix='sem_sidecar_', dir=OUT if os.path.isdir(OUT) else None)
    in_train = (np.arange(len(keys)) % 3 != 0).astype(np.uint8)
    probe = np.arange(len(keys) * 5, dtype=np.float32).reshape(len(keys), 5)
    fams = np.full(len(keys), st.FAMILY_BITS['onto'], np.uint8)
    # duplicate one row (same key, same vector): merged, flags OR-ed
    keys2 = np.concatenate([keys, keys[:1]])
    e2 = np.concatenate([e, e[:1]])
    it2 = np.concatenate([in_train, [1]])
    fams2 = np.concatenate([fams, [st.FAMILY_BITS['sec_findings']]])
    meta = st.write_sidecar(d, keys2, e2, it2, {'rule_version': st.RULE_VERSION, 'teacher_path': 'x'},
                            probe_logits=np.concatenate([probe, probe[:1]]), families=fams2,
                            texts=texts + texts[:1])
    assert meta['n'] == len(keys) and meta['dim'] == e.shape[1]
    r = st.SidecarReader(d)
    assert isinstance(r.emb, np.memmap) and r.n == len(keys) and r.dim == e.shape[1]
    r.check_rule_version()
    assert np.allclose(r.mu, e.mean(0), atol=1e-6)
    idx = r.lookup(keys)
    assert (idx >= 0).all() and np.array_equal(np.asarray(r.keys)[idx], keys)
    rows = r.rows(idx)
    assert rows.dtype == np.float16 and np.allclose(rows.astype(np.float32), e, atol=2e-3)
    assert np.abs(np.linalg.norm(rows.astype(np.float32), axis=1) - 1).max() < 2e-3
    miss = r.lookup(np.array([1, 2, 3], dtype=np.uint64))
    assert (miss == -1).all()
    mixed = np.array([idx[4], -1, idx[0]])
    rr = r.rows(mixed)
    assert (rr[1] == 0).all() and np.allclose(rr[0].astype(np.float32), e[4], atol=2e-3)
    assert r.in_train(mixed).tolist() == [bool(in_train[4]), False, True]
    pl = r.probe_logits(mixed)
    assert pl.shape == (3, 5) and np.allclose(pl[0], probe[4]) and (pl[1] == 0).all()
    assert int(r.families(idx[:1])[0]) == st.FAMILY_BITS['onto'] | st.FAMILY_BITS['sec_findings']
    assert r.texts([idx[7], -1]) == [texts[7], None]
    assert r.lookup_texts([texts[9]])[0] == idx[9]
    assert r.lookup(np.zeros((2, 3), np.uint64)).shape == (2, 3)
    r2 = pickle.loads(pickle.dumps(r))                       # dataloader workers / spawn
    assert np.array_equal(r2.rows(idx[:5]), rows[:5])
    r3 = st.SidecarReader(d, mmap=False)
    assert np.array_equal(r3.lookup(keys), idx)
    try:                                                     # complete sidecar is never overwritten silently
        st.write_sidecar(d, keys, e, in_train, {})
        raise AssertionError('overwrite not refused')
    except FileExistsError:
        pass
    bad = tempfile.mkdtemp(prefix='sem_bad_', dir=OUT if os.path.isdir(OUT) else None)
    try:
        st.write_sidecar(bad, keys, e * 2, in_train, {})
        raise AssertionError('non-normalised rows accepted')
    except ValueError:
        pass
    try:
        st.write_sidecar(bad, keys[:2].repeat(2)[:3], e[:3], in_train[:3], {})
        raise AssertionError('conflicting duplicate accepted')
    except ValueError:
        pass
    # rule mismatch is detected
    m = json.load(open(os.path.join(d, 'meta.json')))
    m['rule_version'] = 'v0'
    json.dump(m, open(os.path.join(d, 'meta.json'), 'w'))
    try:
        st.SidecarReader(d).check_rule_version()
        raise AssertionError('rule mismatch not detected')
    except ValueError:
        pass


def test_builder_json_array_and_limit():
    """collect stage on synthetic files: JSONL + a true JSON array (+ '#N' suffix, overrides)."""
    sys.path.insert(0, os.path.dirname(BUILDER))
    import build_sem_targets as bst
    d = tempfile.mkdtemp(prefix='sem_build_', dir=OUT if os.path.isdir(OUT) else None)
    rows = []
    for i in range(30):
        rows.append({'messages': [{'role': 'user', 'content': '<image>' + P_SEC['sec_impression']},
                                  {'role': 'assistant', 'content': f'Impression number {i % 20}.'}]})
    rows.append({'messages': [{'from': 'human', 'value': P_ONTO}, {'from': 'gpt', 'value': ONTO_JSON}],
                 'sem_target': 'An explicit override target.'})
    jl = os.path.join(d, 'a.jsonl')
    with open(jl, 'w') as f:
        for j, r in enumerate(rows):
            f.write(json.dumps(r) + '\n')
            if j == 3:
                f.write('{"messages": [truncated\n')                          # one corrupt line: skipped + counted
    ja = os.path.join(d, 'b.json')
    with open(ja, 'w') as f:
        json.dump(rows[:5] + [{'messages': [{'role': 'user', 'content': P_MEDREP},
                                            {'role': 'assistant', 'content': json.dumps({'Findings': '* only val'})}]}],
                  f, indent=1)
    assert bst.detect_format(jl) == 'jsonl' and bst.detect_format(ja) == 'json_array'
    assert len(list(bst.iter_json_array(ja, bufsize=64))) == 6            # tiny buffer: incremental path
    out = os.path.join(d, 'side')
    args = bst.parse_args(['--train_datasets', jl + '#10', '--val_datasets', ja, '--out', out, '--collect_only',
                           '--workers', '4', '--chunk_mb', '1'])
    col = bst.collect(args)
    man = col['manifest']
    assert man['n_unique'] == 20 + 1 + 1 + 1, man['n_unique']              # 20 impressions + onto + override + val medrep
    assert man['n_turns_per_family']['sec_impression'] == 35 and man['n_turns_per_family']['override'] == 1
    assert man['bad_lines'] == 1, man['bad_lines']
    assert man['n_unique_in_val'] == 6 and man['n_unique_val_seen_in_train'] == 5
    t2 = dict(zip(col['texts'], col['in_train']))
    assert t2['Findings: * only val'] == 0 and t2[ONTO_CANON] == 1 and t2['An explicit override target.'] == 1
    assert list(col['keys']) == sorted(col['keys'])
    col2 = bst.collect(args)                                              # cached
    assert np.array_equal(col2['keys'], col['keys'])
    # --limit: family-balanced subset
    args_l = bst.parse_args(['--train_datasets', jl, '--out', os.path.join(d, 'lim'), '--collect_only', '--limit', '3'])
    col3 = bst.collect(args_l)
    fams = {int(b) for b in col3['bits']}
    assert len(col3['keys']) == 3 and len(fams) == 3, fams                 # sec_impression, onto, override
    # batch planning respects the token budget
    L = np.array([5, 100, 30, 30, 7, 1000, 64])
    bs = bst.plan_batches(L, 512, 128, 8)
    assert sorted(np.concatenate(bs).tolist()) == list(range(len(L)))
    assert all(len(b) * min(int(L[b].max()), 512) <= 128 or len(b) == 1 for b in bs)



def test_builder_resume_provenance():
    """review TGT-2: a chunk / finished part is reused only when its provenance (teacher fingerprint, dtype,
    max_length, probe head path + sha256, RULE_VERSION) equals the current run's -- a re-run with another dtype or
    probe head re-encodes instead of mixing old rows into a sidecar whose meta records the new settings."""
    import hashlib
    sys.path.insert(0, os.path.dirname(BUILDER))
    import build_sem_targets as bst
    d = tempfile.mkdtemp(prefix='sem_prov_', dir=OUT if os.path.isdir(OUT) else None)
    jl = os.path.join(d, 'a.jsonl')
    with open(jl, 'w') as f:
        for i in range(6):
            f.write(json.dumps({'messages': [{'role': 'user', 'content': '<image>' + P_SEC['sec_impression']},
                                             {'role': 'assistant', 'content': f'Impression number {i}.'}]}) + '\n')
    probes = []
    for n in (1, 2):
        pp = os.path.join(d, f'probe{n}.pt')
        with open(pp, 'wb') as f:
            f.write(b'probe head bytes %d' % n)           # only hashed: the stub teacher never loads it
        probes.append(pp)

    class StubTeacher:
        encoded = []

        def __init__(self, args):
            self.args, self.probe = args, None

        def token_lengths(self, texts):
            return np.array([len(t) for t in texts])

        def encode(self, texts):
            StubTeacher.encoded.extend(texts)
            tag = f'{self.args.dtype}|{self.args.max_length}|{self.args.probe_head}'
            z = np.stack([np.random.default_rng(int.from_bytes(hashlib.sha256((tag + t).encode()).digest()[:8],
                                                                'big')).standard_normal(st.EMB_DIM) for t in texts])
            return (z / np.linalg.norm(z, axis=1, keepdims=True)).astype(np.float32), None

    real_teacher = bst.Teacher
    bst.Teacher = StubTeacher
    info = {'teacher_fingerprint': 'f' * 64, 'fingerprint_expected': 'f' * 64, 'fingerprint_ok': True}

    def run(out, dtype='fp32', probe=None, stop=0):
        argv = ['--train_datasets', jl, '--out', out, '--chunk_texts', '2', '--batch_tokens', '400', '--max_batch', '2',
                '--dtype', dtype, '--device', 'cpu', '--probe_head', probe or 'none', '--workers', '1']
        if stop:
            argv += ['--stop_after_chunks', str(stop)]
        a = bst.parse_args(argv)
        col = bst.collect(a)
        StubTeacher.encoded = []
        try:
            bst.encode_shard(a, col, info)
            done = True
        except SystemExit:
            done = False
        rows = None
        part = os.path.join(out, 'parts', 'part_0.npz')
        if done:
            with np.load(part) as z:
                rows = dict(zip(z['keys'].tolist(), z['emb'].astype(np.float32)))
        return done, len(StubTeacher.encoded), rows

    try:
        clean = {k: run(os.path.join(d, f'clean_{k}'), dtype=dt, probe=pr)[2]
                 for k, dt, pr in (('fp32', 'fp32', None), ('bf16', 'bf16', None), ('p1', 'fp32', probes[0]),
                                   ('p2', 'fp32', probes[1]))}
        same = lambda a, b: a is not None and set(a) == set(b) and all(np.array_equal(a[k], b[k]) for k in a)  # noqa
        # 1. same provenance: a killed run resumes from its chunk (2 of 6 texts not re-encoded)
        o = os.path.join(d, 'resume_same')
        r1 = run(o, stop=1)
        r2 = run(o)
        assert r1[:2] == (False, 2) and r2[0] and r2[1] == 4 and same(r2[2], clean['fp32']), (r1[:2], r2[:2])
        # 2. dtype changed between the kill and the resume: the old chunk is NOT reused
        o = os.path.join(d, 'resume_dtype')
        run(o, dtype='fp32', stop=1)
        r = run(o, dtype='bf16')
        assert r[0] and r[1] == 6 and same(r[2], clean['bf16']), r[:2]
        pj = json.load(open(os.path.join(o, 'parts', 'part_0.json')))
        assert pj['dtype'] == 'bf16' and pj['probe_sha256'] is None and pj['rule_version'] == st.RULE_VERSION
        # 3. another probe head between the kill and the resume: re-encoded
        o = os.path.join(d, 'resume_probe')
        run(o, probe=probes[0], stop=1)
        r = run(o, probe=probes[1])
        assert r[0] and r[1] == 6 and same(r[2], clean['p2']), r[:2]
        # 4. a FINISHED part is skipped only under the same provenance
        r = run(o, probe=probes[1])
        assert r[0] and r[1] == 0 and same(r[2], clean['p2']), r[:2]      # already complete -> skip
        r = run(o, probe=probes[0])
        assert r[0] and r[1] == 6 and same(r[2], clean['p1']), r[:2]      # provenance differs -> re-encode
        r = run(o, probe=probes[0], dtype='bf16')
        assert r[0] and r[1] == 6, r[:2]
        # 5. the SAME probe path with other bytes (a refitted head saved over the old one): re-encoded
        r = run(o, probe=probes[0], dtype='bf16')
        assert r[0] and r[1] == 0, r[:2]
        with open(probes[0], 'wb') as f:
            f.write(b'refitted probe head')
        r = run(o, probe=probes[0], dtype='bf16')
        assert r[0] and r[1] == 6, r[:2]
    finally:
        bst.Teacher = real_teacher


# ============================================================================================
# teacher end-to-end (opt-in)
# ============================================================================================

def verify_sidecar(side_dir, teacher=None, n_manual=2, threads=64):
    """Every collected key hits; unit-norm rows; n_manual rows == independent bs=1 fp32 encode."""
    r = st.SidecarReader(side_dir)
    r.check_rule_version()
    col = [json.loads(ln) for ln in open(os.path.join(side_dir, 'collect', 'targets.jsonl'))]
    keys = np.array([int(c['k']) for c in col], dtype=np.uint64)
    assert np.array_equal(keys, st.target_keys([c['t'] for c in col])), 'collected key != sha1 rule'
    idx = r.lookup(keys)
    assert (idx >= 0).all(), f'{int((idx < 0).sum())} misses'
    rows = r.rows(idx).astype(np.float32)
    nrm = np.linalg.norm(rows, axis=1)
    assert np.abs(nrm - 1).max() < 2e-3, (nrm.min(), nrm.max())
    assert np.array_equal(r.in_train(idx), np.array([bool(c['tr']) for c in col]))
    out = {'n': r.n, 'hits': int((idx >= 0).sum()), 'norm_min': float(nrm.min()), 'norm_max': float(nrm.max()),
           'mu_norm': float(np.linalg.norm(r.mu))}
    if n_manual:
        import torch
        import torch.nn.functional as F
        from transformers import AutoModel, AutoTokenizer
        torch.set_num_threads(threads)
        tp = teacher or r.meta['teacher_path']
        tok = AutoTokenizer.from_pretrained(tp)          # independent path: bs=1 -> no padding at all
        model = AutoModel.from_pretrained(tp, dtype=torch.float32).eval()
        pick = [0, len(col) - 1][:n_manual]
        cos = []
        for j in pick:
            enc = tok([col[j]['t'].strip()], return_tensors='pt', truncation=True,
                      max_length=r.meta['max_length'], add_special_tokens=True)
            assert int(enc['input_ids'][0, -1]) == st.TEACHER_EOT_ID
            with torch.no_grad():
                z = F.normalize(model(**enc).last_hidden_state[0, -1].float(), dim=-1).numpy()
            cos.append(float(z @ rows[j] / np.linalg.norm(rows[j])))
            pl = r.probe_logits(idx[j:j + 1])
            if pl is not None:
                ck = torch.load(r.meta['probe_head'], map_location='cpu', weights_only=False)
                W = ck['state_dict']['fc.weight'].float().numpy()
                b = ck['state_dict']['fc.bias'].float().numpy()
                x = (z - np.asarray(ck['mean']).ravel()) / np.asarray(ck['std']).ravel()
                ref = x @ W.T + b
                out.setdefault('probe_max_abs_diff', []).append(float(np.abs(ref - pl[0].astype(np.float32)).max()))
                thr = np.array([r.meta['probe_thresholds'][t] for t in r.meta['probe_tags']], np.float32)
                sig = lambda v: 1 / (1 + np.exp(-v.astype(np.float64)))            # noqa: E731
                flips = int(((sig(ref) >= thr) != (sig(pl[0].astype(np.float32)) >= thr)).sum())
                out.setdefault('probe_tag_flips_at_thresholds', []).append(flips)
        out['manual_cos'] = cos
        assert min(cos) > 0.99999, cos
    return out


def test_e2e_cpu_build():
    side = os.environ.get('SEM_TEST_SIDECAR')
    if not side and os.environ.get('SEM_TEST_E2E') != '1':
        print('SKIP teacher e2e (set SEM_TEST_E2E=1 to build, or SEM_TEST_SIDECAR=<dir> to check one)')
        return
    if not side:
        side = os.path.join(OUT, 'sidecar_e2e_test')
        common = ['--train_datasets', FILES['DV'], '--val_datasets', FILES['B3V'], '--out', side, '--limit', '6',
                  '--write_texts']
        env = dict(os.environ, CUDA_VISIBLE_DEVICES='', PYTHONPATH=REPO)
        for extra in (['--device', 'cpu', '--dtype', 'fp32', '--batch_tokens', '4096'], ['--merge']):
            r = subprocess.run([PY, BUILDER] + common + extra, env=env, capture_output=True, text=True)
            assert r.returncode == 0, r.stderr[-3000:]
    res = verify_sidecar(side)
    print('e2e sidecar check:', json.dumps(res))


# ============================================================================================

def main():
    import traceback
    tests = [(k, v) for k, v in sorted(globals().items(), key=lambda kv: kv[1].__code__.co_firstlineno
                                         if callable(kv[1]) and hasattr(kv[1], '__code__') else 0)
             if k.startswith('test_') and callable(v)]
    fails = 0
    for name, fn in tests:
        try:
            fn()
            print(f'PASS {name}')
        except Exception:
            fails += 1
            print(f'FAIL {name}')
            traceback.print_exc()
    print(f'{len(tests) - fails}/{len(tests)} passed')
    return 1 if fails else 0


if __name__ == '__main__':
    os.makedirs(OUT, exist_ok=True)
    sys.exit(main())
