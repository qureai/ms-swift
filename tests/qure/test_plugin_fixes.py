#!/usr/bin/env python
"""CPU tests for SPEC section F -- pre-existing bug fixes F1-F4 (each with its escape hatch).

    export PYTHONPATH=<repo> CUDA_VISIBLE_DEVICES=""
    /home/ubuntu/miniconda3/envs/ms-swift-may-s4/bin/python tests/qure/test_plugin_fixes.py
        [-k SUBSTR]      only tests whose name contains SUBSTR
        [--slow]         also the slow ones (== QURE_SLOW=1): two full ckpt-608 CPU loads (F3c)
                         and the real `swift` CLI entry points (F2)

Standalone (exit code 1 on any failure) and pytest-compatible (test_* functions, plain asserts;
slow tests return early unless QURE_SLOW=1). Every plugin-loading case runs in a FRESH child
interpreter, because plugin imports and class patches are process-global -- the same reason the
bugs existed.

  F1  plugin double import: the stage-6c plugin list loaded exactly as swift does
      (swift.utils.import_external_file, list order) -> non-bbox row augmented exactly once,
      bbox row photometric only, each '[...] patched' log line once, sys.modules aliases are one
      object; both import orders; mldecoder_infer + joint; QURE_PLUGIN_DEDUP=0 reproduces the
      legacy double import.
  F2  qure_plugin_mode detection from argv[0] (swift/cli/<verb>.py), overrides, the export
      --to_cached_dataset exception; inference plugins still install; inference template OUTPUT
      identical between the legacy (train) and fixed (infer) mode; real `swift infer/deploy/sft`.
  F3  untie_tied_config / qure_untie_config_plugin on toy models; untie_lm_head.py on synthetic
      checkpoints (dry-run, --apply, idempotency, --all-under, refusals); slow: ckpt-608 through
      swift get_model_processor with the tied vs an untied config -> bit-identical tensors.
  F4  export_lm_only.py on a tiny synthetic checkpoint (drops, index, copied/skipped files,
      untie, SEM chat template, manifest, refusals).
Bulky artefacts go to $QURE_TEST_OUT (default: the session scratchpad tests_out, else a tmpdir).
"""
import glob
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import textwrap

REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
PY = sys.executable
SWIFT_BIN = os.path.join(os.path.dirname(PY), 'swift')
CKPT = os.environ.get('QURE_TEST_CKPT', '/fastdata4/sahil_work/VLM_qure/data_preparation/training_data/'
                      'output/Qwen3.5-2B_stage6b_alltools/v0-20260906-173237/checkpoint-608')
_DEFAULT_OUT = ('/tmp/claude-1000/-fastdata4-sahil-work-VLM-qure-vlm-swift/'
                'b2c1bd9c-54da-4ec4-b14d-2d8473057a20/scratchpad/tests_out')
OUT = os.environ.get('QURE_TEST_OUT') or (_DEFAULT_OUT if os.path.isdir(_DEFAULT_OUT)
                                          else tempfile.mkdtemp(prefix='qure_f_tests_'))
SLOW = os.environ.get('QURE_SLOW', '0') == '1'
PLUGINS_6C = ['mldecoder_joint_plugin', 'qure_augment_plugin', 'qure_ce_weight_plugin',
              'qure_stage3_plugin']
DEPLOY = os.path.join(REPO, 'examples', 'deploy')


# ============================================================================ child helpers
def run_child(code, env=None, timeout=1200, argv_extra=None):
    """Run `code` in a fresh interpreter; returns (stdout, stderr). Fails on non-zero exit."""
    e = dict(os.environ)
    for k in ('QURE_PLUGIN_MODE', 'QURE_PLUGIN_DEDUP', 'QURE_UNTIE_CONFIG'):
        e.pop(k, None)
    e['PYTHONPATH'] = REPO + (os.pathsep + e['PYTHONPATH'] if e.get('PYTHONPATH') else '')
    e['CUDA_VISIBLE_DEVICES'] = ''
    e.update(env or {})
    p = subprocess.run([PY, '-c', textwrap.dedent(code)] + (argv_extra or []), env=e,
                       capture_output=True, text=True, timeout=timeout, cwd=OUT)
    if p.returncode != 0:
        raise AssertionError(f'child failed rc={p.returncode}\n--- stdout\n{p.stdout[-4000:]}\n'
                             f'--- stderr\n{p.stderr[-6000:]}')
    return p.stdout, p.stderr


def result_of(stdout):
    lines = [l for l in stdout.splitlines() if l.startswith('RESULT ')]
    assert lines, f'no RESULT line in child stdout:\n{stdout[-3000:]}'
    return json.loads(lines[-1][len('RESULT '):])


def count(text, needle):
    return sum(needle in line for line in text.splitlines())


# ============================================================================ F1
_AUG_PROBE = r'''
import importlib, json, os, sys
R = {repo!r}
sys.argv = [f'{{R}}/swift/cli/sft.py']
from swift.utils import import_external_file
for f in {pre_files!r}:                # foreign plugin files loaded first (conflict case)
    import_external_file(f)
for m in {pre!r}:                      # imported BEFORE the plugin list (reverse-order case)
    importlib.import_module(m)
for p in {plugins!r}:                  # exactly how BaseArguments._import_external_plugins does it
    import_external_file(f'{{R}}/swift_plugins/{{p}}.py')
from swift.model.register import get_processor
from swift.template.register import get_template
from PIL import Image
import numpy as np
proc = get_processor({ckpt!r}, use_hf=True)
tpl = get_template(proc, default_system='You are a helpful medical assistant.', loss_scale='ignore_empty_think')
tpl.set_mode('train')
copies = []
for n in ('qure_augment_plugin', 'swift_plugins.qure_augment_plugin'):
    m = sys.modules.get(n)
    if m is not None and all(m is not c for c in copies):
        copies.append(m)
cnt = {{'geo': 0, 'photo': 0}}
class Geo:                               # stands in for RandAugment (geometric + intensity)
    def __call__(self, arr):
        cnt['geo'] += 1; return arr
class Photo:                             # stands in for the photometric-only Compose
    def __call__(self, image):
        cnt['photo'] += 1; return {{'image': image}}
if {stand_ins!r}:
    for m in copies:
        m._AUG = Geo(); m._PHOTO = Photo()
def img():
    return Image.fromarray((np.random.RandomState(0).rand(256, 256) * 255).astype('uint8')).convert('RGB')
rows = {{
 'nonbbox': {{'messages': [{{'role': 'user', 'content': '<image>Describe.'}}, {{'role': 'assistant', 'content': 'Normal chest.'}}], 'images': [img()]}},
 'bbox':    {{'messages': [{{'role': 'user', 'content': '<image>Locate.'}}, {{'role': 'assistant', 'content': "[{{'bbox_2d': [1,2,3,4], 'label': 'x'}}]"}}], 'images': [img()]}},
}}
res = {{'n_copies': len(copies)}}
for name, row in rows.items():
    cnt.update(geo=0, photo=0)
    enc = tpl.encode(row)
    assert len(enc['input_ids']) > 0
    res[name] = dict(cnt)
names = {plugins!r} + ['qure_plugin_mode']
res['aliased'] = {{n: (sys.modules.get(n) is not None and sys.modules.get(n) is sys.modules.get('swift_plugins.' + n)) for n in names}}
print('RESULT ' + json.dumps(res), flush=True)
'''

_PATCH_LINES = [
    '[mldecoder] Qwen3_5Template patched',
    '[mldecoder] step probe callback registered',
    '[mldecoder] hooked TunerMixin.prepare_model',
    '[mldecoder] create_optimizer patched',
    '[qure-aug] Qwen3_5Template._preprocess_inputs patched',
    '[qure-ce] liger fixed_fused_linear_cross_entropy wrapped',
    '[stage3] prepare_model wrapped',
    '[stage3] bbox-safe augmentation guard installed',
]


def _aug_probe(pre=(), dedup=None, stand_ins=True, plugins=PLUGINS_6C, pre_files=()):
    code = _AUG_PROBE.format(repo=REPO, pre=list(pre), plugins=list(plugins), ckpt=CKPT,
                             stand_ins=stand_ins, pre_files=list(pre_files))
    env = {} if dedup is None else {'QURE_PLUGIN_DEDUP': dedup}
    out, err = run_child(code, env=env)
    return result_of(out), err


def test_f1_stage6c_list_augments_once():
    res, err = _aug_probe()
    print('   ', res)
    assert res['n_copies'] == 1, res
    assert res['nonbbox'] == {'geo': 1, 'photo': 0}, res          # exactly once
    assert res['bbox'] == {'geo': 0, 'photo': 1}, res              # photometric only
    assert all(res['aliased'].values()), res['aliased']
    for line in _PATCH_LINES:
        assert count(err, line) == 1, (line, count(err, line))


def test_f1_reverse_import_order():
    # `import swift_plugins.X` BEFORE swift loads the list top-level: still one copy each
    res, err = _aug_probe(pre=['swift_plugins.qure_augment_plugin', 'swift_plugins.mldecoder_joint_plugin'])
    print('   ', res)
    assert res['n_copies'] == 1 and res['nonbbox'] == {'geo': 1, 'photo': 0}, res
    assert res['bbox'] == {'geo': 0, 'photo': 1}, res
    assert all(res['aliased'].values()), res['aliased']
    for line in _PATCH_LINES:
        assert count(err, line) == 1, (line, count(err, line))


def test_f1_foreign_earlier_copy_supersedes():
    # an OLD-style copy (no dedup, no once) of qure_augment_plugin from another dir is loaded
    # first, then `import swift_plugins.qure_augment_plugin` runs the repo copy: dedup keeps the
    # earlier copy for both names and once() refuses every patch of the superseded repo copy.
    d = os.path.join(OUT, 'f1_foreign')
    os.makedirs(d, exist_ok=True)
    src = open(os.path.join(REPO, 'swift_plugins', 'qure_augment_plugin.py')).read()
    old = src.replace("_dedup_module(__name__, 'qure_augment_plugin')", 'pass').replace(
        "_once('qure_augment_plugin.patch_template')", 'True')
    assert old != src
    with open(os.path.join(d, 'qure_augment_plugin.py'), 'w') as f:
        f.write(old)
    res, err = _aug_probe(pre_files=[os.path.join(d, 'qure_augment_plugin.py')],
                          pre=['swift_plugins.qure_augment_plugin'])
    print('   ', res)
    assert res['n_copies'] == 1, res
    assert res['nonbbox'] == {'geo': 1, 'photo': 0} and res['bbox'] == {'geo': 0, 'photo': 1}, res
    assert count(err, '[qure-aug] Qwen3_5Template._preprocess_inputs patched') == 1
    assert count(err, 'a different copy is already loaded') == 1
    assert count(err, 'qure_augment_plugin.patch_template refused') == 1


def test_f1_legacy_escape_hatch_reproduces_double():
    res, err = _aug_probe(dedup='0')
    print('   ', res)
    assert res['n_copies'] == 2, res
    assert res['nonbbox'] == {'geo': 2, 'photo': 0}, res           # the legacy double augment
    assert res['bbox'] == {'geo': 1, 'photo': 1}, res              # legacy: geometry leaked
    assert count(err, '[qure-aug] Qwen3_5Template._preprocess_inputs patched') == 2
    assert not res['aliased']['qure_augment_plugin']


def test_f1_real_pipelines_smoke():
    # real RandAugment + photometric Compose (albumentations 1.x) still run end-to-end
    res, err = _aug_probe(stand_ins=False)
    assert res['n_copies'] == 1, res
    assert '[qure-aug] RandAugment ready' in err and 'photometric-only pipeline ready' in err
    assert count(err, 'augmentation failed') == 0, err[-3000:]


_MLD_PROBE = r'''
import json, sys
R = {repo!r}
sys.argv = [f'{{R}}/swift/cli/{verb}.py']
from swift.utils import import_external_file
for p in {plugins!r}:
    import_external_file(f'{{R}}/swift_plugins/{{p}}.py')
j = sys.modules.get('mldecoder_joint_plugin'); js = sys.modules.get('swift_plugins.mldecoder_joint_plugin')
import swift_plugins.mldecoder_infer_plugin as inf
print('RESULT ' + json.dumps({{'joint_one_object': j is not None and j is js,
    'head_cls_from_joint': inf.VisionMLDecoderHead is (j or js).VisionMLDecoderHead,
    'infer_one_object': sys.modules.get('mldecoder_infer_plugin') is sys.modules.get('swift_plugins.mldecoder_infer_plugin')}}))
'''


def test_f1_mldecoder_infer_loads_joint_once():
    for plugins in (['mldecoder_joint_plugin', 'mldecoder_infer_plugin'], ['mldecoder_infer_plugin']):
        out, err = run_child(_MLD_PROBE.format(repo=REPO, plugins=plugins, verb='sft'))
        res = result_of(out)
        print('   ', plugins, res)
        assert all(res.values()), res
        assert count(err, '[mldecoder] Qwen3_5Template patched') == 1, plugins
        assert count(err, '[mldecoder-infer] TransformersEngine.__init__ patched') == 1
    # legacy: joint + infer -> the joint plugin executes twice
    out, err = run_child(_MLD_PROBE.format(repo=REPO, plugins=['mldecoder_joint_plugin', 'mldecoder_infer_plugin'],
                                           verb='sft'), env={'QURE_PLUGIN_DEDUP': '0'})
    assert count(err, '[mldecoder] Qwen3_5Template patched') == 2
    assert not result_of(out)['joint_one_object']


def test_f1_plugin_mode_module_two_names_and_once():
    out, _ = run_child(rf'''
        import json, sys
        sys.path.insert(0, {os.path.join(REPO, 'swift_plugins')!r})
        import qure_plugin_mode as a                     # top-level name first
        import swift_plugins.qure_plugin_mode as b       # then the package name
        r = {{'same': a is b, 'first': a.once('k1'), 'again_other_name': b.once('k1'),
              'other_key': b.once('k2')}}
        print('RESULT ' + json.dumps(r))
    ''')
    assert result_of(out) == {'same': True, 'first': True, 'again_other_name': False, 'other_key': True}
    out, _ = run_child(r'''
        import json
        import swift_plugins.qure_plugin_mode as a
        print('RESULT ' + json.dumps([a.once('k'), a.once('k')]))
    ''', env={'QURE_PLUGIN_DEDUP': '0'})
    assert result_of(out) == [True, True]                 # legacy: never refuses


def test_f1_every_plugin_dedups_first():
    """Static check: the dedup call is the first statement after the docstring in every plugin."""
    import ast
    files = sorted(glob.glob(os.path.join(REPO, 'swift_plugins', '*_plugin.py'))
                   + [os.path.join(REPO, 'swift_plugins', 'qure_graded_loss_scale.py')])
    assert len(files) >= 11, files
    for f in files:
        src = open(f).read()
        body = ast.parse(src).body
        i = 1 if (body and isinstance(body[0], ast.Expr) and isinstance(getattr(body[0], 'value', None), ast.Constant)) else 0
        stmts = body[i:i + 5]
        names = [ast.unparse(s) for s in stmts]
        dedup_at = [k for k, s in enumerate(names) if '_dedup_module(__name__' in s or ('if _AS_PLUGIN' in s and 'dedup' in s)]
        assert dedup_at, (os.path.basename(f), names)
        before = names[:dedup_at[0]]
        assert all(s.startswith(('try:', 'import sys', '_AS_PLUGIN')) for s in before), (os.path.basename(f), before)
        canon = os.path.basename(f)[:-3]
        assert f"'{canon}'" in names[dedup_at[0]], (f, names[dedup_at[0]])


_FAILED_IMPORT_PROBE = r"""
import json, os, sys
R = {repo!r}
{setup}
sys.argv = [f'{{R}}/swift/cli/sft.py']
from swift.utils import import_external_file
res = {{'loaded': [], 'error': None}}
try:
    for p in {plugins!r}:
        import_external_file(f'{{R}}/swift_plugins/{{p}}.py')
        res['loaded'].append(p)
except Exception as e:
    res['error'] = f'{{type(e).__name__}}: {{e}}'
res['left'] = sorted(n for n in {watch!r} if n in sys.modules)
print('RESULT ' + json.dumps(res))
"""


def test_f1_failed_import_stays_loud():
    """review BC-1 / F1-1: a plugin whose body RAISES during import (qure_augment's load-time albumentations
    check; the joint plugin without the foundation repo) must fail the load exactly as before F1 -- the dedup
    alias it registered first must not be handed to a retrying importer as a half-initialised module."""
    fake = os.path.join(OUT, 'f1_fake_alb2')
    os.makedirs(os.path.join(fake, 'albumentations'), exist_ok=True)
    with open(os.path.join(fake, 'albumentations', '__init__.py'), 'w') as f:
        f.write("__version__ = '2.0.0'\n")
    aug = ['qure_augment_plugin', 'swift_plugins.qure_augment_plugin']
    stage3 = ['mldecoder_joint_plugin', 'qure_ce_weight_plugin', 'qure_stage3_plugin']   # augment NOT listed
    cases = [('albumentations missing', "sys.modules['albumentations'] = None", {}, stage3, aug,
              'qure_augment_plugin needs albumentations'),
             ('albumentations 2.x', '', {'PYTHONPATH': fake + os.pathsep + REPO}, stage3, aug,
              'albumentations 2.0.0 is incompatible'),
             ('joint plugin cannot import MLDecoder', '', {'FM_ACCELERATE_REPO': os.path.join(OUT, 'no_such_fm_repo')},
              ['mldecoder_infer_plugin'], ['mldecoder_joint_plugin', 'swift_plugins.mldecoder_joint_plugin'],
              'Could not import MLDecoder')]
    for label, setup, env, plugins, watch, needle in cases:
        for dedup in (None, '0'):
            e = dict(env, **({'QURE_PLUGIN_DEDUP': dedup} if dedup else {}))
            out, _ = run_child(_FAILED_IMPORT_PROBE.format(repo=REPO, setup=setup, plugins=plugins, watch=watch), env=e)
            res = result_of(out)
            print('   ', label, 'dedup=' + str(dedup or 1), res)
            assert res['loaded'] == plugins[:-1] and res['error'] and needle in res['error'], (label, dedup, res)
            assert res['left'] == [], (label, dedup, res)              # no half-initialised alias survives


def test_f1_stale_alias_primitives():
    """_is_stale / drop_failed_import / dedup_module never keep a module a failed import left behind."""
    out, _ = run_child(r"""
        import importlib.machinery, json, sys, types
        import swift_plugins.qure_plugin_mode as m
        def mod(name):
            x = types.ModuleType(name)
            x.__spec__ = importlib.machinery.ModuleSpec(name, None)
            return x
        half = mod('swift_plugins.zz_probe')          # its body raised: importlib popped its own name
        sys.modules['zz_probe'] = half                  # ... but the F1 alias is still there
        r = {'stale': m._is_stale(half)}
        fresh = mod('swift_plugins.zz_probe')          # a new execution under the package name
        sys.modules['swift_plugins.zz_probe'] = fresh
        m.dedup_module('swift_plugins.zz_probe', 'zz_probe')
        r['replaced'] = sys.modules['zz_probe'] is fresh and sys.modules['swift_plugins.zz_probe'] is fresh
        r['healthy_not_stale'] = not m._is_stale(fresh)
        del sys.modules['swift_plugins.zz_probe']       # as importlib does when that body raises too
        r['dropped'] = m.drop_failed_import('zz_probe')
        r['gone'] = 'zz_probe' not in sys.modules and 'swift_plugins.zz_probe' not in sys.modules
        r['keeps_healthy'] = m.drop_failed_import('qure_plugin_mode') == [] and 'qure_plugin_mode' in sys.modules
        print('RESULT ' + json.dumps(r))
    """)
    assert result_of(out) == {'stale': True, 'replaced': True, 'healthy_not_stale': True, 'dropped': ['zz_probe'],
                              'gone': True, 'keeps_healthy': True}, out


# ============================================================================ F2
def test_f2_detect_matrix():
    cli = os.path.join(REPO, 'swift', 'cli')
    cases = [
        ([f'{cli}/infer.py', '--model', 'x'], None, 'infer'),
        ([f'{cli}/deploy.py'], None, 'infer'), ([f'{cli}/eval.py'], None, 'infer'),
        ([f'{cli}/export.py', '--model', 'x'], None, 'infer'), ([f'{cli}/app.py'], None, 'infer'),
        ([f'{cli}/web_ui.py'], None, 'infer'), ([f'{cli}/rollout.py'], None, 'infer'),
        ([f'{cli}/merge_lora.py'], None, 'infer'),
        ([f'{cli}/sft.py'], None, 'train'), ([f'{cli}/pt.py'], None, 'train'),
        ([f'{cli}/rlhf.py'], None, 'train'), ([f'{cli}/sample.py'], None, 'train'),
        ([f'{cli}/_megatron/sft.py'], None, 'train'), ([f'{cli}/_megatron/export.py'], None, 'infer'),
        # export that ENCODES training data stays train (split_*_mldecoder.sh bakes cls_labels)
        ([f'{cli}/export.py', '--to_cached_dataset', 'true', '--dataset', 'd'], None, 'train'),
        ([f'{cli}/export.py', '--to_cached_dataset=True'], None, 'train'),
        ([f'{cli}/export.py', '--to_cached_dataset', '--dataset', 'd'], None, 'train'),
        ([f'{cli}/export.py', '--to_cached_dataset', 'false'], None, 'infer'),
        # not under swift/cli -> argv[0] says nothing; legacy argv[1:] scan / default
        (['/some/where/infer.py'], None, 'train'),
        (['harness.py', 'infer'], None, 'infer'), (['harness.py', '--sft'], None, 'train'),
        (['-c'], None, 'train'),
        # overrides
        ([f'{cli}/infer.py'], 'train', 'train'), ([f'{cli}/sft.py'], 'infer', 'infer'),
        ([f'{cli}/deploy.py'], 'bogus', 'infer'),
    ]
    out, _ = run_child(rf'''
        import json, os, sys
        import swift_plugins.qure_plugin_mode as m
        res = []
        for argv, env, want in {cases!r}:
            sys.argv = argv
            if env is None: os.environ.pop('QURE_PLUGIN_MODE', None)
            else: os.environ['QURE_PLUGIN_MODE'] = env
            mode, src = m._detect()
            res.append([argv, env, want, mode, src])
        print('RESULT ' + json.dumps(res))
    ''')
    bad = [r for r in result_of(out) if r[2] != r[3]]
    assert not bad, bad



def test_f2_export_flag_spellings_match_swift_parser():
    """review F2-1: `export --to_cached_dataset` is recognised however swift's own parser accepts it -- the dash
    alias, unambiguous abbreviations, `=value`, last-wins, RAY_SWIFT_ARGS -- pinned against HfArgumentParser."""
    out, _ = run_child(r"""
        import contextlib, io, json, os, sys
        from transformers import HfArgumentParser
        from swift.arguments import ExportArguments
        from swift.utils.utils import _patch_get_type_hints
        import swift_plugins.qure_plugin_mode as m
        with _patch_get_type_hints():
            p = HfArgumentParser([ExportArguments])
        def swift_says(argv):
            try:
                with contextlib.redirect_stderr(io.StringIO()):
                    ns, _ = p.parse_known_args(argv)
                return bool(ns.to_cached_dataset)
            except SystemExit:
                return None                                # swift refuses the command line itself
        cases = []
        for base in ('to_cached_dataset', 'to-cached-dataset'):
            for n in range(1, len(base) + 1):
                cases += [['--' + base[:n], 'true'], ['--' + base[:n] + '=True'], ['--' + base[:n], 'false']]
        cases += [['--to_cached_dataset'], ['--to_cached_dataset', '--dataset', 'd'], ['--to-cached-dataset', 'no'],
                  ['--to_cached_dataset', 'true', '--to_cached_dataset', 'false'],
                  ['--to_cached', 'false', '--to-cached-dataset', 'yes'], ['--model', 'x'], ['--to_hf', 'true']]
        res = []
        for a in cases:
            sys.argv = ['/x/swift/cli/export.py'] + a
            res.append([a, swift_says(a), m._detect()[0]])
        os.environ['RAY_SWIFT_ARGS'] = json.dumps(['--to-cached-dataset', 'true'])
        sys.argv = ['/x/swift/cli/export.py']
        ray = m._detect()[0]
        print('RESULT ' + json.dumps({'cases': res, 'ray': ray}))
    """)
    r = result_of(out)
    bad = [c for c in r['cases'] if c[1] is not None and (c[2] == 'train') != c[1]]
    assert not bad, bad
    short = [c for c in r['cases'] if c[0][0].lstrip('-') in ('t', 'to', 'to_', 'to-')]
    assert short and all(c[1] is None for c in short), short          # ambiguous: swift itself errors out
    for a in (['--to-cached-dataset', 'true'], ['--to_cached', 'true'], ['--to-c=True']):
        assert [c for c in r['cases'] if c[0] == a][0][1:] == [True, 'train'], a
    assert r['ray'] == 'train'


_TPL_PROBE = r'''
import hashlib, json, sys
R = {repo!r}
sys.argv = [f'{{R}}/swift/cli/infer.py', '--model', {ckpt!r}]
from swift.utils import import_external_file
for p in {plugins!r}:
    import_external_file(f'{{R}}/swift_plugins/{{p}}.py')
import swift_plugins.qure_plugin_mode as qpm
from swift.model.register import get_processor
from swift.template.register import get_template
from swift.infer_engine.protocol import InferRequest
proc = get_processor({ckpt!r}, use_hf=True)
tpl = get_template(proc, default_system='You are a helpful medical assistant.', enable_thinking=False)
img = {img!r}
reqs = [
    {{'messages': [{{'role': 'user', 'content': '<image>Generate a medical report.'}}], 'images': [img]}},
    InferRequest(messages=[{{'role': 'user', 'content': '<image>Describe every finding.'}}], images=[img]),
    {{'messages': [{{'role': 'user', 'content': '<image>Locate.'}}, {{'role': 'assistant', 'content': "[{{'bbox_2d': [1,2,3,4]}}]"}}, {{'role': 'user', 'content': 'Again?'}}], 'images': [img]}},
]
out = {{'mode': qpm.MODE}}
for mode in ('transformers', 'sglang'):
    tpl.set_mode(mode)
    for i, r in enumerate(reqs):
        enc = tpl.encode(r)
        d = {{}}
        for k in ('input_ids', 'labels', 'image_grid_thw', 'pixel_values'):
            v = enc.get(k)
            if v is None: d[k] = None; continue
            if hasattr(v, 'numpy'): v = v.float().numpy() if v.is_floating_point() else v.numpy()
            import numpy as np
            a = np.asarray(v)
            d[k] = hashlib.sha1(a.tobytes()).hexdigest() + str(a.shape)
        d['extra_keys'] = sorted(k for k in enc if k.startswith('cls_'))
        out[f'{{mode}}_{{i}}'] = d
print('RESULT ' + json.dumps(out))
'''


def _tiny_png():
    p = os.path.join(OUT, 'f2_tiny.png')
    if not os.path.exists(p):
        import numpy as np
        from PIL import Image
        Image.fromarray((np.random.RandomState(1).rand(96, 128) * 255).astype('uint8')).convert('RGB').save(p)
    return p


def test_f2_inference_template_output_unchanged():
    """Legacy CLI mode (train patches installed under `swift infer`) vs fixed (infer): the
    model-facing encode output of the pt and sglang paths is identical."""
    code = _TPL_PROBE.format(repo=REPO, ckpt=CKPT, plugins=PLUGINS_6C, img=_tiny_png())
    legacy = result_of(run_child(code, env={'QURE_PLUGIN_MODE': 'train'})[0])
    fixed = result_of(run_child(code)[0])
    assert legacy.pop('mode') == 'train' and fixed.pop('mode') == 'infer'
    for k in fixed:
        lk, fk = dict(legacy[k]), dict(fixed[k])
        # the legacy joint encode attaches per-image cls_* keys that its own _post_encode pops
        # before forward; they never reach the model -> compare only model-facing tensors
        lk.pop('extra_keys'), fk.pop('extra_keys')
        assert lk == fk, (k, lk, fk)
    assert all(not fixed[k]['extra_keys'] for k in fixed)
    print('    legacy extra keys:', {k: legacy[k]['extra_keys'] for k in legacy})


def test_f2_inference_plugins_install_train_patches_skip():
    out, err = run_child(rf'''
        import json, sys
        R = {REPO!r}
        sys.argv = [f'{{R}}/swift/cli/infer.py']
        from swift.template.base import Template
        from swift.template.templates.qwen import Qwen3_5Template
        from swift.pipelines.train.tuner import TunerMixin
        from swift.trainers.mixin import SwiftMixin
        from swift.infer_engine.transformers_engine import TransformersEngine
        from liger_kernel.transformers.model import loss_utils
        before = {{'enc': Qwen3_5Template.encode, '_enc': Qwen3_5Template._encode, 'pre': Qwen3_5Template._preprocess_inputs,
                  'coll': Qwen3_5Template._data_collator, 'prep': TunerMixin.__dict__['prepare_model'],
                  'opt': SwiftMixin.create_optimizer, 'init': SwiftMixin.__init__,
                  'liger': loss_utils.fixed_fused_linear_cross_entropy}}
        eng0, dec0, pgk0 = TransformersEngine.__init__, Qwen3_5Template.decode, Template.prepare_generate_kwargs
        from swift.utils import import_external_file
        for p in {PLUGINS_6C + ['mldecoder_infer_plugin', 'force_presence_plugin', 'record_presence_plugin', 'qure_untie_config_plugin']!r}:
            import_external_file(f'{{R}}/swift_plugins/{{p}}.py')
        after = {{'enc': Qwen3_5Template.encode, '_enc': Qwen3_5Template._encode, 'pre': Qwen3_5Template._preprocess_inputs,
                 'coll': Qwen3_5Template._data_collator, 'prep': TunerMixin.__dict__['prepare_model'],
                 'opt': SwiftMixin.create_optimizer, 'init': SwiftMixin.__init__,
                 'liger': loss_utils.fixed_fused_linear_cross_entropy}}
        import swift_plugins.qure_plugin_mode as qpm
        print('RESULT ' + json.dumps({{'mode': qpm.MODE,
            'train_patches_untouched': {{k: before[k] is after[k] for k in before}},
            'engine_patched': TransformersEngine.__init__ is not eng0,
            'decode_patched': Qwen3_5Template.decode is not dec0,
            'pgk_patched': Template.prepare_generate_kwargs is not pgk0}}))
    ''')
    res = result_of(out)
    print('   ', res)
    assert res['mode'] == 'infer'
    assert all(res['train_patches_untouched'].values()), res
    assert res['engine_patched'] and res['decode_patched'] and res['pgk_patched']
    assert count(err, '[force-presence] installed') == 1 and count(err, '[record-presence] installed') == 1
    assert count(err, '[mldecoder] inference mode') == 1


_ARGV_PROBE = '''import os, sys
import swift_plugins.qure_plugin_mode as m
print('PROBE_MODE', m.MODE, '|', m.MODE_SOURCE, '|', os.path.basename(sys.argv[0]), flush=True)
os._exit(0)
'''


def test_f2_real_cli_entry_points():
    if not SLOW:
        print('    (slow: set QURE_SLOW=1 / --slow)')
        return
    probe = os.path.join(OUT, 'f2_argv_probe_plugin.py')
    with open(probe, 'w') as f:
        f.write(_ARGV_PROBE)
    e = dict(os.environ, PYTHONPATH=REPO, CUDA_VISIBLE_DEVICES='')
    e.pop('QURE_PLUGIN_MODE', None)
    cases = [
        (['infer', '--model', CKPT, '--infer_backend', 'pt'], {}, 'infer'),
        (['deploy', '--model', CKPT, '--infer_backend', 'pt'], {}, 'infer'),
        (['sft', '--model', CKPT, '--dataset', 'AI-ModelScope/alpaca-gpt4-data-en#10'], {}, 'train'),
        (['infer', '--model', CKPT, '--infer_backend', 'pt'], {'QURE_PLUGIN_MODE': 'train'}, 'train'),
        (['export', '--model', CKPT], {}, 'infer'),
        (['export', '--model', CKPT, '--to_cached_dataset', 'true', '--dataset',
          'AI-ModelScope/alpaca-gpt4-data-en#10', '--output_dir', os.path.join(OUT, 'f2_cached_out')], {}, 'train'),
        # review F2-1: the dash alias swift's parser accepts too
        (['export', '--model', CKPT, '--to-cached-dataset', 'true', '--dataset',
          'AI-ModelScope/alpaca-gpt4-data-en#10', '--output_dir', os.path.join(OUT, 'f2_cached_out_dash')], {}, 'train'),
    ]
    for args, extra, want in cases:
        p = subprocess.run([SWIFT_BIN] + args + ['--external_plugins', probe], env=dict(e, **extra),
                           capture_output=True, text=True, timeout=900, cwd=OUT)
        lines = [l for l in (p.stdout + p.stderr).splitlines() if l.startswith('PROBE_MODE')]
        print('   ', args[0], extra, '->', lines, 'rc', p.returncode)
        assert lines, (args, p.stdout[-2000:], p.stderr[-3000:])
        assert lines[-1].split()[1] == want, (args, extra, lines)


# ============================================================================ F3
class _Cfg:
    def __init__(self, tie, text=True):
        self.tie_word_embeddings = tie
        if text:
            self.text_config = type('T', (), {})()
            self.text_config.tie_word_embeddings = tie


def _toy(tied, text=True):
    import torch.nn as nn

    class Toy(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed = nn.Embedding(10, 4)
            self.lm_head = nn.Linear(4, 10, bias=False)
            if tied:
                self.lm_head.weight = self.embed.weight
            self.config = _Cfg(True, text)

        def get_input_embeddings(self):
            return self.embed

        def get_output_embeddings(self):
            return self.lm_head
    return Toy()


def test_f3_untie_tied_config_function():
    sys.path.insert(0, REPO)
    os.environ.setdefault('CUDA_VISIBLE_DEVICES', '')
    from swift_plugins.qure_untie_config_plugin import untie_tied_config
    m = _toy(tied=False)
    assert untie_tied_config(m) is True
    assert m.config.tie_word_embeddings is False and m.config.text_config.tie_word_embeddings is False
    assert untie_tied_config(m) is False                           # idempotent
    t = _toy(tied=True)
    assert untie_tied_config(t) is False                           # genuinely tied: untouched
    assert t.config.tie_word_embeddings is True and t.config.text_config.tie_word_embeddings is True
    w = type('W', (), {})()                                        # DDP/DeepSpeed-style wrapper
    w.module = _toy(tied=False)
    assert untie_tied_config(w) is True and w.module.config.text_config.tie_word_embeddings is False
    n = _toy(tied=False, text=False)
    assert untie_tied_config(n) is True and n.config.tie_word_embeddings is False


def test_f3_untie_plugin_install_and_escape_hatch():
    code = rf'''
        import json, sys
        R = {REPO!r}
        sys.argv = [f'{{R}}/swift/cli/sft.py']
        sys.path.insert(0, f'{{R}}/tests/qure')
        from swift.pipelines.train.tuner import TunerMixin
        TunerMixin.prepare_model = classmethod(lambda cls, args, model, **kw: model)   # stub the real one
        base = TunerMixin.__dict__['prepare_model']
        from swift_plugins.qure_untie_config_plugin import untie_tied_config        # library import
        lib_installs = TunerMixin.__dict__['prepare_model'] is not base
        from swift.utils import import_external_file
        import_external_file(f'{{R}}/swift_plugins/qure_untie_config_plugin.py')
        import_external_file(f'{{R}}/swift_plugins/qure_untie_config_plugin.py')     # listed twice
        from test_plugin_fixes import _toy
        m = _toy(tied=False)
        m2 = object.__new__(TunerMixin).prepare_model(None, m)   # how sft.py calls it (self.prepare_model)
        print('RESULT ' + json.dumps({{'lib_installs': lib_installs,
            'installed': TunerMixin.__dict__['prepare_model'] is not base, 'same_model': m2 is m,
            'tie': [m.config.tie_word_embeddings, m.config.text_config.tie_word_embeddings]}}))
    '''
    out, err = run_child(code)
    res = result_of(out)
    assert res == {'lib_installs': False, 'installed': True, 'same_model': True, 'tie': [False, False]}, res
    assert count(err, '[qure-untie] prepare_model wrapped') == 1
    out, err = run_child(code, env={'QURE_UNTIE_CONFIG': '0'})
    res = result_of(out)
    assert res['installed'] is False and res['tie'] == [True, True], res


def _synthetic_ckpt(root, name, head='distinct', tie=True, extra_keys=True, index=True):
    """Tiny Qwen3.5-shaped checkpoint: 2-3 shards + index, real config/tokenizer files copied."""
    import torch
    from safetensors.torch import save_file
    d = os.path.join(root, name)
    if os.path.isdir(d):
        shutil.rmtree(d)
    os.makedirs(d)
    g = torch.Generator().manual_seed(0)
    emb = torch.randn(64, 8, generator=g).to(torch.bfloat16)
    s1 = {'model.language_model.embed_tokens.weight': emb,
          'model.visual.patch_embed.proj.weight': torch.randn(4, 3, generator=g).to(torch.bfloat16)}
    if head == 'distinct':
        h = emb.float().clone()
        h[5, 3] += 0.25
        s1['lm_head.weight'] = h.to(torch.bfloat16)
    elif head == 'identical':
        s1['lm_head.weight'] = emb.clone()
    s2 = {'model.language_model.layers.0.mlp.down_proj.weight': torch.randn(8, 8, generator=g).to(torch.bfloat16),
          'model.language_model.norm.weight': torch.ones(8, dtype=torch.bfloat16)}
    if extra_keys:
        s1.update({'ml_decoder_head.ml_decoder.decoder.query_embed.weight': torch.randn(3, 4, generator=g),
                   'asl_loss.pos_weights': torch.ones(3), 'asl_loss.neg_weights': torch.ones(3)})
        s2['sem_head.proj.0.weight'] = torch.randn(2, 8, generator=g).to(torch.bfloat16)
        s3 = {'ml_decoder_head.ml_decoder.decoder.embed_standart.weight': torch.randn(2, 2, generator=g)}
    shards = {'model-00001-of-00003.safetensors': s1, 'model-00002-of-00003.safetensors': s2}
    if extra_keys:
        shards['model-00003-of-00003.safetensors'] = s3
    wm, size, nparam = {}, 0, 0
    for fn, sd in shards.items():
        save_file(sd, os.path.join(d, fn), metadata={'format': 'pt'})
        for k, v in sd.items():
            wm[k] = fn
            size += v.numel() * v.element_size()
            nparam += v.numel()
    if index:
        with open(os.path.join(d, 'model.safetensors.index.json'), 'w') as f:
            json.dump({'metadata': {'total_parameters': nparam, 'total_size': size}, 'weight_map': wm}, f, indent=2)
    with open(os.path.join(CKPT, 'config.json')) as f:
        cfg = json.load(f)
    cfg['tie_word_embeddings'] = tie
    cfg['text_config']['tie_word_embeddings'] = tie
    with open(os.path.join(d, 'config.json'), 'w') as f:
        json.dump(cfg, f, indent=2)
    for fn in ('generation_config.json', 'chat_template.jinja', 'tokenizer.json', 'tokenizer_config.json',
               'preprocessor_config.json', 'processor_config.json', 'args.json'):
        shutil.copy2(os.path.join(CKPT, fn), os.path.join(d, fn))
    for fn in ('trainer_state.json', 'training_args.bin', 'rng_state_0.pth', 'scheduler.pt', 'latest',
               'zero_to_fp32.py'):
        with open(os.path.join(d, fn), 'w') as f:
            f.write('dummy')
    os.makedirs(os.path.join(d, 'global_step7'))
    with open(os.path.join(d, 'global_step7', 'mp_rank_00_model_states.pt'), 'w') as f:
        f.write('dummy')
    return d


def _sha(p):
    return hashlib.sha1(open(p, 'rb').read()).hexdigest()


def _run_tool(script, args):
    e = dict(os.environ, PYTHONPATH=REPO, CUDA_VISIBLE_DEVICES='')
    p = subprocess.run([PY, os.path.join(DEPLOY, script)] + args, env=e, capture_output=True,
                       text=True, timeout=900, cwd=OUT)
    return p


def test_f3_untie_lm_head_tool():
    root = os.path.join(OUT, 'f3_untie_ckpts')
    a = _synthetic_ckpt(root, 'run/checkpoint-10')
    b = _synthetic_ckpt(root, 'run/checkpoint-20', head='identical')
    c = _synthetic_ckpt(root, 'run/checkpoint-30', head=None)               # genuinely tied
    u = _synthetic_ckpt(root, 'run/checkpoint-40', tie=False)               # already untied
    sha0 = _sha(os.path.join(a, 'config.json'))
    # dry-run: report, never write
    p = _run_tool('untie_lm_head.py', [a, '--json', os.path.join(root, 'rep.json')])
    assert p.returncode == 0, p.stderr
    rep = json.load(open(os.path.join(root, 'rep.json')))[0]
    assert rep['status'] == 'untie-needed' and rep['tie'] == {'top': True, 'text_config': True}
    assert abs(rep['diff']['max_abs'] - 0.25) < 0.01 and rep['diff']['rows_differ'] == 1, rep['diff']
    assert _sha(os.path.join(a, 'config.json')) == sha0 and not os.path.exists(os.path.join(a, 'config.json.tied_bak'))
    # --apply: backup + both levels false, everything else unchanged
    before = json.load(open(os.path.join(a, 'config.json')))
    p = _run_tool('untie_lm_head.py', [a, '--apply'])
    assert p.returncode == 0 and 'rewritten' in p.stdout, p.stdout
    after = json.load(open(os.path.join(a, 'config.json')))
    assert after['tie_word_embeddings'] is False and after['text_config']['tie_word_embeddings'] is False
    before['tie_word_embeddings'] = False
    before['text_config']['tie_word_embeddings'] = False
    assert after == before
    assert _sha(os.path.join(a, 'config.json.tied_bak')) == sha0
    # idempotent: second --apply is a no-op and never touches the backup
    sha1 = _sha(os.path.join(a, 'config.json'))
    p = _run_tool('untie_lm_head.py', [a, '--apply'])
    assert 'no-op (already untied)' in p.stdout, p.stdout
    assert _sha(os.path.join(a, 'config.json')) == sha1 and _sha(os.path.join(a, 'config.json.tied_bak')) == sha0
    # --all-under: finds the 4 checkpoints (not global_step*), refuses the unsafe ones
    p = _run_tool('untie_lm_head.py', ['--all-under', root, '--apply', '--json', os.path.join(root, 'all.json')])
    assert p.returncode == 0, p.stderr
    reps = {os.path.basename(r['ckpt']): r for r in json.load(open(os.path.join(root, 'all.json')))}
    assert set(reps) == {'checkpoint-10', 'checkpoint-20', 'checkpoint-30', 'checkpoint-40'}, set(reps)
    assert reps['checkpoint-10']['status'] == 'already-untied'
    assert reps['checkpoint-20']['status'] == 'identical' and reps['checkpoint-20']['action'].startswith('skipped')
    assert reps['checkpoint-30']['status'] == 'tied-single-key' and reps['checkpoint-30']['action'].startswith('REFUSED')
    assert reps['checkpoint-40']['status'] == 'already-untied'
    for d in (b, c):
        cfg = json.load(open(os.path.join(d, 'config.json')))
        assert cfg['tie_word_embeddings'] is True and not os.path.exists(os.path.join(d, 'config.json.tied_bak'))
    p = _run_tool('untie_lm_head.py', [b, '--apply', '--even-if-identical'])
    assert json.load(open(os.path.join(b, 'config.json')))['text_config']['tie_word_embeddings'] is False


_LOAD_PROBE = r'''
import json, os, sys, torch
from swift.model.register import get_model_processor
from safetensors import safe_open
ck, tmp = {ck!r}, {tmp!r}
def load(d):
    model, _ = get_model_processor(d, torch_dtype=torch.bfloat16, device_map='cpu', load_model=True)
    h = model.get_output_embeddings().weight.detach().clone()
    e = model.get_input_embeddings().weight.detach().clone()
    info = {{'tied_obj': model.get_output_embeddings().weight is model.get_input_embeddings().weight,
            'cfg_top': model.config.tie_word_embeddings,
            'cfg_text': getattr(model.config, 'text_config', model.config).tie_word_embeddings}}
    del model
    return h, e, info
h1, e1, i1 = load(ck)
h2, e2, i2 = load(tmp)
idx = json.load(open(os.path.join(ck, 'model.safetensors.index.json')))['weight_map']
with safe_open(os.path.join(ck, idx['lm_head.weight']), framework='pt') as f:
    hs = f.get_tensor('lm_head.weight')
# resize_token_embeddings() ends in tie_weights() (transformers modeling_utils.py:2615), which
# recomputes the tie mapping from config.tie_word_embeddings: with the shipped (tied) config it
# RE-TIES and discards the trained lm_head; after untie_tied_config it must not.
sys.path.insert(0, {repo!r})
from swift_plugins.qure_untie_config_plugin import untie_tied_config
retie = {{}}
for fix in (False, True):
    model, _ = get_model_processor(ck, torch_dtype=torch.bfloat16, device_map='cpu', load_model=True)
    changed = untie_tied_config(model) if fix else None
    if fix:
        sv = tmp + '_cfgsave'
        model.config.save_pretrained(sv)
        saved = json.load(open(os.path.join(sv, 'config.json')))
        retie['saved_cfg_tie'] = [saved['tie_word_embeddings'], saved['text_config']['tie_word_embeddings']]
    model.tie_weights()
    w = model.get_output_embeddings().weight
    retie['fixed' if fix else 'shipped'] = {{'changed': changed, 'lm_head_is_embed': w is model.get_input_embeddings().weight,
                                            'lm_head_still_trained': bool(torch.equal(w.detach(), hs))}}
    del model
print('RESULT ' + json.dumps({{'tied_cfg': i1, 'untied_cfg': i2,
    'lm_head_equal': bool(torch.equal(h1, h2)), 'embed_equal': bool(torch.equal(e1, e2)),
    'lm_head_is_saved_lm_head': bool(torch.equal(h1, hs)), 'lm_head_ne_embed': not bool(torch.equal(h1, e1)),
    'retie': retie}}))
'''


def test_f3_ckpt608_untied_config_loads_identically():
    """F3(c): swift get_model_processor(ckpt-608) with the shipped (tied) config vs a temp dir with
    ONLY config.json untied (shards symlinked) -> bit-identical lm_head and embed_tokens."""
    if not SLOW:
        print('    (slow: set QURE_SLOW=1 / --slow)')
        return
    tmp = os.path.join(OUT, 'f3_ckpt608_untied')
    if os.path.isdir(tmp):
        shutil.rmtree(tmp)
    os.makedirs(tmp)
    for fn in os.listdir(CKPT):
        if fn != 'config.json' and os.path.isfile(os.path.join(CKPT, fn)):
            os.symlink(os.path.join(CKPT, fn), os.path.join(tmp, fn))
    cfg = json.load(open(os.path.join(CKPT, 'config.json')))
    cfg['tie_word_embeddings'] = False
    cfg['text_config']['tie_word_embeddings'] = False
    with open(os.path.join(tmp, 'config.json'), 'w') as f:
        json.dump(cfg, f, indent=2)
    out, err = run_child(_LOAD_PROBE.format(ck=CKPT, tmp=tmp, repo=REPO), timeout=3600)
    res = result_of(out)
    print('   ', res)
    assert res['tied_cfg']['tied_obj'] is False and res['untied_cfg']['tied_obj'] is False
    assert res['untied_cfg']['cfg_top'] is False and res['untied_cfg']['cfg_text'] is False
    assert res['lm_head_equal'] and res['embed_equal'] and res['lm_head_is_saved_lm_head'] and res['lm_head_ne_embed']
    assert 'but both are present in the checkpoints, so we will NOT tie them' in err   # HF refused tie (tied cfg)
    rt = res['retie']
    assert rt['shipped']['lm_head_is_embed'] and not rt['shipped']['lm_head_still_trained'], rt   # the latent bug
    assert rt['fixed']['changed'] is True and rt['saved_cfg_tie'] == [False, False], rt
    assert not rt['fixed']['lm_head_is_embed'] and rt['fixed']['lm_head_still_trained'], rt       # fixed



def test_f3_untie_plugin_listed_first():
    """review F3-1: the untie plugin replaces prepare_model with a plain function; every prepare_model wrapper in
    swift_plugins/ now uses the getattr-__func__ pattern, so listing it FIRST no longer crashes the others."""
    import re
    for f in glob.glob(os.path.join(REPO, 'swift_plugins', '*.py')):
        assert not re.search(r'=\s*TunerMixin\.prepare_model\.__func__', open(f).read()), f   # code, not prose
    out, err = run_child(rf"""
        import json, sys
        R = {REPO!r}
        sys.argv = [f'{{R}}/swift/cli/sft.py']
        from swift.utils import import_external_file
        done = []
        for p in ['qure_untie_config_plugin', 'mldecoder_joint_plugin', 'qure_augment_plugin', 'qure_ce_weight_plugin',
                  'qure_stage3_plugin', 'sem_latent_plugin']:
            import_external_file(f'{{R}}/swift_plugins/{{p}}.py')
            done.append(p)
        print('RESULT ' + json.dumps(done))
    """)
    assert len(result_of(out)) == 6
    for line in ('[qure-untie] prepare_model wrapped', '[mldecoder] hooked TunerMixin.prepare_model',
                 '[stage3] prepare_model wrapped', '[sem-latent] prepare_model wrapped'):
        assert count(err, line) == 1, (line, err[-3000:])


def test_f3_sem_untie_without_repo_root():
    """review F3-2: without the repo root on sys.path the SEM plugin still finds untie_tied_config (sibling file,
    private module name: the untie plugin's own patch is NOT installed) instead of skipping the untie."""
    out, _ = run_child(rf"""
        import json, os, sys
        R = {REPO!r}
        try:
            import swift_plugins  # noqa: F401
            importable = True
        except ImportError:
            importable = False
        sys.argv = [f'{{R}}/swift/cli/sft.py']
        from swift.utils import import_external_file
        P = import_external_file(f'{{R}}/swift_plugins/sem_latent_plugin.py')
        fn = P._untie_fn()
        import torch.nn as nn
        class Cfg: tie_word_embeddings = True
        class Toy(nn.Module):
            def __init__(self):
                super().__init__()
                self.e, self.h, self.config = nn.Embedding(5, 3), nn.Linear(3, 5, bias=False), Cfg()
            def get_input_embeddings(self): return self.e
            def get_output_embeddings(self): return self.h
        t = Toy()
        changed = fn(t)
        from swift.pipelines.train.tuner import TunerMixin
        print('RESULT ' + json.dumps({{'swift_plugins_importable': importable, 'changed': changed,
            'tie_after': t.config.tie_word_embeddings, 'plugin_loaded': 'qure_untie_config_plugin' in sys.modules,
            'fn_module': fn.__module__, 'prepare_model': TunerMixin.prepare_model.__qualname__}}))
    """, env={'PYTHONPATH': ''})
    r = result_of(out)
    assert r == {'swift_plugins_importable': False, 'changed': True, 'tie_after': False, 'plugin_loaded': False,
                 'fn_module': '_sem_qure_untie_config', 'prepare_model': 'patch_prepare_model.<locals>.prepare_model'}, r


# ============================================================================ F4
def _keys_of(d):
    from safetensors import safe_open
    out = {}
    for fn in glob.glob(os.path.join(d, '*.safetensors')):
        with safe_open(fn, framework='pt') as f:
            for k in f.keys():
                out[k] = (os.path.basename(fn), f.get_tensor(k))
    return out


def test_f4_export_lm_only():
    import torch
    root = os.path.join(OUT, 'f4_export')
    src = _synthetic_ckpt(root, 'checkpoint-7')
    out = os.path.join(root, 'checkpoint-7-lm')
    if os.path.isdir(out):
        shutil.rmtree(out)
    p = _run_tool('export_lm_only.py', [src, out])
    assert p.returncode == 0, p.stdout + p.stderr
    s, o = _keys_of(src), _keys_of(out)
    drop = ('ml_decoder_head.', 'asl_loss.', 'sem_head.')
    assert set(o) == {k for k in s if not k.startswith(drop)}, sorted(o)
    for k, (_, t) in o.items():
        assert torch.equal(t, s[k][1]), k
    idx = json.load(open(os.path.join(out, 'model.safetensors.index.json')))
    assert {k: v for k, v in idx['weight_map'].items()} == {k: fn for k, (fn, _) in o.items()}
    assert idx['metadata']['total_size'] == sum(t.numel() * t.element_size() for _, t in o.values())
    assert idx['metadata']['total_parameters'] == sum(t.numel() for _, t in o.values())
    assert not os.path.exists(os.path.join(out, 'model-00003-of-00003.safetensors'))   # all-dropped shard
    man = json.load(open(os.path.join(out, 'export_manifest.json')))
    assert man['dropped_keys'] == sorted(k for k in s if k.startswith(drop)) and man['n_dropped'] == 5
    assert man['shards']['model-00003-of-00003.safetensors']['action'] == 'omitted-empty'
    assert man['shards']['model-00001-of-00003.safetensors']['action'] == 'rewritten'
    files = set(os.listdir(out))
    for fn in ('config.json', 'generation_config.json', 'chat_template.jinja', 'tokenizer.json',
               'tokenizer_config.json', 'preprocessor_config.json', 'processor_config.json'):
        assert fn in files, fn
    for fn in ('args.json', 'trainer_state.json', 'training_args.bin', 'rng_state_0.pth', 'scheduler.pt',
               'latest', 'zero_to_fp32.py', 'global_step7'):
        assert fn not in files, fn
    cfg = json.load(open(os.path.join(out, 'config.json')))
    assert cfg['tie_word_embeddings'] is False and cfg['text_config']['tie_word_embeddings'] is False
    assert man['untie']['applied'] is True and man['untie']['source_status'] == 'untie-needed'
    assert json.load(open(os.path.join(src, 'config.json')))['tie_word_embeddings'] is True   # source untouched
    assert man['sem_template']['K'] == 0 and _sha(os.path.join(out, 'chat_template.jinja')) == _sha(os.path.join(src, 'chat_template.jinja'))
    # a shard with nothing dropped is copied byte-for-byte
    src2 = _synthetic_ckpt(root, 'checkpoint-8', extra_keys=False)
    out2 = os.path.join(root, 'checkpoint-8-lm')
    shutil.rmtree(out2, ignore_errors=True)
    p = _run_tool('export_lm_only.py', [src2, out2, '--no-untie', '--keep-args-json'])
    assert p.returncode == 0, p.stdout + p.stderr
    for fn in ('model-00001-of-00003.safetensors', 'model-00002-of-00003.safetensors'):
        assert _sha(os.path.join(out2, fn)) == _sha(os.path.join(src2, fn)), fn
    assert json.load(open(os.path.join(out2, 'config.json')))['tie_word_embeddings'] is True   # --no-untie
    assert 'args.json' in os.listdir(out2)
    # refusals: non-empty out, out inside the source
    assert _run_tool('export_lm_only.py', [src, out]).returncode != 0
    assert _run_tool('export_lm_only.py', [src, os.path.join(src, 'sub')]).returncode != 0



def test_f4_overwrite_removes_stale_outputs():
    """review F4-1: --overwrite removes what a previous export left and this one does not write (args.json from an
    earlier --keep-args-json run would make swift restore training args; stale shards); unknown files stay."""
    root = os.path.join(OUT, 'f4_overwrite')
    src = _synthetic_ckpt(root, 'checkpoint-7')
    out = os.path.join(root, 'checkpoint-7-lm')
    shutil.rmtree(out, ignore_errors=True)
    p = _run_tool('export_lm_only.py', [src, out, '--keep-args-json'])
    assert p.returncode == 0 and 'args.json' in os.listdir(out), p.stdout + p.stderr
    open(os.path.join(out, 'model-00009-of-00009.safetensors'), 'w').write('stale shard')
    open(os.path.join(out, 'optimizer.pt'), 'w').write('never belongs in an export')
    open(os.path.join(out, 'NOTES.txt'), 'w').write('mine')
    os.makedirs(os.path.join(out, 'my_subdir'), exist_ok=True)
    before = set(os.listdir(out))
    p = _run_tool('export_lm_only.py', [src, out, '--dry-run'])                   # plans, removes nothing
    assert p.returncode == 0 and set(os.listdir(out)) == before and 'remove stale' in p.stdout, p.stdout
    p = _run_tool('export_lm_only.py', [src, out, '--overwrite'])
    assert p.returncode == 0, p.stdout + p.stderr
    files = set(os.listdir(out))
    man = json.load(open(os.path.join(out, 'export_manifest.json')))
    assert 'args.json' not in files and 'args.json' not in man['copied_files'], sorted(files)
    assert 'model-00009-of-00009.safetensors' not in files and 'optimizer.pt' not in files, sorted(files)
    assert sorted(man['removed_stale_files']) == ['args.json', 'model-00009-of-00009.safetensors', 'optimizer.pt'], man
    assert 'NOTES.txt' in files and 'my_subdir' in files and man['left_unknown_files'] == ['NOTES.txt', 'my_subdir/']
    assert 'WARNING' in p.stderr and 'NOTES.txt' in p.stderr
    s_, o_ = _keys_of(src), _keys_of(out)                                         # the weights are this export's
    assert set(o_) == {k for k in s_ if not k.startswith(('ml_decoder_head.', 'asl_loss.', 'sem_head.'))}
    idx = json.load(open(os.path.join(out, 'model.safetensors.index.json')))
    assert set(idx['weight_map'].values()) <= files


def test_f4_export_sem_template():
    root = os.path.join(OUT, 'f4_export_sem')
    src = _synthetic_ckpt(root, 'checkpoint-9')
    code = rf'''
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained({src!r})
        n = tok.add_special_tokens({{'additional_special_tokens': [f'<|sem_{{i}}|>' for i in range(8)]}})   # as swift does
        tok.save_pretrained({os.path.join(root, 'tok_sem')!r})
        print('RESULT', n)
    '''
    run_child(code)
    for fn in ('tokenizer.json', 'tokenizer_config.json'):
        shutil.copy2(os.path.join(root, 'tok_sem', fn), os.path.join(src, fn))
    jsrc = open(os.path.join(src, 'chat_template.jinja')).read()
    sem_str = ''.join(f'<|sem_{i}|>' for i in range(8))
    for flag in ([], ['--no-sem-template']):
        out = os.path.join(root, 'checkpoint-9-lm' + ('-nosem' if flag else ''))
        shutil.rmtree(out, ignore_errors=True)
        p = _run_tool('export_lm_only.py', [src, out] + flag)
        assert p.returncode == 0, p.stdout + p.stderr
        man = json.load(open(os.path.join(out, 'export_manifest.json')))['sem_template']
        jout = open(os.path.join(out, 'chat_template.jinja')).read()
        print('   ', flag, man.get('status'), man.get('patched'))
        assert man['K'] == 8, man
        if flag:
            assert man['status'].startswith('skipped') and jout == jsrc
        elif man['status'] == 'installed':            # swift.template.sem_latent present
            assert sem_str in jout and jout != jsrc
            assert 'chat_template.jinja' in man['patched']
        else:                                         # module absent -> warned, unchanged
            assert man['status'].startswith('skipped') and jout == jsrc, man
            assert 'WARNING' in p.stderr
    # module ABSENT (simulated: a None entry in sys.modules makes the lazy import fail)
    out = os.path.join(root, 'checkpoint-9-lm-nomodule')
    shutil.rmtree(out, ignore_errors=True)
    _, err = run_child(rf'''
        import sys
        sys.modules['swift.template.sem_latent'] = None
        sys.path.insert(0, {DEPLOY!r})
        import export_lm_only
        assert export_lm_only.main([{src!r}, {out!r}]) == 0
    ''')
    man = json.load(open(os.path.join(out, 'export_manifest.json')))['sem_template']
    assert man['K'] == 8 and man['status'].startswith('skipped: swift.template.sem_latent unavailable'), man
    assert open(os.path.join(out, 'chat_template.jinja')).read() == jsrc and 'WARNING' in err



# ============================================================================ test hygiene (review BC-3)
def test_bc3_qure_test_modules_leave_env_alone_at_import():
    """Importing any tests/qure module (pytest collection, tests/run.py discovery) must not change the process env
    (CUDA visibility, image-token limits, plugin mode) for other tests collected in the same process."""
    here = os.path.dirname(os.path.abspath(__file__))
    mods = sorted(os.path.basename(f)[:-3] for f in glob.glob(os.path.join(here, 'test_*.py')))
    out, _ = run_child(rf"""
        import importlib, json, os, sys
        sys.path.insert(0, {here!r})
        os.environ['CUDA_VISIBLE_DEVICES'] = '99'
        os.environ.pop('IMAGE_MAX_TOKEN_NUM', None)
        before = dict(os.environ)
        changed = {{}}
        for m in {mods!r}:
            importlib.import_module(m)
            now = dict(os.environ)
            diff = {{k: [before.get(k), now.get(k)] for k in set(before) | set(now) if before.get(k) != now.get(k)}}
            if diff:
                changed[m] = diff
                os.environ.clear(); os.environ.update(before)
        print('RESULT ' + json.dumps(changed))
    """)
    assert result_of(out) == {}, result_of(out)


# ============================================================================ runner
def _main(argv):
    global SLOW
    if '--slow' in argv:
        SLOW = True
        os.environ['QURE_SLOW'] = '1'
    k = argv[argv.index('-k') + 1] if '-k' in argv else None
    tests = [(n, f) for n, f in sorted(globals().items(), key=lambda kv: getattr(kv[1], '__code__', None) and kv[1].__code__.co_firstlineno or 0)
             if n.startswith('test_') and callable(f) and (k is None or k in n)]
    print(f'[qure-tests] {len(tests)} test(s) · out={OUT} · slow={SLOW}', flush=True)
    failed = []
    import time
    for n, f in tests:
        t0 = time.time()
        try:
            f()
            print(f'PASS {n} ({time.time() - t0:.1f}s)', flush=True)
        except Exception as e:  # noqa: BLE001
            failed.append(n)
            print(f'FAIL {n} ({time.time() - t0:.1f}s): {type(e).__name__}: {str(e)[:6000]}', flush=True)
    print(f'[qure-tests] {len(tests) - len(failed)}/{len(tests)} passed' + (f'; FAILED: {failed}' if failed else ''))
    return 1 if failed else 0


if __name__ == '__main__':
    sys.exit(_main(sys.argv[1:]))
