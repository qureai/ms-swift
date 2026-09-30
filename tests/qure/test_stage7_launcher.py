#!/usr/bin/env python
"""Tests for examples/train/liger/20_stage7_sem_latent.sh (spec S5), CPU only, nothing is ever launched.

    CUDA_VISIBLE_DEVICES="" PYTHONPATH=<repo> <env python> tests/qure/test_stage7_launcher.py      (~2-3 min)
    ... -m pytest tests/qure/test_stage7_launcher.py

HOW NOTHING GETS LAUNCHED
Every run points CONDA_ENV at a scratch dir whose bin/ holds a symlink to the real env python and a FAKE
`swift` that only records its argv / env / oom_score_adj (the launcher puts $CONDA_ENV/bin first on PATH).
DRY_RUN=1 runs must not even reach the fake. Synthetic sidecars (16 random unit rows, written with the real
sem_targets.write_sidecar) cover the real default dataset files (MIX=mcr: data_preparation/training_data/stage7_mcr/,
built by build_stage7_mcr_data.py) by size+mtime, so the strict coverage check passes without a real sidecar.

WHAT IS CHECKED
  * bash -n; missing sidecar -> exit 1 + the exact run_build_sem_targets.sh command (OUT, DATASETS, VAL_DATASETS)
  * wrong rule_version / no teacher_fingerprint / --limit smoke build / uncovered or changed dataset file ->
    refused (strict); SEM_SIDECAR_CHECK=warn downgrades coverage + limit only; a complete-but-bad sidecar
    gets a rebuild command into a NEW dir
  * DRY_RUN on a valid sidecar: the python precheck passes (real ckpt-608 tokenizer + both token files through
    swift's get_processor, K=8, no resize, core hook live in train + sglang mode) and the printed command has
    every required flag (both token files in order, sem plugin LAST, packing/padding_free false, liger, zero2,
    flash-attn, non-thinking prefix, ignore_empty_think, run name, the 6+3 MIX=mcr default files), fake swift
    untouched
  * the real (non-dry) path runs `swift sft` with EXACTLY the dry-run argv and the documented env
    (ASL_WEIGHT=0.05 const, MLD_RESTORE_HEAD=1, MLD_FREEZE_HEAD=0, SEM_* defaults, CUDA_VISIBLE_DEVICES,
    oom_score_adj 1000); DETACH=1 does the same from a setsid'd child and returns at once
  * guards: SWIFT_SEM_LATENT=0, QURE_PLUGIN_MODE=infer, RESUME missing, root-disk floor, busy GPUs (DRY_RUN
    warns, real launch refuses), an invalid SEM_* knob (plugin's own SemConfig)
  * overrides: RESUME + IGNORE_DATA_SKIP, DATASETS with a '#N' suffix, EXTRA_ARGS, EXTRA_PLUGINS (sem stays
    last), GPUS -> NPROC_PER_NODE, /localstorage paths resolved to /fastdata4
  * MIX: default mcr (stage7_mcr files, sidecar default stage7_mcr_rule_v1, SKIP_BROKEN_IMAGES=1 exported);
    MIX=6c = the stage-6c mixture + stage7_rule_v1 (the real 6c sidecar when it is built) and no
    SKIP_BROKEN_IMAGES; an explicit SKIP_BROKEN_IMAGES / DATASETS wins; MIX=bogus refused
"""
import json
import os
import shlex
import shutil
import subprocess
import sys
import time

import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
sys.path.insert(0, REPO)
import swift_plugins.sem_targets as st  # noqa: E402

LAUNCHER = os.path.join(REPO, 'examples', 'train', 'liger', '20_stage7_sem_latent.sh')
ENV_PY = '/home/ubuntu/miniconda3/envs/ms-swift-may-s4/bin/python'
SCRATCH = os.environ.get('STAGE7_TEST_OUT') or ('/tmp/claude-1000/-fastdata4-sahil-work-VLM-qure-vlm-swift/'
                                                 'b2c1bd9c-54da-4ec4-b14d-2d8473057a20/scratchpad/tests_out/stage7_launcher')
DATA_DIR = '/fastdata4/sahil_work/VLM_qure/data_preparation/training_data'
CKPT = f'{DATA_DIR}/output/Qwen3.5-2B_stage6b_alltools/v0-20260906-173237/checkpoint-608'
MCR = f'{DATA_DIR}/stage7_mcr'   # MIX=mcr (default): examples/train/liger/sem_latent/build_stage7_mcr_data.py
TRAIN = [f'{MCR}/A_TRAIN_120k.mcr.jsonl', f'{MCR}/B3_TRAIN.mcr.jsonl', f'{MCR}/15_stage5_headguided_TRAIN.mcr.jsonl',
         f'{MCR}/17_refresher_multiturn_TRAIN.mcr.jsonl', f'{MCR}/RET_diverse60k_TRAIN.mcr.jsonl', f'{MCR}/REX_TRAIN.jsonl']
VAL = [f'{MCR}/A_VAL.mcr.jsonl', f'{MCR}/B3_VAL.mcr.jsonl', f'{MCR}/REX_VAL.jsonl']
SIDECAR_MCR = f'{DATA_DIR}/sem_targets/stage7_mcr_rule_v1'
TRAIN_6C = [f'{DATA_DIR}/stage6_toolcalling/A_TRAIN_120k.jsonl', f'{DATA_DIR}/stage6_toolcalling/B3_TRAIN.jsonl',
            f'{DATA_DIR}/15_stage5_headguided_TRAIN.json', f'{DATA_DIR}/17_refresher_multiturn_TRAIN.json',
            f'{DATA_DIR}/stage6_toolcalling/RET_diverse60k_TRAIN.jsonl']
VAL_6C = [f'{DATA_DIR}/stage6_toolcalling/A_VAL.jsonl', f'{DATA_DIR}/stage6_toolcalling/B3_VAL.jsonl']
SIDECAR_6C = f'{DATA_DIR}/sem_targets/stage7_rule_v1'   # the real stage-6c sidecar (built 2026-09-30)
TOKEN_TXT = os.path.join(REPO, 'examples', 'train', 'liger', 'token.txt')
TOKEN_SEM = os.path.join(REPO, 'examples', 'train', 'liger', 'token_sem.txt')
SEM_PLUGIN = os.path.join(REPO, 'swift_plugins', 'sem_latent_plugin.py')

FAKE_SWIFT = r'''#!/usr/bin/env bash
# FAKE swift for tests/qure/test_stage7_launcher.py: records, launches nothing
out="${FAKE_SWIFT_OUT:?}"
"$(dirname "$0")/python" -c 'import json,sys; json.dump(sys.argv[2:], open(sys.argv[1], "w"))' "$out.argv.json" "$@"
env | grep -E '^(SEM_|ASL_|AR_WEIGHT|MLD_|QURE_|CUDA_VISIBLE_DEVICES|NPROC_PER_NODE|PYTHONPATH|HF_HOME|TMPDIR|SKIP_BROKEN_IMAGES)' | sort > "$out.env"
cat /proc/$$/oom_score_adj > "$out.oom"
echo done > "$out.done"
'''


def _setup():
    os.makedirs(SCRATCH, exist_ok=True)
    env_dir = os.path.join(SCRATCH, 'fake_env')
    b = os.path.join(env_dir, 'bin')
    os.makedirs(b, exist_ok=True)
    py = os.path.join(b, 'python')
    if not os.path.islink(py):
        os.symlink(ENV_PY, py)
    with open(os.path.join(b, 'swift'), 'w') as f:
        f.write(FAKE_SWIFT)
    os.chmod(os.path.join(b, 'swift'), 0o755)
    return env_dir


FAKE_ENV = _setup()


def _dsrec(paths, role, bad_size=False):
    out = []
    for p in paths:
        s = os.stat(p)
        out.append({'path': p, 'role': role, 'size': s.st_size + (1 if bad_size else 0), 'mtime': int(s.st_mtime),
                    'format': 'jsonl'})
    return out


def make_sidecar(name, *, rule='v1', fingerprint='f' * 64, limit=None, train=TRAIN, val=VAL, stale=False,
                 as_string=False, families=None):
    d = os.path.join(SCRATCH, 'sidecars', name)
    if os.path.isdir(d):
        shutil.rmtree(d)
    rng = np.random.default_rng(0)
    emb = rng.standard_normal((16, st.EMB_DIM)).astype(np.float32)
    emb /= np.linalg.norm(emb, axis=1, keepdims=True)
    keys = st.target_keys([f'Findings: synthetic {i}' for i in range(16)])
    ds = {'train': _dsrec(train, 'train', bad_size=stale), 'val': _dsrec(val, 'val')}
    meta = {'rule_version': rule, 'teacher_path': '/fastdata4/sahil_work/tag_extractor/bb_23sep',
            'teacher_fingerprint': fingerprint, 'fingerprint_ok': True, 'max_length': 1024, 'dtype': 'fp32',
            'datasets': str(ds) if as_string else ds, 'limit': limit}
    st.write_sidecar(d, keys, emb, np.ones(16, np.uint8), meta, families=families)
    return d


def run(extra_env=None, dry=True, timeout=600):
    env = {k: v for k, v in os.environ.items() if not k.startswith(('SEM_', 'QURE_', 'MLD_', 'ASL_', 'SWIFT_SEM'))}
    env.update({
        'CONDA_ENV': FAKE_ENV, 'CUDA_VISIBLE_DEVICES': '', 'CACHE_ROOT': os.path.join(SCRATCH, 'cache'),
        'CACHE_TMPDIR': os.path.join(SCRATCH, 'tmp'), 'OUTPUT_DIR': os.path.join(SCRATCH, 'out'),
        'OUT_FREE_MIN_GB': '1', 'ALLOW_BUSY_GPUS': '1', 'FAKE_SWIFT_OUT': os.path.join(SCRATCH, 'fake'),
    })
    for k in ('DATASETS', 'VAL_DATASETS', 'BASE_MODEL', 'RESUME', 'EXTRA_ARGS', 'EXTRA_PLUGINS', 'GPUS',
              'NPROC_PER_NODE', 'DRY_RUN', 'DETACH', 'PRECHECK', 'MIX', 'SKIP_BROKEN_IMAGES'):
        env.pop(k, None)
    if dry:
        env['DRY_RUN'] = '1'
    env.update(extra_env or {})
    for suf in ('.argv.json', '.env', '.oom', '.done'):
        p = env['FAKE_SWIFT_OUT'] + suf
        if os.path.exists(p):
            os.remove(p)
    p = subprocess.run(['bash', LAUNCHER], env=env, capture_output=True, text=True, timeout=timeout)
    return p.returncode, p.stdout, p.stderr


def cmd_of(stdout):
    lines = [x for x in stdout.splitlines() if x.startswith('[stage7] CMD:')]
    assert len(lines) == 1, stdout
    return shlex.split(lines[0][len('[stage7] CMD:'):])


def arg_values(cmd, flag):
    """Values after `flag` up to the next --flag."""
    i = cmd.index(flag)
    out = []
    for x in cmd[i + 1:]:
        if x.startswith('--'):
            break
        out.append(x)
    return out


def fake_ran():
    return os.path.exists(os.path.join(SCRATCH, 'fake.done'))


# =========================================================================================================
def test_bash_syntax():
    p = subprocess.run(['bash', '-n', LAUNCHER], capture_output=True, text=True)
    assert p.returncode == 0, p.stderr
    assert os.access(LAUNCHER, os.X_OK)


def test_missing_sidecar_prints_precompute():
    missing = os.path.join(SCRATCH, 'sidecars', 'does_not_exist')
    rc, out, err = run({'SEM_TARGETS_DIR': missing})
    assert rc == 1, err
    assert 'ABORT: SEM_TARGETS_DIR does not exist' in err
    line = [x for x in err.splitlines() if 'run_build_sem_targets.sh' in x]
    assert len(line) == 1, err
    s = line[0]
    assert f"OUT='{missing}'" in s and "setsid nohup bash" in s and s.rstrip().endswith('& disown')
    assert f"DATASETS='{' '.join(TRAIN)}'" in s and f"VAL_DATASETS='{' '.join(VAL)}'" in s
    assert "GPUS='0 1 2 3 4 5 6 7'" in s
    # an existing dir without meta.json (build in progress / never finished) is refused the same way
    part = os.path.join(SCRATCH, 'sidecars', 'partial')
    os.makedirs(part, exist_ok=True)
    rc, out, err = run({'SEM_TARGETS_DIR': part})
    assert rc == 1 and 'meta.json missing' in err and 'run_build_sem_targets.sh' in err, err
    assert not fake_ran()


def test_wrong_rule_version_refused():
    d = make_sidecar('rule_v0', rule='v0')
    rc, out, err = run({'SEM_TARGETS_DIR': d})
    assert rc == 1, err
    assert "rule_version 'v0'" in err and 'rebuild' in err, err
    # complete-but-bad sidecar -> the rebuild command targets a NEW directory
    assert f"OUT='{d}_rebuild_" in err and 'SEM_TARGETS_DIR=' in err, err
    assert not fake_ran()


def test_no_fingerprint_refused():
    d = make_sidecar('no_fp', fingerprint=None)
    rc, out, err = run({'SEM_TARGETS_DIR': d})
    assert rc == 1 and 'teacher_fingerprint' in err, err


def test_limit_and_coverage_strict_vs_warn():
    d = make_sidecar('smoke_like', limit=24, train=TRAIN[4:], as_string=True)   # builder-style str(dict) too
    rc, out, err = run({'SEM_TARGETS_DIR': d})
    assert rc == 1, err
    assert 'SMOKE build' in err
    for f in TRAIN[:4]:
        assert f'train file {f} was not part of the sidecar build' in err, err
    assert f'train file {TRAIN[4]} was not part' not in err
    rc, out, err = run({'SEM_TARGETS_DIR': d, 'SEM_SIDECAR_CHECK': 'warn'})
    assert rc == 0, err
    assert 'WARNING: sidecar is a --limit 24 SMOKE build' in err
    assert '[stage7-precheck] OK (5 warning(s))' in err, err
    cmd_of(out)
    rc, out, err = run({'SEM_TARGETS_DIR': d, 'SEM_SIDECAR_CHECK': 'bogus', 'PRECHECK': '0'})
    assert rc == 1 and 'SEM_SIDECAR_CHECK must be strict|warn' in err


def test_changed_dataset_refused():
    d = make_sidecar('stale', stale=True)
    rc, out, err = run({'SEM_TARGETS_DIR': d})
    assert rc == 1, err
    assert err.count('changed since the sidecar build') == len(TRAIN), err
    # a val file that was only ever built as a TRAIN file is fine (lookups hit; in_train=1)
    d2 = make_sidecar('val_as_train', train=TRAIN + VAL, val=[])
    rc, out, err = run({'SEM_TARGETS_DIR': d2})
    assert rc == 0, err


def test_dry_run_valid():
    d = make_sidecar('good')
    t0 = time.time()
    rc, out, err = run({'SEM_TARGETS_DIR': d})
    assert rc == 0, err
    print(f'  dry run {time.time() - t0:.1f}s')
    assert not fake_ran(), 'DRY_RUN must not launch'
    for s in ('tokenizer: K=8 SEM ids 248211..248218 | len(tokenizer)=248219 config vocab_size=248320',
              'core SEM hook live: train encode 2/2 blocks', 'ml_decoder_head.* 23', 'sem_head.* 0',
              '[stage7-precheck] OK (0 warning(s))', 'DRY_RUN=1: every precheck passed; NOT launching',
              'global batch 1024', 'ASL_WEIGHT=0.05', 'MLD_RESTORE_HEAD=1', 'MLD_FREEZE_HEAD=0'):
        assert s in err, (s, err)
    cmd = cmd_of(out)
    assert cmd[:2] == ['swift', 'sft']
    assert arg_values(cmd, '--model') == [CKPT]
    assert arg_values(cmd, '--new_special_tokens') == [TOKEN_TXT, TOKEN_SEM]
    plugins = arg_values(cmd, '--external_plugins')
    assert [os.path.basename(p) for p in plugins] == ['mldecoder_joint_plugin.py', 'qure_augment_plugin.py',
                                                      'qure_ce_weight_plugin.py', 'qure_stage3_plugin.py',
                                                      'sem_latent_plugin.py']
    assert plugins[-1] == SEM_PLUGIN
    assert arg_values(cmd, '--dataset') == TRAIN and arg_values(cmd, '--val_dataset') == VAL
    want = {'--packing': ['false'], '--padding_free': ['false'], '--use_liger_kernel': ['true'],
            '--deepspeed': ['zero2'], '--attn_impl': ['flash_attention_2'], '--add_non_thinking_prefix': ['true'],
            '--loss_scale': ['ignore_empty_think'], '--run_name': ['Qwen3.5-2B_stage7_sem_latent'],
            '--tuner_type': ['full'], '--lazy_tokenize': ['true'], '--split_dataset_ratio': ['0'],
            '--per_device_train_batch_size': ['8'], '--gradient_accumulation_steps': ['16'],
            '--learning_rate': ['4e-6'], '--save_total_limit': ['3'], '--max_length': ['28000'],
            '--system': ['You are a helpful medical assistant.'], '--report_to': ['tensorboard', 'wandb'],
            '--output_dir': [os.path.join(SCRATCH, 'out')], '--save_only_model': ['false']}
    for k, v in want.items():
        assert arg_values(cmd, k) == v, (k, arg_values(cmd, k))
    assert '--resume_from_checkpoint' not in cmd
    env_line = [x for x in err.splitlines() if x.startswith('[stage7] env: SEM_')][0]
    for kv in ('SEM_W_COS=0.05', 'SEM_W_NCE=0.05', 'SEM_TAU=0.05', 'SEM_NCE_MODE=soft', 'SEM_QUEUE=16384',
               'SEM_POOL=attn', 'SEM_HEAD_LR=2e-4', 'SEM_DETACH_STEPS=20', 'SEM_RESTORE_HEAD=1', 'SEM_FREEZE_HEAD=0',
               'SEM_INIT_EMBED=1', 'SEM_ONLINE_TEACHER=0', 'SEM_SKIP_LEAKED=1', 'SEM_UNTIE_CONFIG=1',
               'SEM_CHAT_TEMPLATE=1', f'SEM_TARGETS_DIR={d}',
               'SEM_PROBE_HEAD=/fastdata4/sahil_work/tag_extractor/lp23sep/head_xray_bce.pt'):
        assert kv in env_line.split(), (kv, env_line)


def test_real_path_runs_same_argv_and_env():
    d = make_sidecar('good_real')
    rc, out, err = run({'SEM_TARGETS_DIR': d, 'PRECHECK': '0'})
    assert rc == 0, err
    dry = cmd_of(out)
    rc, out, err = run({'SEM_TARGETS_DIR': d, 'PRECHECK': '0'}, dry=False)
    assert rc == 0, err
    assert fake_ran(), err
    argv = json.load(open(os.path.join(SCRATCH, 'fake.argv.json')))
    assert ['swift', 'sft'] + argv[1:] == dry and argv[0] == 'sft', (argv[:3], dry[:3])
    env = dict(x.split('=', 1) for x in open(os.path.join(SCRATCH, 'fake.env')).read().splitlines())
    assert env['CUDA_VISIBLE_DEVICES'] == '0,1,2,3,4,5,6,7' and env['NPROC_PER_NODE'] == '8'
    assert env['ASL_WEIGHT'] == '0.05' and env['AR_WEIGHT'] == '1.0'
    assert env['MLD_RESTORE_HEAD'] == '1' and env['MLD_FREEZE_HEAD'] == '0'
    assert env['SEM_TARGETS_DIR'] == d and env['SEM_W_COS'] == '0.05' and env['SEM_W_NCE'] == '0.05'
    assert env['QURE_AUG_BBOX_MODE'] == 'photometric' and env['QURE_W_TRUE'] == '6.0'
    assert env['HF_HOME'] == os.path.join(SCRATCH, 'cache', 'hf')
    assert env['PYTHONPATH'].split(':')[0] == REPO
    assert 'QURE_PLUGIN_MODE' not in env and 'SWIFT_SEM_LATENT' not in env
    assert env['SKIP_BROKEN_IMAGES'] == '1', 'MIX=mcr (default) must export SKIP_BROKEN_IMAGES=1'
    assert open(os.path.join(SCRATCH, 'fake.oom')).read().strip() == '1000'
    assert '[stage7] CMD: swift sft' in err


def test_detach():
    d = make_sidecar('good_detach')
    log_dir = os.path.join(SCRATCH, 'out')
    before = set(os.listdir(log_dir)) if os.path.isdir(log_dir) else set()
    rc, out, err = run({'SEM_TARGETS_DIR': d, 'PRECHECK': '0', 'DETACH': '1'}, dry=False)
    assert rc == 0, err
    assert 'detached: pid' in err, err
    for _ in range(120):
        if fake_ran():
            break
        time.sleep(0.5)
    assert fake_ran(), 'detached child never reached swift'
    new = sorted(set(os.listdir(log_dir)) - before)
    assert any(x.startswith('train_') and x.endswith('.log') for x in new), new
    log = open(os.path.join(log_dir, [x for x in new if x.startswith('train_')][-1])).read()
    assert '[stage7] CMD: swift sft' in log and 'PRECHECK=0' in log, log[-2000:]


def test_guards():
    d = make_sidecar('good_guards')
    cases = [({'SWIFT_SEM_LATENT': '0'}, 'SWIFT_SEM_LATENT=0 disables'),
             ({'QURE_PLUGIN_MODE': 'infer'}, 'QURE_PLUGIN_MODE=infer would skip'),
             ({'RESUME': os.path.join(SCRATCH, 'no_such_ckpt')}, 'RESUME checkpoint missing'),
             ({'ROOT_FREE_MIN_GB': '100000000'}, 'root disk has only'),
             ({'OUT_FREE_MIN_GB': '100000000'}, 'output disk'),
             ({'BASE_MODEL': os.path.join(SCRATCH, 'no_such_base')}, 'base ckpt missing'),
             ({'DATASETS': os.path.join(SCRATCH, 'no_such.jsonl')}, 'dataset missing')]
    for extra, msg in cases:
        rc, out, err = run({'SEM_TARGETS_DIR': d, 'PRECHECK': '0', **extra})
        assert rc == 1 and msg in err, (extra, err)
    rc, out, err = run({'SEM_TARGETS_DIR': d, 'SEM_POOL': 'max'})     # invalid knob -> the plugin's SemConfig
    assert rc == 1 and 'SEM_POOL must be attn|mean|concat' in err, err
    rc, out, err = run({'SEM_TARGETS_DIR': d, 'QURE_PLUGIN_DEDUP': '0', 'PRECHECK': '0'})
    assert rc == 0 and 'LEGACY double plugin import' in err, err
    if shutil.which('nvidia-smi'):
        busy = {'SEM_TARGETS_DIR': d, 'PRECHECK': '0', 'ALLOW_BUSY_GPUS': '0', 'GPU_BUSY_MIB': '-1'}
        rc, out, err = run(busy)
        assert rc == 0 and 'a real launch would ABORT here' in err, err
        rc, out, err = run(busy, dry=False)
        assert rc == 1 and 'GPUs in use' in err and 'never kills anything' in err, err
        assert not fake_ran()


def test_overrides():
    d = make_sidecar('good_over')
    resume = os.path.join(SCRATCH, 'fake_resume_ckpt')
    os.makedirs(resume, exist_ok=True)
    open(os.path.join(resume, 'trainer_state.json'), 'w').write('{}')
    rc, out, err = run({
        'SEM_TARGETS_DIR': d, 'RESUME': resume, 'IGNORE_DATA_SKIP': '1', 'DATA_SEED': '7', 'GPUS': '0,1',
        'DATASETS': ' '.join([TRAIN[0] + '#1000'] + TRAIN[1:]), 'EXTRA_ARGS': '--max_steps 20 --remove_unused_columns false',
        'EXTRA_PLUGINS': os.path.join(REPO, 'swift_plugins', 'qure_untie_config_plugin.py'),
        'BASE_MODEL': CKPT.replace('/fastdata4/', '/localstorage/', 1),
        'SEM_PROBE_HEAD': '/localstorage/sahil_work/tag_extractor/lp23sep/head_xray_bce.pt'})
    assert rc == 0, err
    cmd = cmd_of(out)
    assert arg_values(cmd, '--model') == [CKPT], 'a /localstorage path must resolve to /fastdata4 on this host'
    assert arg_values(cmd, '--resume_from_checkpoint') == [resume] and arg_values(cmd, '--ignore_data_skip') == ['true']
    assert arg_values(cmd, '--data_seed') == ['7']
    assert arg_values(cmd, '--dataset')[0] == TRAIN[0] + '#1000'
    assert cmd[-4:] == ['--max_steps', '20', '--remove_unused_columns', 'false']
    plugins = arg_values(cmd, '--external_plugins')
    assert os.path.basename(plugins[-2]) == 'qure_untie_config_plugin.py' and plugins[-1] == SEM_PLUGIN
    assert 'global batch 256 (2 GPUs x 8 x GAS 16)' in err, err
    assert 'SEM_PROBE_HEAD=/fastdata4/sahil_work/tag_extractor/lp23sep/head_xray_bce.pt' in err
    assert 'RESUMING from' in err



def test_final_argv_checks_and_new_flags():
    """review D1 / P1 / P3 / TGT-1: --ddp_timeout is passed (swift's default hangs ~208 days); the precheck parses the
    FINAL argv (EXTRA_ARGS last) with swift's parser and runs the SEM plugin's refusals on it; an override-bearing
    sidecar adds --remove_unused_columns false (unless EXTRA_ARGS sets it)."""
    d = make_sidecar('good_argv')
    rc, out, err = run({'SEM_TARGETS_DIR': d})
    assert rc == 0, err
    cmd = cmd_of(out)
    assert arg_values(cmd, '--ddp_timeout') == ['10800'] and '--remove_unused_columns' not in cmd, cmd
    assert 'accepted by the SEM plugin' in err and 'loss_scale=ignore_empty_think template_backend=swift' in err, err
    rc, out, err = run({'SEM_TARGETS_DIR': d, 'DDP_TIMEOUT': '900', 'PRECHECK': '0'})
    assert rc == 0 and arg_values(cmd_of(out), '--ddp_timeout') == ['900'], err
    for extra, needle in (('--loss_scale all', 'base strategy "all"'),
                          ('--loss_scale all+ignore_empty_think', 'base strategy "all"'),
                          ('--template_backend jinja', '--template_backend jinja is not supported for training'),
                          ('--loss_scale hermes', 'is non-binary'),
                          ('--enable_channel_loss true', '--enable_channel_loss'),
                          ('--label_smoothing_factor 0.1', '--label_smoothing_factor')):
        rc, out, err = run({'SEM_TARGETS_DIR': d, 'EXTRA_ARGS': extra})
        assert rc == 1 and needle in err and 'python precheck failed' in err, (extra, err[-3000:])
        assert not fake_ran()
    fam = np.full(16, st.FAMILY_BITS[st.OVERRIDE_FAMILY], np.uint8)
    d2 = make_sidecar('override_argv', families=fam)
    rc, out, err = run({'SEM_TARGETS_DIR': d2})
    assert rc == 0, err
    cmd = cmd_of(out)
    assert arg_values(cmd, '--remove_unused_columns') == ['false'] and 'override texts -> --remove_unused_columns false' in err
    rc, out, err = run({'SEM_TARGETS_DIR': d2, 'EXTRA_ARGS': '--remove_unused_columns true'})   # the user's choice wins
    assert rc == 0, err
    cmd = cmd_of(out)
    assert cmd.count('--remove_unused_columns') == 1 and arg_values(cmd, '--remove_unused_columns') == ['true']
    assert 'overrides are IGNORED' in err                                        # ... and the plugin warns


BACKUP_BASE = ('/tmp/claude-1000/-fastdata4-sahil-work-VLM-qure-vlm-swift/b2c1bd9c-54da-4ec4-b14d-2d8473057a20/'
               'scratchpad/pre_change_backup/swift/template/base.py')


def test_hook_negative_control():
    """The 'core SEM hook live' check can FAIL: a swift tree whose base.py is the pre-change backup (no hook) but
    which still has sem_latent.py is put first on the precheck's PYTHONPATH (via a wrapper python)."""
    if not os.path.isfile(BACKUP_BASE):
        print('  SKIP (no pre-change backup of base.py)')
        return
    root = os.path.join(SCRATCH, 'hookless')
    if os.path.isdir(root):
        shutil.rmtree(root)
    shutil.copytree(os.path.join(REPO, 'swift'), os.path.join(root, 'swift'),
                    ignore=shutil.ignore_patterns('__pycache__'))
    shutil.copy2(BACKUP_BASE, os.path.join(root, 'swift', 'template', 'base.py'))
    env_dir = os.path.join(SCRATCH, 'fake_env_hookless')
    os.makedirs(os.path.join(env_dir, 'bin'), exist_ok=True)
    with open(os.path.join(env_dir, 'bin', 'python'), 'w') as f:
        f.write(f'#!/usr/bin/env bash\nexport PYTHONPATH="{root}:$PYTHONPATH"\nexec {ENV_PY} "$@"\n')
    os.chmod(os.path.join(env_dir, 'bin', 'python'), 0o755)
    shutil.copy2(os.path.join(FAKE_ENV, 'bin', 'swift'), os.path.join(env_dir, 'bin', 'swift'))
    d = make_sidecar('good_hookless')
    rc, out, err = run({'SEM_TARGETS_DIR': d, 'CONDA_ENV': env_dir})
    assert rc == 1, err
    assert 'train-mode encode: 0 SEM blocks (want 2)' in err and 'core hook' in err, err
    assert 'python precheck failed' in err and not fake_ran()


def test_mix_defaults():
    """MIX=mcr (default) vs MIX=6c: default DATASETS / VAL_DATASETS / SEM_TARGETS_DIR / SKIP_BROKEN_IMAGES."""
    # default MIX, no SEM_TARGETS_DIR: the stage7_mcr sidecar is the default (missing -> the exact build command)
    rc, out, err = run({'PRECHECK': '0'})
    if os.path.isfile(os.path.join(SIDECAR_MCR, 'meta.json')):
        assert rc == 0 and f'SEM_TARGETS_DIR={SIDECAR_MCR}' in err, err
        cmd = cmd_of(out)
        assert arg_values(cmd, '--dataset') == TRAIN and arg_values(cmd, '--val_dataset') == VAL
    else:
        assert rc == 1 and SIDECAR_MCR in err, err
        line = [x for x in err.splitlines() if 'run_build_sem_targets.sh' in x]
        assert len(line) == 1 and f"OUT='{SIDECAR_MCR}'" in line[0], err
        assert f"DATASETS='{' '.join(TRAIN)}'" in line[0] and f"VAL_DATASETS='{' '.join(VAL)}'" in line[0], line
    assert 'data MIX=mcr (6 train / 3 val files)' in err, err
    # MIX=6c: the stage-6c mixture + its own sidecar dir; SKIP_BROKEN_IMAGES is not exported (legacy behaviour)
    d6 = make_sidecar('good_6c', train=TRAIN_6C, val=VAL_6C)
    rc, out, err = run({'MIX': '6c', 'SEM_TARGETS_DIR': d6})
    assert rc == 0, err
    cmd = cmd_of(out)
    assert arg_values(cmd, '--dataset') == TRAIN_6C and arg_values(cmd, '--val_dataset') == VAL_6C
    assert 'data MIX=6c (5 train / 2 val files)' in err and 'SKIP_BROKEN_IMAGES=<unset>' in err, err
    # the MCR-covering sidecar does not cover the 6c files (strict coverage check)
    rc, out, err = run({'MIX': '6c', 'SEM_TARGETS_DIR': make_sidecar('good_mcr_for_6c')})
    assert rc == 1 and f'train file {TRAIN_6C[0]} was not part of the sidecar build' in err, err
    rc, out, err = run({'MIX': '6c', 'PRECHECK': '0'})
    if os.path.isfile(os.path.join(SIDECAR_6C, 'meta.json')):
        assert rc == 0 and f'SEM_TARGETS_DIR={SIDECAR_6C}' in err, err
    else:
        assert rc == 1 and f"OUT='{SIDECAR_6C}'" in err, err
    # explicit values win; a bad MIX is refused
    rc, out, err = run({'MIX': '6c', 'SEM_TARGETS_DIR': d6, 'PRECHECK': '0', 'SKIP_BROKEN_IMAGES': '1'})
    assert rc == 0 and 'SKIP_BROKEN_IMAGES=1' in err, err
    rc, out, err = run({'SEM_TARGETS_DIR': make_sidecar('good_mix'), 'PRECHECK': '0', 'SKIP_BROKEN_IMAGES': '0'})
    assert rc == 0 and 'SKIP_BROKEN_IMAGES=0' in err, err
    rc, out, err = run({'MIX': '6c', 'SEM_TARGETS_DIR': d6, 'PRECHECK': '0', 'DATASETS': TRAIN_6C[0]})
    assert rc == 0 and arg_values(cmd_of(out), '--dataset') == [TRAIN_6C[0]] and '(DATASETS overridden)' in err, err
    rc, out, err = run({'MIX': 'bogus', 'PRECHECK': '0'})
    assert rc == 1 and "MIX must be mcr|6c, got 'bogus'" in err, err
    assert not fake_ran()


TESTS = [test_bash_syntax, test_missing_sidecar_prints_precompute, test_wrong_rule_version_refused,
         test_no_fingerprint_refused, test_limit_and_coverage_strict_vs_warn, test_changed_dataset_refused,
         test_dry_run_valid, test_real_path_runs_same_argv_and_env, test_detach, test_guards, test_overrides,
         test_final_argv_checks_and_new_flags, test_hook_negative_control, test_mix_defaults]

if __name__ == '__main__':
    n_fail = 0
    t_all = time.time()
    for fn in TESTS:
        t0 = time.time()
        try:
            fn()
            print(f'PASS {fn.__name__} ({time.time() - t0:.1f}s)', flush=True)
        except Exception as e:  # noqa: BLE001
            n_fail += 1
            import traceback
            traceback.print_exc()
            print(f'FAIL {fn.__name__}: {type(e).__name__}: {str(e)[:3000]}', flush=True)
    print(f'{len(TESTS) - n_fail}/{len(TESTS)} passed in {time.time() - t_all:.0f}s')
    sys.exit(1 if n_fail else 0)
