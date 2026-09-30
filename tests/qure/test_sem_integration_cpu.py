#!/usr/bin/env python
"""SEM plugin on REAL weights, CPU only: checkpoint-608 (stage 6b) + SEM tokens + sem_latent_plugin.

RUN (CPU only; standalone or under pytest; several minutes, peak RSS asserted < 40 GB)
    CUDA_VISIBLE_DEVICES="" PYTHONPATH=/fastdata4/sahil_work/VLM_qure/vlm/swift \\
        /home/ubuntu/miniconda3/envs/ms-swift-may-s4/bin/python tests/qure/test_sem_integration_cpu.py
The work runs in a fresh subprocess (plugins patch swift classes process-wide); the parent asserts
on its JSON result and exits non-zero on failure.

WHAT IS PROVEN (real Qwen3.5-2B weights)
  * swift get_model_processor(ckpt-608, new_special_tokens = token.txt + token_sem.txt) on CPU:
    len(tokenizer) 248219, embedding rows 248320 -> NO resize; lm_head is a separate tensor,
    bit-identical to the checkpoint's saved lm_head.weight, different from embed_tokens (the
    untied state is preserved); embed_tokens equals the saved tensor too.
  * the 8 SEM rows (+ the other spare rows) are one identical vector before attach; the plugin's
    prepare_model attach re-initialises exactly the SEM rows (other rows and lm_head unchanged),
    untie sets tie_word_embeddings False at both levels, the SEM chat template is installed.
  * 2 real rows (an onto report row, a B3 tool row) with small resized images through the real
    template / collator / pre_forward_hook, reentrant gradient checkpointing ON (as in the launchers;
    the language_model hook must fire exactly once, with grad): forward + backward once after the
    detach warm-up;
    loss components printed; out.loss - incoming loss == w_cos*L_cos + w_nce*L_nce; grads defined on
    sem_head and non-zero on the SEM embedding rows.
  * sem_head restore path: the trained-shape head is saved as a `sem_head.*` safetensors shard next
    to symlinked checkpoint shards with an index naming it; the plugin is re-attached (fresh random
    head) with model_dir pointing there -> attach restores it bit-exactly via safe_open; the SEM
    rows (now distinct) are left unchanged by the second attach.

ENV
  SEM_IT_DTYPE      bfloat16 (default) | float32
  SEM_IT_IMG_TOKENS IMAGE_MAX_TOKEN_NUM for the encodes (default 256)
  SEM_TEST_OUT      artefact dir (default <scratchpad>/tests_out/sem_plugin), shared with test_sem_plugin.py
"""
import json
import os
import resource
import shutil
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import test_sem_plugin as T  # noqa: E402  (shared helpers: rows, tiny sidecar, tokens; no tests imported)

# review BC-3: T's process env only when this module runs, or around its tests under pytest (restored after)
if __name__ == '__main__':
    T._apply_test_env()
else:
    _qure_module_env = T.module_env_fixture()

OUT_IT = os.path.join(T.OUT, 'integration')
ROW_IDS = ('onto_plain', 'b3_tool')


def _mem():
    """(VmRSS, VmHWM) of this process in GB."""
    v = {}
    for line in open('/proc/self/status'):
        if line.startswith(('VmRSS:', 'VmHWM:')):
            k, n = line.split(':')
            v[k] = int(n.split()[0]) / 1024 / 1024
    return round(v.get('VmRSS', 0), 2), round(v.get('VmHWM', 0), 2)


def _run():
    """Runs INSIDE the subprocess."""
    import torch
    from safetensors import safe_open
    from safetensors.torch import save_file
    T.disable_fla()
    torch.manual_seed(0)
    torch.set_num_threads(int(os.environ.get('OMP_NUM_THREADS', '32')))
    R = {}
    t0 = time.time()
    dtype = getattr(torch, os.environ.get('SEM_IT_DTYPE', 'bfloat16'))
    from swift.model.register import get_model_processor
    model, proc = get_model_processor(T.CKPT, torch_dtype=dtype, device_map='cpu', new_special_tokens=T.sem_tokens())
    R['load_secs'] = time.time() - t0
    R['mem'] = {'after_load': _mem()}
    tok = proc.tokenizer
    E = model.get_input_embeddings().weight
    H = model.get_output_embeddings().weight
    idx = json.load(open(os.path.join(T.CKPT, 'model.safetensors.index.json')))['weight_map']
    ekey = 'model.language_model.embed_tokens.weight'

    def saved(k):
        with safe_open(os.path.join(T.CKPT, idx[k]), framework='pt') as f:
            return f.get_tensor(k)

    Hs, Es = saved('lm_head.weight'), saved(ekey)
    R['len_tok'] = len(tok)
    R['emb_rows'] = int(E.shape[0])
    R['lm_head_is_embed'] = H is E
    R['lm_head_eq_saved'] = bool(torch.equal(H.detach().to(Hs.dtype), Hs))
    R['embed_eq_saved'] = bool(torch.equal(E.detach().to(Es.dtype), Es))
    R['lm_head_minus_embed_max'] = float((H.detach().float() - E.detach().float()).abs().max())
    from swift.template.sem_latent import get_sem_ids
    sem = list(get_sem_ids(tok))
    R['sem_ids'] = [sem[0], sem[-1], len(sem)]
    spare = E.detach()[sem[0]:].double()
    R['spare_spread'] = float((spare - spare[:1]).abs().max())
    R['trained_std_median'] = float(E.detach()[:sem[0]].float().std(0).median())
    # the same ONE-vector criterion the plugin uses (bf16 rounding makes the rows not bit-identical)
    R['spare_rows_identical_before'] = R['spare_spread'] <= 0.05 * R['trained_std_median']
    E0 = E.detach().clone()
    H0 = H.detach().clone()
    # plugin, exactly as --external_plugins loads it
    from swift.utils import import_external_file
    import_external_file(os.path.join(T.REPO, 'swift_plugins', 'sem_latent_plugin.py'))
    P = sys.modules['sem_latent_plugin']
    from swift.template import get_template
    tpl = get_template(proc, **T.TEMPLATE_KW)
    tpl.set_mode('train')
    tpl.model = model
    from types import SimpleNamespace
    args = SimpleNamespace(use_liger_kernel=False, is_adapter=False, tuner_type='full', freeze_parameters_ratio=0.0,
                           freeze_parameters=[], freeze_parameters_regex=None, trainable_parameters=[],
                           trainable_parameters_regex=None, use_galore=False, packing=False, padding_free=False,
                           sequence_parallel_size=1, model_type='qwen3_5')
    from swift.pipelines.train.tuner import TunerMixin
    t1 = time.time()
    model = TunerMixin().prepare_model(args, model, template=tpl)
    R['attach_secs'] = time.time() - t1
    R['mem']['after_attach'] = _mem()
    E1 = model.get_input_embeddings().weight.detach()
    R['sem_rows_reinit'] = bool(not torch.equal(E1[sem], E0[sem]) and (E1[sem][1:] - E1[sem][:1]).abs().max() > 0)
    R['other_rows_unchanged'] = bool(torch.equal(E1[:sem[0]], E0[:sem[0]]) and torch.equal(E1[sem[-1] + 1:],
                                                                                              E0[sem[-1] + 1:]))
    R['lm_head_unchanged'] = bool(torch.equal(model.get_output_embeddings().weight.detach(), H0))
    R['cfg_tie'] = [model.config.tie_word_embeddings, model.config.text_config.tie_word_embeddings]
    R['chat_template_sem'] = '<|sem_0|>' in (proc.chat_template or '') and '<|sem_0|>' in (tok.chat_template or '')
    R['head_params'] = sum(p.numel() for p in model.sem_head.parameters())
    R['head_dtype'] = str(next(model.sem_head.parameters()).dtype)
    # 2 real rows, small resized images
    rows = dict(T.pick_rows())
    encs = [tpl.encode(T.to_swift_row(rows[r]), return_length=True) for r in ROW_IDS]
    batch = tpl.data_collator(encs)
    R['seq'] = list(batch['input_ids'].shape)
    R['sem_st'] = batch['sem_st'].tolist()
    tpl.register_post_encode_hook([model])
    model.train()
    # like the launchers (--gradient_checkpointing true, reentrant under DeepSpeed): also keeps the CPU
    # torch fallback of the 18 GatedDeltaNet layers from holding ~38 GB of autograd state
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': True})
    fires = []
    model.model.language_model.register_forward_hook(
        lambda m, a, o: fires.append((torch.is_grad_enabled(), bool(o.last_hidden_state.requires_grad))))
    P._SCHED['step'] = P.CFG.detach_steps            # past the warm-up: SEM grads reach the LM
    P._STATE['keep_term'] = True
    t2 = time.time()
    out = model(**batch)
    R['fwd_secs'] = time.time() - t2
    R['mem']['after_forward'] = _mem()
    s = dict(P._STATE['last'])
    R['stats'] = s
    R['loss'] = float(out.loss)
    R['diff_vs_incoming'] = float(out.loss) - s['ce']
    R['expected'] = s['w_cos'] * s['cos_loss'] + s['w_nce'] * s['nce_loss']
    t3 = time.time()
    out.loss.backward()
    R['bwd_secs'] = time.time() - t3
    R['lm_hook_fires'] = fires              # [(grad enabled, output requires grad)] per call, fwd + bwd
    R['mem']['after_backward'] = _mem()
    R['head_grads_defined'] = all(p.grad is not None for p in model.sem_head.parameters())
    R['head_grad_abs'] = float(sum(p.grad.float().abs().sum() for p in model.sem_head.parameters()))
    g = model.get_input_embeddings().weight.grad
    R['sem_embed_grad_abs'] = float(g[sem].float().abs().sum()) if g is not None else 0.0
    R['finite'] = bool(torch.isfinite(out.loss))
    print('[it] loss components ' + json.dumps({k: s.get(k) for k in (
        'ce', 'cos_loss', 'nce_loss', 'kl', 'cos_centred', 'cos_raw', 'w_cos', 'w_nce', 'term', 'total', 'n_blk',
        'n_valid')}), flush=True)
    # ---- restore path: save sem_head as a checkpoint shard, re-attach with model_dir there
    saved_sd = {k: v.detach().clone() for k, v in model.sem_head.state_dict().items()}
    ck2 = os.path.join(OUT_IT, 'ckpt_with_sem_head')
    if os.path.isdir(ck2):
        shutil.rmtree(ck2)
    os.makedirs(ck2)
    for fn in os.listdir(T.CKPT):
        src = os.path.join(T.CKPT, fn)
        if os.path.isfile(src) and fn != 'model.safetensors.index.json':
            os.symlink(src, os.path.join(ck2, fn))
    save_file({f'sem_head.{k}': v.contiguous() for k, v in saved_sd.items()},
              os.path.join(ck2, 'model-sem_head.safetensors'))
    ix = json.load(open(os.path.join(T.CKPT, 'model.safetensors.index.json')))
    ix['weight_map'].update({f'sem_head.{k}': 'model-sem_head.safetensors' for k in saved_sd})
    json.dump(ix, open(os.path.join(ck2, 'model.safetensors.index.json'), 'w'))
    for h in model.__dict__.pop('_sem_hook_handles'):
        h.remove()
    del model.sem_head
    model.__dict__.pop('_sem_attached', None)
    model.model_dir = ck2
    E2 = model.get_input_embeddings().weight.detach().clone()
    torch.manual_seed(999)
    fresh = P.SemanticHead(model.get_input_embeddings().weight.shape[1], P.get_reader().dim, len(sem),
                           pool=P.CFG.pool, hidden=P.CFG.head_hidden).to(P.CFG.head_dtype)
    R['fresh_differs'] = not all(torch.equal(fresh.state_dict()[k], v) for k, v in saved_sd.items())
    torch.manual_seed(999)
    P.attach(model, template=tpl)
    R['restored_equal'] = all(torch.equal(model.sem_head.state_dict()[k], v) for k, v in saved_sd.items())
    R['second_attach_rows_unchanged'] = bool(torch.equal(model.get_input_embeddings().weight.detach(), E2))
    R['mem']['end'] = _mem()
    R['peak_rss_gb'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 / 1024
    R['total_secs'] = time.time() - t0
    print('IT_RESULT ' + json.dumps(R, default=float), flush=True)


def test_sem_integration_real_weights_cpu():
    os.makedirs(OUT_IT, exist_ok=True)
    rows, sc, _, _ = T.rows_and_sidecar()
    env = dict(os.environ, PYTHONPATH=T.REPO, QURE_PLUGIN_MODE='train', SEM_TARGETS_DIR=sc, CUDA_VISIBLE_DEVICES='',
               IMAGE_MAX_TOKEN_NUM=os.environ.get('SEM_IT_IMG_TOKENS', '256'), SEM_LOG_EVERY='1',
               OMP_NUM_THREADS=os.environ.get('OMP_NUM_THREADS', '32'))
    log = os.path.join(OUT_IT, 'integration.log')
    t0 = time.time()
    with open(log, 'w') as f:
        p = subprocess.run([sys.executable, os.path.abspath(__file__), '--run'], stdout=f, stderr=subprocess.STDOUT,
                           env=env, timeout=5400, cwd=OUT_IT)
    txt = open(log).read()
    R = None
    for line in txt.splitlines():
        if line.startswith('IT_RESULT '):
            R = json.loads(line[len('IT_RESULT '):])
        if line.startswith('[it] loss components'):
            print(line)
    assert p.returncode == 0 and R is not None, f'integration run failed (rc {p.returncode}); log {log}:\n{txt[-4000:]}'
    print(json.dumps({k: v for k, v in R.items() if k != 'stats'}, indent=1))
    checks = [
        ('no resize: len(tok) 248219, embedding rows 248320, K=8', R['len_tok'] == 248219 and R['emb_rows'] == 248320
         and R['sem_ids'] == [248211, 248218, 8]),
        ('untied lm_head preserved (separate tensor, == saved lm_head.weight, != embed)', not R['lm_head_is_embed']
         and R['lm_head_eq_saved'] and R['embed_eq_saved'] and R['lm_head_minus_embed_max'] > 0),
        ('spare rows identical before attach -> SEM rows re-initialised, other rows + lm_head unchanged',
         R['spare_rows_identical_before'] and R['sem_rows_reinit'] and R['other_rows_unchanged']
         and R['lm_head_unchanged']),
        ('config untied at both levels, SEM chat template installed', R['cfg_tie'] == [False, False]
         and R['chat_template_sem']),
        ('2 rows: 2 valid SEM blocks', R['sem_st'].count(1) == 2 and R['stats']['n_valid'] == 2),
        ('forward: finite, out.loss - incoming == w_cos*L_cos + w_nce*L_nce', R['finite']
         and abs(R['diff_vs_incoming'] - R['expected']) < 2e-3 * max(1.0, abs(R['expected']))),
        ('backward: sem_head grads defined + non-zero, SEM embedding rows get gradient', R['head_grads_defined']
         and R['head_grad_abs'] > 0 and R['sem_embed_grad_abs'] > 0),
        ('restore: re-attach restores the saved sem_head bit-exactly (fresh init differed)', R['fresh_differs']
         and R['restored_equal']),
        ('second attach leaves the (now distinct) SEM rows unchanged', R['second_attach_rows_unchanged']),
        ('language_model hook fired once per micro-step with grad (reentrant checkpointing)',
         R['lm_hook_fires'] == [[True, True]]),
        ('peak RSS < 40 GB', R['peak_rss_gb'] < 40),
    ]
    bad = [c for c, ok in checks if not ok]
    for c, ok in checks:
        print(f'{"PASS" if ok else "FAIL"} {c}')
    print(f'{len(checks) - len(bad)}/{len(checks)} checks; subprocess {time.time() - t0:.0f}s '
          f'(load {R["load_secs"]:.0f}s, fwd {R["fwd_secs"]:.0f}s, bwd {R["bwd_secs"]:.0f}s), peak RSS '
          f'{R["peak_rss_gb"]:.1f} GB; stats {R["stats"]}')
    assert not bad, bad


if __name__ == '__main__':
    if '--run' in sys.argv:
        _run()
        sys.exit(0)
    try:
        test_sem_integration_real_weights_cpu()
    except AssertionError as e:
        print(f'FAILED: {e}')
        sys.exit(1)
    sys.exit(0)
