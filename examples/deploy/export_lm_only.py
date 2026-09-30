#!/usr/bin/env python
"""Make a `-lm` SERVING export of a training checkpoint (F4): LM + vision weights only.

    python examples/deploy/export_lm_only.py <ckpt> <out> [--no-untie] [--no-sem-template]
    python examples/deploy/export_lm_only.py <ckpt> <out> --dry-run      # plan only

WHY
The `checkpoint-*-lm` dirs served with native sglang / vLLM (e.g. stage6b checkpoint-608-lm,
lmlogprob_10k/ckpt1500_lm) were made ad hoc -- no script existed. Training checkpoints carry
tensors no serving engine knows (sglang logs "Parameter ... not found" and skips them; names
with q_proj/k_proj/... would even be RENAMED by its stacked-param mapping) plus GBs of
optimizer state. This script reproduces those exports and adds the two serving fixes:

  1. WEIGHTS. Streams every safetensors shard with `safe_open` and drops
       ml_decoder_head.*   (ML-Decoder cls head, 23 tensors)
       asl_loss.*          (ASL class-weight buffers, 2 tensors)
       sem_head.*          (SEM latent head)
     (+ any --drop-prefix). A shard with nothing to drop is copied byte-for-byte; a shard with
     dropped keys is rewritten from its kept tensors (one shard in memory at a time), keeping
     its safetensors metadata. model.safetensors.index.json is rewritten (weight_map of kept keys,
     total_size / total_parameters recomputed; shards left empty are omitted).
  2. FILES. Copies config / generation_config / tokenizer* / vocab / merges / special tokens /
     chat_template* / *processor*config. NOT copied: optimizer & DeepSpeed state (global_step*/,
     latest, zero_to_fp32.py), rng_state_*.pth, scheduler.pt, trainer_state.json,
     training_args.bin, and args.json (the existing -lm exports have none, so swift restores
     nothing from them; pass --keep-args-json to keep it).
  3. UNTIE (default on; --no-untie to skip). If both `lm_head.weight` and `embed_tokens.weight`
     are kept and differ while config.json says tie_word_embeddings=true, the export's config is
     set to false at the top level AND in text_config, so sglang/vLLM serve the TRAINED lm_head
     instead of lm_head = embed_tokens (see untie_lm_head.py for the engine code citations).
  4. SEM CHAT TEMPLATE (default on; --no-sem-template to skip). If the tokenizer has `<|sem_0|>`,
     the checkpoint's chat_template.jinja (and any `chat_template` string in tokenizer_config.json /
     processor_config.json) is rewritten with swift.template.sem_latent.make_sem_chat_template so a
     native OpenAI server inserts the SEM block exactly where swift's encode does. The module is
     imported lazily; if it is absent (or returns None: patterns not found) the template is left
     unchanged and a WARNING is printed and recorded.
  5. export_manifest.json: source, flags, dropped keys, per-shard actions, copied / skipped files,
     untie and SEM-template outcomes.

Never writes into the source checkpoint; refuses a non-empty <out> unless --overwrite. With
--overwrite, files of the kinds an export writes that THIS export does not write (e.g. args.json from
an earlier --keep-args-json export, stale shards / index / chat template) are REMOVED first, so the
result is exactly this export (swift would otherwise restore training args from a stale args.json);
unrecognised files and directories are left in place with a warning. Both lists are in the manifest.
"""
import argparse
import datetime
import fnmatch
import json
import os
import shutil
import sys

DROP_PREFIXES = ('ml_decoder_head.', 'asl_loss.', 'sem_head.')
COPY_PATTERNS = ('config.json', 'generation_config.json', 'tokenizer*', 'vocab*', 'merges.txt',
                 'special_tokens_map.json', 'added_tokens.json', 'chat_template*',
                 '*processor*config*.json', 'preprocessor_config.json')
NEVER_COPY = ('global_step*', 'latest', 'zero_to_fp32.py', 'rng_state*', 'scheduler.pt',
              'optimizer*', 'trainer_state.json', 'training_args.bin', '*.safetensors',
              'model.safetensors.index.json', 'config.json.tied_bak', 'export_manifest.json')
INDEX = 'model.safetensors.index.json'
_DTYPE_BYTES = {'BOOL': 1, 'U8': 1, 'I8': 1, 'F8_E4M3': 1, 'F8_E5M2': 1, 'I16': 2, 'U16': 2,
                'F16': 2, 'BF16': 2, 'I32': 4, 'U32': 4, 'F32': 4, 'I64': 8, 'U64': 8, 'F64': 8}


def _here_import(name):
    """Import a sibling script (untie_lm_head.py) without needing examples/ on sys.path."""
    d = os.path.dirname(os.path.abspath(__file__))
    if d not in sys.path:
        sys.path.insert(0, d)
    return __import__(name)


def _log(msg):
    print(f'[export-lm] {msg}', flush=True)


def _warn(msg):
    print(f'[export-lm] WARNING: {msg}', file=sys.stderr, flush=True)


def shard_plan(ckpt):
    """{shard: [keys]} and the source index metadata (or None for an index-less checkpoint)."""
    from safetensors import safe_open
    idx_path = os.path.join(ckpt, INDEX)
    if os.path.isfile(idx_path):
        with open(idx_path) as f:
            idx = json.load(f)
        shards = {}
        for k, fn in idx['weight_map'].items():
            shards.setdefault(fn, []).append(k)
        return shards, idx.get('metadata') or {}, True
    shards = {}
    for fn in sorted(os.listdir(ckpt)):
        if fn.endswith('.safetensors'):
            with safe_open(os.path.join(ckpt, fn), framework='pt') as f:
                shards[fn] = list(f.keys())
    if not shards:
        raise SystemExit(f'no safetensors in {ckpt}')
    return shards, {}, False


def sem_tokens_in(tokenizer_json):
    """['<|sem_0|>', ...] contiguous from 0, read from tokenizer.json added_tokens (no load)."""
    if not os.path.isfile(tokenizer_json):
        return []
    with open(tokenizer_json) as f:
        tj = json.load(f)
    have = {t.get('content') for t in tj.get('added_tokens', [])}
    out, i = [], 0
    while f'<|sem_{i}|>' in have:
        out.append(f'<|sem_{i}|>')
        i += 1
    return out


def install_sem_template(out, sem_toks):
    """Rewrite the export's chat template(s) with the SEM-aware version. Returns a status dict."""
    st = {'K': len(sem_toks)}
    try:
        import importlib
        sl = importlib.import_module('swift.template.sem_latent')
    except Exception as e:  # module not written yet / swift not importable
        st['status'] = f'skipped: swift.template.sem_latent unavailable ({type(e).__name__}: {e})'
        _warn(st['status'] + ' -- chat template left UNCHANGED; native servers will not insert '
              'SEM tokens')
        return st
    make = getattr(sl, 'make_sem_chat_template', None)
    if make is None:
        st['status'] = 'skipped: swift.template.sem_latent has no make_sem_chat_template'
        _warn(st['status'])
        return st
    sem_str = ''.join(sem_toks)
    sem_fn = getattr(sl, 'sem_string', None) or getattr(sl, 'get_sem_str', None)
    if sem_fn is not None:
        try:
            from transformers import AutoTokenizer
            tok = AutoTokenizer.from_pretrained(out)
            s = sem_fn(tok)
            if s != sem_str:
                _log(f'sem_string(tokenizer) = {s!r} (tokenizer.json scan gave {sem_str!r}); '
                     'using sem_string')
            sem_str = s
        except Exception as e:
            _warn(f'sem_string(tokenizer) failed ({type(e).__name__}: {e}); using {sem_str!r}')
    st['sem_str'] = sem_str
    if not sem_str:
        st['status'] = 'skipped: SEM disabled (sem_string empty, e.g. SWIFT_SEM_LATENT=0)'
        _warn(st['status'])
        return st
    patched = []
    jpath = os.path.join(out, 'chat_template.jinja')
    if os.path.isfile(jpath):
        with open(jpath) as f:
            src = f.read()
        new = make(src, sem_str)
        if new is None:
            st['status'] = 'skipped: make_sem_chat_template found no expected pattern'
            _warn(st['status'] + ' in chat_template.jinja -- left UNCHANGED')
            return st
        with open(jpath, 'w') as f:
            f.write(new)
        patched.append('chat_template.jinja')
    for fn in ('tokenizer_config.json', 'processor_config.json'):
        p = os.path.join(out, fn)
        if not os.path.isfile(p):
            continue
        with open(p) as f:
            d = json.load(f)
        if isinstance(d.get('chat_template'), str):
            new = make(d['chat_template'], sem_str)
            if new is None:
                _warn(f'{fn}: chat_template pattern not found -- left unchanged')
                continue
            d['chat_template'] = new
            with open(p, 'w') as f:
                json.dump(d, f, indent=2, ensure_ascii=False)
                f.write('\n')
            patched.append(f'{fn}:chat_template')
    st['patched'] = patched
    st['status'] = 'installed' if patched else 'skipped: no chat template found in checkpoint'
    if not patched:
        _warn(st['status'])
    return st


def stale_outputs(out, shards, drop, to_copy, had_index):
    """(stale, unknown) top-level files already in <out> (an --overwrite re-export).

    stale   = files of the kinds an export writes (weights, index, metadata / tokenizer / template
              files, args.json, export_manifest.json) or must never contain (optimizer / rng / trainer
              state) that THIS export will not (re)write -> removed; e.g. an args.json from an earlier
              --keep-args-json run, which `swift infer --model <out>` would otherwise restore.
    unknown = anything else -> left in place (warned); never deletes what it does not recognise.
    """
    if not os.path.isdir(out):
        return [], []
    planned = {fn for fn, ks in shards.items() if any(not k.startswith(drop) for k in ks)}
    planned |= set(to_copy) | {'export_manifest.json'}
    if had_index or len(shards) > 1:
        planned.add(INDEX)
    kinds = COPY_PATTERNS + NEVER_COPY + ('args.json', )
    stale, unknown = [], []
    for fn in sorted(os.listdir(out)):
        path = os.path.join(out, fn)
        if fn in planned or not (os.path.isfile(path) or os.path.islink(path)):
            if fn not in planned and os.path.isdir(path):
                unknown.append(fn + '/')
            continue
        (stale if any(fnmatch.fnmatch(fn, pat) for pat in kinds) else unknown).append(fn)
    return stale, unknown


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('ckpt')
    ap.add_argument('out')
    ap.add_argument('--no-untie', action='store_true', help='keep config tie_word_embeddings as is')
    ap.add_argument('--no-sem-template', action='store_true', help='keep the chat template as is')
    ap.add_argument('--drop-prefix', action='append', default=[], help='extra key prefix to drop')
    ap.add_argument('--keep-args-json', action='store_true', help='also copy args.json')
    ap.add_argument('--overwrite', action='store_true', help='allow a non-empty <out>')
    ap.add_argument('--dry-run', action='store_true', help='print the plan, write nothing')
    a = ap.parse_args(argv)

    from safetensors import safe_open
    from safetensors.torch import save_file

    src, out = os.path.abspath(a.ckpt), os.path.abspath(a.out)
    if not os.path.isfile(os.path.join(src, 'config.json')):
        raise SystemExit(f'{src} has no config.json')
    if out == src or out.startswith(src + os.sep):
        raise SystemExit('refusing to write into (or below) the source checkpoint')
    if os.path.isdir(out) and os.listdir(out) and not (a.overwrite or a.dry_run):
        raise SystemExit(f'{out} is not empty (pass --overwrite)')
    drop = tuple(DROP_PREFIXES) + tuple(a.drop_prefix)

    shards, src_meta, had_index = shard_plan(src)
    dropped = sorted(k for ks in shards.values() for k in ks if k.startswith(drop))
    _log(f'{src} -> {out}: {sum(len(v) for v in shards.values())} keys in {len(shards)} shard(s), '
         f'dropping {len(dropped)} ({", ".join(sorted({k.split(".")[0] for k in dropped})) or "none"})')

    files = sorted(os.listdir(src))
    to_copy = [f for f in files if os.path.isfile(os.path.join(src, f))
               and any(fnmatch.fnmatch(f, p) for p in COPY_PATTERNS)
               and not any(fnmatch.fnmatch(f, p) for p in NEVER_COPY)]
    if 'config.json' not in to_copy:
        to_copy.insert(0, 'config.json')
    if a.keep_args_json and os.path.isfile(os.path.join(src, 'args.json')):
        to_copy.append('args.json')
    skipped = [f for f in files if f not in to_copy and not f.endswith('.safetensors') and f != INDEX]
    _log(f'copy {to_copy}; skip {skipped}')
    stale, unknown = stale_outputs(out, shards, drop, to_copy, had_index)
    if stale or unknown:
        _log(f'{out}: remove stale {stale}' + (f'; leave unknown {unknown}' if unknown else ''))
    if a.dry_run:
        return 0

    os.makedirs(out, exist_ok=True)
    # ---- --overwrite: files a previous export left that THIS export does not write (args.json from an earlier
    #      --keep-args-json run would make swift restore training args; old shards / index / jinja) are removed
    for fn in stale:
        os.remove(os.path.join(out, fn))
    for fn in unknown:
        _warn(f'{out}/{fn} is not something this export writes; left in place')
    # ---- weights, one shard at a time
    new_map, shard_rep, total_size, total_params = {}, {}, 0, 0
    for fn in sorted(shards):
        keys = shards[fn]
        keep = [k for k in keys if not k.startswith(drop)]
        with safe_open(os.path.join(src, fn), framework='pt') as f:
            meta = f.metadata()
            for k in keep:
                sl = f.get_slice(k)
                n = 1
                for s in sl.get_shape():
                    n *= s
                total_params += n
                total_size += n * _DTYPE_BYTES.get(sl.get_dtype(), 0)
            if not keep:
                shard_rep[fn] = {'action': 'omitted-empty', 'n_keys': 0, 'n_dropped': len(keys)}
                continue
            if len(keep) == len(keys):
                shutil.copyfile(os.path.join(src, fn), os.path.join(out, fn))
                action = 'copied'
            else:
                tensors = {k: f.get_tensor(k) for k in keep}
                save_file(tensors, os.path.join(out, fn), metadata=meta)
                del tensors
                action = 'rewritten'
        for k in keep:
            new_map[k] = fn
        shard_rep[fn] = {'action': action, 'n_keys': len(keep), 'n_dropped': len(keys) - len(keep)}
        _log(f'{fn}: {action} ({len(keep)} kept, {len(keys) - len(keep)} dropped)')
    if had_index or len(shard_rep) > 1:
        meta = dict(src_meta)
        meta['total_size'] = total_size
        if 'total_parameters' in src_meta:
            meta['total_parameters'] = total_params
        with open(os.path.join(out, INDEX), 'w') as f:
            json.dump({'metadata': meta, 'weight_map': dict(sorted(new_map.items()))}, f, indent=2)
            f.write('\n')

    # ---- metadata files
    for fn in to_copy:
        shutil.copy2(os.path.join(src, fn), os.path.join(out, fn))

    # ---- untie (decided on the SOURCE tensors, applied to the EXPORT config)
    untie_rep = {'requested': not a.no_untie}
    if not a.no_untie:
        ut = _here_import('untie_lm_head')
        rep = ut.inspect(src, compute_diff=True)
        untie_rep.update(source_status=rep['status'], before=rep.get('tie'),
                         max_abs=(rep.get('diff') or {}).get('max_abs'))
        head_kept = rep.get('lm_head_key') in new_map
        if rep['status'] == 'untie-needed' and head_kept:
            with open(os.path.join(out, 'config.json')) as f:
                cfg = json.load(f)
            new = ut.untied_config(cfg)
            with open(os.path.join(out, 'config.json'), 'w') as f:
                f.write(json.dumps(new, indent=2, ensure_ascii=False) + '\n')
            untie_rep.update(applied=True, after=ut.tie_flags(new))
            _log(f"untied config (source max|lm_head-embed|={untie_rep['max_abs']:.6g}): "
                 f"{untie_rep['before']} -> {untie_rep['after']}")
        else:
            untie_rep['applied'] = False
            _log(f"untie not applied (source status: {rep['status']})")

    # ---- SEM-aware chat template
    sem_toks = sem_tokens_in(os.path.join(out, 'tokenizer.json'))
    if not sem_toks:
        sem_rep = {'K': 0, 'status': 'n/a (tokenizer has no <|sem_0|>)'}
    elif a.no_sem_template:
        sem_rep = {'K': len(sem_toks), 'status': 'skipped (--no-sem-template)'}
    else:
        sem_rep = install_sem_template(out, sem_toks)
    _log(f"SEM chat template: {sem_rep['status']}")

    # ---- verify + manifest
    got = {}
    for fn in {v for v in new_map.values()}:
        with safe_open(os.path.join(out, fn), framework='pt') as f:
            for k in f.keys():
                got[k] = fn
    assert got == new_map, 'export verification failed: shard contents != index'
    assert not any(k.startswith(drop) for k in got)
    manifest = {
        'tool': 'examples/deploy/export_lm_only.py',
        'created': datetime.datetime.now().isoformat(timespec='seconds'),
        'source': src, 'out': out,
        'flags': {'untie': not a.no_untie, 'sem_template': not a.no_sem_template,
                  'drop_prefixes': list(drop), 'keep_args_json': a.keep_args_json},
        'n_keys_source': sum(len(v) for v in shards.values()), 'n_keys_kept': len(new_map),
        'n_dropped': len(dropped), 'dropped_keys': dropped,
        'shards': shard_rep, 'copied_files': to_copy, 'skipped_files': skipped,
        'removed_stale_files': stale, 'left_unknown_files': unknown,
        'untie': untie_rep, 'sem_template': sem_rep,
    }
    with open(os.path.join(out, 'export_manifest.json'), 'w') as f:
        json.dump(manifest, f, indent=2)
    _log(f'done: {len(new_map)} keys kept, {len(dropped)} dropped -> {out}/export_manifest.json')
    return 0


if __name__ == '__main__':
    sys.exit(main())
