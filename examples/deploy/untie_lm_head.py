#!/usr/bin/env python
"""Report (and optionally fix) the train/serve lm_head mismatch in Qwen3.5 checkpoints (F3).

    python examples/deploy/untie_lm_head.py <ckpt>                      # dry-run (default)
    python examples/deploy/untie_lm_head.py <ckpt> --apply              # rewrite config.json
    python examples/deploy/untie_lm_head.py --all-under <run_dir>       # every checkpoint below
    python examples/deploy/untie_lm_head.py --all-under <dir> --no-diff # fast: keys/config only

THE MISMATCH
Our checkpoints store a separate, TRAINED `lm_head.weight` next to
`model.language_model.embed_tokens.weight` (they diverged from stage 2 on), but config.json says
`tie_word_embeddings: true` at the top level and in text_config.
  * HF / swift (training, --infer_backend pt): both keys present -> NOT tied
    (transformers/modeling_utils.py:2489-2503). The separate lm_head is what was trained/evaluated.
  * sglang 0.5.10: Qwen3.5's LM config is `config.text_config` (sglang/srt/models/qwen3_vl.py:1094);
    text_config.tie_word_embeddings=true -> `lm_head = model.embed_tokens` (qwen3_vl.py:1105-1106)
    and the embedding is copied into it on load (qwen3_5.py:1386-1397). With false, a separate
    ParallelLMHead is built (qwen3_vl.py:1108-1113) and `lm_head.weight` is loaded by name
    (qwen3_5.py:1430-1442).
  * vLLM 0.20.2: `config = hf_text_config` (vllm/model_executor/models/qwen3_5.py:468), tied ->
    `lm_head = model.embed_tokens` (:498-499), untied -> ParallelLMHead loaded from `lm_head.*` via
    hf_to_vllm_mapper (qwen3_vl.py:1619-1623, qwen3_5.py:704-709). For VL models built through
    init_vllm_registered_model it also copies the TOP-LEVEL flag into the text config
    (vllm/config/vllm.py:616-621).
  * HF reads the TOP-LEVEL flag (modeling_utils.py:2421), incl. for a later resize re-tie.
  => serving engines use lm_head = the trained INPUT embedding. Setting the flag false at BOTH
     levels makes every engine use the trained lm_head. Weights are never touched.

WHAT --apply DOES
Rewrites config.json with tie_word_embeddings=false at the top level and in text_config (and
llm_config/language_config if present). The original is copied once to config.json.tied_bak (an
existing backup is never overwritten). Idempotent: an already-untied config is not rewritten.
It REFUSES to untie when `lm_head.weight` is absent (a genuinely tied checkpoint -- untying it
would give sglang/vLLM an uninitialised lm_head) and, unless --even-if-identical, when both
tensors are bit-identical (tied serving is then already correct).

max|lm_head - embed| is computed by streaming row chunks with safetensors `safe_open().get_slice`
(never a whole shard in memory).
"""
import argparse
import json
import os
import shutil
import sys
from typing import Dict, List, Optional, Tuple

BACKUP_SUFFIX = '.tied_bak'
SUB_KEYS = ('text_config', 'llm_config', 'language_config')
EMBED_CANDIDATES = ('model.language_model.embed_tokens.weight', 'model.embed_tokens.weight',
                    'language_model.model.embed_tokens.weight')
HEAD_CANDIDATES = ('lm_head.weight', 'language_model.lm_head.weight')
SKIP_DIRS = {'global_step', 'runs', 'images', 'eval_predictions', '__pycache__', '.git'}


# --------------------------------------------------------------------------------- weights
def weight_map(ckpt: str) -> Dict[str, str]:
    """key -> shard filename, from model.safetensors.index.json or the shard headers."""
    idx = os.path.join(ckpt, 'model.safetensors.index.json')
    if os.path.isfile(idx):
        with open(idx) as f:
            return dict(json.load(f)['weight_map'])
    from safetensors import safe_open
    out = {}
    for fn in sorted(os.listdir(ckpt)):
        if fn.endswith('.safetensors'):
            with safe_open(os.path.join(ckpt, fn), framework='pt') as f:
                for k in f.keys():
                    out[k] = fn
    return out


def _pick(wm: Dict[str, str], candidates, suffix: str) -> Optional[str]:
    for k in candidates:
        if k in wm:
            return k
    hits = [k for k in wm if k.endswith(suffix) and 'visual' not in k and 'mtp' not in k]
    return hits[0] if len(hits) == 1 else None


def locate(ckpt: str) -> Tuple[Optional[str], Optional[str], Dict[str, str]]:
    wm = weight_map(ckpt)
    return _pick(wm, HEAD_CANDIDATES, 'lm_head.weight'), _pick(wm, EMBED_CANDIDATES,
                                                                 'embed_tokens.weight'), wm


def diff_stats(ckpt: str, wm: Dict[str, str], head_key: str, emb_key: str,
               chunk_rows: int = 16384) -> dict:
    """max / mean |lm_head - embed| and #rows that differ, streamed in row chunks."""
    import torch
    from safetensors import safe_open
    fh = safe_open(os.path.join(ckpt, wm[head_key]), framework='pt')
    fe = fh if wm[emb_key] == wm[head_key] else safe_open(os.path.join(ckpt, wm[emb_key]),
                                                          framework='pt')
    sh, se = fh.get_slice(head_key), fe.get_slice(emb_key)
    shape_h, shape_e = list(sh.get_shape()), list(se.get_shape())
    res = {'shape_lm_head': shape_h, 'shape_embed': shape_e, 'dtype': sh.get_dtype()}
    if shape_h != shape_e:
        res.update(identical=False, max_abs=None, mean_abs=None, rows_differ=None,
                   note='shape mismatch')
        return res
    n = shape_h[0]
    mx, tot, tot_h, cnt, rows = 0.0, 0.0, 0.0, 0, 0
    for i in range(0, n, chunk_rows):
        a = sh[i:i + chunk_rows].float()
        b = se[i:i + chunk_rows].float()
        d = (a - b).abs()
        mx = max(mx, float(d.max())) if d.numel() else mx
        tot += float(d.sum(dtype=torch.float64))
        tot_h += float(a.abs().sum(dtype=torch.float64))
        cnt += d.numel()
        rows += int((d.amax(dim=1) > 0).sum()) if d.dim() == 2 else int((d > 0).sum())
    res.update(max_abs=mx, mean_abs=(tot / cnt if cnt else 0.0), rows_differ=rows, rows=n,
               mean_abs_lm_head=(tot_h / cnt if cnt else 0.0), identical=(mx == 0.0))
    return res


# --------------------------------------------------------------------------------- config
def tie_flags(cfg: dict) -> Dict[str, Optional[bool]]:
    out = {'top': cfg.get('tie_word_embeddings')}
    for k in SUB_KEYS:
        if isinstance(cfg.get(k), dict):
            out[k] = cfg[k].get('tie_word_embeddings')
    return out


def untied_config(cfg: dict) -> dict:
    new = json.loads(json.dumps(cfg))
    new['tie_word_embeddings'] = False
    for k in SUB_KEYS:
        if isinstance(new.get(k), dict):
            new[k]['tie_word_embeddings'] = False
    return new


def write_config(ckpt: str, new: dict) -> str:
    """Atomic rewrite; the ORIGINAL is backed up once to config.json.tied_bak."""
    path = os.path.join(ckpt, 'config.json')
    bak = path + BACKUP_SUFFIX
    if not os.path.exists(bak):
        shutil.copy2(path, bak)
    tmp = path + '.untie_tmp'
    with open(tmp, 'w') as f:
        f.write(json.dumps(new, indent=2, ensure_ascii=False) + '\n')
    with open(tmp) as f:
        assert json.load(f) == new
    os.replace(tmp, path)
    return bak


# --------------------------------------------------------------------------------- main logic
def inspect(ckpt: str, compute_diff: bool = True) -> dict:
    rep = {'ckpt': os.path.abspath(ckpt)}
    cpath = os.path.join(ckpt, 'config.json')
    if not os.path.isfile(cpath):
        rep['status'] = 'no-config'
        return rep
    with open(cpath) as f:
        cfg = json.load(f)
    rep['tie'] = tie_flags(cfg)
    try:
        head_key, emb_key, wm = locate(ckpt)
    except Exception as e:
        rep['status'] = f'no-weights ({type(e).__name__}: {e})'
        return rep
    rep['lm_head_key'], rep['embed_key'] = head_key, emb_key
    if emb_key is None:
        rep['status'] = 'no-embed-key'
        return rep
    if head_key is None:
        rep['status'] = 'tied-single-key'          # genuinely tied: must stay tied
        return rep
    rep['shards'] = {head_key: wm[head_key], emb_key: wm[emb_key]}
    if compute_diff:
        rep['diff'] = diff_stats(ckpt, wm, head_key, emb_key)
    any_tied = any(v is True for v in rep['tie'].values())
    if not any_tied:
        rep['status'] = 'already-untied'
    elif compute_diff and rep['diff'].get('identical'):
        rep['status'] = 'identical'
    else:
        rep['status'] = 'untie-needed'
    return rep


def apply(ckpt: str, rep: dict, even_if_identical: bool = False) -> str:
    st = rep.get('status')
    if st == 'already-untied':
        return 'no-op (already untied)'
    if st == 'identical' and not even_if_identical:
        return 'skipped (tensors identical: tied serving is already correct; --even-if-identical)'
    if st not in ('untie-needed', 'identical'):
        return f'REFUSED ({st})'
    with open(os.path.join(ckpt, 'config.json')) as f:
        cfg = json.load(f)
    bak = write_config(ckpt, untied_config(cfg))
    return f'config.json rewritten (tie_word_embeddings=false at all levels); backup {bak}'


def find_checkpoints(root: str) -> List[str]:
    out = []
    for d, dirs, files in os.walk(root, followlinks=False):
        dirs[:] = sorted(x for x in dirs if not any(x.startswith(s) for s in SKIP_DIRS))
        if 'config.json' in files and ('model.safetensors.index.json' in files
                                       or any(f.endswith('.safetensors') for f in files)):
            out.append(d)
    return out


def _fmt(rep: dict) -> str:
    lines = [f"[untie] {rep['ckpt']}"]
    if 'tie' in rep:
        lines.append('  config: tie_word_embeddings ' + ' '.join(f'{k}={v}' for k, v in rep['tie'].items()))
    if rep.get('lm_head_key') or rep.get('embed_key'):
        lines.append(f"  keys:   lm_head={rep.get('lm_head_key')}  embed={rep.get('embed_key')}")
    d = rep.get('diff')
    if d:
        if d.get('max_abs') is None:
            lines.append(f"  diff:   {d.get('note')} {d['shape_lm_head']} vs {d['shape_embed']}")
        else:
            lines.append(f"  diff:   shape={d['shape_lm_head']} dtype={d['dtype']} "
                         f"max|lm_head-embed|={d['max_abs']:.6g} mean={d['mean_abs']:.6g} "
                         f"(mean|lm_head|={d['mean_abs_lm_head']:.6g}) "
                         f"rows_differ={d['rows_differ']}/{d['rows']}")
    msg = {
        'untie-needed': 'UNTIE NEEDED -- sglang/vLLM currently serve lm_head = embed_tokens, '
                        'not the trained lm_head',
        'already-untied': 'ok (already untied)',
        'identical': 'both keys present but bit-identical -- tied serving is correct',
        'tied-single-key': 'genuinely tied (no lm_head.weight) -- leave tied',
    }.get(rep['status'], rep['status'])
    lines.append(f'  status: {msg}')
    if 'action' in rep:
        lines.append(f"  action: {rep['action']}")
    return '\n'.join(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('ckpt', nargs='*', help='checkpoint dir(s)')
    ap.add_argument('--all-under', action='append', default=[], metavar='DIR',
                    help='walk DIR for checkpoints (dirs with config.json + safetensors)')
    ap.add_argument('--apply', action='store_true', help='rewrite config.json (default: dry-run)')
    ap.add_argument('--no-diff', action='store_true', help='skip max|lm_head-embed| (fast)')
    ap.add_argument('--even-if-identical', action='store_true',
                    help='also untie when both tensors are bit-identical')
    ap.add_argument('--json', help='write the per-checkpoint reports to this JSON file')
    a = ap.parse_args(argv)
    ckpts = list(a.ckpt)
    for root in a.all_under:
        ckpts += find_checkpoints(root)
    if not ckpts:
        ap.error('give a checkpoint dir or --all-under DIR')
    reports = []
    for c in ckpts:
        rep = inspect(c, compute_diff=not a.no_diff)
        if a.apply:
            rep['action'] = apply(c, rep, a.even_if_identical)
        else:
            rep['action'] = ('dry-run (pass --apply to rewrite config.json; backup -> '
                             f'config.json{BACKUP_SUFFIX})' if rep['status'] == 'untie-needed'
                             else 'dry-run')
        reports.append(rep)
        print(_fmt(rep), flush=True)
    n_need = sum(r['status'] == 'untie-needed' for r in reports)
    print(f'[untie] {len(reports)} checkpoint(s): {n_need} need untying'
          + ('' if a.apply else ' (dry-run, nothing written)'))
    if a.json:
        with open(a.json, 'w') as f:
            json.dump(reports, f, indent=2)
    return 0


if __name__ == '__main__':
    sys.exit(main())
