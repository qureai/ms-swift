#!/usr/bin/env python
"""Offline TEACHER SIDECAR for SEM latent-token supervision (spec S3).

For every supervised report turn of the training/validation mixture this computes the frozen
teacher embedding z* ONCE, so training never runs the 4B teacher. Targets come from the SAME
pure module the training plugin uses (swift_plugins/sem_targets.py: T1 families, canonical
"Findings: ...\\nSummary: ...\\nImpression: ..." text, sha1 key), so a training lookup can only
miss when the data changed after the build.

PIPELINE (each stage is resumable; re-running the same command continues where it stopped)
  1. COLLECT  stream every --train_datasets / --val_datasets file line by line (the files are
              GBs; all of the stage-6c ones are JSONL even when named .json -- detected from the
              first byte; a true JSON array is streamed with ijson when installed, else with a
              raw_decode incremental parser), byte-range chunks in parallel (--workers).
              sem_targets.iter_row_targets per row (+ explicit 'sem_targets'/'sem_target' row
              overrides), dedup by key (a key with two different texts is a hard error).
              -> <out>/collect/targets.jsonl (sorted by key: k, t, f=family bits, tr=in_train,
                 n=turn count) + manifest.json (per-family / per-file stats). Reused while the
                 dataset files (size, mtime), RULE_VERSION and --limit are unchanged; guarded by
                 an flock so concurrent shards collect once.
  2. ENCODE   shard i of N takes the keys with key % N == i, sorts them by teacher length and
              encodes length-bucketed batches (<= --batch_tokens padded tokens, <= --max_batch).
              Teacher recipe (map_teacher.md; parity with SentenceTransformer 7e-9):
                AutoTokenizer(padding_side='left'); AutoModel(dtype); text.strip();
                tok(texts, padding=True, truncation=True, max_length=--max_length,
                    add_special_tokens=True)  -> EOT <|endoftext|> 151643 ASSERTED at [:, -1]
                h = last_hidden_state[:, -1];  z = F.normalize(h.float(), dim=-1)
              No instruction prompt. Probe logits (lp23sep head_xray_bce.pt: standardise with the
              SAVED mean/std, linear) are computed in the same pass from the fp32 z, so tags can
              never desynchronise from z. CUDA OOM halves the batch and retries.
              Chunk files parts/part_{i}.cNNNNN.npz (every --chunk_texts) make a killed shard
              resume mid-way; the finished shard is parts/part_{i}.npz + part_{i}.json. An flock
              per part means a re-launched shard waits for a still-running one, then skips.
              Every chunk / part records its PROVENANCE (teacher fingerprint, dtype, max_length,
              probe head path + sha256, RULE_VERSION) and is reused only when that record AND the
              keys equal the current run's: a re-run with another teacher / dtype / max_length /
              probe head re-encodes instead of mixing old rows into a sidecar whose meta says new.
  3. MERGE    (--merge) all N parts -> sort by key, dedup, check coverage against the collection,
              mu = mean over the unique rows, and write the sidecar at <out>/ (layout in the
              sem_targets.py header; meta.json is written LAST, so its presence = complete).

TEACHER CHECKS
  The teacher's fingerprint (sha256 over 16 x 64 KB tensor-byte samples per safetensors shard;
  a COPY of tag_extractor.backbone_fingerprint, not an import of that file) must equal
  heads.json 'backbone_fingerprint' of the probe bundle (bb_23sep == lp23sep: 36741eaf96c44658...),
  else the build refuses (--allow_fingerprint_mismatch to override; recorded in meta either way).
  Paths are resolved across storage roots (/localstorage <-> /fastdata3 <-> /fastdata4).

USAGE
  # tiny CPU smoke (2 files, ~24 texts)
  python build_sem_targets.py --train_datasets A.jsonl --val_datasets B3_VAL.jsonl --out DIR \\
      --limit 24 --device cpu --dtype fp32 --write_texts            # collect + encode shard 0/1
  python build_sem_targets.py ...same... --merge                    # -> sidecar in DIR
  # full build: run_build_sem_targets.sh (collect once, one shard per GPU, merge)
  # text census only (no teacher): --collect_only [--token_stats]

ENV
  SEM_TEACHER_PATH, SEM_PROBE_HEAD   default --teacher / --probe_head (sem_targets.py)
  CUDA_VISIBLE_DEVICES               the shard's GPU (the shell wrapper sets one per shard)
"""
import argparse
import collections
import fcntl
import glob
import hashlib
import json
import logging
import math
import os
import platform
import re
import struct
import sys
import time
from contextlib import contextmanager
from typing import Any, Dict, Iterator, List, Optional, Tuple

import numpy as np

_REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..', '..'))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)
from swift_plugins import sem_targets as st  # noqa: E402  (pure: stdlib + numpy)


def get_logger() -> logging.Logger:
    lg = logging.getLogger('sem_build')
    if not lg.handlers:
        h = logging.StreamHandler(sys.stderr)
        h.setFormatter(logging.Formatter('[%(levelname)s:sem_build %(asctime)s] %(message)s', '%H:%M:%S'))
        lg.addHandler(h)
        lg.setLevel(logging.INFO)
        lg.propagate = False
    return lg


logger = get_logger()

# ============================================================================================
# dataset streaming
# ============================================================================================


def resolve_dataset(spec: str) -> str:
    """swift dataset spec -> local file. A '#N' sampling suffix is dropped: the sidecar must
    cover the WHOLE file (swift samples rows at train time)."""
    p = re.sub(r'#\d+$', '', spec.strip())
    p = st.resolve_storage_path(p)
    if not os.path.isfile(p):
        raise FileNotFoundError(f'[sem-build] dataset file not found: {spec}')
    return p


def detect_format(path: str) -> str:
    """'jsonl' or 'json_array' from the first non-space byte (+ a parse of the first line)."""
    with open(path, 'rb') as f:
        head = f.read(1 << 16)
    s = head.lstrip()
    if not s:
        return 'jsonl'
    if s[:1] == b'[':
        return 'json_array'
    with open(path, 'rb') as f:
        first = f.readline()
    try:
        json.loads(first)
        return 'jsonl'
    except Exception:
        raise ValueError(f'[sem-build] {path}: starts with {s[:1]!r} but its first line is not a JSON object '
                         '(pretty-printed JSON objects are not supported; use JSONL or a JSON array)')


def iter_json_array(path: str, bufsize: int = 1 << 22) -> Iterator[dict]:
    """Stream the elements of a top-level JSON array without loading the file."""
    try:
        import ijson  # type: ignore
        with open(path, 'rb') as f:
            for obj in ijson.items(f, 'item', use_float=True):
                yield obj
        return
    except ImportError:
        pass
    dec = json.JSONDecoder()
    with open(path, 'r', encoding='utf-8') as f:
        buf = f.read(bufsize)
        i = buf.find('[')
        if i < 0:
            raise ValueError(f'[sem-build] {path}: not a JSON array')
        buf = buf[i + 1:]
        eof = False
        while True:
            buf = buf.lstrip(' \t\r\n,')
            if buf.startswith(']'):
                return
            if not buf:
                if eof:
                    return
                more = f.read(bufsize)
                eof = not more
                buf += more
                continue
            try:
                obj, end = dec.raw_decode(buf)
            except json.JSONDecodeError:
                if eof:
                    raise
                more = f.read(bufsize)
                eof = not more
                buf += more
                continue
            yield obj
            buf = buf[end:]


BAD_LINES: List[Tuple[str, int]] = []      # (path, offset) of unparsable lines seen by this process


def iter_jsonl_range(path: str, start: int, end: int) -> Iterator[Tuple[int, dict]]:
    """(line offset, row) for lines STARTING in [start, end]; unparsable lines are skipped and
    recorded in BAD_LINES (reported by collect; swift would fail on them at train time)."""
    with open(path, 'rb') as f:
        if start > 0:
            f.seek(start - 1)
            f.readline()                        # finish the line that straddles `start`
        while True:
            off = f.tell()
            if off > end:
                break
            line = f.readline()
            if not line:
                break
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except ValueError:
                BAD_LINES.append((path, off))
                continue
            yield off, row


def iter_jsonl_spread(path: str, n: int) -> Iterator[dict]:
    """--limit sampling: the first complete line after each of `n` evenly spaced byte offsets
    (deterministic, whole-file coverage -- several training files are sorted by template)."""
    size = os.path.getsize(path)
    seen = set()
    with open(path, 'rb') as f:
        for i in range(n):
            pos = size * i // n
            f.seek(max(0, pos - 1))
            if pos > 0:
                f.readline()
            off = f.tell()
            if off in seen:
                continue
            line = f.readline()
            if not line.strip():
                continue
            seen.add(off)
            try:
                row = json.loads(line)
            except ValueError:
                BAD_LINES.append((path, off))
                continue
            yield row


def raw_answer_key(content: Any) -> int:
    """sha1 of the cleaned RAW answer (what the data-reader census counted), for reporting."""
    t = st.clean_assistant_text(content)
    return int.from_bytes(hashlib.sha1(t.encode('utf-8')).digest()[:8], 'big')


def _collect_rows(rows: Iterator[dict], max_rows: Optional[int]) -> Dict[str, Any]:
    targets: Dict[int, list] = {}
    raw_keys: Dict[str, set] = collections.defaultdict(set)
    c = collections.Counter()
    for row in rows:
        c['rows'] += 1
        tg = st.iter_row_targets(row)
        if tg:
            c['rows_with_target'] += 1
        msgs = None
        for i, fam, text in tg:
            c[f'turns:{fam}'] += 1
            k = st.target_key(text)
            e = targets.get(k)
            if e is None:
                targets[k] = [text, st.FAMILY_BITS[fam], 1]
            else:
                if e[0] != text:
                    raise RuntimeError(f'[sem-build] sha1[:8] key collision: {k} for two different texts')
                e[1] |= st.FAMILY_BITS[fam]
                e[2] += 1
            if fam != st.OVERRIDE_FAMILY and i >= 0:
                msgs = msgs if msgs is not None else st.row_messages(row)
                raw_keys[fam].add(raw_answer_key(msgs[i]['content']))
        c[f'targets_per_row:{min(len(tg), 5)}'] += 1
        if max_rows is not None and c['rows'] >= max_rows:
            break
    return {'targets': targets, 'counts': c, 'raw_keys': dict(raw_keys)}


def _limit_subset(res: Dict[str, Any], cap: int) -> Dict[str, Any]:
    """--limit: a deterministic FAMILY-BALANCED subset of `cap` targets (round-robin over the
    families present, keys ascending inside a family), so a smoke run touches every family."""
    by_fam = collections.defaultdict(list)
    for k, (_, bits, _) in sorted(res['targets'].items()):
        by_fam[(bits & -bits).bit_length()].append(k)      # lowest family bit
    pick, queues = [], [by_fam[f] for f in sorted(by_fam)]
    while len(pick) < cap and any(queues):
        for q in queues:
            if q and len(pick) < cap:
                pick.append(q.pop(0))
    res = dict(res)
    res['targets'] = {k: res['targets'][k] for k in pick}
    return res


def _collect_job(job: Tuple[str, str, str, int, int, Optional[int], Optional[int]]) -> Tuple[str, str, Dict[str, Any]]:
    path, role, fmt, start, end, max_rows, cap = job
    if fmt == 'json_array':
        rows = iter_json_array(path)
    elif cap is not None and max_rows:
        rows = iter_jsonl_spread(path, max_rows)
    else:
        rows = (r for _, r in iter_jsonl_range(path, start, end))
    del BAD_LINES[:]
    res = _collect_rows(rows, max_rows)
    res['counts']['bad_lines'] += len(BAD_LINES)
    if BAD_LINES:
        res['bad_lines'] = list(BAD_LINES[:20])
    return path, role, (res if cap is None else _limit_subset(res, cap))


def dataset_fingerprint(paths_roles: List[Tuple[str, str]]) -> List[Dict[str, Any]]:
    out = []
    for p, role in paths_roles:
        s = os.stat(p)
        out.append({'path': p, 'role': role, 'size': s.st_size, 'mtime': int(s.st_mtime), 'format': detect_format(p)})
    return out


@contextmanager
def flocked(path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'a+') as fh:
        fcntl.flock(fh.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(fh.fileno(), fcntl.LOCK_UN)


def collect(args) -> Dict[str, Any]:
    """Stage 1 -> {'keys','texts','bits','in_train','count','manifest'} (from cache when valid)."""
    cdir = os.path.join(args.out, 'collect')
    tpath, mpath = os.path.join(cdir, 'targets.jsonl'), os.path.join(cdir, 'manifest.json')
    pr = [(resolve_dataset(p), 'train') for p in args.train_datasets] + \
         [(resolve_dataset(p), 'val') for p in (args.val_datasets or [])]
    if not pr:
        raise ValueError('[sem-build] no datasets given')
    want = {'rule_version': st.RULE_VERSION, 'limit': args.limit,
            'limit_scan_rows': args.limit_scan_rows if args.limit is not None else None,
            'datasets': dataset_fingerprint(pr)}
    with flocked(os.path.join(cdir, '.lock')):
        if os.path.isfile(mpath) and os.path.isfile(tpath):
            with open(mpath) as f:
                man = json.load(f)
            if all(man.get(k) == v for k, v in want.items()):
                logger.info(f'[sem-build] collect: reusing {tpath} ({man["n_unique"]} unique targets)')
                return _load_collect(tpath, man)
            logger.info('[sem-build] collect: cache is stale (datasets / rule_version / limit changed) -> recollecting')
        t0 = time.time()
        jobs = []
        cap_per_file = None if args.limit is None else max(1, math.ceil(args.limit / len(pr)))
        for (p, role), d in zip(pr, want['datasets']):
            if args.limit is not None:
                jobs.append((p, role, d['format'], 0, d['size'], args.limit_scan_rows, cap_per_file))
            elif d['format'] == 'json_array':
                jobs.append((p, role, d['format'], 0, d['size'], None, None))
            else:
                k = max(1, d['size'] // (args.chunk_mb << 20))
                for c in range(k):
                    jobs.append((p, role, 'jsonl', c * d['size'] // k,
                                 (c + 1) * d['size'] // k - 1 if c < k - 1 else d['size'], None, None))
        logger.info(f'[sem-build] collect: {len(pr)} files, {len(jobs)} jobs, workers={args.workers}'
                    + (f', limit={args.limit} ({cap_per_file}/file, family-balanced from '
                       f'{args.limit_scan_rows} rows spread over each file)' if args.limit is not None else ''))
        merged: Dict[int, list] = {}
        per_file = collections.defaultdict(collections.Counter)
        totals = collections.Counter()
        raw_keys = {'train': collections.defaultdict(set), 'val': collections.defaultdict(set)}

        def absorb(path, role, res):
            for p_, off in res.get('bad_lines', ()):
                logger.warning(f'[sem-build] collect: unparsable JSON line skipped: {p_} @ byte {off}')
            per_file[path].update(res['counts'])
            totals.update(res['counts'])
            for fam, ks in res['raw_keys'].items():
                raw_keys[role][fam] |= ks
            for k, (text, bits, n) in res['targets'].items():
                e = merged.get(k)
                if e is None:
                    merged[k] = [text, bits, n, role == 'train', role == 'val']
                else:
                    if e[0] != text:
                        raise RuntimeError(f'[sem-build] sha1[:8] key collision: {k}')
                    e[1] |= bits
                    e[2] += n
                    e[3] = e[3] or role == 'train'
                    e[4] = e[4] or role == 'val'

        if args.limit is not None or args.workers <= 1 or len(jobs) == 1:
            for j in jobs:
                absorb(*_collect_job(j))
        else:
            from multiprocessing import get_context
            with get_context('fork').Pool(min(args.workers, len(jobs))) as pool:
                for done, res in enumerate(pool.imap_unordered(_collect_job, jobs), 1):
                    absorb(*res)
                    if done % 50 == 0 or done == len(jobs):
                        logger.info(f'[sem-build] collect: {done}/{len(jobs)} jobs, {len(merged)} unique targets')
        keys = sorted(merged)
        os.makedirs(cdir, exist_ok=True)
        with open(tpath + '.tmp', 'w', encoding='utf-8') as f:
            for k in keys:
                t, b, n, tr, _ = merged[k]
                f.write(json.dumps({'k': str(k), 't': t, 'f': b, 'tr': int(tr), 'n': n}, ensure_ascii=False) + '\n')
        os.replace(tpath + '.tmp', tpath)
        man = dict(want)
        man.update(collect_stats(merged, totals, per_file, raw_keys))
        man['seconds'] = round(time.time() - t0, 1)
        man['created'] = time.strftime('%Y-%m-%dT%H:%M:%S%z')
        with open(mpath + '.tmp', 'w') as f:
            json.dump(man, f, indent=1)
        os.replace(mpath + '.tmp', mpath)
        logger.info(f'[sem-build] collect: {man["n_unique"]} unique targets from {totals["rows"]} rows '
                    f'in {man["seconds"]}s -> {tpath}')
        return _load_collect(tpath, man)


def collect_stats(merged, totals, per_file, raw_keys) -> Dict[str, Any]:
    fams = list(st.FAMILY_BITS)
    uniq = {f: 0 for f in fams}
    uniq_train = {f: 0 for f in fams}
    multi = 0
    n_train = n_val = n_val_seen = 0
    for text, bits, n, tr, va in merged.values():
        n_train += bool(tr)
        n_val += bool(va)
        n_val_seen += bool(va and tr)
        multi += bin(bits).count('1') > 1
        for f in fams:
            if bits & st.FAMILY_BITS[f]:
                uniq[f] += 1
                uniq_train[f] += bool(tr)
    raw_train = {f: len(v) for f, v in raw_keys['train'].items()}
    raw_all = {f: len(raw_keys['train'].get(f, set()) | raw_keys['val'].get(f, set()))
               for f in set(raw_keys['train']) | set(raw_keys['val'])}
    raw_union_train = len(set().union(*raw_keys['train'].values())) if raw_keys['train'] else 0
    return {
        'n_unique': len(merged), 'n_unique_in_train': n_train, 'n_unique_val_only': len(merged) - n_train,
        'n_unique_in_val': n_val, 'n_unique_val_seen_in_train': n_val_seen,
        'n_texts_per_family': uniq, 'n_texts_per_family_in_train': uniq_train,
        'n_texts_multi_family': multi,
        'n_turns_per_family': {f: totals.get(f'turns:{f}', 0) for f in fams},
        'rows': totals.get('rows', 0), 'rows_with_target': totals.get('rows_with_target', 0),
        'bad_lines': totals.get('bad_lines', 0),
        'targets_per_row': {k.split(':')[1]: v for k, v in sorted(totals.items()) if k.startswith('targets_per_row:')},
        'raw_unique_answers_per_family_train': raw_train, 'raw_unique_answers_per_family_all': raw_all,
        'raw_unique_answers_train_union': raw_union_train,
        'per_file': {p: {'rows': c.get('rows', 0), 'rows_with_target': c.get('rows_with_target', 0),
                         'turns': {f: c.get(f'turns:{f}', 0) for f in fams if c.get(f'turns:{f}', 0)}}
                     for p, c in per_file.items()},
    }


def _load_collect(tpath: str, man: Dict[str, Any]) -> Dict[str, Any]:
    keys, texts, bits, tr, cnt = [], [], [], [], []
    with open(tpath, encoding='utf-8') as f:
        for line in f:
            r = json.loads(line)
            keys.append(int(r['k']))
            texts.append(r['t'])
            bits.append(r['f'])
            tr.append(r['tr'])
            cnt.append(r['n'])
    return {'keys': np.asarray(keys, dtype=np.uint64), 'texts': texts, 'bits': np.asarray(bits, np.uint8),
            'in_train': np.asarray(tr, np.uint8), 'count': np.asarray(cnt, np.int64), 'manifest': man}


# ============================================================================================
# teacher
# ============================================================================================


def backbone_fingerprint(model_dir: str, samples: int = 16, chunk: int = 1 << 16) -> Optional[str]:
    """COPY of tag_extractor.backbone_fingerprint (lp23sep/tag_extractor.py): sha256 over 16 x
    64 KB of tensor bytes sampled evenly through every *.safetensors shard (sorted), JSON
    header skipped. Tells apart consecutive checkpoints of one run. None without shards."""
    shards = sorted(glob.glob(os.path.join(model_dir, '*.safetensors')))
    if not shards:
        return None
    h = hashlib.sha256()
    for s in shards:
        size = os.path.getsize(s)
        with open(s, 'rb') as f:
            start = 8 + struct.unpack('<Q', f.read(8))[0]
            span = size - start - chunk
            for i in range(samples):
                f.seek(start + span * i // (samples - 1))
                h.update(f.read(chunk))
    return h.hexdigest()


def check_teacher(args) -> Dict[str, Any]:
    fp = backbone_fingerprint(args.teacher)
    heads_json = args.heads_json
    if heads_json is None and args.probe_head:
        heads_json = os.path.join(os.path.dirname(args.probe_head), 'heads.json')
    heads_json = st.resolve_storage_path(heads_json) if heads_json else None
    expected = None
    if heads_json and os.path.isfile(heads_json):
        with open(heads_json) as f:
            expected = json.load(f).get('backbone_fingerprint')
    ok = expected is not None and fp == expected
    info = {'teacher_fingerprint': fp, 'fingerprint_expected': expected, 'fingerprint_ok': ok,
            'heads_json': heads_json}
    if ok:
        logger.info(f'[sem-build] teacher fingerprint ok: {fp[:16]} == heads.json ({heads_json})')
    elif expected is None:
        msg = f'[sem-build] no heads.json backbone_fingerprint to check the teacher against ({heads_json})'
        if not args.allow_fingerprint_mismatch:
            raise SystemExit(msg + ' -- pass --heads_json or --allow_fingerprint_mismatch')
        logger.warning(msg)
    else:
        msg = (f'[sem-build] TEACHER FINGERPRINT MISMATCH: {args.teacher} -> {fp} but {heads_json} expects '
               f'{expected}: the probe head was fitted on a different backbone')
        if not args.allow_fingerprint_mismatch:
            raise SystemExit(msg + ' (--allow_fingerprint_mismatch to override)')
        logger.warning(msg)
    return info


class Teacher:
    """bb_23sep embedder + (optional) lp23sep linear probe, one forward per batch."""

    def __init__(self, args):
        import torch
        import transformers
        from transformers import AutoModel, AutoTokenizer
        try:
            transformers.utils.logging.disable_progress_bar()      # no per-tensor tqdm in shard logs
        except Exception:
            pass
        self.torch = torch
        self.max_length = args.max_length
        if args.device == 'cpu':
            torch.set_num_threads(args.threads)
        self.device = torch.device(args.device)
        self.dtype = {'bf16': torch.bfloat16, 'fp32': torch.float32, 'fp16': torch.float16}[args.dtype]
        self.tok = AutoTokenizer.from_pretrained(args.teacher, padding_side='left')
        self.tok.padding_side = 'left'
        self.eot = self.tok.convert_tokens_to_ids('<|endoftext|>')
        if self.eot != st.TEACHER_EOT_ID:
            logger.warning(f'[sem-build] teacher EOT id {self.eot} != expected {st.TEACHER_EOT_ID}')
        t0 = time.time()
        try:
            self.model = AutoModel.from_pretrained(args.teacher, dtype=self.dtype, attn_implementation='sdpa')
        except TypeError:                         # transformers < 4.56
            self.model = AutoModel.from_pretrained(args.teacher, torch_dtype=self.dtype, attn_implementation='sdpa')
        self.model.to(self.device).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        logger.info(f'[sem-build] teacher {type(self.model).__name__} {args.dtype} on {self.device} '
                    f'loaded in {time.time() - t0:.0f}s (hidden {self.model.config.hidden_size})')
        self.probe = None
        if args.probe_head:
            self.probe = self._load_probe(args.probe_head)

    def _load_probe(self, path: str) -> Dict[str, Any]:
        torch = self.torch
        try:
            ck = torch.load(path, map_location='cpu', weights_only=True)
        except Exception:
            ck = torch.load(path, map_location='cpu', weights_only=False)   # trusted local bundle
        sd = ck['state_dict']
        W, b = sd['fc.weight'].float(), sd['fc.bias'].float()
        kind = (ck.get('head_cfg') or {}).get('kind', 'linear')
        if kind != 'linear':
            raise SystemExit(f'[sem-build] probe head kind {kind!r} is not linear')
        std = ck.get('standardize', False)
        mean = torch.as_tensor(np.asarray(ck['mean'], dtype=np.float32)).reshape(1, -1) if std else None
        sd_ = torch.as_tensor(np.asarray(ck['std'], dtype=np.float32)).reshape(1, -1) if std else None
        tags = list(ck.get('tags') or [])
        thr = ck.get('thresholds') or {}
        logger.info(f'[sem-build] probe {path}: {W.shape[0]} tags, standardize={bool(std)}')
        dev = self.device
        return {'path': path, 'W': W.to(dev), 'b': b.to(dev), 'mean': None if mean is None else mean.to(dev),
                'std': None if sd_ is None else sd_.to(dev), 'standardize': bool(std), 'tags': tags,
                'thresholds': {k: float(v) for k, v in thr.items()}, 'n_tags': int(W.shape[0])}

    def token_lengths(self, texts: List[str]) -> np.ndarray:
        out = []
        for i in range(0, len(texts), 4096):
            enc = self.tok(texts[i:i + 4096], add_special_tokens=True, truncation=False)['input_ids']
            out.extend(len(x) for x in enc)
        return np.asarray(out, dtype=np.int32)

    def encode(self, texts: List[str]) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """-> (z float32 [B,2560] unit-norm, probe logits float32 [B,T] or None)."""
        torch = self.torch
        import torch.nn.functional as F
        texts = [t.strip() for t in texts]
        enc = self.tok(texts, padding=True, truncation=True, max_length=self.max_length,
                       add_special_tokens=True, return_tensors='pt')
        ids, am = enc['input_ids'], enc['attention_mask']
        if not bool((ids[:, -1] == self.eot).all()) or not bool((am[:, -1] == 1).all()):
            raise RuntimeError('[sem-build] teacher input does not end in <|endoftext|> for every row '
                               '(left padding / add_special_tokens broken): targets would be wrong')
        try:
            with torch.inference_mode():
                out = self.model(input_ids=ids.to(self.device), attention_mask=am.to(self.device))
                z = F.normalize(out.last_hidden_state[:, -1].float(), dim=-1)
                if not bool(torch.isfinite(z).all()):
                    raise RuntimeError(f'[sem-build] non-finite teacher embedding in a batch of {len(texts)} x '
                                       f'{ids.shape[1]} (attention kernel / padding problem): refusing to write it')
                lg = None
                if self.probe is not None:
                    p = self.probe
                    x = (z - p['mean']) / p['std'] if p['standardize'] else z
                    lg = x @ p['W'].T + p['b']
        except torch.cuda.OutOfMemoryError:
            if len(texts) == 1:
                raise
            torch.cuda.empty_cache()
            h = len(texts) // 2
            logger.warning(f'[sem-build] CUDA OOM at batch {len(texts)} x {ids.shape[1]} -> splitting')
            z1, l1 = self.encode(texts[:h])
            z2, l2 = self.encode(texts[h:])
            return np.concatenate([z1, z2]), (None if l1 is None else np.concatenate([l1, l2]))
        return z.cpu().numpy(), (None if lg is None else lg.float().cpu().numpy())


def plan_batches(lengths: np.ndarray, max_length: int, batch_tokens: int, max_batch: int) -> List[np.ndarray]:
    """Length-bucketed batches over indices sorted by (truncated) length, longest first."""
    eff = np.minimum(lengths, max_length)
    order = np.lexsort((np.arange(len(eff)), -eff))
    batches, cur, cur_max = [], [], 0
    for i in order:
        L = int(eff[i])
        m = max(cur_max, L)
        if cur and ((len(cur) + 1) * m > batch_tokens or len(cur) >= max_batch):
            batches.append(np.asarray(cur))
            cur, m = [], L
        cur.append(i)
        cur_max = m
    if cur:
        batches.append(np.asarray(cur))
    return batches


def _file_sha256(path: Optional[str]) -> Optional[str]:
    if not path or not os.path.isfile(path):
        return None
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def encode_provenance(args, teacher_info: Dict[str, Any]) -> Dict[str, Any]:
    """Everything that changes an embedding row. A chunk / part file is reused ONLY when its record equals
    the current one (a re-run with another teacher, dtype, max_length, probe head or rule re-encodes)."""
    return {'teacher_fingerprint': teacher_info.get('teacher_fingerprint'), 'dtype': args.dtype,
            'max_length': int(args.max_length), 'probe_head': args.probe_head or None,
            'probe_sha256': _file_sha256(args.probe_head), 'rule_version': st.RULE_VERSION}


def _prov_json(prov: Dict[str, Any]) -> str:
    return json.dumps(prov, sort_keys=True)


def encode_shard(args, col: Dict[str, Any], teacher_info: Dict[str, Any]) -> str:
    pdir = os.path.join(args.out, 'parts')
    os.makedirs(pdir, exist_ok=True)
    i, N = args.shard_id, args.num_shards
    part, pjson = os.path.join(pdir, f'part_{i}.npz'), os.path.join(pdir, f'part_{i}.json')
    keys_all = col['keys']
    sel = np.nonzero(keys_all % np.uint64(N) == np.uint64(i))[0]
    keys = keys_all[sel]
    prov = encode_provenance(args, teacher_info)
    pjs_ = _prov_json(prov)
    with flocked(os.path.join(pdir, f'part_{i}.lock')):
        if os.path.isfile(part) and os.path.isfile(pjson):
            with open(pjson) as f:
                pj = json.load(f)
            with np.load(part) as z:
                same = z['keys'].shape == keys.shape and bool(np.array_equal(z['keys'], keys))
            same_prov = _prov_json({k: pj.get(k) for k in prov}) == pjs_
            if same and same_prov and pj.get('num_shards') == N:
                logger.info(f'[sem-build] shard {i}/{N}: {part} already complete ({len(keys)} texts) -> skip')
                return part
            why = ('keys differ' if not same else 'num_shards differs' if same_prov else
                   f'provenance differs: {({k: (pj.get(k), v) for k, v in prov.items() if pj.get(k) != v})}')
            logger.info(f'[sem-build] shard {i}/{N}: existing part is stale ({why}) -> re-encoding')
        texts = [col['texts'][j] for j in sel]
        t0 = time.time()
        teacher = Teacher(args)
        lengths = teacher.token_lengths(texts)
        batches = plan_batches(lengths, args.max_length, args.batch_tokens, args.max_batch)
        # chunks = groups of consecutive batches (deterministic given selection + batch knobs)
        chunks, cur, cn = [], [], 0
        for b in batches:
            cur.append(b)
            cn += len(b)
            if cn >= args.chunk_texts:
                chunks.append(cur)
                cur, cn = [], 0
        if cur:
            chunks.append(cur)
        ntrunc = int((lengths > args.max_length).sum())
        logger.info(f'[sem-build] shard {i}/{N}: {len(texts)} texts, {int(np.minimum(lengths, args.max_length).sum())} '
                    f'tokens, {len(batches)} batches in {len(chunks)} chunks, {ntrunc} truncated at {args.max_length} '
                    f'(max {int(lengths.max()) if len(lengths) else 0})')
        E = np.zeros((len(texts), st.EMB_DIM), dtype=np.float16)
        P = None
        done_tok = 0
        for cj, ch in enumerate(chunks):
            idx = np.concatenate(ch)
            cpath = os.path.join(pdir, f'part_{i}.c{cj:05d}.npz')
            if os.path.isfile(cpath):
                with np.load(cpath) as z:
                    c_prov = str(z['prov']) if 'prov' in z.files else None
                    if c_prov != pjs_:
                        logger.info(f'[sem-build] shard {i}/{N}: chunk {cj + 1}/{len(chunks)} at {cpath} has another '
                                    f'provenance ({c_prov}) -> re-encoding it')
                    elif np.array_equal(z['keys'], keys[idx]):
                        logger.info(f'[sem-build] shard {i}/{N}: chunk {cj + 1}/{len(chunks)} reused from {cpath}')
                        E[idx] = z['emb']
                        if 'probe' in z.files:
                            P = P if P is not None else np.zeros((len(texts), z['probe'].shape[1]), np.float16)
                            P[idx] = z['probe']
                        continue
            ce, cp = [], []
            for b in ch:
                z, lg = teacher.encode([texts[j] for j in b])
                ce.append(z)
                cp.append(lg)
                done_tok += int(np.minimum(lengths[b], args.max_length).sum())
            ze = np.concatenate(ce).astype(np.float16)
            E[idx] = ze
            extra = {}
            if cp[0] is not None:
                zp = np.concatenate(cp).astype(np.float16)
                P = P if P is not None else np.zeros((len(texts), zp.shape[1]), np.float16)
                P[idx] = zp
                extra['probe'] = zp
            np.savez(cpath + '.tmp.npz', keys=keys[idx], emb=ze, prov=np.array(pjs_), **extra)
            os.replace(cpath + '.tmp.npz', cpath)
            el = time.time() - t0
            logger.info(f'[sem-build] shard {i}/{N}: chunk {cj + 1}/{len(chunks)} ({len(idx)} texts), '
                        f'{done_tok / max(el, 1e-6):.0f} tok/s')
            if args.stop_after_chunks and cj + 1 >= args.stop_after_chunks and cj + 1 < len(chunks):
                logger.warning(f'[sem-build] shard {i}/{N}: --stop_after_chunks {args.stop_after_chunks} -> exiting '
                               '(test hook; re-run to resume from the chunk files)')
                raise SystemExit(3)
        extra = {} if P is None else {'probe': P}
        np.savez(part + '.tmp.npz', keys=keys, emb=E, ntok=lengths, **extra)
        os.replace(part + '.tmp.npz', part)
        import torch
        import transformers
        pj = {'shard_id': i, 'num_shards': N, 'n': len(texts), 'n_truncated': ntrunc,
              'tokens': int(np.minimum(lengths, args.max_length).sum()), 'seconds': round(time.time() - t0, 1),
              'device': args.device, 'dtype': args.dtype, 'max_length': args.max_length,
              'batch_tokens': args.batch_tokens, 'max_batch': args.max_batch, 'teacher_path': args.teacher,
              'probe_head': args.probe_head, 'rule_version': st.RULE_VERSION, 'host': platform.node(),
              'torch': torch.__version__, 'transformers': transformers.__version__, **teacher_info, **prov}
        if teacher.probe is not None:
            pj.update({'probe_tags': teacher.probe['tags'], 'probe_thresholds': teacher.probe['thresholds'],
                       'probe_standardize': teacher.probe['standardize']})
        with open(pjson + '.tmp', 'w') as f:
            json.dump(pj, f, indent=1)
        os.replace(pjson + '.tmp', pjson)
        for c in glob.glob(os.path.join(pdir, f'part_{i}.c*.npz')):
            os.remove(c)
        logger.info(f'[sem-build] shard {i}/{N}: wrote {part} in {pj["seconds"]}s')
        return part


# ============================================================================================
# merge
# ============================================================================================


def merge(args, col: Dict[str, Any]) -> Dict[str, Any]:
    if os.path.isfile(os.path.join(args.out, st.SIDECAR_FILES['meta'])) and not args.overwrite:
        logger.info(f'[sem-build] merge: sidecar already complete at {args.out} (--overwrite to rebuild)')
        with open(os.path.join(args.out, st.SIDECAR_FILES['meta'])) as f:
            return json.load(f)
    pdir = os.path.join(args.out, 'parts')
    N = args.num_shards
    if not os.path.isfile(os.path.join(pdir, f'part_0.json')):
        raise SystemExit(f'[sem-build] merge: no parts in {pdir}')
    with open(os.path.join(pdir, 'part_0.json')) as f:
        N_rec = json.load(f)['num_shards']
    if N != N_rec:
        logger.info(f'[sem-build] merge: parts were built with num_shards={N_rec} (arg {N}) -> using {N_rec}')
        N = N_rec
    pjs, K, E, P, T = [], [], [], [], []
    for i in range(N):
        pj, pz = os.path.join(pdir, f'part_{i}.json'), os.path.join(pdir, f'part_{i}.npz')
        if not (os.path.isfile(pj) and os.path.isfile(pz)):
            raise SystemExit(f'[sem-build] merge: part {i}/{N} missing ({pz}) -- run that shard first')
        with open(pj) as f:
            pjs.append(json.load(f))
        with np.load(pz) as z:
            K.append(z['keys'])
            E.append(z['emb'])
            T.append(z['ntok'])
            P.append(z['probe'] if 'probe' in z.files else None)
    for k in ('max_length', 'dtype', 'teacher_fingerprint', 'rule_version', 'num_shards', 'probe_head', 'probe_sha256'):
        vals = {json.dumps(p.get(k)) for p in pjs}
        if len(vals) != 1:
            raise SystemExit(f'[sem-build] merge: parts disagree on {k}: {vals}')
    if pjs[0]['rule_version'] != st.RULE_VERSION:
        raise SystemExit(f'[sem-build] merge: parts built under rule {pjs[0]["rule_version"]} != {st.RULE_VERSION}')
    has_probe = all(p is not None for p in P)
    keys = np.concatenate(K)
    emb = np.concatenate(E)
    ntok = np.concatenate(T)
    probe = np.concatenate(P) if has_probe else None
    # coverage vs the collection
    ck = col['keys']
    pos = np.searchsorted(ck, keys)
    known = (pos < len(ck)) & (ck[np.minimum(pos, len(ck) - 1)] == keys)
    if not known.all():
        raise SystemExit(f'[sem-build] merge: {int((~known).sum())} part keys are not in the collection '
                         '(stale parts from another dataset list?) -- delete parts/ and re-encode')
    missing = len(np.setdiff1d(ck, keys))
    if missing and not args.allow_missing:
        raise SystemExit(f'[sem-build] merge: {missing} collected keys have no embedding (--allow_missing to accept)')
    in_train = col['in_train'][pos]
    bits = col['bits'][pos]
    texts = [col['texts'][j] for j in pos] if args.write_texts else None
    ml = pjs[0]['max_length']
    ntr = int((ntok > ml).sum())
    man = col['manifest']
    meta = {
        'rule_version': st.RULE_VERSION, 'teacher_path': pjs[0]['teacher_path'],
        'teacher_fingerprint': pjs[0]['teacher_fingerprint'], 'fingerprint_expected': pjs[0]['fingerprint_expected'],
        'fingerprint_ok': pjs[0]['fingerprint_ok'], 'max_length': ml, 'dtype': pjs[0]['dtype'],
        'teacher_recipe': {'padding_side': 'left', 'add_special_tokens': True, 'eot_id': st.TEACHER_EOT_ID,
                           'pooling': 'last_hidden_state[:, -1]', 'normalize': 'F.normalize(fp32)',
                           'strip': True, 'prompt': None},
        'datasets': {'train': [d for d in man['datasets'] if d['role'] == 'train'],
                     'val': [d for d in man['datasets'] if d['role'] == 'val']},
        'limit': man.get('limit'),
        'probe_head': pjs[0]['probe_head'] if has_probe else None,
        'probe_tags': pjs[0].get('probe_tags') if has_probe else None,
        'probe_thresholds': pjs[0].get('probe_thresholds') if has_probe else None,
        'probe_standardize': pjs[0].get('probe_standardize') if has_probe else None,
        'probe_logits': 'raw linear logits (sigmoid -> prob; pred = prob >= probe_thresholds[tag])' if has_probe else None,
        'truncation_rate': ntr / max(1, len(ntok)), 'n_truncated': ntr,
        'ntok_p50': float(np.percentile(ntok, 50)) if len(ntok) else None,
        'ntok_max': int(ntok.max()) if len(ntok) else None,
        'n_missing': int(missing), 'num_shards': N,
        'collect': {k: man.get(k) for k in ('n_unique', 'n_unique_in_train', 'n_unique_in_val',
                                           'n_unique_val_seen_in_train', 'n_turns_per_family',
                                           'n_texts_per_family', 'rows', 'rows_with_target')},
        'encode_seconds': sum(p['seconds'] for p in pjs), 'encode_tokens': sum(p['tokens'] for p in pjs),
        'host': platform.node(), 'created': time.strftime('%Y-%m-%dT%H:%M:%S%z'),
    }
    meta = st.write_sidecar(args.out, keys, emb, in_train, meta, probe_logits=probe, families=bits, ntok=ntok,
                            texts=texts, overwrite=args.overwrite)
    logger.info(f'[sem-build] merge: sidecar {args.out}: n={meta["n"]} dim={meta["dim"]} |mu|={meta["mu_norm"]:.3f} '
                f'truncated {ntr} ({100 * meta["truncation_rate"]:.3f}%) in_train {meta["n_in_train"]} '
                f'probe={has_probe} texts={texts is not None}')
    return meta


# ============================================================================================
# CLI
# ============================================================================================


def report_collect(col: Dict[str, Any], args) -> None:
    man = col['manifest']
    print(json.dumps({k: man[k] for k in ('n_unique', 'n_unique_in_train', 'n_unique_val_only', 'n_unique_in_val',
                                          'n_unique_val_seen_in_train', 'n_texts_per_family',
                                          'n_texts_per_family_in_train', 'n_texts_multi_family', 'n_turns_per_family',
                                          'rows', 'rows_with_target', 'targets_per_row',
                                          'raw_unique_answers_per_family_train', 'raw_unique_answers_train_union',
                                          'seconds') if k in man}, indent=1))
    if args.token_stats and len(col['texts']):
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(args.teacher, padding_side='left')
        L = []
        for i in range(0, len(col['texts']), 8192):
            L.extend(len(x) for x in tok(col['texts'][i:i + 8192], add_special_tokens=True)['input_ids'])
        L = np.asarray(L)
        bits = col['bits']
        out = {'all': L}
        for f, b in st.FAMILY_BITS.items():
            m = (bits & b) != 0
            if m.any():
                out[f] = L[m]
        print('teacher tokens (EOT incl.) per unique text:')
        for f, x in out.items():
            print(f'  {f:15s} n={len(x):7d} p50={np.percentile(x, 50):6.0f} p90={np.percentile(x, 90):6.0f} '
                  f'p99={np.percentile(x, 99):6.0f} max={x.max():6d} >512={(x > 512).mean():.4f} '
                  f'>{args.max_length}={(x > args.max_length).mean():.5f}')


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--train_datasets', nargs='+', required=True)
    ap.add_argument('--val_datasets', nargs='*', default=[])
    ap.add_argument('--out', required=True, help='sidecar directory (collect/ and parts/ live inside it)')
    ap.add_argument('--teacher', default=None, help=f'default $SEM_TEACHER_PATH or {st.DEFAULT_TEACHER}')
    ap.add_argument('--probe_head', default=None, help=f"default $SEM_PROBE_HEAD or {st.DEFAULT_PROBE_HEAD}; 'none' = no probe")
    ap.add_argument('--heads_json', default=None, help='fingerprint reference (default: heads.json beside the probe head)')
    ap.add_argument('--max_length', type=int, default=st.TEACHER_MAX_LENGTH)
    ap.add_argument('--batch_tokens', type=int, default=32768, help='max padded tokens per teacher batch')
    ap.add_argument('--max_batch', type=int, default=256)
    ap.add_argument('--chunk_texts', type=int, default=4096, help='resume granularity inside a shard')
    ap.add_argument('--dtype', choices=['bf16', 'fp32', 'fp16'], default='bf16')
    ap.add_argument('--device', default='cuda', help='cuda | cuda:N | cpu')
    ap.add_argument('--threads', type=int, default=min(64, os.cpu_count() or 1), help='torch CPU threads (--device cpu)')
    ap.add_argument('--num_shards', type=int, default=1)
    ap.add_argument('--shard_id', type=int, default=0)
    ap.add_argument('--merge', action='store_true', help='merge all parts into the sidecar (no encoding)')
    ap.add_argument('--collect_only', action='store_true', help='text collection + stats only (no teacher)')
    ap.add_argument('--token_stats', action='store_true', help='with --collect_only: teacher token-length stats')
    ap.add_argument('--limit', type=int, default=None,
                    help='smoke: cap on unique texts (split evenly over files, family-balanced)')
    ap.add_argument('--limit_scan_rows', type=int, default=3000,
                    help='--limit: rows sampled per file (evenly spaced byte offsets; first rows for JSON arrays)')
    ap.add_argument('--write_texts', action='store_true', help='also write texts.jsonl (merge)')
    ap.add_argument('--workers', type=int, default=min(96, os.cpu_count() or 1), help='collect processes')
    ap.add_argument('--chunk_mb', type=int, default=32, help='collect byte-range chunk size')
    ap.add_argument('--allow_fingerprint_mismatch', action='store_true')
    ap.add_argument('--allow_missing', action='store_true')
    ap.add_argument('--overwrite', action='store_true', help='merge: replace a complete sidecar')
    ap.add_argument('--stop_after_chunks', type=int, default=0, help=argparse.SUPPRESS)   # resume test hook
    a = ap.parse_args(argv)
    a.teacher = st.resolve_storage_path(a.teacher) if a.teacher else st.default_teacher_path()
    if a.probe_head is None:
        a.probe_head = st.default_probe_head_path()
    elif a.probe_head.lower() in ('none', '', '0'):
        a.probe_head = None
    else:
        a.probe_head = st.resolve_storage_path(a.probe_head)
    if not 0 <= a.shard_id < a.num_shards:
        ap.error('--shard_id must be in [0, num_shards)')
    a.out = os.path.abspath(a.out)
    return a


def main(argv=None) -> int:
    args = parse_args(argv)
    os.makedirs(args.out, exist_ok=True)
    col = collect(args)
    if args.collect_only:
        report_collect(col, args)
        return 0
    if args.merge:
        merge(args, col)
        return 0
    if not os.path.isdir(args.teacher):
        raise SystemExit(f'[sem-build] teacher not found: {args.teacher}')
    if args.probe_head and not os.path.isfile(args.probe_head):
        raise SystemExit(f'[sem-build] probe head not found: {args.probe_head} (--probe_head none to skip)')
    info = check_teacher(args)
    encode_shard(args, col, info)
    return 0


if __name__ == '__main__':
    sys.exit(main())
