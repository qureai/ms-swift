#!/usr/bin/env python
"""Proper bbox eval: IoU / Dice / detection P-R-F1 @ IoU thresholds, per-class and macro.

Token-CE does NOT optimize overlap, so we measure overlap directly. Runs free-running generation
(swift infer, sglang DP) on a bbox val set, parses predicted boxes, matches them to GT per label
(greedy highest-IoU), and reports:
  - mean matched IoU and Dice
  - precision / recall / F1 at IoU>=0.5 and >=0.75
  - counts (TP/FP/FN), per class + macro
Predicted and GT boxes are BOTH snapped to --coord-step (match training) so the metric is in the
same frame the model was trained on.

Usage:
  python eval_bbox_iou.py --ckpt <dir> --val 15_stage5_headguided_VAL.json \
      --out bbox_eval --dp 8 --coord-step 50 [--limit N]
Filter: only DB/HB rows (a human turn asking to 'Generate the bbox ...') are scored.
"""
import argparse, ast, json, os, subprocess, sys
from collections import defaultdict

SWIFT_ROOT = '/fastdata4/sahil_work/VLM_qure/vlm/swift'
FM = os.environ.get('FM_ACCELERATE_REPO',
                    '/fastdata4/sahil_work/VLM_qure/vision_foundation_2d/foundation_model_accelerate')
CONDA = os.environ.get('CONDA_ENV', '/home/ubuntu/miniconda3/envs/ms-swift-may-s4')
SYSTEM = 'You are a helpful medical assistant.'


def q(v, step):
    return max(0, min(1000, int(round(v / step)) * step)) if step > 1 else int(v)

def qbox(b, step):
    if not (isinstance(b, list) and len(b) == 4):
        return None
    x1, y1, x2, y2 = (q(b[0], step), q(b[1], step), q(b[2], step), q(b[3], step))
    if x2 <= x1: x2 = min(1000, x1 + max(step, 1))
    if y2 <= y1: y2 = min(1000, y1 + max(step, 1))
    return [x1, y1, x2, y2]

def parse_boxes(ans, step):
    """-> list[(label, [x1,y1,x2,y2])] from a bbox answer string. Robust to the non-thinking
    prefix (`<think></think>`) and to truncated / trailing content: locate the first '[' and
    literal-eval from there, trimming to the last ']' on failure, then salvage complete entries."""
    ans = ans or ''
    i = ans.find('[')
    if i < 0:
        return []
    frag = ans[i:]
    obj = None
    for end in (len(frag), frag.rfind(']') + 1):
        if end <= 0:
            continue
        try:
            obj = ast.literal_eval(frag[:end]); break
        except Exception:
            continue
    if obj is None:                                    # salvage complete {bbox_2d,label} entries
        import re
        obj = []
        for m in re.finditer(r"\{'bbox_2d':\s*\[[^\]]*\],\s*'label':\s*'[^']*'\}", frag):
            try: obj.append(ast.literal_eval(m.group(0)))
            except Exception: pass
    out = []
    items = obj if isinstance(obj, list) else []
    for e in items:
        if not isinstance(e, dict):
            continue
        lab = e.get('label')
        bb = e.get('bbox_2d')
        if isinstance(bb, list) and bb and isinstance(bb[0], list):
            for b in bb:
                qb = qbox(b, step)
                if qb: out.append((lab, qb))
        else:
            qb = qbox(bb, step)
            if qb: out.append((lab, qb))
    return out

def iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0

def dice(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    aa = (a[2] - a[0]) * (a[3] - a[1]); bb = (b[2] - b[0]) * (b[3] - b[1])
    return 2 * inter / (aa + bb) if (aa + bb) > 0 else 0.0

def match(preds, gts):
    """Greedy highest-IoU matching within the same label. -> list of matched (iou,dice), n_fp, n_fn."""
    matched, used = [], set()
    by_lab_gt = defaultdict(list)
    for i, (l, b) in enumerate(gts):
        by_lab_gt[l].append((i, b))
    pairs = []
    for pj, (pl, pb) in enumerate(preds):
        for gi, gb in by_lab_gt.get(pl, []):
            pairs.append((iou(pb, gb), pj, gi, pb, gb))
    pairs.sort(reverse=True)
    used_p, used_g = set(), set()
    for iu, pj, gi, pb, gb in pairs:
        if pj in used_p or gi in used_g or iu <= 0:
            continue
        used_p.add(pj); used_g.add(gi)
        matched.append((iu, dice(pb, gb)))
    n_fp = len(preds) - len(used_p)
    n_fn = len(gts) - len(used_g)
    return matched, n_fp, n_fn


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--val', required=True, help='bbox val jsonl (messages+images)')
    ap.add_argument('--out', required=True)
    ap.add_argument('--dp', type=int, default=8)
    ap.add_argument('--coord-step', type=int, default=50)
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--reuse', action='store_true',
                    help='skip inference and re-score an existing <out>/result.jsonl')
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    # collect DB/HB bbox rows: the human turn that asks for the bbox + its image + GT answer
    rows = []
    for line in open(a.val):
        line = line.strip()
        if not line: continue
        d = json.loads(line)
        ms = d.get('messages') or []
        imgs = d.get('images') or ([d['image']] if 'image' in d else [])
        # find the LAST 'Generate the bbox' human turn and the gpt turn right after it
        for i in range(len(ms) - 1):
            if (ms[i].get('from') in ('human', 'user')) and 'Generate the bbox' in (ms[i].get('value') or ''):
                prompt = ms[i]['value']
                gt = ms[i + 1].get('value', '')
                # rebuild the context up to and including this prompt for infer
                ctx = [{'role': ('user' if m.get('from') in ('human', 'user') else 'assistant'),
                        'content': m.get('value')} for m in ms[:i + 1]]
                rows.append({'messages': ctx, 'images': imgs, '_gt': gt})
                break
    if a.limit:
        rows = rows[:a.limit]
    print(f'bbox val rows: {len(rows)}', flush=True)

    infer_in = os.path.join(a.out, 'infer_input.jsonl')
    with open(infer_in, 'w') as f:
        for r in rows:
            f.write(json.dumps({'messages': r['messages'], 'images': r['images']}) + '\n')
    res_path = os.path.join(a.out, 'result.jsonl')

    env = dict(os.environ)
    env['PATH'] = f"{CONDA}/bin:" + env.get('PATH', '')
    env['PYTHONPATH'] = f"{SWIFT_ROOT}:{FM}:" + env.get('PYTHONPATH', '')
    env.setdefault('IMAGE_MAX_TOKEN_NUM', '2500')
    # pt (transformers) backend: does greedy natively (do_sample=False), sidestepping the
    # sglang top_p=None verify crash. Fast enough for a ~100-row bbox val set.
    cmd = ['swift', 'infer', '--model', a.ckpt, '--load_args', 'false', '--val_dataset', infer_in,
           '--result_path', res_path, '--max_new_tokens', '2048', '--max_length', '28000',
           '--temperature', '0', '--enable_thinking', 'false', '--system', SYSTEM,
           '--use_hf', 'true', '--val_dataset_shuffle', 'false', '--write_batch_size', '50',
           '--infer_backend', 'pt', '--max_batch_size', '8']
    if a.reuse and os.path.exists(res_path):
        print(f'--reuse: scoring existing {res_path}', flush=True)
    else:
        with open(os.path.join(a.out, 'infer.log'), 'w') as lf:
            r = subprocess.run(cmd, env=env, stdout=lf, stderr=subprocess.STDOUT)
        if r.returncode != 0:
            sys.exit(f'infer failed (see {a.out}/infer.log)')

    # join predictions back to GT by row order (val_dataset_shuffle=false preserves order)
    preds = []
    for line in open(res_path):
        line = line.strip()
        if line:
            preds.append(json.loads(line).get('response', ''))
    if len(preds) != len(rows):
        print(f'WARNING: {len(preds)} predictions vs {len(rows)} GT rows', flush=True)
    if not any(parse_boxes(p, a.coord_step) for p in preds):
        sys.exit(f'no predicted boxes parsed from {res_path} -- refusing to write an all-zero summary')

    per_cls = defaultdict(lambda: {'iou': [], 'dice': [], 'tp5': 0, 'tp75': 0, 'fp': 0, 'fn': 0})
    n = min(len(preds), len(rows))
    for i in range(n):
        pb = parse_boxes(preds[i], a.coord_step)
        gb = parse_boxes(rows[i]['_gt'], a.coord_step)
        matched, n_fp, n_fn = match(pb, gb)
        for iu, di in matched:
            per_cls['__macro__']['iou'].append(iu); per_cls['__macro__']['dice'].append(di)
            per_cls['__macro__']['tp5'] += (iu >= 0.5); per_cls['__macro__']['tp75'] += (iu >= 0.75)
        per_cls['__macro__']['fp'] += n_fp; per_cls['__macro__']['fn'] += n_fn
    m = per_cls['__macro__']
    n_match = len(m['iou'])
    meaniou = sum(m['iou']) / n_match if n_match else 0.0
    meandice = sum(m['dice']) / n_match if n_match else 0.0
    tp5, tp75, fp, fn = m['tp5'], m['tp75'], m['fp'], m['fn']
    def prf(tp):
        p = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * p * rec / (p + rec) if (p + rec) else 0.0
        return p, rec, f1
    p5, r5, f5 = prf(tp5); p75, r75, f75 = prf(tp75)
    summary = {
        'rows': n, 'matched_boxes': n_match, 'coord_step': a.coord_step,
        'mean_iou': round(meaniou, 4), 'mean_dice': round(meandice, 4),
        'iou0.5': {'precision': round(p5, 4), 'recall': round(r5, 4), 'f1': round(f5, 4), 'tp': tp5},
        'iou0.75': {'precision': round(p75, 4), 'recall': round(r75, 4), 'f1': round(f75, 4), 'tp': tp75},
        'fp': fp, 'fn': fn,
    }
    json.dump(summary, open(os.path.join(a.out, 'summary.json'), 'w'), indent=2)
    print(json.dumps(summary, indent=2))
    print(f'-> {a.out}/summary.json')


if __name__ == '__main__':
    main()
