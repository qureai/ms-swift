#!/usr/bin/env python
"""Batched ML-Decoder HEAD scoring — vision forward + head only, NO text generation.

    python score_head_batch.py --ckpt <dir> --input <jsonl> --out <jsonl> --shard i --num-shards N

Why this exists: the head is a forward-hook on `model.visual`, which sglang/vLLM never fire
(they reimplement the model). So head scores require the HF (pt) model. But we do NOT need
generation — the head reads pooled vision features, produced in the prefill's vision forward.
This script runs ONLY `model.visual` + the head over batches of images, which is cheap and
high-throughput, and shards across GPUs by `--shard`.

Output: one row per image {"key","image","probs":[132]}.
"""
import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor

os.environ.setdefault('QURE_PLUGIN_MODE', 'infer')
os.environ.setdefault('IMAGE_MAX_TOKEN_NUM', '2500')

import torch

SWIFT_ROOT = '/fastdata4/sahil_work/VLM_qure/vlm/swift'
if SWIFT_ROOT not in sys.path:
    sys.path.insert(0, SWIFT_ROOT)

from swift.infer_engine import InferRequest, RequestConfig, TransformersEngine  # noqa: E402
from swift_plugins.mldecoder_infer_plugin import attach_for_inference, _TAGS  # noqa: E402


def load_rows(path, shard, num):
    out = []
    for i, line in enumerate(open(path)):
        line = line.strip()
        if not line or (num > 1 and i % num != shard):
            continue
        d = json.loads(line)
        imgs = d.get('images') or ([d['image']] if 'image' in d else [])
        imgs = [x['path'] if isinstance(x, dict) else x for x in imgs]
        if imgs:
            out.append({'key': d.get('key') or os.path.basename(imgs[0]), 'image': imgs[0]})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--input', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--shard', type=int, default=0)
    ap.add_argument('--num-shards', type=int, default=1)
    ap.add_argument('--batch', type=int, default=128)
    ap.add_argument('--io-workers', type=int, default=24,
                    help='threads to preload images from NFS in parallel (the real bottleneck)')
    a = ap.parse_args()

    rows = load_rows(a.input, a.shard, a.num_shards)
    print(f'shard {a.shard}/{a.num_shards}: {len(rows)} images', flush=True)

    eng = TransformersEngine(a.ckpt, max_batch_size=1, torch_dtype=torch.bfloat16,
                             attn_impl='sdpa', use_hf=True)
    model = eng.model.module if hasattr(eng.model, 'module') else eng.model
    template = eng.template
    attach_for_inference(model)                       # loads ml_decoder_head.* from the ckpt
    head = model.ml_decoder_head
    model.eval()

    from PIL import Image
    CLS = 'Classify the abnormalities in the given chest x-ray.'   # prompt text is irrelevant

    def encode_one(r):
        # load the PIL image (IO-bound NFS read — this is what we parallelize) then encode
        try:
            im = Image.open(r['image'])
            im = im.convert('RGB') if im.mode not in ('L', 'RGB') else im
            im.load()
            ir = InferRequest(messages=[{'role': 'user', 'content': f'{CLS}\n<image>'}], images=[im])
            enc = template.encode(ir)
            inp = enc['inputs'] if isinstance(enc, dict) and 'inputs' in enc else enc
            pv, gt = inp.get('pixel_values'), inp.get('image_grid_thw')
            if pv is None or gt is None:
                return None
            return (pv, gt, r['key'], r['image'])
        except Exception:
            return None

    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    pool = ThreadPoolExecutor(max_workers=a.io_workers)
    t0 = time.time()
    done = 0
    with open(a.out, 'w') as fo:
        for b0 in range(0, len(rows), a.batch):
            batch = rows[b0:b0 + a.batch]
            pvs, grids, keys, imgs = [], [], [], []
            for res in pool.map(encode_one, batch):        # parallel NFS load + encode
                if res is None:
                    continue
                pv, gt, k, img = res
                pvs.append(pv); grids.append(gt); keys.append(k); imgs.append(img)
            if not pvs:
                continue
            pixel_values = torch.cat(pvs, 0).to(model.device, dtype=torch.bfloat16)
            grid_thw = torch.cat(grids, 0).to(model.device)
            with torch.no_grad():
                vout = model.visual(pixel_values, grid_thw)
                logits = head(vout.last_hidden_state, vout.pooler_output, grid_thw)
                probs = torch.sigmoid(logits.float()).cpu()
            for i, k in enumerate(keys):
                fo.write(json.dumps({'key': k, 'image': imgs[i],
                                     'probs': [round(v, 5) for v in probs[i].tolist()]}) + '\n')
            fo.flush()
            done += len(keys)
            dt = time.time() - t0
            print(f'[{done}/{len(rows)}] {dt/max(done,1):.3f}s/img · '
                  f'ETA {(len(rows)-done)*dt/max(done,1)/60:.0f}m', flush=True)
    print(f'done shard {a.shard}: {done} imgs in {(time.time()-t0)/60:.1f}m -> {a.out}', flush=True)
    # tags order for downstream AUC
    json.dump(_TAGS, open(a.out + '.tags.json', 'w'))


if __name__ == '__main__':
    main()
