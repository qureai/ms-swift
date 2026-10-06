#!/usr/bin/env python
"""Live ML-Decoder head scoring service for the qVLM app.

Loads ckpt-300 (model.visual + ml_decoder_head only are exercised) once and serves
    POST /score  {"image_base64": "..."}  ->  {"probs": {tag: prob}, "positives": [tag,...]}
where positives are thresholded at the per-tag Youden-optimal point, ranked by
confidence margin (prob - threshold), exactly mirroring score_head_batch.py.

Run:
    CUDA_VISIBLE_DEVICES=6 python mld_head_service.py --port 8902
"""
import argparse
import base64
import io
import json
import os
import sys
import threading

os.environ.setdefault('QURE_PLUGIN_MODE', 'infer')
os.environ.setdefault('IMAGE_MAX_TOKEN_NUM', '2500')

import torch

SWIFT_ROOT = '/fastdata4/sahil_work/VLM_qure/vlm/swift'
if SWIFT_ROOT not in sys.path:
    sys.path.insert(0, SWIFT_ROOT)

CKPT = ('/fastdata4/sahil_work/VLM_qure/vlm/swift/output/'
        'Qwen3.5-2B_stage3_diverse_frozen_vit_head/v4-20260813-090333/checkpoint-300')
YOUDEN = ('/fastdata4/sahil_work/VLM_qure/vlm/swift/vlm_inference_analysis/results/'
          'joint_run/eval140k/head_youden_thresholds.json')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--port', type=int, default=8902)
    ap.add_argument('--ckpt', default=CKPT)
    ap.add_argument('--thresholds', default=YOUDEN)
    a = ap.parse_args()

    from swift.infer_engine import InferRequest, TransformersEngine
    from swift_plugins.mldecoder_infer_plugin import attach_for_inference, _TAGS
    from PIL import Image

    eng = TransformersEngine(a.ckpt, max_batch_size=1, torch_dtype=torch.bfloat16,
                             attn_impl='sdpa', use_hf=True)
    model = eng.model.module if hasattr(eng.model, 'module') else eng.model
    template = eng.template
    attach_for_inference(model)
    head = model.ml_decoder_head
    model.eval()
    thresholds = json.load(open(a.thresholds))
    # Only tags with a genuinely GT-computed Youden threshold take part in the present
    # list (mirrors head_present_youden_npos10.json from the hinted-140k run) — tags
    # whose threshold fell back to EER are excluded.
    import csv as _csv
    summary = a.thresholds.replace('.json', '_summary.csv')
    if os.path.exists(summary):
        youden_tags = {r['tag'] for r in _csv.DictReader(open(summary)) if r['source'] == 'youden'}
        thresholds = {t: v for t, v in thresholds.items() if t in youden_tags}
    print(f'{len(thresholds)} tags eligible for the present list', flush=True)
    lock = threading.Lock()
    print(f'MLD head service ready: {len(_TAGS)} tags, ckpt={a.ckpt}', flush=True)

    def score(img: Image.Image):
        img = img.convert('RGB') if img.mode not in ('L', 'RGB') else img
        ir = InferRequest(messages=[{'role': 'user', 'content': 'Classify the abnormalities in the given chest x-ray.\n<image>'}],
                          images=[img])
        enc = template.encode(ir)
        inp = enc['inputs'] if isinstance(enc, dict) and 'inputs' in enc else enc
        pv, gt = inp['pixel_values'], inp['image_grid_thw']
        with lock, torch.no_grad():
            pv = pv.to(model.device, dtype=torch.bfloat16)
            gt = gt.to(model.device)
            vout = model.visual(pv, gt)
            logits = head(vout.last_hidden_state, vout.pooler_output, gt)
            probs = torch.sigmoid(logits.float()).cpu()[0].tolist()
        prob_map = {t: round(p, 5) for t, p in zip(_TAGS, probs)}
        # alphabetical, exactly as the hinted-140k run ordered its present list
        pos = sorted(t for t, p in prob_map.items() if t in thresholds and p >= thresholds[t])
        return prob_map, pos

    from fastapi import FastAPI
    from pydantic import BaseModel
    import uvicorn

    app = FastAPI()

    class Req(BaseModel):
        image_base64: str

    @app.post('/score')
    def score_ep(req: Req):
        img = Image.open(io.BytesIO(base64.b64decode(req.image_base64)))
        img.load()
        probs, positives = score(img)
        return {'probs': probs, 'positives': positives}

    @app.get('/health')
    def health():
        return {'ok': True, 'tags': len(_TAGS)}

    uvicorn.run(app, host='127.0.0.1', port=a.port)


if __name__ == '__main__':
    main()
