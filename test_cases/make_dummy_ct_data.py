#!/usr/bin/env python
"""Generate synthetic CT volumes (.nii.gz) + a swift-format JSONL dataset for testing.

We have no real CT dataset path yet, so this fabricates a handful of small random HU volumes and a
chat dataset that references them via the <video> tag (volumes ride the video channel — no new token).

Usage:
    python test_cases/make_dummy_ct_data.py --n 4 --out test_cases/dummy_ct_data
Produces:
    <out>/vol_000.nii.gz ...        synthetic volumes (HU range ~[-1000, 1000])
    <out>/ct_dummy.jsonl            {"messages": [...], "videos": ["<abs path>.nii.gz"]}
"""
import argparse
import json
import os

import numpy as np


def make_volume(path, shape=(64, 128, 128), seed=0):
    rng = np.random.default_rng(seed)
    # crude HU-like content: smooth-ish body in [-1000, 1000]
    vol = rng.normal(loc=-200.0, scale=400.0, size=shape).astype(np.float32)
    vol = np.clip(vol, -1000, 1000)
    try:
        import nibabel as nib
        affine = np.eye(4, dtype=np.float32)
        nib.save(nib.Nifti1Image(vol, affine), path)
    except Exception as e:  # pragma: no cover
        raise RuntimeError(f'nibabel is required to write {path}: {e}')


QUESTIONS = [
    'What abnormality is visible in this chest CT?',
    'Describe the findings in this CT volume.',
    'Is there any evidence of nodules in this scan?',
    'Summarize the key observations from this CT.',
]
ANSWERS = [
    'No acute abnormality is identified in this synthetic volume.',
    'The synthetic volume shows uniform low-attenuation regions consistent with test data.',
    'No nodules are detected in this synthetic scan.',
    'This is randomly generated test data with no clinical findings.',
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=4)
    ap.add_argument('--out', default=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'dummy_ct_data'))
    ap.add_argument('--shape', default='64,128,128', help='D,H,W')
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    shape = tuple(int(x) for x in args.shape.split(','))
    jsonl_path = os.path.join(args.out, 'ct_dummy.jsonl')
    rows = []
    for i in range(args.n):
        vol_path = os.path.join(args.out, f'vol_{i:03d}.nii.gz')
        make_volume(vol_path, shape=shape, seed=i)
        rows.append({
            'messages': [
                {'role': 'user', 'content': f'<video>{QUESTIONS[i % len(QUESTIONS)]}'},
                {'role': 'assistant', 'content': ANSWERS[i % len(ANSWERS)]},
            ],
            'videos': [os.path.abspath(vol_path)],
        })
    with open(jsonl_path, 'w', encoding='utf-8') as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')
    print(f'Wrote {args.n} volumes (shape {shape}) and dataset -> {jsonl_path}')


if __name__ == '__main__':
    main()
