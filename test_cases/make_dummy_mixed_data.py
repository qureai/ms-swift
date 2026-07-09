#!/usr/bin/env python
"""Generate a mixed 2D-image + 3D-CT-volume dataset covering all routing cases.

Writes, in order, one sample of each case (so with batch_size=1, no-shuffle, steps 1/2/3 hit each):
  1. image-only   (<image>)          -> 2D encoder only (3D runs a dummy forward)
  2. volume-only  (<video> .nii.gz)  -> 3D encoder only (2D not invoked on Qwen3-VL)
  3. both         (<image><video>)   -> 2D and 3D encoders

Usage: python test_cases/make_dummy_mixed_data.py --out test_cases/dummy_mixed_data
"""
import argparse
import json
import os

import numpy as np


def make_image(path, size=224, seed=0):
    from PIL import Image
    rng = np.random.default_rng(seed)
    arr = rng.integers(0, 255, size=(size, size, 3), dtype=np.uint8)
    Image.fromarray(arr).save(path)


def make_volume(path, shape=(64, 64, 64), seed=0):
    import nibabel as nib
    rng = np.random.default_rng(seed + 100)
    vol = np.clip(rng.normal(-200.0, 400.0, size=shape), -1000, 1000).astype(np.float32)
    nib.save(nib.Nifti1Image(vol, np.eye(4, dtype=np.float32)), path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'dummy_mixed_data'))
    ap.add_argument('--img-size', type=int, default=224)
    ap.add_argument('--vol-shape', default='64,64,64')
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    vshape = tuple(int(x) for x in args.vol_shape.split(','))

    img = os.path.join(args.out, 'img.png')
    vol = os.path.join(args.out, 'vol.nii.gz')
    make_image(img, size=args.img_size)
    make_volume(vol, shape=vshape)
    img, vol = os.path.abspath(img), os.path.abspath(vol)

    rows = [
        {  # 1. image only
            'messages': [{'role': 'user', 'content': '<image>What is shown in this image?'},
                         {'role': 'assistant', 'content': 'A synthetic test image.'}],
            'images': [img],
        },
        {  # 2. volume only
            'messages': [{'role': 'user', 'content': '<video>Describe this CT volume.'},
                         {'role': 'assistant', 'content': 'A synthetic test CT volume.'}],
            'videos': [vol],
        },
        {  # 3. both
            'messages': [{'role': 'user', 'content': '<image><video>Compare the image and the CT volume.'},
                         {'role': 'assistant', 'content': 'Both are synthetic test data.'}],
            'images': [img],
            'videos': [vol],
        },
    ]
    path = os.path.join(args.out, 'ct_mixed.jsonl')
    with open(path, 'w', encoding='utf-8') as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')
    print(f'Wrote {len(rows)} samples (image-only, volume-only, both) -> {path}')


if __name__ == '__main__':
    main()
