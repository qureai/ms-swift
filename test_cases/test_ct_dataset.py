#!/usr/bin/env python
"""Dataset-loading tests through swift's real `load_dataset` pipeline (CPU; no model weights).

Validates:
  * the dummy CT JSONL loads and rows carry `messages` + `videos`;
  * the existing 2D pretraining JSON loads (a small slice);
  * a loaded CT row encodes through the `qwen2_5_vl_ct` template into training tensors.

Run (env swift_2026, repo root):
    USE_HF=1 python test_cases/test_ct_dataset.py
"""
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
os.environ.setdefault('USE_HF', '1')

GREEN, RED, RESET = '\033[92m', '\033[91m', '\033[0m'
CT_JSONL = os.path.join(REPO_ROOT, 'test_cases/dummy_ct_data/ct_dummy.jsonl')
DATA_2D = '/cache/fast_data_nas8/vlm_team_data/janhavi/11_Nov_pretraining_data_without_bbox.json'
QWEN = os.environ.get('QWEN_MODEL', 'Qwen/Qwen2.5-VL-3B-Instruct')


def _ok(m):
    print(f'{GREEN}PASS{RESET} {m}')


def _fail(m):
    print(f'{RED}FAIL{RESET} {m}')


def main():
    from swift.dataset import load_dataset
    fails = 0

    # 1) dummy CT JSONL
    assert os.path.exists(CT_JSONL), f'run make_dummy_ct_data.py first; missing {CT_JSONL}'
    ct_train, _ = load_dataset([CT_JSONL], num_proc=1)
    row = ct_train[0]
    if len(ct_train) > 0 and 'messages' in row and 'videos' in row:
        _ok(f'CT dataset loaded via swift: {len(ct_train)} rows; row keys={sorted(row)[:6]}')
    else:
        _fail(f'CT dataset row missing messages/videos: keys={sorted(row)}')
        fails += 1

    # 2) existing 2D pretraining data (small slice)
    if os.path.exists(DATA_2D):
        try:
            d2d_train, _ = load_dataset([f'{DATA_2D}#8'], num_proc=1)
            _ok(f'2D pretraining data loaded via swift: {len(d2d_train)} rows (sliced)')
        except Exception as e:  # noqa
            _fail(f'2D data load failed: {type(e).__name__}: {e}')
            fails += 1
    else:
        print(f'  (skip 2D data; not found at {DATA_2D})')

    # 3) encode a loaded CT row through the CT template
    from swift.model import get_model_processor
    from swift.template import get_template
    _, processor = get_model_processor(QWEN, load_model=False)
    template = get_template(
        processor, template_type='qwen2_5_vl_ct', ct_windows=['lung', 'bone'],
        ct_volume_size='32,64,64', vision_3d_max_tokens=8)
    template.set_mode('train')
    enc = template.encode(ct_train[0])
    n_video = sum(int(t == template.video_token_id) for t in enc['input_ids'])
    if n_video == 8 and enc.get('pixel_values_volumes') is not None:
        _ok(f'loaded CT row encodes: {n_video} video tokens, pixel_values_volumes '
            f'{tuple(enc["pixel_values_volumes"].shape)}')
    else:
        _fail(f'CT row encode: n_video={n_video}, pvv={enc.get("pixel_values_volumes") is not None}')
        fails += 1

    print(f'\n{"ALL GOOD" if fails == 0 else f"{fails} FAILURE(S)"}')
    sys.exit(1 if fails else 0)


if __name__ == '__main__':
    main()
