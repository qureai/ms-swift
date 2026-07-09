#!/usr/bin/env python
"""Encode/collate tests for the CT template (`qwen2_5_vl_ct`).

Validates, without loading model weights (processor only):
  * a `<video>` CT volume expands to exactly `vision_3d_max_tokens` video tokens;
  * `pixel_values_volumes` has shape (n_vol, T, C, H, W) with C == number of windows;
  * a synthetic `video_grid_thw` is emitted whose merged token count equals `vision_3d_max_tokens`;
  * the data collator stacks volumes across a batch.

Run (env swift_2026, repo root):
    USE_HF=1 python test_cases/test_ct_template.py
"""
import json
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
os.environ.setdefault('USE_HF', '1')

GREEN, RED, RESET = '\033[92m', '\033[91m', '\033[0m'
QWEN = os.environ.get('QWEN_MODEL', 'Qwen/Qwen2.5-VL-3B-Instruct')
N_TOKENS = 8
WINDOWS = ['lung', 'mediastinum', 'bone']  # + full_range base -> C = 4
VOLUME_SIZE = '32,64,64'  # (D, H, W)


def _ok(m):
    print(f'{GREEN}PASS{RESET} {m}')


def _fail(m):
    print(f'{RED}FAIL{RESET} {m}')


def main():
    from swift.model import get_model_processor
    from swift.template import get_template

    ds_path = os.path.join(REPO_ROOT, 'test_cases/dummy_ct_data/ct_dummy.jsonl')
    assert os.path.exists(ds_path), f'run make_dummy_ct_data.py first; missing {ds_path}'
    rows = [json.loads(l) for l in open(ds_path)]

    print(f'=== loading processor for {QWEN} ===')
    _, processor = get_model_processor(QWEN, load_model=False)
    template = get_template(
        processor,
        template_type='qwen2_5_vl_ct',
        ct_windows=WINDOWS,
        ct_window_base='full_range',
        ct_volume_size=VOLUME_SIZE,
        vision_3d_max_tokens=N_TOKENS,
    )
    template.set_mode('train')
    video_token_id = template.video_token_id
    merge_length = processor.image_processor.merge_size**2
    fails = 0

    # ---- single-sample encode ----
    enc = template.encode(rows[0])
    input_ids = enc['input_ids']
    n_video = sum(int(t == video_token_id) for t in input_ids)
    if n_video == N_TOKENS:
        _ok(f'volume expands to {N_TOKENS} <video> tokens')
    else:
        _fail(f'expected {N_TOKENS} video tokens, got {n_video}')
        fails += 1

    pvv = enc.get('pixel_values_volumes')
    exp_shape = (1, 32, len(WINDOWS) + 1, 64, 64)  # (n_vol, T=D, C=base+windows, H, W)
    if pvv is not None and tuple(pvv.shape) == exp_shape:
        _ok(f'pixel_values_volumes shape {tuple(pvv.shape)} (C={len(WINDOWS)+1})')
    else:
        _fail(f'pixel_values_volumes shape {None if pvv is None else tuple(pvv.shape)}, expected {exp_shape}')
        fails += 1

    vgt = enc.get('video_grid_thw')
    if vgt is not None and int(vgt[0].prod() // merge_length) == N_TOKENS:
        _ok(f'video_grid_thw {vgt.tolist()} -> {N_TOKENS} merged tokens')
    else:
        _fail(f'video_grid_thw {None if vgt is None else vgt.tolist()} does not yield {N_TOKENS} merged tokens')
        fails += 1

    # ---- batch collation ----
    encs = [template.encode(r) for r in rows[:2]]
    batch = template.data_collator(encs)
    pvv_b = batch.get('pixel_values_volumes')
    if pvv_b is not None and tuple(pvv_b.shape) == (2, 32, len(WINDOWS) + 1, 64, 64):
        _ok(f'collated pixel_values_volumes shape {tuple(pvv_b.shape)}')
    else:
        _fail(f'collated pixel_values_volumes shape {None if pvv_b is None else tuple(pvv_b.shape)}')
        fails += 1
    # position_ids must be computable (Qwen rope over the synthetic video grid)
    if 'position_ids' in batch or 'input_ids' in batch:
        _ok('data_collator produced a batch (rope/position handling ran without error)')
    else:
        _fail('data_collator did not produce input_ids/position_ids')
        fails += 1

    fails += test_volume_routing(template)

    print(f'\n{"ALL GOOD" if fails == 0 else f"{fails} FAILURE(S)"}')
    sys.exit(1 if fails else 0)


def test_volume_routing(template):
    """Validate embedding-time volume routing (_splice_volume_embeds) with a tiny dummy 3D encoder."""
    import torch
    import torch.nn as nn
    from swift.model.vision_3d import Vision3DBackbone

    fails = 0
    hidden, enc_dim = 64, 32
    n_tokens = template.num_volume_tokens()
    c = len(WINDOWS) + 1
    vtok = template.video_token_id

    class DummyEnc(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv3d(c, enc_dim, kernel_size=4, stride=4)  # (1,c,D,H,W) -> (1,enc_dim,d,h,w)

        def forward(self, x):
            return self.conv(x)

    backbone = Vision3DBackbone(DummyEnc(), encoder_dim=enc_dim, llm_hidden_size=hidden, max_tokens=n_tokens)

    class _Cfg:
        video_token_id = vtok
        image_token_id = -999

    class MockModel:
        config = _Cfg()

    model = MockModel()
    model.visual_3d = backbone

    # sequence: [text, text, <video>*n_tokens, text]
    input_ids = torch.tensor([[1, 1] + [vtok] * n_tokens + [2]])
    seq_len = input_ids.shape[1]
    volume = torch.zeros(1, 32, c, 16, 16)  # (n_vol, T, C, H, W)
    inputs = {'input_ids': input_ids, 'pixel_values_volumes': volume}
    inputs_embeds = torch.zeros(1, seq_len, hidden)

    out = template._splice_volume_embeds(inputs_embeds, inputs, model)
    video_pos = (input_ids[0] == vtok)
    filled = (out[0][video_pos].abs().sum(dim=-1) > 0).all().item()
    text_untouched = (out[0][~video_pos].abs().sum() == 0).item()
    if out.shape == (1, seq_len, hidden) and filled and text_untouched:
        _ok(f'volume embeds scattered into {n_tokens} <video> positions (text untouched)')
    else:
        _fail(f'routing: shape={tuple(out.shape)}, filled={filled}, text_untouched={text_untouched}')
        fails += 1

    # no-volume batch: dummy forward keeps visual_3d alive without changing embeds
    inputs_novol = {'input_ids': torch.tensor([[1, 2, 3]])}
    emb0 = torch.zeros(1, 3, hidden)
    out2 = template._splice_volume_embeds(emb0, inputs_novol, model)
    if out2.shape == (1, 3, hidden) and torch.allclose(out2, emb0):
        _ok('no-volume batch: dummy forward ran, embeds unchanged')
    else:
        _fail('no-volume dummy-forward path altered embeds or crashed')
        fails += 1
    return fails


if __name__ == '__main__':
    main()
