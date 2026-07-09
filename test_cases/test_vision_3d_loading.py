#!/usr/bin/env python
"""Model-loading tests for the optional 3D vision encoder (`--vision_3d_model`).

Validates that a vision encoder can be loaded + extracted + 3D-ified + wrapped for several model
families used as the 3D backbone:
  * atlas    -> YalaLab/Pillar0-ChestCT   (custom code, trust_remote_code=True, natively 3D)
  * qwen     -> Qwen/Qwen2.5-VL-3B-Instruct (full VLM, extract `visual`, Conv2d->Conv3d)
  * internvl -> OpenGVLab/InternVL3-1B-hf  (full VLM, extract `vision_model`, Conv2d->Conv3d)

Run examples (inside the swift repo root, env swift_2026):
    python test_cases/test_vision_3d_loading.py --quick               # conv-inflate unit checks only
    python test_cases/test_vision_3d_loading.py --models atlas        # just the Atlas/Pillar0 case
    python test_cases/test_vision_3d_loading.py --models atlas,qwen,internvl

Each model case is isolated: a failure/unavailability is reported but does not abort the others.
"""
import argparse
import os
import sys
import traceback

import torch
import torch.nn as nn

# Make the repo root importable when run directly from test_cases/.
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

# Prefer the HuggingFace hub (the user is logged into HF in swift_2026).
os.environ.setdefault('USE_HF', '1')

DEFAULT_MODELS = {
    'atlas': dict(model='YalaLab/Pillar0-ChestCT', trust_remote_code=True, module_path=None),
    'qwen': dict(model='Qwen/Qwen2.5-VL-3B-Instruct', trust_remote_code=False, module_path=None),
    'internvl': dict(model='OpenGVLab/InternVL3-1B-hf', trust_remote_code=True, module_path=None),
}

GREEN, RED, RESET = '\033[92m', '\033[91m', '\033[0m'


def _ok(msg):
    print(f'{GREEN}PASS{RESET} {msg}')


def _fail(msg):
    print(f'{RED}FAIL{RESET} {msg}')


# --------------------------------------------------------------------------------------
# Fast unit checks (no downloads): Conv2d -> Conv3d inflation
# --------------------------------------------------------------------------------------
def test_conv_inflate():
    from swift.model.vision_3d.conv_inflate import (conv2d_to_conv3d, ensure_conv3d_patch_embed,
                                                    find_patch_embed_conv)
    failures = 0

    # (a) basic shape + weight inflation, same channel count
    conv2d = nn.Conv2d(3, 16, kernel_size=14, stride=14, bias=False)
    conv3d = conv2d_to_conv3d(conv2d, temporal_patch_size=2, in_channels=3, inflate=True)
    assert isinstance(conv3d, nn.Conv3d), 'expected Conv3d'
    assert conv3d.kernel_size == (2, 14, 14), conv3d.kernel_size
    assert conv3d.stride == (2, 14, 14), conv3d.stride
    assert conv3d.in_channels == 3 and conv3d.out_channels == 16
    # inflated weight summed over temporal dim should ~equal the original 2D weight
    summed = conv3d.weight.data.sum(dim=2)
    if torch.allclose(summed, conv2d.weight.data, atol=1e-5):
        _ok('conv_inflate: temporal inflation preserves summed response')
    else:
        _fail('conv_inflate: temporal-inflated weights do not sum to the 2D weights')
        failures += 1

    # (b) channel adaptation: 3 -> 1 (single CT window)
    conv3d_1ch = conv2d_to_conv3d(conv2d, temporal_patch_size=2, in_channels=1, inflate=True)
    if conv3d_1ch.in_channels == 1 and conv3d_1ch.weight.shape == (16, 1, 2, 14, 14):
        _ok('conv_inflate: channel adaptation 3 -> 1')
    else:
        _fail(f'conv_inflate: channel adaptation produced shape {tuple(conv3d_1ch.weight.shape)}')
        failures += 1

    # (c) channel adaptation: 3 -> 4 (multi-window)
    conv3d_4ch = conv2d_to_conv3d(conv2d, temporal_patch_size=2, in_channels=4, inflate=True)
    if conv3d_4ch.in_channels == 4 and conv3d_4ch.weight.shape == (16, 4, 2, 14, 14):
        _ok('conv_inflate: channel adaptation 3 -> 4')
    else:
        _fail(f'conv_inflate: channel adaptation produced shape {tuple(conv3d_4ch.weight.shape)}')
        failures += 1

    # (d) ensure_conv3d_patch_embed swaps a 2D embed in a tiny module, and is a no-op on a 3D one
    mod2d = nn.Sequential(nn.Conv2d(3, 8, 14, 14), nn.GELU())
    swapped = ensure_conv3d_patch_embed(mod2d, in_channels=1, temporal_patch_size=2, inflate=True)
    name, conv = find_patch_embed_conv(mod2d)
    if swapped and isinstance(conv, nn.Conv3d):
        _ok('conv_inflate: ensure_conv3d_patch_embed swapped a 2D patch embed')
    else:
        _fail('conv_inflate: ensure_conv3d_patch_embed did not swap a 2D patch embed')
        failures += 1

    mod3d = nn.Sequential(nn.Conv3d(1, 8, (2, 14, 14), (2, 14, 14)), nn.GELU())
    swapped2 = ensure_conv3d_patch_embed(mod3d, in_channels=1, temporal_patch_size=2, inflate=True)
    if not swapped2:
        _ok('conv_inflate: ensure_conv3d_patch_embed is a no-op on a native 3D embed')
    else:
        _fail('conv_inflate: ensure_conv3d_patch_embed wrongly swapped a native 3D embed')
        failures += 1

    return failures


# --------------------------------------------------------------------------------------
# Heavy checks (downloads): load + extract + 3D-ify a real model's vision tower
# --------------------------------------------------------------------------------------
def test_load_model(key, spec, in_channels=1, llm_hidden=2048):
    from swift.model.vision_3d.backbone import Vision3DBackbone
    from swift.model.vision_3d.conv_inflate import ensure_conv3d_patch_embed, find_patch_embed_conv
    from swift.model.vision_3d.loader import extract_vision_tower, load_vision_3d_source_model
    from swift.model.vision_3d.attach import infer_encoder_dim

    print(f'\n=== [{key}] loading {spec["model"]} (trust_remote_code={spec["trust_remote_code"]}) ===')
    source = load_vision_3d_source_model(
        spec['model'], trust_remote_code=spec['trust_remote_code'], torch_dtype=torch.float32)
    print(f'  source class: {source.__class__.__name__}')

    tower = extract_vision_tower(source, module_path=spec['module_path'], model_type=None)
    print(f'  extracted tower: {tower.__class__.__name__}')
    n_params = sum(p.numel() for p in tower.parameters())
    print(f'  tower params: {n_params/1e6:.1f}M')

    name, conv = find_patch_embed_conv(tower)
    print(f'  patch-embed conv: {name} -> {conv.__class__.__name__ if conv is not None else None}')
    was_2d = isinstance(conv, nn.Conv2d)
    ensure_conv3d_patch_embed(tower, in_channels=in_channels, temporal_patch_size=2, inflate=True)
    _, conv_after = find_patch_embed_conv(tower)
    assert isinstance(conv_after, nn.Conv3d), 'patch embed is not Conv3d after ensure_conv3d_patch_embed'
    if was_2d:
        # a real swap happened: in_channels must match the requested CT windowing-channel count
        assert conv_after.in_channels == in_channels, \
            f'expected in_channels={in_channels} after swap, got {conv_after.in_channels}'
    # native-3D encoders (e.g. Atlas, in_channels=11) keep their own channel count
    print(f'  patch-embed conv after: {conv_after.__class__.__name__} '
          f'(in_channels={conv_after.in_channels}, was_2d={was_2d})')

    enc_dim = infer_encoder_dim(tower) or llm_hidden
    backbone = Vision3DBackbone(tower, encoder_dim=enc_dim, llm_hidden_size=llm_hidden, max_tokens=256)
    assert backbone.proj[-1].out_features == llm_hidden
    print(f'  Vision3DBackbone built: encoder_dim={enc_dim} -> llm_hidden={llm_hidden}, '
          f'proj_params={sum(p.numel() for p in backbone.proj.parameters())/1e3:.1f}K')
    _ok(f'[{key}] loaded + extracted + 3D-ified + wrapped')
    del source, tower, backbone
    return 0


def test_atlas_adapter(spec, reduced_size=(128, 128, 128), max_tokens=16):
    """Validate AtlasAdapter pre-pool token extraction on the real Pillar0 tower (reduced image size)."""
    from swift.model.vision_3d.atlas import AtlasAdapter, atlas_modality, atlas_token_dim, is_atlas_tower
    from swift.model.vision_3d.conv_inflate import find_patch_embed_conv
    from swift.model.vision_3d.loader import load_vision_3d_source_model

    print('\n=== [atlas-adapter] pre-pool token extraction on real Pillar0 weights (reduced size) ===')
    tower = load_vision_3d_source_model(spec['model'], trust_remote_code=True, torch_dtype=torch.float32)
    if not is_atlas_tower(tower):
        _fail(f'[atlas-adapter] expected MultiModalAtlas, got {tower.__class__.__name__}')
        return 1
    modality = atlas_modality(tower)
    _, conv = find_patch_embed_conv(tower)
    in_ch = conv.in_channels
    # shrink the configured image_size so the multiscale layout matches our small test input
    tower.model_config['modalities'][modality]['image_size'] = list(reduced_size)
    adapter = AtlasAdapter(modality)
    d, h, w = reduced_size
    vol = torch.zeros(1, d, in_ch, h, w)  # (n_vol, T=D, C, H, W)
    with torch.no_grad():
        out = adapter(tower, vol, max_tokens)
    cfg_dim = atlas_token_dim(tower)
    print(f'  modality={modality} in_channels={in_ch} token_dim(config)={cfg_dim} -> adapter out {tuple(out.shape)}')
    if out.dim() == 2 and out.shape[0] == max_tokens:
        _ok(f'[atlas-adapter] produced {max_tokens} tokens of dim {out.shape[1]}')
        if cfg_dim is not None and out.shape[1] != cfg_dim:
            print(f'  NOTE: actual token dim {out.shape[1]} != config guess {cfg_dim} '
                  '(attach should infer enc_dim from a dry run, not the config).')
        return 0
    _fail(f'[atlas-adapter] unexpected output shape {tuple(out.shape)}')
    return 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--quick', action='store_true', help='only run conv-inflate unit checks (no downloads)')
    ap.add_argument('--models', default='atlas,qwen,internvl', help='comma-separated subset of {atlas,qwen,internvl}')
    ap.add_argument('--in-channels', type=int, default=1, help='CT windowing channels (sets Conv3d in_channels)')
    ap.add_argument('--atlas-model', default=None)
    ap.add_argument('--qwen-model', default=None)
    ap.add_argument('--internvl-model', default=None)
    args = ap.parse_args()

    total_fail = 0
    print('--- conv-inflate unit checks ---')
    total_fail += test_conv_inflate()
    if args.quick:
        print(f'\n{"ALL GOOD" if total_fail == 0 else f"{total_fail} FAILURE(S)"}')
        sys.exit(1 if total_fail else 0)

    specs = {k: dict(v) for k, v in DEFAULT_MODELS.items()}
    for k, override in (('atlas', args.atlas_model), ('qwen', args.qwen_model), ('internvl', args.internvl_model)):
        if override:
            specs[k]['model'] = override

    for key in [m.strip() for m in args.models.split(',') if m.strip()]:
        if key not in specs:
            _fail(f'unknown model key `{key}`')
            total_fail += 1
            continue
        try:
            total_fail += test_load_model(key, specs[key], in_channels=args.in_channels)
        except Exception as e:  # noqa
            _fail(f'[{key}] {type(e).__name__}: {e}')
            traceback.print_exc()
            total_fail += 1

    if 'atlas' in [m.strip() for m in args.models.split(',')]:
        try:
            total_fail += test_atlas_adapter(specs['atlas'])
        except Exception as e:  # noqa
            _fail(f'[atlas-adapter] {type(e).__name__}: {e}')
            traceback.print_exc()
            total_fail += 1

    print(f'\n{"ALL GOOD" if total_fail == 0 else f"{total_fail} FAILURE(S)"}')
    sys.exit(1 if total_fail else 0)


if __name__ == '__main__':
    main()
