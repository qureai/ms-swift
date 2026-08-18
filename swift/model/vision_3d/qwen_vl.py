# Copyright (c) ModelScope Contributors. All rights reserved.
"""Photon-style adapter: use a Qwen-VL vision tower (Qwen2/2.5/3-VL) as the 3D encoder.

A Qwen-VL vision tower processes *video* as flattened patches + a `grid_thw`. A CT volume is a
temporal stack of slices, so we treat it as a video: patchify the `(T, C, H, W)` volume into Qwen's
exact `(num_patches, C*tps*ps*ps)` layout with a real `grid_thw`, and call the (Conv3d-inflated)
`visual(patches, grid_thw)`. This mirrors what Photon did, but as a pluggable adapter so it coexists
with the Atlas path.

`ct_volume_size` (D, H, W) must be divisible so the grid is valid:
  T % tps == 0, H % (ps*m) == 0, W % (ps*m) == 0  (Qwen defaults: ps=14, tps=2, m=2 -> H,W % 28 == 0).
"""
from typing import Optional

import torch
import torch.nn as nn

from swift.utils import get_logger

logger = get_logger()

_QWEN_VISION_HINTS = ('Qwen2VisionTransformer', 'Qwen2_5_VisionTransformer', 'Qwen3VLVision', 'Qwen3_5VLVision')


def is_qwen_vl_tower(module: nn.Module) -> bool:
    """True if `module` is a Qwen-VL vision tower (needs the video-style grid adapter)."""
    name = module.__class__.__name__
    return any(h in name for h in _QWEN_VISION_HINTS) or ('Qwen' in name and 'Vision' in name)


def _vision_attr(tower, *names, default=None):
    cfg = getattr(tower, 'config', None)
    for n in names:
        if getattr(tower, n, None) is not None:
            return getattr(tower, n)
        if cfg is not None and getattr(cfg, n, None) is not None:
            return getattr(cfg, n)
    return default


def qwen_vl_token_dim(tower: nn.Module) -> Optional[int]:
    """Output feature dim of the Qwen-VL vision tower (post-merger == LLM hidden size)."""
    dim = _vision_attr(tower, 'out_hidden_size', 'hidden_size')
    return int(dim) if dim else None


class QwenVLVideoAdapter:
    """Drive a Qwen-VL vision tower on CT volumes by patchifying them as video.

    Callable contract (matches Vision3DBackbone):
        (encoder, vols: (n_vol, T, C, H, W), max_tokens, grid) -> (n_vol * max_tokens, out_dim)
    """

    def __init__(self, patch_size: int = 14, temporal_patch_size: int = 2, merge_size: int = 2):
        self.patch_size = patch_size
        self.temporal_patch_size = temporal_patch_size
        self.merge_size = merge_size

    @classmethod
    def from_tower(cls, tower: nn.Module) -> 'QwenVLVideoAdapter':
        return cls(
            patch_size=int(_vision_attr(tower, 'patch_size', default=14)),
            temporal_patch_size=int(_vision_attr(tower, 'temporal_patch_size', default=2)),
            merge_size=int(_vision_attr(tower, 'spatial_merge_size', 'merge_size', default=2)),
        )

    def _patchify(self, volume: torch.Tensor):
        """(T, C, H, W) -> (flattened patches, grid_thw) in Qwen's exact layout."""
        tps, ps, m = self.temporal_patch_size, self.patch_size, self.merge_size
        t, c, h, w = volume.shape
        assert t % tps == 0 and h % (ps * m) == 0 and w % (ps * m) == 0, (
            f'QwenVLVideoAdapter: volume (T={t},H={h},W={w}) must satisfy T%{tps}==0 and H,W%{ps*m}==0; '
            'set --ct_volume_size accordingly.')
        gt, gh, gw = t // tps, h // ps, w // ps
        x = volume.reshape(gt, tps, c, gh // m, m, ps, gw // m, m, ps)
        x = x.permute(0, 3, 6, 4, 7, 2, 1, 5, 8).contiguous()
        patches = x.reshape(gt * gh * gw, c * tps * ps * ps)
        grid_thw = torch.tensor([[gt, gh, gw]], dtype=torch.long, device=volume.device)
        return patches, grid_thw

    def __call__(self, encoder, vols, max_tokens, grid=None, pool=True):
        # pool=True -> mean-pool each volume to max_tokens and concat -> (n_vol * max_tokens, out_dim).
        # pool=False -> return the FULL per-volume tokens (for the Perceiver resampler) -> (n_vol, M, out_dim).
        from .backbone import pool_tokens_to
        outs = []
        enc_dtype = next(encoder.parameters()).dtype
        for v in vols:  # (T, C, H, W)
            patches, grid_thw = self._patchify(v)
            out = encoder(patches.to(enc_dtype), grid_thw=grid_thw)
            # Qwen-VL vision returns BaseModelOutputWithPooling: `pooler_output` is the post-merger
            # tokens at out_hidden_size (the vision tokens fed to the LLM); prefer it, else fall back.
            tokens = getattr(out, 'pooler_output', None)
            if tokens is None:
                tokens = getattr(out, 'last_hidden_state', out)
            outs.append(pool_tokens_to(tokens, max_tokens) if pool else tokens)
        return torch.cat(outs, dim=0) if pool else torch.stack(outs, dim=0)
