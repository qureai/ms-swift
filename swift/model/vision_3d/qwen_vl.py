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

    @staticmethod
    def _tower_tokens(out):
        """Extract the post-merger vision tokens from the tower output (tensor / *WithPooling /
        *WithDeepstackFeatures all expose pooler_output = the merged tokens)."""
        tokens = getattr(out, 'pooler_output', None)
        if tokens is None:
            tokens = getattr(out, 'last_hidden_state', out)
        return tokens

    @staticmethod
    def _tower_has_window_attention(encoder) -> bool:
        """Qwen2.5-VL-style towers reorder ALL tokens into windows across the packed sequence (see
        `get_window_index`), so a volume's features change with its batch-mates -> concatenating volumes
        is NOT equivalent to encoding them alone. Qwen3-VL / Qwen3.5 towers have no window attention
        (`window_size`/`fullatt_block_indexes` absent): attention is pure cu_seqlens block-diagonal and
        rotary position ids reset per media, so batching IS equivalent. Detect the windowed case to guard."""
        cfg = getattr(encoder, 'config', None)
        if cfg is None:
            return False
        return bool(getattr(cfg, 'window_size', None)) or bool(getattr(cfg, 'fullatt_block_indexes', None))

    def __call__(self, encoder, vols, max_tokens, grid=None, pool=True, variable=False):
        # BATCHED by default for windowless towers (Qwen3-VL / Qwen3.5): concat every volume's patches into
        # ONE forward with a multi-row grid_thw. The tower builds cu_seqlens from grid_thw so attention is
        # block-diagonal per volume, and position ids reset per media -- so the output equals encoding each
        # volume alone, but in a single tower forward (far better GPU utilisation than a per-volume Python
        # loop). Windowed towers (Qwen2.5-VL) reorder tokens across the whole packed sequence and are NOT
        # batch-equivalent -> fall back to the per-volume loop. (Atlas uses its own AtlasAdapter.)
        # pool=True     -> mean-pool each volume to max_tokens, concat -> (n_vol * max_tokens, out_dim).
        # pool=False    -> stack the FULL per-volume tokens (for the Perceiver resampler) -> (n_vol, M, out_dim).
        # variable=True -> NO pooling/stacking: concat each volume's native (variable) token count
        #                  -> (sum_i M_i, out_dim). Requires equal-M only when stacking, so variable is safe
        #                  with differently-sized volumes (the variable-resolution CT path).
        from .backbone import pool_tokens_to
        enc = next(encoder.parameters())
        enc_dtype, enc_device = enc.dtype, enc.device

        if len(vols) > 1 and not self._tower_has_window_attention(encoder):
            patches, grids = [], []
            for v in vols:  # (T, C, H, W)
                p, g = self._patchify(v.to(enc_device))
                patches.append(p)
                grids.append(g)
            grid_thw = torch.cat(grids, dim=0)  # (n_vol, 3): one row per volume -> cu_seqlens isolates each
            tokens = self._tower_tokens(encoder(torch.cat(patches, dim=0).to(enc_dtype), grid_thw=grid_thw))
            m2 = self.merge_size**2
            counts = [int(g[0, 0] * g[0, 1] * g[0, 2]) // m2 for g in grids]  # merged tokens per volume
            per_vol = list(torch.split(tokens, counts, dim=0))
        else:
            per_vol = []
            for v in vols:  # (T, C, H, W)
                patches, grid_thw = self._patchify(v.to(enc_device))
                per_vol.append(self._tower_tokens(encoder(patches.to(enc_dtype), grid_thw=grid_thw)))

        if variable:
            return torch.cat(per_vol, dim=0)  # (sum_i M_i, out_dim) -- native per-volume counts, no pooling
        outs = [pool_tokens_to(t, max_tokens) if pool else t for t in per_vol]
        return torch.cat(outs, dim=0) if pool else torch.stack(outs, dim=0)
