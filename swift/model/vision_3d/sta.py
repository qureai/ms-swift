# Copyright (c) ModelScope Contributors. All rights reserved.
"""Spatial-Temporal Attention (STA) for the Qwen-VL 3D encoder.

A Qwen-VL vision tower attends only WITHIN each temporal group (its `cu_seqlens` are per-frame), so it
never mixes information across the z-axis (slices) -- and both its RoPE and its learned pos-embed are
2D (H,W) only. STA fixes that by injecting a lightweight **temporal self-attention** into each vision
block, between the existing spatial attention and the MLP (each with its own pre-norm + residual):

    h = h + spatial_attn(norm1(h))     # unchanged, per-frame (cu_seqlens)
    h = h + temporal_attn(norm3(h))    # NEW: across the gt temporal groups, per spatial location
    h = h + mlp(norm2(h))              # unchanged

Design choices (cf. STAVEQ2, arXiv:2603.05950 style, adapted to Qwen3.5):
  * Temporal attention uses `round(num_heads * scale)` heads (scale=0.25 -> 4x fewer) with the SAME
    head_dim as spatial -> fewer params.
  * A separate **1D RoPE** over the temporal (gt) axis -- the principled z-position signal the tower
    otherwise lacks (distinct from the 2D spatial RoPE).
  * The temporal output projection is **zero-initialised**, so at step 0 the temporal path contributes
    exactly 0 and the STA-augmented tower reproduces the original tower bit-for-bit (pretrained spatial
    modelling preserved). It is NOT frozen -- it trains up from zero.
  * The per-volume reshape is driven by `grid_thw` (one row per volume), so STA is correct for a SINGLE
    volume, a BATCH of volumes (concatenated), and VARIABLE-resolution volumes (different gt/gh/gw) --
    unlike the reference's fixed uniform-stride reshape.

Entirely opt-in: nothing here runs unless `--vision_3d_sta true` is passed (or a reloaded checkpoint's
`vision_3d_config` has `sta_enabled`). When off, the tower is untouched.
"""
import math
from typing import List, Optional

import torch
import torch.nn as nn
from transformers.modeling_layers import GradientCheckpointingLayer

from swift.utils import get_logger

logger = get_logger()

DEFAULT_TEMPORAL_SCALE = 0.25  # temporal heads = round(spatial_heads * scale); 0.25 => 4x fewer heads


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


class VisionTemporalAttention(nn.Module):
    """1D temporal self-attention across a volume's `gt` temporal groups, per spatial location.

    Input/output token order is preserved (gt-major, matching the tower's patch layout), so the block's
    residual stays aligned. Runs per-volume using `grid_thw` -> supports batched + variable-resolution.
    """

    def __init__(self, hidden_size: int, num_heads: int, head_dim: int, rope_theta: float = 10000.0,
                 window: int = 0):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.neck_dim = num_heads * head_dim
        self.rope_theta = rope_theta
        self.window = int(window)   # 0 => full z attention; k>0 => band-diagonal local attention over +/- k groups
        self.qkv = nn.Linear(hidden_size, self.neck_dim * 3, bias=True)
        self.proj = nn.Linear(self.neck_dim, hidden_size, bias=True)
        # zero-init the output projection -> identity at start (preserve pretrained spatial modelling)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def _rope_cos_sin(self, gt: int, device, dtype):
        """1D RoPE cos/sin of shape (gt, head_dim) over temporal positions 0..gt-1."""
        inv_freq = 1.0 / (self.rope_theta**(
            torch.arange(0, self.head_dim, 2, device=device, dtype=torch.float32) / self.head_dim))
        freqs = torch.outer(torch.arange(gt, device=device, dtype=torch.float32), inv_freq)  # (gt, hd/2)
        emb = torch.cat((freqs, freqs), dim=-1)  # (gt, hd)
        return emb.cos().to(dtype), emb.sin().to(dtype)

    def forward(self, hidden_states: torch.Tensor, grid_thw: torch.Tensor) -> torch.Tensor:
        assert grid_thw is not None, (
            'VisionTemporalAttention needs grid_thw (per-volume gt,gh,gw); it is stashed on the tower by '
            'install_sta\'s forward-pre-hook. If you see this, STA was not installed correctly.')
        H, hd = self.num_heads, self.head_dim
        dev, dt = hidden_states.device, hidden_states.dtype
        outs: List[torch.Tensor] = []
        offset = 0
        for row in grid_thw:
            gt, gh, gw = int(row[0]), int(row[1]), int(row[2])
            s = gh * gw          # spatial tokens per temporal group
            n = gt * s           # this volume's token span (gt-major)
            x = hidden_states[offset:offset + n]
            offset += n
            qkv = self.qkv(x).view(gt, s, 3, H, hd)   # gt-major split
            q, k, v = qkv.unbind(2)                    # each (gt, s, H, hd)
            # temporal axis = gt, batch = (spatial s, heads H)
            q = q.permute(1, 2, 0, 3)                  # (s, H, gt, hd)
            k = k.permute(1, 2, 0, 3)
            v = v.permute(1, 2, 0, 3)
            cos, sin = self._rope_cos_sin(gt, dev, dt)  # (gt, hd)
            cos, sin = cos[None, None], sin[None, None]  # (1,1,gt,hd) -> broadcast over (s,H)
            q = q * cos + _rotate_half(q) * sin
            k = k * cos + _rotate_half(k) * sin
            # Flash/mem-efficient attention via SDPA (FlashAttention-2 / mem-efficient kernel on CUDA, math
            # fallback on CPU) over the temporal axis; replaces the eager matmul+softmax that materialised the
            # full (s,H,gt,gt) score matrix. SDPA's default scale is 1/sqrt(head_dim), matching `/sqrt(hd)`.
            attn_mask = None
            if self.window > 0 and gt > 2 * self.window + 1:
                # local temporal attention: a group attends only to +/- window neighbours along z (reduces the
                # uniformity/rank pressure of averaging over the whole z-extent). No-op when gt is small. The
                # band always includes the diagonal, so no query row is fully masked (SDPA stays finite).
                ar = torch.arange(gt, device=dev)
                keep = (ar[None, :] - ar[:, None]).abs() <= self.window      # (gt, gt) True = attend (band)
                attn_mask = keep[None, None]                                 # (1,1,gt,gt) broadcast over (s,H)
            o = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)  # (s, H, gt, hd)
            o = o.permute(2, 0, 1, 3).reshape(n, self.neck_dim)  # -> (gt, s, H, hd) -> (n, neck) gt-major
            outs.append(o)
        return self.proj(torch.cat(outs, dim=0))       # (N_total, hidden)


class STAVisionBlock(GradientCheckpointingLayer):
    """A Qwen-VL vision block augmented with temporal attention. Reuses the ORIGINAL block's pretrained
    submodules (norm1/attn/norm2/mlp) and inserts a new norm3 + temporal attention between spatial
    attention and the MLP. `grid_thw` is read from the tower (stashed by install_sta's pre-hook)."""

    def __init__(self, orig_block: nn.Module, temp_attn: VisionTemporalAttention, norm3: nn.Module):
        super().__init__()
        self.norm1 = orig_block.norm1
        self.attn = orig_block.attn
        self.norm2 = orig_block.norm2
        self.mlp = orig_block.mlp
        self.norm3 = norm3
        self.temp_attn = temp_attn
        # `_sta_grid_thw` is set on THIS block each forward by install_sta's pre-hook (a plain tensor
        # attribute -- deliberately NOT a module ref, which would create a tower<->block cycle that breaks
        # .train()/.state_dict()/.to()).

    def forward(self, hidden_states, cu_seqlens, rotary_pos_emb=None, position_embeddings=None, **kwargs):
        hidden_states = hidden_states + self.attn(
            self.norm1(hidden_states),
            cu_seqlens=cu_seqlens,
            rotary_pos_emb=rotary_pos_emb,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        grid_thw = getattr(self, '_sta_grid_thw', None)
        hidden_states = hidden_states + self.temp_attn(self.norm3(hidden_states), grid_thw)
        hidden_states = hidden_states + self.mlp(self.norm2(hidden_states))
        return hidden_states


def install_sta(tower: nn.Module,
                *,
                temporal_scale: float = DEFAULT_TEMPORAL_SCALE,
                num_temporal_heads: Optional[int] = None,
                window: int = 0,
                block_indexes: Optional[List[int]] = None) -> nn.Module:
    """In-place: wrap each (selected) block of a Qwen-VL vision `tower` with an STAVisionBlock and register
    a forward-pre-hook that stashes `grid_thw` for the temporal reshape. Returns the tower."""
    from .qwen_vl import is_qwen_vl_tower
    if not is_qwen_vl_tower(tower):
        raise ValueError('vision_3d/STA: temporal attention is only supported for a Qwen-VL vision tower '
                         '(it uses the Qwen patch grid). Got a non-Qwen tower.')
    cfg = tower.config
    hidden = int(getattr(cfg, 'hidden_size'))
    n_spatial = int(getattr(cfg, 'num_heads'))
    head_dim = hidden // n_spatial
    n_temporal = int(num_temporal_heads) if num_temporal_heads else max(1, round(n_spatial * temporal_scale))
    p = next(tower.parameters())
    idxs = list(block_indexes) if block_indexes is not None else list(range(len(tower.blocks)))

    for i in idxs:
        orig = tower.blocks[i]
        temp_attn = VisionTemporalAttention(hidden, n_temporal, head_dim, window=window).to(device=p.device, dtype=p.dtype)
        norm3 = nn.LayerNorm(hidden, eps=1e-6).to(device=p.device, dtype=p.dtype)
        sta_block = STAVisionBlock(orig, temp_attn, norm3)
        # inherit the checkpointing flag so --gradient_checkpointing keeps working
        sta_block.gradient_checkpointing = getattr(orig, 'gradient_checkpointing', False)
        tower.blocks[i] = sta_block

    def _stash_grid_thw(module, args, kwargs):
        grid_thw = kwargs.get('grid_thw')
        if grid_thw is None and len(args) >= 2:
            grid_thw = args[1]
        for blk in module.blocks:
            if isinstance(blk, STAVisionBlock):
                blk._sta_grid_thw = grid_thw  # plain tensor attr; read in STAVisionBlock.forward

    tower.register_forward_pre_hook(_stash_grid_thw, with_kwargs=True)
    tower._sta_enabled = True
    tower._sta_num_temporal_heads = n_temporal
    tower._sta_block_indexes = idxs
    logger.info(
        f'vision_3d/STA: injected temporal attention into {len(idxs)}/{len(tower.blocks)} vision blocks '
        f'(temporal_heads={n_temporal} vs spatial_heads={n_spatial}, head_dim={head_dim}); output '
        'projection zero-initialised (tower is identity-equivalent until trained).')
    return tower
