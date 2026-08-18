# Copyright (c) ModelScope Contributors. All rights reserved.
"""Perceiver-style resampler for the 3D vision encoder.

The default 3D path collapses the encoder's (often huge) token set to `vision_3d_max_tokens` by a blind
`adaptive_avg_pool1d` (a mean over tokens) and then an MLP projector. This module is a learned
alternative: a small set of learnable latent queries cross-attend to the encoder tokens and are refined
over a few layers, so the model *learns which information to route* into the fixed token budget instead
of averaging it away (Flamingo/Perceiver-IO style). It replaces both the mean-pool and the MLP projector:
it maps `input_dim -> output_dim` and emits exactly `num_latents` (== vision_3d_max_tokens) tokens.

Cross-attention runs at `dim` (defaults to the encoder dim, not the LLM hidden size) so attending over a
large key set (e.g. Atlas's ~66k pre-pool tokens) stays memory-cheap; only the `num_latents` outputs are
projected up to `output_dim`.
"""
from typing import Optional

import torch
import torch.nn as nn


class PerceiverResampler(nn.Module):

    def __init__(self,
                 input_dim: int,
                 output_dim: int,
                 num_latents: int,
                 depth: int = 2,
                 num_heads: int = 8,
                 ff_mult: int = 4,
                 dim: Optional[int] = None):
        super().__init__()
        dim = dim or input_dim
        if dim % num_heads != 0:  # keep a valid head split (Atlas dim 384 % 8 == 0)
            num_heads = next((h for h in (8, 6, 4, 2, 1) if dim % h == 0), 1)
        self.num_latents = num_latents
        self.input_proj = nn.Linear(input_dim, dim) if input_dim != dim else nn.Identity()
        self.latents = nn.Parameter(torch.randn(num_latents, dim) * 0.02)
        self.layers = nn.ModuleList([
            nn.ModuleList([
                nn.LayerNorm(dim),  # norm latents (query)
                nn.LayerNorm(dim),  # norm context (key/value)
                nn.MultiheadAttention(dim, num_heads, batch_first=True),
                nn.LayerNorm(dim),  # norm before FFN
                nn.Sequential(nn.Linear(dim, dim * ff_mult), nn.GELU(), nn.Linear(dim * ff_mult, dim)),
            ]) for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(dim)
        self.out_proj = nn.Linear(dim, output_dim)

    @property
    def dtype(self) -> torch.dtype:
        return self.out_proj.weight.dtype

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, M, input_dim) encoder tokens -> (B, num_latents, output_dim) resampled tokens."""
        x = self.input_proj(x.to(self.dtype))  # (B, M, dim)
        latents = self.latents.unsqueeze(0).expand(x.shape[0], -1, -1)  # (B, num_latents, dim)
        for norm_q, norm_kv, attn, norm_ff, ff in self.layers:
            q = norm_q(latents)
            kv = norm_kv(x)
            attn_out, _ = attn(q, kv, kv, need_weights=False)  # need_weights=False -> memory-efficient sdpa path
            latents = latents + attn_out
            latents = latents + ff(norm_ff(latents))
        return self.out_proj(self.norm(latents))  # (B, num_latents, output_dim)
