# Copyright (c) ModelScope Contributors. All rights reserved.
"""`Vision3DBackbone`: wraps an extracted/converted vision tower + a projector into the LLM subspace.

The wrapper exposes a uniform contract regardless of the underlying encoder:
  * `forward(pixel_values_volumes, grid=None) -> (n_vol * max_tokens, llm_hidden_size)` token features;
  * `num_tokens(grid)` so the template can expand exactly the right number of <video> placeholders.

An optional `adapter` (callable) handles encoders whose `forward` is non-standard (e.g. Atlas/Pillar0,
which takes a `{modality: tensor}` dict and pools to a global vector — the adapter taps the pre-pool
token features instead). When `adapter is None`, a best-effort generic path is used.
"""
from typing import Callable, Optional

import torch
import torch.nn as nn


def pool_tokens_to(feats: torch.Tensor, n_tokens: int) -> torch.Tensor:
    """Adaptive-average-pool a (M, dim) token sequence to exactly `n_tokens` tokens (no-op if already M)."""
    if feats.shape[0] == n_tokens:
        return feats
    pooled = torch.nn.functional.adaptive_avg_pool1d(
        feats.transpose(0, 1).unsqueeze(0).float(), n_tokens).squeeze(0).transpose(0, 1)
    return pooled.to(feats.dtype)


class Vision3DBackbone(nn.Module):

    def __init__(self,
                 encoder: nn.Module,
                 *,
                 encoder_dim: int,
                 llm_hidden_size: int,
                 max_tokens: Optional[int] = None,
                 adapter: Optional[Callable] = None):
        super().__init__()
        self.encoder = encoder
        self.encoder_dim = encoder_dim
        self.llm_hidden_size = llm_hidden_size
        self.max_tokens = max_tokens
        self.adapter = adapter
        # MLP projector encoder_dim -> llm_hidden_size (the "3D encoder MLP")
        self.proj = nn.Sequential(
            nn.Linear(encoder_dim, llm_hidden_size),
            nn.GELU(),
            nn.Linear(llm_hidden_size, llm_hidden_size),
        )

    @property
    def dtype(self) -> torch.dtype:
        return self.proj[0].weight.dtype

    def _to_tokens(self, feats: torch.Tensor) -> torch.Tensor:
        """Coerce an encoder output into a (tokens, encoder_dim) sequence."""
        if feats.dim() == 5:  # (B, C, D, H, W) feature map -> (B*D*H*W, C)
            feats = feats.permute(0, 2, 3, 4, 1).reshape(-1, feats.shape[1])
        elif feats.dim() == 3:  # (B, N, C) -> (B*N, C)
            feats = feats.reshape(-1, feats.shape[-1])
        elif feats.dim() != 2:  # 2 -> already (N, C)
            raise ValueError(f'vision_3d: unexpected encoder output rank {feats.dim()} (shape {tuple(feats.shape)}).')
        return feats

    def _encode_one(self, volume: torch.Tensor) -> torch.Tensor:
        """Run the encoder on a single (T, C, H, W) volume -> (M, encoder_dim) tokens (generic path)."""
        x = volume.permute(1, 0, 2, 3).unsqueeze(0)  # (1, C, D, H, W)
        feats = self.encoder(x.to(next(self.encoder.parameters()).dtype))
        if hasattr(feats, 'last_hidden_state'):
            feats = feats.last_hidden_state
        return self._to_tokens(feats)

    def forward(self, pixel_values_volumes: torch.Tensor, grid=None) -> torch.Tensor:
        """Map a batch of volumes (n_vol, T, C, H, W) to (n_vol * max_tokens, llm_hidden_size)."""
        if self.max_tokens is None:
            raise ValueError('Vision3DBackbone requires max_tokens (== vision_3d_max_tokens) to be set for volumes.')
        vols = pixel_values_volumes
        if vols.dim() == 4:  # single volume -> add volume-batch axis
            vols = vols.unsqueeze(0)
        if self.adapter is not None:
            # adapter handles the encoder's native input/forward and returns (n_vol * max_tokens, encoder_dim)
            feats = self.adapter(self.encoder, vols, self.max_tokens, grid)
        else:
            feats = torch.cat([pool_tokens_to(self._encode_one(v), self.max_tokens) for v in vols], dim=0)
        return self.proj(feats.to(self.dtype))

    def num_tokens(self, grid=None) -> Optional[int]:
        """Tokens emitted per volume (== the pooling budget)."""
        return self.max_tokens
