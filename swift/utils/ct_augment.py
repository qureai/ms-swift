# Copyright (c) ModelScope Contributors. All rights reserved.
"""Probabilistic CT-volume augmentation, ported from Photon's `VolumeAugmenter`.

With probability `prob` (default 0.15) one of six corruptions is applied to a `(T, C, H, W)` volume
(temporal axis = T, spatial axes = H/W). These are deliberately aggressive — shuffles, heavy masking,
noise, blackout — to discourage the model from ignoring the volume. Applied at data-loading time
during training only.
"""
from typing import Callable, List, Tuple

import torch


def _random_mask(x: torch.Tensor, dim: int, keep_ratio: float) -> torch.Tensor:
    """Zero out all but a random `keep_ratio` fraction of slices along `dim`."""
    n = x.shape[dim]
    k = max(1, int(round(n * keep_ratio)))
    keep = torch.zeros(n, dtype=torch.bool)
    keep[torch.randperm(n)[:k]] = True
    shape = [1] * x.dim()
    shape[dim] = n
    return x * keep.view(shape).to(x.dtype)


def _temporal_shuffle(x: torch.Tensor) -> torch.Tensor:
    return x[torch.randperm(x.shape[0])]


def _temporal_spatial_shuffle(x: torch.Tensor) -> torch.Tensor:
    # shuffle temporal (T) and one spatial axis (H)
    return x[torch.randperm(x.shape[0])][:, :, torch.randperm(x.shape[2])]


class VolumeAugmenter:
    """Callable applying one random corruption to a (T, C, H, W) volume with probability `prob`."""

    def __init__(self, prob: float = 0.15):
        self.prob = float(prob)
        # (relative weight, fn); weights need not sum to 1 (matches Photon's 0.16*5 + 0.20)
        self.augmentations: List[Tuple[float, Callable[[torch.Tensor], torch.Tensor]]] = [
            (0.16, _temporal_shuffle),
            (0.16, _temporal_spatial_shuffle),
            (0.16, lambda x: _random_mask(x, dim=0, keep_ratio=0.2)),  # keep 20% of temporal frames
            (0.16, lambda x: _random_mask(x, dim=2, keep_ratio=0.1)),  # keep 10% of spatial rows
            (0.16, torch.rand_like),  # random noise
            (0.20, lambda x: x * 0),  # blackout
        ]

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if torch.rand(1).item() >= self.prob:
            return x
        r = torch.rand(1).item()
        cumulative = 0.0
        for weight, fn in self.augmentations:
            cumulative += weight
            if r <= cumulative:
                return fn(x)
        return x
