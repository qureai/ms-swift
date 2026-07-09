# Copyright (c) ModelScope Contributors. All rights reserved.
"""Multi-channel HU windowing for CT volumes.

A CT volume is normalised into one channel per requested *window* (a center/width in Hounsfield
Units that highlights a particular tissue). Channel 0 is always the full-range base `[-1000, 1000]`
window (unless disabled); each name in `ct_windows` is appended as an extra channel. The number of
channels produced must equal the 3D encoder's first-conv `in_channels`.

Lives under `swift.utils` (a dependency-light location) so both the data-side template and the
model-side encoder attachment can import it without creating an import cycle.
"""
from typing import List, Optional

import torch

# center/width (Hounsfield Units) per anatomical window.
ANATOMICAL_WINDOWS = {
    'lung': {'center': -600, 'width': 1500},
    'mediastinum': {'center': 50, 'width': 400},
    'abdomen': {'center': 40, 'width': 400},
    'liver': {'center': 80, 'width': 150},
    'bone': {'center': 400, 'width': 1800},
    'brain': {'center': 40, 'width': 80},
    'subdural': {'center': 75, 'width': 215},
    'stroke': {'center': 40, 'width': 40},
    'temporal_bone': {'center': 600, 'width': 2800},
    'soft_tissue': {'center': 50, 'width': 350},
    # base full-range channel: [-1000, 1000] == center 0, width 2000
    'full_range': {'center': 0, 'width': 2000},
}


def apply_hu_windowing(vol: torch.Tensor,
                       window: str,
                       mean: Optional[float] = None,
                       std: Optional[float] = None) -> torch.Tensor:
    """Normalise a HU volume according to `window` (a name in ANATOMICAL_WINDOWS, or 'minmax')."""
    if window == 'minmax':
        v = vol.clone().float()
        v = v - v.min()
        ans = v / (v.max() + 1e-6)
    else:
        if window not in ANATOMICAL_WINDOWS:
            raise ValueError(f"ct_windowing: unknown window '{window}'. "
                             f'Valid: {sorted(ANATOMICAL_WINDOWS)} or "minmax".')
        cfg = ANATOMICAL_WINDOWS[window]
        center, width = cfg['center'], cfg['width']
        low, high = center - width / 2, center + width / 2
        v = vol.clamp(low, high)
        ans = (v - low) / width
    if mean is not None and std is not None:
        ans = (ans - mean) / std
    return ans


def resolve_ct_window_list(ct_windows: Optional[List[str]], ct_window_base: str = 'full_range') -> List[str]:
    """Return the ordered window list, prepending the base channel unless `ct_window_base == 'none'`.

    ct_windows: extra window names (e.g. ['lung', 'mediastinum', 'bone']); None/empty -> base only.
    ct_window_base: 'full_range' (default), 'minmax', or 'none'.
    """
    windows: List[str] = []
    if ct_window_base and ct_window_base != 'none':
        windows.append(ct_window_base)
    # accept both space-separated (['lung','bone']) and comma-separated (['lung,bone']) CLI forms
    for entry in (ct_windows or []):
        for w in str(entry).split(','):
            w = w.strip()
            if w:
                windows.append(w)
    if not windows:
        # avoid a zero-channel volume if the base was disabled and no windows were given
        windows = ['full_range']
    return windows


def num_ct_channels(ct_windows: Optional[List[str]], ct_window_base: str = 'full_range') -> int:
    """Number of channels the windowing will produce (== the 3D encoder's required `in_channels`)."""
    return len(resolve_ct_window_list(ct_windows, ct_window_base))


def build_windowed_volume(vol: torch.Tensor,
                          windows: List[str],
                          mean: Optional[float] = None,
                          std: Optional[float] = None) -> torch.Tensor:
    """Apply each window to a (D, H, W) HU volume and stack along a new channel axis -> (C, D, H, W)."""
    assert vol.dim() == 3, f'expected (D, H, W) HU volume, got shape {tuple(vol.shape)}'
    channels = [apply_hu_windowing(vol, w, mean=mean, std=std) for w in windows]
    return torch.stack(channels, dim=0)
