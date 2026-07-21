# Copyright (c) ModelScope Contributors. All rights reserved.
"""Loading CT volumes into the `(T, C, H, W)` tensor the 3D encoder consumes.

Pipeline: read HU volume (NIfTI / DICOM dir / .npy) -> resize (trilinear) to a target (D, H, W) ->
multi-channel HU windowing -> `(C, D, H, W)` -> permute to `(T=D, C, H, W)` (depth slices act as the
temporal axis, matching how volumes ride the <video> channel).

Kept dependency-light (torch + lazy IO libs) so it can be unit-tested without importing swift.
"""
from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F

from .ct_windowing import build_windowed_volume, resolve_ct_window_list

DEFAULT_CT_VOLUME_SIZE: Tuple[int, int, int] = (64, 224, 224)  # (D, H, W)


def parse_volume_size(volume_size, default: Tuple[int, int, int] = DEFAULT_CT_VOLUME_SIZE) -> Tuple[int, int, int]:
    """Parse a 'D,H,W' string (or tuple/None) into a (D, H, W) int tuple."""
    if volume_size is None:
        return default
    if isinstance(volume_size, str):
        parts = [int(x) for x in volume_size.replace(' ', '').split(',') if x]
    else:
        parts = list(volume_size)
    assert len(parts) == 3, f'volume_size must be D,H,W (3 ints), got {volume_size}'
    return tuple(int(x) for x in parts)


def _resolve_media_path(path: str) -> str:
    """Resolve an `s3://` URI to a local file via the rnd-data-lake content cache; pass local paths through.

    Volumes stored in the rnd-data-lake live at `s3://rnd-data-lake/safetensors/<UID>.safetensors`; the
    training rows carry that URI in the `videos` field and it is resolved to a local blob here (downloaded
    on a cache miss). The resolved blob is content-addressed and has *no* file extension, so callers must
    keep the original URI for format dispatch and open the returned local path.
    """
    if not path.startswith('s3://'):
        return path
    try:
        from rnd_data_lake.cache import get_cached_path
    except ImportError as e:
        raise ImportError(
            'Loading an s3:// volume requires the `rnd_data_lake` package (pip install -e '
            '<rnd-data-lake>) and the `nas` boto profile configured (see its README).') from e
    return str(get_cached_path(path))


def read_hu_volume(path: str) -> torch.Tensor:
    """Read a CT volume into a float32 (D, H, W) tensor of HU values.

    `path` may be a local file/dir or an `s3://` rnd-data-lake URI (resolved via the content cache).
    Supports .safetensors, .npy/.npz, NIfTI (.nii/.nii.gz) via nibabel, and (fallback) anything SimpleITK reads.
    """
    import os
    import numpy as np

    lower = path.lower()  # dispatch format on the ORIGINAL path (an s3 blob has no extension)
    local_path = _resolve_media_path(path)  # s3:// -> local cache file; local paths unchanged
    if lower.endswith('.safetensors'):
        from safetensors.torch import load_file
        tensors = load_file(local_path)
        # prefer a conventionally-named volume tensor, else the first entry
        key = next((k for k in ('volume', 'image', 'data', 'arr', 'pixel_values') if k in tensors), None)
        key = key or next(iter(tensors))
        t = tensors[key].float().squeeze()  # drop singleton (e.g. channel/batch) dims
        assert t.dim() == 3, f'safetensors volume `{key}` must reduce to (D, H, W), got {tuple(t.shape)}'
        return t
    if lower.endswith('.npy'):
        arr = np.load(local_path)
    elif lower.endswith('.npz'):
        npz = np.load(local_path)
        arr = npz[list(npz.keys())[0]]
    elif lower.endswith('.nii') or lower.endswith('.nii.gz'):
        import nibabel as nib
        # nibabel volumes are (X, Y, Z); transpose to (Z=D, Y=H, X=W) for depth-first slicing
        arr = np.asarray(nib.load(local_path).get_fdata())
        arr = np.transpose(arr, (2, 1, 0))
    else:
        # DICOM directory or other format via SimpleITK
        import SimpleITK as sitk
        if os.path.isdir(local_path):
            reader = sitk.ImageSeriesReader()
            reader.SetFileNames(reader.GetGDCMSeriesFileNames(local_path))
            img = reader.Execute()
        else:
            img = sitk.ReadImage(local_path)
        arr = sitk.GetArrayFromImage(img)  # already (D, H, W)
    return torch.as_tensor(np.ascontiguousarray(arr), dtype=torch.float32)


def resize_volume(vol: torch.Tensor, size: Tuple[int, int, int]) -> torch.Tensor:
    """Trilinear-resize a (D, H, W) volume to `size` (D, H, W). Uses torch (no MONAI dependency)."""
    assert vol.dim() == 3, f'expected (D, H, W), got {tuple(vol.shape)}'
    resized = F.interpolate(vol[None, None].float(), size=tuple(size), mode='trilinear', align_corners=False)
    return resized[0, 0]


def load_ct_volume(path: str,
                   ct_windows: Optional[List[str]] = None,
                   ct_window_base: str = 'full_range',
                   volume_size=None,
                   augment: bool = False,
                   augment_prob: float = 0.15) -> torch.Tensor:
    """Full pipeline: path -> (T, C, H, W) windowed, resized volume tensor.

    augment: apply Photon-style probabilistic augmentation (training only). augment_prob: its probability.
    """
    vol = read_hu_volume(path)
    vol = resize_volume(vol, parse_volume_size(volume_size))
    windows = resolve_ct_window_list(ct_windows, ct_window_base)
    vol = build_windowed_volume(vol, windows)  # (C, D, H, W)
    vol = vol.permute(1, 0, 2, 3).contiguous()  # (T=D, C, H, W)
    if augment:
        from .ct_augment import VolumeAugmenter
        vol = VolumeAugmenter(prob=augment_prob)(vol)
    return vol
