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


# Keys, in priority order, under which a CT volume is stored inside a multi-array container
# (.safetensors / .npz). 'image' = datalake volumes; 'ct_data' = PE volumes (which also carry a
# separate 'mask_data' array we must NEVER load as the volume).
_VOLUME_KEYS = ('ct_data', 'volume', 'image', 'data', 'arr', 'pixel_values')
# Substrings that mark an auxiliary (non-volume) array -- excluded from the fallback pick.
_NON_VOLUME_KEY_HINTS = ('mask', 'seg', 'label', 'annotation')


def _pick_volume_key(keys) -> str:
    """Choose the CT-volume array key from a multi-array container's keys.

    1) first known volume key (`_VOLUME_KEYS`); else
    2) first key that doesn't look like a mask/segmentation (`_NON_VOLUME_KEY_HINTS`); else
    3) the first key. Raises if there are no keys.
    """
    keys = list(keys)
    if not keys:
        raise ValueError('CT volume container has no arrays.')
    for k in _VOLUME_KEYS:
        if k in keys:
            return k
    non_aux = [k for k in keys if not any(h in k.lower() for h in _NON_VOLUME_KEY_HINTS)]
    return (non_aux or keys)[0]


def read_hu_volume(path: str) -> torch.Tensor:
    """Read a CT volume into a float32 (D, H, W) tensor of HU values.

    `path` may be a local file/dir or an `s3://` rnd-data-lake URI (resolved via the content cache).
    Supports .safetensors, .npy/.npz, NIfTI (.nii/.nii.gz) via nibabel, and (fallback) anything SimpleITK reads.
    Multi-array containers (.safetensors/.npz) may hold several arrays (e.g. PE volumes store the CT under
    `ct_data` alongside a `mask_data`); the CT array is selected via `_pick_volume_key`.
    """
    import os
    import numpy as np

    lower = path.lower()  # dispatch format on the ORIGINAL path (an s3 blob has no extension)
    local_path = _resolve_media_path(path)  # s3:// -> local cache file; local paths unchanged
    if lower.endswith('.safetensors'):
        from safetensors.torch import load_file
        tensors = load_file(local_path)
        key = _pick_volume_key(tensors.keys())
        t = tensors[key].float().squeeze()  # drop singleton (e.g. channel/batch) dims
        assert t.dim() == 3, f'safetensors volume `{key}` must reduce to (D, H, W), got {tuple(t.shape)}'
        return t
    if lower.endswith('.npy'):
        arr = np.load(local_path)
    elif lower.endswith('.npz'):
        npz = np.load(local_path)
        arr = npz[_pick_volume_key(npz.keys())]
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


def smart_resize_dims(native_dhw, patch_size: int, temporal_patch_size: int, merge_size: int,
                      max_tokens: int) -> Tuple[int, int, int]:
    """Pick a (D, H, W) target for VARIABLE-resolution CT (when `--ct_volume_size` is unset).

    The target (a) is divisible for the Qwen grid (D % tps == 0, H,W % (patch*merge) == 0), (b) stays as
    close to the native shape/aspect as possible, and (c) yields AT MOST `max_tokens` merged tokens, where
    merged tokens = (D/tps) * (H/(patch*merge)) * (W/(patch*merge)). So each volume keeps ~native detail up
    to the `vision_3d_max_tokens` ceiling; larger volumes are isotropically downscaled to fit.
    """
    import math
    D, H, W = (int(x) for x in native_dhw)
    unit_t = int(temporal_patch_size)
    unit_s = int(patch_size) * int(merge_size)

    def snap(x, unit):
        return max(unit, int(round(x / unit)) * unit)

    def tok(d, h, w):
        return (d // unit_t) * (h // unit_s) * (w // unit_s)

    Dt, Ht, Wt = snap(D, unit_t), snap(H, unit_s), snap(W, unit_s)
    if tok(Dt, Ht, Wt) > max_tokens:
        s = (max_tokens / tok(Dt, Ht, Wt))**(1.0 / 3.0)  # isotropic shrink factor
        Dt = max(unit_t, int(math.floor(D * s / unit_t)) * unit_t)
        Ht = max(unit_s, int(math.floor(H * s / unit_s)) * unit_s)
        Wt = max(unit_s, int(math.floor(W * s / unit_s)) * unit_s)
        # floor-snapping can still overshoot slightly; trim the largest dim one unit at a time until it fits
        while tok(Dt, Ht, Wt) > max_tokens:
            if Wt >= Ht and Wt > unit_s:
                Wt -= unit_s
            elif Ht > unit_s:
                Ht -= unit_s
            elif Dt > unit_t:
                Dt -= unit_t
            else:
                break
    return (Dt, Ht, Wt)


def load_ct_volume(path: str,
                   ct_windows: Optional[List[str]] = None,
                   ct_window_base: str = 'full_range',
                   volume_size=None,
                   augment: bool = False,
                   augment_prob: float = 0.15,
                   patch_size: Optional[int] = None,
                   temporal_patch_size: Optional[int] = None,
                   merge_size: Optional[int] = None,
                   max_tokens: Optional[int] = None,
                   normalize: str = 'none',
                   out_dtype: Optional[torch.dtype] = None) -> torch.Tensor:
    """Full pipeline: path -> (T, C, H, W) windowed, resized volume tensor.

    volume_size given -> resize every volume to that fixed (D,H,W). volume_size None -> VARIABLE resolution:
    the native shape is snapped to a divisible, `max_tokens`-capped target via `smart_resize_dims` (requires
    patch_size/temporal_patch_size/merge_size/max_tokens), so different volumes yield different token counts.
    augment: apply Photon-style probabilistic augmentation (training only). augment_prob: its probability.
    out_dtype: cast the FINAL tensor to this dtype (e.g. bfloat16/float16) to halve host-RAM + H2D cost. All
        math (read/resize/windowing/augment) stays float32; only the stored result is cast. None -> float32.
    """
    vol = read_hu_volume(path)
    if volume_size is None:
        assert None not in (patch_size, temporal_patch_size, merge_size, max_tokens), (
            'variable-resolution CT (volume_size=None) requires patch_size, temporal_patch_size, merge_size '
            'and max_tokens (pass --ct_volume_size for fixed resolution instead).')
        target = smart_resize_dims(tuple(vol.shape), patch_size, temporal_patch_size, merge_size, max_tokens)
    else:
        target = parse_volume_size(volume_size)
    vol = resize_volume(vol, target)
    windows = resolve_ct_window_list(ct_windows, ct_window_base)
    vol = build_windowed_volume(vol, windows)  # (C, D, H, W), each channel windowed to ~[0, 1]
    if normalize == 'standardize':
        # Match the pretrained vision tower's expected input statistics (~zero-mean, unit-std per channel).
        # Windowing yields [0,1] channels (mean ~0.15, std ~0.27); fed raw, that small, non-zero-mean signal
        # lets the patch-embed's large pretrained bias dominate -> the encoder emits input-INDEPENDENT
        # (constant) features (diagnosed root cause of zero grounding). Standardize each channel over its
        # voxels so the patch-embed weight actually drives the output. Math stays float32.
        mu = vol.mean(dim=(1, 2, 3), keepdim=True)
        sd = vol.std(dim=(1, 2, 3), keepdim=True)
        vol = (vol - mu) / (sd + 1e-5)
    vol = vol.permute(1, 0, 2, 3).contiguous()  # (T=D, C, H, W)
    if augment:
        from .ct_augment import VolumeAugmenter
        vol = VolumeAugmenter(prob=augment_prob)(vol)
    if out_dtype is not None and out_dtype != vol.dtype:
        vol = vol.to(out_dtype)  # cast only the stored result (values are ~[0,1] post-windowing -> safe)
    return vol
