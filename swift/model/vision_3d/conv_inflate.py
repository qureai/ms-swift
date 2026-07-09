# Copyright (c) ModelScope Contributors. All rights reserved.
"""Utilities to turn a 2D-conv patch embedding into a 3D-conv one.

When the vision tower extracted from `--vision_3d_model` uses an `nn.Conv2d` for its patch
embedding (i.e. it is really a 2D image encoder), we replace that conv with an `nn.Conv3d` so it
can ingest volumetric (T, C, H, W) input. Pretrained 2D weights are optionally *inflated* across the
new temporal axis. Natively-3D encoders (e.g. Pillar0/Atlas, which already use `nn.Conv3d`) are left
untouched.
"""
from typing import Optional, Tuple

import torch
import torch.nn as nn

from swift.utils import get_logger

logger = get_logger()


def find_patch_embed_conv(module: nn.Module) -> Tuple[Optional[str], Optional[nn.Module]]:
    """Return (dotted_name, conv) of the first Conv2d/Conv3d encountered (the patch embed).

    The patch-embedding conv is, in every ViT-style tower we care about, the earliest conv in a
    depth-first walk. Returns (None, None) if no conv is found.
    """
    for name, m in module.named_modules():
        if isinstance(m, (nn.Conv2d, nn.Conv3d)):
            return name, m
    return None, None


def conv2d_to_conv3d(conv2d: nn.Conv2d,
                     *,
                     temporal_patch_size: int = 2,
                     in_channels: Optional[int] = None,
                     inflate: bool = True) -> nn.Conv3d:
    """Build an `nn.Conv3d` mirroring `conv2d`, with a temporal kernel/stride of `temporal_patch_size`.

    in_channels: input channel count (number of CT windowing channels). The pretrained 2D weights are
        temporally inflated only when `inflate` and the channel count is unchanged; on a channel
        mismatch (or `inflate=False`) the conv is left freshly initialised (to be trained).
    """
    out_ch = conv2d.out_channels
    in_ch = in_channels or conv2d.in_channels
    kh, kw = conv2d.kernel_size
    sh, sw = conv2d.stride
    ph, pw = (conv2d.padding if isinstance(conv2d.padding, tuple) else (conv2d.padding, conv2d.padding))
    has_bias = conv2d.bias is not None

    conv3d = nn.Conv3d(
        in_ch,
        out_ch,
        kernel_size=(temporal_patch_size, kh, kw),
        stride=(temporal_patch_size, sh, sw),
        padding=(0, ph, pw),
        bias=has_bias,
    )
    conv3d = conv3d.to(device=conv2d.weight.device, dtype=conv2d.weight.dtype)

    if inflate and in_ch == conv2d.in_channels:
        with torch.no_grad():
            # inflate across temporal dim and divide by T so the summed response matches the 2D conv
            w3d = conv2d.weight.data.unsqueeze(2).repeat(1, 1, temporal_patch_size, 1, 1) / temporal_patch_size
            conv3d.weight.data.copy_(w3d.to(conv3d.weight.dtype))
            if has_bias:
                conv3d.bias.data.copy_(conv2d.bias.data.to(conv3d.bias.dtype))
    return conv3d


def reinit_conv3d_in_channels(conv3d: nn.Conv3d, in_channels: int) -> nn.Conv3d:
    """Return a fresh `nn.Conv3d` matching `conv3d`'s geometry but with `in_channels` inputs.

    Used when an already-3D encoder's native input-channel count differs from the CT windowing-channel
    count: the patch embed is reinitialised (and trained), while the rest of the encoder keeps its
    pretrained weights.
    """
    new_conv = nn.Conv3d(
        in_channels,
        conv3d.out_channels,
        kernel_size=conv3d.kernel_size,
        stride=conv3d.stride,
        padding=conv3d.padding,
        bias=conv3d.bias is not None,
    )
    return new_conv.to(device=conv3d.weight.device, dtype=conv3d.weight.dtype)


def _set_submodule(root: nn.Module, dotted_name: str, new_module: nn.Module) -> None:
    """Replace the submodule at `dotted_name` (supports numeric indices for Sequential/ModuleList)."""
    parts = dotted_name.split('.')
    obj = root
    for p in parts[:-1]:
        obj = obj[int(p)] if p.isdigit() else getattr(obj, p)
    last = parts[-1]
    if last.isdigit():
        obj[int(last)] = new_module
    else:
        setattr(obj, last, new_module)


def ensure_conv3d_patch_embed(tower: nn.Module,
                              *,
                              in_channels: int,
                              temporal_patch_size: int = 2,
                              inflate: bool = True) -> bool:
    """If `tower`'s patch-embed conv is 2D, replace it in-place with a 3D conv.

    Returns True if a swap happened, False if the tower was already 3D (or had no conv).
    """
    name, conv = find_patch_embed_conv(tower)
    if conv is None:
        logger.warning('vision_3d: no Conv2d/Conv3d patch embed found in the extracted vision tower; '
                       'leaving it unchanged.')
        return False

    if isinstance(conv, nn.Conv3d):
        if conv.in_channels == in_channels:
            logger.info(f'vision_3d: patch embed `{name}` is already nn.Conv3d with in_channels={in_channels}; '
                        'no change needed.')
            return False
        # native 3D, but its input channels differ from the CT windowing-channel count -> reinit
        logger.info(f'vision_3d: patch embed `{name}` is Conv3d(in_channels={conv.in_channels}) but '
                    f'ct_num_channels={in_channels}; reinitialising the patch embed to {in_channels} channels '
                    '(pretrained patch-embed weights dropped; rest of the encoder kept).')
        _set_submodule(tower, name, reinit_conv3d_in_channels(conv, in_channels))
        return True

    reuse = inflate and (conv.in_channels == in_channels)
    logger.info(f'vision_3d: converting 2D patch embed `{name}` (in={conv.in_channels}, out={conv.out_channels}, '
                f'kernel={conv.kernel_size}) -> Conv3d(in={in_channels}, temporal_patch_size={temporal_patch_size}); '
                f"weights={'temporally inflated' if reuse else 'reinitialised'}.")
    conv3d = conv2d_to_conv3d(
        conv, temporal_patch_size=temporal_patch_size, in_channels=in_channels, inflate=inflate)
    _set_submodule(tower, name, conv3d)
    return True
