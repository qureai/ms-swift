# Copyright (c) ModelScope Contributors. All rights reserved.
"""Pillar0 / Atlas (`clip_multimodal_atlas`) support.

These checkpoints are CLIP-style: a `MultiModalAtlas` vision tower aligned to a text tower whose
config points at an external/relative path (`./cached_text_embedding/...`). Loading the full model
via `AutoModel` therefore fails on the text side — and we only want the vision tower anyway. So we
build *just* the vision tower (using the repo's own `_build_vision_tower`) and load only the
`model.visual.*` weights, leaving the text tower untouched.
"""
import importlib.util
import os
import sys
import types
from typing import Optional

import torch
import torch.nn as nn

from swift.utils import get_logger

logger = get_logger()

ATLAS_MODEL_TYPES = ('clip_multimodal_atlas', )
_VISUAL_WEIGHT_PREFIX = 'model.visual.'
_REMOTE_PKG = 'swift_pillar0_atlas_remote'


def _import_remote_module(snapshot_dir: str, module_name: str):
    """Import `<module_name>.py` from a custom-code snapshot dir as part of a synthetic package.

    The snapshot's files use intra-repo relative imports (`from .multimodal_atlas import ...`). We
    register a package whose `__path__` is the snapshot dir so those relative imports resolve, then
    import the requested module. This avoids transformers' dynamic-module hashing, which mishandles
    symlinked HF snapshot dirs.
    """
    if _REMOTE_PKG not in sys.modules:
        pkg = types.ModuleType(_REMOTE_PKG)
        pkg.__path__ = [snapshot_dir]
        sys.modules[_REMOTE_PKG] = pkg
    else:
        sys.modules[_REMOTE_PKG].__path__ = [snapshot_dir]
    full_name = f'{_REMOTE_PKG}.{module_name}'
    if full_name in sys.modules:
        return sys.modules[full_name]
    spec = importlib.util.spec_from_file_location(full_name, os.path.join(snapshot_dir, f'{module_name}.py'))
    module = importlib.util.module_from_spec(spec)
    sys.modules[full_name] = module
    spec.loader.exec_module(module)
    return module


def is_atlas_config(config) -> bool:
    """True if `config` describes a Pillar0/Atlas CLIP model whose vision tower is a MultiModalAtlas."""
    if getattr(config, 'model_type', None) in ATLAS_MODEL_TYPES:
        return True
    vision_cfg = getattr(config, 'vision_cfg', None)
    if isinstance(vision_cfg, dict) and vision_cfg.get('model_name') == 'multimodal_atlas':
        return True
    return False


def is_atlas_tower(module: nn.Module) -> bool:
    """True if `module` is a MultiModalAtlas vision tower (needs the AtlasAdapter for token features)."""
    return module.__class__.__name__ == 'MultiModalAtlas'


class AtlasAdapter:
    """Extract per-patch token features from a `MultiModalAtlas` tower.

    `MultiModalAtlas.forward` returns a *pooled* global vector (CLIP-style). For LLM consumption we
    need the pre-pool token sequence instead, so we tap the inputs to the tower's final
    `AdaptiveMaxPool1d` (called once per multiscale level) via a forward-pre-hook, concatenate the
    levels along the token axis, and adaptive-pool to the requested per-volume token budget.

    Callable signature matches `Vision3DBackbone`'s adapter contract:
        (encoder, vols: (n_vol, T, C, H, W), max_tokens, grid) -> (n_vol * max_tokens, encoder_dim)
    """

    def __init__(self, modality: str):
        self.modality = modality

    def __call__(self, encoder, vols, max_tokens, grid=None):
        from .backbone import pool_tokens_to
        outs = []
        for v in vols:  # (T, C, H, W)
            x = v.permute(1, 0, 2, 3).unsqueeze(0)  # (1, C, D, H, W)
            tokens = self._encode_tokens(encoder, x)  # (M, encoder_dim)
            outs.append(pool_tokens_to(tokens, max_tokens))
        return torch.cat(outs, dim=0)

    def _encode_tokens(self, encoder, x):
        captured = []

        def _pre_hook(_module, inputs):
            captured.append(inputs[0])  # (B, C, N_level)

        handle = encoder.maxpool.register_forward_pre_hook(_pre_hook)
        try:
            encoder({self.modality: x.to(next(encoder.parameters()).dtype)})
        finally:
            handle.remove()
        if not captured:
            raise RuntimeError('AtlasAdapter: no pre-pool features captured; the tower may not use `maxpool`.')
        feats = torch.cat(captured, dim=2).transpose(1, 2)  # (B, sum_N, C)
        return feats[0]  # (sum_N, C) for B == 1


def atlas_modality(encoder: nn.Module) -> str:
    """The modality key the Atlas tower was built for (e.g. 'chest_ct')."""
    return next(iter(encoder.patch_embeds.keys()))


def atlas_token_dim(encoder: nn.Module) -> Optional[int]:
    """Best-effort per-token feature dim for the Atlas tower (== AtlasStage width)."""
    try:
        return int(encoder.model_config['model']['atlas_args'].get('embed_dim')
                   or encoder.model_config['model']['atlas_args'].get('dim'))
    except Exception:  # noqa
        return None


def build_atlas_vision_tower(model_path: str,
                             config,
                             *,
                             torch_dtype: Optional[torch.dtype] = None) -> nn.Module:
    """Build the Atlas `MultiModalAtlas` vision tower and load only its `model.visual.*` weights."""
    from safetensors.torch import load_file

    modeling_clip = _import_remote_module(model_path, 'modeling_clip')
    build_vision_tower = modeling_clip._build_vision_tower
    embed_dim = getattr(config, 'embed_dim')
    vision_cfg = getattr(config, 'vision_cfg')
    visual = build_vision_tower(embed_dim, vision_cfg)

    weights_path = os.path.join(model_path, 'model.safetensors')
    state_dict = load_file(weights_path)
    vision_state = {
        k[len(_VISUAL_WEIGHT_PREFIX):]: v
        for k, v in state_dict.items() if k.startswith(_VISUAL_WEIGHT_PREFIX)
    }
    missing, unexpected = visual.load_state_dict(vision_state, strict=False)
    logger.info(f'vision_3d/atlas: built MultiModalAtlas vision tower and loaded {len(vision_state)} tensors '
                f'(missing={len(missing)}, unexpected={len(unexpected)}).')
    # Atlas mixes float32 positional math with its weights; keep the encoder in float32 to avoid
    # dtype clashes. Only the downstream projector runs in the LLM dtype (set in `attach`).
    return visual.float()
