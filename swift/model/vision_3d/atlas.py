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

    def __call__(self, encoder, vols, max_tokens, grid=None, pool=True):
        # Encode ALL volumes of the micro-batch in ONE Atlas forward (Atlas supports batch:
        # MultiModalAtlas.forward uses bsz=image.shape[0]). Batching means BatchNorm3d normalizes over the
        # whole micro-batch (per GPU) rather than per single volume -> more stable stats. All volumes are
        # the same --ct_volume_size, so their token counts match and the batch is rectangular.
        # pool=True  -> mean-pool each volume to max_tokens, flatten -> (n_vol * max_tokens, encoder_dim).
        # pool=False -> return the FULL per-volume tokens (for the Perceiver resampler) -> (n_vol, M, enc_dim).
        x = vols.permute(0, 2, 1, 3, 4).contiguous()  # (n_vol, C, D, H, W)
        feats = self._encode_tokens(encoder, x)  # (n_vol, sum_N, encoder_dim)
        if not pool:
            return feats
        pooled = self._pool_tokens(feats, max_tokens)  # (n_vol, max_tokens, encoder_dim)
        # row-major flatten keeps volume 0's tokens first, then volume 1, ... (the <video> position order)
        return pooled.reshape(-1, pooled.shape[-1])

    @staticmethod
    def _pool_tokens(feats: torch.Tensor, n_tokens: int) -> torch.Tensor:
        """Adaptive-avg-pool a (B, M, C) token sequence to (B, n_tokens, C) along the token axis."""
        if feats.shape[1] == n_tokens:
            return feats
        pooled = torch.nn.functional.adaptive_avg_pool1d(feats.transpose(1, 2).float(), n_tokens).transpose(1, 2)
        return pooled.to(feats.dtype)

    def _encode_tokens(self, encoder, x):
        """Run the Atlas tower on a batch `x` = (B, C, D, H, W); return pre-pool tokens (B, sum_N, C)."""
        captured = []

        def _pre_hook(_module, inputs):
            captured.append(inputs[0])  # (B, C, N_level)

        handle = encoder.maxpool.register_forward_pre_hook(_pre_hook)
        enc_dtype = next(encoder.parameters()).dtype
        # Atlas hard-casts its relative-position coords to fp32 (multimodal_msa.RelativePosEmb); when the
        # tower runs in bf16/fp16 that fp32 tensor clashes with the bf16 `cpb_mlp` Linear
        # ('expected mat1 and mat2 to have the same dtype'). Running the forward under autocast makes the
        # Linear cast that fp32 input to the compute dtype (fixing the clash) while keeping norm/softmax in
        # fp32 for stability. For a genuinely fp32 tower autocast is disabled -> path is byte-identical.
        use_autocast = enc_dtype in (torch.bfloat16, torch.float16)
        try:
            with torch.autocast(device_type=x.device.type, dtype=enc_dtype, enabled=use_autocast):
                encoder({self.modality: x.to(enc_dtype)})
        finally:
            handle.remove()
        if not captured:
            raise RuntimeError('AtlasAdapter: no pre-pool features captured; the tower may not use `maxpool`.')
        # each captured level is (B, C, N_level); concat over the token axis -> (B, C, sum_N) -> (B, sum_N, C)
        return torch.cat(captured, dim=2).transpose(1, 2)


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
    # Historically the tower was forced to float32 because Atlas's relative-position math hard-casts an
    # intermediate to fp32 (multimodal_msa.RelativePosEmb), which clashes with bf16 weights. That single
    # clash is now handled by an autocast wrap in `AtlasAdapter._encode_tokens`, so the tower can run in
    # the run's dtype (bf16 -> ~half the memory + enables flash-attention on the LLM). Default to fp32
    # when no dtype is requested (preserves the original numerically-safe behaviour).
    return visual.to(dtype=torch_dtype) if torch_dtype is not None else visual.float()
