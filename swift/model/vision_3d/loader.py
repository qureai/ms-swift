# Copyright (c) ModelScope Contributors. All rights reserved.
"""Load a model by name and extract its vision encoder, to be reused as a 3D backbone.

`--vision_3d_model` may point at:
  * a full VLM (e.g. Qwen2.5-VL, InternVL) -> we extract just its vision tower, or
  * a dedicated vision/3D encoder (e.g. YalaLab/Pillar0-ChestCT, custom-code) -> used whole.

Resolution of *where* the vision tower lives is tiered:
  1. explicit `--vision_3d_module_path` (dotted attribute path) — always wins;
  2. a registry keyed by architecture / model_type;
  3. a class-name heuristic (shallowest submodule whose class name looks vision-towerish).
"""
from typing import Optional

import torch
import torch.nn as nn

from swift.utils import get_logger

logger = get_logger()

# model_type / lowercased-arch substring -> dotted attribute path of the vision tower
VISION_TOWER_ATTRS = {
    'qwen2_vl': 'visual',
    'qwen2_5_vl': 'visual',
    'qwen3_vl': 'visual',
    'llava': 'vision_tower',
    'llava_next': 'vision_tower',
    'paligemma': 'vision_tower',
    'gemma3': 'vision_tower',
    'internvl': 'vision_model',
    'internvl_chat': 'vision_model',
    'clip': 'vision_model',
    'siglip': 'vision_model',
}

# class-name substrings that mark a vision tower (used by the heuristic fallback)
VISION_CLASS_HINTS = ('VisionTransformer', 'VisionModel', 'VisionTower', 'VisionEncoder', 'Atlas', 'ImageEncoder')


def _resolve_model_path(model_id_or_path: str, *, use_hf: Optional[bool], hub_token: Optional[str]) -> str:
    """Resolve a hub id to a local dir (downloading if needed); pass local paths through."""
    import os
    if os.path.isdir(model_id_or_path):
        return model_id_or_path
    from huggingface_hub import snapshot_download
    return snapshot_download(model_id_or_path, token=hub_token)


def load_vision_3d_source_model(model_id_or_path: str,
                                *,
                                trust_remote_code: bool = False,
                                torch_dtype: Optional[torch.dtype] = None,
                                use_hf: Optional[bool] = None,
                                hub_token: Optional[str] = None) -> nn.Module:
    """Load the source model that holds the 3D/vision encoder.

    Tries several Auto* classes in order so that both full VLMs and bare vision encoders load. The
    full object is returned; the vision tower is extracted separately by `extract_vision_tower`.
    """
    import transformers
    path = _resolve_model_path(model_id_or_path, use_hf=use_hf, hub_token=hub_token)

    # Pillar0/Atlas (CLIP-style, custom code): build only the vision tower (the text side points at a
    # relative cached-embedding path and would fail / is unwanted).
    from transformers import AutoConfig
    try:
        config = AutoConfig.from_pretrained(path, trust_remote_code=trust_remote_code)
    except Exception as e:  # noqa
        config = None
        logger.info(f'vision_3d: AutoConfig load failed ({type(e).__name__}); proceeding without config branch.')
    if config is not None:
        from .atlas import build_atlas_vision_tower, is_atlas_config
        if is_atlas_config(config):
            logger.info('vision_3d: detected Atlas (clip_multimodal_atlas); building vision tower only.')
            return build_atlas_vision_tower(path, config, torch_dtype=torch_dtype)

    common = dict(trust_remote_code=trust_remote_code, torch_dtype=torch_dtype, low_cpu_mem_usage=True)

    # For custom-code models, AutoModel + auto_map is the right entry. For standard VLMs, the
    # image-text-to-text / vision2seq classes expose the vision tower; AutoModel is the last resort.
    if trust_remote_code:
        order = ['AutoModel', 'AutoModelForImageTextToText', 'AutoModelForVision2Seq', 'AutoModelForCausalLM']
    else:
        order = ['AutoModelForImageTextToText', 'AutoModelForVision2Seq', 'AutoModelForCausalLM', 'AutoModel']

    last_err = None
    for cls_name in order:
        cls = getattr(transformers, cls_name, None)
        if cls is None:
            continue
        try:
            model = cls.from_pretrained(path, **common)
            logger.info(f'vision_3d: loaded source model `{model_id_or_path}` via {cls_name} '
                        f'({model.__class__.__name__}).')
            return model
        except Exception as e:  # noqa
            last_err = e
            logger.info(f'vision_3d: {cls_name}.from_pretrained failed ({type(e).__name__}); trying next.')
    raise RuntimeError(f'vision_3d: could not load `{model_id_or_path}` with any Auto* class.') from last_err


def _get_submodule(root: nn.Module, dotted: str) -> Optional[nn.Module]:
    obj = root
    for p in dotted.split('.'):
        if p == '':
            continue
        if p.isdigit():
            try:
                obj = obj[int(p)]
            except Exception:
                return None
        elif hasattr(obj, p):
            obj = getattr(obj, p)
        else:
            return None
    return obj if isinstance(obj, nn.Module) else None


def _heuristic_find_vision_tower(model: nn.Module) -> Optional[nn.Module]:
    """Return the shallowest submodule whose class name matches a vision-tower hint."""
    best = None
    best_depth = None
    for name, m in model.named_modules():
        if name == '':
            continue
        cls_name = m.__class__.__name__
        if any(h in cls_name for h in VISION_CLASS_HINTS):
            depth = name.count('.')
            if best_depth is None or depth < best_depth:
                best, best_depth = m, depth
    return best


def extract_vision_tower(model: nn.Module,
                         *,
                         module_path: Optional[str] = None,
                         model_type: Optional[str] = None) -> nn.Module:
    """Extract the vision tower submodule from a loaded model (see module docstring for tiers)."""
    # Tier 1: explicit override
    if module_path:
        tower = _get_submodule(model, module_path)
        if tower is None:
            raise ValueError(f'vision_3d: --vision_3d_module_path `{module_path}` did not resolve to an nn.Module '
                             f'inside {model.__class__.__name__}.')
        logger.info(f'vision_3d: extracted vision tower via module_path `{module_path}` ({tower.__class__.__name__}).')
        return tower

    # The loaded object may already BE the vision tower (e.g. an Atlas tower built directly).
    if any(h in model.__class__.__name__ for h in VISION_CLASS_HINTS):
        logger.info(f'vision_3d: loaded model is itself a vision tower ({model.__class__.__name__}); using as-is.')
        return model

    # Tier 2: architecture registry (match by model_type or lowercased class-name substring)
    candidates = []
    if model_type:
        candidates.append(model_type)
    cls_lower = model.__class__.__name__.lower()
    for key, attr in VISION_TOWER_ATTRS.items():
        if (model_type and key == model_type) or key.replace('_', '') in cls_lower.replace('_', ''):
            tower = _get_submodule(model, attr)
            if tower is not None:
                logger.info(f'vision_3d: extracted vision tower via registry key `{key}` -> `{attr}` '
                            f'({tower.__class__.__name__}).')
                return tower

    # Tier 3: class-name heuristic
    tower = _heuristic_find_vision_tower(model)
    if tower is not None:
        logger.info(f'vision_3d: extracted vision tower via class-name heuristic ({tower.__class__.__name__}).')
        return tower

    # Last resort: maybe the loaded object IS the encoder
    logger.warning('vision_3d: could not locate a distinct vision tower; using the loaded model as-is. '
                   'Pass --vision_3d_module_path if this is wrong.')
    return model
