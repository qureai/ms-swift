# Copyright (c) ModelScope Contributors. All rights reserved.
"""Orchestration: load + extract + 3D-ify a vision encoder and attach it as `model.visual_3d`."""
from typing import Optional

import torch
import torch.nn as nn

from swift.utils import get_logger
from .backbone import Vision3DBackbone
from .conv_inflate import ensure_conv3d_patch_embed
from .loader import extract_vision_tower, load_vision_3d_source_model

logger = get_logger()

# Temporal kernel/stride used when converting a 2D patch-embed conv to 3D.
DEFAULT_TEMPORAL_PATCH_SIZE = 2


def get_llm_hidden_size(model: nn.Module) -> int:
    """Find the host LLM hidden size, looking through common nested-config layouts."""
    cfg = getattr(model, 'config', None)
    if cfg is None:
        raise ValueError('vision_3d: host model has no `config`; cannot determine LLM hidden size.')
    if getattr(cfg, 'hidden_size', None):
        return cfg.hidden_size
    for sub in ('text_config', 'llm_config', 'language_config'):
        sub_cfg = getattr(cfg, sub, None)
        if sub_cfg is not None and getattr(sub_cfg, 'hidden_size', None):
            return sub_cfg.hidden_size
    raise ValueError('vision_3d: could not determine host LLM hidden_size from model config.')


def infer_encoder_dim(tower: nn.Module) -> Optional[int]:
    """Infer the vision tower's output feature dim from common config/attribute names."""
    cfg = getattr(tower, 'config', None)
    attr_names = ('out_hidden_size', 'output_dim', 'projection_dim', 'hidden_size', 'embed_dim', 'num_features',
                  'width')
    for attr in attr_names:
        if cfg is not None and getattr(cfg, attr, None):
            return int(getattr(cfg, attr))
        val = getattr(tower, attr, None)
        if isinstance(val, int):
            return val
    return None


def attach_vision_3d_encoder(model: nn.Module, processor, args) -> Vision3DBackbone:
    """Load `args.vision_3d_model`'s vision encoder, make it 3D, and set it as `model.visual_3d`."""
    logger.info(f'vision_3d: attaching secondary 3D vision encoder from `{args.vision_3d_model}` '
                f'(trust_remote_code={args.vision_3d_trust_remote_code}).')

    source = load_vision_3d_source_model(
        args.vision_3d_model,
        trust_remote_code=args.vision_3d_trust_remote_code,
        torch_dtype=args.torch_dtype,
        use_hf=getattr(args, 'use_hf', None),
        hub_token=getattr(args, 'hub_token', None),
    )
    tower = extract_vision_tower(source, module_path=args.vision_3d_module_path, model_type=None)

    # Number of input channels = number of CT windowing channels (set by the template args; default 1).
    in_channels = int(getattr(args, 'ct_num_channels', None) or 1)
    ensure_conv3d_patch_embed(
        tower,
        in_channels=in_channels,
        temporal_patch_size=DEFAULT_TEMPORAL_PATCH_SIZE,
        inflate=args.vision_3d_inflate_weights,
    )

    llm_hidden = get_llm_hidden_size(model)

    # Atlas towers pool to a global vector by default; use the AtlasAdapter to tap per-patch tokens.
    from .atlas import AtlasAdapter, atlas_modality, atlas_token_dim, is_atlas_tower
    from .qwen_vl import QwenVLVideoAdapter, is_qwen_vl_tower, qwen_vl_token_dim
    adapter = None
    if is_atlas_tower(tower):
        adapter = AtlasAdapter(atlas_modality(tower))
        # Atlas's multiscale layout is derived from the modality's configured image_size, so it must
        # match the actual input size produced by the template (`--ct_volume_size`). Sync it here.
        vol_size = getattr(args, 'ct_volume_size', None)
        if vol_size:
            from swift.utils.ct_volume_io import parse_volume_size
            size = list(parse_volume_size(vol_size))
            tower.model_config['modalities'][adapter.modality]['image_size'] = size
            logger.info(f'vision_3d/atlas: set encoder image_size={size} to match --ct_volume_size '
                        '(must be large enough to yield the checkpoint\'s scale count, e.g. >=128^3).')
        else:
            logger.warning('vision_3d/atlas: --ct_volume_size not set; using the encoder\'s native image_size. '
                           'Set --ct_volume_size to control volume resolution (and keep >=128^3 for Atlas).')
        enc_dim = atlas_token_dim(tower) or infer_encoder_dim(tower) or llm_hidden
    elif is_qwen_vl_tower(tower):
        # Photon-style: drive a Qwen-VL vision tower on volumes patchified as video.
        adapter = QwenVLVideoAdapter.from_tower(tower)
        enc_dim = qwen_vl_token_dim(tower) or infer_encoder_dim(tower) or llm_hidden
        logger.info(f'vision_3d: using QwenVLVideoAdapter (patch_size={adapter.patch_size}, '
                    f'temporal_patch_size={adapter.temporal_patch_size}, merge_size={adapter.merge_size}); '
                    f'token_dim={enc_dim}. Ensure --ct_volume_size dims are divisible: T%tps==0, H,W%(ps*merge)==0.')
    else:
        enc_dim = infer_encoder_dim(tower)
        if enc_dim is None:
            logger.warning(f'vision_3d: could not infer encoder output dim; defaulting projector input to LLM hidden '
                           f'size ({llm_hidden}).')
            enc_dim = llm_hidden
    logger.info(f'vision_3d: projector {enc_dim} -> {llm_hidden} (in_channels={in_channels}, '
                f'max_tokens={args.vision_3d_max_tokens}).')

    backbone = Vision3DBackbone(
        tower,
        encoder_dim=enc_dim,
        llm_hidden_size=llm_hidden,
        max_tokens=args.vision_3d_max_tokens,
        adapter=adapter,
    )
    target_dtype = getattr(model, 'dtype', None) or args.torch_dtype or torch.float32
    try:
        backbone = backbone.to(device=model.device)
    except Exception:  # device_map / sharded models may not expose a single .device
        pass
    # The projector must match the LLM dtype; the encoder keeps its own (e.g. Atlas stays float32,
    # and Vision3DBackbone casts the encoder output to the projector dtype before projecting).
    backbone.proj.to(dtype=target_dtype)

    model.visual_3d = backbone
    # release any non-vision parts of the source model (e.g. a full VLM's LLM tower)
    del source
    logger.info(f'vision_3d: attached model.visual_3d = {backbone.__class__.__name__}'
                f'(encoder={backbone.encoder.__class__.__name__}).')
    return backbone
