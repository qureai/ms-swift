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


def _load_trained_vision_3d_weights(backbone: nn.Module, model_dir: Optional[str]) -> int:
    """Overlay trained `visual_3d.*` weights from a checkpoint dir onto a freshly-built backbone.

    When `--model` points at a checkpoint produced by this fork, its safetensors carry the trained
    `visual_3d.encoder.*` / `visual_3d.proj.*` tensors. The base VLM load drops them (unknown to the
    host architecture), so we build the backbone skeleton, then copy those trained tensors in here.
    Returns the number of tensors loaded (0 => fresh training from a base VLM, nothing to overlay).
    """
    import glob
    import os
    if not model_dir or not os.path.isdir(model_dir):
        return 0
    prefix = 'visual_3d.'
    state = {}
    shards = sorted(glob.glob(os.path.join(model_dir, '*.safetensors')))
    if shards:
        from safetensors import safe_open
        for shard in shards:
            with safe_open(shard, framework='pt') as f:
                for k in f.keys():
                    if k.startswith(prefix):
                        state[k[len(prefix):]] = f.get_tensor(k)
    else:
        for b in sorted(glob.glob(os.path.join(model_dir, 'pytorch_model*.bin'))):
            for k, v in torch.load(b, map_location='cpu').items():
                if k.startswith(prefix):
                    state[k[len(prefix):]] = v
    if not state:
        return 0
    missing, unexpected = backbone.load_state_dict(state, strict=False)
    logger.info(f'vision_3d: overlaid {len(state)} trained visual_3d tensors from checkpoint `{model_dir}` '
                f'(missing={len(missing)}, unexpected={len(unexpected)}).')
    return len(state)


def attach_vision_3d_encoder(model: nn.Module, processor, args) -> Vision3DBackbone:
    """Load `args.vision_3d_model`'s vision encoder, make it 3D, and set it as `model.visual_3d`.

    Architecture parameters (in_channels, encoder_dim, max_tokens, temporal_patch_size, volume size)
    are taken from a `vision_3d_config` saved on `model.config` when present (i.e. reloading a trained
    2D+3D checkpoint), so the rebuilt skeleton matches the saved weights exactly; otherwise they come
    from the CLI args (fresh training). After the skeleton is built, trained `visual_3d.*` weights are
    overlaid from the checkpoint dir if present.
    """
    saved = getattr(getattr(model, 'config', None), 'vision_3d_config', None) or {}
    logger.info(f'vision_3d: attaching secondary 3D vision encoder from `{args.vision_3d_model}` '
                f'(trust_remote_code={args.vision_3d_trust_remote_code}, '
                f'from_saved_config={bool(saved)}).')

    source = load_vision_3d_source_model(
        args.vision_3d_model,
        trust_remote_code=args.vision_3d_trust_remote_code,
        torch_dtype=args.torch_dtype,
        use_hf=getattr(args, 'use_hf', None),
        hub_token=getattr(args, 'hub_token', None),
    )
    tower = extract_vision_tower(source, module_path=args.vision_3d_module_path, model_type=None)

    # Number of input channels = number of CT windowing channels (set by the template args; default 1).
    # On reload, prefer the value baked into the saved config so the patch embed matches the weights.
    in_channels = int(saved.get('in_channels') or getattr(args, 'ct_num_channels', None) or 1)
    temporal_patch_size = int(saved.get('temporal_patch_size') or DEFAULT_TEMPORAL_PATCH_SIZE)
    max_tokens = args.vision_3d_max_tokens if args.vision_3d_max_tokens is not None else saved.get('vision_3d_max_tokens')
    ensure_conv3d_patch_embed(
        tower,
        in_channels=in_channels,
        temporal_patch_size=temporal_patch_size,
        inflate=args.vision_3d_inflate_weights,
    )

    llm_hidden = get_llm_hidden_size(model)

    # Atlas towers pool to a global vector by default; use the AtlasAdapter to tap per-patch tokens.
    from .atlas import AtlasAdapter, atlas_modality, atlas_token_dim, is_atlas_tower
    from .qwen_vl import QwenVLVideoAdapter, is_qwen_vl_tower, qwen_vl_token_dim
    adapter = None
    adapter_type = None
    if is_atlas_tower(tower):
        adapter_type = 'atlas'
        adapter = AtlasAdapter(atlas_modality(tower))
        # Atlas's multiscale layout is derived from the modality's configured image_size, so it must
        # match the actual input size produced by the template (`--ct_volume_size`). Sync it here.
        vol_size = getattr(args, 'ct_volume_size', None) or saved.get('ct_volume_size')
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
        adapter_type = 'qwen_vl'
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
    # On reload, the saved encoder_dim wins so the projector shape matches the trained proj weights.
    enc_dim = int(saved.get('encoder_dim') or enc_dim)

    # Token aggregator: 'none' -> mean-pool + MLP projector (default); 'perceiver' -> learned resampler.
    # On reload the saved settings win so the rebuilt module matches the trained weights.
    resampler_type = saved.get('resampler') or getattr(args, 'vision_3d_resampler', 'none')
    resampler_depth = int(saved.get('resampler_depth') or getattr(args, 'vision_3d_resampler_depth', 2))
    resampler_heads = int(saved.get('resampler_heads') or getattr(args, 'vision_3d_resampler_heads', 8))
    resampler = None
    if resampler_type == 'perceiver':
        from .resampler import PerceiverResampler
        resampler = PerceiverResampler(
            input_dim=enc_dim, output_dim=llm_hidden, num_latents=max_tokens,
            depth=resampler_depth, num_heads=resampler_heads)
        logger.info(f'vision_3d: token aggregator = PerceiverResampler (latents={max_tokens}, '
                    f'depth={resampler_depth}, heads={resampler_heads}); replaces mean-pool + MLP projector.')
    else:
        logger.info(f'vision_3d: projector {enc_dim} -> {llm_hidden} (in_channels={in_channels}, '
                    f'max_tokens={max_tokens}).')

    backbone = Vision3DBackbone(
        tower,
        encoder_dim=enc_dim,
        llm_hidden_size=llm_hidden,
        max_tokens=max_tokens,
        adapter=adapter,
        resampler=resampler,
    )
    target_dtype = getattr(model, 'dtype', None) or args.torch_dtype or torch.float32
    try:
        backbone = backbone.to(device=model.device)
    except Exception:  # device_map / sharded models may not expose a single .device
        pass
    # The projector/resampler must match the LLM dtype; the encoder keeps its own (e.g. Atlas float32,
    # and its output is cast to the aggregator dtype before projecting/resampling).
    backbone.proj.to(dtype=target_dtype)
    if backbone.resampler is not None:
        backbone.resampler.to(dtype=target_dtype)

    # Reloading a trained checkpoint: overlay the trained visual_3d weights on top of the skeleton
    # (which currently holds freshly-loaded pretrained-encoder + randomly-initialised projector weights).
    _load_trained_vision_3d_weights(backbone, getattr(args, 'model_dir', None))

    model.visual_3d = backbone
    # Record how the 3D encoder was built so the checkpoint is self-describing: this dict is written to
    # config.json by save_pretrained, and read back above (`saved`) when the checkpoint is reloaded.
    if getattr(model, 'config', None) is not None:
        model.config.vision_3d_config = {
            'vision_3d_model': args.vision_3d_model,
            'vision_3d_module_path': args.vision_3d_module_path,
            'vision_3d_trust_remote_code': bool(args.vision_3d_trust_remote_code),
            'vision_3d_inflate_weights': bool(args.vision_3d_inflate_weights),
            'vision_3d_max_tokens': max_tokens,
            'in_channels': in_channels,
            'encoder_dim': enc_dim,
            'llm_hidden_size': int(llm_hidden),
            'adapter_type': adapter_type,
            'resampler': resampler_type,
            'resampler_depth': resampler_depth,
            'resampler_heads': resampler_heads,
            'temporal_patch_size': temporal_patch_size,
            'ct_volume_size': getattr(args, 'ct_volume_size', None) or saved.get('ct_volume_size'),
            'ct_windows': list(getattr(args, 'ct_windows', None) or []) or saved.get('ct_windows'),
            'ct_window_base': getattr(args, 'ct_window_base', None) or saved.get('ct_window_base'),
        }
    # release any non-vision parts of the source model (e.g. a full VLM's LLM tower)
    del source
    logger.info(f'vision_3d: attached model.visual_3d = {backbone.__class__.__name__}'
                f'(encoder={backbone.encoder.__class__.__name__}).')
    return backbone
