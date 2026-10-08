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
    # Under DeepSpeed ZeRO-3 the freshly-built backbone params are partitioned (numel()==0 shards),
    # so a plain load_state_dict raises a size mismatch. Gather them for the load, then let deepspeed
    # re-partition on context exit. `GatheredParameters` is a no-op for non-ZeRO-3 params.
    zero3_params = [p for p in backbone.parameters() if hasattr(p, 'ds_id')]
    if zero3_params:
        import deepspeed
        with deepspeed.zero.GatheredParameters(zero3_params, modifier_rank=0):
            missing, unexpected = backbone.load_state_dict(state, strict=False)
    else:
        missing, unexpected = backbone.load_state_dict(state, strict=False)
    logger.info(f'vision_3d: overlaid {len(state)} trained visual_3d tensors from checkpoint `{model_dir}` '
                f'(missing={len(missing)}, unexpected={len(unexpected)}, zero3={bool(zero3_params)}).')
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
        attn_impl=getattr(args, 'attn_impl', None),   # FA2/sdpa for the 3D encoder's spatial attention
    )
    tower = extract_vision_tower(source, module_path=args.vision_3d_module_path, model_type=None)

    # Number of input channels = number of CT windowing channels (set by the template args; default 1).
    # On reload, prefer the value baked into the saved config so the patch embed matches the weights.
    in_channels = int(saved.get('in_channels') or getattr(args, 'ct_num_channels', None) or 1)
    temporal_patch_size = int(saved.get('temporal_patch_size') or DEFAULT_TEMPORAL_PATCH_SIZE)
    max_tokens = args.vision_3d_max_tokens if args.vision_3d_max_tokens is not None else saved.get('vision_3d_max_tokens')
    # variable-resolution CT: when --ct_volume_size is unset, each volume keeps its native (variable)
    # token count, capped at `max_tokens`. Only supported for the Qwen-VL 3D encoder, and incompatible
    # with the Perceiver resampler (both asserted below).
    vol_size_cfg = getattr(args, 'ct_volume_size', None) or saved.get('ct_volume_size')
    variable_tokens = vol_size_cfg is None
    # Spatial-Temporal Attention (temporal/z-axis self-attention in the vision blocks); opt-in, Qwen-VL only.
    sta_enabled = bool(getattr(args, 'vision_3d_sta', False)) or bool(saved.get('sta_enabled'))
    sta_num_temporal_heads = (getattr(args, 'vision_3d_sta_num_temporal_heads', None)
                              or saved.get('sta_num_temporal_heads'))
    sta_window = int(getattr(args, 'vision_3d_sta_window', 0) or saved.get('sta_window') or 0)
    # Slow-Fast: enabled by args OR by a reloaded checkpoint's saved config; structure must be rebuilt
    # (proj_fast) BEFORE the trained weights are overlaid on resume.
    slowfast_enabled = bool(getattr(args, 'vision_3d_slowfast', False)) or bool(saved.get('slowfast_enabled'))
    _fg = saved.get('fast_grid') or getattr(args, 'vision_3d_fast_grid', '4,8,8')
    fast_grid = tuple(int(x) for x in (_fg.split(',') if isinstance(_fg, str) else _fg))
    fast_pool = saved.get('fast_pool') or getattr(args, 'vision_3d_fast_pool', 'avgmax')
    slow_max_tokens = int(saved.get('slow_max_tokens') or getattr(args, 'vision_3d_slow_max_tokens', 4096))
    crossattn_every_n = int(saved.get('crossattn_every_n') or getattr(args, 'vision_3d_crossattn_every_n', 2) or 2)
    ensure_conv3d_patch_embed(
        tower,
        in_channels=in_channels,
        temporal_patch_size=temporal_patch_size,
        inflate=args.vision_3d_inflate_weights,
        patch_init=getattr(args, 'vision_3d_patch_init', 'random'),
    )

    llm_hidden = get_llm_hidden_size(model)

    # Atlas towers pool to a global vector by default; use the AtlasAdapter to tap per-patch tokens.
    from .atlas import AtlasAdapter, atlas_modality, atlas_token_dim, is_atlas_tower
    from .qwen_vl import QwenVLVideoAdapter, is_qwen_vl_tower, qwen_vl_token_dim
    adapter = None
    adapter_type = None
    if is_atlas_tower(tower):
        adapter_type = 'atlas'
        if variable_tokens:
            raise ValueError(
                'vision_3d/atlas: the Atlas encoder requires a fixed --ct_volume_size (its multiscale layout '
                'is derived from a configured image_size). Variable-length CT tokens (no --ct_volume_size) '
                'are only supported for the Qwen-VL 3D encoder. Pass --ct_volume_size D,H,W.')
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

    # Spatial-Temporal Attention: inject temporal (z-axis) self-attention into the Qwen-VL vision blocks.
    # Must run BEFORE _load_trained_vision_3d_weights so the temporal params exist to receive trained weights.
    if sta_enabled:
        if adapter_type != 'qwen_vl':
            raise ValueError('--vision_3d_sta (temporal attention) requires a Qwen-VL 3D encoder; got '
                             f'adapter_type={adapter_type!r}. Atlas models it natively (3D convs).')
        from .sta import install_sta
        install_sta(tower, num_temporal_heads=sta_num_temporal_heads, window=sta_window)
        sta_num_temporal_heads = getattr(tower, '_sta_num_temporal_heads', sta_num_temporal_heads)

    # Token aggregator: 'none' -> mean-pool + MLP projector (default); 'perceiver' -> learned resampler.
    # On reload the saved settings win so the rebuilt module matches the trained weights.
    resampler_type = saved.get('resampler') or getattr(args, 'vision_3d_resampler', 'none')
    resampler_depth = int(saved.get('resampler_depth') or getattr(args, 'vision_3d_resampler_depth', 2))
    resampler_heads = int(saved.get('resampler_heads') or getattr(args, 'vision_3d_resampler_heads', 8))
    if variable_tokens and resampler_type != 'none':
        raise ValueError(
            f'vision_3d: variable-length CT tokens (no --ct_volume_size) are incompatible with '
            f'--vision_3d_resampler {resampler_type} -- a resampler emits a FIXED num_latents, which '
            'contradicts per-volume variable token counts. Use --vision_3d_resampler none, or set '
            '--ct_volume_size for fixed-length tokens.')
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
        variable_tokens=variable_tokens,
        slowfast=slowfast_enabled,
        fast_grid=fast_grid,
        fast_pool=fast_pool,
        slow_max_tokens=slow_max_tokens,
    )
    if slowfast_enabled:
        logger.info(f'vision_3d: SLOW-FAST on -> fast_grid={fast_grid} (fast_count={backbone.num_fast_tokens()}), '
                    f'fast_pool={fast_pool}, slow_max_tokens={slow_max_tokens}.')
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
    if backbone.proj_fast is not None:
        backbone.proj_fast.to(dtype=target_dtype)
    # Recipe: zero-init the projector OUTPUT Linear so step-0 sends no gradient into the tower through the
    # projector (done BEFORE the trained-weight overlay, so a resume restores the real weights over the zeros).
    if bool(getattr(args, 'vision_3d_proj_zero_init', False)):
        # BASE/slow projector: zero-init its output so step-0 sends no gradient into the tower through proj.
        if backbone.proj is not None and isinstance(backbone.proj[-1], nn.Linear):
            nn.init.zeros_(backbone.proj[-1].weight)
            if backbone.proj[-1].bias is not None:
                nn.init.zeros_(backbone.proj[-1].bias)
        logger.info('vision_3d: base projector output Linear zero-initialised (proj[-1]).')
    # FAST projector init -- a SEPARATE knob. Zero-init was found to seed a rank-~1 fast channel (the fast
    # <video> tokens collapsed while the pre-proj encoder stayed high-rank); 'orthogonal' starts it full-rank.
    # Unset -> follow proj_zero_init (backward-compatible). Runs BEFORE the trained-weight overlay (resume-safe).
    if backbone.proj_fast is not None and isinstance(backbone.proj_fast[-1], nn.Linear):
        pf_init = str(getattr(args, 'vision_3d_proj_fast_init', '') or '').lower()
        if not pf_init:
            pf_init = 'zero' if bool(getattr(args, 'vision_3d_proj_zero_init', False)) else 'default'
        w = backbone.proj_fast[-1]
        if pf_init == 'orthogonal':
            # orthogonal_ uses QR (geqrf), which is NOT implemented for bf16 on CUDA -> init in fp32, cast back.
            with torch.no_grad():
                o = torch.empty(w.weight.shape, dtype=torch.float32, device=w.weight.device)
                nn.init.orthogonal_(o)
                w.weight.copy_(o.to(w.weight.dtype))
            if w.bias is not None:
                nn.init.zeros_(w.bias)
            logger.info('vision_3d: proj_fast[-1] ORTHOGONAL-initialised (full-rank fast channel).')
        elif pf_init == 'zero':
            nn.init.zeros_(w.weight)
            if w.bias is not None:
                nn.init.zeros_(w.bias)
            logger.info('vision_3d: proj_fast[-1] zero-initialised.')

    # Optional auxiliary ML-Decoder multi-label head on the PRE-projector encoder tokens (ASL grounding
    # objective; a constant/collapsed feature cannot minimise it). Built before the weight overlay so its
    # params exist to receive trained weights on reload. Training-only: lambda 0 => skip entirely.
    aux_lambda = float(getattr(args, 'vision_3d_aux_mldecoder_lambda', 0.0) or 0.0)
    if aux_lambda > 0:
        import pandas as pd
        from .ml_decoder import Vision3DMLDecoderHead
        tags = [t.strip() for t in open(args.vision_3d_aux_tags) if t.strip()]
        backbone.aux_head = Vision3DMLDecoderHead(enc_dim=enc_dim, num_classes=len(tags)).to(
            device=getattr(model, 'device', None) or 'cpu', dtype=target_dtype)
        backbone.aux_lambda = aux_lambda
        tdf = pd.read_parquet(args.vision_3d_aux_label_table)
        table = {}
        for uid, idxs in zip(tdf['series_uid'].tolist(), tdf['tag_idxs'].tolist()):
            vec = torch.zeros(len(tags), dtype=torch.bool)
            vec[list(idxs)] = True
            table[str(uid)] = vec
        backbone.aux_table = table
        logger.info(f'vision_3d: aux ML-Decoder head ON (num_classes={len(tags)}, lambda={aux_lambda}, '
                    f'label table {len(table)} volumes).')

    # Slow-Fast: install the gated cross-attention (SLOW branch) BEFORE the weight overlay. The cross-attn
    # modules live on the backbone (`visual_3d.cross_blocks.*`) via forward hooks on the LLM decoder layers,
    # so they are (a) covered by the `visual_3d.*` overlay below on resume, and (b) targeted by the
    # `--vit_lr`/vision_tower optimizer group -- while the decoder layers' own params are never renamed.
    sf_layer_idxs = None
    if slowfast_enabled:
        from .slowfast import install_slowfast
        sf_layer_idxs = install_slowfast(model, backbone, slow_dim=enc_dim, every_n=crossattn_every_n)
        logger.info(f'vision_3d: slow-fast cross-attention installed on decoder layers {sf_layer_idxs} '
                    f'(slow_dim={enc_dim}, every_n={crossattn_every_n}).')

    # Reloading a trained checkpoint: overlay the trained visual_3d weights on top of the skeleton (which
    # currently holds freshly-loaded pretrained-encoder + randomly-initialised projector + cross_block weights).
    _load_trained_vision_3d_weights(backbone, getattr(args, 'model_dir', None))

    model.visual_3d = backbone
    # Register `visual_3d` as a vision tower on the model's arch keys so swift's `--vit_lr` optimizer
    # group targets the 3D encoder + projector (otherwise the multimodal optimizer, which builds ONLY
    # vit/aligner/llm groups from these prefixes, would leave every `visual_3d.*` param OUT of the
    # optimizer -> the encoder would silently receive no updates when vit_lr/aligner_lr is set).
    # Harmless for non-CT models (no such params). No-op unless model_meta.model_arch is present.
    try:
        arch = getattr(getattr(model, 'model_meta', None), 'model_arch', None)
        if arch is not None and getattr(arch, 'vision_tower', None) is not None:
            vt = arch.vision_tower
            vt = [vt] if isinstance(vt, str) else list(vt)
            if 'visual_3d' not in vt:
                vt.append('visual_3d')
                arch.vision_tower = vt
                logger.info(f'vision_3d: registered `visual_3d` in model_arch.vision_tower -> {vt} '
                            '(so --vit_lr targets the 3D encoder + projector).')
            # Recipe: split the NEW adapter params (proj/proj_fast/cross_blocks) into the `aligner` group so
            # they train at --aligner_lr (HIGH) while the pretrained encoder stays in the vit group at --vit_lr
            # (LOW). The vit group REJECTS aligner-prefixed params, so this is a clean disjoint 2-way split.
            if bool(getattr(args, 'vision_3d_split_lr_groups', False)):
                al = getattr(arch, 'aligner', None)
                al = ([] if al is None else ([al] if isinstance(al, str) else list(al)))
                # HIGH-LR group = the NEW params: inflated patch-embed + projector(s) + cross-attn. The
                # pretrained encoder BLOCKS (and the STA temporal attn nested in them) stay in the vit group.
                # ('visual_3d.proj' also matches 'visual_3d.proj_fast'.)
                for pfx in ('visual_3d.encoder.patch_embed', 'visual_3d.proj', 'visual_3d.cross_blocks'):
                    if pfx not in al:
                        al.append(pfx)
                arch.aligner = al
                logger.info(f'vision_3d: split_lr_groups ON -> new params (patch_embed/proj/cross_blocks) under '
                            f'model_arch.aligner={al} (train at --aligner_lr); pretrained blocks stay at --vit_lr.')
    except Exception as e:  # noqa: BLE001 -- never fail attach over an LR-grouping convenience
        logger.warning(f'vision_3d: could not register visual_3d in model_arch.vision_tower ({e}); '
                       '--vit_lr will NOT apply to the 3D encoder.')
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
            'variable_tokens': variable_tokens,
            'sta_enabled': sta_enabled,
            'sta_num_temporal_heads': sta_num_temporal_heads if sta_enabled else None,
            'sta_window': sta_window if sta_enabled else None,
            'slowfast_enabled': slowfast_enabled,
            'fast_grid': list(fast_grid) if slowfast_enabled else None,
            'fast_pool': fast_pool if slowfast_enabled else None,
            'slow_max_tokens': slow_max_tokens if slowfast_enabled else None,
            'crossattn_every_n': crossattn_every_n if slowfast_enabled else None,
            'crossattn_layer_idxs': sf_layer_idxs,
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
