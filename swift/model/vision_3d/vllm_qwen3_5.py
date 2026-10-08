# Copyright (c) ModelScope Contributors. All rights reserved.
"""vLLM integration for the secondary 3D CT vision tower (``model.visual_3d``).

vanilla vLLM's ``Qwen3_5ForConditionalGeneration`` only knows the 2D ``visual`` tower + the
``language_model``; it has no module for the Qure ``visual_3d.*`` weights and its weight loader
raises on them. This module defines a *drop-in subclass* that:

  * builds ``self.visual_3d`` from the checkpoint's ``vision_3d_config`` (same construction path as
    the HF trainer -- `Qwen3_5VisionModel` + Conv3d-inflated patch embed + STA -- but rebuilt from
    the host ``vision_config`` instead of re-downloading the 8GB source model) and overlays the
    trained ``visual_3d.*`` weights from the checkpoint dir, so **the 3D weights are loaded**; and
  * mirrors the 2D ``visual`` multimodal flow for a ``volume`` modality: CT volumes (patchified as
    Qwen video patches + a ``volume_grid_thw``) are routed through ``visual_3d``, split per-volume,
    and merged into the ``<video>`` placeholder positions -- exactly how ``_process_video_input``
    handles ``self.visual``.

Backwards compatibility (important): :func:`register_vision_3d_vllm_models` only overrides the vLLM
model registry **when the checkpoint config actually carries ``vision_3d_config``**. A normal 2D-only
Qwen3.5 checkpoint is never touched -- it keeps the pristine upstream class, and nothing 3D is built
or loaded.

NOTE (input pipeline): the model-side ``volume`` forward path below is complete and mirrors video.
Feeding CT volumes end-to-end through a vLLM *request* additionally needs (a) the multimodal
*processor* to declare the ``volume`` modality and (b) swift's rollout template to emit volumes to
vLLM. Those are only exercised by CT-volume RL/inference (not 2D-image rollout) and are added
alongside that data. Until then this subclass behaves byte-for-byte like the vanilla class for
image/video/text, while keeping the trained 3D tower loaded and ready.
"""
from typing import Iterable, Optional

import torch
from torch import nn

from swift.utils import get_logger

logger = get_logger()


# --------------------------------------------------------------------------------------------------
# Build visual_3d from the checkpoint's saved vision_3d_config (no source-model download).
# --------------------------------------------------------------------------------------------------
from contextlib import contextmanager


def _read_visual_3d_state(model_dir: Optional[str]) -> dict:
    """Read the trained ``visual_3d.*`` tensors from a checkpoint dir (safetensors or .bin), with the
    ``visual_3d.`` prefix stripped so keys match ``Vision3DBackbone`` param/buffer names."""
    import glob
    import os
    state = {}
    if not model_dir or not os.path.isdir(model_dir):
        return state
    prefix = 'visual_3d.'
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
    return state


def _install_trained_visual_3d_weights(backbone, model_dir: Optional[str]) -> int:
    """Install trained ``visual_3d.*`` weights by direct ``.data`` assignment (robust to empty / meta
    param data left by vLLM's build context). Returns the number of tensors installed."""
    state = _read_visual_3d_state(model_dir)
    if not state:
        return 0
    targets = dict(backbone.named_parameters())
    targets.update(dict(backbone.named_buffers()))
    installed, missing, unexpected = 0, [], []
    for k, tgt in targets.items():
        src = state.get(k)
        if src is None:
            missing.append(k)
            continue
        tgt.data = src.to(dtype=tgt.dtype) if tgt.dtype.is_floating_point else src
        installed += 1
    unexpected = [k for k in state if k not in targets]
    logger.info('vLLM visual_3d: installed %d trained tensors from `%s` (missing=%d, unexpected=%d).',
                installed, model_dir, len(missing), len(unexpected))
    if missing:
        logger.warning('vLLM visual_3d: %d params had no checkpoint tensor (sample: %s).',
                       len(missing), missing[:5])
    return installed


@contextmanager
def _exclude_from_deepspeed_zero3():
    """Build modules with real (unpartitioned) params even under an active ZeRO-3 ``zero.Init()``
    monkeypatch. ``Init(enabled=False)`` becomes the top-of-stack context, so modules created inside
    skip ZeRO-3 partitioning.

    We do NOT gate on ``is_deepspeed_zero3_enabled()`` -- swift configures ZeRO-3 directly via
    deepspeed (not HF's TrainerDeepSpeedConfig), so that check can be False while ``zero.Init`` is
    still active. Entering is wrapped so it safely no-ops when there is no deepspeed runtime (e.g. a
    plain CPU build)."""
    ctx = None
    try:
        import deepspeed
        ctx = deepspeed.zero.Init(enabled=False)
        ctx.__enter__()
    except Exception:  # noqa -- deepspeed absent / not initialised: nothing to exclude from
        ctx = None
    try:
        yield
    finally:
        if ctx is not None:
            try:
                ctx.__exit__(None, None, None)
            except Exception:  # noqa
                pass


def build_visual_3d_from_config(hf_config, model_dir: Optional[str], device, dtype):
    """Rebuild the ``Vision3DBackbone`` (== ``model.visual_3d``) from ``hf_config.vision_3d_config``.

    Mirrors :func:`swift.model.vision_3d.attach.attach_vision_3d_encoder`, but constructs the tower
    skeleton from the host ``vision_config`` (architecturally identical to the recorded
    ``vision_3d_model`` for same-family checkpoints, e.g. Qwen3.5) rather than loading the source
    model -- so it is self-contained and cheap. The trained ``visual_3d.*`` weights from
    ``model_dir`` are then overlaid, fully replacing the freshly-initialised skeleton.
    """
    from .backbone import Vision3DBackbone
    from .conv_inflate import ensure_conv3d_patch_embed
    from .qwen_vl import QwenVLVideoAdapter, is_qwen_vl_tower

    v3d = dict(getattr(hf_config, 'vision_3d_config', None) or {})
    if not v3d:
        raise ValueError('build_visual_3d_from_config: hf_config has no vision_3d_config.')

    adapter_type = v3d.get('adapter_type')
    resampler_type = v3d.get('resampler') or 'none'
    if adapter_type != 'qwen_vl' or resampler_type != 'none':
        # Only the Qwen-VL video adapter with the mean-pool/variable projector is supported in vLLM
        # today; Atlas / Perceiver-resampler 3D encoders need their own vLLM port.
        raise NotImplementedError(
            f'vLLM visual_3d: only adapter_type="qwen_vl" + resampler="none" is supported '
            f'(got adapter_type={adapter_type!r}, resampler={resampler_type!r}).')

    # Build the whole skeleton on CPU, EXCLUDED from deepspeed ZeRO-3. In colocate GRPO the vLLM
    # engine shares the process with a ZeRO-3 policy; ZeRO-3's `zero.Init()` globally hooks module
    # construction and would partition visual_3d's params to empty ([0]) shards -> state_dict load
    # fails. visual_3d is not a deepspeed-managed module, so build it fully materialised.
    with _exclude_from_deepspeed_zero3(), torch.device('cpu'):
        # Tower skeleton == the host's own Qwen3.5 vision architecture (same class as the 2D `visual`).
        from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5VisionModel
        tower = Qwen3_5VisionModel(hf_config.vision_config)

        # 2D patch-embed conv -> Conv3d with the CT windowing-channel count (weights overwritten below).
        ensure_conv3d_patch_embed(
            tower,
            in_channels=int(v3d['in_channels']),
            temporal_patch_size=int(v3d.get('temporal_patch_size') or 2),
            inflate=bool(v3d.get('vision_3d_inflate_weights', True)),
        )

        # Spatial-Temporal Attention (z-axis) -- must be installed before loading trained weights so the
        # temporal params exist to receive them.
        if v3d.get('sta_enabled'):
            from .sta import install_sta
            if not is_qwen_vl_tower(tower):
                raise ValueError('vLLM visual_3d: sta_enabled but the rebuilt tower is not a Qwen-VL tower.')
            install_sta(tower, num_temporal_heads=v3d.get('sta_num_temporal_heads'))

        adapter = QwenVLVideoAdapter.from_tower(tower)
        backbone = Vision3DBackbone(
            tower,
            encoder_dim=int(v3d['encoder_dim']),
            llm_hidden_size=int(v3d['llm_hidden_size']),
            max_tokens=v3d.get('vision_3d_max_tokens'),
            adapter=adapter,
            resampler=None,
            variable_tokens=bool(v3d.get('variable_tokens', False)),
        )

    # Install the trained visual_3d.* weights by DIRECT `.data` assignment (not load_state_dict).
    # vLLM builds models under a context that can leave params empty (shape [0]) to be filled by its
    # own weight loader; load_state_dict then fails with size mismatch. Assigning the checkpoint tensor
    # straight into `param.data` materialises each param to the correct shape, robust to empty / meta /
    # partitioned initial states. visual_3d is not a deepspeed-managed module, so full params are right.
    n = _install_trained_visual_3d_weights(backbone, model_dir)
    if n == 0:
        logger.warning(
            'vLLM visual_3d: no trained visual_3d.* weights found in `%s`; the 3D tower is running '
            'with freshly-initialised weights.', model_dir)

    backbone = backbone.to(device=device)
    # projector matches LLM dtype; the encoder keeps its own (cast at aggregation time in the backbone).
    backbone.proj.to(dtype=dtype)
    if backbone.resampler is not None:
        backbone.resampler.to(dtype=dtype)
    logger.info('vLLM visual_3d: built + loaded model.visual_3d = %s(encoder=%s) on %s (%d tensors overlaid).',
                backbone.__class__.__name__, backbone.encoder.__class__.__name__, device, n)
    return backbone


# --------------------------------------------------------------------------------------------------
# Custom vLLM model: vanilla Qwen3.5 + a `volume` modality routed through visual_3d.
# --------------------------------------------------------------------------------------------------
def _make_qwen3_5_with_3d_cls():
    """Build the subclass lazily so this module imports without vLLM present (e.g. in swift_july)."""
    from vllm.model_executor.models.qwen3_5 import (Qwen3_5ForConditionalGeneration, Qwen3_5ProcessingInfo)
    from vllm.model_executor.models.qwen3_vl import (Qwen3VLDummyInputsBuilder, Qwen3VLMultiModalProcessor)
    from vllm.model_executor.models.utils import AutoWeightsLoader, maybe_prefix  # noqa: F401
    from vllm.multimodal import MULTIMODAL_REGISTRY

    @MULTIMODAL_REGISTRY.register_processor(
        Qwen3VLMultiModalProcessor,
        info=Qwen3_5ProcessingInfo,
        dummy_inputs=Qwen3VLDummyInputsBuilder,
    )
    class Qwen3_5With3DForConditionalGeneration(Qwen3_5ForConditionalGeneration):
        """Qwen3.5 VL + secondary CT (`visual_3d`) tower. Registered only for 3D checkpoints."""

        def __init__(self, *, vllm_config, prefix: str = 'model'):
            super().__init__(vllm_config=vllm_config, prefix=prefix)
            hf_config = vllm_config.model_config.hf_config
            model_dir = vllm_config.model_config.model
            # BACKWARDS COMPATIBILITY: only build the 3D tower for a 2D+3D checkpoint. For a normal
            # 2D-only Qwen3.5 checkpoint (no vision_3d_config) this subclass is byte-for-byte the
            # vanilla class -- nothing 3D is constructed or loaded.
            self._has_3d = bool(getattr(hf_config, 'vision_3d_config', None))
            if self._has_3d:
                # place visual_3d on the same device/dtype as the (already built) 2D visual tower
                ref_param = next(self.visual.parameters())
                self.visual_3d = build_visual_3d_from_config(
                    hf_config, model_dir, device=ref_param.device, dtype=ref_param.dtype)
            self._video_token_id = getattr(hf_config, 'video_token_id', None)

        # -- weight loading: visual_3d.* is loaded from disk in __init__, so skip it in the stream ----
        def load_weights(self, weights: Iterable[tuple]) -> set:
            # Ignore SFT-only auxiliary prefixes that vanilla vLLM has no module for and that are not
            # needed for text generation / rollout:
            #   visual_3d.*        -> secondary CT tower (overlaid from disk in __init__ when 3D)
            #   asl_loss.*         -> Asymmetric-Loss classification head (training-time only)
            #   ml_decoder_head.*  -> ML-Decoder multi-label classification head (training-time only)
            # All are no-ops on a truly vanilla checkpoint (no such tensors present). (The `model.`
            # prefix and `lm_head` are the real weights; vLLM's hf_to_vllm_mapper remaps `model.` ->
            # `visual`/`language_model`.)
            loader = AutoWeightsLoader(
                self,
                skip_prefixes=['mtp.'],
                ignore_unexpected_prefixes=['visual_3d.', 'asl_loss.', 'ml_decoder_head.'],
            )
            return loader.load_weights(weights, mapper=self.hf_to_vllm_mapper)

        # -- volume modality: mirror _process_video_input, but through visual_3d --------------------
        def _parse_and_validate_volume_input(self, **kwargs):
            pixel_values_volumes = kwargs.get('pixel_values_volumes', None)
            volume_grid_thw = kwargs.get('volume_grid_thw', None)
            if pixel_values_volumes is None:
                return None
            if volume_grid_thw is None:
                raise ValueError('visual_3d: pixel_values_volumes given without volume_grid_thw.')
            return {'pixel_values_volumes': pixel_values_volumes, 'volume_grid_thw': volume_grid_thw}

        def _process_volume_input(self, volume_input) -> tuple:
            """Patchified volumes + grid -> per-volume token embeddings (mirrors _process_video_input)."""
            from .qwen_vl import QwenVLVideoAdapter
            grid_thw = volume_input['volume_grid_thw']
            patches = volume_input['pixel_values_volumes'].type(self.visual_3d.dtype)
            # run the (Conv3d-inflated + STA) Qwen vision tower on the packed patches; the tower builds
            # cu_seqlens from grid_thw so attention is block-diagonal per volume (same as the adapter's
            # batched path), then project the post-merger tokens into the LLM subspace.
            tower_out = self.visual_3d.encoder(patches, grid_thw=grid_thw)
            tokens = QwenVLVideoAdapter._tower_tokens(tower_out)
            volume_embeds = self.visual_3d.proj(tokens.to(self.visual_3d.dtype))
            merge = self.visual.spatial_merge_size
            sizes = (grid_thw.prod(-1) // merge // merge).tolist()
            return tuple(volume_embeds.split(sizes))

        def embed_multimodal(self, **kwargs):
            volume_input = self._parse_and_validate_volume_input(**kwargs) if self._has_3d else None
            # image/video/text: identical to the vanilla class.
            base = super().embed_multimodal(**kwargs)
            if volume_input is None:
                return base
            volume_embeddings = self._process_volume_input(volume_input)
            if base is None:
                return tuple(volume_embeddings)
            return tuple(list(base) + list(volume_embeddings))

    return Qwen3_5With3DForConditionalGeneration


# --------------------------------------------------------------------------------------------------
# Conditional registration -- called before the vLLM engine is built.
# --------------------------------------------------------------------------------------------------
_REGISTERED = False


def register_qwen3_5_3d_model() -> bool:
    """Unconditionally override ``Qwen3_5ForConditionalGeneration`` with the 3D-aware subclass.

    Safe for 2D-only checkpoints because the subclass self-gates on ``vision_3d_config`` (builds
    nothing 3D when it is absent). This is the spawn-safe entry point: it is invoked as a vLLM
    ``general_plugin`` so it runs in every vLLM process, including out-of-process EngineCore workers
    (which do NOT inherit a parent process's runtime ``ModelRegistry.register_model`` call).
    """
    global _REGISTERED
    if _REGISTERED:
        return True
    try:
        from vllm import ModelRegistry
        cls = _make_qwen3_5_with_3d_cls()
        ModelRegistry.register_model('Qwen3_5ForConditionalGeneration', cls)
    except Exception as e:  # noqa -- never let a plugin import break vLLM startup
        logger.warning('vLLM visual_3d: could not register the 3D model subclass (%s: %s); '
                       '2D+3D checkpoints will fall back to the vanilla class.', type(e).__name__, e)
        return False
    _REGISTERED = True
    logger.info('vLLM visual_3d: registered %s for Qwen3_5ForConditionalGeneration '
                '(self-gates on vision_3d_config).', cls.__name__)
    return True


def vllm_general_plugin() -> None:
    """vLLM ``general_plugins`` entry point (see setup.py). Runs in every vLLM process."""
    register_qwen3_5_3d_model()


def register_vision_3d_vllm_models(hf_config) -> bool:
    """If ``hf_config`` carries ``vision_3d_config``, override the vLLM model registry so the 3D-aware
    subclass is used for ``Qwen3_5ForConditionalGeneration``. No-op for 2D-only checkpoints.

    Returns True if an override was registered.
    """
    if not getattr(hf_config, 'vision_3d_config', None):
        return False  # 2D-only: leave vanilla vLLM untouched (backwards compatible).
    archs = list(getattr(hf_config, 'architectures', None) or [])
    if 'Qwen3_5ForConditionalGeneration' not in archs:
        logger.info('vLLM visual_3d: vision_3d_config present but architectures=%s is not '
                    'Qwen3_5ForConditionalGeneration; skipping 3D registration.', archs)
        return False
    return register_qwen3_5_3d_model()
