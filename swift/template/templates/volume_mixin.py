# Copyright (c) ModelScope Contributors. All rights reserved.
"""`Volume3DTemplateMixin`: adds CT-volume support to a 2D VL template.

Volumes ride the existing `<video>` channel (no new token). Compose this mixin onto a host VL
template, e.g. `class Qwen2_5_VL_CT_Template(Volume3DTemplateMixin, Qwen2_5VLTemplate): ...`.

Host-agnostic responsibilities live here: detecting + loading CT volumes into `(T, C, H, W)` tensors
(multi-channel HU windowing), batching them in the collator, and the 3D routing
(`_splice_volume_embeds`: volumes -> `model.visual_3d`, spliced into `<video>` token positions, with a
zeroed dummy forward when a batch has no volumes so the encoder stays in the autograd graph).

`_post_encode` here is the Qwen2.5-VL style (swift splices *both* image and volume features into
`inputs_embeds`). Hosts whose HF model merges vision internally (e.g. Qwen3-VL with deepstack) provide
their own `_post_encode` (splice volumes only, keep `pixel_values`). Per-token-expansion / rope lives
in each host template's `_encode`.

Set env `VISION_3D_DEBUG=1` to log, from the actual forward pass: each vision encoder's output shape
(via forward hooks) and image/video token routing per batch.
"""
import os
from typing import Any, Dict, List

import torch

from swift.utils import get_logger
from swift.utils.ct_volume_io import load_ct_volume, parse_volume_size

logger = get_logger()


class Volume3DTemplateMixin:
    # video paths with these extensions are treated as CT volumes (routed to the 3D encoder)
    volume_extensions = ('.nii', '.nii.gz', '.npy', '.npz', '.safetensors')

    # ---- volume detection & loading -------------------------------------------------------------
    def is_volume_path(self, video: Any) -> bool:
        return isinstance(video, str) and any(video.lower().endswith(ext) for ext in self.volume_extensions)

    def load_volume(self, path: str) -> torch.Tensor:
        """Read a CT volume into `(T, C, H, W)` (windowing + resize; augment during training)."""
        augment = bool(getattr(self, 'ct_augment', False)) and self.is_training
        return load_ct_volume(
            path,
            ct_windows=self.ct_windows,
            ct_window_base=self.ct_window_base,
            volume_size=self.ct_volume_size,
            augment=augment,
            augment_prob=getattr(self, 'ct_augment_prob', 0.15))

    def num_volume_tokens(self) -> int:
        """How many LLM tokens each volume expands to (== the 3D encoder's pooled token budget)."""
        if self.vision_3d_max_tokens is None:
            raise ValueError('CT template requires `--vision_3d_max_tokens` (tokens emitted per CT volume).')
        return int(self.vision_3d_max_tokens)

    # ---- data collation --------------------------------------------------------------------------
    def _data_collator_mm_data(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        res = super()._data_collator_mm_data(batch)
        volumes = [b['pixel_values_volumes'] for b in batch if b.get('pixel_values_volumes') is not None]
        if volumes:
            # each item is (n_vol_i, T, C, H, W); concat over the volume axis -> (total_vols, T, C, H, W)
            res['pixel_values_volumes'] = torch.concat(volumes, dim=0)
        return res

    # ---- debug instrumentation (VISION_3D_DEBUG=1) ----------------------------------------------
    @property
    def _ct_debug(self) -> bool:
        return os.environ.get('VISION_3D_DEBUG', '0') == '1'

    @staticmethod
    def _shape_str(output) -> str:
        if torch.is_tensor(output):
            return f'tensor{tuple(output.shape)} dtype={output.dtype}'
        lhs = getattr(output, 'last_hidden_state', None)
        po = getattr(output, 'pooler_output', None)
        if lhs is not None or po is not None:
            return (f'{type(output).__name__}(last_hidden_state='
                    f'{tuple(lhs.shape) if lhs is not None else None}, '
                    f'pooler_output={tuple(po.shape) if po is not None else None})')
        if isinstance(output, (tuple, list)):
            return '[' + ', '.join(tuple(o.shape).__repr__() if torch.is_tensor(o) else type(o).__name__
                                   for o in output) + ']'
        return type(output).__name__

    def _register_debug_hooks(self, model) -> None:
        """Forward-hook `model.visual` (2D) and `model.visual_3d` (3D) to log their output shapes."""
        if getattr(self, '_ct_debug_hooked', False):
            return
        self._ct_debug_hooked = True

        def make_hook(tag):
            def hook(_module, _args, output):
                logger.info(f'[vision_3d_debug] {tag} encoder output: {self._shape_str(output)}')
            return hook

        visual = getattr(model, 'visual', None)
        if visual is not None:
            visual.register_forward_hook(make_hook('2D (model.visual)'))
        visual_3d = getattr(model, 'visual_3d', None)
        if visual_3d is not None:
            visual_3d.register_forward_hook(make_hook('3D (model.visual_3d)'))
        logger.info(f'[vision_3d_debug] registered hooks: 2D={visual is not None}, 3D={visual_3d is not None}')

    def _debug_tokens(self, inputs, model) -> None:
        input_ids = inputs['input_ids']
        img_id = getattr(model.config, 'image_token_id', getattr(self, 'image_token_id', None))
        vid_id = getattr(model.config, 'video_token_id', getattr(self, 'video_token_id', None))
        n_img = int((input_ids == img_id).sum()) if img_id is not None else -1
        n_vid = int((input_ids == vid_id).sum()) if vid_id is not None else -1
        pv = inputs.get('pixel_values')
        pvv = inputs.get('pixel_values_volumes')
        logger.info(f'[vision_3d_debug] batch input_ids={tuple(input_ids.shape)} | image_tokens={n_img} '
                    f'video_tokens={n_vid} | images_present={pv is not None} '
                    f'(pixel_values={tuple(pv.shape) if pv is not None else None}) | '
                    f'volumes_present={pvv is not None} '
                    f'(pixel_values_volumes={tuple(pvv.shape) if pvv is not None else None})')

    def _debug_start(self, model, inputs) -> None:
        if self._ct_debug:
            self._register_debug_hooks(model)
            self._debug_tokens(inputs, model)

    # ---- embedding-time dual routing (Qwen2.5-VL style: splice both here) ------------------------
    def _post_encode(self, model, inputs: Dict[str, Any]) -> Dict[str, Any]:
        self._debug_start(model, inputs)
        input_ids = inputs['input_ids']
        base_model = self.get_base_model(model)
        if hasattr(base_model.model, 'embed_tokens'):
            inputs_embeds = base_model.model.embed_tokens(input_ids)
        else:
            inputs_embeds = base_model.model.language_model.embed_tokens(input_ids)
        if self.is_training:
            # Training: swift owns both splices. 2D path: images -> model.visual (also emits a zeroed
            # dummy for text-only/volume-only batches so the 2D tower stays in the DDP graph).
            inputs_embeds = self._get_inputs_embeds_hf(inputs_embeds, inputs, model.visual, self.processor,
                                                       model.config)
            # 3D path: CT volumes -> model.visual_3d, spliced into <video> token positions.
            inputs_embeds = self._splice_volume_embeds(inputs_embeds, inputs, model)
            return {'inputs_embeds': inputs_embeds}
        # Inference/generation with NO CT volume: leave everything to native HF (return inputs
        # untouched) so the 2D image / text path is byte-for-byte identical to upstream ms-swift.
        if inputs.get('pixel_values_volumes') is None:
            return inputs
        # Inference/generation WITH a CT volume: splice ONLY the 3D volume here (HF has no knowledge of
        # model.visual_3d) and pass the 2D image tensors through so the HF forward still encodes images
        # with model.visual and merges them into our inputs_embeds itself -- 2D handling stays native.
        inputs_embeds = self._splice_volume_embeds(inputs_embeds, inputs, model)
        out = {'inputs_embeds': inputs_embeds}
        # Only forward vision kwargs the host model's forward actually accepts. Qwen2.5-VL takes
        # `second_per_grid_ts`, but other hosts that reuse this template (e.g. Qwen3.5) do not, and HF's
        # `generate` raises on unexpected model_kwargs. Filter by the base model's forward signature.
        import inspect
        try:
            # `model` is the object generate() is called on -- match what HF validates against.
            accepted = set(inspect.signature(model.forward).parameters)
        except (TypeError, ValueError):
            accepted = None
        for key in ('pixel_values', 'pixel_values_videos', 'image_grid_thw', 'video_grid_thw', 'second_per_grid_ts'):
            if inputs.get(key) is not None and (accepted is None or key in accepted):
                out[key] = inputs[key]
        return out

    def _dummy_volume(self) -> torch.Tensor:
        """A tiny zero volume matching the configured channel/size, to keep visual_3d in the graph."""
        from swift.utils.ct_windowing import num_ct_channels
        c = num_ct_channels(self.ct_windows, self.ct_window_base)
        d, h, w = parse_volume_size(self.ct_volume_size)
        return torch.zeros(1, d, c, h, w)  # (n_vol=1, T=D, C, H, W)

    def _splice_volume_embeds(self, inputs_embeds: torch.Tensor, inputs: Dict[str, Any], model) -> torch.Tensor:
        visual_3d = getattr(model, 'visual_3d', None)
        if visual_3d is None:
            return inputs_embeds
        input_ids = inputs['input_ids']
        video_token_id = getattr(model.config, 'video_token_id', getattr(self, 'video_token_id', None))
        pixel_values_volumes = inputs.get('pixel_values_volumes')

        if pixel_values_volumes is None:
            # no volumes in this batch: dummy forward so visual_3d params receive gradient (DDP-safe)
            if self.is_training:
                # dtype is handled inside the backbone/adapter (encoder may differ from the projector)
                volume_embeds = visual_3d(self._dummy_volume().to(inputs_embeds.device))
                if self._ct_debug:
                    logger.info(f'[vision_3d_debug] no volumes -> 3D dummy forward, embeds {tuple(volume_embeds.shape)} '
                                '(added * 0, keeps visual_3d in graph)')
                inputs_embeds = inputs_embeds + volume_embeds.mean().to(inputs_embeds.device, inputs_embeds.dtype) * 0.
            return inputs_embeds

        volume_embeds = visual_3d(pixel_values_volumes, grid=inputs.get('volume_grid_thw'))
        n_video = int((input_ids == video_token_id).sum())
        if self._ct_debug:
            logger.info(f'[vision_3d_debug] 3D real forward: pixel_values_volumes {tuple(pixel_values_volumes.shape)} '
                        f'-> projected volume_embeds {tuple(volume_embeds.shape)} scattered into {n_video} <video> tokens')
        volume_mask = (input_ids == video_token_id).unsqueeze(-1).expand_as(inputs_embeds).to(inputs_embeds.device)
        volume_embeds = volume_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
        inputs_embeds = inputs_embeds.masked_scatter(volume_mask, volume_embeds)
        return inputs_embeds
