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

    @property
    def _variable_ct(self) -> bool:
        """True when `--ct_volume_size` is unset -> variable-resolution CT (per-volume token counts,
        capped at `vision_3d_max_tokens`). Only supported for the Qwen-VL 3D encoder (not Atlas), and
        incompatible with the Perceiver resampler (both asserted at encoder-attach time)."""
        return getattr(self, 'ct_volume_size', None) is None

    def _ct_geometry(self):
        """(patch_size, temporal_patch_size, merge_size) of the 3D ENCODER's patch grid.

        Prefer the geometry read from `--vision_3d_model`'s own config (threaded onto the template), so
        variable-resolution CT is correct for ANY Qwen vision tower even when it differs from the host
        (e.g. host Qwen3.5 patch=16 + 3D encoder Qwen2.5-VL patch=14). Falls back to the host processor's
        geometry only when the encoder geometry couldn't be read (then it must match the host)."""
        p = getattr(self, 'vision_3d_patch_size', None)
        tps = getattr(self, 'vision_3d_temporal_patch_size', None)
        m = getattr(self, 'vision_3d_merge_size', None)
        if p and tps and m:
            return int(p), int(tps), int(m)
        ip = self.processor.image_processor
        return (int(getattr(ip, 'patch_size', 16)), int(getattr(ip, 'temporal_patch_size', 2)),
                int(getattr(ip, 'merge_size', 2)))

    def _volume_token_len(self, vol: torch.Tensor) -> int:
        """Merged <video> token count a loaded (T,C,H,W) volume expands to == (T/tps)*(H/p)*(W/p)/merge^2.
        Slow-fast: the <video> placeholder carries only the FIXED fast_count (the encoder still receives the
        full volume for the slow branch -- the resize ceiling `num_volume_tokens()` is unchanged)."""
        if getattr(self, 'vision_3d_slowfast', False):
            return int(self.vision_3d_fast_count)
        p, tps, m = self._ct_geometry()
        t, _c, h, w = vol.shape
        return int((t // tps) * (h // p) * (w // p) // (m * m))

    def _init_placeholder_tokens(self):
        # The CT templates inherit a HARDCODED video_token_id from their 2D parent (Qwen2VLTemplate=151656),
        # which is WRONG for Qwen3.5 (its video token is 248057). With the mismatch, `_encode`'s
        # findall(input_ids, self.video_token_id) matches nothing -> the <video> placeholder is never
        # expanded to the encoder's token count -> only 1 visual token is spliced and the LLM never receives
        # the volume. Resolve video_token_id from the model config/processor so it ALWAYS matches the token
        # the chat template emits and the splice scatters into (model.config.video_token_id).
        super()._init_placeholder_tokens()
        vid = getattr(self.processor, 'video_token_id', None)
        if vid is None:
            cfg = getattr(getattr(self, 'model_info', None), 'config', None)
            vid = getattr(cfg, 'video_token_id', None)
        if vid is not None and int(vid) != int(getattr(self, 'video_token_id', -1)):
            from swift.utils import get_logger
            get_logger().info(f'vision_3d: video_token_id {getattr(self, "video_token_id", None)} -> {int(vid)} '
                              '(resolved from model config so <video> expansion matches the splice).')
            self.video_token_id = int(vid)

    @staticmethod
    def _volume_uid(path) -> str:
        """series_uid == basename of the volume path without extension (key into the aux label table)."""
        import os
        p = path if isinstance(path, str) else (path[0] if path else '')
        return os.path.basename(str(p)).replace('.safetensors', '')

    def load_volume(self, path: str) -> torch.Tensor:
        """Read a CT volume into `(T, C, H, W)` (windowing + resize; augment during training).
        Fixed mode: resize to `--ct_volume_size`. Variable mode (size unset): smart-resize to a divisible,
        `vision_3d_max_tokens`-capped target near the native shape."""
        augment = bool(getattr(self, 'ct_augment', False)) and self.is_training
        kw = {}
        if self._variable_ct:
            p, tps, m = self._ct_geometry()
            kw = dict(patch_size=p, temporal_patch_size=tps, merge_size=m, max_tokens=self.num_volume_tokens())
        out_dtype = {'float16': torch.float16, 'bfloat16': torch.bfloat16}.get(
            getattr(self, 'ct_load_dtype', 'float32'))  # None -> keep float32
        return load_ct_volume(
            path,
            ct_windows=self.ct_windows,
            ct_window_base=self.ct_window_base,
            volume_size=self.ct_volume_size,
            augment=augment,
            augment_prob=getattr(self, 'ct_augment_prob', 0.15),
            normalize=getattr(self, 'ct_normalize', 'none'),
            out_dtype=out_dtype,
            **kw)

    def num_volume_tokens(self) -> int:
        """LLM tokens per volume. Fixed mode: the exact per-volume count. Variable mode: the MAXIMUM
        allowed per volume (the smart-resize ceiling). Set by `--vision_3d_max_tokens`."""
        if self.vision_3d_max_tokens is None:
            raise ValueError('CT template requires `--vision_3d_max_tokens` (tokens emitted per CT volume).')
        return int(self.vision_3d_max_tokens)

    def _prepare_volumes(self, volumes):
        """Loaded (T,C,H,W) volumes -> (pixel_values_volumes, video_grid_thw, token_lens).
        Fixed: stack into one tensor + constant token count. Variable: keep a LIST (sizes differ, can't
        stack) + per-volume token counts. The synthetic grid [1,2,2N] makes rope assign exactly N positions."""
        if self._variable_ct:
            token_lens = [self._volume_token_len(v) for v in volumes]
            pixel_values_volumes = list(volumes)
        else:
            n = self.num_volume_tokens()
            token_lens = [n] * len(volumes)
            pixel_values_volumes = torch.stack(volumes, dim=0)
        video_grid_thw = torch.tensor([[1, 2, 2 * n] for n in token_lens], dtype=torch.long)
        return pixel_values_volumes, video_grid_thw, token_lens

    # ---- data collation --------------------------------------------------------------------------
    def _data_collator_mm_data(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        res = super()._data_collator_mm_data(batch)
        volumes = [b['pixel_values_volumes'] for b in batch if b.get('pixel_values_volumes') is not None]
        if volumes:
            if all(torch.is_tensor(v) for v in volumes):
                # fixed mode: each item is (n_vol_i, T, C, H, W); concat -> (total_vols, T, C, H, W)
                res['pixel_values_volumes'] = torch.concat(volumes, dim=0)
            else:
                # variable mode: each item is a LIST of (T,C,H,W) (sizes differ); flatten to one list,
                # preserving sample order so it aligns with the <video> token order for masked_scatter.
                flat = []
                for v in volumes:
                    flat.extend(list(v) if isinstance(v, (list, tuple)) else [v])
                res['pixel_values_volumes'] = flat
        # Flatten per-volume series_uids in the SAME order (for the aux ML-Decoder label lookup).
        uids = [u for b in batch if b.get('volume_uids') is not None for u in b['volume_uids']]
        if uids:
            res['volume_uids'] = uids
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

    def _visual_2d_trainable(self, model) -> bool:
        """True if the 2D vision tower (`model.visual`) has any trainable params. Cached (requires_grad
        doesn't change mid-training). When False, the no-image dummy 2D forward is skipped entirely."""
        cached = getattr(self, '_visual2d_trainable_cache', None)
        if cached is None:
            visual = getattr(model, 'visual', None)
            cached = bool(visual is not None and any(p.requires_grad for p in visual.parameters()))
            self._visual2d_trainable_cache = cached
        return cached

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
            # Training: swift owns both splices. 2D path: images -> model.visual. `_get_inputs_embeds_hf`
            # also emits a zeroed (`* 0`) dummy image for batches with NO images, purely to keep the 2D
            # tower's params in the autograd/DDP graph (avoids "param received no grad" under DDP). That
            # dummy is only needed when the 2D tower is TRAINABLE: a frozen tower has no grads to sync, so
            # for a no-image batch we skip the 2D path entirely -- otherwise, under ZeRO-3, the dummy also
            # force-gathers the whole frozen 2D tower every step for nothing (measured ~166 ms/step waste).
            has_2d_media = inputs.get('pixel_values') is not None or inputs.get('pixel_values_videos') is not None
            if has_2d_media or self._visual_2d_trainable(model):
                inputs_embeds = self._get_inputs_embeds_hf(inputs_embeds, inputs, model.visual, self.processor,
                                                           model.config)
            # 3D path: CT volumes -> model.visual_3d, spliced into <video> token positions.
            inputs_embeds = self._splice_volume_embeds(inputs_embeds, inputs, model)
            self._stash_slow_tokens(model, input_ids)          # slow-fast: push slow K/V onto cross-attn layers
            # Aux ML-Decoder grounding loss on the pre-projector tokens (stashed by the splice's encoder
            # forward); added to the LM loss by the trainer. None when the head is off / no labeled volumes.
            self._aux_mldecoder_loss = self._compute_aux_mldecoder_loss(model, inputs)
            return {'inputs_embeds': inputs_embeds}
        # Inference/generation with NO CT volume: leave everything to native HF (return inputs
        # untouched) so the 2D image / text path is byte-for-byte identical to upstream ms-swift.
        if inputs.get('pixel_values_volumes') is None:
            return inputs
        # Inference/generation WITH a CT volume: splice ONLY the 3D volume here (HF has no knowledge of
        # model.visual_3d) and pass the 2D image tensors through so the HF forward still encodes images
        # with model.visual and merges them into our inputs_embeds itself -- 2D handling stays native.
        inputs_embeds = self._splice_volume_embeds(inputs_embeds, inputs, model)
        self._stash_slow_tokens(model, input_ids)              # slow-fast: push slow K/V onto cross-attn layers
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

    def _compute_aux_mldecoder_loss(self, model, inputs: Dict[str, Any]):
        """lambda * ASL over the pre-projector encoder tokens, for volumes whose series_uid is in the aux
        label table (others masked out). Returns a scalar tensor, or None when the head is off / no
        labeled volumes this batch. The encoder tokens were stashed on the backbone by the splice's forward."""
        bb = getattr(model, 'visual_3d', None)
        head = getattr(bb, 'aux_head', None) if bb is not None else None
        if head is None or not getattr(bb, 'aux_lambda', 0):
            return None
        feats, uids, table = bb._aux_feats, inputs.get('volume_uids'), (bb.aux_table or {})
        if not feats or not uids or len(feats) != len(uids):
            return None
        # Run the head on ALL volumes so it stays in the autograd graph on EVERY rank every step (identical
        # DDP/NCCL collective sequence -- otherwise ranks with vs without labeled volumes desync -> NCCL
        # ALLREDUCE timeout/SIGABRT). ASL is applied only to labeled volumes; if none are labeled on this
        # rank, a 0-weighted term keeps the head params in the graph with a zero gradient.
        logits = head(feats)  # (n_vol, num_classes)
        keep = [i for i, u in enumerate(uids) if table.get(u) is not None]
        if keep:
            targets = torch.stack([table[uids[i]] for i in keep]).to(logits.device)
            loss = head.loss_fn(logits[keep], targets)
        else:
            loss = logits.sum() * 0.0
        return bb.aux_lambda * loss

    def _stash_slow_tokens(self, model, input_ids):
        """Slow-fast: pad the per-volume SLOW tokens (stashed on the backbone by its forward) to
        (n_vol, T_max, enc_dim) + a validity mask, and set them on the backbone (`_slow_kv_padded` /
        `_slow_mask_padded`) where the decoder-layer cross-attn hooks read them in the upcoming LLM forward.
        Cleared (identity) when off, no volumes, or the volume<->sample mapping isn't a safe 1:1."""
        bb = getattr(model, 'visual_3d', None)
        if bb is None or not getattr(bb, 'slowfast', False):
            return
        slow = getattr(bb, '_slow_tokens', None)
        B = int(input_ids.shape[0])
        if not slow or len(slow) != B:                        # no volumes / not a 1:1 mapping (e.g. multi-volume)
            if slow and len(slow) != B and getattr(self, '_ct_debug', False):
                logger.info(f'[vision_3d_debug] slow-fast: {len(slow)} volumes != batch {B}; cross-attn skipped')
            # DISTRIBUTED-SAFE: during training, still hand the cross-attn an ALL-MASKED (B,1) dummy so the
            # cross_blocks stay in the autograd graph on EVERY rank (uniform param usage -> no DeepSpeed/DDP
            # grad-sync hang when some ranks have volumes and others don't / have multi-volume samples). The
            # fully-masked attention contributes exactly 0 (nan_to_num), so the LM output is unchanged. At
            # inference we can safely skip (no backward).
            if self.is_training:
                dev = input_ids.device
                bb._slow_kv_padded = torch.zeros(B, 1, bb.encoder_dim, device=dev, dtype=bb.dtype)
                bb._slow_mask_padded = torch.zeros(B, 1, dtype=torch.bool, device=dev)   # all False = all masked
            else:
                bb._slow_kv_padded = None
                bb._slow_mask_padded = None
            return
        cap = getattr(bb, 'slow_max_tokens', 0) or 0
        slow = [t[:cap] if (cap and t.shape[0] > cap) else t for t in slow]   # cap K/V memory (truncate)
        T_max = max(t.shape[0] for t in slow)
        D, dev, dt = slow[0].shape[1], slow[0].device, slow[0].dtype
        kv = torch.zeros(B, T_max, D, device=dev, dtype=dt)
        mask = torch.zeros(B, T_max, dtype=torch.bool, device=dev)
        for i, t in enumerate(slow):
            kv[i, :t.shape[0]] = t
            mask[i, :t.shape[0]] = True
        bb._slow_kv_padded = kv
        bb._slow_mask_padded = mask

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
