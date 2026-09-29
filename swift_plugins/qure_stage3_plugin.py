#!/usr/bin/env python
"""Stage-3 add-on plugin — OPT-IN ONLY. Loaded exclusively by
examples/train/liger/13_aug_qure_2b_stage3_diverse_frozen_vit_head.sh; no other run is
affected, and the shared plugins (mldecoder_joint / qure_augment / qure_ce_weight) are
untouched. Everything here layers on top of them via wrappers.

MUST be listed LAST in --external_plugins (after mldecoder_joint_plugin and
qure_augment_plugin), because it wraps the patches those plugins install.

What it adds (each independently env-gated):

1. HEAD RESTORE (MLD_RESTORE_HEAD, default 1)
   mldecoder_joint_plugin.attach_head runs in prepare_model — AFTER from_pretrained — so when
   --model points at a JOINT checkpoint, its ml_decoder_head.* tensors were already dropped as
   "unexpected" and the freshly attached head is RANDOM. This wrapper re-loads them from the
   checkpoint's safetensors right after attach (same recovery mldecoder_infer_plugin does for
   inference). Without this, stage 3 would silently train/carry a random head.

2. HEAD FREEZE (MLD_FREEZE_HEAD, default 0 — the stage-3 script sets 1)
   Sets requires_grad=False on the head after restore. mldecoder_joint_plugin's
   patch_head_param_group filters on requires_grad, so no optimizer group is created; the head
   is still part of the module tree and is saved into every checkpoint unchanged.
   Combine with ASL_WEIGHT=0 to skip the classification loss (frozen head + frozen ViT means
   ASL has no trainable path anyway).

3. BBOX-SAFE AUGMENTATION (QURE_AUG_SKIP_BBOX, default 1)
   qure_augment_plugin's pipeline includes geometric transforms (ElasticTransform /
   GridDistortion / 10-15% crop-zoom) that move pixels while a conversation's bbox coordinates
   stay fixed — training grounding turns on augmented images teaches WRONG boxes. This wrapper
   detects conversations mentioning `bbox_2d` (bbox generation answers, region-identify
   prompts, comparison rows) and suppresses augmentation for exactly those rows by zeroing
   qure_augment_plugin.AUG_PROB around the wrapped call. Non-bbox rows keep the full pipeline.
   (Dataloader workers are separate processes and calls within a worker are sequential, so the
   temporary module-global flip is safe.)
"""
import glob
import os

from swift.utils import get_logger

logger = get_logger()

_flag = lambda name, default: os.environ.get(name, default).lower() in ('1', 'true', 'yes')
RESTORE_HEAD = _flag('MLD_RESTORE_HEAD', '1')
FREEZE_HEAD = _flag('MLD_FREEZE_HEAD', '0')
SKIP_BBOX_AUG = _flag('QURE_AUG_SKIP_BBOX', '1')


def _restore_head_weights(model):
    """Load ml_decoder_head.* from the model's source checkpoint into the attached head."""
    head = getattr(model, 'ml_decoder_head', None)
    if head is None:
        logger.warning('[stage3] no ml_decoder_head on model — is mldecoder_joint_plugin '
                       'listed before qure_stage3_plugin?')
        return
    src = getattr(model, 'model_dir', None) or getattr(model, 'name_or_path', None) \
        or getattr(getattr(model, 'config', None), '_name_or_path', None)
    if not (src and os.path.isdir(src)):
        logger.warning(f'[stage3] cannot locate model dir ({src!r}) — head NOT restored')
        return
    from safetensors.torch import load_file
    sd = {}
    for f in sorted(glob.glob(os.path.join(src, '*.safetensors'))):
        for k, v in load_file(f).items():
            if k.startswith('ml_decoder_head.'):
                sd[k[len('ml_decoder_head.'):]] = v
    if not sd:
        logger.error(f'[stage3] NO ml_decoder_head.* tensors in {src} — the attached head is '
                     'RANDOM. For stage 3 the base model must be a joint checkpoint.')
        return
    missing, unexpected = head.load_state_dict(sd, strict=False)
    logger.info(f'[stage3] restored {len(sd)} head tensors from {src} '
                f'(missing={len(missing)}, unexpected={len(unexpected)})')


def patch_prepare_model():
    """Wrap (the already-patched) TunerMixin.prepare_model: restore + optionally freeze."""
    from swift.pipelines.train.tuner import TunerMixin

    # After mldecoder_joint_plugin, prepare_model is a plain function; fresh it's a classmethod.
    orig = getattr(TunerMixin.prepare_model, '__func__', TunerMixin.prepare_model)

    def prepare_model(cls, args, model, **kwargs):
        model = orig(cls, args, model, **kwargs)
        base = model.module if hasattr(model, 'module') else model
        if RESTORE_HEAD:
            _restore_head_weights(base)
        if FREEZE_HEAD:
            head = getattr(base, 'ml_decoder_head', None)
            if head is not None:
                head.requires_grad_(False)
                head.eval()
                logger.info('[stage3] head FROZEN: weights fixed, no optimizer group '
                            '(param-group patch filters on requires_grad), still saved in ckpts')
        return model

    TunerMixin.prepare_model = prepare_model
    logger.info(f'[stage3] prepare_model wrapped (restore_head={RESTORE_HEAD}, '
                f'freeze_head={FREEZE_HEAD})')


def patch_bbox_safe_augment():
    """Wrap (the already-patched) Qwen3_5Template._preprocess_inputs with a bbox guard."""
    try:
        import swift_plugins.qure_augment_plugin as qa
    except Exception:
        import qure_augment_plugin as qa  # external_plugins may load it top-level
    from swift.template.templates.qwen import Qwen3_5Template

    from PIL import Image
    # bbox rows: 'photometric' = apply intensity-only augs (bbox-safe), 'skip' = no aug at all.
    BBOX_AUG_MODE = os.environ.get('QURE_AUG_BBOX_MODE', 'photometric').lower()
    orig = Qwen3_5Template._preprocess_inputs
    stats = {'photo': 0, 'skipped': 0}

    def _preprocess_inputs(self, inputs):
        is_bbox = False
        if getattr(self, 'is_training', False):
            for m in (getattr(inputs, 'messages', None) or []):
                c = m.get('content') if isinstance(m, dict) else None
                if isinstance(c, str) and 'bbox_2d' in c:
                    is_bbox = True
                    break
        if not is_bbox:
            return orig(self, inputs)               # non-bbox: full geometric+photometric aug

        # bbox conversation: never apply the GEOMETRIC pipeline (it would move pixels while the
        # box coordinates stay fixed). Disable the full augment inside orig, then optionally
        # apply the photometric-only pipeline (which leaves coordinates valid).
        old = qa.AUG_PROB
        qa.AUG_PROB = 0.0
        try:
            orig(self, inputs)
        finally:
            qa.AUG_PROB = old
        if BBOX_AUG_MODE == 'photometric' and getattr(self, 'is_training', False):
            imgs = getattr(inputs, 'images', None) or []
            for i, im in enumerate(imgs):
                if isinstance(im, Image.Image):
                    imgs[i] = qa.augment_pil_photometric(im)
            inputs.images = imgs
            stats['photo'] += 1
            if stats['photo'] in (1, 1000, 10000):
                logger.info(f"[stage3] {stats['photo']} bbox conversations photometric-augmented "
                            '(geometry suppressed, box coords preserved)')
        else:
            stats['skipped'] += 1
        return

    Qwen3_5Template._preprocess_inputs = _preprocess_inputs
    logger.info(f'[stage3] bbox-safe augmentation guard installed (mode={BBOX_AUG_MODE})')


from swift_plugins.qure_plugin_mode import is_inference  # noqa: E402

if is_inference():
    logger.info('[stage3] inference mode — no patches installed')
else:
    patch_prepare_model()
    if SKIP_BBOX_AUG:
        patch_bbox_safe_augment()
    logger.info('[stage3] plugin loaded (opt-in; shared plugins untouched)')
