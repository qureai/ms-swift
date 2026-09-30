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

   F1 BUG FIX. This guard used to `import swift_plugins.qure_augment_plugin`, which -- because
   --external_plugins had already imported the file as top-level `qure_augment_plugin` --
   EXECUTED IT A SECOND TIME: a second RandAugment wrapper (non-bbox images augmented twice) and
   AUG_PROB zeroed on the copy that was NOT doing the geometric augmentation (bbox rows still got
   ElasticTransform/crop/rotate). Now every plugin registers itself under both names
   (qure_plugin_mode.dedup_module), the augment module is resolved from sys.modules (the copy
   whose wrapper is installed), and AUG_PROB is zeroed on every loaded copy.
   QURE_PLUGIN_DEDUP=0 restores the legacy resolution and double import exactly.
"""
# F1 -- must stay the first statements of the module (qure_plugin_mode.dedup_module)
try:
    from swift_plugins.qure_plugin_mode import (dedup_module as _dedup_module, once as _once,
                                                drop_failed_import as _drop_failed_import)
except ImportError:  # repo root not on sys.path; swift put this plugin's own dir there
    from qure_plugin_mode import (dedup_module as _dedup_module, once as _once,
                                  drop_failed_import as _drop_failed_import)
_dedup_module(__name__, 'qure_stage3_plugin')

import glob  # noqa: E402
import os  # noqa: E402
import sys  # noqa: E402

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
    if not _once('qure_stage3_plugin.patch_prepare_model'):
        return  # F1: already wrapped in this process
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


_AUG_NAMES = ('qure_augment_plugin', 'swift_plugins.qure_augment_plugin')


def _legacy_import_augment():
    try:
        import swift_plugins.qure_augment_plugin as qa
    except Exception:
        # a body that RAISED (e.g. the load-time albumentations check) leaves its F1 alias behind:
        # drop it so this retry re-executes the file and raises the same error (pre-F1 behaviour)
        # instead of silently returning the half-initialised module
        _drop_failed_import('qure_augment_plugin')
        try:
            import qure_augment_plugin as qa  # external_plugins may load it top-level
        except Exception:
            _drop_failed_import('qure_augment_plugin')  # this attempt's alias too: nothing half-built stays
            raise
    return qa


def _resolve_augment_modules(current_fn):
    """(primary, all_copies) of qure_augment_plugin, as actually installed.

    F1: resolve from sys.modules FIRST -- --external_plugins imported the file top-level before
    this plugin, and importing it under the other name is what used to execute it a second time.
    With dedup on both names are one object, so `all_copies` has one entry; every distinct
    loaded copy is returned anyway so the guard zeroes AUG_PROB on all of them. `primary` is the
    copy whose wrapper is the current `_preprocess_inputs` (the one that runs the geometry).
    Only if no copy is loaded is it imported (legacy behaviour: stage3 then pulls it in).
    QURE_PLUGIN_DEDUP=0 -> the legacy resolution (`import swift_plugins.qure_augment_plugin`).
    """
    from swift_plugins.qure_plugin_mode import _dedup_enabled
    if not _dedup_enabled():
        qa = _legacy_import_augment()
        return qa, [qa]
    mods = []
    for n in _AUG_NAMES:
        m = sys.modules.get(n)
        if m is not None and all(m is not x for x in mods):
            mods.append(m)
    if not mods:
        mods = [_legacy_import_augment()]
    installed = sys.modules.get(getattr(current_fn, '__module__', None) or '')
    primary = installed if any(installed is m for m in mods) else mods[0]
    if len(mods) > 1:
        logger.warning(f'[stage3] {len(mods)} distinct qure_augment_plugin copies are loaded '
                       '(QURE_PLUGIN_DEDUP off, or a copy from another file); AUG_PROB is zeroed '
                       'on all of them for bbox rows')
    return primary, mods


def patch_bbox_safe_augment():
    """Wrap (the already-patched) Qwen3_5Template._preprocess_inputs with a bbox guard."""
    if not _once('qure_stage3_plugin.patch_bbox_safe_augment'):
        return  # F1: already wrapped in this process
    from swift.template.templates.qwen import Qwen3_5Template

    from PIL import Image
    # bbox rows: 'photometric' = apply intensity-only augs (bbox-safe), 'skip' = no aug at all.
    BBOX_AUG_MODE = os.environ.get('QURE_AUG_BBOX_MODE', 'photometric').lower()
    # resolve BEFORE capturing orig (the legacy order): if the resolution has to import the
    # augment plugin, that import installs its wrapper, which orig must then include
    qa, qa_all = _resolve_augment_modules(Qwen3_5Template._preprocess_inputs)
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
        old = [m.AUG_PROB for m in qa_all]
        for m in qa_all:
            m.AUG_PROB = 0.0
        try:
            orig(self, inputs)
        finally:
            for m, v in zip(qa_all, old):
                m.AUG_PROB = v
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
