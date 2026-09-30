#!/usr/bin/env python
"""InternVL port of qure_augment_plugin.py — wire the foundation_model_accelerate RandAugment
into InternVL3.5 training.

    --external_plugins swift_plugins/qure_augment_internvl_plugin.py

Same pipeline as the Qwen plugin (transforms_qure.RandAugment):
  geometric (albumentations ReplayCompose)
    OneOf(p=0.3): ElasticTransform(alpha=120, sigma=6, alpha_affine=3.6) | GridDistortion
    OneOf(p=0.5): RandomCrop(10-15% reduction of the image's OWN H,W) | Rotate(limit=15)
  intensity (torchvision): scale -> clip -> gamma -> brightness -> contrast -> clip -> ToTensor

The ONLY difference from qure_augment_plugin.py is the patch target: InternVL loads images to PIL
and dynamically tiles them in Internvl2Template, so we augment each PIL image in
`Internvl2Template._preprocess_inputs` (after load, BEFORE transform_image tiles it). Augmenting
the source image means every dynamic tile of it is derived from the augmented pixels.

CACHING/EPOCHS: with --lazy_tokenize true (as in the joint run) encode reruns each epoch, so a
fresh augmentation is drawn per sample per epoch. Env knobs: QURE_AUG_PROB (default 1.0),
QURE_AUG_CROP_MIN/MAX (0.10/0.15), QURE_AUG_IM_SIZE (None = keep native/dynamic resolution).

ONE COPY PER PROCESS (F1): the first statements register this module under both
`qure_augment_internvl_plugin` and `swift_plugins.qure_augment_internvl_plugin`, and
patch_template() is once()-guarded (a second copy would augment every image twice).
QURE_PLUGIN_DEDUP=0 restores the legacy behaviour (see qure_plugin_mode.py).
"""
# F1 -- must stay the first statements of the module (qure_plugin_mode.dedup_module)
try:
    from swift_plugins.qure_plugin_mode import dedup_module as _dedup_module, once as _once
except ImportError:  # repo root not on sys.path; swift put this plugin's own dir there
    from qure_plugin_mode import dedup_module as _dedup_module, once as _once
_dedup_module(__name__, 'qure_augment_internvl_plugin')

import os  # noqa: E402
import sys  # noqa: E402

import numpy as np
from PIL import Image

from swift.utils import get_logger

logger = get_logger()

FM_REPO = os.environ.get(
    'FM_ACCELERATE_REPO',
    '/localstorage/sahil_work/VLM_qure/vision_foundation_2d/foundation_model_accelerate')
AUG_PROB = float(os.environ.get('QURE_AUG_PROB', '1.0'))
_IM = os.environ.get('QURE_AUG_IM_SIZE')
IM_SIZE = int(_IM) if _IM else None
CROP_MIN = float(os.environ.get('QURE_AUG_CROP_MIN', '0.10'))
CROP_MAX = float(os.environ.get('QURE_AUG_CROP_MAX', '0.15'))

if FM_REPO not in sys.path:
    sys.path.insert(0, FM_REPO)

from swift_plugins.qure_plugin_mode import is_inference  # noqa: E402

if not is_inference():
    try:
        import albumentations as _A
    except ImportError as _e:
        raise ImportError('qure_augment_internvl_plugin needs albumentations (pin: 1.3.1). '
                          f'pip install albumentations==1.3.1 · original: {_e}') from _e
    if int(_A.__version__.split('.')[0]) >= 2:
        raise ImportError(f'albumentations {_A.__version__} is incompatible: this pipeline uses '
                          'alpha_affine/always_apply, removed in 2.x. Pin albumentations==1.3.1.')

_AUG = None
_STATS = {'seen': 0, 'augmented': 0, 'failed': 0}


def get_augmenter():
    global _AUG
    if _AUG is None:
        from swift.template.transforms_qure import RandAugment
        _AUG = RandAugment(im_size=IM_SIZE, crop_reduction_min=CROP_MIN, crop_reduction_max=CROP_MAX)
        logger.info(f'[qure-aug-internvl] RandAugment ready · im_size={IM_SIZE} '
                    f'crop_reduction={CROP_MIN}-{CROP_MAX} p={AUG_PROB}')
    return _AUG


def augment_pil(img: Image.Image) -> Image.Image:
    import random
    import torch
    _STATS['seen'] += 1
    if AUG_PROB < 1.0 and random.random() >= AUG_PROB:
        return img
    try:
        arr = np.asarray(img.convert('L') if img.mode != 'L' else img)
        out = get_augmenter()(arr)
        if isinstance(out, torch.Tensor):
            a = out.detach().cpu().float().numpy()
            if a.ndim == 3:
                a = a.transpose(1, 2, 0)
            a = np.clip(a, 0.0, 1.0) * 255.0
            a = a.astype(np.uint8)
            if a.ndim == 2:
                a = np.repeat(a[:, :, None], 3, axis=2)
            elif a.shape[2] == 1:
                a = np.repeat(a, 3, axis=2)
            res = Image.fromarray(a[:, :, :3], mode='RGB')
        else:
            a = np.asarray(out)
            if a.ndim == 2:
                a = np.repeat(a[:, :, None], 3, axis=2)
            res = Image.fromarray(a.astype(np.uint8)[:, :, :3], mode='RGB')
        _STATS['augmented'] += 1
        if _STATS['augmented'] in (1, 100, 1000, 10000):
            logger.info(f'[qure-aug-internvl] {_STATS["augmented"]} images augmented '
                        f'(e.g. {img.size} -> {res.size}) · failures={_STATS["failed"]}')
        return res
    except Exception as e:
        _STATS['failed'] += 1
        if _STATS['failed'] in (1, 10, 100):
            logger.warning(f'[qure-aug-internvl] augmentation failed ({_STATS["failed"]}x), using '
                           f'the original image: {type(e).__name__}: {e}')
        return img


def patch_template():
    """Augment inside Internvl2Template._preprocess_inputs, after PIL load, training only."""
    if not _once('qure_augment_internvl_plugin.patch_template'):
        return  # F1: already wrapped in this process
    from swift.template.templates.internvl import Internvl2Template

    orig = Internvl2Template._preprocess_inputs

    def _preprocess_inputs(self, inputs):
        orig(self, inputs)
        # self.is_training is False for every inference backend and for the eval pass inside
        # training (swift/template/base.py), so augmentation never leaks into generation/eval.
        if not self.is_training:
            return
        imgs = getattr(inputs, 'images', None) or []
        for i, im in enumerate(imgs):
            if isinstance(im, Image.Image):
                imgs[i] = augment_pil(im)
        inputs.images = imgs

    Internvl2Template._preprocess_inputs = _preprocess_inputs
    logger.info('[qure-aug-internvl] Internvl2Template._preprocess_inputs patched (train-time only)')


if is_inference():
    logger.info('[qure-aug-internvl] inference mode — augmentation NOT patched.')
else:
    patch_template()
    logger.info('[qure-aug-internvl] plugin loaded')
