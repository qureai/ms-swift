#!/usr/bin/env python
"""Wire the foundation_model_accelerate augmentation pipeline into swift VLM training.

    --external_plugins swift_plugins/qure_augment_plugin.py

`swift/template/transforms_qure.py` already ports the pipeline but NOTHING imported it, so no
swift training run has ever applied these augmentations. This plugin is the missing link.

Pipeline, matching foundation_model_accelerate/data/utils/transform_controller.py RandAugment
(the pipeline used by the mldecoder / new-tag-extraction configs):

  geometric (albumentations ReplayCompose)
    OneOf(p=0.3): ElasticTransform(alpha=120, sigma=6, alpha_affine=3.6) | GridDistortion
    OneOf(p=0.5): RandomCrop(10-15% reduction of the image's OWN H,W) | Rotate(limit=15)
    [optional] Resize(im_size, im_size)          -- OFF by default, see IM_SIZE below
  intensity (torchvision Compose over the numpy image)
    scale -> clip -> gamma -> brightness -> contrast -> clip -> ToTensor

The zoom/crop branch deliberately differs from the original `Resize(1100,1100) ->
RandomCrop(960,960)`: that forced a square aspect ratio and made the crop fraction depend on
the input resolution. Here it is a straight 10-15% reduction of the real H,W with no resize.

IM_SIZE is None by default. Qwen3.5's vision tower is native-resolution and the processor
already handles sizing via IMAGE_MAX_TOKEN_NUM, so forcing a fixed square would discard that.
Set QURE_AUG_IM_SIZE to an int only to reproduce the fixed-size CNN setup.

CACHING CAVEAT: with `--cached_dataset` the augmentation is applied ONCE at encode time and
frozen into the cache, so every epoch sees identical pixels. That is harmless at
num_train_epochs=1 (each sample is seen once anyway) but wrong for multi-epoch training --
use `--lazy_tokenize true` there so encode, and therefore augmentation, reruns each epoch.

ONE COPY PER PROCESS (F1). The first statements below register this module under both
`qure_augment_plugin` (how --external_plugins imports it) and `swift_plugins.qure_augment_plugin`
(how qure_stage3_plugin used to import it), and patch_template() is once()-guarded. Before this,
the second name re-executed the file and stacked a SECOND RandAugment wrapper: non-bbox images
were augmented twice and bbox rows still got geometric augmentation. QURE_PLUGIN_DEDUP=0 restores
the legacy double import (see qure_plugin_mode.py).
"""
# F1 -- must stay the first statements of the module (qure_plugin_mode.dedup_module)
try:
    from swift_plugins.qure_plugin_mode import dedup_module as _dedup_module, once as _once
except ImportError:  # repo root not on sys.path; swift put this plugin's own dir there
    from qure_plugin_mode import dedup_module as _dedup_module, once as _once
_dedup_module(__name__, 'qure_augment_plugin')

import os  # noqa: E402
import sys  # noqa: E402

import numpy as np
from PIL import Image

from swift.utils import get_logger

logger = get_logger()

FM_REPO = os.environ.get(
    'FM_ACCELERATE_REPO',
    '/fastdata4/sahil_work/VLM_qure/vision_foundation_2d/foundation_model_accelerate')
# probability of augmenting a given training image
AUG_PROB = float(os.environ.get('QURE_AUG_PROB', '1.0'))
_IM = os.environ.get('QURE_AUG_IM_SIZE')
IM_SIZE = int(_IM) if _IM else None
CROP_MIN = float(os.environ.get('QURE_AUG_CROP_MIN', '0.10'))
CROP_MAX = float(os.environ.get('QURE_AUG_CROP_MAX', '0.15'))
# Photometric-only pipeline (bbox-SAFE: no pixel is moved, so box coordinates stay valid).
# Applied to bbox conversations by qure_stage3_plugin when QURE_AUG_BBOX_MODE=photometric.
PHOTO_PROB = float(os.environ.get('QURE_AUG_PHOTO_PROB', '1.0'))

# transforms_qure imports `data.utils.transform_utils` from the foundation repo
if FM_REPO not in sys.path:
    sys.path.insert(0, FM_REPO)

# Fail at LOAD time if the dependency is missing. Without this the per-image try/except
# below quietly returns the original image, so a run would train with NO augmentation while
# only emitting a warning at the 1st/10th/100th failure -- easy to miss in a 9-hour log.
# albumentations must be 1.x: this pipeline uses ElasticTransform(alpha_affine=...) and
# always_apply=, both removed in 2.x. The foundation repo pins 1.3.1.
#
# The check is TRAIN-ONLY. swift reloads external_plugins from the checkpoint's args.json, so
# this plugin is imported by `swift infer` too -- and there, raising on a missing/incompatible
# albumentations would kill an inference run over a dependency that augmentation alone needs
# and that inference never touches.
from swift_plugins.qure_plugin_mode import is_inference  # noqa: E402

if not is_inference():
    try:
        import albumentations as _A
    except ImportError as _e:
        raise ImportError('qure_augment_plugin needs albumentations (pin: 1.3.1). '
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
        _AUG = RandAugment(im_size=IM_SIZE, crop_reduction_min=CROP_MIN,
                           crop_reduction_max=CROP_MAX)
        logger.info(f'[qure-aug] RandAugment ready · im_size={IM_SIZE} '
                    f'crop_reduction={CROP_MIN}-{CROP_MAX} p={AUG_PROB}')
    return _AUG


def augment_pil(img: Image.Image) -> Image.Image:
    """PIL -> augmented PIL. RandAugment works on numpy and returns a CHW float tensor."""
    import random
    import torch
    _STATS['seen'] += 1
    if AUG_PROB < 1.0 and random.random() >= AUG_PROB:
        return img
    try:
        arr = np.asarray(img.convert('L') if img.mode != 'L' else img)
        out = get_augmenter()(arr)
        if isinstance(out, torch.Tensor):
            # CHW float in ~[0,1] -> HWC uint8 RGB, which is what the processor wants
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
        if _STATS['augmented'] in (1, 100, 1000):
            logger.info(f'[qure-aug] {_STATS["augmented"]} images augmented '
                        f'(e.g. {img.size} -> {res.size}) · failures={_STATS["failed"]}')
        return res
    except Exception as e:
        _STATS['failed'] += 1
        if _STATS['failed'] in (1, 10, 100):
            logger.warning(f'[qure-aug] augmentation failed ({_STATS["failed"]}x), using the '
                           f'original image: {type(e).__name__}: {e}')
        return img


_PHOTO = None

def get_photometric():
    """Albumentations photometric-only Compose. Every op alters INTENSITY, never geometry, so a
    box's [x1,y1,x2,y2] stays correct — safe for bbox / region-identify / comparison rows."""
    global _PHOTO
    if _PHOTO is None:
        import albumentations as A
        _PHOTO = A.Compose([
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
            A.RandomGamma(gamma_limit=(80, 120), p=0.4),
            A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=0.3),          # CXR contrast
            A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
            A.OneOf([
                A.GaussianBlur(blur_limit=(3, 5)),
                A.MedianBlur(blur_limit=3),
                A.Sharpen(alpha=(0.1, 0.3), lightness=(0.9, 1.1)),
            ], p=0.3),
            A.ImageCompression(quality_lower=60, quality_upper=100, p=0.2),  # jpeg-artifact robustness
            A.Downscale(scale_min=0.7, scale_max=0.95, p=0.15),             # resolution robustness
        ])
        logger.info('[qure-aug] photometric-only pipeline ready (brightness/contrast/gamma/CLAHE/'
                    'noise/blur/sharpen/jpeg/downscale) — bbox-safe')
    return _PHOTO


def augment_pil_photometric(img: Image.Image) -> Image.Image:
    """PIL -> photometric-augmented PIL (no geometry). Used for bbox conversations."""
    import random
    if PHOTO_PROB < 1.0 and random.random() >= PHOTO_PROB:
        return img
    try:
        a = np.asarray(img.convert('RGB'))                 # HWC uint8
        out = get_photometric()(image=a)['image']
        res = Image.fromarray(np.asarray(out)[:, :, :3].astype(np.uint8), mode='RGB')
        _STATS['augmented'] += 1
        return res
    except Exception as e:
        _STATS['failed'] += 1
        if _STATS['failed'] in (1, 10, 100):
            logger.warning(f'[qure-aug] photometric augmentation failed ({_STATS["failed"]}x): '
                           f'{type(e).__name__}: {e}')
        return img


def patch_template():
    """Augment inside _preprocess_inputs, after images are loaded to PIL, training only."""
    if not _once('qure_augment_plugin.patch_template'):
        return  # F1: already wrapped in this process -- never stack a second augmentation
    from swift.template.templates.qwen import Qwen3_5Template

    orig = Qwen3_5Template._preprocess_inputs

    def _preprocess_inputs(self, inputs):
        orig(self, inputs)
        # Second, independent guard. `is_training` is
        #   self.mode not in {'transformers', 'vllm', 'lmdeploy', 'sglang'}
        # (swift/template/base.py:1590), so it is False for EVERY inference backend and for the
        # eval pass inside training. Augmentation therefore cannot leak into generation or into
        # a val metric even if this plugin were installed by mistake.
        if not self.is_training:
            return
        imgs = getattr(inputs, 'images', None) or []
        for i, im in enumerate(imgs):
            if isinstance(im, Image.Image):
                imgs[i] = augment_pil(im)
        inputs.images = imgs

    Qwen3_5Template._preprocess_inputs = _preprocess_inputs
    logger.info('[qure-aug] Qwen3_5Template._preprocess_inputs patched (train-time only)')


if is_inference():
    logger.info('[qure-aug] inference mode — augmentation NOT patched. Generation and eval '
                'always see unaugmented images.')
else:
    patch_template()
    logger.info('[qure-aug] plugin loaded')
