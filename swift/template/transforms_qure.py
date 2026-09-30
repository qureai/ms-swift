"""Qure medical-image augmentation transforms for VLM training.

Adapted from the vision_foundation_2d augmentation pipeline. The ONLY behavioural
change vs. the original is the zoom/crop branch of ``RandAugment`` (see below).

Original zoom/crop branch:
    A.Resize(1100, 1100) -> A.RandomCrop(960, 960)      # fixed sizes

Corrected zoom/crop branch (this file):
    resize to a random SQUARE S x S, with S ~ Uniform[800, 1600],
    then RandomCrop to round(0.85 * S) x round(0.85 * S)   # 15% H/W reduction

Interpretation note: "maintaining height and width" is read here as a square
resize (H == W == S), matching the original Resize(1100, 1100) behaviour — only
the target size is now randomized. If you instead meant "preserve aspect ratio",
swap A.Resize(S, S) for A.LongestMaxSize(max_size=S) in _build_transforms().

NOTE ON OUTPUT: like the original, these callables return a torch tensor
(CHW float, values in ~[0, 1]) via new_tsfms.ToTensor. A Swift VLM processor
expects a PIL RGB image, so the integration layer (added later) must convert
the tensor back to a PIL image before handing it to the processor.
"""

from random import randint, uniform

import albumentations as A
import numpy as np
from cv2 import INTER_AREA
from numpy import int64
from torchvision.transforms import Compose

from data.utils import transform_utils as new_tsfms


def modify_mask(original_mask):
    flag = 0
    if original_mask.min() == -100:
        original_mask[original_mask == -100] = 2.0
        original_mask = original_mask.astype(dtype=np.float32)
        flag = 1
    else:
        original_mask = original_mask.astype(dtype=np.float32)
    return flag, original_mask


def mask_replay_augment(seg_classes, replay, mask_dict):
    for tag in seg_classes:
        original_mask = mask_dict["segmentation_target"][tag]

        """
        THis is done so that if we encounter invalid seg maps (-100,-100) then we
        convert them to 2 so that after the augmentation they are reverted back,
        this is important as if they are not done then the value will shift from
        -100 to something like -100.0001 and then this would cause an error or
        worse the loss would be propogated , it is important to make it
        dtype=int64 as after the augmentation 2.0 sometimes becomes 2.00005
        and it wont be converted back to orginal -100 value.
        """
        flag, original_mask = modify_mask(original_mask)

        tf_mask = A.ReplayCompose.replay(replay, image=original_mask)
        tf_mask_image = tf_mask["image"].astype(dtype=int64)

        if flag:
            tf_mask_image[tf_mask_image == 2.0] = -100
        else:
            tf_mask_image = (tf_mask_image > 0) * 1

        mask_dict["segmentation_target"][tag] = tf_mask_image.astype(dtype=int64)

    return mask_dict


class Resize_Scale:
    def __init__(self, im_size) -> None:
        self.im_size = im_size
        self.albumentation_transforms = A.ReplayCompose(
            [
                A.Resize(
                    self.im_size,
                    self.im_size,
                    interpolation=INTER_AREA,
                    always_apply=True,
                ),
            ]
        )

        self.transforms = Compose(
            [
                new_tsfms.ToTensor(),
            ]
        )

    def __call__(self, original_image):
        try:
            tf_image = self.albumentation_transforms(image=original_image)
            original_image = new_tsfms.rescale(tf_image["image"])
            original_image = new_tsfms.scale(original_image)
            original_image = new_tsfms.clip(original_image)
            augmented_image = original_image[np.newaxis, :, :]  # convert to rgb
            augmented_image = np.repeat(augmented_image, 3, axis=0)
            augmented_image = augmented_image.transpose(1, 2, 0)
            augmented_image = self.transforms(augmented_image)

        except Exception as e:
            print("\033[91m" + "applying transform error" + "\033[0m")
            print(e)
            return original_image

        return augmented_image


class RandAugment:
    def __init__(
        self,
        im_size=None,
        crop_reduction_min=0.10,
        crop_reduction_max=0.15,
    ) -> None:
        # im_size=None keeps the ORIGINAL resolution: Qwen3.5 is a native-resolution
        # vision tower, so squashing every study to a fixed square throws away the
        # dynamic-resolution behaviour the processor already handles (governed by
        # IMAGE_MAX_TOKEN_NUM). Pass an int only to reproduce the fixed-size CNN setup.
        self.im_size = im_size
        # Zoom/crop is a straight 10-15% reduction of the image's OWN height and width,
        # with NO resize step -- so the crop is always relative to the real resolution
        # instead of a fixed 1100x1100 intermediate.
        self.crop_reduction_min = crop_reduction_min
        self.crop_reduction_max = crop_reduction_max

        self.transforms = Compose(
            [
                new_tsfms.scale,
                new_tsfms.clip,
                new_tsfms.gamma,
                new_tsfms.brightness,
                new_tsfms.contrast,
                new_tsfms.clip,
                new_tsfms.ToTensor(),
                # new_tsfms.ToTensor(),
            ]
        )

    def _build_transforms(self, height, width):
        """Build the albumentation pipeline for one image.

        `height`/`width` are the ORIGINAL image dimensions: the zoom/crop branch crops to
        (1-r) of them with r ~ Uniform[crop_reduction_min, crop_reduction_max], sampled
        fresh per call. There is deliberately no Resize before the crop -- the original
        pipeline went Resize(1100, 1100) -> RandomCrop(960, 960), which both forced a square
        aspect ratio and made the crop fraction depend on the input size.
        """
        r = uniform(self.crop_reduction_min, self.crop_reduction_max)
        crop_h = max(1, int(round(height * (1.0 - r))))
        crop_w = max(1, int(round(width * (1.0 - r))))

        return A.ReplayCompose(
            [
                A.OneOf(
                    [
                        A.ElasticTransform(
                            alpha=120,
                            sigma=120 * 0.05,
                            alpha_affine=120 * 0.03,
                            p=0.5,
                        ),
                        A.GridDistortion(p=0.5),
                    ],
                    p=0.3,
                ),
                A.OneOf(
                    [
                        # 10-15% crop of the image's own H,W -- no resize.
                        A.RandomCrop(crop_h, crop_w, always_apply=True),
                        A.Rotate(limit=15, p=0.5),
                    ],
                    p=0.5,
                ),
            ] + ([
                A.Resize(
                    self.im_size,
                    self.im_size,
                    interpolation=INTER_AREA,
                    always_apply=True,
                ),
            ] if self.im_size else [])
        )

    def __call__(self, original_image):
        try:
            h, w = original_image.shape[:2]
            albumentation_transforms = self._build_transforms(h, w)
            tf_image = albumentation_transforms(image=original_image)
            original_image = new_tsfms.rescale(tf_image["image"])
            original_image = new_tsfms.scale(original_image)
            original_image = new_tsfms.clip(original_image)
            augmented_image = original_image[np.newaxis, :, :]
            augmented_image = np.repeat(augmented_image, 3, axis=0)
            augmented_image = augmented_image.transpose(1, 2, 0)
            augmented_image = self.transforms(augmented_image)

        except Exception as e:
            print("\033[91m" + "applying transform error" + "\033[0m")
            print(e)
            return original_image

        return augmented_image
