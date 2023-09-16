from typing import Union

import albumentations as albu
from albumentations.pytorch import ToTensorV2

TRANSFORM_TYPE = Union[albu.BasicTransform, albu.BaseCompose]


def get_transforms(
    width: int,
    height: int,
    preprocessing: bool = True,
    augmentations: bool = True,
    postprocessing: bool = True,
) -> TRANSFORM_TYPE:
    transforms = []

    if preprocessing:
        transforms.append(albu.Resize(height=height, width=width))

    if augmentations:
        transforms.extend(
            [
                albu.HorizontalFlip(p=0.5),
                albu.VerticalFlip(p=0.5),
                albu.ShiftScaleRotate(shift_limit=0.0625,  scale_limit=0.1, rotate_limit=180, p=0.5),
            ],
        )

    if postprocessing:
        transforms.extend([albu.Normalize(), ToTensorV2()])

    return albu.Compose(transforms)
