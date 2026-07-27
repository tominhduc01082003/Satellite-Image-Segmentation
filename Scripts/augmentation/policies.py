"""
policies.py
===========

Albumentations policies for online data augmentation.
"""

from __future__ import annotations

import albumentations as A
import cv2
import sys
from pathlib import Path
from albumentations.pytorch import ToTensorV2

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
from Scripts.config.config import (
    BACKGROUND_AUG,
    VERY_SPARSE_AUG,
    SPARSE_AUG,
    NORMAL_AUG,
    DENSE_AUG,
)

# ImageNet Normalization


IMAGENET_MEAN = (
    0.485,
    0.456,
    0.406,
)

IMAGENET_STD = (
    0.229,
    0.224,
    0.225,
)


# Sampling Weights


SAMPLING_WEIGHTS = {
    "background": BACKGROUND_AUG,
    "very_sparse": VERY_SPARSE_AUG,
    "sparse": SPARSE_AUG,
    "normal": NORMAL_AUG,
    "dense": DENSE_AUG,
}


# Sampling Weight


def get_sampling_weight(level: str) -> float:
    """
    Return sampling weight of a patch level.
    """

    level = level.lower()

    if level not in SAMPLING_WEIGHTS:
        raise ValueError(f"Unknown level: {level}")

    return float(SAMPLING_WEIGHTS[level])


# Base Transform


def get_base_transform(
    level: str,
) -> A.Compose:
    """
    Return augmentation pipeline without
    Normalize() and ToTensorV2().
    """

    level = level.lower()

    # Background

    if level == "background":

        transforms = [
            A.NoOp(),
        ]

    # Very Sparse

    elif level == "very_sparse":

        transforms = [
            A.HorizontalFlip(
                p=0.5,
            ),
            A.VerticalFlip(
                p=0.5,
            ),
            A.RandomRotate90(
                p=0.5,
            ),
        ]

    # Sparse

    elif level == "sparse":

        transforms = [
            A.HorizontalFlip(
                p=0.5,
            ),
            A.VerticalFlip(
                p=0.5,
            ),
            A.RandomRotate90(
                p=0.5,
            ),
            A.Affine(
                translate_percent=(-0.05, 0.05),  # Tương đương shift_limit
                scale=(0.95, 1.05),  # Tương đương scale_limit
                rotate=(-15, 15),  # Tương đương rotate_limit
                interpolation=cv2.INTER_LINEAR,
                border_mode=cv2.BORDER_CONSTANT,
                fill=0,
                fill_mask=0,
                p=0.50,
            ),
            A.RandomBrightnessContrast(
                brightness_limit=0.15,
                contrast_limit=0.15,
                p=0.30,
            ),
        ]

    # Normal

    elif level == "normal":

        transforms = [
            A.HorizontalFlip(
                p=0.5,
            ),
            A.VerticalFlip(
                p=0.5,
            ),
            A.RandomRotate90(
                p=0.5,
            ),
            A.Affine(
                translate_percent=(-0.08, 0.08),  # Tương đương shift_limit
                scale=(0.92, 1.08),  # Tương đương scale_limit
                rotate=(-20, 20),  # Tương đương rotate_limit
                interpolation=cv2.INTER_LINEAR,
                border_mode=cv2.BORDER_CONSTANT,
                fill=0,
                fill_mask=0,
                p=0.60,
            ),
            A.RandomBrightnessContrast(
                brightness_limit=0.20,
                contrast_limit=0.20,
                p=0.50,
            ),
            A.GaussNoise(
                std_range=(0.04, 0.08),
                p=0.20,
            ),
        ]

    # Dense

    elif level == "dense":

        transforms = [
            A.HorizontalFlip(
                p=0.5,
            ),
            A.VerticalFlip(
                p=0.5,
            ),
            A.RandomRotate90(
                p=0.5,
            ),
            A.Affine(
                translate_percent=(-0.1, 0.1),  # Tương đương shift_limit
                scale=(0.9, 1.1),  # Tương đương scale_limit
                rotate=(-20, 20),  # Tương đương rotate_limit
                interpolation=cv2.INTER_LINEAR,
                border_mode=cv2.BORDER_CONSTANT,
                fill=0,
                fill_mask=0,
                p=0.70,
            ),
            A.RandomBrightnessContrast(
                brightness_limit=0.20,
                contrast_limit=0.20,
                p=0.50,
            ),
            A.GaussNoise(
                std_range=(0.08, 0.15),
                p=0.30,
            ),
            A.GaussianBlur(
                blur_limit=(3, 5),
                p=0.20,
            ),
        ]

    else:

        raise ValueError(f"Unknown level: {level}")

    return A.Compose(
        transforms,
        is_check_shapes=False,
    )


# Train Transform


def get_train_transform(level: str) -> A.Compose:
    """
    Training transform.
    Augmentation + Normalize + ToTensor.
    """

    base_transform = get_base_transform(level)

    transforms = list(base_transform.transforms)

    transforms.extend(
        [
            A.Normalize(
                mean=IMAGENET_MEAN,
                std=IMAGENET_STD,
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ]
    )

    return A.Compose(
        transforms,
        is_check_shapes=False,
    )


# Validation Transform


def get_validation_transform() -> A.Compose:
    """
    Validation transform.
    No augmentation.
    """

    return A.Compose(
        [
            A.Normalize(
                mean=IMAGENET_MEAN,
                std=IMAGENET_STD,
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ],
        is_check_shapes=False,
    )


# Test Transform


def get_test_transform() -> A.Compose:
    """
    Test transform.
    No augmentation.
    """

    return A.Compose(
        [
            A.Normalize(
                mean=IMAGENET_MEAN,
                std=IMAGENET_STD,
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ],
        is_check_shapes=False,
    )


# Policy Factory


def get_policy(
    split: str,
    level: str = "normal",
) -> A.Compose:
    """
    Return transform according to dataset split.

    Train -> augmentation + tensor

    Val/Test -> normalize + tensor
    """

    split = split.strip().lower()

    if split == "train":
        return get_train_transform(level)

    if split == "val":
        return get_validation_transform()

    if split == "validation":
        return get_validation_transform()

    if split == "test":
        return get_test_transform()

    raise ValueError(f"Unknown split: {split}")


# Print Policies


def print_policies() -> None:
    """
    Print all augmentation pipelines.
    """

    print("=" * 70)
    print("TRAIN POLICIES")
    print("=" * 70)

    levels = [
        "background",
        "very_sparse",
        "sparse",
        "normal",
        "dense",
    ]

    for level in levels:

        print(f"\n[{level.upper()}]")

        transform = get_train_transform(level)

        for t in transform.transforms:
            print(f"  - {t}")

    print("\n" + "=" * 70)
    print("VALIDATION POLICY")
    print("=" * 70)

    for t in get_validation_transform().transforms:
        print(f"  - {t}")

    print("\n" + "=" * 70)
    print("TEST POLICY")
    print("=" * 70)

    for t in get_test_transform().transforms:
        print(f"  - {t}")


# Main


def main() -> None:

    print_policies()

    print("\n" + "=" * 70)
    print("SAMPLING WEIGHTS")
    print("=" * 70)

    for level in [
        "background",
        "very_sparse",
        "sparse",
        "normal",
        "dense",
    ]:

        weight = get_sampling_weight(level)

        print(f"{level:<15} : {weight}")

    print("\n" + "=" * 70)
    print("POLICY FACTORY TEST")
    print("=" * 70)

    print(
        "Train :",
        type(get_policy("Train", "dense")).__name__,
    )

    print(
        "Val   :",
        type(get_policy("Val")).__name__,
    )

    print(
        "Test  :",
        type(get_policy("Test")).__name__,
    )

    print("\nPolicy test completed successfully.")


# Entry Point


if __name__ == "__main__":
    main()
