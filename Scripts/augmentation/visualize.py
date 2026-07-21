"""
visualize.py
============

Visual debugger for Albumentations.
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import rasterio
import matplotlib.pyplot as plt
import albumentations as A

ROOT_DIR = Path(__file__).resolve().parent.parent.parent

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from Scripts.config.config import PATCHES_DIR
from Scripts.augmentation.policies import get_base_transform

TRAIN_IMAGE_DIR = PATCHES_DIR / "Train" / "Images"

TRAIN_MASK_DIR = PATCHES_DIR / "Train" / "Masks"


def read_patch(
    image_path: Path,
    mask_path: Path,
):
    """
    Read image patch and mask patch.
    """

    with rasterio.open(image_path) as src:

        image = src.read()

    image = np.transpose(
        image,
        (1, 2, 0),
    )

    if image.shape[2] > 3:

        image = image[:, :, :3]

    image = np.nan_to_num(image)

    image = np.clip(
        image,
        0,
        255,
    ).astype(np.uint8)

    with rasterio.open(mask_path) as src:

        mask = src.read(1)

    mask = np.nan_to_num(mask)

    mask = mask.astype(np.uint8)

    return image, mask


# ==========================================================
# Get Sample Patch
# ==========================================================


def get_sample_patch():
    """
    Get one patch for visualization.
    """

    image_files = sorted(TRAIN_IMAGE_DIR.glob("*.tif"))

    if len(image_files) == 0:

        raise FileNotFoundError(f"No patch found in:\n{TRAIN_IMAGE_DIR}")

    image_path = image_files[5] if len(image_files) > 5 else image_files[0]

    mask_path = TRAIN_MASK_DIR / image_path.name.replace(
        ".tif",
        "_mask.tif",
    )

    if not mask_path.exists():

        raise FileNotFoundError(f"Mask not found:\n{mask_path}")

    print("=" * 70)
    print("Visualization Sample")
    print("=" * 70)
    print(f"Image : {image_path.name}")
    print(f"Mask  : {mask_path.name}")
    print()

    return read_patch(
        image_path=image_path,
        mask_path=mask_path,
    )


# ==========================================================
# Draw Largest Object
# ==========================================================


def draw_single_bbox_with_coords(
    image: np.ndarray,
    mask: np.ndarray,
):
    """
    Draw bounding box and center coordinate
    of the largest connected component.
    """

    output = image.copy()

    binary = (mask > 0).astype(np.uint8)

    contours, _ = cv2.findContours(
        binary,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    if len(contours) == 0:

        cv2.putText(
            output,
            "NO OBJECT",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 0, 0),
            2,
            cv2.LINE_AA,
        )

        return output

    largest = max(
        contours,
        key=cv2.contourArea,
    )

    area = cv2.contourArea(
        largest,
    )

    if area < 5:

        cv2.putText(
            output,
            "OBJECT TOO SMALL",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 0, 0),
            2,
            cv2.LINE_AA,
        )

        return output

    x, y, w, h = cv2.boundingRect(
        largest,
    )

    center_x = x + w // 2

    center_y = y + h // 2

    # Bounding Box

    cv2.rectangle(
        output,
        (x, y),
        (x + w, y + h),
        (255, 0, 0),
        2,
    )

    # Center Point

    cv2.circle(
        output,
        (center_x, center_y),
        4,
        (255, 255, 0),
        -1,
    )

    # Coordinate

    cv2.putText(
        output,
        f"({center_x},{center_y})",
        (center_x + 8, center_y - 8),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 0),
        2,
        cv2.LINE_AA,
    )

    # Area

    cv2.putText(
        output,
        f"Area={int(area)}",
        (x, max(20, y - 10)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 255, 255),
        2,
        cv2.LINE_AA,
    )

    return output


# ==========================================================
# Show Result
# ==========================================================


def show_result(
    title: str,
    image: np.ndarray,
    mask: np.ndarray,
    aug_image: np.ndarray,
    aug_mask: np.ndarray,
):
    """
    Display original image/mask and augmented image/mask.
    """

    fig, axes = plt.subplots(
        1,
        4,
        figsize=(22, 6),
    )

    try:
        fig.canvas.manager.set_window_title(title)
    except Exception:
        pass

    fig.suptitle(
        title,
        fontsize=18,
        fontweight="bold",
    )

    axes[0].imshow(image)
    axes[0].set_title("Original Image")

    axes[1].imshow(mask, cmap="gray")
    axes[1].set_title("Original Mask")

    axes[2].imshow(aug_image)
    axes[2].set_title("Augmented Image")

    axes[3].imshow(aug_mask, cmap="gray")
    axes[3].set_title("Augmented Mask")

    for ax in axes:
        ax.axis("off")

    plt.tight_layout()
    plt.show()


# ==========================================================
# Build Test Suite
# ==========================================================


def build_test_suite(level: str = "dense"):
    """
    Build test suite from policies.py.

    Mỗi transform được tách riêng để kiểm tra.
    """

    compose = get_base_transform(level)

    suite = []

    for transform in compose.transforms:

        transform.p = 1.0

        suite.append(
            (
                transform.__class__.__name__,
                A.Compose(
                    [transform],
                    is_check_shapes=False,
                ),
            )
        )

    return suite


# ==========================================================
# Visualize One Augmentation
# ==========================================================


def visualize_single_augmentation(
    image: np.ndarray,
    mask: np.ndarray,
    name: str,
    transform: A.Compose,
):
    """
    Hiển thị kết quả trước và sau augment.
    """

    print("=" * 70)
    print(f"Testing Augmentation : {name}")
    print("=" * 70)

    original = draw_single_bbox_with_coords(
        image,
        mask,
    )

    augmented = transform(
        image=image,
        mask=mask,
    )

    aug_image = augmented["image"]
    aug_mask = augmented["mask"]

    augmented_draw = draw_single_bbox_with_coords(
        aug_image,
        aug_mask,
    )

    show_result(
        title=f"Testing : {name}",
        image=original,
        mask=mask,
        aug_image=augmented_draw,
        aug_mask=aug_mask,
    )


# ==========================================================
# Visualize All Augmentations
# ==========================================================


def visualize_augmentations(
    level: str = "dense",
):
    """
    Hiển thị lần lượt toàn bộ augmentation
    của level được chọn.
    """

    image_files = sorted(TRAIN_IMAGE_DIR.glob("*.tif"))

    if len(image_files) == 0:

        raise FileNotFoundError("Không tìm thấy patch image.")

    image_path = image_files[min(5, len(image_files) - 1)]

    mask_path = TRAIN_MASK_DIR / image_path.name.replace(
        ".tif",
        "_mask.tif",
    )

    image, mask = read_patch(
        image_path,
        mask_path,
    )

    suite = build_test_suite(level)

    total = len(suite)

    print("=" * 70)
    print(f"LEVEL : {level.upper()}")
    print(f"TOTAL AUGMENTATIONS : {total}")
    print("=" * 70)

    for index, (name, transform) in enumerate(
        suite,
        start=1,
    ):

        print()
        print("-" * 70)
        print(f"[{index}/{total}]")
        print(f"Current Augmentation : {name}")
        print("Close figure to continue...")
        print("-" * 70)

        visualize_single_augmentation(
            image=image,
            mask=mask,
            name=name,
            transform=transform,
        )

    print()
    print("=" * 70)
    print("ALL AUGMENTATIONS FINISHED")
    print("=" * 70)


# ==========================================================
# Main
# ==========================================================


def main():
    """
    Visual Debugger Entry Point.
    """
    print("=" * 70)
    print("ALBUMENTATIONS VISUAL DEBUGGER")
    print("=" * 70)

    visualize_augmentations(level="dense")

    print("Tất cả augmentation hoạt động hoàn tất.")


# ==========================================================
# Entry
# ==========================================================

if __name__ == "__main__":
    main()
