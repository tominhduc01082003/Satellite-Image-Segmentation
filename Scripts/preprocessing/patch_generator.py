"""
patch_generator.py
==================

Generate image-mask patches using a sliding window.

Pipeline
--------
Image (.tif)
        +
Mask (.tif)
        │
        ▼
Sliding Window
        │
        ▼
Save Image Patch
Save Mask Patch
        │
        ▼
patch_index.csv
"""

from __future__ import annotations

from pathlib import Path
from typing import Generator

import numpy as np
import rasterio
import csv
import sys
from rasterio.windows import Window
from rasterio.windows import transform as window_transform
from tqdm import tqdm

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from Scripts.config.config import (
    IMAGES_DIR,
    MASKS_DIR,
    PATCHES_DIR,
    REPORT_DIR,
    SPLITS,
    PATCH_SIZE,
    PATCH_STRIDE,
    KEEP_PARTIAL,
    PATCH_MASK_SUFFIX,
    IMAGE_GLOB,
    PATCH_IMAGE_SUFFIX,
)

# ==========================================================
# Utility
# ==========================================================


def create_output_dirs(split: str) -> tuple[Path, Path]:
    """
    Create output directories if they do not exist.

    Resource/
        Patches/
            Train/
                Images/
                Masks/
    """

    image_dir = PATCHES_DIR / split / "Images"
    mask_dir = PATCHES_DIR / split / "Masks"

    image_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)

    return image_dir, mask_dir


def get_mask_path(image_path: Path) -> Path:
    """
    Convert

    img_1.tif

    ->

    img_1_mask.tif
    """

    split = image_path.parent.name

    return MASKS_DIR / split / (image_path.stem + PATCH_MASK_SUFFIX)


# ==========================================================
# Patch Coordinate Generator
# ==========================================================


def sliding_window(
    width: int,
    height: int,
    patch_size: int,
    stride: int,
):
    """
    Generate patch coordinates.

    Always returns PATCH_SIZE x PATCH_SIZE windows.
    The last window is shifted to touch the image border.
    """

    if patch_size > width or patch_size > height:
        raise ValueError(
            f"Patch size ({patch_size}) is larger than image size "
            f"({width} x {height})"
        )

    x_positions = list(range(0, width - patch_size + 1, stride))
    y_positions = list(range(0, height - patch_size + 1, stride))

    # Add last column
    if x_positions[-1] != width - patch_size or not x_positions:
        x_positions.append(width - patch_size)

    # Add last row
    if y_positions[-1] != height - patch_size or not y_positions:
        y_positions.append(height - patch_size)

    for y in y_positions:
        for x in x_positions:
            yield x, y


# ==========================================================
# Read Patch
# ==========================================================


def read_patch(
    dataset: rasterio.io.DatasetReader,
    x: int,
    y: int,
    patch_size: int,
):
    """
    Read one patch from raster.

    Returns
    -------
    patch
    transform
    """

    window = Window(
        col_off=x,
        row_off=y,
        width=patch_size,
        height=patch_size,
    )

    patch = dataset.read(window=window)

    transform = window_transform(
        window,
        dataset.transform,
    )

    return patch, transform


# ==========================================================
# Save Patch
# ==========================================================


def save_patch(
    patch: np.ndarray,
    profile: dict,
    transform,
    output_path: Path,
):
    """
    Save patch to GeoTIFF.

    Parameters
    ----------
    patch

    profile

    transform

    output_path
    """

    profile = profile.copy()

    profile.update(
        {
            "height": patch.shape[1],
            "width": patch.shape[2],
            "transform": transform,
        }
    )

    with rasterio.open(
        output_path,
        "w",
        **profile,
    ) as dst:

        dst.write(patch)


# ==========================================================
# Patch Name
# ==========================================================


def build_patch_name(
    image_name: str,
    patch_index: int,
) -> tuple[str, str]:
    """
    Example
    -------

    img_1_patch_0001.tif

    img_1_patch_0001_mask.tif
    """

    stem = f"{image_name}_patch_{patch_index:04d}"

    image_name = stem + PATCH_IMAGE_SUFFIX
    mask_name = stem + PATCH_MASK_SUFFIX

    return image_name, mask_name


# ==========================================================
# Process One Image
# ==========================================================
def process_image(
    image_path: Path,
    mask_path: Path,
    image_output_dir: Path,
    mask_output_dir: Path,
    csv_writer,
) -> int:
    """
    Process one image-mask pair.

    Returns
    -------
    int
        Number of generated patches.
    """

    if not image_path.exists():
        raise FileNotFoundError(image_path)

    if not mask_path.exists():
        raise FileNotFoundError(mask_path)

    patch_count = 0

    with rasterio.open(image_path) as image_ds, rasterio.open(mask_path) as mask_ds:

        # Check size
        if image_ds.width != mask_ds.width or image_ds.height != mask_ds.height:
            raise ValueError(
                f"Image and mask size mismatch:\n"
                f"{image_path.name}\n"
                f"{mask_path.name}"
            )

        image_profile = image_ds.profile.copy()
        mask_profile = mask_ds.profile.copy()

        for x, y in sliding_window(
            width=image_ds.width,
            height=image_ds.height,
            patch_size=PATCH_SIZE,
            stride=PATCH_STRIDE,
        ):

            image_patch, image_transform = read_patch(
                image_ds,
                x,
                y,
                PATCH_SIZE,
            )

            mask_patch, mask_transform = read_patch(
                mask_ds,
                x,
                y,
                PATCH_SIZE,
            )

            patch_count += 1

            image_patch_name, mask_patch_name = build_patch_name(
                image_path.stem,
                patch_count,
            )

            image_patch_path = image_output_dir / image_patch_name
            mask_patch_path = mask_output_dir / mask_patch_name

            save_patch(
                image_patch,
                image_profile,
                image_transform,
                image_patch_path,
            )

            save_patch(
                mask_patch,
                mask_profile,
                mask_transform,
                mask_patch_path,
            )

            csv_writer.writerow(
                [
                    image_patch_name,
                    mask_patch_name,
                    image_path.name,
                    x,
                    y,
                    PATCH_SIZE,
                    PATCH_SIZE,
                ]
            )

    return patch_count


# ==========================================================
# Patch Index
# ==========================================================


def create_patch_index():
    """
    Create patch_index.csv
    """

    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    csv_path = REPORT_DIR / "patch_index.csv"

    csv_file = open(
        csv_path,
        "w",
        newline="",
        encoding="utf-8",
    )

    writer = csv.writer(csv_file)

    writer.writerow(
        [
            "image_patch",
            "mask_patch",
            "source_image",
            "x",
            "y",
            "patch_width",
            "patch_height",
        ]
    )

    return csv_file, writer


# ==========================================================
# Process Split
# ==========================================================


def process_split(
    split: str,
    csv_writer,
):
    """
    Process Train / Val / Test
    """

    image_dir = IMAGES_DIR / split

    image_output_dir, mask_output_dir = create_output_dirs(split)

    image_files = sorted(image_dir.glob(IMAGE_GLOB))

    if len(image_files) == 0:
        print(f"[WARNING] No image found in {image_dir}")
        return

    total_patch = 0

    for image_path in tqdm(
        image_files,
        desc=f"{split}",
        unit="image",
    ):

        mask_path = get_mask_path(image_path)

        total_patch += process_image(
            image_path=image_path,
            mask_path=mask_path,
            image_output_dir=image_output_dir,
            mask_output_dir=mask_output_dir,
            csv_writer=csv_writer,
        )

    print(f"{split}: " f"{len(image_files)} images -> " f"{total_patch} patches")


# ==========================================================
# Generate Patches
# ==========================================================


def generate_patches():
    """
    Generate patches for all dataset splits.
    """

    print("=" * 60)
    print("PATCH GENERATOR")
    print("=" * 60)
    print(f"Patch Size : {PATCH_SIZE}")
    print(f"Stride     : {PATCH_STRIDE}")
    print(f"Output     : {PATCHES_DIR}")
    print()

    csv_file, csv_writer = create_patch_index()

    try:

        for split in SPLITS:

            process_split(
                split=split,
                csv_writer=csv_writer,
            )

    finally:

        csv_file.close()

    print()
    print("=" * 60)
    print("Patch generation completed.")
    print(f"Patch index saved to:")
    print(REPORT_DIR / "patch_index.csv")
    print("=" * 60)


# ==========================================================
# Main
# ==========================================================


def main():

    generate_patches()


if __name__ == "__main__":

    main()
