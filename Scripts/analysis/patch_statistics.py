"""
patch_statistics.py
===================

Statistics for every generated mask patch.
"""

from __future__ import annotations
import sys
from pathlib import Path
import csv
import cv2
import numpy as np
import rasterio
from tqdm import tqdm

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from Scripts.config.config import (
    PATCHES_DIR,
    REPORT_DIR,
    SPLITS,
)

# ==========================================================
# Utility
# ==========================================================


def get_patch_dirs(split: str) -> tuple[Path, Path]:
    """
    Return image patch directory and mask patch directory.
    """

    image_dir = PATCHES_DIR / split / "Images"
    mask_dir = PATCHES_DIR / split / "Masks"

    if not image_dir.exists():
        raise FileNotFoundError(image_dir)

    if not mask_dir.exists():
        raise FileNotFoundError(mask_dir)

    return image_dir, mask_dir


def get_mask_files(split: str) -> list[Path]:
    """
    Get all mask patches.
    """

    _, mask_dir = get_patch_dirs(split)

    return sorted(mask_dir.glob("*_mask.tif"))


# ==========================================================
# Read Mask
# ==========================================================


def read_mask(mask_path: Path) -> np.ndarray:
    """
    Read mask patch.

    Returns
    -------
    ndarray (H, W)
    """

    with rasterio.open(mask_path) as src:

        mask = src.read(1)

    return mask


# ==========================================================
# Pixel Statistics
# ==========================================================


def calculate_pixel_statistics(mask: np.ndarray) -> dict:
    """
    Calculate foreground/background statistics.

    Returns
    -------
    dict
    """

    total_pixels = mask.size

    object_pixels = int(np.count_nonzero(mask))

    background_pixels = total_pixels - object_pixels

    object_ratio = object_pixels / total_pixels

    background_ratio = background_pixels / total_pixels

    return {
        "total_pixels": total_pixels,
        "object_pixels": object_pixels,
        "background_pixels": background_pixels,
        "object_ratio": object_ratio,
        "background_ratio": background_ratio,
    }


# ==========================================================
# Connected Components
# ==========================================================


def count_objects(mask: np.ndarray) -> int:
    """
    Count connected objects.

    Background = 0
    Foreground > 0
    """

    binary = (mask > 0).astype(np.uint8)

    num_labels, _, _, _ = cv2.connectedComponentsWithStats(
        binary,
        connectivity=8,
    )

    # Remove background label
    return max(0, num_labels - 1)


# ==========================================================
# Process One Patch
# ==========================================================


def process_patch(mask_path: Path) -> dict:
    """
    Process one mask patch.

    Returns
    -------
    dict
    """

    mask = read_mask(mask_path)

    stats = calculate_pixel_statistics(mask)

    stats["num_objects"] = count_objects(mask)

    stats["patch_name"] = mask_path.name

    stats["split"] = mask_path.parent.parent.name

    stats["height"] = mask.shape[0]

    stats["width"] = mask.shape[1]

    return stats


# ==========================================================
# CSV
# ==========================================================


def create_statistics_csv():
    """
    Create patch_statistics.csv
    """

    csv_path = REPORT_DIR / "patch_statistics.csv"

    csv_file = open(
        csv_path,
        "w",
        newline="",
        encoding="utf-8",
    )

    writer = csv.writer(csv_file)

    writer.writerow(
        [
            "split",
            "patch_name",
            "width",
            "height",
            "total_pixels",
            "object_pixels",
            "background_pixels",
            "object_ratio",
            "background_ratio",
            "num_objects",
        ]
    )

    return csv_file, writer


# ==========================================================
# Write One Row
# ==========================================================


def write_statistics(
    writer,
    stats: dict,
):
    """
    Write one patch statistics.
    """

    writer.writerow(
        [
            stats["split"],
            stats["patch_name"],
            stats["width"],
            stats["height"],
            stats["total_pixels"],
            stats["object_pixels"],
            stats["background_pixels"],
            f'{stats["object_ratio"]:.6f}',
            f'{stats["background_ratio"]:.6f}',
            stats["num_objects"],
        ]
    )


# ==========================================================
# Process Split
# ==========================================================


def process_split(
    split: str,
    writer,
):
    """
    Process one dataset split.
    """

    mask_files = get_mask_files(split)

    if len(mask_files) == 0:
        print(f"[WARNING] No mask patches found in {split}")
        return

    for mask_path in tqdm(
        mask_files,
        desc=split,
        unit="patch",
    ):

        stats = process_patch(mask_path)

        write_statistics(
            writer,
            stats,
        )


# ==========================================================
# Generate Statistics
# ==========================================================


def generate_statistics():
    """
    Generate patch_statistics.csv
    """

    print("=" * 60)
    print("PATCH STATISTICS")
    print("=" * 60)

    csv_file, writer = create_statistics_csv()

    try:

        for split in SPLITS:

            process_split(
                split=split,
                writer=writer,
            )

    finally:

        csv_file.close()

    print()
    print("=" * 60)
    print("Patch statistics completed.")
    print(f"Saved to: {REPORT_DIR / 'patch_statistics.csv'}")
    print("=" * 60)


# ==========================================================
# Main
# ==========================================================


def main():

    generate_statistics()


if __name__ == "__main__":

    main()
