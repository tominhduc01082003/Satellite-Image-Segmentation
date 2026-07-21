"""
create_dataset.py
=================

PyTorch Dataset for patch-based building segmentation.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import rasterio
import torch
from torch.utils.data import Dataset

ROOT_DIR = Path(__file__).resolve().parent.parent.parent

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from Scripts.config.config import (
    PATCHES_DIR,
    REPORT_DIR,
    SPLITS,
    BALANCED_PATCH_CSV,
)

from Scripts.augmentation.policies import get_policy


# ==========================================================
# Patch Dataset
# ==========================================================
class PatchDataset(Dataset):
    """
    Patch Dataset.

    Đọc patch đã được tạo sẵn trong Resource/Patches.
    Chỉ sử dụng các patch được giữ lại trong balanced_patch.csv.
    """

    def __init__(
        self,
        split: str,
    ):
        super().__init__()

        split = split.capitalize()

        if split not in SPLITS:
            raise ValueError(f"Split must be one of {SPLITS}")

        self.split = split

        # --------------------------------------------------
        # Patch folders
        # --------------------------------------------------

        self.image_dir = PATCHES_DIR / split / "Images"

        self.mask_dir = PATCHES_DIR / split / "Masks"

        if not self.image_dir.exists():
            raise FileNotFoundError(self.image_dir)

        if not self.mask_dir.exists():
            raise FileNotFoundError(self.mask_dir)

        # --------------------------------------------------
        # Balanced CSV
        # --------------------------------------------------

        if not BALANCED_PATCH_CSV.exists():
            raise FileNotFoundError(BALANCED_PATCH_CSV)

        df = pd.read_csv(
            BALANCED_PATCH_CSV,
        )

        # --------------------------------------------------
        # Filter split
        # --------------------------------------------------

        df = df[df["split"].str.lower() == split.lower()].copy()

        # --------------------------------------------------
        # Keep policy
        # --------------------------------------------------

        if "keep" in df.columns:

            df = df[df["keep"] == True].copy()

        df.reset_index(
            drop=True,
            inplace=True,
        )

        self.data = df

        # --------------------------------------------------
        # Transform
        # --------------------------------------------------

        self.default_transform = get_policy(
            split=self.split,
            level="normal",
        )

        print(f"[{self.split}] Loaded {len(self.data)} patches.")

    # ==========================================================
    # Dataset Length
    # ==========================================================

    def __len__(self):
        """
        Return number of patches.
        """

        return len(self.data)

    # ======================================================
    # Read Image
    # ======================================================

    def _read_image(
        self,
        image_path: Path,
    ):
        """
        Read image patch.
        """

        with rasterio.open(image_path) as src:

            image = src.read()

        image = image.transpose(
            1,
            2,
            0,
        )

        # Chỉ lấy 3 kênh đầu nếu nhiều hơn
        if image.shape[-1] > 3:
            image = image[:, :, :3]

        image = image.astype("float32")

        return image

    # ======================================================
    # Read Mask
    # ======================================================

    def _read_mask(
        self,
        mask_path: Path,
    ):
        """
        Read mask patch.
        """

        with rasterio.open(mask_path) as src:

            mask = src.read(1)

        return mask.astype("int64")

    # ======================================================
    # Get Item
    # ======================================================

    def __getitem__(
        self,
        index: int,
    ):
        """
        Return one training sample.
        """

        row = self.data.iloc[index]

        patch_name = row["patch_name"]
        # BẮT ĐẦU FIX LỖI TÊN FILE
        # Tự động gọt bỏ chữ "_mask" nếu nó bị dính vào tên ảnh gốc
        if patch_name.endswith("_mask.tif"):
            patch_name = patch_name.replace("_mask.tif", ".tif")

        level = row.get(
            "level",
            "normal",
        )

        image_path = self.image_dir / patch_name

        mask_name = patch_name.replace(
            ".tif",
            "_mask.tif",
        )

        mask_path = self.mask_dir / mask_name

        if not image_path.exists():
            raise FileNotFoundError(image_path)

        if not mask_path.exists():
            raise FileNotFoundError(mask_path)

        image = self._read_image(
            image_path,
        )

        mask = self._read_mask(
            mask_path,
        )

        # --------------------------------------------------
        # Augmentation Policy
        # --------------------------------------------------

        transform = get_policy(
            split=self.split,
            level=level,
        )

        transformed = transform(
            image=image,
            mask=mask,
        )

        image = transformed["image"]

        mask = transformed["mask"]

        # --------------------------------------------------
        # Metadata
        # --------------------------------------------------

        sample = {
            "image": image,
            "mask": mask.long(),
            "patch_name": patch_name,
            "level": level,
            "split": self.split,
        }

        if "foreground_ratio" in row:
            sample["foreground_ratio"] = float(row["foreground_ratio"])

        if "background_ratio" in row:
            sample["background_ratio"] = float(row["background_ratio"])

        if "object_count" in row:
            sample["object_count"] = int(row["object_count"])

        if "sampling_weight" in row:
            sample["sampling_weight"] = float(row["sampling_weight"])

        return sample


# ==========================================================
# Dataset Factory
# ==========================================================


def create_dataset(
    split: str,
):
    """
    Create PatchDataset.
    """

    split = split.capitalize()

    if split not in SPLITS:
        raise ValueError(f"Split must be one of {SPLITS}")

    dataset = PatchDataset(
        split=split,
    )

    return dataset


# ==========================================================
# Dataset Info
# ==========================================================


def print_dataset_info(
    dataset: PatchDataset,
    name: str = "Dataset",
):
    """
    Print dataset information.
    """

    print("=" * 70)
    print(name)
    print("=" * 70)

    print(f"Split           : {dataset.split}")
    print(f"Image Folder    : {dataset.image_dir}")
    print(f"Mask Folder     : {dataset.mask_dir}")
    print(f"Number of Patch : {len(dataset)}")

    if len(dataset) > 0:

        levels = dataset.data["level"].value_counts().sort_index()

        print("\nPatch Levels")

        for level, count in levels.items():

            print(f"  {level:<15}: {count}")

    print()


# ==========================================================
# Main
# ==========================================================


def main():
    """
    Test PatchDataset.
    """

    print("=" * 70)
    print("PATCH DATASET TEST")
    print("=" * 70)

    for split in SPLITS:

        try:

            dataset = create_dataset(split)

            print_dataset_info(
                dataset,
                split,
            )

            if len(dataset) == 0:

                print("Dataset rỗng.\n")
                continue

            sample = dataset[0]

            print("First Sample")
            print("-" * 70)

            print(f"Patch Name      : {sample['patch_name']}")

            print(f"Level           : {sample['level']}")

            print(f"Image Shape     : {tuple(sample['image'].shape)}")

            print(f"Mask Shape      : {tuple(sample['mask'].shape)}")

            print(f"Image Dtype     : {sample['image'].dtype}")

            print(f"Mask Dtype      : {sample['mask'].dtype}")

            if "sampling_weight" in sample:

                print(f"Sampling Weight : {sample['sampling_weight']}")

            print()

        except Exception as e:

            print(f"[ERROR] {split}: {e}")
            print()


# ==========================================================
# Entry Point
# ==========================================================

if __name__ == "__main__":
    main()
