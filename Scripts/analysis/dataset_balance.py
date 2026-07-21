"""
dataset_balance.py
==================

Balance dataset according to patch statistics.
"""

from __future__ import annotations

import random

import pandas as pd

import json
import sys
from pathlib import Path
import matplotlib.pyplot as plt

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from Scripts.config.config import (
    PATCH_STATISTICS_CSV,
    PATCH_LEVEL_CSV,
    BALANCED_PATCH_CSV,
    DATASET_REPORT_JSON,
    PATCH_DISTRIBUTION_PNG,
    BACKGROUND_THRESHOLD,
    VERY_SPARSE_THRESHOLD,
    SPARSE_THRESHOLD,
    NORMAL_THRESHOLD,
    BACKGROUND_KEEP_RATIO,
    VERY_SPARSE_KEEP_RATIO,
    SPARSE_KEEP_RATIO,
    NORMAL_KEEP_RATIO,
    DENSE_KEEP_RATIO,
    BACKGROUND_AUG,
    VERY_SPARSE_AUG,
    SPARSE_AUG,
    NORMAL_AUG,
    DENSE_AUG,
    BALANCE_SPLITS,
)

# ==========================================================
# Random Seed
# ==========================================================

RANDOM_SEED = 42

random.seed(RANDOM_SEED)


# ==========================================================
# Read CSV
# ==========================================================


def load_patch_statistics() -> pd.DataFrame:
    """
    Load patch_statistics.csv.
    """

    if not PATCH_STATISTICS_CSV.exists():
        raise FileNotFoundError(f"File not found:\n{PATCH_STATISTICS_CSV}")

    return pd.read_csv(PATCH_STATISTICS_CSV)


# ==========================================================
# Patch Classification
# ==========================================================


def classify_patch(object_ratio: float) -> str:
    """
    Classify patch according to object ratio.
    """

    if object_ratio <= BACKGROUND_THRESHOLD:
        return "background"

    elif object_ratio <= VERY_SPARSE_THRESHOLD:
        return "very_sparse"

    elif object_ratio <= SPARSE_THRESHOLD:
        return "sparse"

    elif object_ratio <= NORMAL_THRESHOLD:
        return "normal"

    else:
        return "dense"


# ==========================================================
# Keep Ratio
# ==========================================================


def get_keep_ratio(level: str) -> float:
    """
    Return keep ratio.
    """

    keep_ratio = {
        "background": BACKGROUND_KEEP_RATIO,
        "very_sparse": VERY_SPARSE_KEEP_RATIO,
        "sparse": SPARSE_KEEP_RATIO,
        "normal": NORMAL_KEEP_RATIO,
        "dense": DENSE_KEEP_RATIO,
    }

    return keep_ratio[level]


# ==========================================================
# Augmentation Times
# ==========================================================


def get_augmentation_times(level: str) -> int:
    """
    Return augmentation times.
    """

    augmentation = {
        "background": BACKGROUND_AUG,
        "very_sparse": VERY_SPARSE_AUG,
        "sparse": SPARSE_AUG,
        "normal": NORMAL_AUG,
        "dense": DENSE_AUG,
    }

    return augmentation[level]


# ==========================================================
# Build Patch Levels
# ==========================================================


def build_patch_levels(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add level, keep_ratio and augment_times.
    """

    df = df.copy()

    df["level"] = df["object_ratio"].apply(
        classify_patch,
    )

    keep_ratio = []
    augment_times = []

    for _, row in df.iterrows():

        if row["split"] in BALANCE_SPLITS:

            keep_ratio.append(get_keep_ratio(row["level"]))

            augment_times.append(get_augmentation_times(row["level"]))

        else:

            keep_ratio.append(1.0)

            augment_times.append(0)

    df["keep_ratio"] = keep_ratio

    df["augment_times"] = augment_times

    return df


# ==========================================================
# Random Keep
# ==========================================================


def random_keep(df: pd.DataFrame) -> pd.DataFrame:
    """
    Randomly keep patches according to keep_ratio.
    """

    df = df.copy()

    df["keep"] = df["keep_ratio"].apply(lambda ratio: random.random() <= ratio)

    return df


# ==========================================================
# Save patch_level.csv
# ==========================================================


def save_patch_level(df: pd.DataFrame) -> None:
    """
    Save patch level information.
    """

    columns = [
        "split",
        "patch_name",
        "width",
        "height",
        "object_ratio",
        "background_ratio",
        "num_objects",
        "level",
        "keep_ratio",
        "augment_times",
    ]

    df[columns].to_csv(
        PATCH_LEVEL_CSV,
        index=False,
    )

    print(f"[INFO] Saved: {PATCH_LEVEL_CSV}")


# ==========================================================
# Build balanced dataframe
# ==========================================================


def build_balanced_dataframe(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Randomly keep patches according to keep_ratio.
    """

    balanced_df = random_keep(df)

    balanced_df = balanced_df[balanced_df["keep"]].copy()

    balanced_df.reset_index(
        drop=True,
        inplace=True,
    )

    return balanced_df


# ==========================================================
# Save balanced_patch.csv
# ==========================================================


def save_balanced_patch(
    balanced_df: pd.DataFrame,
) -> None:
    """
    Save balanced dataset.
    """

    columns = [
        "split",
        "patch_name",
        "width",
        "height",
        "object_ratio",
        "background_ratio",
        "num_objects",
        "level",
        "keep_ratio",
        "augment_times",
    ]

    balanced_df[columns].to_csv(
        BALANCED_PATCH_CSV,
        index=False,
    )

    print(f"[INFO] Saved: {BALANCED_PATCH_CSV}")


# ==========================================================
# Generate patch level & balanced csv
# ==========================================================


def generate_balance_table(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Generate patch_level.csv and balanced_patch.csv.
    """

    level_df = build_patch_levels(df)

    save_patch_level(level_df)

    balanced_df = build_balanced_dataframe(
        level_df,
    )

    save_balanced_patch(
        balanced_df,
    )

    return balanced_df


# ==========================================================
# Dataset Report
# ==========================================================


def generate_dataset_report(
    level_df: pd.DataFrame,
    balanced_df: pd.DataFrame,
) -> dict:
    """
    Generate dataset report.
    """

    report = {
        "total_patches": int(len(level_df)),
        "kept_patches": int(len(balanced_df)),
        "removed_patches": int(len(level_df) - len(balanced_df)),
        "keep_ratio": round(
            len(balanced_df) / len(level_df),
            4,
        ),
        "levels": {},
    }

    level_order = [
        "background",
        "very_sparse",
        "sparse",
        "normal",
        "dense",
    ]

    for level in level_order:

        subset = level_df[level_df["level"] == level]
        kept = balanced_df[balanced_df["level"] == level]

        report["levels"][level] = {
            "total": int(len(subset)),
            "kept": int(len(kept)),
            "removed": int(len(subset) - len(kept)),
            "augment_times": (
                get_augmentation_times(level) if "Train" in BALANCE_SPLITS else 0
            ),
        }

    with open(
        DATASET_REPORT_JSON,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            report,
            f,
            indent=4,
            ensure_ascii=False,
        )

    print(f"[INFO] Saved: {DATASET_REPORT_JSON}")

    return report


# ==========================================================
# Patch Distribution
# ==========================================================


def plot_patch_distribution(
    level_df: pd.DataFrame,
) -> None:
    """
    Plot patch distribution.
    """

    level_order = [
        "background",
        "very_sparse",
        "sparse",
        "normal",
        "dense",
    ]

    counts = level_df["level"].value_counts().reindex(level_order, fill_value=0)

    plt.figure(figsize=(8, 5))

    plt.bar(
        counts.index,
        counts.values,
    )

    plt.title("Patch Distribution")

    plt.xlabel("Patch Level")

    plt.ylabel("Number of Patches")

    plt.grid(
        axis="y",
        linestyle="--",
        alpha=0.4,
    )

    plt.tight_layout()

    plt.savefig(
        PATCH_DISTRIBUTION_PNG,
        dpi=300,
    )

    plt.close()

    print(f"[INFO] Saved: {PATCH_DISTRIBUTION_PNG}")


# ==========================================================
# Generate Balance
# ==========================================================


def generate_dataset_balance() -> None:
    """
    Generate all balance reports.
    """

    print("=" * 60)
    print("DATASET BALANCE")
    print("=" * 60)

    # Read statistics
    df = load_patch_statistics()

    # Build patch levels
    level_df = build_patch_levels(df)

    # Save patch_level.csv
    save_patch_level(level_df)

    # Build balanced dataset
    balanced_df = build_balanced_dataframe(level_df)

    # Save balanced_patch.csv
    save_balanced_patch(balanced_df)

    # Save dataset_report.json
    generate_dataset_report(
        level_df=level_df,
        balanced_df=balanced_df,
    )

    # Save distribution figure
    plot_patch_distribution(level_df)

    print()
    print("=" * 60)
    print("Dataset balance completed.")
    print(f"Patch Level       : {PATCH_LEVEL_CSV}")
    print(f"Balanced Dataset  : {BALANCED_PATCH_CSV}")
    print(f"Dataset Report    : {DATASET_REPORT_JSON}")
    print(f"Distribution Plot : {PATCH_DISTRIBUTION_PNG}")
    print("=" * 60)


# ==========================================================
# Main
# ==========================================================


def main():

    generate_dataset_balance()


if __name__ == "__main__":

    main()
