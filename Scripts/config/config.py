from pathlib import Path

import yaml

# ==========================================================
# PROJECT
# ==========================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

YAML_PATH = Path(__file__).resolve().parent / "config.yaml"

# ==========================================================
# LOAD YAML
# ==========================================================

try:
    with open(YAML_PATH, "r", encoding="utf-8") as f:
        _yaml_data = yaml.safe_load(f)

except FileNotFoundError:
    raise FileNotFoundError(f"Không tìm thấy file cấu hình:\n{YAML_PATH}")

# ==========================================================
# PATHS
# ==========================================================

PATHS = _yaml_data["paths"]

IMAGES_DIR = PROJECT_ROOT / PATHS["images_dir"]

LABELS_DIR = PROJECT_ROOT / PATHS["labels_dir"]

MASKS_DIR = PROJECT_ROOT / PATHS["masks_dir"]

PATCHES_DIR = PROJECT_ROOT / PATHS["patches_dir"]

REPORT_DIR = PROJECT_ROOT / PATHS["reports_dir"]

TRAIN_LABELS_DIR = PROJECT_ROOT / PATHS["train_labels_dir"]

VAL_LABELS_DIR = PROJECT_ROOT / PATHS["val_labels_dir"]

TEST_LABELS_DIR = PROJECT_ROOT / PATHS["test_labels_dir"]

DEFAULT_CHECK_SHP_PATH = TRAIN_LABELS_DIR / PATHS["default_check_shp"]

# ==========================================================
# PREPROCESSING
# ==========================================================

PREPROCESSING = _yaml_data.get("preprocessing", {})

MIN_AREA = float(
    PREPROCESSING.get(
        "overlap_min_area",
        1e-10,
    )
)

GRID_SIZE = float(
    PREPROCESSING.get(
        "overlap_grid_size",
        0.001,
    )
)

MAX_ITER = int(
    PREPROCESSING.get(
        "max_iterations",
        20,
    )
)

SPLITS = PREPROCESSING.get(
    "splits",
    [
        "Train",
        "Val",
        "Test",
    ],
)

CLASS_COL = PREPROCESSING.get(
    "class_column",
    "class_id",
)

MASK_SUFFIX = PREPROCESSING.get(
    "mask_suffix",
    "_mask",
)

# ==========================================================
# PATCH CONFIG
# ==========================================================

PATCH_CONFIG = _yaml_data.get(
    "patch",
    {},
)

PATCH_SIZE = int(
    PATCH_CONFIG.get(
        "patch_size",
        256,
    )
)

PATCH_STRIDE = int(
    PATCH_CONFIG.get(
        "stride",
        64,
    )
)

KEEP_PARTIAL = bool(
    PATCH_CONFIG.get(
        "keep_partial",
        False,
    )
)

PATCH_IMAGE_SUFFIX = PATCH_CONFIG.get(
    "image_suffix",
    ".tif",
)

PATCH_MASK_SUFFIX = PATCH_CONFIG.get(
    "mask_suffix",
    "_mask.tif",
)

# ==========================================================
# DATASET
# ==========================================================

DATASET_CONFIG = _yaml_data.get(
    "dataset",
    {},
)

IMAGE_GLOB = DATASET_CONFIG.get(
    "image_glob",
    "*.tif",
)

MASK_GLOB = DATASET_CONFIG.get(
    "mask_glob",
    "*_mask.tif",
)
# ==========================================================
# REPORT FILES
# ==========================================================

REPORT_FILES = _yaml_data.get(
    "report_files",
    {},
)

PATCH_INDEX_CSV = REPORT_DIR / REPORT_FILES.get(
    "patch_index_csv",
    "patch_index.csv",
)

PATCH_STATISTICS_CSV = REPORT_DIR / REPORT_FILES.get(
    "patch_statistics_csv",
    "patch_statistics.csv",
)

PATCH_LEVEL_CSV = REPORT_DIR / REPORT_FILES.get(
    "patch_level_csv",
    "patch_level.csv",
)

BALANCED_PATCH_CSV = REPORT_DIR / REPORT_FILES.get(
    "balanced_patch_csv",
    "balanced_patch.csv",
)

DATASET_REPORT_JSON = REPORT_DIR / REPORT_FILES.get(
    "dataset_report_json",
    "dataset_report.json",
)

PATCH_DISTRIBUTION_PNG = REPORT_DIR / REPORT_FILES.get(
    "patch_distribution_png",
    "patch_distribution.png",
)

# ==========================================================
# CREATE DIRECTORIES
# ==========================================================

PATCHES_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

REPORT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

# ==========================================================
# DATASET BALANCE
# ==========================================================

BALANCE_CONFIG = _yaml_data.get(
    "balance",
    {},
)
# ==========================================================
# BALANCE SPLITS
# ==========================================================

BALANCE_SPLITS = BALANCE_CONFIG.get(
    "apply_to",
    [
        "Train",
    ],
)
# ----------------------------------------------------------
# Thresholds
# ----------------------------------------------------------

THRESHOLDS = BALANCE_CONFIG.get(
    "thresholds",
    {},
)

BACKGROUND_THRESHOLD = float(
    THRESHOLDS.get(
        "background",
        0.00,
    )
)

VERY_SPARSE_THRESHOLD = float(
    THRESHOLDS.get(
        "very_sparse",
        0.01,
    )
)

SPARSE_THRESHOLD = float(
    THRESHOLDS.get(
        "sparse",
        0.10,
    )
)

NORMAL_THRESHOLD = float(
    THRESHOLDS.get(
        "normal",
        0.40,
    )
)

# ----------------------------------------------------------
# Keep Ratio
# ----------------------------------------------------------

KEEP_RATIO = BALANCE_CONFIG.get(
    "keep_ratio",
    {},
)

BACKGROUND_KEEP_RATIO = float(
    KEEP_RATIO.get(
        "background",
        0.10,
    )
)

VERY_SPARSE_KEEP_RATIO = float(
    KEEP_RATIO.get(
        "very_sparse",
        1.00,
    )
)

SPARSE_KEEP_RATIO = float(
    KEEP_RATIO.get(
        "sparse",
        1.00,
    )
)

NORMAL_KEEP_RATIO = float(
    KEEP_RATIO.get(
        "normal",
        1.00,
    )
)

DENSE_KEEP_RATIO = float(
    KEEP_RATIO.get(
        "dense",
        1.00,
    )
)

# ----------------------------------------------------------
# Augmentation
# ----------------------------------------------------------

AUGMENTATION = BALANCE_CONFIG.get(
    "augmentation",
    {},
)

BACKGROUND_AUG = int(
    AUGMENTATION.get(
        "background",
        4,
    )
)

VERY_SPARSE_AUG = int(
    AUGMENTATION.get(
        "very_sparse",
        6,
    )
)

SPARSE_AUG = int(
    AUGMENTATION.get(
        "sparse",
        1,
    )
)

NORMAL_AUG = int(
    AUGMENTATION.get(
        "normal",
        1,
    )
)

DENSE_AUG = int(
    AUGMENTATION.get(
        "dense",
        1,
    )
)

# ==========================================================
# TRAINING
# ==========================================================

TRAINING_CONFIG = _yaml_data.get(
    "training",
    {},
)

EPOCHS = int(
    TRAINING_CONFIG.get(
        "epochs",
        20,
    )
)

BATCH_SIZE = int(
    TRAINING_CONFIG.get(
        "batch_size",
        8,
    )
)

NUM_WORKERS = int(
    TRAINING_CONFIG.get(
        "num_workers",
        4,
    )
)

LEARNING_RATE = float(
    TRAINING_CONFIG.get(
        "learning_rate",
        1e-4,
    )
)

WEIGHT_DECAY = float(
    TRAINING_CONFIG.get(
        "weight_decay",
        1e-4,
    )
)

OPTIMIZER = TRAINING_CONFIG.get(
    "optimizer",
    "adamw",
)

SCHEDULER = TRAINING_CONFIG.get(
    "scheduler",
    "cosine",
)
LOSS_NAME = TRAINING_CONFIG.get(
    "loss",
    "cross_entropy",
)

MIN_LR = float(
    TRAINING_CONFIG.get(
        "min_lr",
        1e-6,
    )
)

WARMUP_EPOCHS = int(
    TRAINING_CONFIG.get(
        "warmup_epochs",
        5,
    )
)

USE_AMP = bool(
    TRAINING_CONFIG.get(
        "amp",
        True,
    )
)

GRAD_CLIP = float(
    TRAINING_CONFIG.get(
        "grad_clip",
        1.0,
    )
)

SEED = int(
    TRAINING_CONFIG.get(
        "seed",
        42,
    )
)

SHUFFLE = bool(
    TRAINING_CONFIG.get(
        "shuffle",
        False,
    )
)

PIN_MEMORY = bool(
    TRAINING_CONFIG.get(
        "pin_memory",
        True,
    )
)

DROP_LAST = bool(
    TRAINING_CONFIG.get(
        "drop_last",
        True,
    )
)

SAVE_BEST_ONLY = bool(
    TRAINING_CONFIG.get(
        "save_best_only",
        True,
    )
)

SAVE_INTERVAL = int(
    TRAINING_CONFIG.get(
        "save_interval",
        5,
    )
)

EARLY_STOPPING = int(
    TRAINING_CONFIG.get(
        "early_stopping",
        15,
    )
)

OUTPUTS_DIR = PROJECT_ROOT / PATHS["outputs_dir"]

CHECKPOINT_DIR = OUTPUTS_DIR / TRAINING_CONFIG.get(
    "checkpoint_dir",
    "Checkpoints",
)

LOG_DIR = OUTPUTS_DIR / TRAINING_CONFIG.get(
    "log_dir",
    "Logs",
)

USE_TENSORBOARD = bool(
    TRAINING_CONFIG.get(
        "tensorboard",
        True,
    )
)
OUTPUTS_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

CHECKPOINT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

LOG_DIR.mkdir(
    parents=True,
    exist_ok=True,
)
# ==========================================================
# MODEL
# ==========================================================
MODEL_CONFIG = _yaml_data.get(
    "model",
    {},
)

MODEL_NAME = MODEL_CONFIG.get(
    "architecture",
    "unet",
)

ENCODER_NAME = MODEL_CONFIG.get(
    "encoder_name",
    "resnet34",
)

ENCODER_WEIGHTS = MODEL_CONFIG.get(
    "encoder_weights",
    "imagenet",
)

IN_CHANNELS = int(
    MODEL_CONFIG.get(
        "in_channels",
        3,
    )
)

NUM_CLASSES = int(
    MODEL_CONFIG.get(
        "num_classes",
        2,
    )
)
