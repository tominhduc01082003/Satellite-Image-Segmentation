# Satellite Image Segmentation

Semantic segmentation framework for satellite imagery using deep learning and GIS preprocessing. This project provides a complete pipeline from QGIS vector annotations to model training, prediction, and evaluation.

---

# Overview

This project is designed to build a complete semantic segmentation workflow for satellite images.

The pipeline combines Geographic Information System (GIS) preprocessing with deep learning to transform manually annotated vector labels into raster masks, generate training patches, perform dataset analysis and balancing, apply data augmentation, and finally train and evaluate convolutional neural network (CNN) segmentation models.

The project supports the entire workflow, including:

- GIS data validation
- Vector geometry checking
- Overlap detection and correction
- Rasterization
- Patch generation
- Dataset statistics
- Dataset balancing
- Data augmentation
- PyTorch dataset construction
- Model training
- Prediction
- Model evaluation

---

# Features

- Convert QGIS vector labels (`.shp`) into raster masks (`.tif`)
- Validate invalid geometries automatically
- Detect and fix overlapping polygons
- Generate image and mask patches
- Analyze foreground/background distribution
- Balance datasets before training
- Support configurable data augmentation
- Build PyTorch datasets
- Train CNN segmentation models
- Predict segmentation masks
- Evaluate model performance using common segmentation metrics

---

# Table of Contents

- [Overview](#overview)
- [Structure Project](#structure-project)
- [Pipeline](#pipeline)
- [Installation](#installation)
- [Dataset Preparation](#dataset-preparation)
- [Data Analysis](#data-analysis)
- [Training](#training)
- [Inference](#inference)
- [Evaluation](#evaluation)
- [Results](#results)
- [Contact](#contact)

---

# Structure Project

```python
BUILD_CNN_SEGMENTATION/
├── .venv/                              # Môi trường ảo Python
│
├── Resource/                           # Toàn bộ dữ liệu của dự án
│   ├── Images/                         # Ảnh vệ tinh gốc (.tif)
│   │   ├── Train/
│   │   │   ├── img_1.tif
│   │   │   ├── img_2.tif
│   │   │   └── ...
│   │   ├── Val/
│   │   └── Test/
│   │
│   ├── Labels/                         # Nhãn Vector được tạo bằng QGIS
│   │   ├── Train/
│   │   │   ├── img_1.shp
│   │   │   ├── img_1.dbf
│   │   │   ├── img_1.prj
│   │   │   ├── img_1.shx
│   │   │   └── ...
│   │   ├── Val/
│   │   └── Test/
│   │
│   ├── Masks/                          # Ground Truth Mask sau khi Rasterize
│   │   ├── Train/
│   │   │   ├── img_1_mask.tif
│   │   │   ├── img_2_mask.tif
│   │   │   └── ...
│   │   ├── Val/
│   │   └── Test/
│   │
│   ├── Patches/                        # Các mảnh ảnh cắt nhỏ để train model
│   │   ├── Train/
│   │   │   ├── Images/
│   │   │   │   ├── img_1_patch_0001.tif
│   │   │   │   ├── img_1_patch_0002.tif
│   │   │   │   └── ...
│   │   │   └── Masks/
│   │   │       ├── img_1_patch_0001_mask.tif
│   │   │       ├── img_1_patch_0002_mask.tif
│   │   │       └── ...
│   │   │
│   │   ├── Val/
│   │   │   ├── Images/
│   │   │   └── Masks/
│   │   └── Test/
│   │       ├── Images/
│   │       └── Masks/
│   │
│   └── pro.qgz                         # Project QGIS
│
├── Scripts/                            # Thư mục mã nguồn chính
│   ├── analysis/                       # Thống kê & Phân tích dữ liệu
│   │   ├── __init__.py
│   │   ├── dataset_balance.py
│   │   └── patch_statistics.py
│   │
│   ├── augmentation/                   # Tăng cường dữ liệu
│   │   ├── __init__.py
│   │   ├── visualize.py
│   │   └── policies.py
│   │
│   ├── config/                         # Cấu hình hệ thống
│   │   ├── config.py
│   │   └── config.yaml
│   │
│   ├── dataset/                        # PyTorch Dataset
│   │   ├── __init__.py
│   │   └── create_dataset.py
│   │
│   ├── pipeline/                       # Luồng chạy huấn luyện, dự đoán, đánh giá
│   │   ├── __init__.py
│   │   ├── evaluate.py
│   │   ├── predict.py
│   │   └── train.py
│   │
│   └── preprocessing/                  # Tiền xử lý hình học & GIS
│       ├── __init__.py
│       ├── check_geometry.py
│       ├── check_overlap.py
│       ├── fix_overlap.py
│       ├── rasterize.py
│       └── patch_generator.py
│
├── Models/                             # Lưu checkpoint mô hình (.pth)
├── Outputs/                            # Kết quả dự đoán và logs
├── Report/                             # Kết quả thống kê và biểu đồ phân tích
│   ├── dataset_report.json
│   ├── patch_distribution.png
│   ├── patch_index.csv
│   ├── patch_statistics.csv
│   ├── patch_level.csv
│   └── balanced_patch.csv
│
├── .gitignore
├── README.md
└── requirements.txt
```

---

# Pipeline

```python
┌──────────────────── PREPROCESSING ────────────────────┐
Label → Check Geometry → Check Overlap → Fix Overlap
                                          │
                                          ▼
                                     Rasterize
└───────────────────────────────────────────────────────┘

┌────────────────── DATA PREPARATION ───────────────────┐
Rasterize → Patch Generator → Patch Statistics → Dataset Balance
                                                   │
                                                   ▼
                                            Augmentation
                                                   │
                                                   ▼
                                            Create Dataset
└───────────────────────────────────────────────────────┘

┌──────────── TRAINING & EVALUATION ────────────────────┐
Create Dataset → Train → Predict → Evaluate
└───────────────────────────────────────────────────────┘
```

---

# Installation

## Clone Repository

```bash
git clone https://github.com/tominhduc01082003/Satellite-Image-Segmentation
# put pro in BUILD_CNN_SEGMENTATION /
cd BUILD_CNN_SEGMENTATION
```

## Create Virtual Environment

### Windows

```bash
python -m venv .venv
# requirement python version 3.12.8
.venv\Scripts\activate
```

### Linux / macOS

```bash
python3 -m venv .venv

source .venv/bin/activate
```

## Install Dependencies

```bash
pip install -r requirements.txt
```

---

# Dataset Preparation

## 1. Check Geometry

```bash
python -m Scripts.preprocessing.check_geometry
```

Validate all vector geometries and repair invalid polygons.

---

## 2. Check Overlap

```bash
python -m Scripts.preprocessing.check_overlap
```

Detect overlapping polygons inside each annotation layer.

---

## 3. Fix Overlap

```bash
python -m Scripts.preprocessing.fix_overlap
```

Automatically remove or repair overlapping regions.

---

## 4. Rasterize

```bash
python -m Scripts.preprocessing.rasterize
```

Convert QGIS shapefiles into raster masks.

---

## 5. Generate Patches

```bash
python -m Scripts.preprocessing.patch_generator
```

Slice large satellite images into training patches.

---

# Data Analysis

## Patch Statistics

```bash
python -m Scripts.analysis.patch_statistics
```

Generate statistics for every generated patch.

---

## Dataset Balance

```bash
python -m Scripts.analysis.dataset_balance
```

Analyze foreground/background ratios and remove unnecessary empty patches.

---

## Data Augmentation

Visualize augmentation policies before training.

```bash
python -m Scripts.augmentation.visualize
```

---

# Training

Configure hyperparameters inside

```text
Scripts/config/config.py
```

and

```text
Scripts/config/config.yaml
```

Create PyTorch Dataset

```bash
python -m Scripts.dataset.create_dataset
```

Train Model

```bash
python -m Scripts.pipeline.train
```

Model checkpoints are automatically saved inside

```text
Models/
```

---

# Inference

Generate prediction masks.

```bash
python -m Scripts.pipeline.predict
```

Prediction outputs are saved into

```text
Outputs/
```

---

# Evaluation

Evaluate segmentation performance.

```bash
python -m Scripts.pipeline.evaluate
```

Typical metrics include

- Dice Score
- mIoU
- Precision
- Recall
- Accuracy

---

# Results

Generated reports include

- Dataset statistics
- Patch distribution
- Dataset balance
- Prediction outputs
- Evaluation metrics

The following results were obtained on the **Test** dataset.

| Metric         |           Value |
| -------------- | --------------: |
| Model          | UNet (ResNet34) |
| Test Samples   |              49 |
| Test Loss      |          0.2822 |
| Dice Score     |      **85.00%** |
| IoU            |      **73.91%** |
| Precision      |          78.22% |
| Recall         |          93.08% |
| F1-Score       |          85.00% |
| Pixel Accuracy |          94.10% |

- Reports are automatically saved inside

```text
Report/

```

---

# Contact

- To Minh Duc (ducto020803@gmail.com)
