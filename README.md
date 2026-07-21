### Structure project

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

```python
Pipeline :
Label

↓

Check Geometry

↓

Check Overlap

↓

Fix Overlap

↓

Rasterize

↓

Patch Generator

↓

Patch Statistics

↓

Dataset Balance

↓

Augmentation

↓

Create Dataset

↓

Train

↓

Predict

↓

Evaluate
```
