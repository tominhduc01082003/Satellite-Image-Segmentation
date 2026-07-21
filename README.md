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
.Viết full code ko cần giải thích loằng ngoằng nhiều ,tốn token quá,giải thích chỗ cần thiết thôi
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

```python
Bước 1 (quan trọng nhất)
✅ patch_generator.py
Cắt patch theo patch_size và stride.
Lưu toàn bộ patch ảnh và mask vào Resource/Patches.
Bước 2
✅ patch_statistics.py
Thống kê từng patch (tỷ lệ pixel đối tượng, nền, số đối tượng nếu cần).
Xuất patch_statistics.csv.
Bước 3
✅ dataset_balance.py
Phân loại patch theo mức độ chứa đối tượng.
Sinh dataset_report.json và biểu đồ phân bố.
Bước 4
✅ augmentation/
Chỉ tăng cường cho các nhóm patch thiếu (ví dụ ít patch chứa nhiều đối tượng).
Bước 5
✅ Cập nhật create_dataset.py
Đọc dữ liệu từ thư mục Resource/Patches thay vì ảnh gốc.
Bước 6
✅ Viết lại train.py
Loại bỏ RandomGeoSampler, dùng DataLoader đọc trực tiếp các patch đã được tạo và cân bằng.
```

### patch_generator.py :

```python
Phần 1
Import
Đọc config
Các hàm tiện ích
Hàm cắt patch
Hàm lưu patch
Phần 2
Hàm xử lý một cặp ảnh–mask
Ghi patch_index.csv
Kiểm tra lỗi
Progress bar
Phần 3
Hàm generate_patches()
main()
Chạy cho Train / Val / Test
```

### patch_statistics.py :

```python
Phần 1

Import
Đọc config
Utility functions
Hàm đọc mask
Hàm tính pixel statistics

Phần 2

Hàm đếm object (Connected Components)
Hàm xử lý một patch
Ghi một dòng vào CSV

Phần 3

Thống kê toàn bộ Train/Val/Test
Sinh patch_statistics.csv
main()
```

### dataset_balance.py

```python
Mình sẽ chia module này thành 3 phần
Phần 1
Import
Đọc CSV
Phân loại patch
Utility
Phần 2
Sinh balanced_patch.csv
Sinh dataset_report.json
Phần 3
Vẽ biểu đồ
main()
```

### policies.py

```python
Phần 1
Import
Đọc config
Constants
get_sampling_weight()
get_base_transform()

≈ 130–180 dòng

Phần 2
get_train_transform()
get_validation_transform()
get_test_transform()
get_policy()

≈ 100–150 dòng

Phần 3
print_policies()
main()
Test toàn bộ policy

≈ 50–80 dòng
```

### visualize.py

```python
Phần 1 (~150 dòng)
Import
Đọc config
read_patch()
draw_single_bbox_with_coords()
show_result()
Phần 2 (~150 dòng)
build_test_suite() (tự lấy toàn bộ transform từ policies.py)
visualize_single_augmentation()
visualize_augmentations()
Phần 3 (~50 dòng)
main()
In tên augment đang kiểm tra
Hiển thị:
Ảnh gốc
Mask gốc
Ảnh sau augment (BBox + tọa độ mới)
Mask sau augment
Đóng cửa sổ tự chuyển sang augment tiếp theo
```

### create_dataset.py

```python
Phần 1 (~130 dòng): Import + PatchDataset.__init__() + đọc balanced_patch.csv.
Phần 2 (~130 dòng): __len__(), __getitem__(), đọc TIFF, áp dụng get_policy().
Phần 3 (~80 dòng): create_dataset(), print_dataset_info(), main().
```

### train.py

```python
Phần 1: Import, Config, Seed, Logging, Metrics, Checkpoint.
Phần 2: Dataset, Sampler, Model, Loss, Optimizer, Scheduler, AMP.
Phần 3: train_one_epoch() và validate_one_epoch() (Dice/IoU tính toàn epoch).
Phần 4: Resume, EarlyStopping, CSV/TXT Logging, main().
```
