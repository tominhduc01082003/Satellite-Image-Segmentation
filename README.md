### Structure project

```python
BUILD_CNN_SEGMENTATION/
│
├── .venv/                          # Môi trường ảo Python
│
├── Resource/                       # Toàn bộ dữ liệu của dự án
│   │
│   ├── Images/                     # Ảnh vệ tinh gốc (.tif)
│   │   ├── Train/
│   │   │   ├── img_1.tif
│   │   │   ├── img_2.tif
│   │   │   └── ...
│   │   ├── Val/
│   │   └── Test/
│   │
│   ├── Labels/                     # Nhãn Vector được tạo bằng QGIS
│   │   ├── Train/
│   │   │   ├── img_1.shp
│   │   │   ├── img_1.dbf
│   │   │   ├── img_1.prj
│   │   │   ├── img_1.shx
│   │   │   └── ...
│   │   ├── Val/
│   │   └── Test/
│   │
│   ├── Masks/                      # Ground Truth Mask sau khi Rasterize
│   │   ├── Train/
│   │   │   ├── img_1_mask.tif
│   │   │   ├── img_2_mask.tif
│   │   │   └── ...
│   │   ├── Val/
│   │   └── Test/
│   │
│   └── pro.qgz                     # Project QGIS
│
├── Scripts/                        # Các script tiền xử lý dữ liệu
│   ├── Check_Geometry.py
│   ├── Check_Overlap.py
│   ├── Fix_Overlap.py
│   ├── Rasterize.py
│   ├── CreateDataset.py
│   ├── Train.py
│   ├── Predict.py
│   └── Evaluate.py
|
├── README.md
├── requirements.txt
└── .gitignore
```
