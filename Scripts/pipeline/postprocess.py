"""
postprocess.py
==============

Production-Ready GIS Post-processing Pipeline.
Converts Predicted Rasters (.tif) to Professional Vector Shapefiles (.shp).
Applies Noise Filtering (Min Area) and Polygon Regularization (Simplification).
"""

import os
import sys
import time
from pathlib import Path

import rasterio
from rasterio.features import shapes
import geopandas as gpd
from shapely.geometry import shape, MultiPolygon, Polygon
from shapely.validation import make_valid
from tqdm import tqdm

# ==========================================================
# CẤU HÌNH ĐƯỜNG DẪN
# ==========================================================
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# Đọc kết quả từ thư mục Predictions của bước trước
PREDICTS_DIR = ROOT_DIR / "Outputs" / "Predictions"

# Thư mục lưu Shapefile đầu ra
VECTORS_DIR = ROOT_DIR / "Outputs" / "Vectors"
VECTORS_DIR.mkdir(parents=True, exist_ok=True)

# ==========================================================
# CẤU HÌNH THUẬT TOÁN HẬU XỬ LÝ (POST-PROCESSING)
# ==========================================================

# 1. Khử nhiễu: Diện tích tối thiểu để được công nhận là 1 tòa nhà
# Lưu ý: Nếu hệ tọa độ (CRS) của ảnh gốc tính bằng MÉT (Ví dụ: VN-2000, UTM), số này tương đương 20 mét vuông.
# Nếu hệ tọa độ tính bằng ĐỘ (EPSG:4326), bạn phải chỉnh số này thành rất nhỏ (VD: 0.0000001)
MIN_AREA_THRESHOLD = 20.0

# 2. Làm mượt: Độ nới lỏng khi làm phẳng các đường ziczac của pixel.
# Số càng lớn đường viền càng thẳng, nhưng nếu quá lớn nhà sẽ bị biến dạng.
SIMPLIFY_TOLERANCE = 0.5

# 3. Ép vuông góc (Tùy chọn): Bật True nếu muốn tất cả các ngôi nhà biến thành hình hộp chữ nhật hoàn hảo.
# Rất hữu ích với các khu quy hoạch đô thị.
FORCE_RECTANGLE = False

# ==========================================================
# HÀM CHUYỂN ĐỔI VÀ XỬ LÝ LÕI
# ==========================================================


def process_raster_to_vector(tif_path: Path, shp_path: Path):
    """
    Chuyển Raster thành Vector, Lọc nhiễu và Làm mượt đa giác.
    """
    # 1. Đọc file dự đoán
    with rasterio.open(tif_path) as src:
        mask = src.read(1)
        transform = src.transform
        crs = src.crs

    # Nếu ảnh dự đoán hoàn toàn rỗng (không có nhà nào)
    if mask.max() == 0:
        print(f"[-] Bỏ qua {tif_path.name} vì không phát hiện tòa nhà nào.")
        return

    # 2. Chuyển đổi khối pixel thành Polygons (Raster to Vector)
    # shapes() sẽ trả về các cặp (geometry, value). Ta chỉ lấy các vùng có value = 1 (Tòa nhà).
    polygons = []
    for geom_dict, value in shapes(mask, mask=(mask == 1), transform=transform):
        if value == 1:
            geom = shape(geom_dict)
            polygons.append(geom)

    # 3. Bắt đầu bộ lọc và tinh chỉnh (Filtering & Regularization)
    processed_geoms = []

    for geom in polygons:
        # Sửa lỗi đa giác tự cắt (Self-intersection do nhiễu pixel)
        if not geom.is_valid:
            geom = make_valid(geom)

        # Nếu sau khi sửa mà nó biến thành tập hợp nhiều mảnh, tách ra
        if isinstance(geom, MultiPolygon):
            geoms_to_check = list(geom.geoms)
        else:
            geoms_to_check = [geom]

        for g in geoms_to_check:
            # BỎ QUA CÁC ĐA GIÁC QUÁ NHỎ (Nhiễu hạt tiêu)
            if g.area < MIN_AREA_THRESHOLD:
                continue

            # LÀM MƯỢT RĂNG CƯA BẰNG THUẬT TOÁN DOUGLAS-PEUCKER
            g = g.simplify(SIMPLIFY_TOLERANCE, preserve_topology=True)

            # ÉP VUÔNG VỨC (Minimum Bounding Rectangle) NẾU CẦN
            if FORCE_RECTANGLE:
                g = g.minimum_rotated_rectangle

            processed_geoms.append(g)

    if len(processed_geoms) == 0:
        print(
            f"[-] Bỏ qua {tif_path.name} vì toàn bộ các dự đoán đều là rác nhỏ hơn mức quy định."
        )
        return

    # 4. Ghi ra định dạng Shapefile cho QGIS
    # Tạo GeoDataFrame
    gdf = gpd.GeoDataFrame(geometry=processed_geoms, crs=crs)

    # Lưu thành file Shapefile
    gdf.to_file(shp_path, driver="ESRI Shapefile")

    print(
        f"[+] Hoàn tất: {shp_path.name} | Tìm thấy {len(processed_geoms)} tòa nhà hợp lệ."
    )


# ==========================================================
# CHƯƠNG TRÌNH CHÍNH
# ==========================================================


def main():
    print("=" * 80)
    print("BẮT ĐẦU CHUYỂN ĐỔI GIS & HẬU XỬ LÝ (POST-PROCESSING)")
    print("=" * 80)

    if not PREDICTS_DIR.exists():
        print(f"[!] Lỗi: Không tìm thấy thư mục {PREDICTS_DIR}")
        print("Vui lòng chạy file predict.py trước!")
        return

    # Quét tất cả các file .tif trong thư mục Predictions
    tif_files = list(PREDICTS_DIR.glob("*_pred.tif"))

    if len(tif_files) == 0:
        print(f"[!] Thư mục {PREDICTS_DIR} trống. Không có kết quả nào để chuyển đổi.")
        return

    print(f"[*] Cấu hình Lọc: Loại bỏ nhà nhỏ hơn {MIN_AREA_THRESHOLD} đơn vị vuông.")
    print(f"[*] Cấu hình Mượt: Simplify Tolerance = {SIMPLIFY_TOLERANCE}")
    print(f"[*] Ép vuông vức : {'BẬT' if FORCE_RECTANGLE else 'TẮT'}\n")

    start_time = time.time()

    for tif_path in tqdm(tif_files, desc="Đang chuyển đổi", unit="file"):
        shp_name = tif_path.name.replace("_pred.tif", "_buildings.shp")
        shp_path = VECTORS_DIR / shp_name

        process_raster_to_vector(tif_path, shp_path)

    print("=" * 80)
    print(f"🎉 XUẤT GIS HOÀN TẤT TRONG {(time.time() - start_time):.2f} GIÂY!")
    print(f"-> Thư mục chứa Shapefile: {VECTORS_DIR}")
    print(
        "-> Bạn có thể mở QGIS, ném ảnh gốc và file .shp này vào để báo cáo ngay lập tức!"
    )


if __name__ == "__main__":
    main()
