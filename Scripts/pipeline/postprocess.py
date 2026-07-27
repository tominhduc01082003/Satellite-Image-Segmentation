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

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

PREDICTS_DIR = ROOT_DIR / "Outputs" / "Predictions"

# Shapefile output
VECTORS_DIR = ROOT_DIR / "Outputs" / "Vectors"
VECTORS_DIR.mkdir(parents=True, exist_ok=True)


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

    with rasterio.open(tif_path) as src:
        mask = src.read(1)
        transform = src.transform
        crs = src.crs

    if mask.max() == 0:
        print(f"[-] Bỏ qua {tif_path.name} vì không phát hiện tòa nhà nào.")
        return

    # Chuyển đổi khối pixel thành Polygons
    # shapes() return tuple (geometry, value).Chỉ lấy các vùng có value = 1.
    polygons = []
    for geom_dict, value in shapes(mask, mask=(mask == 1), transform=transform):
        if value == 1:
            geom = shape(geom_dict)
            polygons.append(geom)

    processed_geoms = []

    for geom in polygons:
        if not geom.is_valid:
            geom = make_valid(geom)

        if isinstance(geom, MultiPolygon):
            geoms_to_check = list(geom.geoms)
        else:
            geoms_to_check = [geom]

        for g in geoms_to_check:
            if g.area < MIN_AREA_THRESHOLD:
                continue

            g = g.simplify(SIMPLIFY_TOLERANCE, preserve_topology=True)

            # ÉP VUÔNG
            if FORCE_RECTANGLE:
                g = g.minimum_rotated_rectangle

            processed_geoms.append(g)

    if len(processed_geoms) == 0:
        print(
            f"[-] Bỏ qua {tif_path.name} vì toàn bộ các dự đoán đều là rác nhỏ hơn mức quy định."
        )
        return

    gdf = gpd.GeoDataFrame(geometry=processed_geoms, crs=crs)

    gdf.to_file(shp_path, driver="ESRI Shapefile")

    print(
        f"[+] Hoàn tất: {shp_path.name} | Tìm thấy {len(processed_geoms)} tòa nhà hợp lệ."
    )


def main():
    print("=" * 80)
    print("BẮT ĐẦU CHUYỂN ĐỔI GIS & HẬU XỬ LÝ (POST-PROCESSING)")
    print("=" * 80)

    if not PREDICTS_DIR.exists():
        print(f"[!] Lỗi: Không tìm thấy thư mục {PREDICTS_DIR}")
        print("Vui lòng chạy file predict.py trước!")
        return

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
    print(f"XUẤT GIS HOÀN TẤT TRONG {(time.time() - start_time):.2f} GIÂY!")
    print(f"-> Thư mục chứa Shapefile: {VECTORS_DIR}")


if __name__ == "__main__":
    main()
