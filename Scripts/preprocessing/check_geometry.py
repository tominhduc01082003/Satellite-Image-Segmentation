import sys
from pathlib import Path
import geopandas as gpd
from shapely.validation import explain_validity

ROOT_DIR = (
    Path(__file__).resolve().parent.parent.parent
)  # trả về thư mục gốc Build_CNN_Segmentation
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))  # thêm vô biến môi trường
from Scripts.config import config


def check_shapefile(shp_path):
    if not shp_path.exists():
        print(f"[LỖI] Không tìm thấy đường dẫn: {shp_path}")
        return

    gdf = gpd.read_file(shp_path)

    print("-" * 80)
    print(f"File: {shp_path.name}")
    print(f"Total features: {len(gdf)}")
    print("-" * 80)

    invalid_count = 0
    for idx, row in gdf.iterrows():
        geom = row["geometry"]
        is_valid = geom.is_valid

        print("\n" + "-" * 60)
        print(f"Feature id: {idx}")
        print(f"Valid: {is_valid}")
        print(f"Geometry type: {geom.geom_type}")

        multipart = geom.geom_type.startswith("Multi")
        print(f"Multipart: {multipart}")

        if is_valid:
            print("Reason: OK")
        else:
            invalid_count += 1
            print(f"Reason: {explain_validity(geom)}")

    print("\n" + "-" * 80)
    print(f"Invalid count: {invalid_count}")
    print("-" * 80)


if __name__ == "__main__":

    shp_path = config.DEFAULT_CHECK_SHP_PATH
    print(f"Đang kiểm tra hình học tại: {shp_path}")
    check_shapefile(shp_path)
