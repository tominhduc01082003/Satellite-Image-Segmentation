from pathlib import Path
import geopandas as gpd
from shapely import set_precision
import sys

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
from Scripts.config.config import MIN_AREA, GRID_SIZE
from Scripts.config import config


def check_overlap(shp_path):

    gdf = gpd.read_file(shp_path)

    # Ép lưới tọa độ để triệt tiêu sai số đọc/ghi file Shapefile
    # gdf["geometry"] = gdf["geometry"].apply(
    #     lambda x: set_precision(x, GRID_SIZE) if x is not None else None
    # )

    print("=" * 80)
    print("FILE:", shp_path.name)
    print("TOTAL:", len(gdf))
    print("=" * 80)

    count = 0

    for i in range(len(gdf)):

        gi = gdf.iloc[i].geometry
        if gi is None:
            continue

        for j in range(i + 1, len(gdf)):

            gj = gdf.iloc[j].geometry
            if gj is None:
                continue

            if not gi.intersects(gj):
                continue

            inter = gi.intersection(gj)

            if inter.is_empty:
                continue

            if inter.area <= MIN_AREA:
                continue

            count += 1

            print("-" * 60)
            print(f"{i} ↔ {j}")
            print(f"area = {inter.area:.12f}")

    print("=" * 80)

    if count == 0:
        print("OK - No overlap")
    else:
        print("OVERLAP:", count)


if __name__ == "__main__":

    sample = config.DEFAULT_CHECK_SHP_PATH
    check_overlap(sample)
