import sys
from pathlib import Path
import geopandas as gpd
from shapely import set_precision
from shapely.validation import make_valid

# THIẾT LẬP IMPORT TỪ THƯ MỤC GỐC

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
from Scripts.config import config

# Geometry Repair


def repair_geometry(geom):
    if geom is None:
        return geom
    try:
        geom = make_valid(geom)
    except Exception:
        pass

    try:
        geom = geom.buffer(0)
    except Exception:
        pass

    try:
        geom = set_precision(
            geom,
            grid_size=config.GRID_SIZE,
        )
    except Exception:
        pass

    return geom


# Count overlap


def count_overlaps(gdf):
    overlaps = []
    n = len(gdf)

    for i in range(n):
        geom_i = gdf.loc[i, "geometry"]
        for j in range(i + 1, n):
            geom_j = gdf.loc[j, "geometry"]

            if not geom_i.intersects(geom_j):
                continue

            inter = geom_i.intersection(geom_j)

            if inter.is_empty:
                continue

            area = inter.area
            if area <= config.MIN_AREA:
                continue

            overlaps.append((i, j, area))

    return overlaps


# Fix overlap


def fix_overlaps(gdf):
    gdf = gdf.copy()
    print("\n=== STEP 1 : Repair Geometries ===")

    for idx in range(len(gdf)):
        gdf.at[idx, "geometry"] = repair_geometry(gdf.loc[idx, "geometry"])

    print("Done.")
    print("\n=== STEP 2 : Remove Overlaps ===")

    iteration = 0
    # Đã dùng cấu hình MAX_ITER
    while iteration < config.MAX_ITER:
        iteration += 1
        overlaps = count_overlaps(gdf)

        if len(overlaps) == 0:
            print("\nNo overlap remaining.")
            break

        print(f"\nIteration {iteration} | overlaps = {len(overlaps)}")
        fixed_this_round = 0

        for i, j, area in overlaps:
            print(f"Fix {i} ↔ {j} | area = {area:.30f}")

            geom_i = gdf.loc[i, "geometry"]
            geom_j = gdf.loc[j, "geometry"]

            # cắt phần overlap khỏi feature j
            new_geom_j = geom_j.difference(geom_i)
            new_geom_j = repair_geometry(new_geom_j)

            gdf.at[j, "geometry"] = new_geom_j
            fixed_this_round += 1

        if fixed_this_round == 0:
            break

    print("\n=== FINAL CHECK ===")
    overlaps = count_overlaps(gdf)

    if len(overlaps) == 0:
        print("SUCCESS")
        print("No overlap detected.")
    else:
        print(f"Still found {len(overlaps)} overlap(s)")
        for i, j, area in overlaps:
            print(f"{i} ↔ {j} | area = {area:.30f}")

    return gdf


# Main
if __name__ == "__main__":
    target_shp = config.DEFAULT_CHECK_SHP_PATH

    print("=" * 80)
    print("INPUT :", target_shp.name)
    print("=" * 80)

    gdf = gpd.read_file(target_shp)
    fixed = fix_overlaps(gdf)

    fixed.to_file(target_shp)

    print("\nSaved:")
    print(target_shp)
