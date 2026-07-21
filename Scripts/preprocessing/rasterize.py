import sys
from pathlib import Path
import geopandas as gpd
import numpy as np
import rasterio
from rasterio.features import rasterize
from tqdm import tqdm

# ==========================================================
# THIẾT LẬP IMPORT TỪ THƯ MỤC GỐC
# ==========================================================
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# Import cấu hình tập trung
from Scripts.config import config

# ==========================================================
# Rasterize One Image
# ==========================================================


def rasterize_one(image_path: Path, shp_path: Path, output_path: Path):
    print(f"\nProcessing: {image_path.name}")

    # Read raster information
    with rasterio.open(image_path) as src:
        transform = src.transform
        crs = src.crs
        width = src.width
        height = src.height
        profile = src.profile.copy()

    # Read shapefile
    gdf = gpd.read_file(shp_path)

    if gdf.empty:
        raise RuntimeError(f"{shp_path.name} has no polygons.")

    # CRS check
    if gdf.crs != crs:
        print("CRS mismatch -> Reprojecting...")
        gdf = gdf.to_crs(crs)

    # Geometry list
    shapes = []
    for _, row in gdf.iterrows():
        geometry = row.geometry

        if geometry is None:
            continue
        if geometry.is_empty:
            continue

        # Lấy tên cột chứa ID từ file cấu hình (thay vì fix cứng "class_id")
        class_id = int(row[config.CLASS_COL])
        shapes.append((geometry, class_id))

    # Rasterize
    mask = rasterize(
        shapes=shapes,
        out_shape=(height, width),
        fill=0,
        transform=transform,
        dtype=np.uint8,
    )

    # Save
    # Trong qgis nodata = 0 (để làm trong suốt các viền đen background có giá trị 0)
    # khi copy profile từ ảnh gốc src.profile.copy()
    # thì vô tình copy luôn nodata = 0 nên khi update để nodata=None
    profile.update(count=1, dtype=rasterio.uint8, compress="lzw", nodata=None)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with rasterio.open(output_path, "w", **profile) as dst:
        dst.write(mask, 1)

    unique = np.unique(mask)

    print(f"Saved -> {output_path.name}")
    print(f"Mask Values : {unique}")


# ==========================================================
# Main Process Loop
# ==========================================================


def process_split(split):
    # Lấy đường dẫn động từ config kết hợp với split hiện tại
    image_dir = config.IMAGES_DIR / split
    label_dir = config.LABELS_DIR / split
    mask_dir = config.MASKS_DIR / split

    tif_files = sorted(image_dir.glob("*.tif"))

    if len(tif_files) == 0:
        print(f"\nNo images found in {image_dir}")
        return

    print("=" * 60)
    print(f"SPLIT: {split}")
    print("=" * 60)

    for image_path in tqdm(tif_files):
        stem = image_path.stem
        shp_path = label_dir / f"{stem}.shp"

        if not shp_path.exists():
            print(f"Missing Label : {shp_path.name}")
            continue

        output_path = mask_dir / f"{stem}{config.MASK_SUFFIX}.tif"

        rasterize_one(image_path=image_path, shp_path=shp_path, output_path=output_path)


# ==========================================================
# Entry Point
# ==========================================================

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("Rasterize Building Labels")
    print("=" * 60)

    # Duyệt qua các thư mục (Train, Val, Test) lấy từ file cấu hình
    for split in config.SPLITS:
        process_split(split)

    print("\nFinished.")
