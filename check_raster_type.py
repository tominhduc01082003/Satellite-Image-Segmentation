import rasterio

with rasterio.open(
    f"E:/To_Duc/Computer_Vision/QGis/Build_CNN_Segmentation/Resource/Images/Train/img_1.tif"
) as src:
    print(src.count)
    print(src.dtypes)
