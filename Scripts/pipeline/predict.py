"""
predict.py
==========

Production-Ready Inference Pipeline for Large Satellite Imagery.
Uses Sliding Window with Hann Blending to prevent border artifacts.
Maintains original Georeferencing (CRS & Transform) for QGIS compatibility.
"""

import os
import sys
import math
import time
from pathlib import Path

import cv2
import numpy as np
import rasterio
import torch
from tqdm import tqdm
import segmentation_models_pytorch as smp
from torch.amp import autocast

# ==========================================================
# CẤU HÌNH ĐƯỜNG DẪN & IMPORTS
# ==========================================================
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# Import mọi thứ trực tiếp từ config.py để đồng bộ với config.yaml
from Scripts.config.config import (
    MODEL_NAME,
    ENCODER_NAME,
    ENCODER_WEIGHTS,
    IN_CHANNELS,
    NUM_CLASSES,
    CHECKPOINT_DIR,
    BATCH_SIZE,
    IMAGES_DIR,
    OUTPUTS_DIR,
    PATCH_SIZE,
)

# Thư mục Test và Thư mục lưu kết quả tự động theo config
TEST_IMAGES_DIR = IMAGES_DIR / "Test"
OUTPUT_PREDICTS_DIR = OUTPUTS_DIR / "Predictions"
OUTPUT_PREDICTS_DIR.mkdir(parents=True, exist_ok=True)

# THÔNG SỐ TRƯỢT KHI SUY LUẬN (INFERENCE)
# Bắt buộc Overlap 50% (Patch_size // 2) để thuật toán Hann Window khử viền tốt nhất
STRIDE = PATCH_SIZE // 2

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# ==========================================================
# HÀM HỖ TRỢ TOÁN HỌC (BLENDING WINDOW)
# ==========================================================


def get_hann_window(size: int) -> np.ndarray:
    """Tạo ma trận trọng số 2D (Hann Window) để khử viền khi ghép patch."""
    win = np.hanning(size)
    win_2d = np.outer(win, win)
    return win_2d.astype(np.float32)


def preprocess_patch(patch: np.ndarray) -> torch.Tensor:
    """Chuẩn hóa ảnh về ImageNet chuẩn để đưa vào Model."""
    patch = patch.astype(np.float32) / 255.0
    patch = (patch - IMAGENET_MEAN) / IMAGENET_STD
    patch = patch.transpose(2, 0, 1)  # (H, W, C) -> (C, H, W)
    return torch.from_numpy(patch)


# ==========================================================
# LOAD MÔ HÌNH TỪ CHECKPOINT
# ==========================================================


def load_model(checkpoint_name="best_model.pth"):
    print("=" * 80)
    print(f"[*] Đang tải mô hình từ: {checkpoint_name}...")
    model_kwargs = {
        "encoder_name": ENCODER_NAME,
        "encoder_weights": None,  # Không cần tải tạ ImageNet lúc predict
        "in_channels": IN_CHANNELS,
        "classes": NUM_CLASSES,
    }

    if MODEL_NAME.lower() == "unet":
        model = smp.Unet(**model_kwargs)
    elif MODEL_NAME.lower() == "unetplusplus":
        model = smp.UnetPlusPlus(**model_kwargs)
    elif MODEL_NAME.lower() == "deeplabv3plus":
        model = smp.DeepLabV3Plus(**model_kwargs)
    else:
        raise ValueError(f"Unsupported model: {MODEL_NAME}")

    chkpt_path = CHECKPOINT_DIR / checkpoint_name
    if not chkpt_path.exists():
        raise FileNotFoundError(f"Không tìm thấy {chkpt_path}. Bạn đã train xong chưa?")

    checkpoint = torch.load(chkpt_path, map_location=DEVICE, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(DEVICE)
    model.eval()
    print(
        f"[*] Tải thành công! Best Dice lúc train: {checkpoint.get('best_dice', 'N/A')}"
    )
    print("=" * 80)
    return model


# ==========================================================
# HÀM SUY LUẬN TRÊN ẢNH LỚN (SLIDING WINDOW)
# ==========================================================


def predict_large_image(model, image_path: Path, output_path: Path):
    """Đọc ảnh lớn, trượt cắt, dự đoán, ghép lại và lưu TIFF."""
    print(f"Đang xử lý: {image_path.name}")
    start_time = time.time()

    # 1. Đọc ảnh gốc & Lấy profile hệ tọa độ
    with rasterio.open(image_path) as src:
        profile = src.profile
        img = src.read()  # (C, H, W)
        img = img.transpose(1, 2, 0)  # (H, W, C)

        if img.shape[-1] > 3:
            img = img[:, :, :3]

    H, W, _ = img.shape

    # 2. Tính toán Padding (Để khung trượt không bị lọt ra ngoài)
    pad_h = math.ceil(max(H - PATCH_SIZE, 0) / STRIDE) * STRIDE + PATCH_SIZE - H
    pad_w = math.ceil(max(W - PATCH_SIZE, 0) / STRIDE) * STRIDE + PATCH_SIZE - W

    # Dùng reflect padding để rìa ảnh không bị dải đen
    padded_img = cv2.copyMakeBorder(img, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT_101)
    padded_H, padded_W, _ = padded_img.shape

    # 3. Chuẩn bị Ma trận chứa kết quả
    if NUM_CLASSES == 1:
        pred_sum = np.zeros((padded_H, padded_W), dtype=np.float32)
    else:
        pred_sum = np.zeros((NUM_CLASSES, padded_H, padded_W), dtype=np.float32)

    weight_sum = np.zeros((padded_H, padded_W), dtype=np.float32)
    window_2d = get_hann_window(PATCH_SIZE)

    # 4. Tạo danh sách tọa độ (y, x) cần cắt
    y_coords = list(range(0, padded_H - PATCH_SIZE + 1, STRIDE))
    x_coords = list(range(0, padded_W - PATCH_SIZE + 1, STRIDE))
    coords = [(y, x) for y in y_coords for x in x_coords]
    total_patches = len(coords)

    # 5. Xử lý theo Batch (Predict tốn ít RAM hơn nên nhân đôi Batch_size)
    infer_batch_size = BATCH_SIZE * 2
    pbar = tqdm(
        total=total_patches,
        desc="Đang trượt",
        unit="patch",
        leave=False,
        dynamic_ncols=True,
    )

    for i in range(0, total_patches, infer_batch_size):
        batch_coords = coords[i : i + infer_batch_size]
        batch_patches = []

        for y, x in batch_coords:
            patch = padded_img[y : y + PATCH_SIZE, x : x + PATCH_SIZE, :]
            batch_patches.append(preprocess_patch(patch))

        batch_tensor = torch.stack(batch_patches).to(DEVICE)

        with torch.no_grad():
            with autocast(device_type=DEVICE.type):
                outputs = model(batch_tensor)
                if NUM_CLASSES == 1:
                    probs = torch.sigmoid(outputs).squeeze(1).cpu().numpy()
                else:
                    probs = torch.softmax(outputs, dim=1).cpu().numpy()

        # Ghép kết quả lại với Hann Window
        for j, (y, x) in enumerate(batch_coords):
            if NUM_CLASSES == 1:
                pred_sum[y : y + PATCH_SIZE, x : x + PATCH_SIZE] += probs[j] * window_2d
            else:
                pred_sum[:, y : y + PATCH_SIZE, x : x + PATCH_SIZE] += (
                    probs[j] * window_2d
                )
            weight_sum[y : y + PATCH_SIZE, x : x + PATCH_SIZE] += window_2d

        pbar.update(len(batch_coords))
    pbar.close()

    # 6. Chia trung bình các vùng chồng lấp và Cắt bỏ Padding
    final_pred = pred_sum / (weight_sum + 1e-7)

    if NUM_CLASSES == 1:
        final_pred = final_pred[:H, :W]
        final_mask = (final_pred > 0.5).astype(np.uint8)
    else:
        final_pred = final_pred[:, :H, :W]
        final_mask = np.argmax(final_pred, axis=0).astype(np.uint8)

    # 7. Lưu file kết quả (Giữ nguyên tọa độ GIS)
    profile.update(count=1, dtype=rasterio.uint8, compress="lzw", nodata=None)

    with rasterio.open(output_path, "w", **profile) as dst:
        dst.write(final_mask, 1)

    print(
        f"-> Đã lưu: {output_path.name} (Thời gian: {(time.time()-start_time):.1f}s)\n"
    )


# ==========================================================
# CHƯƠNG TRÌNH CHÍNH
# ==========================================================


def main():
    model = load_model("best_model.pth")

    if not TEST_IMAGES_DIR.exists():
        print(f"[!] Không tìm thấy thư mục Test tại: {TEST_IMAGES_DIR}")
        return

    image_files = list(TEST_IMAGES_DIR.glob("*.tif"))
    if not image_files:
        print(f"[!] Thư mục {TEST_IMAGES_DIR} không có ảnh .tif nào.")
        return

    print(f"[*] Tìm thấy {len(image_files)} ảnh cần xử lý.")

    for img_path in image_files:
        out_name = img_path.name.replace(".tif", "_pred.tif")
        out_path = OUTPUT_PREDICTS_DIR / out_name
        predict_large_image(model, img_path, out_path)

    print("=" * 80)
    print(f"🎉 HOÀN TẤT! Toàn bộ kết quả đã được lưu tại:\n {OUTPUT_PREDICTS_DIR}")
    print("-> Bạn có thể kéo các file _pred.tif này vào QGIS ngay bây giờ.")


if __name__ == "__main__":
    main()
