"""
evaluate.py
===========

Production-Ready Evaluation Pipeline on the Test Set.
Computes comprehensive metrics: Dice, IoU, Precision, Recall, F1-Score.
Saves quantitative reports for formal research or reporting.
"""

from __future__ import annotations

import os
import sys
import time
import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from tqdm import tqdm
import segmentation_models_pytorch as smp
from torch.amp import autocast
from torch.utils.data import DataLoader

# ==========================================================
# CẤU HÌNH ĐƯỜNG DẪN & IMPORTS
# ==========================================================
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from Scripts.config.config import (
    MODEL_NAME,
    ENCODER_NAME,
    ENCODER_WEIGHTS,
    IN_CHANNELS,
    NUM_CLASSES,
    CHECKPOINT_DIR,
    BATCH_SIZE,
    IMAGES_DIR,
    MASKS_DIR,
    OUTPUTS_DIR,
    NUM_WORKERS,
)
from Scripts.dataset.create_dataset import create_dataset

# Thư mục lưu báo cáo đánh giá
EVAL_DIR = OUTPUTS_DIR / "Evaluation"
EVAL_DIR.mkdir(parents=True, exist_ok=True)
EVAL_LOG = EVAL_DIR / "test_evaluation_report.txt"
EVAL_CSV = EVAL_DIR / "test_metrics_per_image.csv"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==========================================================
# HÀM HỖ TRỢ LOGGER
# ==========================================================


def write_eval_log(message: str, print_to_console: bool = True):
    if print_to_console:
        print(message)
    with open(EVAL_LOG, "a", encoding="utf-8") as f:
        f.write(message + "\n")


# ==========================================================
# LOAD MÔ HÌNH
# ==========================================================


def load_evaluation_model(checkpoint_name="best_model.pth"):
    print("=" * 80)
    print(f"[*] Đang tải mô hình đánh giá từ: {checkpoint_name}...")
    model_kwargs = {
        "encoder_name": ENCODER_NAME,
        "encoder_weights": None,
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
        raise FileNotFoundError(
            f"Không tìm thấy {chkpt_path}. Vui lòng kiểm tra lại quá trình huấn luyện!"
        )

    checkpoint = torch.load(chkpt_path, map_location=DEVICE, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(DEVICE)
    model.eval()

    print(
        f"[*] Tải thành công mô hình từ Epoch {checkpoint.get('epoch', 'N/A')} với Best Dice: {checkpoint.get('best_dice', 'N/A'):.4f}"
    )
    print("=" * 80)
    return model


# ==========================================================
# HÀM TÍNH TOÁN CÁC CHỈ SỐ TOÀN DIỆN (GLOBAL METRICS)
# ==========================================================


def calculate_detailed_metrics(outputs: torch.Tensor, masks: torch.Tensor):
    """
    Tính toán True Positives (TP), False Positives (FP), False Negatives (FN).
    Đảm bảo độ chính xác tuyệt đối bằng float64.
    """
    tp = torch.tensor(0.0, dtype=torch.float64, device=outputs.device)
    fp = torch.tensor(0.0, dtype=torch.float64, device=outputs.device)
    fn = torch.tensor(0.0, dtype=torch.float64, device=outputs.device)
    tn = torch.tensor(0.0, dtype=torch.float64, device=outputs.device)

    if NUM_CLASSES == 1:
        preds = (torch.sigmoid(outputs) > 0.5).float().view(-1)
        targets = masks.float().view(-1)

        tp += (preds * targets).sum(dtype=torch.float64)
        fp += (preds * (1 - targets)).sum(dtype=torch.float64)
        fn += ((1 - preds) * targets).sum(dtype=torch.float64)
        tn += ((1 - preds) * (1 - targets)).sum(dtype=torch.float64)
    else:
        preds = torch.argmax(outputs, dim=1).view(-1)
        targets = masks.long().view(-1)
        # Đánh giá trên các lớp Foreground (Bỏ qua nền 0)
        for cls in range(1, NUM_CLASSES):
            p_cls = (preds == cls).float()
            t_cls = (targets == cls).float()

            tp += (p_cls * t_cls).sum(dtype=torch.float64)
            fp += (p_cls * (1 - t_cls)).sum(dtype=torch.float64)
            fn += ((1 - p_cls) * t_cls).sum(dtype=torch.float64)
            tn += ((1 - p_cls) * (1 - t_cls)).sum(dtype=torch.float64)

    return tp.item(), fp.item(), fn.item(), tn.item()


# ==========================================================
# CHƯƠNG TRÌNH ĐÁNH GIÁ CHÍNH
# ==========================================================


def main():
    # Khởi tạo file log báo cáo
    with open(EVAL_LOG, "w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write(
            f"TEST SET EVALUATION REPORT - {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        )
        f.write("=" * 80 + "\n\n")

    model = load_evaluation_model("best_model.pth")

    # 1. Load tập Test thông qua create_dataset chuẩn của dự án
    try:
        test_dataset = create_dataset("Test")
    except Exception as e:
        write_eval_log(f"[!] Không thể load tập Test: {e}")
        return

    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        drop_last=False,
    )

    write_eval_log(
        f"[*] Tổng số mẫu trên tập Test cần đánh giá: {len(test_dataset)} patches\n"
    )

    # 2. Vòng lặp đánh giá toàn bộ tập Test
    total_tp, total_fp, total_fn, total_tn = 0.0, 0.0, 0.0, 0.0
    total_loss = 0.0

    # Loss function tương ứng cấu hình
    criterion = nn.BCEWithLogitsLoss() if NUM_CLASSES == 1 else nn.CrossEntropyLoss()

    pbar = tqdm(test_loader, desc="Đang đánh giá Test Set", dynamic_ncols=True)

    per_image_records = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(pbar, 1):
            images = batch["image"].to(DEVICE, non_blocking=True)
            masks = (
                batch["mask"].unsqueeze(1).float()
                if NUM_CLASSES == 1
                else batch["mask"].long()
            ).to(DEVICE, non_blocking=True)

            with autocast(device_type=DEVICE.type):
                outputs = model(images)
                loss = criterion(outputs, masks)

            total_loss += loss.item()
            tp, fp, fn, tn = calculate_detailed_metrics(outputs, masks)

            total_tp += tp
            total_fp += fp
            total_fn += fn
            total_tn += tn

            # Tính metric riêng lẻ từng batch/patch nếu cần lưu chi tiết
            b_dice = (2.0 * tp + 1e-7) / (2.0 * tp + fp + fn + 1e-7)
            b_iou = (tp + 1e-7) / (tp + fp + fn + 1e-7)

            patch_names = batch.get(
                "patch_name", [f"patch_{batch_idx}_{i}" for i in range(images.size(0))]
            )
            for i, name in enumerate(patch_names):
                per_image_records.append(
                    {"patch_name": name, "dice": b_dice, "iou": b_iou}
                )

    # 3. Tổng hợp và Tính toán các chỉ số khoa học cuối cùng
    avg_loss = total_loss / len(test_loader)

    # Dice & IoU toàn cục (Global Metrics)
    final_dice = (2.0 * total_tp + 1e-7) / (2.0 * total_tp + total_fp + total_fn + 1e-7)
    final_iou = (total_tp + 1e-7) / (total_tp + total_fp + total_fn + 1e-7)

    # Precision, Recall, F1-Score
    precision = (total_tp + 1e-7) / (total_tp + total_fp + 1e-7)
    recall = (total_tp + 1e-7) / (total_tp + total_fn + 1e-7)
    f1_score = (2.0 * precision * recall + 1e-7) / (precision + recall + 1e-7)
    accuracy = (total_tp + total_tn + 1e-7) / (
        total_tp + total_tn + total_fp + total_fn + 1e-7
    )

    # 4. Xuất Báo Cáo
    report_text = (
        f"KẾT QUẢ ĐÁNH GIÁ TRÊN TẬP TEST:\n"
        f"----------------------------------------\n"
        f"Model Architecture : {MODEL_NAME.upper()} ({ENCODER_NAME})\n"
        f"Total Test Samples : {len(test_dataset)}\n"
        f"Test Loss          : {avg_loss:.4f}\n"
        f"----------------------------------------\n"
        f"Global Dice Score  : {final_dice:.4f} ({final_dice*100:.2f}%)\n"
        f"Global IoU Score   : {final_iou:.4f} ({final_iou*100:.2f}%)\n"
        f"Precision          : {precision:.4f} ({precision*100:.2f}%)\n"
        f"Recall (Sensitivity): {recall:.4f} ({recall*100:.2f}%)\n"
        f"F1-Score           : {f1_score:.4f} ({f1_score*100:.2f}%)\n"
        f"Pixel Accuracy     : {accuracy:.4f} ({accuracy*100:.2f}%)\n"
        f"========================================\n"
    )

    write_eval_log(report_text)

    # Lưu chi tiết metric từng patch ra CSV để phục vụ phân tích sâu
    df_records = pd.DataFrame(per_image_records)
    df_records.to_csv(EVAL_CSV, index=False)

    print("=" * 80)
    print(f"🎉 ĐÁNH GIÁ TẬP TEST HOÀN TẤT!")
    print(f"-> Báo cáo chi tiết đã được lưu tại:\n {EVAL_LOG}")
    print(f"-> Bảng số liệu từng patch lưu tại:\n {EVAL_CSV}")
    print("=" * 80)


if __name__ == "__main__":
    main()
