"""
train.py
=========

Production-Ready Training pipeline for patch-based building segmentation.
Includes: AverageMeter, TQDM, GPU Peak Memory, NaN Check, Grad Norm, True Resume, and Auto-Plotting.
"""

from __future__ import annotations

import os
import sys
import time
import random
import shutil
import datetime
import platform
import warnings
from pathlib import Path

# TẮT TOÀN BỘ CẢNH BÁO ĐỂ KHÔNG LÀM HỎNG THANH TQDM
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from tqdm import tqdm
import segmentation_models_pytorch as smp

from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.optim import AdamW, Adam, SGD
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    StepLR,
    ReduceLROnPlateau,
    OneCycleLR,
)
from torch.amp import GradScaler, autocast

# ==========================================================
# CẤU HÌNH ĐƯỜNG DẪN & IMPORTS
# ==========================================================

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from Scripts.config.config import (
    YAML_PATH,
    SEED,
    BATCH_SIZE,
    NUM_WORKERS,
    PIN_MEMORY,
    DROP_LAST,
    EPOCHS,
    LEARNING_RATE,
    WEIGHT_DECAY,
    MIN_LR,
    USE_AMP,
    GRAD_CLIP,
    LOSS_NAME,
    OPTIMIZER,
    SCHEDULER,
    EARLY_STOPPING,
    SAVE_INTERVAL,
    MODEL_NAME,
    ENCODER_NAME,
    ENCODER_WEIGHTS,
    IN_CHANNELS,
    NUM_CLASSES,
    LOG_DIR,
    CHECKPOINT_DIR,
)

from Scripts.dataset.create_dataset import create_dataset
from Scripts.augmentation.policies import get_sampling_weight

OUTPUTS_DIR = LOG_DIR.parent
VIS_DIR = OUTPUTS_DIR / "Visualizations"
LOG_DIR.mkdir(parents=True, exist_ok=True)
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
VIS_DIR.mkdir(parents=True, exist_ok=True)

LOG_FILE = LOG_DIR / "training_log.txt"
CSV_HISTORY = LOG_DIR / "training_history.csv"

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
IMAGENET_STD = np.array([0.229, 0.224, 0.225])

# ==========================================================
# CÔNG CỤ HỖ TRỢ (UTILITIES & METERS)
# ==========================================================


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def reset_log():
    with open(LOG_FILE, "w", encoding="utf-8") as f:
        f.write("=" * 80 + "\nPRODUCTION TRAINING LOG\n" + "=" * 80 + "\n")


def write_log(message: str):
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(message + "\n")


class AverageMeter:
    """Computes and stores the average and current value."""

    def __init__(self, name: str, fmt: str = ":f"):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val, self.avg, self.sum, self.count = 0.0, 0.0, 0.0, 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def get_peak_gpu_mem() -> str:
    if torch.cuda.is_available():
        peak = torch.cuda.max_memory_allocated() / (1024**3)
        return f"{peak:.1f}G"
    return "N/A"


def count_parameters(model):
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    return trainable, frozen


# ==========================================================
# LOSS & METRICS
# ==========================================================


class BCEDiceLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = smp.losses.DiceLoss(mode="binary", from_logits=True)

    def forward(self, p, t):
        return self.bce(p, t) + self.dice(p, t)


class MultiClassCEDiceLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.ce = nn.CrossEntropyLoss()
        self.dice = smp.losses.DiceLoss(mode="multiclass", from_logits=True)

    def forward(self, p, t):
        return self.ce(p, t) + self.dice(p, t)


def calculate_metrics(outputs: torch.Tensor, masks: torch.Tensor):
    """Tính bằng torch.float64 để đảm bảo Macro/Micro độ chính xác cao nhất"""
    inter = torch.tensor(0.0, dtype=torch.float64, device=outputs.device)
    pred_a = torch.tensor(0.0, dtype=torch.float64, device=outputs.device)
    tgt_a = torch.tensor(0.0, dtype=torch.float64, device=outputs.device)

    if NUM_CLASSES == 1:
        preds = (torch.sigmoid(outputs) > 0.5).float().view(-1)
        targets = masks.float().view(-1)
        inter += (preds * targets).sum(dtype=torch.float64)
        pred_a += preds.sum(dtype=torch.float64)
        tgt_a += targets.sum(dtype=torch.float64)
    else:
        preds = torch.argmax(outputs, dim=1).view(-1)
        targets = masks.long().view(-1)
        for cls in range(1, NUM_CLASSES):
            p_cls = (preds == cls).float()
            t_cls = (targets == cls).float()
            inter += (p_cls * t_cls).sum(dtype=torch.float64)
            pred_a += p_cls.sum(dtype=torch.float64)
            tgt_a += t_cls.sum(dtype=torch.float64)

    return inter.item(), pred_a.item(), tgt_a.item()


# ==========================================================
# VISUALIZATION & PLOTTING
# ==========================================================


def save_validation_visualization(images, masks, outputs, epoch):
    img = images[0].cpu().numpy().transpose(1, 2, 0)
    img = np.clip(IMAGENET_STD * img + IMAGENET_MEAN, 0, 1)

    gt = masks[0].cpu().numpy()
    if NUM_CLASSES == 1:
        pred = (torch.sigmoid(outputs[0]) > 0.5).cpu().numpy().squeeze(0)
        gt = gt.squeeze(0)
    else:
        pred = torch.argmax(outputs[0], dim=0).cpu().numpy()

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(img)
    axes[0].set_title("Input Image")
    axes[1].imshow(gt, cmap="gray")
    axes[1].set_title("Ground Truth")
    axes[2].imshow(pred, cmap="gray")
    axes[2].set_title(f"Prediction (Epoch {epoch})")

    for ax in axes:
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(VIS_DIR / f"val_epoch_{epoch:03d}.png")
    plt.close()


def plot_training_curves():
    if not CSV_HISTORY.exists():
        return
    df = pd.read_csv(CSV_HISTORY)
    plt.figure(figsize=(18, 5))

    plt.subplot(1, 3, 1)
    plt.plot(
        df["epoch"], df["train_loss"], label="Train Loss", marker="o", markersize=3
    )
    plt.plot(df["epoch"], df["val_loss"], label="Val Loss", marker="o", markersize=3)
    plt.title("Loss Curve")
    plt.xlabel("Epoch")
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 3, 2)
    plt.plot(
        df["epoch"], df["train_dice"], label="Train Dice", marker="o", markersize=3
    )
    plt.plot(df["epoch"], df["val_dice"], label="Val Dice", marker="o", markersize=3)
    plt.title("Dice Curve")
    plt.xlabel("Epoch")
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 3, 3)
    plt.plot(df["epoch"], df["lr"], label="Learning Rate", color="purple")
    plt.title("LR Schedule")
    plt.xlabel("Epoch")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(OUTPUTS_DIR / "training_curves.png")
    plt.close()


# ==========================================================
# MAIN LOOP
# ==========================================================


def main():
    reset_log()
    seed_everything(SEED)
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(
        f"[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] KHỞI TẠO HỆ THỐNG TRÊN {DEVICE.type.upper()}"
    )
    write_log(
        f"[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] KHỞI TẠO HỆ THỐNG TRÊN {DEVICE.type.upper()}"
    )

    if YAML_PATH.exists():
        shutil.copy(YAML_PATH, OUTPUTS_DIR / "config_backup.yaml")

    train_dataset = create_dataset("Train")
    val_dataset = create_dataset("Val")

    train_levels = train_dataset.data["level"].tolist()
    train_weights = [get_sampling_weight(lvl) for lvl in train_levels]
    sampler = WeightedRandomSampler(
        weights=train_weights, num_samples=len(train_weights), replacement=True
    )

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=BATCH_SIZE,
        sampler=sampler,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        drop_last=DROP_LAST,
        persistent_workers=True if NUM_WORKERS > 0 else False,
    )
    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        drop_last=False,
        persistent_workers=True if NUM_WORKERS > 0 else False,
    )

    model_kwargs = {
        "encoder_name": ENCODER_NAME,
        "encoder_weights": ENCODER_WEIGHTS,
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

    model = model.to(DEVICE)
    trainable_p, frozen_p = count_parameters(model)
    write_log(f"Model: {MODEL_NAME.upper()} | Encoder: {ENCODER_NAME}")
    write_log(
        f"Params: {trainable_p/1e6:.2f}M Trainable | {frozen_p/1e6:.2f}M Frozen | {(trainable_p+frozen_p)/1e6:.2f}M Total"
    )

    loss_name = LOSS_NAME.lower()
    if NUM_CLASSES == 1:
        if loss_name == "dice":
            criterion = smp.losses.DiceLoss(mode="binary", from_logits=True)
        elif loss_name == "bce_dice":
            criterion = BCEDiceLoss()
        else:
            criterion = nn.BCEWithLogitsLoss()
    else:
        if loss_name == "ce_dice":
            criterion = MultiClassCEDiceLoss()
        else:
            criterion = nn.CrossEntropyLoss()

    opt_name = OPTIMIZER.lower()
    if opt_name == "adam":
        optimizer = Adam(
            model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
        )
    elif opt_name == "adamw":
        optimizer = AdamW(
            model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
        )
    elif opt_name == "sgd":
        optimizer = SGD(
            model.parameters(),
            lr=LEARNING_RATE,
            momentum=0.9,
            weight_decay=WEIGHT_DECAY,
        )

    sched_name = SCHEDULER.lower()
    if sched_name == "cosine":
        scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=MIN_LR)
    elif sched_name == "plateau":
        scheduler = ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=5)
    else:
        scheduler = None

    scaler = GradScaler(device=DEVICE.type, enabled=(USE_AMP and DEVICE.type == "cuda"))

    # ==========================================================
    # RESUME & RANDOM STATE LOGIC
    # ==========================================================
    start_epoch, best_dice, best_loss, best_epoch = 1, -1.0, float("inf"), 1
    patience = 0
    resume_file = CHECKPOINT_DIR / "last_model.pth"

    if resume_file.exists():
        try:
            chkpt = torch.load(resume_file, map_location=DEVICE)
            model.load_state_dict(chkpt["model_state_dict"])
            optimizer.load_state_dict(chkpt["optimizer_state_dict"])
            if scheduler and chkpt.get("scheduler_state_dict"):
                scheduler.load_state_dict(chkpt["scheduler_state_dict"])
            if scaler and chkpt.get("scaler_state_dict"):
                scaler.load_state_dict(chkpt["scaler_state_dict"])

            if "random_state" in chkpt:
                random.setstate(chkpt["random_state"])
            if "np_random_state" in chkpt:
                np.random.set_state(chkpt["np_random_state"])
            if "torch_rng_state" in chkpt:
                torch.set_rng_state(chkpt["torch_rng_state"])
            if "cuda_rng_state" in chkpt and torch.cuda.is_available():
                torch.cuda.set_rng_state_all(chkpt["cuda_rng_state"])

            start_epoch = chkpt.get("epoch", 0) + 1
            best_dice = chkpt.get("best_dice", -1.0)
            best_loss = chkpt.get("best_loss", float("inf"))
            best_epoch = start_epoch - 1
            print(
                f"[*] Phục hồi an toàn từ Checkpoint (Epoch {start_epoch-1}). Best Dice: {best_dice:.4f}"
            )
        except Exception as e:
            print(f"[!] Checkpoint bị hỏng ({e}). Bắt đầu huấn luyện từ đầu!")

    if start_epoch == 1 and CSV_HISTORY.exists():
        CSV_HISTORY.unlink()

    # ==========================================================
    # CLOSURES
    # ==========================================================
    def save_checkpoint(epoch_num, b_dice, b_loss, filename):
        checkpoint = {
            "epoch": epoch_num,
            "best_dice": b_dice,
            "best_loss": b_loss,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
            "scaler_state_dict": scaler.state_dict() if scaler else None,
            "random_state": random.getstate(),
            "np_random_state": np.random.get_state(),
            "torch_rng_state": torch.get_rng_state(),
            "cuda_rng_state": (
                torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
            ),
            "date": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "torch_version": torch.__version__,
            "hostname": platform.node(),
            "model_name": MODEL_NAME,
            "encoder": ENCODER_NAME,
            "batch_size": BATCH_SIZE,
            "learning_rate": LEARNING_RATE,
        }
        torch.save(checkpoint, CHECKPOINT_DIR / filename)

    def train_one_epoch(epoch: int):
        model.train()
        loss_m = AverageMeter("Loss")
        norm_m = AverageMeter("GradNorm")
        inter_sum, pred_sum, target_sum = 0.0, 0.0, 0.0

        pbar = tqdm(
            train_loader, desc=f"Train Ep {epoch:03d}", leave=False, dynamic_ncols=True
        )
        start_time = time.time()

        for batch_idx, batch in enumerate(pbar, 1):
            images = batch["image"].to(DEVICE, non_blocking=True)
            masks = (
                batch["mask"].unsqueeze(1).float()
                if NUM_CLASSES == 1
                else batch["mask"].long()
            )
            masks = masks.to(DEVICE, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with autocast(device_type=DEVICE.type, enabled=scaler.is_enabled()):
                outputs = model(images)
                loss = criterion(outputs, masks)

            if not torch.isfinite(loss):
                raise RuntimeError(
                    f"Loss is {loss.item()}. Ngừng huấn luyện để tránh hỏng model (NaN/Inf)."
                )

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            scaler.step(optimizer)
            scaler.update()

            if scheduler and isinstance(scheduler, OneCycleLR):
                scheduler.step()

            loss_m.update(loss.item(), images.size(0))
            if torch.isfinite(grad_norm):
                norm_m.update(grad_norm.item())

            inter, pred_a, target_a = calculate_metrics(outputs.detach(), masks)
            inter_sum += inter
            pred_sum += pred_a
            target_sum += target_a

            if batch_idx % 5 == 0:
                fps = (batch_idx * BATCH_SIZE) / (time.time() - start_time)
                pbar.set_postfix(
                    {
                        "Loss": f"{loss_m.avg:.4f}",
                        "Norm": f"{norm_m.avg:.2f}",
                        "GPU": get_peak_gpu_mem(),
                        "Img/s": f"{fps:.1f}",
                    }
                )

        if scheduler and not isinstance(scheduler, (ReduceLROnPlateau, OneCycleLR)):
            scheduler.step()
        ep_dice = (2.0 * inter_sum + 1e-7) / (pred_sum + target_sum + 1e-7)
        ep_iou = (inter_sum + 1e-7) / (pred_sum + target_sum - inter_sum + 1e-7)
        return loss_m.avg, ep_dice, ep_iou

    @torch.no_grad()
    def validate_one_epoch(epoch: int):
        model.eval()
        loss_m = AverageMeter("ValLoss")
        inter_sum, pred_sum, target_sum = 0.0, 0.0, 0.0
        start_time = time.time()
        total_images = 0

        pbar = tqdm(
            val_loader, desc=f"Valid Ep {epoch:03d}", leave=False, dynamic_ncols=True
        )

        for batch_idx, data in enumerate(pbar, 1):
            images = data["image"].to(DEVICE, non_blocking=True)
            masks = (
                data["mask"].unsqueeze(1).float()
                if NUM_CLASSES == 1
                else data["mask"].long()
            )
            masks = masks.to(DEVICE, non_blocking=True)
            total_images += images.size(0)

            with autocast(device_type=DEVICE.type, enabled=scaler.is_enabled()):
                outputs = model(images)
                loss = criterion(outputs, masks)

            loss_m.update(loss.item(), images.size(0))
            inter, pred_a, target_a = calculate_metrics(outputs, masks)
            inter_sum += inter
            pred_sum += pred_a
            target_sum += target_a

            if batch_idx == 1 and (epoch % 5 == 0 or epoch == 1):
                save_validation_visualization(images, masks, outputs, epoch)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        ep_dice = (2.0 * inter_sum + 1e-7) / (pred_sum + target_sum + 1e-7)
        ep_iou = (inter_sum + 1e-7) / (pred_sum + target_sum - inter_sum + 1e-7)
        ms_per_img = ((time.time() - start_time) / total_images) * 1000

        if scheduler and isinstance(scheduler, ReduceLROnPlateau):
            scheduler.step(ep_dice)
        return loss_m.avg, ep_dice, ep_iou, ms_per_img

    # ==========================================================
    # TIẾN HÀNH CHẠY HUẤN LUYỆN
    # ==========================================================
    for epoch in range(start_epoch, EPOCHS + 1):
        epoch_start = time.time()

        t_loss, t_dice, t_iou = train_one_epoch(epoch)
        v_loss, v_dice, v_iou, ms_img = validate_one_epoch(epoch)

        epoch_time = time.time() - epoch_start
        current_lr = optimizer.param_groups[0]["lr"]

        msg = (
            f"\n[Epoch {epoch:03d}] Time: {epoch_time:.1f}s | Val Speed: {ms_img:.1f}ms/img | Peak GPU: {get_peak_gpu_mem()} | LR: {current_lr:.6e}\n"
            f"  Train -> Loss: {t_loss:.4f} | Dice: {t_dice:.4f} | IoU: {t_iou:.4f}\n"
            f"  Valid -> Loss: {v_loss:.4f} | Dice: {v_dice:.4f} | IoU: {v_iou:.4f}"
        )
        print(msg)
        write_log(msg)

        hist_df = pd.DataFrame(
            [
                {
                    "epoch": epoch,
                    "train_loss": t_loss,
                    "val_loss": v_loss,
                    "train_dice": t_dice,
                    "val_dice": v_dice,
                    "train_iou": t_iou,
                    "val_iou": v_iou,
                    "lr": current_lr,
                }
            ]
        )
        hist_df.to_csv(
            CSV_HISTORY, mode="a", header=not CSV_HISTORY.exists(), index=False
        )

        is_best = False
        if v_dice > best_dice:
            is_best = True
        elif v_dice == best_dice and v_loss < best_loss:
            is_best = True

        if is_best:
            best_dice, best_loss, best_epoch = v_dice, v_loss, epoch
            patience = 0
            save_checkpoint(epoch, best_dice, best_loss, "best_model.pth")
            write_msg = (
                f"  🔥 KỶ LỤC MỚI (Best Dice: {best_dice:.4f} | Loss: {best_loss:.4f})"
            )
            print(write_msg)
            write_log(write_msg)
        else:
            patience += 1
            write_msg = f"  ⚠️ Không cải thiện ({patience}/{EARLY_STOPPING})"
            print(write_msg)
            write_log(write_msg)

        if epoch % SAVE_INTERVAL == 0:
            save_checkpoint(epoch, best_dice, best_loss, f"epoch_{epoch:03d}.pth")
        save_checkpoint(epoch, best_dice, best_loss, "last_model.pth")

        if patience >= EARLY_STOPPING:
            stop_msg = "\n❌ KÍCH HOẠT DỪNG SỚM (EARLY STOPPING)!"
            print(stop_msg)
            write_log(stop_msg)
            break

    plot_training_curves()
    done_msg = (
        "\n"
        + "=" * 80
        + f"\n🎉 HUẤN LUYỆN HOÀN TẤT! Best Epoch: {best_epoch} | Best Dice: {best_dice:.4f}\n"
        + "=" * 80
    )
    print(done_msg)
    write_log(done_msg)


if __name__ == "__main__":
    if os.name == "nt":
        import torch.multiprocessing

        torch.multiprocessing.freeze_support()
    main()
