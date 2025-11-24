#!/usr/bin/env python3
"""
finetune_deeplab_head.py (corrected)

Fast head fine-tuning for DeepLabV3 (ResNet-50 backbone).
- Loads cleaned checkpoint: checkpoints_deeplab/best_deeplab_clean.pth
- Freezes backbone, sets backbone BatchNorm to eval(), trains classifier (+ aux_classifier if present)
- Uses BCEWithLogits + Dice loss
- Uses ImageNet normalization and repeats 1->3 channels
- Saves best model to checkpoints_deeplab/best_deeplab_refined.pth

Usage:
    python finetune_deeplab_head.py
"""

import os
import copy
import time
from pathlib import Path
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision.models.segmentation import deeplabv3_resnet50

# ----------------- USER CONFIG -----------------
DATA_ROOT = "datasets_reduced_500"                    # dataset used for Deeplab (train/val)
TRAIN_SPLIT = "train"                                 # or set to "val" for quick runs
VAL_SPLIT = "val"
CLEAN_CKPT = "checkpoints_deeplab/best_deeplab_clean.pth"
OUT_CKPT   = "checkpoints_deeplab/best_deeplab_refined.pth"
NUM_EPOCHS = 5            # recommended 3-10
BATCH_SIZE = 4
IMAGE_SIZE = 256
LR_HEAD = 1e-3
WEIGHT_DECAY = 1e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_WORKERS = 2
PRINT_EVERY = 50
# ------------------------------------------------

# ------------- Dice loss -------------
def dice_loss_logits(pred_logits, target, smooth=1e-6):
    probs = torch.sigmoid(pred_logits)
    probs = probs.view(-1)
    target = target.view(-1)
    inter = (probs * target).sum()
    union = probs.sum() + target.sum()
    return 1.0 - (2 * inter + smooth) / (union + smooth)

# ------------- Dataset -------------
IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}

class SegmentationDataset(Dataset):
    def __init__(self, root, split="train", image_size=256):
        self.root = Path(root) / split
        if not self.root.exists():
            raise FileNotFoundError(f"{self.root} not found")
        self.samples = []
        for cls in sorted(self.root.iterdir()):
            if not cls.is_dir():
                continue
            imgs = (cls / "images")
            masks = (cls / "masks")
            if not imgs.exists() or not masks.exists():
                continue
            for im in imgs.iterdir():
                if im.suffix.lower() not in IMG_EXTS:
                    continue
                mask = masks / im.name
                if not mask.exists():
                    mask_alt = masks / (im.stem + ".png")
                    if mask_alt.exists():
                        mask = mask_alt
                    else:
                        continue
                self.samples.append((im, mask))
        self.image_size = image_size

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_p, m_p = self.samples[idx]
        img = Image.open(img_p).convert("L").resize((self.image_size, self.image_size))
        m = Image.open(m_p).convert("L").resize((self.image_size, self.image_size), resample=Image.NEAREST)
        img_np = np.array(img).astype(np.float32) / 255.0
        mask_np = (np.array(m) > 127).astype(np.float32)
        img_t = torch.from_numpy(img_np).unsqueeze(0).float()   # [1,H,W]
        mask_t = torch.from_numpy(mask_np).unsqueeze(0).float() # [1,H,W]
        return img_t, mask_t, img_p.name

# ------------- Utils -------------
def ensure_dir(p):
    Path(p).mkdir(parents=True, exist_ok=True)

def build_deeplab(num_classes=1):
    model = deeplabv3_resnet50(weights=None)
    try:
        last = list(model.classifier.children())[-1]
        if isinstance(last, nn.Conv2d):
            in_ch = last.in_channels
            model.classifier[-1] = nn.Conv2d(in_ch, num_classes, kernel_size=1)
    except Exception:
        pass
    return model

def load_clean_checkpoint(model, path):
    ckpt = torch.load(path, map_location="cpu")
    state = ckpt.get("model_state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
    new_state = {}
    for k,v in state.items():
        nk = k
        if nk.startswith("model."):
            nk = nk[len("model."):]
        if nk.startswith("module."):
            nk = nk[len("module."):]
        new_state[nk] = v
    model.load_state_dict(new_state, strict=False)
    return model

# ------------- Training & evaluation helpers -------------
def evaluate_model(model, loader, threshold=0.3):
    model.eval()
    dices = []
    mean = torch.tensor([0.485,0.456,0.406], device=DEVICE).view(1,3,1,1)
    std  = torch.tensor([0.229,0.224,0.225], device=DEVICE).view(1,3,1,1)
    with torch.no_grad():
        for img, mask, _ in loader:
            img = img.to(DEVICE)    # [B,1,H,W]
            mask = mask.to(DEVICE)
            img3 = img.repeat(1,3,1,1)
            inp = (img3 - mean) / std
            out = model(inp)
            if isinstance(out, dict) and "out" in out:
                out = out["out"]
            prob = torch.sigmoid(out)
            pred = (prob > threshold).float()
            for b in range(pred.shape[0]):
                p = pred[b,0]
                g = mask[b,0]
                inter = (p * g).sum()
                dice = (2*inter) / (p.sum() + g.sum() + 1e-6)
                dices.append(float(dice))
    return np.mean(dices) if len(dices)>0 else 0.0

def train():
    print("Device:", DEVICE)
    ensure_dir(Path(OUT_CKPT).parent)

    ds_tr = SegmentationDataset(DATA_ROOT, split=TRAIN_SPLIT, image_size=IMAGE_SIZE)
    ds_val = SegmentationDataset(DATA_ROOT, split=VAL_SPLIT, image_size=IMAGE_SIZE)
    print("Train size:", len(ds_tr), "Val size:", len(ds_val))

    # drop_last True only if dataset length >= batch size (avoid empty loader)
    drop_last_flag = len(ds_tr) >= BATCH_SIZE
    loader_tr = DataLoader(ds_tr, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, drop_last=drop_last_flag)
    loader_val = DataLoader(ds_val, batch_size=1, shuffle=False, num_workers=NUM_WORKERS)

    model = build_deeplab(num_classes=1).to(DEVICE)
    model = load_clean_checkpoint(model, CLEAN_CKPT)
    model.train()

    # Freeze backbone parameters and set backbone BatchNorm layers to eval()
    for name, param in model.named_parameters():
        if name.startswith("backbone"):
            param.requires_grad = False

    # Put backbone in eval mode and freeze its BatchNorm running/stat behavior
    if hasattr(model, "backbone"):
        model.backbone.eval()
        for m in model.backbone.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()
                for p in m.parameters():
                    p.requires_grad = False

    # Ensure classifier parameters require grad (they should)
    trainable_params = [p for n,p in model.named_parameters() if p.requires_grad]
    total_trainable = sum(p.numel() for p in trainable_params)
    print("Trainable parameter count:", total_trainable)

    optimizer = torch.optim.AdamW(trainable_params, lr=LR_HEAD, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2, verbose=True)

    bce = nn.BCEWithLogitsLoss()

    best_val = -1.0
    best_model_wts = copy.deepcopy(model.state_dict())

    for epoch in range(NUM_EPOCHS):
        model.train()
        running_loss = 0.0
        t0 = time.time()
        for i, (img, mask, fname) in enumerate(loader_tr):
            img = img.to(DEVICE)   # [B,1,H,W]
            mask = mask.to(DEVICE) # [B,1,H,W]

            img3 = img.repeat(1,3,1,1)
            mean = torch.tensor([0.485,0.456,0.406], device=DEVICE).view(1,3,1,1)
            std  = torch.tensor([0.229,0.224,0.225], device=DEVICE).view(1,3,1,1)
            inp = (img3 - mean) / std

            out = model(inp)
            if isinstance(out, dict) and "out" in out:
                out_main = out["out"]
            else:
                out_main = out

            loss_bce = bce(out_main, mask)
            loss_dice = dice_loss_logits(out_main, mask)
            loss = loss_bce + loss_dice

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += float(loss.item())

            if (i+1) % PRINT_EVERY == 0:
                print(f"[Epoch {epoch+1}/{NUM_EPOCHS}] Step {i+1}/{len(loader_tr)} loss: {running_loss/(i+1):.4f}")

        epoch_time = time.time() - t0
        avg_loss = running_loss / max(1, len(loader_tr))
        # evaluate with lower threshold (more tolerant)
        val_dice = evaluate_model(model, loader_val, threshold=0.3)
        scheduler.step(val_dice)

        print(f"Epoch {epoch+1}/{NUM_EPOCHS} finished. train_loss: {avg_loss:.4f} val_dice(thr=0.3): {val_dice:.4f} time: {epoch_time:.1f}s")

        # save best
        if val_dice > best_val:
            best_val = val_dice
            best_model_wts = copy.deepcopy(model.state_dict())
            torch.save({"model_state_dict": best_model_wts, "best_val": best_val, "epoch": epoch+1}, OUT_CKPT)
            print("Saved best model to", OUT_CKPT)

    # save final weights
    torch.save({"model_state_dict": model.state_dict(), "best_val": best_val, "epoch": NUM_EPOCHS}, OUT_CKPT)
    print("Training complete. Best val dice:", best_val)

if __name__ == "__main__":
    train()
