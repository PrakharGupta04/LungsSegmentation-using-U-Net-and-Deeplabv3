# train_deeplab_fixed2.py
"""
DeepLabV3-ResNet50 fine-tuning for lung segmentation (final patch)
- Fixes classifier replacement by updating only the final conv to output 1 channel
- Ensures output shape [B,1,256,256]
- Uses IMAGE_SIZE = 256 (as you preprocessed)
- Saves checkpoints to checkpoints_deeplab/
"""

import argparse
import json
import time
from pathlib import Path
import random
import sys

import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, ConcatDataset

import torchvision
from torchvision.models.segmentation import deeplabv3_resnet50, DeepLabV3_ResNet50_Weights

# ---------------- CONFIG ----------------
class Config:
    DATA_ROOT = "datasets_reduced_500"
    CHECKPOINT_DIR = Path("checkpoints_deeplab")
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    IMAGE_SIZE = 256        # your images are already 256x256
    OUTPUT_SIZE = 256       # final mask size
    BATCH_SIZE = 2          # drop to 1 if OOM
    NUM_EPOCHS = 15
    LR = 1e-4
    WD = 1e-4

    SEED = 42

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    NUM_WORKERS = 0
    PIN_MEMORY = True

    LOG_EVERY = 10
    SAVE_EVERY = 5

config = Config()

# reproducibility
torch.manual_seed(config.SEED)
random.seed(config.SEED)
np.random.seed(config.SEED)

# ---------------- Dataset ----------------
IMG_EXTS = {'.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp'}

class LungSegDataset(Dataset):
    def __init__(self, images_dir, masks_dir, image_size=config.IMAGE_SIZE):
        self.images_dir = Path(images_dir)
        self.masks_dir = Path(masks_dir)
        self.image_size = int(image_size)
        self.pairs = self._find_pairs()
        if len(self.pairs) == 0:
            print(f"  ⚠️  WARNING: No pairs found in {images_dir}")

    def _find_pairs(self):
        if not self.images_dir.exists() or not self.masks_dir.exists():
            return []
        image_files = sorted([p for p in self.images_dir.glob("*") if p.suffix.lower() in IMG_EXTS])
        mask_files = sorted([p for p in self.masks_dir.glob("*") if p.suffix.lower() in IMG_EXTS])
        mask_map = {m.stem: m for m in mask_files}
        pairs = []
        for im in image_files:
            if im.stem in mask_map:
                pairs.append((im, mask_map[im.stem]))
            else:
                # fuzzy fallback
                found = False
                for k, m in mask_map.items():
                    if k in im.stem or im.stem in k:
                        pairs.append((im, m))
                        found = True
                        break
                if not found:
                    # skip unmatched
                    pass
        return pairs

    def _get_bbox_from_mask(self, mask_np):
        rows = np.any(mask_np, axis=1)
        cols = np.any(mask_np, axis=0)
        if not rows.any() or not cols.any():
            return np.array([0, 0, self.image_size, self.image_size])
        ymin, ymax = np.where(rows)[0][[0, -1]]
        xmin, xmax = np.where(cols)[0][[0, -1]]
        mx = int((xmax - xmin) * 0.05)
        my = int((ymax - ymin) * 0.05)
        xmin = max(0, xmin - mx)
        ymin = max(0, ymin - my)
        xmax = min(self.image_size, xmax + mx)
        ymax = min(self.image_size, ymax + my)
        return np.array([xmin, ymin, xmax, ymax])

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        img_path, mask_path = self.pairs[idx]
        # Force RGB input (3 channels) because torchvision pretrained models expect 3-channel images.
        image = Image.open(img_path).convert("RGB").resize((self.image_size, self.image_size), Image.BILINEAR)
        mask = Image.open(mask_path).convert("L").resize((self.image_size, self.image_size), Image.NEAREST)

        image_np = np.array(image).astype(np.float32) / 255.0  # H,W,3
        mask_np = (np.array(mask) > 127).astype(np.float32)   # H,W (binary)

        bbox = self._get_bbox_from_mask(mask_np)

        image_tensor = torch.from_numpy(image_np).permute(2,0,1).float()  # 3,H,W
        mask_tensor = torch.from_numpy(mask_np).unsqueeze(0).float()      # 1,H,W
        bbox_tensor = torch.from_numpy(bbox).float()
        return image_tensor, mask_tensor, bbox_tensor

# ---------------- Dataloaders ----------------
def create_dataloaders(data_root=config.DATA_ROOT, batch_size=None):
    if batch_size is None:
        batch_size = config.BATCH_SIZE
    root = Path(data_root)
    if not root.exists():
        raise FileNotFoundError(f"Data root not found: {root.resolve()}")

    datasets_by_split = {'train': [], 'val': []}
    for split in ['train', 'val']:
        split_dir = root / split
        if not split_dir.exists():
            print(f"⚠️  Split dir missing: {split_dir} (continue)")
            continue
        print(f"\nLoading split: {split} from {split_dir}")
        for class_dir in sorted(split_dir.iterdir()):
            if not class_dir.is_dir(): continue
            images_dir = class_dir / "images"
            masks_dir = class_dir / "masks"
            if images_dir.exists() and masks_dir.exists():
                ds = LungSegDataset(images_dir, masks_dir, image_size=config.IMAGE_SIZE)
                if len(ds) > 0:
                    datasets_by_split[split].append(ds)
                else:
                    print(f"  ⚠️  No pairs inside: {class_dir}")
            else:
                print(f"  ⚠️  Missing images or masks in: {class_dir}")

    train_ds = ConcatDataset(datasets_by_split['train']) if datasets_by_split['train'] else None
    val_ds = ConcatDataset(datasets_by_split['val']) if datasets_by_split['val'] else None

    def mk_loader(ds, shuffle):
        if ds is None:
            return None
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                          num_workers=config.NUM_WORKERS, pin_memory=config.PIN_MEMORY,
                          drop_last=shuffle)
    train_loader = mk_loader(train_ds, True)
    val_loader = mk_loader(val_ds, False)

    # summary
    print("\nDATALOADER SUMMARY")
    if train_loader:
        try:
            print(f"Train samples: {len(train_loader.dataset)}  batches: {len(train_loader)}")
        except Exception:
            pass
    else:
        print("Train loader: NONE")

    if val_loader:
        try:
            print(f"Val samples:   {len(val_loader.dataset)}  batches: {len(val_loader)}")
        except Exception:
            pass
    else:
        print("Val loader: NONE (will only train)")

    return train_loader, val_loader

# ---------------- Model wrapper ----------------
class DeepLabForLungs(nn.Module):
    def __init__(self, pretrained_backbone=True, num_classes=1):
        super().__init__()
        # set weights properly
        if pretrained_backbone:
            weights = DeepLabV3_ResNet50_Weights.COCO_WITH_VOC_LABELS_V1
            self.model = deeplabv3_resnet50(weights=weights, aux_loss=True)
        else:
            self.model = deeplabv3_resnet50(weights=None, aux_loss=False)

        # Instead of replacing the entire classifier (which broke channels), we locate the
        # final conv in the classifier and replace only that conv to output `num_classes` channels.
        # This preserves ASPP internals and avoids in/out channel mismatches.
        replaced = False
        try:
            # Traverse modules in classifier in reversed order to find last Conv2d
            for i, m in reversed(list(enumerate(self.model.classifier))):
                if isinstance(m, nn.Conv2d):
                    in_ch = m.in_channels
                    # replace this conv with same kernel but out_channels=num_classes
                    new_conv = nn.Conv2d(in_ch, num_classes, kernel_size=1)
                    # keep surrounding modules: rebuild classifier sequence
                    cls_modules = list(self.model.classifier)
                    cls_modules[i] = new_conv
                    self.model.classifier = nn.Sequential(*cls_modules)
                    replaced = True
                    break
        except Exception:
            replaced = False

        if not replaced:
            # Fallback: assume ASPP output channels = 256 and set a simple classifier
            aspp_out = 256
            self.model.classifier = nn.Sequential(
                nn.Conv2d(aspp_out, 256, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                nn.Conv2d(256, num_classes, kernel_size=1)
            )

    def forward(self, x):
        out = self.model(x)['out']
        # ensure output is float32 logits
        return out

# ---------------- Loss & metrics ----------------
def dice_loss(pred, target, smooth=1e-6):
    pred = torch.sigmoid(pred)
    inter = (pred * target).sum(dim=(1,2,3))
    union = pred.sum(dim=(1,2,3)) + target.sum(dim=(1,2,3))
    dice = (2*inter + smooth) / (union + smooth)
    return 1 - dice.mean()

def dice_score(pred, target, th=0.5, smooth=1e-6):
    pred_bin = (torch.sigmoid(pred) > th).float()
    inter = (pred_bin * target).sum()
    union = pred_bin.sum() + target.sum()
    return ((2*inter + smooth) / (union + smooth)).item()

# ---------------- Training ----------------
def train(model, train_loader, val_loader):
    device = config.DEVICE
    model = model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=config.LR, weight_decay=config.WD)
    criterion = nn.BCEWithLogitsLoss()

    best_val = -1.0
    history = {'train_loss': [], 'val_dice': []}

    for epoch in range(1, config.NUM_EPOCHS + 1):
        model.train()
        running_loss = 0.0
        nb = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{config.NUM_EPOCHS} [Train]")
        for imgs, masks, _ in pbar:
            imgs = imgs.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)

            optimizer.zero_grad()
            outputs = model(imgs)  # logits [B, C, H_out, W_out]

            # Force the final output to be single-channel and size OUTPUT_SIZE
            if outputs.shape[1] != 1:
                # If model outputs multi-channel, reduce by taking first channel (shouldn't happen if replaced)
                outputs = outputs[:, :1, :, :]

            outputs_resized = nn.functional.interpolate(outputs, size=(config.OUTPUT_SIZE, config.OUTPUT_SIZE), mode='bilinear', align_corners=False)
            masks_resized = nn.functional.interpolate(masks, size=(config.OUTPUT_SIZE, config.OUTPUT_SIZE), mode='nearest')

            loss_bce = criterion(outputs_resized, masks_resized)
            loss_d = dice_loss(outputs_resized, masks_resized)
            loss = 0.5 * loss_bce + 0.5 * loss_d

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            running_loss += loss.item()
            nb += 1
            if nb % config.LOG_EVERY == 0:
                pbar.set_postfix({'loss': f"{running_loss/nb:.4f}"})
        avg_loss = running_loss / max(1, nb)
        history['train_loss'].append(avg_loss)

        # validation
        val_dice = 0.0
        val_batches = 0
        if val_loader is not None:
            model.eval()
            with torch.no_grad():
                for imgs, masks, _ in tqdm(val_loader, desc=f"Epoch {epoch}/{config.NUM_EPOCHS} [Val]", leave=False):
                    imgs = imgs.to(device, non_blocking=True)
                    masks = masks.to(device, non_blocking=True)
                    outputs = model(imgs)
                    if outputs.shape[1] != 1:
                        outputs = outputs[:, :1, :, :]
                    outputs_resized = nn.functional.interpolate(outputs, size=(config.OUTPUT_SIZE, config.OUTPUT_SIZE), mode='bilinear', align_corners=False)
                    masks_resized = nn.functional.interpolate(masks, size=(config.OUTPUT_SIZE, config.OUTPUT_SIZE), mode='nearest')
                    val_dice += dice_score(outputs_resized, masks_resized)
                    val_batches += 1
        val_dice = (val_dice / val_batches) if val_batches > 0 else 0.0
        history['val_dice'].append(val_dice)

        print(f"\nEpoch {epoch} summary -> Train loss: {avg_loss:.4f}  Val Dice: {val_dice:.4f}")

        # save best
        if val_dice > best_val:
            best_val = val_dice
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val': best_val,
                'history': history
            }, config.CHECKPOINT_DIR / "best_deeplab.pth")
            print(f"★ New best model saved (dice={best_val:.4f})")

        if epoch % config.SAVE_EVERY == 0:
            torch.save({'epoch': epoch, 'model_state_dict': model.state_dict()}, config.CHECKPOINT_DIR / f"deeplab_epoch{epoch:03d}.pth")

    # write history
    with open(config.CHECKPOINT_DIR / "history.json", "w") as f:
        json.dump(history, f, indent=2)

    print("\nTraining complete. Best val dice:", best_val)
    return model, history

# ---------------- CLI & MAIN ----------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", default=config.DATA_ROOT)
    p.add_argument("--batch_size", type=int, default=config.BATCH_SIZE)
    p.add_argument("--epochs", type=int, default=config.NUM_EPOCHS)
    p.add_argument("--lr", type=float, default=config.LR)
    p.add_argument("--no_pretrained", action="store_true", help="Disable pretrained backbone weights")
    return p.parse_args()

def main():
    args = parse_args()
    config.NUM_EPOCHS = args.epochs
    config.BATCH_SIZE = args.batch_size
    config.LR = args.lr

    data_root = args.data_root

    print("\nCONFIG")
    print(f"DATA_ROOT: {data_root}")
    print(f"DEVICE: {config.DEVICE}")
    print(f"IMAGE_SIZE: {config.IMAGE_SIZE}, BATCH_SIZE: {config.BATCH_SIZE}")
    print(f"CHECKPOINT_DIR: {config.CHECKPOINT_DIR.resolve()}\n")

    train_loader, val_loader = create_dataloaders(data_root, batch_size=config.BATCH_SIZE)
    if train_loader is None:
        print("❌ No training data found. Exiting.")
        return

    # create model
    model = DeepLabForLungs(pretrained_backbone=not args.no_pretrained, num_classes=1)
    model.to(config.DEVICE)

    # sanity GPU check
    print("CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        try:
            print("GPU:", torch.cuda.get_device_name(0))
        except Exception:
            pass

    train(model, train_loader, val_loader)

if __name__ == "__main__":
    main()
