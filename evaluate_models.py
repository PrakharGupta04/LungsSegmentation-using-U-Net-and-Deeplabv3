#!/usr/bin/env python3
"""
evaluate_models.py

Run with:
    python evaluate_models.py

Defaults (no CLI args needed):
- UNet dataset: data/val/
- DeepLab dataset: datasets_reduced_500/val/
- UNet checkpoint: checkpoints/best_model.pth
- DeepLab checkpoint: checkpoints_deeplab/best_deeplab.pth
- Output: results/comparison_<timestamp>/

Features:
- fixes 1-channel vs 3-channel mismatch automatically
- evaluates a subset (default eval_subset=100)
- saves N_samples examples (default n_samples=3) showing both models side-by-side
- saves JSON report, summary, plots, and debug logs
"""
import json
from pathlib import Path
from datetime import datetime
import random
import os
import sys
import traceback

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision.models.segmentation import deeplabv3_resnet50
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# ----------------------------
# USER / PROJECT DEFAULTS
# ----------------------------
UNET_DATA_ROOT = "data"                     # dataset used for UNet (has train/val/test splits)
DEEPLAB_DATA_ROOT = "datasets_reduced_500"  # dataset used for DeepLab (reduced)
UNET_CHECKPOINT = "checkpoints/best_model.pth"
DEEPLAB_CHECKPOINT = "checkpoints_deeplab/best_deeplab_refined.pth"

OUTPUT_ROOT = "results"

# Evaluation hyperparams (change here if needed)
EVAL_SUBSET = 100   # number of images to compute metrics on (clamped to dataset size)
N_SAMPLES = 3       # number of visual example images to save (combined montage)
THRESHOLD = 0.5     # binarization threshold
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# A specific image you provided earlier (diagnostics will be printed if present)
DIAG_IMAGE_PATH = Path("/mnt/data/016884af-7d2c-493c-b102-9e9549a0ee12.png")

# ----------------------------
# Utilities
# ----------------------------
def ensure_dir(p):
    Path(p).mkdir(parents=True, exist_ok=True)

def dice_coef(pred, target, smooth=1e-6):
    pred = pred.reshape(-1)
    target = target.reshape(-1)
    inter = (pred * target).sum()
    union = pred.sum() + target.sum()
    return (2 * inter + smooth) / (union + smooth)

def iou_coef(pred, target, smooth=1e-6):
    pred = pred.reshape(-1)
    target = target.reshape(-1)
    inter = (pred * target).sum()
    union = pred.sum() + target.sum() - inter
    return (inter + smooth) / (union + smooth)

def binarize_probs(probs, thr=0.5):
    return (probs > thr).float()

def model_input_channels(model):
    # inspect first parameter to infer required input channels
    for p in model.parameters():
        # param shape like (out_ch, in_ch, k, k) for conv weight
        if p.dim() >= 2:
            shape = p.shape
            if len(shape) >= 2:
                in_ch = shape[1]
                return int(in_ch)
    return None

def adapt_input_channels(img_tensor, required_in_ch):
    # img_tensor: [B,C,H,W]
    c = img_tensor.shape[1]
    if c == required_in_ch:
        return img_tensor
    if c == 1 and required_in_ch == 3:
        return img_tensor.repeat(1,3,1,1)
    if c == 3 and required_in_ch == 1:
        return img_tensor.mean(dim=1, keepdim=True)
    # fallback - try to slice or repeat
    if c < required_in_ch:
        return img_tensor.repeat(1, required_in_ch // c + 1, 1, 1)[:, :required_in_ch, ...]
    else:
        return img_tensor[:, :required_in_ch, ...]

# ----------------------------
# Dataset (expects <root>/<split>/<class>/images & masks)
# ----------------------------
IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}

class SimpleLungDataset(Dataset):
    def __init__(self, root, split="val", image_size=256):
        self.root = Path(root) / split
        if not self.root.exists():
            raise FileNotFoundError(f"Dataset split not found: {self.root}")
        self.image_size = image_size
        self.pairs = []
        # Walk disease/class folders
        for class_folder in sorted(self.root.iterdir()):
            if not class_folder.is_dir(): 
                continue
            imgs_dir = class_folder / "images"
            masks_dir = class_folder / "masks"
            if not imgs_dir.exists():
                continue
            img_files = sorted([p for p in imgs_dir.iterdir() if p.suffix.lower() in IMG_EXTS])
            # try matching masks by filename
            mask_map = {m.name: m for m in masks_dir.iterdir() if m.suffix.lower() in IMG_EXTS} if masks_dir.exists() else {}
            for im in img_files:
                if im.name in mask_map:
                    self.pairs.append((im, mask_map[im.name]))
                else:
                    # try matching by stem
                    candidate = masks_dir / (im.stem + ".png")
                    if candidate.exists():
                        self.pairs.append((im, candidate))
                    else:
                        # last resort: skip
                        continue

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        im_path, mask_path = self.pairs[idx]
        img = Image.open(im_path).convert("L").resize((self.image_size, self.image_size))
        mask = Image.open(mask_path).convert("L").resize((self.image_size, self.image_size))
        img_arr = np.array(img).astype(np.float32) / 255.0
        mask_arr = (np.array(mask) > 127).astype(np.float32)
        img_t = torch.from_numpy(img_arr).unsqueeze(0).float()  # 1,H,W
        mask_t = torch.from_numpy(mask_arr).unsqueeze(0).float() # 1,H,W
        return img_t, mask_t, im_path.name

# ----------------------------
# Model helpers
# ----------------------------
class SmallUNet(nn.Module):
    def __init__(self, in_ch=1, out_ch=1):
        super().__init__()
        def C(in_c, out_c):
            return nn.Sequential(
                nn.Conv2d(in_c, out_c, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_c, out_c, 3, padding=1),
                nn.ReLU(inplace=True)
            )
        self.enc1 = C(in_ch, 32)
        self.pool = nn.MaxPool2d(2)
        self.enc2 = C(32,64)
        self.enc3 = C(64,128)
        self.up2 = nn.ConvTranspose2d(128,64,2,2)
        self.dec2 = C(128,64)
        self.up1 = nn.ConvTranspose2d(64,32,2,2)
        self.dec1 = C(64,32)
        self.final = nn.Conv2d(32,out_ch,1)

    def forward(self,x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        d2 = self.up2(e3)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)
        d1 = self.up1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)
        out = self.final(d1)
        return out

def build_deeplab(num_classes=1):
    model = deeplabv3_resnet50(weights=None)
    # replace classifier final conv to 1 channel if necessary
    try:
        # some torchvision versions have classifier as nn.Sequential; adapt last conv
        last = list(model.classifier.children())[-1]
        if isinstance(last, nn.Conv2d):
            in_ch = last.in_channels
            model.classifier[-1] = nn.Conv2d(in_ch, num_classes, kernel_size=1)
    except Exception:
        # fallback: simple classifier
        model.classifier = nn.Sequential(
            nn.Conv2d(256,256,3,padding=1,bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256,num_classes,1)
        )
    return model

def load_checkpoint(model, ckpt_path):
    ckpt_p = Path(ckpt_path)
    if not ckpt_p.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_p}")
    payload = torch.load(str(ckpt_p), map_location="cpu")
    # try common keys
    if isinstance(payload, dict):
        if "model_state_dict" in payload:
            state = payload["model_state_dict"]
        elif "state_dict" in payload:
            state = payload["state_dict"]
        else:
            # maybe full state dict
            state = payload
    else:
        state = payload
    # load with strict=False to avoid minor key mismatches
    missing = None
    try:
        model.load_state_dict(state, strict=False)
    except Exception as e:
        # try to be robust: if keys are nested like module., strip prefix
        new_state = {}
        for k,v in state.items():
            nk = k
            if k.startswith("module."):
                nk = k[len("module."):]
            new_state[nk] = v
        model.load_state_dict(new_state, strict=False)
    model.to(DEVICE)
    model.eval()
    return model

# ----------------------------
# Evaluation & visualization
# ----------------------------
def eval_models_on_subset(unet, deeplab, loader, out_dir, subset_idx, n_samples=3, threshold=0.5, debug_for=None):
    """
    subset_idx : list of dataset indices to evaluate
    loader : DataLoader with batch_size=1 used to fetch items by index via dataset
    """
    ds = loader.dataset
    results_unet = []
    results_deeplab = []
    vis_samples = []
    debug_lines = []

    # determine expected in-channels
    unet_req_ch = model_input_channels(unet) or 1
    deeplab_req_ch = model_input_channels(deeplab) or 3

    # iterate subset indices
    for idx in subset_idx:
        img_t, mask_t, fname = ds[idx]  # img_t: [1,H,W]
        img = img_t.unsqueeze(0).to(DEVICE)  # [B=1, C=1, H, W] already unsqueezed to 1 in dataset; double-check
        # if dataset returns [C,H,W] we did img_t.unsqueeze(0) -> [1,C,H,W]
        if img.dim() == 3:
            img = img.unsqueeze(0)  # safety: ensure batch dim
        mask = mask_t.unsqueeze(0).to(DEVICE) if mask_t.dim()==3 else mask_t.to(DEVICE)

        # prepare inputs per-model
        img_for_unet = adapt_input_channels(img, unet_req_ch)
        img_for_deeplab = adapt_input_channels(img, deeplab_req_ch)

        # forward UNet
        with torch.no_grad():
            try:
                out_unet = unet(img_for_unet)
                if isinstance(out_unet, dict) and 'out' in out_unet:
                    out_unet = out_unet['out']
            except Exception as e:
                # log and continue
                debug_lines.append(f"UNet forward error on {fname}: {repr(e)}")
                out_unet = torch.zeros((1,1,img.shape[-2], img.shape[-1]), device=DEVICE)

            # forward DeepLab
            try:
                out_deeplab = deeplab(img_for_deeplab)
                if isinstance(out_deeplab, dict) and 'out' in out_deeplab:
                    out_deeplab = out_deeplab['out']
            except Exception as e:
                debug_lines.append(f"DeepLab forward error on {fname}: {repr(e)}")
                out_deeplab = torch.zeros((1,1,img.shape[-2], img.shape[-1]), device=DEVICE)

        # ensure single output channel and resize if necessary
        if out_unet.shape[1] != 1:
            out_unet = out_unet[:, :1, ...]
        if out_deeplab.shape[1] != 1:
            out_deeplab = out_deeplab[:, :1, ...]

        if out_unet.shape[-2:] != mask.shape[-2:]:
            out_unet = nn.functional.interpolate(out_unet, size=mask.shape[-2:], mode='bilinear', align_corners=False)
        if out_deeplab.shape[-2:] != mask.shape[-2:]:
            out_deeplab = nn.functional.interpolate(out_deeplab, size=mask.shape[-2:], mode='bilinear', align_corners=False)

        # probabilities
        prob_unet = torch.sigmoid(out_unet)
        prob_deeplab = torch.sigmoid(out_deeplab)

        # debug stats for this sample
        debug_lines.append(f"Sample: {fname}")
        debug_lines.append(f"  UNet logits min/max/mean: {float(out_unet.min()):.6f}/{float(out_unet.max()):.6f}/{float(out_unet.mean()):.6f}")
        debug_lines.append(f"  UNet probs min/max/mean:  {float(prob_unet.min()):.6f}/{float(prob_unet.max()):.6f}/{float(prob_unet.mean()):.6f}")
        debug_lines.append(f"  DeepLab logits min/max/mean: {float(out_deeplab.min()):.6f}/{float(out_deeplab.max()):.6f}/{float(out_deeplab.mean()):.6f}")
        debug_lines.append(f"  DeepLab probs min/max/mean:  {float(prob_deeplab.min()):.6f}/{float(prob_deeplab.max()):.6f}/{float(prob_deeplab.mean()):.6f}")

        # binarize
        pred_unet = (prob_unet > threshold).float()
        pred_deeplab = (prob_deeplab > threshold).float()

        # gather metrics
        p_u = pred_unet.cpu().numpy().astype(np.uint8)[0,0]
        p_d = pred_deeplab.cpu().numpy().astype(np.uint8)[0,0]
        g = mask.cpu().numpy().astype(np.uint8)[0,0]

        def metrics_from_arrays(p_arr, g_arr):
            flat_p = p_arr.flatten()
            flat_g = g_arr.flatten()
            try:
                acc = float(accuracy_score(flat_g, flat_p))
                prec = float(precision_score(flat_g, flat_p, zero_division=0))
                rec = float(recall_score(flat_g, flat_p, zero_division=0))
                f1 = float(f1_score(flat_g, flat_p, zero_division=0))
            except Exception:
                acc = prec = rec = f1 = 0.0
            dice = float(dice_coef(torch.tensor(p_arr), torch.tensor(g_arr)))
            iou = float(iou_coef(torch.tensor(p_arr), torch.tensor(g_arr)))
            return acc, prec, rec, f1, dice, iou

        acc_u, prec_u, rec_u, f1_u, dice_u, iou_u = metrics_from_arrays(p_u, g)
        acc_d, prec_d, rec_d, f1_d, dice_d, iou_d = metrics_from_arrays(p_d, g)

        results_unet.append({
            "image": fname,
            "accuracy": acc_u, "precision": prec_u, "recall": rec_u, "f1": f1_u, "dice": dice_u, "iou": iou_u
        })
        results_deeplab.append({
            "image": fname,
            "accuracy": acc_d, "precision": prec_d, "recall": rec_d, "f1": f1_d, "dice": dice_d, "iou": iou_d
        })

        # collect some samples for visualization (we will pick later)
        vis_samples.append({
            "name": fname,
            "img": img.cpu().numpy()[0,0],  # H,W
            "gt": g,
            "pred_unet": p_u,
            "pred_deeplab": p_d
        })

        # If debugging for a specific filename, include model numeric details
        if debug_for is not None and fname == debug_for.name:
            debug_lines.append("--- Detailed debug for provided image ---")
            debug_lines.append(f"Image: {debug_for}")
            debug_lines.append(f"UNet prob stats: min {float(prob_unet.min()):.6f}, max {float(prob_unet.max()):.6f}, mean {float(prob_unet.mean()):.6f}")
            debug_lines.append(f"DeepLab prob stats: min {float(prob_deeplab.min()):.6f}, max {float(prob_deeplab.max()):.6f}, mean {float(prob_deeplab.mean()):.6f}")

    # aggregate
    def aggregate(results_list):
        if len(results_list) == 0:
            return {k: 0.0 for k in ["accuracy","precision","recall","f1","dice","iou"]}
        agg = {
            "accuracy": float(np.mean([r["accuracy"] for r in results_list])),
            "precision": float(np.mean([r["precision"] for r in results_list])),
            "recall": float(np.mean([r["recall"] for r in results_list])),
            "f1": float(np.mean([r["f1"] for r in results_list])),
            "dice": float(np.mean([r["dice"] for r in results_list])),
            "iou": float(np.mean([r["iou"] for r in results_list]))
        }
        return agg

    # select n_samples random vis samples from vis_samples (clamped)
    random.shuffle(vis_samples)
    vis_choice = vis_samples[:min(len(vis_samples), n_samples)]

    # produce combined montages (one montage per chosen sample)
    vis_dir = Path(out_dir) / "vis_examples"
    ensure_dir(vis_dir)
    for vs in vis_choice:
        name = vs["name"]
        img = vs["img"]
        gt = vs["gt"]
        pu = vs["pred_unet"]
        pd = vs["pred_deeplab"]
        # build figure: Image | GT | UNet Pred | DeepLab Pred | Overlay UNet | Overlay DeepLab
        fig, axs = plt.subplots(1,6, figsize=(18,3))
        axs[0].imshow(img, cmap="gray"); axs[0].set_title("Image"); axs[0].axis("off")
        axs[1].imshow(gt, cmap="gray"); axs[1].set_title("GT Mask"); axs[1].axis("off")
        axs[2].imshow(pu, cmap="gray"); axs[2].set_title("UNet Pred"); axs[2].axis("off")
        axs[3].imshow(pd, cmap="gray"); axs[3].set_title("DeepLab Pred"); axs[3].axis("off")
        # overlays: colorize predictions vs GT
        overlay_u = np.stack([img, img, img], axis=-1).copy()
        overlay_u[pu > 0.5, 0] = 1.0
        overlay_u[gt > 0.5, 1] = 1.0
        axs[4].imshow(overlay_u); axs[4].set_title("Overlay UNet (R=pred,G=gt)"); axs[4].axis("off")

        overlay_d = np.stack([img, img, img], axis=-1).copy()
        overlay_d[pd > 0.5, 0] = 1.0
        overlay_d[gt > 0.5, 1] = 1.0
        axs[5].imshow(overlay_d); axs[5].set_title("Overlay DeepLab (R=pred,G=gt)"); axs[5].axis("off")

        plt.tight_layout()
        safe_name = "".join([c if c.isalnum() or c in "._-" else "_" for c in name])
        outfile = vis_dir / f"{safe_name}_comparison.png"
        plt.savefig(outfile, bbox_inches="tight")
        plt.close(fig)

    return {
        "unet": {"per_image": results_unet, "aggregate": aggregate(results_unet)},
        "deeplab": {"per_image": results_deeplab, "aggregate": aggregate(results_deeplab)},
        "debug_lines": debug_lines
    }

# ----------------------------
# Main orchestrator
# ----------------------------
def main():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(OUTPUT_ROOT) / f"comparison_{timestamp}"
    ensure_dir(out_dir)
    debug_file = out_dir / "debug_logs.txt"

    try:
        # load datasets
        print("Loading datasets...")
        ds_unet = SimpleLungDataset(UNET_DATA_ROOT, split="val")
        ds_deeplab = SimpleLungDataset(DEEPLAB_DATA_ROOT, split="val")
        print(f"UNet dataset size: {len(ds_unet)}")
        print(f"DeepLab dataset size: {len(ds_deeplab)}")

        # choose subset of indices to evaluate (shared across both datasets by filename intersection)
        # Create a list of filenames present in both datasets (so we can compare both models on same images)
        names_unet = {p[0].name: idx for idx, p in enumerate(ds_unet.pairs)}
        names_deeplab = {p[0].name: idx for idx, p in enumerate(ds_deeplab.pairs)}
        common_names = sorted(set(names_unet.keys()).intersection(set(names_deeplab.keys())))
        if len(common_names) == 0:
            # fallback: use whichever dataset has more
            print("Warning: no common filenames between UNet and DeepLab datasets. Comparing separately.")
            # use UNet dataset if available
            common_list = list(names_unet.keys()) if len(names_unet)>0 else list(names_deeplab.keys())
        else:
            common_list = common_names

        random.shuffle(common_list)
        subset_list = common_list[:min(len(common_list), EVAL_SUBSET)]

        # Build index lists for each dataset corresponding to chosen subset filenames
        subset_idx_unet = [names_unet[nm] for nm in subset_list if nm in names_unet]
        subset_idx_deeplab = [names_deeplab[nm] for nm in subset_list if nm in names_deeplab]

        # If there are no common items, just evaluate respective subsets separately
        evaluate_common = len(subset_idx_unet) > 0 and len(subset_idx_deeplab) > 0

        # DataLoaders not strictly needed for index-based access but create them for compatibility
        loader_unet = DataLoader(ds_unet, batch_size=1, shuffle=False)
        loader_deeplab = DataLoader(ds_deeplab, batch_size=1, shuffle=False)

        # Load models (try to import user's UNet class if exists)
        print("Loading UNet checkpoint...")
        unet_model = None
        # Try to import a UNet class from common filenames (unet_model.py, train_unet.py, model_unet.py)
        import importlib
        for candidate in ("unet_model", "model_unet", "train_unet"):
            try:
                mod = importlib.import_module(candidate)
                if hasattr(mod, "UNet"):
                    UnetClass = getattr(mod, "UNet")
                    unet_model = UnetClass(in_channels=1, out_channels=1) if "in_channels" in UnetClass.__init__.__code__.co_varnames else UnetClass()
                    print(f"Imported UNet class from {candidate}.py")
                    break
            except Exception:
                continue
        if unet_model is None:
            unet_model = SmallUNet(in_ch=1, out_ch=1)
            print("Using fallback SmallUNet.")

        unet = load_checkpoint(unet_model, UNET_CHECKPOINT)

        print("Loading DeepLab checkpoint...")
        deeplab = build_deeplab(num_classes=1)
        deeplab = load_checkpoint(deeplab, DEEPLAB_CHECKPOINT)

        # Evaluate (if we have common indices use the same images; else evaluate separately)
        results = {"unet": None, "deeplab": None, "summary": None}
        debug_lines = []

        if evaluate_common:
            print(f"Evaluating on {len(subset_idx_unet)} common samples (subset size capped to {EVAL_SUBSET})...")
            # Create unified loader for comparison by indexing into each dataset separately in eval function
            # We'll use ds_unet for fetching since subset indices are for that dataset; but we need both models to accept same image.
            # For simplicity, craft a temporary dataset that uses the filenames and their paths from ds_unet (which are also present in ds_deeplab)
            # Build a small wrapper dataset consisting of pairs of file paths from ds_unet where names in subset_list
            temp_pairs = [ds_unet.pairs[i] for i in subset_idx_unet]
            # Create a custom DataLoader that uses ds_unet indexing
            temp_loader = DataLoader(ds_unet, batch_size=1, shuffle=False)
            # Use the subset indexes directly
            eval_res = eval_models_on_subset(unet, deeplab, temp_loader, out_dir, subset_idx_unet, n_samples=N_SAMPLES, threshold=THRESHOLD, debug_for=DIAG_IMAGE_PATH if DIAG_IMAGE_PATH.exists() else None)
            results["unet"] = eval_res["unet"]
            results["deeplab"] = eval_res["deeplab"]
            debug_lines.extend(eval_res["debug_lines"])
        else:
            # Evaluate UNet on its own subset
            print("No common filenames. Evaluating UNet and DeepLab independently on their own subsets (clamped by EVAL_SUBSET).")
            idxs_u = list(range(min(len(ds_unet), EVAL_SUBSET)))
            idxs_d = list(range(min(len(ds_deeplab), EVAL_SUBSET)))
            res_u = eval_models_on_subset(unet, deeplab, DataLoader(ds_unet, batch_size=1), out_dir, idxs_u, n_samples=N_SAMPLES, threshold=THRESHOLD, debug_for=DIAG_IMAGE_PATH if DIAG_IMAGE_PATH.exists() else None)
            res_d = eval_models_on_subset(unet, deeplab, DataLoader(ds_deeplab, batch_size=1), out_dir, idxs_d, n_samples=N_SAMPLES, threshold=THRESHOLD, debug_for=DIAG_IMAGE_PATH if DIAG_IMAGE_PATH.exists() else None)
            results["unet"] = res_u["unet"]
            results["deeplab"] = res_d["deeplab"]
            debug_lines.extend(res_u["debug_lines"])
            debug_lines.extend(res_d["debug_lines"])

        # Save JSON report
        report = {
            "timestamp": timestamp,
            "device": str(DEVICE),
            "unet_checkpoint": str(Path(UNET_CHECKPOINT).resolve()),
            "deeplab_checkpoint": str(Path(DEEPLAB_CHECKPOINT).resolve()),
            "unet": results["unet"],
            "deeplab": results["deeplab"]
        }
        report_path = Path(out_dir) / "comparison_report.json"
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)

        # Save summary.json (top-level aggregates)
        summary = {
            "unet_mean": results["unet"]["aggregate"] if results["unet"] else {},
            "deeplab_mean": results["deeplab"]["aggregate"] if results["deeplab"] else {}
        }
        with open(Path(out_dir) / "summary.json", "w") as f:
            json.dump(summary, f, indent=2)

        # Save debug logs
        with open(debug_file, "w") as f:
            if debug_lines:
                f.write("\n".join(debug_lines))
            else:
                f.write("No debug lines collected.\n")

        # Make a bar chart comparing metrics (Dice, IoU, F1, Accuracy)
        metrics = ["dice", "iou", "f1", "accuracy"]
        unet_vals = [summary["unet_mean"].get(m, 0.0) for m in metrics]
        deeplab_vals = [summary["deeplab_mean"].get(m, 0.0) for m in metrics]
        x = np.arange(len(metrics))
        width = 0.35

        fig, ax = plt.subplots(figsize=(8,4))
        ax.bar(x - width/2, unet_vals, width)
        ax.bar(x + width/2, deeplab_vals, width)
        ax.set_xticks(x)
        ax.set_xticklabels(metrics)
        ax.set_ylabel("Score")
        ax.set_title("Model comparison")
        ax.legend(["UNet","DeepLab"])
        plt.tight_layout()
        plt_path = Path(out_dir) / "metrics_comparison.png"
        plt.savefig(plt_path)
        plt.close(fig)

        print(f"Evaluation finished. Results saved to: {out_dir}")
        print(f"- JSON report: {report_path}")
        print(f"- Summary: {Path(out_dir)/'summary.json'}")
        print(f"- Metrics plot: {plt_path}")
        print(f"- Visual examples: {Path(out_dir)/'vis_examples/'}")
        print(f"- Debug log: {debug_file}")

    except Exception as e:
        # write traceback to debug file and stderr
        tb = traceback.format_exc()
        with open(out_dir / "debug_exception.txt", "w") as f:
            f.write(tb)
        print("An exception occurred during evaluation. See debug_exception.txt in the results folder.")
        raise

if __name__ == "__main__":
    main()
