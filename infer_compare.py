#!/usr/bin/env python3
"""
infer_compare.py

Run this script to perform inference on a single chest X-ray image using:
 - UNet checkpoint: checkpoints/best_model.pth
 - DeepLab checkpoint: checkpoints_deeplab/best_deeplab_refined.pth

Saves outputs under ./inference_results/<timestamp>/

Default sample input (from session): /mnt/data/016884af-7d2c-493c-b102-9e9549a0ee12.png
"""
import argparse
from pathlib import Path
from datetime import datetime
import json
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torchvision.transforms.functional as TF

# ----------------- Defaults (change if you want) -----------------
DEFAULT_IMAGE = "/mnt/data/016884af-7d2c-493c-b102-9e9549a0ee12.png"
UNET_CHECKPOINT = "checkpoints/best_model.pth"
DEEPLAB_CHECKPOINT = "checkpoints_deeplab/best_deeplab_refined.pth"
IMAGE_SIZE = 256
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
THRESHOLD_UNET = 0.5
THRESHOLD_DEEPLAB = 0.3
# -----------------------------------------------------------------

def ensure_dir(p):
    Path(p).mkdir(parents=True, exist_ok=True)

# small fallback UNet in case user's UNet class is not importable
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
    from torchvision.models.segmentation import deeplabv3_resnet50
    model = deeplabv3_resnet50(weights=None)
    try:
        last = list(model.classifier.children())[-1]
        if isinstance(last, nn.Conv2d):
            in_ch = last.in_channels
            model.classifier[-1] = nn.Conv2d(in_ch, num_classes, kernel_size=1)
    except Exception:
        pass
    return model

def load_unet_checkpoint(model, path):
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"UNet checkpoint not found: {p}")
    ck = torch.load(str(p), map_location="cpu")
    state = ck.get("model_state_dict", ck) if isinstance(ck, dict) else ck
    # strip module. prefix if present
    if any(k.startswith("module.") for k in state.keys()):
        state = {k.replace("module.",""):v for k,v in state.items()}
    model.load_state_dict(state, strict=False)
    return model

def load_deeplab_checkpoint(model, path):
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"DeepLab checkpoint not found: {p}")
    ck = torch.load(str(p), map_location="cpu")
    state = ck.get("model_state_dict", ck) if isinstance(ck, dict) else ck
    # clean prefixes
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

def preprocess_image(pth, size=256):
    img = Image.open(pth).convert("L").resize((size,size))
    arr = np.array(img).astype(np.float32) / 255.0
    t = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0).float()  # [1,1,H,W]
    return t, img

def adapt_input_channels(img_tensor, required_in_ch):
    # img_tensor: [B,C,H,W]
    c = img_tensor.shape[1]
    if c == required_in_ch:
        return img_tensor
    if c == 1 and required_in_ch == 3:
        return img_tensor.repeat(1,3,1,1)
    if c == 3 and required_in_ch == 1:
        return img_tensor.mean(dim=1, keepdim=True)
    if c < required_in_ch:
        return img_tensor.repeat(1, required_in_ch // c + 1, 1, 1)[:, :required_in_ch, ...]
    else:
        return img_tensor[:, :required_in_ch, ...]

def normalize_imagenet(img3):
    mean = torch.tensor([0.485,0.456,0.406], device=img3.device).view(1,3,1,1)
    std = torch.tensor([0.229,0.224,0.225], device=img3.device).view(1,3,1,1)
    return (img3 - mean) / std

def save_prob_mask_overlay(out_dir, name, orig_pil, prob_map, mask_bin, cmap="gray", alpha=0.5):
    # prob_map, mask_bin are numpy HxW arrays (0..1)
    out_dir = Path(out_dir)
    ensure_dir(out_dir)
    # save prob
    prob_img = (prob_map * 255).astype(np.uint8)
    Image.fromarray(prob_img).save(out_dir / f"{name}_prob.png")
    # save mask
    mask_img = (mask_bin * 255).astype(np.uint8)
    Image.fromarray(mask_img).save(out_dir / f"{name}_mask.png")
    # overlay: pred in red over grayscale orig
    fig, ax = plt.subplots(1,1,figsize=(4,4))
    ax.imshow(orig_pil, cmap="gray")
    ax.imshow(mask_bin, cmap="Reds", alpha=alpha)
    ax.axis("off")
    fig.savefig(out_dir / f"{name}_overlay.png", bbox_inches="tight", pad_inches=0)
    plt.close(fig)

def run_inference(image_path, out_root, show=False):
    out_root = Path(out_root)
    ensure_dir(out_root)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = out_root / f"run_{stamp}"
    ensure_dir(run_dir)

    print("Loading image:", image_path)
    img_t, orig_pil = preprocess_image(image_path, size=IMAGE_SIZE)
    img_t = img_t.to(DEVICE)

    # ---------------- UNet ----------------
    # Try to import user's UNet class if available
    unet_model = None
    try:
        import importlib
        mod = importlib.import_module("unet_model")  # tries unet_model.py
        if hasattr(mod, "UNet"):
            UnetClass = getattr(mod, "UNet")
            try:
                unet_model = UnetClass(in_channels=1, out_channels=1)
            except Exception:
                unet_model = UnetClass()
            print("Imported UNet from unet_model.py")
    except Exception:
        pass

    if unet_model is None:
        # fallback
        unet_model = SmallUNet(in_ch=1, out_ch=1)
        print("Using fallback SmallUNet.")

    # load checkpoint
    try:
        unet_model = load_unet_checkpoint(unet_model, UNET_CHECKPOINT)
        print("Loaded UNet checkpoint.")
    except Exception as e:
        print("Warning: failed to load UNet checkpoint:", e)
    unet_model = unet_model.to(DEVICE)
    unet_model.eval()

    # forward unet
    with torch.no_grad():
        # adapt channels if needed
        req_ch_unet = next(iter(unet_model.parameters())).shape[1] if any(True for _ in unet_model.parameters()) else 1
        input_unet = adapt_input_channels(img_t, req_ch_unet)
        out_u = unet_model(input_unet)
        if isinstance(out_u, dict) and "out" in out_u:
            out_u = out_u["out"]
        prob_u = torch.sigmoid(out_u)
        prob_u_np = prob_u.cpu().numpy()[0,0]
        mask_u = (prob_u_np > THRESHOLD_UNET).astype(np.uint8)

    save_prob_mask_overlay(run_dir, "unet", orig_pil, prob_u_np, mask_u)
    print("Saved UNet outputs to", run_dir)

    # ---------------- DeepLab ----------------
    deeplab = build_deeplab(num_classes=1)
    try:
        deeplab = load_deeplab_checkpoint(deeplab, DEEPLAB_CHECKPOINT)
        print("Loaded DeepLab checkpoint.")
    except Exception as e:
        print("Warning: failed to load DeepLab checkpoint:", e)
    deeplab = deeplab.to(DEVICE)
    deeplab.eval()

    with torch.no_grad():
        # DeepLab expects 3-channel normalized input
        input_dl = adapt_input_channels(img_t, 3)
        input_dl = normalize_imagenet(input_dl.to(DEVICE))
        out_d = deeplab(input_dl)
        if isinstance(out_d, dict) and "out" in out_d:
            out_d = out_d["out"]
        prob_d = torch.sigmoid(out_d)
        prob_d_np = prob_d.cpu().numpy()[0,0]
        mask_d = (prob_d_np > THRESHOLD_DEEPLAB).astype(np.uint8)

    save_prob_mask_overlay(run_dir, "deeplab", orig_pil, prob_d_np, mask_d)
    print("Saved DeepLab outputs to", run_dir)

    # Optionally display results inline (if show=True; useful in notebooks)
    if show:
        fig, axes = plt.subplots(1,3,figsize=(12,4))
        axes[0].imshow(orig_pil, cmap="gray"); axes[0].set_title("Image"); axes[0].axis("off")
        axes[1].imshow(prob_u_np, cmap="inferno"); axes[1].set_title("UNet prob"); axes[1].axis("off")
        axes[2].imshow(prob_d_np, cmap="inferno"); axes[2].set_title("DeepLab prob"); axes[2].axis("off")
        plt.show()

    # Save a small JSON summary
    summary = {
        "image": str(image_path),
        "unet": {
            "threshold": THRESHOLD_UNET,
            "mean_prob": float(prob_u_np.mean()),
            "max_prob": float(prob_u_np.max()),
            "mask_sum": int(mask_u.sum())
        },
        "deeplab": {
            "threshold": THRESHOLD_DEEPLAB,
            "mean_prob": float(prob_d_np.mean()),
            "max_prob": float(prob_d_np.max()),
            "mask_sum": int(mask_d.sum())
        },
        "output_folder": str(run_dir)
    }
    with open(run_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("Inference finished. Results in:", run_dir)
    return run_dir

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, default=DEFAULT_IMAGE, help="Path to input X-ray image")
    parser.add_argument("--out", type=str, default="inference_results", help="Output root folder")
    parser.add_argument("--show", action="store_true", help="Show quick preview (if running in notebook)")
    args = parser.parse_args()
    image_path = args.image
    run_dir = run_inference(image_path, args.out, show=args.show)
    print("Done. Check folder:", run_dir)

if __name__ == "__main__":
    main()
