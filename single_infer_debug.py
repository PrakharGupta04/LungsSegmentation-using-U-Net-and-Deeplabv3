# single_infer_debug_norm.py
import torch, numpy as np
from PIL import Image
from torchvision.models.segmentation import deeplabv3_resnet50
import torch.nn as nn
from pathlib import Path
import imageio
import sys

# EDIT THIS: point to one concrete image file (include extension)
img_path = Path("data/val/COVID/images/COVID-1.png")   # <-- change to your exact filename if needed
ckpt_deeplab = "checkpoints_deeplab/best_deeplab_clean.pth"

ckpt_unet = "checkpoints/best_model.pth"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_img(p):
    im = Image.open(p).convert("L").resize((256,256))
    arr = np.array(im).astype(np.float32)/255.0
    t = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0).float().to(device)  # [1,1,H,W]
    return t

def build_deeplab():
    m = deeplabv3_resnet50(weights=None)
    try:
        last = list(m.classifier.children())[-1]
        if isinstance(last, nn.Conv2d):
            in_ch = last.in_channels
            m.classifier[-1] = nn.Conv2d(in_ch, 1, 1)
    except Exception:
        pass
    return m

# load image
if not img_path.exists():
    print("ERROR: image not found:", img_path)
    sys.exit(1)
img = load_img(img_path)

# load UNet (try to import project's UNet, fallback to small UNet)
unet = None
try:
    import importlib
    mod = importlib.import_module("unet_model")
    if hasattr(mod, "UNet"):
        UnetClass = getattr(mod, "UNet")
        # try sensible init
        try:
            unet = UnetClass(in_channels=1, out_channels=1)
        except Exception:
            unet = UnetClass()
except Exception:
    pass

if unet is None:
    # small UNet fallback
    class SmallUNet(nn.Module):
        def __init__(self, in_ch=1, out_ch=1):
            super().__init__()
            def C(in_c, out_c):
                return nn.Sequential(nn.Conv2d(in_c,out_c,3,padding=1), nn.ReLU(),
                                     nn.Conv2d(out_c,out_c,3,padding=1), nn.ReLU())
            self.enc1 = C(in_ch, 32); self.pool = nn.MaxPool2d(2)
            self.enc2 = C(32,64); self.up1 = nn.ConvTranspose2d(64,32,2,2)
            self.dec1 = C(64,32); self.final = nn.Conv2d(32,out_ch,1)
        def forward(self,x):
            e1 = self.enc1(x); e2 = self.enc2(self.pool(e1))
            u1 = self.up1(e2); d1 = self.dec1(torch.cat([u1,e1],dim=1))
            return self.final(d1)
    unet = SmallUNet()

# load checkpoints (strip module. if needed)
def load_ckpt_into(model, path):
    ck = torch.load(path, map_location="cpu")
    state = ck.get("model_state_dict", ck) if isinstance(ck, dict) else ck
    # strip module. prefix if present
    if any(k.startswith("module.") for k in state.keys()):
        state = {k.replace("module.",""):v for k,v in state.items()}
    model.load_state_dict(state, strict=False)
    model.to(device); model.eval()
    return model

# load models
try:
    unet = load_ckpt_into(unet, ckpt_unet)
except Exception as e:
    print("Warning loading UNet checkpoint:", e)

deeplab = build_deeplab()
try:
    deeplab = load_ckpt_into(deeplab, ckpt_deeplab)
except Exception as e:
    print("Warning loading DeepLab checkpoint:", e)

# Prepare inputs
# UNet: keep single-channel (0-1)
img_unet = img.to(device)               # [1,1,256,256]

# DeepLab: repeat to 3 channels and apply ImageNet normalization
img3 = img.repeat(1,3,1,1).to(device)   # [1,3,256,256]
mean = torch.tensor([0.485,0.456,0.406], device=device).view(1,3,1,1)
std  = torch.tensor([0.229,0.224,0.225], device=device).view(1,3,1,1)
img3_norm = (img3 - mean) / std

# Forward and print stats
with torch.no_grad():
    out_u = unet(img_unet)
    out_d = deeplab(img3_norm)
    if isinstance(out_d, dict) and 'out' in out_d:
        out_d = out_d['out']

pu = torch.sigmoid(out_u)
pd = torch.sigmoid(out_d)

print("UNet logits min/max/mean:", float(out_u.min()), float(out_u.max()), float(out_u.mean()))
print("UNet probs min/max/mean: ", float(pu.min()), float(pu.max()), float(pu.mean()))
print("DeepLab logits min/max/mean:", float(out_d.min()), float(out_d.max()), float(out_d.mean()))
print("DeepLab probs min/max/mean: ", float(pd.min()), float(pd.max()), float(pd.mean()))

# write prob maps
imageio.imwrite("debug_unet_prob.png", (pu.cpu().numpy()[0,0]*255).astype('uint8'))
imageio.imwrite("debug_deeplab_prob.png", (pd.cpu().numpy()[0,0]*255).astype('uint8'))
print("Saved debug_unet_prob.png and debug_deeplab_prob.png")
