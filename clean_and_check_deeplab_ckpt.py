# clean_and_check_deeplab_ckpt.py
import torch
from pathlib import Path
from torchvision.models.segmentation import deeplabv3_resnet50
import torch.nn as nn
import sys

ORIG = Path("checkpoints_deeplab/best_deeplab.pth")
CLEAN = Path("checkpoints_deeplab/best_deeplab_clean.pth")
BACKUP = Path("checkpoints_deeplab/best_deeplab_backup.pth")

if not ORIG.exists():
    print("Original checkpoint not found:", ORIG)
    sys.exit(1)

print("Loading original checkpoint...")
ckpt = torch.load(str(ORIG), map_location="cpu")
# obtain the state dict (handles different checkpoint formats)
state = ckpt.get("model_state_dict", ckpt) if isinstance(ckpt, dict) else ckpt

# back up original (only if backup doesn't already exist)
if not BACKUP.exists():
    print("Saving a backup to", BACKUP)
    torch.save(ckpt, str(BACKUP))

print("First few keys in original state dict:")
for i,k in enumerate(list(state.keys())[:20]):
    print(i, k)
print("... total keys:", len(state))

# Build cleaned state dict: remove leading "model." and "module." if present
clean_state = {}
for k,v in state.items():
    new_k = k
    if new_k.startswith("model."):
        new_k = new_k[len("model."):]
    if new_k.startswith("module."):
        new_k = new_k[len("module."):]
    clean_state[new_k] = v

print("\nFirst few keys after cleaning:")
for i,k in enumerate(list(clean_state.keys())[:20]):
    print(i, k)
print("... total cleaned keys:", len(clean_state))

# Save cleaned checkpoint (preserve same outer structure if possible)
if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
    new_ckpt = ckpt.copy()
    new_ckpt["model_state_dict"] = clean_state
else:
    new_ckpt = clean_state

torch.save(new_ckpt, str(CLEAN))
print("\nSaved cleaned checkpoint to", CLEAN)

# Now test loading the cleaned state into a deeplab model
print("\nBuilding a fresh DeepLab model and attempting to load the cleaned state...")
model = deeplabv3_resnet50(weights=None)
# ensure classifier final conv has out_channels=1
try:
    last = list(model.classifier.children())[-1]
    if isinstance(last, nn.Conv2d):
        in_ch = last.in_channels
        model.classifier[-1] = nn.Conv2d(in_ch, 1, kernel_size=1)
except Exception:
    pass

res = model.load_state_dict(clean_state, strict=False)
print("\nResult of load_state_dict(strict=False):")
print(res)  # shows missing_keys and unexpected_keys

# If load looks OK, optionally save the model-only state dict cleaned
print("\nIf missing_keys is empty or only contains small unrelated items, try inference next.")
