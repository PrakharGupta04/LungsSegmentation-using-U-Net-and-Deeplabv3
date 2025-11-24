# check_deeplab_state.py
import torch
from torchvision.models.segmentation import deeplabv3_resnet50
import torch.nn as nn
from pathlib import Path

CKPT = "checkpoints_deeplab/best_deeplab.pth"

def build_deeplab_for_check(num_classes=1):
    model = deeplabv3_resnet50(weights=None)
    # try to make classifier final conv have out_channels=num_classes if present
    try:
        last = list(model.classifier.children())[-1]
        if isinstance(last, nn.Conv2d):
            in_ch = last.in_channels
            model.classifier[-1] = nn.Conv2d(in_ch, num_classes, kernel_size=1)
    except Exception:
        pass
    return model

ckpt = torch.load(CKPT, map_location="cpu")
print("Loaded checkpoint keys:", list(ckpt.keys()))
state = ckpt.get("model_state_dict", ckpt)
print("Number of state_dict keys:", len(state))

# print first 30 keys + shapes
print("\nFirst 30 state_dict keys and shapes:")
for i,(k,v) in enumerate(state.items()):
    print(f"{i:02d}: {k} -> {tuple(v.shape)}")
    if i >= 29:
        break

# try strict load to see missing/unexpected keys
model = build_deeplab_for_check(num_classes=1)
res = model.load_state_dict(state, strict=False)  # load non-strict so we capture mismatches
print("\nload_state_dict(strict=False) returned:")
print(res)  # a NamedTuple with missing_keys and unexpected_keys

# also show if any keys start with 'module.'
has_module_prefix = any(k.startswith("module.") for k in state.keys())
print("\nAny 'module.' prefixes present in keys?", has_module_prefix)
