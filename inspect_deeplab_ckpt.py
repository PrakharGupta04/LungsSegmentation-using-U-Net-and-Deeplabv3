# import torch

# ckpt = torch.load("checkpoints_deeplab/best_deeplab.pth", map_location="cpu")
# print("Type:", type(ckpt))
# print("Length:", len(ckpt))
# print("Keys:", list(ckpt.keys())[:50])

import torch
from torchvision.models.segmentation import deeplabv3_resnet50
import torch.nn as nn

ckpt = torch.load("checkpoints_deeplab/best_deeplab.pth", map_location="cpu")
state = ckpt["model_state_dict"]

model = deeplabv3_resnet50(weights=None)
# match your classifier (you trained with 1 output channel)
model.classifier[-1] = nn.Conv2d(256, 1, kernel_size=1)

res = model.load_state_dict(state, strict=False)
print(res)
