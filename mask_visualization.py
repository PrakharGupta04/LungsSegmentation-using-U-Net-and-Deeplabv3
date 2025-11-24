

import os
import cv2
import random
import matplotlib.pyplot as plt

# CONFIGURATION
SPLIT = "train"        
CLASS_NAME = "COVID"   


# PATH SETUP

img_dir = f"data/{SPLIT}/{CLASS_NAME}/images"
mask_dir = f"data/{SPLIT}/{CLASS_NAME}/masks"

# Ensure paths exist
if not os.path.exists(img_dir):
    raise FileNotFoundError(f" Image folder not found: {img_dir}")
if not os.path.exists(mask_dir):
    raise FileNotFoundError(f" Mask folder not found: {mask_dir}")

# PICK RANDOM IMAGE

file_list = os.listdir(img_dir)
if len(file_list) == 0:
    raise FileNotFoundError(f" No images found in {img_dir}")

random_img = "COVID-2.png"
img_path = os.path.join(img_dir, random_img)
mask_path = os.path.join(mask_dir, random_img)

print(f" Showing file: {random_img}")
print(f" Location: {img_path}")


# LOAD IMAGES

img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

if img is None or mask is None:
    raise ValueError(f" Could not load image or mask for {random_img}")


# VISUALIZE

plt.figure(figsize=(11, 4))
plt.suptitle(f"{CLASS_NAME} | {SPLIT} Split | File: {random_img}", fontsize=12, fontweight='bold')

# 1️ Original Image
plt.subplot(1, 3, 1)
plt.imshow(img, cmap='gray')
plt.title("X-ray Image")
plt.axis('off')

# 2️ Mask
plt.subplot(1, 3, 2)
plt.imshow(mask, cmap='gray')
plt.title("Lung Mask (Binary)")
plt.axis('off')

# 3️ Overlay
overlay = cv2.addWeighted(img, 0.7, mask, 0.3, 0)
plt.subplot(1, 3, 3)
plt.imshow(overlay, cmap='gray')
plt.title("Overlay (Image + Mask)")
plt.axis('off')

plt.tight_layout()
plt.show()
