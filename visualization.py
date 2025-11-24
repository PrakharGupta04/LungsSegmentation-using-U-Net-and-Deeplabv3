import os
import cv2
import random
import matplotlib.pyplot as plt
import numpy as np

BASE_DIR = "data/train"

# Pick a random class (e.g., COVID, Normal)
classes = [c for c in os.listdir(BASE_DIR) if os.path.isdir(os.path.join(BASE_DIR, c))]
chosen_class = random.choice(classes)
image_dir = os.path.join(BASE_DIR, chosen_class, "images")

# Pick a random image from that class
image_files = [f for f in os.listdir(image_dir) if f.lower().endswith((".png", ".jpg", ".jpeg"))]
if not image_files:
    print(f"⚠️ No image files found in {image_dir}")
    exit()

chosen_file = random.choice(image_files)
sample_image_path = os.path.join(image_dir, chosen_file)
original = cv2.imread(sample_image_path, cv2.IMREAD_GRAYSCALE)

if original is None:
    print(f" Could not load {sample_image_path}")
    exit()

print(f" Visualizing: {sample_image_path}")
print(f" Class Label: {chosen_class}")

# Step 1️ — Resize
resized = cv2.resize(original, (256, 256))

# Step 2️ — Median Blur (noise removal)
median_blurred = cv2.medianBlur(resized, 3)

# Step 3️ — CLAHE (contrast enhancement)
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
clahe_applied = clahe.apply(median_blurred)

# Step 4️ — Normalize [0, 1]
normalized = clahe_applied.astype(np.float32) / 255.0

# --- Display ---
titles = [
    "Original",
    "After Median Blur (Noise Removed)",
    "After CLAHE (Contrast Enhanced)",
    "Final Normalized Image"
]
images = [resized, median_blurred, clahe_applied, normalized]

plt.figure(figsize=(12, 6))
for i in range(4):
    plt.subplot(1, 4, i + 1)
    plt.imshow(images[i], cmap='gray')
    plt.title(titles[i], fontsize=9)
    plt.axis('off')

plt.suptitle(f"Preprocessing Pipeline — {chosen_class}\nFile: {chosen_file}", fontsize=12)
plt.tight_layout()
plt.show()
