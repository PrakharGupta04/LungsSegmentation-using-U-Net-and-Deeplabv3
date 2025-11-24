import cv2
import numpy as np
from pathlib import Path
import shutil

# ======================
# CONFIG — PRESET PATHS
# ======================
ROOT = Path(__file__).parent
PNEUMONIA_DIR = ROOT / "datasets" / "pneumonia"
IMAGES_DIR = PNEUMONIA_DIR / "images"
MASKS_DIR = PNEUMONIA_DIR / "masks"

OUT_IMAGES = PNEUMONIA_DIR / "images_clean"
OUT_MASKS  = PNEUMONIA_DIR / "masks_clean"

RESIZE_TO = 256  # matching resolution
MASK_SUFFIXES = ["_mask", "-mask", "mask", "_seg", "-seg", "seg"]


# ======================
# IMAGE LOADING HELPERS
# ======================
def load_gray(path):
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise RuntimeError(f"Cannot read: {path}")
    img = cv2.resize(img, (RESIZE_TO, RESIZE_TO), interpolation=cv2.INTER_AREA)
    return img.astype(np.float32) / 255.0


def load_mask(path):
    m = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if m is None:
        raise RuntimeError(f"Cannot read mask: {path}")
    m = cv2.resize(m, (RESIZE_TO, RESIZE_TO), interpolation=cv2.INTER_NEAREST)
    _, m = cv2.threshold(m, 127, 1, cv2.THRESH_BINARY)
    return m.astype(np.float32)


def compute_score(img, mask):
    s = (img * mask).sum()
    denom = mask.sum()
    if denom <= 0:
        return -1.0
    return float(s / (denom + 1e-9))


# ======================
# MATCHING FUNCTION
# ======================
def greedy_match(score_mat):
    scores = score_mat.copy()
    pairs = []
    n, m = scores.shape
    while True:
        i, j = np.unravel_index(np.argmax(scores), scores.shape)
        if scores[i, j] <= 0 or scores[i, j] == -np.inf:
            break
        pairs.append((i, j, scores[i, j]))
        scores[i, :] = -np.inf
        scores[:, j] = -np.inf
    return pairs


# ======================
# MAIN SCRIPT
# ======================
def main():
    print("=== LUNGSEG — PNEUMONIA AUTO-RENAMING ===")

    if not IMAGES_DIR.exists() or not MASKS_DIR.exists():
        print("❌ ERROR: pneumonia/images or pneumonia/masks folder not found!")
        return

    image_files = sorted([p for p in IMAGES_DIR.iterdir() if p.is_file()])
    mask_files  = sorted([p for p in MASKS_DIR.iterdir() if p.is_file()])

    print(f"Found {len(image_files)} images and {len(mask_files)} masks.")

    # Backup original
    if not (IMAGES_DIR.parent / "images_backup").exists():
        shutil.copytree(IMAGES_DIR, IMAGES_DIR.parent / "images_backup")
        print("✔ Backup created: pneumonia/images_backup")

    if not (MASKS_DIR.parent / "masks_backup").exists():
        shutil.copytree(MASKS_DIR, MASKS_DIR.parent / "masks_backup")
        print("✔ Backup created: pneumonia/masks_backup")

    print("Loading images...")
    imgs = [load_gray(p) for p in image_files]

    print("Loading masks...")
    masks = [load_mask(p) for p in mask_files]

    n, m = len(imgs), len(masks)
    score_mat = np.zeros((n, m), dtype=np.float32)

    print("Computing match scores...")
    for i in range(n):
        for j in range(m):
            score_mat[i, j] = compute_score(imgs[i], masks[j])

    print("Assigning best pairs...")
    assignments = greedy_match(score_mat)

    print(f"✔ Found {len(assignments)} image–mask matches.")

    OUT_IMAGES.mkdir(exist_ok=True)
    OUT_MASKS.mkdir(exist_ok=True)

    # Sort by image filename for stable numbering
    assignments_sorted = sorted(assignments, key=lambda x: image_files[x[0]].name)

    used_imgs = set()
    used_masks = set()
    count = 1

    for (i, j, score) in assignments_sorted:
        if i in used_imgs or j in used_masks:
            continue
        used_imgs.add(i)
        used_masks.add(j)

        new_name = f"pneumonia_{count:03d}.bmp"

        shutil.copy2(image_files[i], OUT_IMAGES / new_name)
        shutil.copy2(mask_files[j],  OUT_MASKS  / new_name)

        count += 1

    unmatched_images = [image_files[i].name for i in range(n) if i not in used_imgs]
    unmatched_masks  = [mask_files[j].name  for j in range(m) if j not in used_masks]

    print("\n=== FINAL REPORT ===")
    print(f"✔ Total matched pairs: {len(used_imgs)}")
    print(f"• Unmatched images: {len(unmatched_images)}")
    print(f"• Unmatched masks:  {len(unmatched_masks)}")

    print("\nRenamed files saved in:")
    print(f"  → {OUT_IMAGES}")
    print(f"  → {OUT_MASKS}")

    print("\nNext step:")
    print("1️⃣ Delete old pneumonia/images and pneumonia/masks")
    print("2️⃣ Rename images_clean → images")
    print("3️⃣ Rename masks_clean → masks")
    print("4️⃣ Run: python preprocess.py")


if __name__ == "__main__":
    main()
