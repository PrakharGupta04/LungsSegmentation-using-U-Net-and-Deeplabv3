# preprocess.py
import os
import shutil
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from pathlib import Path

# CONFIG
SOURCE_DIR = "datasets"     # your raw datasets root (COVID, NORMAL, PNEUMONIA)
DEST_DIR = "data"           # preprocessed output (train/val/test)
IMAGE_SIZE = (256, 256)
TEST_SPLIT = 0.1
VAL_SPLIT = 0.1
APPLY_CLAHE = True
APPLY_MEDIAN_BLUR = True
APPLY_GAUSSIAN_BLUR = False
NORMALIZE = True
RANDOM_STATE = 42

# tolerant mask suffixes to try when matching
MASK_SUFFIXES = ["_mask", "-mask", "mask", "_seg", "-seg", "seg"]

def preprocess_image(img_path):
    img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Could not read image: {img_path}")
    if APPLY_MEDIAN_BLUR:
        img = cv2.medianBlur(img, 3)
    if APPLY_GAUSSIAN_BLUR:
        img = cv2.GaussianBlur(img, (3,3), 0)
    if APPLY_CLAHE:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        img = clahe.apply(img)
    img = cv2.resize(img, IMAGE_SIZE, interpolation=cv2.INTER_AREA)
    if NORMALIZE:
        img = img.astype(np.float32) / 255.0
        img = (img * 255).astype(np.uint8)  # keep as uint8 on disk but processed numerical range used later in loader
    return img

def preprocess_mask(mask_path):
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise ValueError(f"Could not read mask: {mask_path}")
    mask = cv2.resize(mask, IMAGE_SIZE, interpolation=cv2.INTER_NEAREST)
    _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    return mask

def find_matching_mask(img_path, mask_files_set):
    """
    Try tolerant matching strategies:
    1) exact stem match (same base filename without extension)
    2) image stem + common mask suffixes
    3) substring match: if mask filename contains image stem (dangerous fallback)
    """
    img_stem = img_path.stem  # filename without extension
    # 1) exact stem match
    for ext in ['.png', '.jpg', '.jpeg', '.tif', '.bmp']:
        candidate = f"{img_stem}{ext}"
        if candidate in mask_files_set:
            return candidate
    # 2) try suffix variations
    for suf in MASK_SUFFIXES:
        cand = f"{img_stem}{suf}"
        for ext in ['.png', '.jpg', '.jpeg', '.tif', '.bmp']:
            candidate = f"{cand}{ext}"
            if candidate in mask_files_set:
                return candidate
    # 3) substring match (less strict) - choose first mask that contains stem
    for m in mask_files_set:
        if img_stem in Path(m).stem:
            return m
    return None

def prepare_class(class_name, src_root, dest_root):
    src_images_dir = Path(src_root) / class_name / "images"
    src_masks_dir = Path(src_root) / class_name / "masks"

    if not src_images_dir.exists() or not src_masks_dir.exists():
        print(f"Missing images or masks for class {class_name}. Skipping.")
        return {"class": class_name, "matched": 0, "unmatched_images": [], "unmatched_masks": []}

    all_images = sorted([p for p in src_images_dir.iterdir() if p.is_file()])
    all_masks = sorted([p for p in src_masks_dir.iterdir() if p.is_file()])

    mask_names_set = {p.name for p in all_masks}
    matched_pairs = []
    unmatched_images = []
    matched_masks = set()

    for img_path in all_images:
        match_name = find_matching_mask(img_path, mask_names_set)
        if match_name is None:
            unmatched_images.append(str(img_path.name))
        else:
            matched_pairs.append((img_path, src_masks_dir / match_name))
            matched_masks.add(match_name)

    # any masks not matched?
    unmatched_masks = [m for m in mask_names_set if m not in matched_masks]

    # create destination folders
    images_out_dir = Path(dest_root) / "images_tmp" / class_name
    masks_out_dir = Path(dest_root) / "masks_tmp" / class_name
    images_out_dir.mkdir(parents=True, exist_ok=True)
    masks_out_dir.mkdir(parents=True, exist_ok=True)

    # preprocess and save matched pairs
    for img_path, mask_path in matched_pairs:
        try:
            img_proc = preprocess_image(img_path)
            mask_proc = preprocess_mask(mask_path)
            out_img_name = img_path.name
            out_mask_name = mask_path.name
            cv2.imwrite(str(images_out_dir / out_img_name), img_proc)
            cv2.imwrite(str(masks_out_dir / out_mask_name), mask_proc)
        except Exception as e:
            print(f"Error processing pair {img_path} / {mask_path}: {e}")

    return {
        "class": class_name,
        "matched": len(matched_pairs),
        "unmatched_images": unmatched_images,
        "unmatched_masks": unmatched_masks
    }

def split_and_move(dest_root, output_root):
    """
    Takes the tmp images/masks folders created and splits them into train/val/test structure.
    """
    tmp_images_root = Path(dest_root) / "images_tmp"
    tmp_masks_root = Path(dest_root) / "masks_tmp"

    if not tmp_images_root.exists():
        print("No preprocessed data found to split.")
        return

    for class_dir in sorted(tmp_images_root.iterdir()):
        if not class_dir.is_dir(): 
            continue
        class_name = class_dir.name
        image_files = sorted(list(class_dir.iterdir()))
        mask_dir = tmp_masks_root / class_name
        mask_files = sorted(list(mask_dir.iterdir())) if mask_dir.exists() else []

        # Build mapping by filename (ensured in prepare_class)
        X = image_files
        y = mask_files

        if len(X) == 0:
            continue

        # use sklearn split
        X_train, X_rest, y_train, y_rest = train_test_split(X, y, test_size=(TEST_SPLIT+VAL_SPLIT), random_state=RANDOM_STATE)
        # split X_rest into val and test
        if TEST_SPLIT + VAL_SPLIT > 0:
            rel_test = TEST_SPLIT / (TEST_SPLIT + VAL_SPLIT)
            X_val, X_test, y_val, y_test = train_test_split(X_rest, y_rest, test_size=rel_test, random_state=RANDOM_STATE)
        else:
            X_val, X_test, y_val, y_test = [], [], [], []

        groups = {
            "train": (X_train, y_train),
            "val": (X_val, y_val),
            "test": (X_test, y_test)
        }

        for split_name, (imgs, masks) in groups.items():
            imgs_out = Path(output_root) / split_name / class_name / "images"
            masks_out = Path(output_root) / split_name / class_name / "masks"
            imgs_out.mkdir(parents=True, exist_ok=True)
            masks_out.mkdir(parents=True, exist_ok=True)
            for im, ma in zip(imgs, masks):
                shutil.move(str(im), str(imgs_out / im.name))
                shutil.move(str(ma), str(masks_out / ma.name))

def main():
    src_root = Path(SOURCE_DIR)
    out_root = Path(DEST_DIR)

    # clear destination
    if out_root.exists():
        shutil.rmtree(out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    class_dirs = [d.name for d in src_root.iterdir() if d.is_dir()]
    report = []
    for c in class_dirs:
        print(f"Processing class: {c}")
        r = prepare_class(c, SOURCE_DIR, DEST_DIR)
        report.append(r)
        print(f"  Matched pairs: {r['matched']}")
        if r['unmatched_images']:
            print(f"  Unmatched images ({len(r['unmatched_images'])}): {r['unmatched_images'][:5]}{'...' if len(r['unmatched_images'])>5 else ''}")
        if r['unmatched_masks']:
            print(f"  Unmatched masks ({len(r['unmatched_masks'])}): {r['unmatched_masks'][:5]}{'...' if len(r['unmatched_masks'])>5 else ''}")

    # split and move into train/val/test
    split_and_move(DEST_DIR, DEST_DIR)

    print("\n=== PREPROCESSING SUMMARY ===")
    for r in report:
        print(f"{r['class']}: matched={r['matched']}, unmatched_images={len(r['unmatched_images'])}, unmatched_masks={len(r['unmatched_masks'])}")

    print("\nPreprocessing complete. Preprocessed dataset available in the 'data/' folder (train/val/test).")

if __name__ == "__main__":
    main()
