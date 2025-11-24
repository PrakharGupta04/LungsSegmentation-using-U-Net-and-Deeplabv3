import argparse
from pathlib import Path
import random
import shutil
import math
from PIL import Image, ImageOps
import os

IMG_EXTS = {'.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp'}


def safe_clear_folder(path: Path):
    """Delete out_root if exists, then recreate."""
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def get_pairs(class_dir: Path):
    """Find image-mask pairs inside class/images and class/masks."""
    img_dir = class_dir / "images"
    mask_dir = class_dir / "masks"

    if not img_dir.exists() or not mask_dir.exists():
        return []

    img_files = [p for p in img_dir.iterdir() if p.suffix.lower() in IMG_EXTS]
    mask_files = [p for p in mask_dir.iterdir() if p.suffix.lower() in IMG_EXTS]

    mask_map = {m.stem: m for m in mask_files}
    pairs = [(im, mask_map[im.stem]) for im in img_files if im.stem in mask_map]
    return pairs


def save_pair(src_img, src_mask, dst_img, dst_mask):
    dst_img.parent.mkdir(parents=True, exist_ok=True)
    dst_mask.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src_img, dst_img)
    shutil.copy2(src_mask, dst_mask)


def augment(img_path, mask_path, dst_img, dst_mask, op):
    img = Image.open(img_path).convert("RGB")
    mask = Image.open(mask_path).convert("L")

    if op == 0:
        img_a = ImageOps.mirror(img)
        mask_a = ImageOps.mirror(mask)
    elif op == 1:
        img_a = ImageOps.flip(img)
        mask_a = ImageOps.flip(mask)
    elif op == 2:
        img_a = img.rotate(90, expand=True)
        mask_a = mask.rotate(90, expand=True)
    elif op == 3:
        img_a = img.rotate(270, expand=True)
        mask_a = mask.rotate(270, expand=True)
    else:
        img_a = img.transpose(Image.TRANSPOSE)
        mask_a = mask.transpose(Image.TRANSPOSE)

    dst_img.parent.mkdir(parents=True, exist_ok=True)
    dst_mask.parent.mkdir(parents=True, exist_ok=True)

    img_a.save(dst_img)
    mask_a.save(dst_mask)


def split_train_val(pairs, val_frac, seed):
    random.Random(seed).shuffle(pairs)
    n_val = int(math.ceil(len(pairs) * val_frac))
    return pairs[n_val:], pairs[:n_val]


def generate_name(cls, index):
    return f"{cls.lower()}_{index:04d}.png"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_root", required=True)
    parser.add_argument("--out_root", required=True)
    parser.add_argument("--target_per_class", type=int, default=500)
    parser.add_argument("--val_frac", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    dataset_root = Path(args.dataset_root)
    out_root = Path(args.out_root)
    target = args.target_per_class
    seed = args.seed

    # Overwrite output safely
    print(f"🧹 Clearing output folder: {out_root}")
    safe_clear_folder(out_root)

    class_dirs = [d for d in dataset_root.iterdir() if d.is_dir()]

    for class_dir in class_dirs:
        cls = class_dir.name
        print(f"\n➡ Class: {cls}")

        pairs = get_pairs(class_dir)
        total = len(pairs)
        print(f"  Found {total} pairs")

        need_aug = total < target

        # sample or keep all
        if need_aug:
            sampled = pairs
            print(f"  Will augment {cls}: {total} → {target}")
        else:
            sampled = random.Random(seed).sample(pairs, target)
            print(f"  Undersampled to {target}")

        # Split
        train_pairs, val_pairs = split_train_val(sampled, args.val_frac, seed)

        # Save originals (with sequential numbering)
        idx = 1
        for im, mask in train_pairs:
            name = generate_name(cls, idx)
            save_pair(
                im,
                mask,
                out_root / "train" / cls / "images" / name,
                out_root / "train" / cls / "masks" / name,
            )
            idx += 1

        for im, mask in val_pairs:
            name = generate_name(cls, idx)
            save_pair(
                im,
                mask,
                out_root / "val" / cls / "images" / name,
                out_root / "val" / cls / "masks" / name,
            )
            idx += 1

        # Augment
        if need_aug:
            aug_needed = target - len(sampled)
            print(f"  ✨ Generating {aug_needed} augmentations")

            base = train_pairs or sampled

            for i in range(aug_needed):
                src_im, src_mask = base[i % len(base)]
                name = generate_name(cls, idx)
                augment(
                    src_im,
                    src_mask,
                    out_root / "train" / cls / "images" / name,
                    out_root / "train" / cls / "masks" / name,
                    op=i,
                )
                idx += 1

    print("\n✅ DONE! Reduced dataset created at:", out_root)


if __name__ == "__main__":
    main()
