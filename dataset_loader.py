# # dataset_loader.py
# """
# Robust dataset loader for LungsSeg (final version)
# - Safe defaults for Windows (num_workers=0)
# - Optional cache_images to preload dataset into RAM (use only if you have memory)
# - Strict mask binarization and dtype handling
# - Lightweight augmentations by default; heavy augment off by default
# """

# from pathlib import Path
# import cv2
# import numpy as np
# import torch
# from torch.utils.data import Dataset, DataLoader, ConcatDataset
# import albumentations as A
# from albumentations.pytorch import ToTensorV2
# from typing import Tuple, List

# IMG_SIZE = (256, 256)

# def get_transforms(train: bool = True, heavy: bool = False):
#     if train:
#         if heavy:
#             return A.Compose([
#                 A.HorizontalFlip(p=0.5),
#                 A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=10, p=0.5),
#                 A.RandomBrightnessContrast(p=0.5),
#                 A.Normalize(mean=(0.5,), std=(0.5,)),
#                 ToTensorV2(),
#             ])
#         else:
#             return A.Compose([
#                 A.HorizontalFlip(p=0.5),
#                 A.Normalize(mean=(0.5,), std=(0.5,)),
#                 ToTensorV2(),
#             ])
#     else:
#         return A.Compose([
#             A.Normalize(mean=(0.5,), std=(0.5,)),
#             ToTensorV2(),
#         ])


# class LungSegDataset(Dataset):
#     def __init__(self,
#                  images_dir: str,
#                  masks_dir: str,
#                  transforms=None,
#                  image_size: Tuple[int,int]=IMG_SIZE,
#                  cache_images: bool=False):
#         self.images_dir = Path(images_dir)
#         self.masks_dir = Path(masks_dir)
#         self.transforms = transforms
#         self.image_size = image_size
#         self.cache_images = cache_images

#         image_files = sorted([p for p in self.images_dir.iterdir() if p.is_file()])
#         mask_files = sorted([p for p in self.masks_dir.iterdir() if p.is_file()])
#         masks_map = {p.name: p for p in mask_files}

#         self.pairs = []
#         unmatched = []
#         for img in image_files:
#             if img.name in masks_map:
#                 self.pairs.append((img, masks_map[img.name] ))
#                 continue
#             # tolerant matching
#             stem = img.stem
#             found = None
#             for suf in ["_mask","-mask","mask","_seg","-seg","seg"]:
#                 for ext in ['.png','.jpg','.jpeg','.tif','.bmp','.bmp']:
#                     candidate = f"{stem}{suf}{ext}"
#                     if candidate in masks_map:
#                         found = masks_map[candidate]; break
#                 if found: break
#             if not found:
#                 for mname, mpath in masks_map.items():
#                     if stem in Path(mname).stem:
#                         found = mpath; break
#             if found:
#                 self.pairs.append((img, found))
#             else:
#                 unmatched.append(img.name)

#         if len(unmatched) > 0:
#             print(f"[dataset_loader] Warning: {len(unmatched)} images have no matched masks. Examples: {unmatched[:5]}")

#         self._cache = {}
#         if self.cache_images:
#             print("[dataset_loader] Caching images into RAM (this may use lots of RAM).")
#             for img_p, mask_p in self.pairs:
#                 self._cache[img_p.name] = (self._read_image(img_p), self._read_mask(mask_p))

#     def __len__(self):
#         return len(self.pairs)

#     def _read_image(self, path: Path):
#         img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
#         if img is None:
#             raise IOError(f"Failed to read image {path}")
#         img = cv2.resize(img, self.image_size, interpolation=cv2.INTER_AREA)
#         return img

#     def _read_mask(self, path: Path):
#         mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
#         if mask is None:
#             raise IOError(f"Failed to read mask {path}")
#         mask = cv2.resize(mask, self.image_size, interpolation=cv2.INTER_NEAREST)
#         _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
#         return mask

#     def __getitem__(self, idx):
#         img_path, mask_path = self.pairs[idx]
#         if self.cache_images and img_path.name in self._cache:
#             img, mask = self._cache[img_path.name]
#         else:
#             img = self._read_image(img_path)
#             mask = self._read_mask(mask_path)

#         # HWC single channel
#         img_hwc = np.expand_dims(img, axis=-1)    # H,W,1
#         mask_hwc = np.expand_dims(mask, axis=-1)  # H,W,1

#         if self.transforms:
#             aug = self.transforms(image=img_hwc, mask=mask_hwc)
#             img_aug = aug['image']
#             mask_aug = aug['mask']
#         else:
#             img_aug = (img_hwc.astype('float32') / 255.0)
#             mask_aug = (mask_hwc.astype('float32') / 255.0)

#         # Normalize types and shapes -> image: [C,H,W], mask: [1,H,W], float32
#         # Handle numpy or torch outputs robustly
#         if isinstance(img_aug, np.ndarray):
#             img_np = img_aug.astype('float32') / 255.0 if img_aug.max() > 1.0 else img_aug.astype('float32')
#             img_tensor = torch.from_numpy(img_np).permute(2,0,1).contiguous().float()
#         elif isinstance(img_aug, torch.Tensor):
#             img_tensor = img_aug.float()
#             if img_tensor.ndim == 2:
#                 img_tensor = img_tensor.unsqueeze(0)
#             elif img_tensor.ndim == 3 and img_tensor.shape[0] not in (1,3):
#                 img_tensor = img_tensor.permute(2,0,1).contiguous()
#         else:
#             raise TypeError("Unsupported image type from augmentation")

#         if isinstance(mask_aug, np.ndarray):
#             mask_np = (mask_aug.astype('float32') / 255.0) if mask_aug.max() > 1.0 else mask_aug.astype('float32')
#             if mask_np.ndim == 2:
#                 mask_tensor = torch.from_numpy(mask_np).unsqueeze(0).contiguous().float()
#             elif mask_np.ndim == 3 and mask_np.shape[-1] == 1:
#                 mask_tensor = torch.from_numpy(mask_np).transpose(2,0,1).contiguous().float()
#             else:
#                 mask_tensor = torch.from_numpy(mask_np).permute(2,0,1).contiguous().float()
#         elif isinstance(mask_aug, torch.Tensor):
#             mask_tensor = mask_aug.float()
#             if mask_tensor.ndim == 2:
#                 mask_tensor = mask_tensor.unsqueeze(0)
#             elif mask_tensor.ndim == 3 and mask_tensor.shape[0] not in (1,):
#                 if mask_tensor.shape[2] == 1:
#                     mask_tensor = mask_tensor.permute(2,0,1).contiguous()
#                 else:
#                     mask_tensor = mask_tensor.permute(2,0,1).contiguous()
#         else:
#             raise TypeError("Unsupported mask type from augmentation")

#         # Binarize and ensure only 0/1
#         mask_tensor = (mask_tensor > 0.5).float()
#         # final sanity checks
#         if not torch.isfinite(img_tensor).all():
#             raise ValueError("Non-finite values in image tensor")
#         if not torch.isfinite(mask_tensor).all():
#             raise ValueError("Non-finite values in mask tensor")

#         return img_tensor, mask_tensor


# def _make_concat_datasets(split_dir: Path, transform, cache_images=False):
#     datasets = []
#     if not split_dir.exists():
#         return datasets
#     for class_dir in sorted(split_dir.iterdir()):
#         images_dir = class_dir / "images"
#         masks_dir = class_dir / "masks"
#         if images_dir.exists() and masks_dir.exists():
#             datasets.append(LungSegDataset(str(images_dir), str(masks_dir), transforms=transform, cache_images=cache_images))
#     return datasets


# def get_dataloaders(data_root="data", batch_size=8, num_workers=0, pin_memory=True, cache_images=False, heavy_augment=False):
#     data_root = Path(data_root)
#     train_trans = get_transforms(train=True, heavy=heavy_augment)
#     val_trans = get_transforms(train=False, heavy=False)

#     train_ds_list = _make_concat_datasets(data_root/"train", train_trans, cache_images=cache_images)
#     val_ds_list = _make_concat_datasets(data_root/"val", val_trans, cache_images=False)
#     test_ds_list = _make_concat_datasets(data_root/"test", val_trans, cache_images=False)

#     train_ds = ConcatDataset(train_ds_list) if train_ds_list else None
#     val_ds = ConcatDataset(val_ds_list) if val_ds_list else None
#     test_ds = ConcatDataset(test_ds_list) if test_ds_list else None

#     def make_loader(ds, shuffle):
#         if ds is None:
#             class Empty(torch.utils.data.Dataset):
#                 def __len__(self): return 0
#                 def __getitem__(self, idx): raise IndexError
#             return DataLoader(Empty(), batch_size=batch_size, shuffle=False, num_workers=0)
#         return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=pin_memory)

#     train_loader = make_loader(train_ds, shuffle=True)
#     val_loader = make_loader(val_ds, shuffle=False)
#     test_loader = make_loader(test_ds, shuffle=False)
#     return train_loader, val_loader, test_loader


# if __name__ == "__main__":
#     tr, val, tst = get_dataloaders(data_root="data", batch_size=4, num_workers=0)
#     print("Train batches:", len(tr), "Val batches:", len(val), "Test batches:", len(tst))


"""
Optimized dataset loader for lung segmentation (RTX 1650 optimized)
- Fast loading with PIL instead of CV2
- Smart memory management
- Robust transforms with proper normalization
- Windows-safe defaults
"""

from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from PIL import Image
import albumentations as A
from typing import Tuple

IMG_SIZE = (256, 256)


def get_transforms(train: bool = True):
    """
    Simple, stable transforms without ToTensorV2 (manual conversion for reliability).
    """
    if train:
        return A.Compose([
            A.HorizontalFlip(p=0.5),
            A.ShiftScaleRotate(
                shift_limit=0.05, 
                scale_limit=0.05, 
                rotate_limit=15, 
                border_mode=0,
                p=0.3
            ),
            A.OneOf([
                A.GaussianBlur(blur_limit=(3, 5), p=1.0),
            ], p=0.2),
            A.RandomBrightnessContrast(
                brightness_limit=0.15,
                contrast_limit=0.15,
                p=0.3
            ),
            A.Normalize(mean=(0.5,), std=(0.5,)),
            # Don't use ToTensorV2 - we'll convert manually
        ])
    else:
        return A.Compose([
            A.Normalize(mean=(0.5,), std=(0.5,)),
            # Don't use ToTensorV2 - we'll convert manually
        ])


class LungSegDataset(Dataset):
    """
    Fast, memory-efficient dataset for lung segmentation.
    Uses PIL for faster loading than CV2.
    """
    
    def __init__(
        self,
        images_dir: str,
        masks_dir: str,
        transforms=None,
        image_size: Tuple[int, int] = IMG_SIZE,
        cache_images: bool = False
    ):
        self.images_dir = Path(images_dir)
        self.masks_dir = Path(masks_dir)
        self.transforms = transforms
        self.image_size = image_size
        self.cache_images = cache_images
        
        # Find all image-mask pairs with robust matching
        self.pairs = self._find_pairs()
        
        print(f"[Dataset] {self.images_dir.name}: Found {len(self.pairs)} image-mask pairs")
        
        # Optional caching
        self._cache = {}
        if self.cache_images and len(self.pairs) > 0:
            print(f"[Dataset] Caching {len(self.pairs)} images into RAM...")
            for idx in range(len(self.pairs)):
                img_path, mask_path = self.pairs[idx]
                self._cache[idx] = (
                    self._load_image(img_path),
                    self._load_mask(mask_path)
                )
            print("[Dataset] Caching complete!")
    
    def _find_pairs(self):
        """Find matching image-mask pairs with fuzzy matching."""
        image_files = sorted([
            p for p in self.images_dir.iterdir() 
            if p.suffix.lower() in {'.png', '.jpg', '.jpeg', '.tif', '.bmp'}
        ])
        
        mask_files = sorted([
            p for p in self.masks_dir.iterdir() 
            if p.suffix.lower() in {'.png', '.jpg', '.jpeg', '.tif', '.bmp'}
        ])
        
        # Create mask lookup
        masks_by_stem = {}
        for mf in mask_files:
            # Clean stem (remove _mask, -mask, mask suffixes)
            stem = mf.stem
            for suffix in ['_mask', '-mask', 'mask', '_seg', '-seg', 'seg']:
                if stem.endswith(suffix):
                    stem = stem[:-len(suffix)]
                    break
            masks_by_stem[stem] = mf
        
        # Match images to masks
        pairs = []
        unmatched = []
        
        for img_file in image_files:
            img_stem = img_file.stem
            
            # Try exact stem match first
            if img_stem in masks_by_stem:
                pairs.append((img_file, masks_by_stem[img_stem]))
                continue
            
            # Try fuzzy matching
            found = False
            for mask_stem, mask_path in masks_by_stem.items():
                if mask_stem in img_stem or img_stem in mask_stem:
                    pairs.append((img_file, mask_path))
                    found = True
                    break
            
            if not found:
                unmatched.append(img_file.name)
        
        if unmatched:
            print(f"[Dataset] Warning: {len(unmatched)} unmatched images (first 5): {unmatched[:5]}")
        
        return pairs
    
    def _load_image(self, path: Path) -> np.ndarray:
        """Load and resize grayscale image using PIL (faster than CV2)."""
        img = Image.open(path).convert('L')  # Convert to grayscale
        img = img.resize(self.image_size, Image.BILINEAR)
        return np.array(img, dtype=np.uint8)
    
    def _load_mask(self, path: Path) -> np.ndarray:
        """Load and binarize mask."""
        mask = Image.open(path).convert('L')
        mask = mask.resize(self.image_size, Image.NEAREST)
        mask = np.array(mask, dtype=np.uint8)
        # Strict binarization
        mask = (mask > 127).astype(np.uint8) * 255
        return mask
    
    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):
        # Load from cache or disk
        if self.cache_images and idx in self._cache:
            img, mask = self._cache[idx]
        else:
            img_path, mask_path = self.pairs[idx]
            img = self._load_image(img_path)
            mask = self._load_mask(mask_path)
        
        # Add channel dimension for albumentations (H, W, C)
        img_hwc = np.expand_dims(img, axis=-1)
        mask_hwc = np.expand_dims(mask, axis=-1)
        
        # Apply augmentations (returns numpy arrays)
        if self.transforms:
            augmented = self.transforms(image=img_hwc, mask=mask_hwc)
            img_aug = augmented['image']    # numpy [H, W, C]
            mask_aug = augmented['mask']    # numpy [H, W, C]
        else:
            # Just normalize manually
            img_aug = img_hwc.astype(np.float32) / 255.0
            mask_aug = mask_hwc.astype(np.float32) / 255.0
        
        # Ensure float32 and proper range
        if img_aug.dtype != np.float32:
            img_aug = img_aug.astype(np.float32)
        if mask_aug.dtype != np.float32:
            mask_aug = mask_aug.astype(np.float32)
        
        # Manual conversion to tensor [C, H, W]
        # permute from [H, W, C] to [C, H, W]
        img_tensor = torch.from_numpy(img_aug).permute(2, 0, 1).contiguous()
        mask_tensor = torch.from_numpy(mask_aug).permute(2, 0, 1).contiguous()
        
        # Binarize mask strictly to 0 or 1
        mask_tensor = (mask_tensor > 0.5).float()
        
        # Ensure everything is float32
        img_tensor = img_tensor.float()
        mask_tensor = mask_tensor.float()
        
        # Final safety checks
        assert img_tensor.shape[0] == 1, f"Image has wrong channels: {img_tensor.shape}"
        assert mask_tensor.shape[0] == 1, f"Mask has wrong channels: {mask_tensor.shape}"
        assert torch.isfinite(img_tensor).all(), "Non-finite values in image"
        assert torch.isfinite(mask_tensor).all(), "Non-finite values in mask"
        
        return img_tensor, mask_tensor


def create_dataloaders(
    data_root: str = "data",
    batch_size: int = 8,
    num_workers: int = 0,
    pin_memory: bool = True,
    cache_train: bool = False
):
    """
    Create train/val/test dataloaders with proper configuration.
    
    Args:
        data_root: Root directory containing train/val/test folders
        batch_size: Batch size (8-12 recommended for RTX 1650)
        num_workers: 0 for Windows, 2-4 for Linux
        pin_memory: True if using CUDA
        cache_train: Cache training data in RAM (only if you have 16GB+ RAM)
    """
    data_root = Path(data_root)
    
    train_transform = get_transforms(train=True)
    val_transform = get_transforms(train=False)
    
    # Helper to load all datasets from a split directory
    def load_split_datasets(split_dir: Path, transform, cache: bool):
        datasets = []
        if not split_dir.exists():
            print(f"[DataLoader] Warning: {split_dir} does not exist")
            return datasets
        
        for class_dir in sorted(split_dir.iterdir()):
            if not class_dir.is_dir():
                continue
            
            images_dir = class_dir / "images"
            masks_dir = class_dir / "masks"
            
            if images_dir.exists() and masks_dir.exists():
                ds = LungSegDataset(
                    str(images_dir),
                    str(masks_dir),
                    transforms=transform,
                    cache_images=cache
                )
                if len(ds) > 0:
                    datasets.append(ds)
        
        return datasets
    
    # Load datasets
    train_datasets = load_split_datasets(data_root / "train", train_transform, cache_train)
    val_datasets = load_split_datasets(data_root / "val", val_transform, False)
    test_datasets = load_split_datasets(data_root / "test", val_transform, False)
    
    # Combine datasets
    train_ds = ConcatDataset(train_datasets) if train_datasets else None
    val_ds = ConcatDataset(val_datasets) if val_datasets else None
    test_ds = ConcatDataset(test_datasets) if test_datasets else None
    
    # Create dataloaders with proper error handling
    def make_loader(dataset, shuffle: bool):
        if dataset is None or len(dataset) == 0:
            # Return empty loader
            class EmptyDataset(Dataset):
                def __len__(self):
                    return 0
                def __getitem__(self, idx):
                    raise IndexError("Empty dataset")
            
            return DataLoader(
                EmptyDataset(),
                batch_size=1,
                shuffle=False,
                num_workers=0
            )
        
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=True if shuffle else False,  # Drop last incomplete batch in training
            persistent_workers=False  # Don't persist workers on Windows
        )
    
    train_loader = make_loader(train_ds, shuffle=True)
    val_loader = make_loader(val_ds, shuffle=False)
    test_loader = make_loader(test_ds, shuffle=False)
    
    print(f"\n[DataLoader] Created dataloaders:")
    print(f"  Train: {len(train_loader)} batches ({len(train_ds) if train_ds else 0} samples)")
    print(f"  Val:   {len(val_loader)} batches ({len(val_ds) if val_ds else 0} samples)")
    print(f"  Test:  {len(test_loader)} batches ({len(test_ds) if test_ds else 0} samples)")
    
    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    # Test the dataloader
    print("Testing dataloader...")
    train_loader, val_loader, test_loader = create_dataloaders(
        data_root="data",
        batch_size=4,
        num_workers=0,
        cache_train=False
    )
    
    if len(train_loader) > 0:
        imgs, masks = next(iter(train_loader))
        print(f"\nSample batch:")
        print(f"  Images: {imgs.shape}, dtype={imgs.dtype}, range=[{imgs.min():.3f}, {imgs.max():.3f}]")
        print(f"  Masks:  {masks.shape}, dtype={masks.dtype}, unique={masks.unique().tolist()}")
        print("\n✓ Dataloader test passed!")
    else:
        print("\n✗ No data found. Check your data directory structure.")