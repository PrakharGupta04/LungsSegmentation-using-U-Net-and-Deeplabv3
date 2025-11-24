"""
Quick test script to verify UNet model and data pipeline
Run this before training to catch issues early
"""

import torch
import torch.nn as nn
from dataset_loader import create_dataloaders
from unet_model import UNet

def test_model():
    print("="*60)
    print("TESTING MODEL AND DATA PIPELINE")
    print("="*60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")
    
    # Test 1: Load data
    print("\n[1/4] Testing data loading...")
    try:
        train_loader, val_loader, test_loader = create_dataloaders(
            data_root="data",
            batch_size=2,  # Small batch for testing
            num_workers=0,
            pin_memory=False,
            cache_train=False
        )
        print(f"✓ Data loaders created successfully")
        print(f"  Train batches: {len(train_loader)}")
    except Exception as e:
        print(f"✗ Data loading failed: {e}")
        return False
    
    # Test 2: Get a batch
    print("\n[2/4] Testing batch retrieval...")
    try:
        images, masks = next(iter(train_loader))
        print(f"✓ Batch retrieved successfully")
        print(f"  Images: {images.shape}, dtype={images.dtype}")
        print(f"  Masks:  {masks.shape}, dtype={masks.dtype}")
        print(f"  Image range: [{images.min():.3f}, {images.max():.3f}]")
        print(f"  Mask range:  [{masks.min():.3f}, {masks.max():.3f}]")
        print(f"  Mask unique: {masks.unique().tolist()}")
        
        # Check for NaN
        if not torch.isfinite(images).all():
            print("✗ WARNING: Non-finite values in images!")
            return False
        if not torch.isfinite(masks).all():
            print("✗ WARNING: Non-finite values in masks!")
            return False
            
    except Exception as e:
        print(f"✗ Batch retrieval failed: {e}")
        return False
    
    # Test 3: Create model
    print("\n[3/4] Testing model creation...")
    try:
        model = UNet(in_channels=1, out_channels=1).to(device)
        
        # Xavier initialization
        for m in model.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        
        print(f"✓ Model created successfully")
        print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
    except Exception as e:
        print(f"✗ Model creation failed: {e}")
        return False
    
    # Test 4: Forward pass
    print("\n[4/4] Testing forward pass...")
    try:
        model.eval()
        images = images.to(device)
        masks = masks.to(device)
        
        with torch.no_grad():
            logits = model(images)
        
        print(f"✓ Forward pass successful")
        print(f"  Output shape: {logits.shape}")
        print(f"  Output range: [{logits.min():.3f}, {logits.max():.3f}]")
        
        # Check for NaN
        if not torch.isfinite(logits).all():
            print("✗ CRITICAL: Model output contains NaN/Inf!")
            print("  This indicates a problem with the model architecture or initialization")
            return False
        
        # Test loss computation
        print("\n[BONUS] Testing loss computation...")
        criterion = nn.BCEWithLogitsLoss()
        loss = criterion(logits, masks)
        print(f"  BCE Loss: {loss.item():.6f}")
        
        if not torch.isfinite(loss):
            print("✗ CRITICAL: Loss is NaN/Inf!")
            return False
        
        print(f"✓ Loss computation successful")
        
    except Exception as e:
        print(f"✗ Forward pass failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\n" + "="*60)
    print("✓ ALL TESTS PASSED!")
    print("="*60)
    print("\nYou can now run: python train_unet.py")
    return True

if __name__ == "__main__":
    success = test_model()
    if not success:
        print("\n✗ Tests failed! Fix the issues above before training.")
        exit(1)