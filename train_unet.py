"""
Ultra-optimized UNet training for RTX 1650 (4GB VRAM)
- Optimized batch size and learning rate
- Stable loss function with proper numerical stability
- Smart memory management
- Gradient accumulation for effective larger batches
- Comprehensive logging and checkpointing
- Windows-safe signal handling
"""

import os
import sys
import time
import signal
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
try:
    from torch.amp import autocast, GradScaler
except ImportError:
    from torch.cuda.amp import autocast, GradScaler
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from tqdm import tqdm

from dataset_loader import create_dataloaders
from unet_model import UNet


# ==================== CONFIGURATION ====================
class Config:
    # Data
    DATA_ROOT = "data"
    IMG_SIZE = 256
    
    # Training - Optimized for RTX 1650 + 8GB RAM (NO AMP)
    BATCH_SIZE = 6  # Safe without AMP
    ACCUMULATION_STEPS = 2  # Effective batch size = 12
    NUM_EPOCHS = 50
    
    # Optimizer - Balanced settings
    LEARNING_RATE = 1e-4  # Conservative but effective
    WEIGHT_DECAY = 1e-5
    MIN_LR = 1e-7
    
    # Scheduler
    T_0 = 10  # Cosine annealing restart every 10 epochs
    T_MULT = 2  # Double the restart interval
    
    # Regularization
    GRAD_CLIP = 1.0
    LABEL_SMOOTHING = 0.0  # Can help with overfitting
    
    # Hardware
    NUM_WORKERS = 0  # Windows safe
    PIN_MEMORY = True
    
    # Checkpointing
    CHECKPOINT_DIR = Path("checkpoints")
    SAVE_EVERY = 5  # Save checkpoint every N epochs
    
    # Logging
    LOG_EVERY = 20  # Log every N batches
    
    # Mixed precision - DISABLED due to NaN issues on GTX 1650
    USE_AMP = False  # Causes NaN with this GPU/PyTorch version
    
    # Data caching (DO NOT USE - only 8GB RAM!)
    CACHE_TRAIN = False  # Never cache with 8GB RAM

config = Config()
config.CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)


# ==================== LOSS FUNCTIONS ====================
class DiceBCELoss(nn.Module):
    """
    Combined Dice + BCE loss with numerical stability.
    More stable than separate losses.
    """
    
    def __init__(self, dice_weight=0.5, smooth=1e-6):
        super().__init__()
        self.dice_weight = dice_weight
        self.bce_weight = 1 - dice_weight
        self.smooth = smooth
        self.bce = nn.BCEWithLogitsLoss()
    
    def forward(self, logits, targets):
        # BCE loss
        bce_loss = self.bce(logits, targets)
        
        # Dice loss
        probs = torch.sigmoid(logits)
        
        # Flatten
        probs_flat = probs.view(-1)
        targets_flat = targets.view(-1)
        
        intersection = (probs_flat * targets_flat).sum()
        dice_score = (2. * intersection + self.smooth) / (
            probs_flat.sum() + targets_flat.sum() + self.smooth
        )
        dice_loss = 1 - dice_score
        
        # Combined loss
        total_loss = self.bce_weight * bce_loss + self.dice_weight * dice_loss
        
        return total_loss, bce_loss, dice_loss


# ==================== METRICS ====================
def calculate_dice_score(logits, targets, threshold=0.5, smooth=1e-6):
    """Calculate Dice score from logits."""
    with torch.no_grad():
        probs = torch.sigmoid(logits)
        preds = (probs > threshold).float()
        
        # Flatten
        preds_flat = preds.view(-1)
        targets_flat = targets.view(-1)
        
        intersection = (preds_flat * targets_flat).sum()
        dice = (2. * intersection + smooth) / (
            preds_flat.sum() + targets_flat.sum() + smooth
        )
        
        return dice.item()


def calculate_iou(logits, targets, threshold=0.5, smooth=1e-6):
    """Calculate IoU (Jaccard index)."""
    with torch.no_grad():
        probs = torch.sigmoid(logits)
        preds = (probs > threshold).float()
        
        # Flatten
        preds_flat = preds.view(-1)
        targets_flat = targets.view(-1)
        
        intersection = (preds_flat * targets_flat).sum()
        union = preds_flat.sum() + targets_flat.sum() - intersection
        
        iou = (intersection + smooth) / (union + smooth)
        
        return iou.item()


# ==================== TRAINING FUNCTIONS ====================
def train_one_epoch(model, loader, criterion, optimizer, scaler, device, epoch):
    """Train for one epoch with gradient accumulation."""
    model.train()
    
    running_loss = 0.0
    running_bce = 0.0
    running_dice_loss = 0.0
    running_dice_score = 0.0
    running_iou = 0.0
    num_batches = 0
    
    optimizer.zero_grad()
    
    pbar = tqdm(loader, desc=f"Epoch {epoch}/{config.NUM_EPOCHS}")
    
    for batch_idx, (images, masks) in enumerate(pbar):
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        
        # Check for NaN in input data
        if not torch.isfinite(images).all():
            print(f"\n[ERROR] Non-finite values in input images at batch {batch_idx}")
            continue
        if not torch.isfinite(masks).all():
            print(f"\n[ERROR] Non-finite values in input masks at batch {batch_idx}")
            continue
        
        # Forward pass (no AMP to avoid NaN issues)
        logits = model(images)
        
        # Check logits before loss computation
        if not torch.isfinite(logits).all():
            print(f"\n[ERROR] Non-finite logits at batch {batch_idx}")
            print(f"  Images: min={images.min():.3f}, max={images.max():.3f}")
            debug_path = config.CHECKPOINT_DIR / "debug_nan_logits.pth"
            torch.save({
                'model': model.state_dict(),
                'batch_idx': batch_idx,
            }, debug_path)
            sys.exit(1)
        
        loss, bce_loss, dice_loss = criterion(logits, masks)
            
            # Scale loss by accumulation steps
        loss = loss / config.ACCUMULATION_STEPS
        
        # Check for NaN/Inf
        if not torch.isfinite(loss):
            print(f"\n[ERROR] Non-finite loss detected at epoch {epoch}, batch {batch_idx}")
            print(f"  Loss: {loss.item()}")
            print(f"  Logits: min={logits.min():.3f}, max={logits.max():.3f}")
            print(f"  Targets: min={masks.min():.3f}, max={masks.max():.3f}")
            
            # Save debug checkpoint
            debug_path = config.CHECKPOINT_DIR / "debug_nan.pth"
            torch.save({
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
                'batch': batch_idx,
            }, debug_path)
            print(f"  Saved debug checkpoint to {debug_path}")
            sys.exit(1)
        
        # Backward pass
        if config.USE_AMP:
            scaler.scale(loss).backward()
        else:
            loss.backward()
        
        # Update weights every ACCUMULATION_STEPS
        if (batch_idx + 1) % config.ACCUMULATION_STEPS == 0:
            # Gradient clipping
            if config.USE_AMP:
                scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.GRAD_CLIP)
            
            # Optimizer step
            if config.USE_AMP:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad()
        
        # Calculate metrics
        dice_score = calculate_dice_score(logits.detach(), masks)
        iou_score = calculate_iou(logits.detach(), masks)
        
        # Update running stats
        running_loss += loss.item() * config.ACCUMULATION_STEPS
        running_bce += bce_loss.item()
        running_dice_loss += dice_loss.item()
        running_dice_score += dice_score
        running_iou += iou_score
        num_batches += 1
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f'{running_loss/num_batches:.4f}',
            'dice': f'{running_dice_score/num_batches:.4f}',
            'iou': f'{running_iou/num_batches:.4f}'
        })
        
        # Detailed logging
        if (batch_idx + 1) % config.LOG_EVERY == 0:
            avg_loss = running_loss / num_batches
            avg_dice = running_dice_score / num_batches
            avg_iou = running_iou / num_batches
            print(f"\n  Batch {batch_idx+1}/{len(loader)}: "
                  f"loss={avg_loss:.4f}, dice={avg_dice:.4f}, iou={avg_iou:.4f}")
    
    # Final epoch statistics
    epoch_loss = running_loss / num_batches
    epoch_bce = running_bce / num_batches
    epoch_dice_loss = running_dice_loss / num_batches
    epoch_dice_score = running_dice_score / num_batches
    epoch_iou = running_iou / num_batches
    
    return {
        'loss': epoch_loss,
        'bce': epoch_bce,
        'dice_loss': epoch_dice_loss,
        'dice_score': epoch_dice_score,
        'iou': epoch_iou
    }


@torch.no_grad()
def validate(model, loader, criterion, device):
    """Validate the model."""
    model.eval()
    
    running_loss = 0.0
    running_dice_score = 0.0
    running_iou = 0.0
    num_batches = 0
    
    pbar = tqdm(loader, desc="Validation", leave=False)
    
    for images, masks in pbar:
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        
        # No AMP for validation either
        logits = model(images)
        loss, _, _ = criterion(logits, masks)
        
        dice_score = calculate_dice_score(logits, masks)
        iou_score = calculate_iou(logits, masks)
        
        running_loss += loss.item()
        running_dice_score += dice_score
        running_iou += iou_score
        num_batches += 1
        
        pbar.set_postfix({
            'loss': f'{running_loss/num_batches:.4f}',
            'dice': f'{running_dice_score/num_batches:.4f}'
        })
    
    return {
        'loss': running_loss / num_batches,
        'dice_score': running_dice_score / num_batches,
        'iou': running_iou / num_batches
    }


# ==================== UTILITIES ====================
def save_checkpoint(model, optimizer, scheduler, scaler, epoch, best_dice, filename):
    """Save training checkpoint."""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'scaler_state_dict': scaler.state_dict(),
        'best_dice': best_dice,
    }
    torch.save(checkpoint, filename)
    print(f"  Saved checkpoint: {filename}")


def load_checkpoint(model, optimizer, scheduler, scaler, filename):
    """Load training checkpoint."""
    if not Path(filename).exists():
        print(f"  No checkpoint found at {filename}")
        return 0, 0.0
    
    print(f"  Loading checkpoint: {filename}")
    checkpoint = torch.load(filename, map_location='cpu')
    
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    if 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    if 'scaler_state_dict' in checkpoint and config.USE_AMP:
        scaler.load_state_dict(checkpoint['scaler_state_dict'])
    
    epoch = checkpoint.get('epoch', 0)
    best_dice = checkpoint.get('best_dice', 0.0)
    
    print(f"  Resumed from epoch {epoch}, best dice: {best_dice:.4f}")
    return epoch, best_dice


def initialize_weights(model):
    """Initialize model weights with careful initialization to avoid NaN."""
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            # Use Xavier uniform instead of Kaiming for better stability
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.ConvTranspose2d):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)


# ==================== MAIN TRAINING LOOP ====================
def main():
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
    
    # Device setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n{'='*60}")
    print(f"  TRAINING CONFIGURATION")
    print(f"{'='*60}")
    print(f"  Device: {device}")
    
    if torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  CUDA Version: {torch.version.cuda}")
        print(f"  Available VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    print(f"\n  Batch size: {config.BATCH_SIZE}")
    print(f"  Accumulation steps: {config.ACCUMULATION_STEPS}")
    print(f"  Effective batch size: {config.BATCH_SIZE * config.ACCUMULATION_STEPS}")
    print(f"  Learning rate: {config.LEARNING_RATE}")
    print(f"  Epochs: {config.NUM_EPOCHS}")
    print(f"{'='*60}\n")
    
    # Create dataloaders
    print("Loading datasets...")
    train_loader, val_loader, test_loader = create_dataloaders(
        data_root=config.DATA_ROOT,
        batch_size=config.BATCH_SIZE,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY,
        cache_train=config.CACHE_TRAIN
    )
    
    if len(train_loader) == 0:
        print("\n[ERROR] No training data found! Check your data directory structure.")
        print("Expected structure:")
        print("  data/")
        print("    train/")
        print("      class_name/")
        print("        images/")
        print("        masks/")
        sys.exit(1)
    
    # Create model
    print("\nInitializing model...")
    model = UNet(in_channels=1, out_channels=1).to(device)
    initialize_weights(model)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    
    # Loss and optimizer
    criterion = DiceBCELoss(dice_weight=0.5)  # Balanced BCE and Dice
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY,
        betas=(0.9, 0.999),
        eps=1e-8  # Add epsilon for numerical stability
    )
    
    # Scheduler with warm restarts
    scheduler = CosineAnnealingWarmRestarts(
        optimizer,
        T_0=config.T_0,
        T_mult=config.T_MULT,
        eta_min=config.MIN_LR
    )
    
    # AMP scaler - compatible with all PyTorch versions
    try:
        # PyTorch 2.0+ style
        scaler = GradScaler('cuda', enabled=config.USE_AMP)
    except TypeError:
        # PyTorch 1.x style (fallback)
        scaler = GradScaler(enabled=config.USE_AMP)
    
    # Load checkpoint if exists
    checkpoint_path = config.CHECKPOINT_DIR / "latest.pth"
    start_epoch, best_dice = load_checkpoint(
        model, optimizer, scheduler, scaler, checkpoint_path
    )
    
    # Training loop
    print(f"\n{'='*60}")
    print("  STARTING TRAINING")
    print(f"{'='*60}\n")
    
    try:
        for epoch in range(start_epoch + 1, config.NUM_EPOCHS + 1):
            epoch_start = time.time()
            
            # Train
            train_metrics = train_one_epoch(
                model, train_loader, criterion, optimizer, scaler, device, epoch
            )
            
            # Validate
            if len(val_loader) > 0:
                val_metrics = validate(model, val_loader, criterion, device)
            else:
                val_metrics = {'loss': 0.0, 'dice_score': 0.0, 'iou': 0.0}
            
            # Update scheduler
            scheduler.step()
            
            epoch_time = time.time() - epoch_start
            
            # Print epoch summary
            print(f"\n{'='*60}")
            print(f"  EPOCH {epoch}/{config.NUM_EPOCHS} SUMMARY ({epoch_time:.1f}s)")
            print(f"{'='*60}")
            print(f"  Train: loss={train_metrics['loss']:.4f}, "
                  f"dice={train_metrics['dice_score']:.4f}, "
                  f"iou={train_metrics['iou']:.4f}")
            print(f"  Val:   loss={val_metrics['loss']:.4f}, "
                  f"dice={val_metrics['dice_score']:.4f}, "
                  f"iou={val_metrics['iou']:.4f}")
            print(f"  LR: {optimizer.param_groups[0]['lr']:.2e}")
            
            if torch.cuda.is_available():
                print(f"  VRAM: {torch.cuda.max_memory_allocated(device)/1024**3:.2f} GB")
                torch.cuda.reset_peak_memory_stats(device)
            
            print(f"{'='*60}\n")
            
            # Save best model
            current_dice = val_metrics['dice_score'] if len(val_loader) > 0 else train_metrics['dice_score']
            
            if current_dice > best_dice:
                best_dice = current_dice
                best_path = config.CHECKPOINT_DIR / "best_model.pth"
                save_checkpoint(model, optimizer, scheduler, scaler, epoch, best_dice, best_path)
                print(f"  ★ New best model! Dice: {best_dice:.4f}\n")
            
            # Save periodic checkpoint
            if epoch % config.SAVE_EVERY == 0:
                save_checkpoint(
                    model, optimizer, scheduler, scaler, epoch, best_dice,
                    config.CHECKPOINT_DIR / f"epoch_{epoch:03d}.pth"
                )
            
            # Always save latest
            save_checkpoint(
                model, optimizer, scheduler, scaler, epoch, best_dice,
                config.CHECKPOINT_DIR / "latest.pth"
            )
    
    except KeyboardInterrupt:
        print("\n\n[INTERRUPTED] Saving checkpoint...")
        save_checkpoint(
            model, optimizer, scheduler, scaler, epoch, best_dice,
            config.CHECKPOINT_DIR / "interrupted.pth"
        )
        print("Checkpoint saved. Exiting.")
        sys.exit(0)
    
    print(f"\n{'='*60}")
    print("  TRAINING COMPLETE!")
    print(f"{'='*60}")
    print(f"  Best Dice Score: {best_dice:.4f}")
    print(f"  Best model saved at: {config.CHECKPOINT_DIR / 'best_model.pth'}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()