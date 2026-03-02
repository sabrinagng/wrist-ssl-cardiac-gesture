"""
Train 2D CNN-based Masked Autoencoder for EMG Pretraining
===========================================================
Uses time-frequency representation from JSON data.
"""

import os
import sys
import argparse
import datetime
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from tqdm import tqdm

from model_2d import EMGMaskedAE2D
from json_dataset_2d import JSONDataset2D

# Default paths
JSON_PATH = '/root/autodl-fs/segraw_EMG_allgestures_allusers.json'


def train_epoch(model, dataloader, optimizer, device, alpha_mask=1.0, alpha_vis=0.01):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    
    for x in tqdm(dataloader, desc='Train', leave=False):
        x = x.to(device, non_blocking=True)
        
        # Forward
        recon, mask = model(x)
        loss = model.compute_loss(x, recon, mask, alpha_mask=alpha_mask, alpha_vis=alpha_vis)
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(dataloader)


def evaluate(model, dataloader, device, alpha_mask=1.0, alpha_vis=0.01):
    """Evaluate model."""
    model.eval()
    total_loss = 0.0
    
    with torch.no_grad():
        for x in tqdm(dataloader, desc='Eval', leave=False):
            x = x.to(device, non_blocking=True)
            
            recon, mask = model(x)
            loss = model.compute_loss(x, recon, mask, alpha_mask=alpha_mask, alpha_vis=alpha_vis)
            
            total_loss += loss.item()
    
    return total_loss / len(dataloader)


def main():
    parser = argparse.ArgumentParser(description='Train 2D CNN MAE')
    parser.add_argument('--json-path', type=str, default=JSON_PATH,
                        help='Path to segraw_EMG_allgestures_allusers.json')
    parser.add_argument('--transform', type=str, default='stft',
                        choices=['stft', 'cwt', 'logmel'],
                        help='Time-frequency transform type')
    parser.add_argument('--n-fft', type=int, default=256,
                        help='FFT size for STFT/LogMel')
    parser.add_argument('--hop-length', type=int, default=64,
                        help='Hop length for time-frequency transform')
    parser.add_argument('--n-scales', type=int, default=128,
                        help='Number of scales for CWT')
    parser.add_argument('--n-mels', type=int, default=128,
                        help='Number of mel bins for LogMel')
    parser.add_argument('--f-min', type=float, default=20.0,
                        help='Minimum frequency (Hz)')
    parser.add_argument('--f-max', type=float, default=450.0,
                        help='Maximum frequency (Hz)')
    parser.add_argument('--fs', type=float, default=2000.0,
                        help='Sampling frequency (Hz)')
    parser.add_argument('--target-len', type=int, default=8192,
                        help='Target length for resampling')
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=3e-4,
                        help='Learning rate')
    parser.add_argument('--weight-decay', type=float, default=0.01)
    parser.add_argument('--warmup-pct', type=float, default=0.05,
                        help='Percentage of steps for linear warmup')
    parser.add_argument('--eta-min', type=float, default=1e-6,
                        help='Minimum learning rate for cosine annealing')
    parser.add_argument('--mask-ratio', type=float, default=0.3,
                        help='Mask ratio')
    parser.add_argument('--block-size', type=str, default='8,8',
                        help='Block size for 2D masking (freq,time)')
    parser.add_argument('--mask-type', type=str, default='random',
                        choices=['random', 'uniform'],
                        help='Mask type')
    parser.add_argument('--alpha-mask', type=float, default=1.0,
                        help='Weight for masked tokens loss')
    parser.add_argument('--alpha-vis', type=float, default=0.005,
                        help='Weight for visible tokens loss')
    parser.add_argument('--dataset-norm', action='store_true',
                        help='Apply normalization in dataset')
    parser.add_argument('--channels', type=str, default='6,10',
                        help='Channels to select (comma-separated, 0-indexed)')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Output directory for checkpoints')
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Parse block size
    block_size = tuple(map(int, args.block_size.split(',')))
    
    # Parse channels
    channels = [int(c.strip()) for c in args.channels.split(',')] if args.channels else None
    
    # Create output directory
    if args.output_dir is None:
        ts = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
        args.output_dir = f'/root/autodl-fs/emg_pretrain/pretrain/checkpoints/cnn_mae_2d_{args.transform}/{ts}'
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load dataset
    print("\n" + "="*70)
    print("Loading dataset...")
    print("="*70)
    print(f"JSON path: {args.json_path}")
    print(f"Transform: {args.transform.upper()}")
    print(f"Dataset normalization: {args.dataset_norm}")
    
    dataset = JSONDataset2D(
        json_path=args.json_path,
        transform_type=args.transform,
        n_fft=args.n_fft,
        hop_length=args.hop_length,
        n_scales=args.n_scales,
        f_min=args.f_min,
        f_max=args.f_max,
        fs=args.fs,
        n_mels=args.n_mels,
        target_len=args.target_len,
        normalize=args.dataset_norm,
        channels=channels,
    )
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                           num_workers=4, pin_memory=True)
    
    print(f"Dataset size: {len(dataset)}")
    print(f"Batches per epoch: {len(dataloader)}")
    
    # Get sample to determine input shape
    sample = dataset[0]
    in_ch = sample.shape[0]
    print(f"Input shape: {sample.shape}")
    print(f"Input channels: {in_ch}")
    
    # Create model
    print("\n" + "="*70)
    print("Creating model...")
    print("="*70)
    
    model = EMGMaskedAE2D(
        in_ch=in_ch,
        mask_ratio=args.mask_ratio,
        block_size=block_size,
        mask_type=args.mask_type,
    ).to(device)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Mask ratio: {args.mask_ratio}")
    print(f"Block size: {block_size}")
    print(f"Mask type: {args.mask_type}")
    
    # Optimizer
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    # Scheduler with warmup
    total_steps = len(dataloader) * args.epochs
    warmup_steps = int(total_steps * args.warmup_pct)
    
    def lr_lambda(step):
        if step < warmup_steps:
            return step / warmup_steps
        else:
            # Cosine annealing after warmup
            progress = (step - warmup_steps) / (total_steps - warmup_steps)
            import math
            return args.eta_min / args.lr + (1 - args.eta_min / args.lr) * 0.5 * (1 + math.cos(progress * math.pi))
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    print(f"Total steps: {total_steps}")
    print(f"Warmup steps: {warmup_steps}")
    print(f"Initial LR: {args.lr}")
    print(f"Min LR: {args.eta_min}")
    
    # Training loop
    print("\n" + "="*70)
    print("Training...")
    print("="*70)
    
    best_loss = float('inf')
    best_epoch = 0
    
    train_losses = []
    
    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(
            model, dataloader, optimizer, device,
            alpha_mask=args.alpha_mask, alpha_vis=args.alpha_vis
        )
        
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        
        train_losses.append(train_loss)
        
        if train_loss < best_loss:
            best_loss = train_loss
            best_epoch = epoch
        
        if epoch % 5 == 0 or epoch == args.epochs:
            print(f"Epoch {epoch:3d}/{args.epochs}  "
                  f"Train Loss: {train_loss:.6f}  "
                  f"LR: {current_lr:.2e}  "
                  f"Best: {best_loss:.6f} (ep {best_epoch})")
        
        # Save checkpoint
        if epoch % 5 == 0 or epoch == args.epochs:
            checkpoint_path = os.path.join(args.output_dir, f'ckpt_cnn_mae_2d_epoch_{epoch}.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_loss': train_loss,
                'args': vars(args),
            }, checkpoint_path)
    
    # Final summary
    print("\n" + "="*70)
    print("Training Complete")
    print("="*70)
    print(f"Best Train Loss: {best_loss:.6f} (epoch {best_epoch})")
    print(f"Final Train Loss: {train_losses[-1]:.6f}")
    print(f"\nCheckpoints saved to: {args.output_dir}")
    
    # Save training report
    report_data = {
        'args': vars(args),
        'best_epoch': best_epoch,
        'best_train_loss': float(best_loss),
        'final_train_loss': float(train_losses[-1]),
        'train_losses': train_losses,
    }
    
    report_path = os.path.join(args.output_dir, 'training_report.json')
    with open(report_path, 'w') as f:
        json.dump(report_data, f, indent=2)
    
    print(f"Training report: {report_path}")


if __name__ == '__main__':
    main()
