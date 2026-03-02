"""
2D CNN-based Masked Autoencoder for EMG Pretraining
====================================================
Uses time-frequency representation (STFT/CWT/LogMel) as input.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def mean_var_norm_2d(x: torch.Tensor, eps: float = 1e-6):
    """
    Per-sample, per-channel mean-variance normalization over spatial dimensions.
    
    Args:
        x: (B, C, H, W) tensor (time-frequency representation)
        eps: small value to avoid division by zero
    
    Returns:
        x_norm: normalized tensor
        mean: mean values (B, C, 1, 1)
        std: std values (B, C, 1, 1)
    """
    # Compute mean and std over H and W dimensions
    mean = x.mean(dim=[2, 3], keepdim=True)
    var = x.var(dim=[2, 3], keepdim=True, unbiased=False)
    std = torch.sqrt(var + eps)
    x_norm = (x - mean) / std
    return x_norm, mean, std


class EMGMaskedAE2D(nn.Module):
    """
    2D CNN-based Masked Autoencoder for EMG time-frequency representations.
    """
    
    def __init__(
        self,
        in_ch=2,
        encoder_channels=[64, 128, 256, 512],
        decoder_channels=[256, 128, 64, 2],
        kernel_sizes=[7, 5, 3, 3],
        strides=[2, 2, 2, 2],
        mask_ratio=0.5,
        block_size=(8, 8),  # (freq_block, time_block) for 2D masking
        mask_type='random',
    ):
        super().__init__()
        self.in_ch = in_ch
        self.mask_ratio = mask_ratio
        self.block_size = block_size
        self.mask_type = mask_type
        
        # Encoder: 2D CNN
        encoder_layers = []
        in_c = in_ch
        for out_c, k, s in zip(encoder_channels, kernel_sizes, strides):
            encoder_layers.extend([
                nn.Conv2d(in_c, out_c, k, stride=s, padding=k//2),
                nn.BatchNorm2d(out_c),
                nn.ReLU(inplace=True),
            ])
            in_c = out_c
        self.encoder = nn.Sequential(*encoder_layers)
        
        # Decoder: 2D Transposed CNN
        decoder_layers = []
        in_c = encoder_channels[-1]
        for out_c, k, s in zip(decoder_channels, kernel_sizes[::-1], strides[::-1]):
            decoder_layers.extend([
                nn.ConvTranspose2d(in_c, out_c, k, stride=s, padding=k//2, output_padding=s-1),
                nn.BatchNorm2d(out_c),
                nn.ReLU(inplace=True) if out_c != decoder_channels[-1] else nn.Identity(),
            ])
            in_c = out_c
        self.decoder = nn.Sequential(*decoder_layers)
        
    def forward(self, x, mask=None):
        """
        Args:
            x: (B, C, H, W) input time-frequency representation
            mask: optional boolean mask (B, H, W)
        
        Returns:
            recon: reconstructed time-frequency representation (B, C, H, W)
            mask: applied mask (B, H, W)
        """
        B, C, H, W = x.shape
        
        # Normalize input
        x_norm, mean, std = mean_var_norm_2d(x)
        
        # Generate mask if not provided
        if mask is None:
            mask = self.generate_mask(B, H, W, x.device)
        
        # Apply mask (set masked regions to 0)
        # mask: (B, H, W), x_masked: (B, C, H, W)
        # Expand mask to (B, 1, H, W) and broadcast to (B, C, H, W)
        x_masked = x_norm.clone()
        mask_expanded = mask.unsqueeze(1)  # (B, 1, H, W)
        x_masked[mask_expanded.expand_as(x_masked)] = 0.0
        
        # Encode
        z = self.encoder(x_masked)
        
        # Decode
        recon_norm = self.decoder(z)
        
        # Denormalize
        recon = recon_norm * std + mean
        
        return recon, mask
    
    def generate_mask(self, B, H, W, device):
        """Generate 2D mask for time-frequency representation."""
        if self.mask_type == 'random':
            # Random block masking
            freq_blocks = H // self.block_size[0]
            time_blocks = W // self.block_size[1]
            if freq_blocks == 0:
                freq_blocks = 1
            if time_blocks == 0:
                time_blocks = 1
            
            total_blocks = freq_blocks * time_blocks
            num_mask = int(total_blocks * self.mask_ratio)
            
            mask = torch.zeros(B, H, W, dtype=torch.bool, device=device)
            for b in range(B):
                block_indices = torch.randperm(total_blocks, device=device)[:num_mask]
                for bi in block_indices:
                    f_idx = bi // time_blocks
                    t_idx = bi % time_blocks
                    f_start = f_idx * self.block_size[0]
                    f_end = min(f_start + self.block_size[0], H)
                    t_start = t_idx * self.block_size[1]
                    t_end = min(t_start + self.block_size[1], W)
                    mask[b, f_start:f_end, t_start:t_end] = True
        else:
            # Uniform random masking
            num_mask = int(H * W * self.mask_ratio)
            mask = torch.zeros(B, H, W, dtype=torch.bool, device=device)
            for b in range(B):
                flat_indices = torch.randperm(H * W, device=device)[:num_mask]
                f_indices = flat_indices // W
                t_indices = flat_indices % W
                mask[b, f_indices, t_indices] = True
        
        return mask
    
    def compute_loss(self, x, recon, mask, alpha_mask=1.0, alpha_vis=0.01):
        """
        Compute reconstruction loss.
        
        Args:
            x: original input (B, C, H, W)
            recon: reconstruction (B, C, H', W') - may have different size due to decoder
            mask: mask (B, H, W)
            alpha_mask: weight for masked tokens
            alpha_vis: weight for visible tokens

        Returns:
            loss: scalar tensor
        """
        # Normalize for loss computation
        x_norm, _, _ = mean_var_norm_2d(x)
        recon_norm, _, _ = mean_var_norm_2d(recon)
        
        # Resize recon to match input size if needed
        if recon_norm.shape[2:] != x_norm.shape[2:]:
            recon_norm = F.interpolate(
                recon_norm, size=x_norm.shape[2:], mode='bilinear', align_corners=False
            )
        
        # Expand mask to match channel dimension: (B, H, W) -> (B, 1, H, W)
        mask_expanded = mask.unsqueeze(1)  # (B, 1, H, W)
        
        # L1 loss
        # Mask expanded: (B, 1, H, W) -> (B, C, H, W)
        mask_ch = mask_expanded.expand_as(recon_norm)  # (B, C, H, W)
        loss_mask = F.l1_loss(
            recon_norm[mask_ch],
            x_norm[mask_ch],
            reduction='mean'
        )
        loss_vis = F.l1_loss(
            recon_norm[~mask_ch],
            x_norm[~mask_ch],
            reduction='mean'
        )
        
        loss = alpha_mask * loss_mask + alpha_vis * loss_vis
        return loss
