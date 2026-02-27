"""
model_overlap_peak.py
ECG MAE with BOTH patch overlap AND peak-adaptive masking.
{1,1} configuration: has Peak detection, has Overlap
"""
import math
from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F

from peak_detector_neurokit import ECGPeakDetector


@dataclass
class MAEConfig:
    in_channels: int = 2
    patch_size: int = 64
    embed_dim: int = 128
    depth: int = 6
    num_heads: int = 8
    mlp_ratio: float = 4.0
    decoder_embed_dim: int = 64
    decoder_depth: int = 4
    decoder_num_heads: int = 4
    mask_ratio: float = 0.55
    max_patches: int = 512
    overlap_ratio: float = 0.5
    use_adaptive_masking: bool = True
    sampling_frequency: float = 500.0


class MLP(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4.0, dropout=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, int(dim * mlp_ratio), dropout)

    def forward(self, x):
        x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x), need_weights=False)[0]
        return x + self.mlp(self.norm2(x))


def sincos_pos_embed(dim, n, device):
    """1D sinusoidal positional embedding [1, N, dim]."""
    pos = torch.arange(n, device=device).float().unsqueeze(1)
    div = torch.exp(torch.arange(0, dim, 2, device=device).float() * (-math.log(10000.0) / dim))
    pe = torch.zeros(n, dim, device=device)
    pe[:, 0::2] = torch.sin(pos * div)
    pe[:, 1::2] = torch.cos(pos * div)
    return pe.unsqueeze(0)


class ECGMAE(nn.Module):
    """MAE for 1D multi-channel ECG with peak-adaptive masking. Input: [B, C, L]."""

    def __init__(self, cfg: MAEConfig):
        super().__init__()
        self.cfg = cfg
        self.in_channels = cfg.in_channels
        self.patch_size = cfg.patch_size
        self.mask_ratio = cfg.mask_ratio
        self.max_patches = cfg.max_patches
        self.overlap_ratio = cfg.overlap_ratio
        self.stride = max(1, int(cfg.patch_size * (1 - cfg.overlap_ratio)))
        self.use_adaptive_masking = cfg.use_adaptive_masking

        # Peak detector (NeuroKit2-based)
        if cfg.use_adaptive_masking:
            self.peak_detector = ECGPeakDetector(sampling_rate=cfg.sampling_frequency)

        # Input norm & patch embed (stride for overlap)
        self.input_norm = nn.InstanceNorm1d(cfg.in_channels, affine=False, eps=1e-6)
        self.patch_embed = nn.Sequential(
            nn.Conv1d(cfg.in_channels, cfg.embed_dim * 2, cfg.patch_size, self.stride),
            nn.GELU(),
            nn.Conv1d(cfg.embed_dim * 2, cfg.embed_dim, 1),
            nn.BatchNorm1d(cfg.embed_dim),
            nn.GELU(),
        )

        # Encoder
        self.encoder_blocks = nn.ModuleList([
            TransformerBlock(cfg.embed_dim, cfg.num_heads, cfg.mlp_ratio)
            for _ in range(cfg.depth)
        ])
        self.encoder_norm = nn.LayerNorm(cfg.embed_dim)

        # Decoder
        self.mask_token = nn.Parameter(torch.zeros(1, 1, cfg.decoder_embed_dim))
        self.decoder_embed = nn.Linear(cfg.embed_dim, cfg.decoder_embed_dim)
        self.decoder_blocks = nn.ModuleList([
            TransformerBlock(cfg.decoder_embed_dim, cfg.decoder_num_heads, cfg.mlp_ratio)
            for _ in range(cfg.decoder_depth)
        ])
        self.decoder_norm = nn.LayerNorm(cfg.decoder_embed_dim)

        # Prediction head
        patch_dim = cfg.in_channels * cfg.patch_size
        self.pred_head = nn.Sequential(
            nn.Linear(cfg.decoder_embed_dim, cfg.decoder_embed_dim * 2),
            nn.GELU(),
            nn.Linear(cfg.decoder_embed_dim * 2, patch_dim),
        )
        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.mask_token, std=0.02)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def patchify(self, x):
        """[B, C, L] -> [B, N, C*P] with optional overlap."""
        B, C, L = x.shape
        P, S = self.patch_size, self.stride
        patches = x.unfold(2, P, S)  # [B, C, N, P]
        return patches.permute(0, 2, 1, 3).reshape(B, patches.shape[2], C * P)

    def unpatchify(self, patches, L):
        """[B, N, C*P] -> [B, C, L] with overlap averaging."""
        B, N, _ = patches.shape
        C, P, S = self.in_channels, self.patch_size, self.stride
        patches = patches.view(B, N, C, P).permute(0, 2, 1, 3)  # [B, C, N, P]
        patches_flat = patches.permute(0, 1, 3, 2).reshape(B, C * P, N)
        output = F.fold(patches_flat, output_size=(1, L), kernel_size=(1, P), stride=(1, S))
        output = output.squeeze(2)  # [B, C, L]
        ones = torch.ones_like(patches_flat)
        divisor = F.fold(ones, output_size=(1, L), kernel_size=(1, P), stride=(1, S))
        divisor = divisor.squeeze(2).clamp(min=1)
        return output / divisor

    def get_peak_weights(self, x, r_peaks=None):
        """Compute per-patch weights based on R-peak presence for overlapping patches.
        
        Patches with R-peaks get low weight (kept), patches without get high weight (masked).
        """
        B, C, L = x.shape
        P, S = self.patch_size, self.stride
        N = (L - P) // S + 1

        if r_peaks is None and self.use_adaptive_masking:
            # Auto-detect R-peaks using NeuroKit2 peak detector
            r_peaks_list = []
            for b in range(B):
                signal = x[b, 0].detach().cpu().numpy()
                peaks, _ = self.peak_detector.detect_r_peaks(signal, return_indices=False)
                r_peaks_list.append(torch.from_numpy(peaks).to(x.device))
            r_peaks = torch.stack(r_peaks_list, dim=0)

        if r_peaks is not None:
            weights = torch.zeros(B, N, device=x.device)
            for i in range(N):
                start = i * S
                end = start + P
                if end <= r_peaks.shape[1]:
                    patch_peaks = r_peaks[:, start:end]
                    # Low weight for peak patches → kept; high weight for no-peak → masked
                    weights[:, i] = 1.0 - patch_peaks.mean(dim=-1) + 0.1
            return weights
        else:
            return torch.ones(B, N, device=x.device)

    def adaptive_masking(self, x, weights, mask_ratio):
        """Adaptive masking based on peak weights.
        
        Modulates random noise by weights so that peak-rich patches
        (low weight) are more likely to be kept.
        """
        B, N, D = x.shape
        len_keep = int(N * (1 - mask_ratio))

        # Weighted noise: low weight → small noise → sorted first → kept
        noise = torch.rand(B, N, device=x.device) * weights
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, 1, ids_keep.unsqueeze(-1).expand(-1, -1, D))

        mask = torch.ones(B, N, device=x.device)
        mask[:, :len_keep] = 0
        mask = torch.gather(mask, 1, ids_restore)

        return x_masked, mask, ids_restore

    @staticmethod
    def random_masking(x, mask_ratio):
        """Standard random masking. Returns: x_masked, mask, ids_restore."""
        B, N, D = x.shape
        len_keep = int(N * (1 - mask_ratio))
        noise = torch.rand(B, N, device=x.device)
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, 1, ids_keep.unsqueeze(-1).expand(-1, -1, D))
        mask = torch.ones(B, N, device=x.device)
        mask[:, :len_keep] = 0
        mask = torch.gather(mask, 1, ids_restore)
        return x_masked, mask, ids_restore

    def forward(self, x, r_peaks=None):
        """x: [B, C, L], r_peaks: [B, L] optional -> loss, pred, mask."""
        assert x.shape[1] == self.in_channels

        x_normed = self.input_norm(x)
        B, C, L = x_normed.shape
        target = self.patchify(x_normed)  # [B, N, C*P]
        N = target.shape[1]

        # Encode
        x_embed = self.patch_embed(x_normed).transpose(1, 2)
        x_tokens = x_embed + sincos_pos_embed(self.cfg.embed_dim, N, x.device)

        # Masking: adaptive during training if enabled, random otherwise
        if self.use_adaptive_masking and self.training:
            weights = self.get_peak_weights(x, r_peaks)
            x_masked, mask, ids_restore = self.adaptive_masking(x_tokens, weights, self.mask_ratio)
        else:
            x_masked, mask, ids_restore = self.random_masking(x_tokens, self.mask_ratio)

        for blk in self.encoder_blocks:
            x_masked = blk(x_masked)
        encoded = self.encoder_norm(x_masked)

        # Decode
        dec = self.decoder_embed(encoded)
        N_keep, N_mask = dec.shape[1], N - dec.shape[1]
        dec_full = torch.cat([dec, self.mask_token.expand(B, N_mask, -1)], dim=1)
        dec_full = torch.gather(dec_full, 1, ids_restore.unsqueeze(-1).expand(-1, -1, dec.shape[-1]))
        dec_full = dec_full + sincos_pos_embed(self.cfg.decoder_embed_dim, N, x.device)

        for blk in self.decoder_blocks:
            dec_full = blk(dec_full)
        pred_patches = self.pred_head(self.decoder_norm(dec_full))  # [B, N, C*P]

        # Loss computation
        if self.overlap_ratio > 0:
            pred_signal = self.unpatchify(pred_patches, L)
            target_signal = x_normed
            mask_expanded = mask.unsqueeze(1).unsqueeze(-1)
            mask_expanded = mask_expanded.expand(B, C, N, self.patch_size)
            mask_time = F.fold(
                mask_expanded.reshape(B, C * self.patch_size, N),
                output_size=(1, L),
                kernel_size=(1, self.patch_size),
                stride=(1, self.stride),
            ).squeeze(2)
            mask_time = (mask_time > 0).float()
            loss = (F.l1_loss(pred_signal, target_signal, reduction='none') * mask_time).sum() / mask_time.sum().clamp(min=1)
        else:
            loss = (F.l1_loss(pred_patches, target, reduction='none').mean(-1) * mask).sum() / mask.sum()

        return loss, pred_patches, mask

    def encode(self, x, n_blocks=None):
        """Encode only (for downstream tasks). x: [B, C, L] -> [B, N, D]

        Args:
            n_blocks: if set, only run the first *n_blocks* encoder blocks
                      (e.g. 4 out of 6).  None → use all blocks.
        """
        x_normed = self.input_norm(x)
        x_embed = self.patch_embed(x_normed).transpose(1, 2)  # [B, N, D]
        N = x_embed.shape[1]
        x_tokens = x_embed + sincos_pos_embed(self.cfg.embed_dim, N, x.device)

        blocks = self.encoder_blocks if n_blocks is None else self.encoder_blocks[:n_blocks]
        for blk in blocks:
            x_tokens = blk(x_tokens)

        return self.encoder_norm(x_tokens)


def build_model_for_dataset(
    window_size,
    in_channels,
    patch_size=64,
    embed_dim=128,
    depth=6,
    num_heads=8,
    mlp_ratio=4.0,
    decoder_embed_dim=64,
    decoder_depth=4,
    decoder_num_heads=4,
    mask_ratio=0.55,
    overlap_ratio=0.5,
    use_adaptive_masking=True,
    sampling_frequency=500.0,
):
    """Build ECGMAE model with BOTH overlap AND peak-adaptive masking."""
    stride = max(1, int(patch_size * (1 - overlap_ratio)))
    num_patches = (window_size - patch_size) // stride + 1 if overlap_ratio > 0 else window_size // patch_size
    return ECGMAE(MAEConfig(
        in_channels=in_channels,
        patch_size=patch_size,
        embed_dim=embed_dim,
        depth=depth,
        num_heads=num_heads,
        mlp_ratio=mlp_ratio,
        decoder_embed_dim=decoder_embed_dim,
        decoder_depth=decoder_depth,
        decoder_num_heads=decoder_num_heads,
        mask_ratio=mask_ratio,
        max_patches=num_patches,
        overlap_ratio=overlap_ratio,
        use_adaptive_masking=use_adaptive_masking,
        sampling_frequency=sampling_frequency,
    ))
