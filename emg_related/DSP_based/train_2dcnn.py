"""
EMG Gesture Classification - Simple 2D CNN on Time-Frequency Representations
=============================================================================
Vanilla 2D CNN (no residual connections) on 2-channel EMG spectrograms/scalograms.

Architecture:
    Conv2d(2,32,3) -> BN -> ReLU -> MaxPool
    Conv2d(32,64,3) -> BN -> ReLU -> MaxPool
    Conv2d(64,128,3) -> BN -> ReLU -> MaxPool
    Conv2d(128,256,3) -> BN -> ReLU -> MaxPool
    AdaptiveAvgPool2d(1) -> FC(256,128) -> ReLU -> Dropout -> FC(128,9)

Evaluation modes:
    --eval-mode cross   : Cross-subject (S01-S16 train, S17-S20 test) [default]
    --eval-mode intra   : Intra-subject (rep-based split, no leakage)

Time-frequency transforms:
    --transform stft    : Short-Time Fourier Transform [default]
    --transform cwt     : Continuous Wavelet Transform (Morlet, GPU-accelerated)
    --transform logmel  : Log-Mel Spectrogram

Stack features (appended to frequency axis, channels stay 2):
    --stack rms,mav,wl,psd : any combination

Intra-subject options:
    --n-folds 5         : 5-fold rep-based CV (default 1 = single split)
    --augment           : Enable data augmentation + mixup
    --warmup-epochs 5   : Linear LR warmup

Usage:
    python train_cnn2d.py --window 16s --transform cwt --eval-mode intra
    python train_cnn2d.py --window 16s --transform stft --stack rms,mav,wl
    python train_cnn2d.py --window 16s --transform logmel --eval-mode cross
    python train_cnn2d.py --window 16s --eval-mode intra --n-folds 5 --augment

    """

import gc
import numpy as np
import argparse
import os
import sys
import json
from datetime import datetime
from time import perf_counter

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.amp import autocast, GradScaler

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns


# ==================== Logging ====================

class Tee:
    def __init__(self, filepath):
        self.terminal = sys.stdout
        self.log = open(filepath, 'w', encoding='utf-8')
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()
    def flush(self):
        self.terminal.flush()
        self.log.flush()
    def close(self):
        self.log.close()


# ==================== Constants ====================

GESTURE_NAMES = {
    1: "Hand Close", 2: "Hand Open", 3: "Wrist Flex", 4: "Wrist Ext",
    5: "Point Index", 7: "Little Finger", 8: "Tripod",
    9: "Thumb Flex", 10: "Middle Finger",
}
EXCLUDED_GESTURES = {6}
FS_EMG = 2000

VALID_STACK_FEATURES = {'rms', 'mav', 'wl', 'psd'}


def parse_stack(stack_str):
    if not stack_str or stack_str.lower() == 'none':
        return []
    parts = [p.strip().lower() for p in stack_str.split(',')]
    invalid = set(parts) - VALID_STACK_FEATURES
    if invalid:
        raise ValueError(f"Invalid stack features: {invalid}. Valid: {sorted(VALID_STACK_FEATURES)}")
    seen = set()
    result = []
    for p in parts:
        if p not in seen:
            seen.add(p)
            result.append(p)
    return result


# ==================== Data Loading ====================

def load_raw_windowed_data(data_dir, label_dir, subjects):
    """Load raw windowed EMG data with repetition info."""
    emg_list, labels, subj_arr, rep_arr = [], [], [], []
    for subj in subjects:
        subj_data_dir = os.path.join(data_dir, subj)
        subj_label_dir = os.path.join(label_dir, subj)
        if not os.path.exists(subj_data_dir):
            print(f"  [!] {subj} not found, skipping")
            continue
        npz_files = sorted([f for f in os.listdir(subj_data_dir) if f.endswith('.npz')])
        count = 0
        for fname in npz_files:
            data_path = os.path.join(subj_data_dir, fname)
            label_path = os.path.join(subj_label_dir, fname)
            if not os.path.exists(label_path):
                continue
            data_npz = np.load(data_path, allow_pickle=True)
            gesture_id = int(data_npz['gesture_id'])
            if gesture_id in EXCLUDED_GESTURES:
                continue
            emg_list.append(data_npz['emg'].astype(np.float32))
            labels.append(gesture_id)
            subj_arr.append(subj)
            rep_arr.append(int(data_npz['repetition']))
            count += 1
        print(f"  [OK] {subj}: {count} samples")
    return emg_list, np.array(labels), np.array(subj_arr), np.array(rep_arr)


# ==================== Rep-Based Splitting ====================

def rep_based_split(rep_arr, test_frac=0.2, val_frac=0.15, seed=42):
    """Split indices by repetition to prevent data leakage from overlapping windows.

    All clips from the same repetition go to the same set (train/val/test).
    With 10 reps, test_frac=0.2, val_frac=0.15:
        test:  2 reps (20%)
        val:   1 rep  (~12.5% of remaining 8)
        train: 7 reps
    """
    unique_reps = sorted(set(rep_arr))
    n_reps = len(unique_reps)
    n_test = max(1, round(n_reps * test_frac))
    n_val = max(1, round((n_reps - n_test) * val_frac))

    rng = np.random.RandomState(seed)
    shuffled = list(rng.permutation(unique_reps))

    test_reps = set(shuffled[:n_test])
    val_reps = set(shuffled[n_test:n_test + n_val])
    train_reps = set(shuffled[n_test + n_val:])

    train_idx = np.where(np.isin(rep_arr, list(train_reps)))[0]
    val_idx = np.where(np.isin(rep_arr, list(val_reps)))[0]
    test_idx = np.where(np.isin(rep_arr, list(test_reps)))[0]

    return train_idx, val_idx, test_idx, train_reps, val_reps, test_reps


def generate_kfold_splits(rep_arr, n_folds=5, seed=42):
    """Generate K-fold splits based on repetitions (Leave-2-Reps-Out).

    With 10 reps and 5 folds, each fold uses:
        test:  2 reps (every rep tested exactly once across all folds)
        val:   1 rep  (rotates across folds)
        train: 7 reps

    Returns list of (train_idx, val_idx, test_idx, train_reps, val_reps, test_reps).
    """
    unique_reps = sorted(set(rep_arr))
    n_reps = len(unique_reps)
    rng = np.random.RandomState(seed)
    shuffled = list(rng.permutation(unique_reps))

    reps_per_fold = n_reps // n_folds
    if reps_per_fold < 1:
        raise ValueError(f"Not enough reps ({n_reps}) for {n_folds} folds")

    folds = []
    for k in range(n_folds):
        start = k * reps_per_fold
        test_reps = set(shuffled[start:start + reps_per_fold])

        # Val = next rep after test block (wrapping)
        val_pos = (start + reps_per_fold) % n_reps
        while shuffled[val_pos] in test_reps:
            val_pos = (val_pos + 1) % n_reps
        val_reps = {shuffled[val_pos]}

        train_reps = set(shuffled) - test_reps - val_reps

        train_idx = np.where(np.isin(rep_arr, list(train_reps)))[0]
        val_idx = np.where(np.isin(rep_arr, list(val_reps)))[0]
        test_idx = np.where(np.isin(rep_arr, list(test_reps)))[0]

        folds.append((train_idx, val_idx, test_idx, train_reps, val_reps, test_reps))

    return folds


# ==================== Dataset ====================

class EMGDataset(Dataset):
    def __init__(self, emg_list, labels, augment=False,
                 noise_std=0.05, scale_range=(0.8, 1.2),
                 shift_frac=0.05, ch_drop_p=0.2):
        self.emg = torch.from_numpy(
            np.stack([e.T for e in emg_list], axis=0).astype(np.float32))
        self.labels = torch.from_numpy(labels.astype(np.int64))
        self.augment = augment
        self.noise_std = noise_std
        self.scale_range = scale_range
        self.shift_frac = shift_frac
        self.ch_drop_p = ch_drop_p

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        x = self.emg[idx]  # (C, T)
        if self.augment:
            x = self._apply_augment(x)
        return x, self.labels[idx]

    def _apply_augment(self, x):
        """Apply random augmentations to raw EMG signal (C, T)."""
        C, T = x.shape
        # 1. Additive Gaussian noise (p=0.8)
        if torch.rand(1).item() < 0.8:
            sigma = torch.rand(1).item() * self.noise_std
            x = x + torch.randn_like(x) * sigma
        # 2. Random per-channel amplitude scaling (p=0.8)
        if torch.rand(1).item() < 0.8:
            lo, hi = self.scale_range
            scale = torch.empty(C, 1).uniform_(lo, hi)
            x = x * scale
        # 3. Random circular time shift (p=0.5)
        if torch.rand(1).item() < 0.5:
            max_shift = int(T * self.shift_frac)
            shift = torch.randint(-max_shift, max_shift + 1, (1,)).item()
            x = torch.roll(x, shifts=shift, dims=-1)
        # 4. Channel dropout: zero one channel (p=ch_drop_p)
        if torch.rand(1).item() < self.ch_drop_p:
            ch = torch.randint(0, C, (1,)).item()
            x[ch] = 0.0
        return x


# ==================== Time-Frequency Transforms ====================

class SpectrogramTransform(nn.Module):
    def __init__(self, n_fft=256, hop_length=64, normalized=True):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.normalized = normalized
        self.register_buffer('window', torch.hann_window(n_fft))

    def forward(self, x):
        B, C, T = x.shape
        specs = []
        for ch in range(C):
            stft = torch.stft(x[:, ch, :], n_fft=self.n_fft, hop_length=self.hop_length,
                              win_length=self.n_fft, window=self.window,
                              return_complex=True, normalized=self.normalized)
            specs.append(torch.log1p(stft.abs()))
        return torch.stack(specs, dim=1)


class CWTTransform(nn.Module):
    def __init__(self, n_scales=128, f_min=20.0, f_max=450.0, fs=2000,
                 sigma=6.0, hop_length=64):
        super().__init__()
        self.hop_length = hop_length
        self.n_scales = n_scales
        freqs = np.logspace(np.log10(f_min), np.log10(f_max), n_scales)
        filters_real, filters_imag = [], []
        max_len = 0
        for f in freqs:
            scale = sigma / (2 * np.pi * f)
            t_half = int(4 * scale * fs)
            t = np.arange(-t_half, t_half + 1) / fs
            gaussian = np.exp(-t**2 / (2 * scale**2))
            w_real = gaussian * np.cos(2 * np.pi * f * t)
            w_imag = gaussian * np.sin(2 * np.pi * f * t)
            norm = np.sqrt(np.sum(w_real**2 + w_imag**2))
            filters_real.append(w_real / norm)
            filters_imag.append(w_imag / norm)
            max_len = max(max_len, len(w_real))
        if max_len % 2 == 0:
            max_len += 1
        weight_real = np.zeros((n_scales, 1, max_len), dtype=np.float32)
        weight_imag = np.zeros((n_scales, 1, max_len), dtype=np.float32)
        for i, (wr, wi) in enumerate(zip(filters_real, filters_imag)):
            pad_l = (max_len - len(wr)) // 2
            pad_r = max_len - len(wr) - pad_l
            weight_real[i, 0] = np.pad(wr, (pad_l, pad_r))
            weight_imag[i, 0] = np.pad(wi, (pad_l, pad_r))
        self.register_buffer('weight_real', torch.from_numpy(weight_real))
        self.register_buffer('weight_imag', torch.from_numpy(weight_imag))
        self.pad_size = max_len // 2
        print(f"  [CWT] {n_scales} scales, f=[{f_min:.0f}-{f_max:.0f}]Hz, "
              f"filter_len={max_len}, hop={hop_length}")

    def forward(self, x):
        B, C, T = x.shape
        specs = []
        for ch in range(C):
            sig = x[:, ch:ch+1, :]
            sig_padded = F.pad(sig, (self.pad_size, self.pad_size), mode='reflect')
            real = F.conv1d(sig_padded, self.weight_real)
            imag = F.conv1d(sig_padded, self.weight_imag)
            mag = torch.log1p(torch.sqrt(real**2 + imag**2 + 1e-8))
            if self.hop_length > 1:
                mag = F.avg_pool1d(mag, self.hop_length, self.hop_length)
            specs.append(mag)
        return torch.stack(specs, dim=1)


class LogMelTransform(nn.Module):
    def __init__(self, n_fft=256, hop_length=64, n_mels=128,
                 f_min=20.0, f_max=450.0, fs=2000):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.register_buffer('window', torch.hann_window(n_fft))
        mel_fb = self._build_mel_fb(n_mels, n_fft, fs, f_min, f_max)
        self.register_buffer('mel_fb', torch.from_numpy(mel_fb))
        print(f"  [LogMel] n_mels={n_mels}, n_fft={n_fft}, hop={hop_length}")

    @staticmethod
    def _hz_to_mel(hz): return 2595.0 * np.log10(1.0 + hz / 700.0)
    @staticmethod
    def _mel_to_hz(mel): return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)

    def _build_mel_fb(self, n_mels, n_fft, fs, f_min, f_max):
        n_freqs = n_fft // 2 + 1
        mel_pts = np.linspace(self._hz_to_mel(f_min), self._hz_to_mel(f_max), n_mels + 2)
        hz_pts = self._mel_to_hz(mel_pts)
        bins = np.floor((n_fft + 1) * hz_pts / fs).astype(np.int64)
        fb = np.zeros((n_mels, n_freqs), dtype=np.float32)
        for i in range(n_mels):
            l, c, r = bins[i], bins[i+1], bins[i+2]
            if c > l:
                for j in range(l, c):
                    fb[i, j] = (j - l) / (c - l)
            if r > c:
                for j in range(c, r):
                    fb[i, j] = (r - j) / (r - c)
        return fb

    def forward(self, x):
        B, C, T = x.shape
        specs = []
        for ch in range(C):
            stft = torch.stft(x[:, ch, :], n_fft=self.n_fft, hop_length=self.hop_length,
                              win_length=self.n_fft, window=self.window,
                              return_complex=True, normalized=True)
            power = stft.abs().pow(2)
            specs.append(torch.log1p(torch.matmul(self.mel_fb, power)))
        return torch.stack(specs, dim=1)


# ==================== Stack Feature Layer ====================

class StackFeatureLayer(nn.Module):
    """Sliding-window envelope features appended to frequency axis."""
    def __init__(self, stack_list, window_size=256, hop_length=64):
        super().__init__()
        self.stack_list = stack_list
        self.window_size = window_size
        self.hop_length = hop_length
        self.n_features = len(stack_list)
        print(f"  [Stack] features={stack_list}, window={window_size}, hop={hop_length}")

    def forward(self, raw_signal, target_time_len):
        B, C, T = raw_signal.shape
        x_unfold = raw_signal.unfold(2, self.window_size, self.hop_length)
        features = []
        for feat_name in self.stack_list:
            if feat_name == 'rms':
                feat = torch.sqrt(torch.mean(x_unfold ** 2, dim=-1) + 1e-8)
            elif feat_name == 'mav':
                feat = torch.mean(torch.abs(x_unfold), dim=-1)
            elif feat_name == 'wl':
                feat = torch.sum(torch.abs(torch.diff(x_unfold, dim=-1)), dim=-1)
            elif feat_name == 'psd':
                feat = torch.mean(x_unfold ** 2, dim=-1)
            else:
                continue
            features.append(feat)
        stack = torch.stack(features, dim=2)
        n_windows = stack.shape[-1]
        if n_windows != target_time_len:
            stack = F.interpolate(
                stack.reshape(B * C, self.n_features, n_windows),
                size=target_time_len, mode='linear', align_corners=False
            ).reshape(B, C, self.n_features, target_time_len)
        return torch.log1p(stack)


# ==================== Simple 2D CNN ====================

class SimpleCNN2d(nn.Module):
    """Vanilla 2D CNN - no residual connections, just conv+bn+relu+pool."""
    def __init__(self, in_channels=2, num_classes=9, dropout=0.3,
                 transform_type='stft', n_fft=256, hop_length=64,
                 n_scales=128, f_min=20.0, f_max=450.0, fs=2000, n_mels=128,
                 stack_list=None):
        super().__init__()
        self.stack_layer = None
        if stack_list:
            self.stack_layer = StackFeatureLayer(
                stack_list, window_size=n_fft, hop_length=hop_length)
        if transform_type == 'cwt':
            self.tf_transform = CWTTransform(n_scales, f_min, f_max, fs, hop_length=hop_length)
        elif transform_type == 'logmel':
            self.tf_transform = LogMelTransform(n_fft, hop_length, n_mels, f_min, f_max, fs)
        else:
            self.tf_transform = SpectrogramTransform(n_fft, hop_length)
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256), nn.ReLU(inplace=True), nn.MaxPool2d(2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Linear(256, 128), nn.ReLU(inplace=True),
            nn.Dropout(dropout), nn.Linear(128, num_classes),
        )

    def forward(self, x):
        tf_out = self.tf_transform(x)
        if self.stack_layer is not None:
            stack_out = self.stack_layer(x, tf_out.shape[-1])
            tf_out = torch.cat([tf_out, stack_out], dim=2)
        x = self.features(tf_out)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


# ==================== Training / Evaluation ====================

def mixup_data(x, y, alpha=0.2):
    if alpha <= 0:
        return x, y, y, 1.0
    lam = np.random.beta(alpha, alpha)
    lam = max(lam, 1 - lam)
    index = torch.randperm(x.size(0), device=x.device)
    mixed_x = lam * x + (1 - lam) * x[index]
    return mixed_x, y, y[index], lam


def mixup_criterion(criterion, logits, y_a, y_b, lam):
    return lam * criterion(logits, y_a) + (1 - lam) * criterion(logits, y_b)


def train_epoch(model, loader, criterion, optimizer, device, scaler, mixup_alpha=0.0):
    model.train()
    total_loss, correct, total = 0, 0, 0
    use_amp = scaler is not None
    use_mixup = mixup_alpha > 0
    for emg, y in loader:
        emg = emg.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        if use_mixup:
            emg, y_a, y_b, lam = mixup_data(emg, y, mixup_alpha)
        optimizer.zero_grad(set_to_none=True)
        if use_amp:
            with autocast('cuda'):
                logits = model(emg)
                if use_mixup:
                    loss = mixup_criterion(criterion, logits, y_a, y_b, lam)
                else:
                    loss = criterion(logits, y)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(emg)
            if use_mixup:
                loss = mixup_criterion(criterion, logits, y_a, y_b, lam)
            else:
                loss = criterion(logits, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
        total_loss += loss.item() * len(y)
        if use_mixup:
            correct += (lam * (logits.argmax(1) == y_a).float()
                        + (1 - lam) * (logits.argmax(1) == y_b).float()).sum().item()
        else:
            correct += (logits.argmax(1) == y).sum().item()
        total += len(y)
    return total_loss / total, correct / total


def evaluate(model, loader, criterion, device, use_amp=False):
    model.eval()
    total_loss = 0
    all_preds, all_labels = [], []
    with torch.no_grad():
        for emg, y in loader:
            emg = emg.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            if use_amp:
                with autocast('cuda'):
                    logits = model(emg)
                    loss = criterion(logits, y)
            else:
                logits = model(emg)
                loss = criterion(logits, y)
            total_loss += loss.item() * len(y)
            all_preds.extend(logits.argmax(1).cpu().numpy())
            all_labels.extend(y.cpu().numpy())
    return total_loss / len(all_labels), accuracy_score(all_labels, all_preds), \
           np.array(all_preds), np.array(all_labels)


# ==================== Visualization ====================

def plot_confusion_matrix(y_true, y_pred, output_path, title="Confusion Matrix"):
    labels = sorted(set(y_true) | set(y_pred))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    names = [GESTURE_NAMES.get(l, f"G{l}") for l in labels]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0],
                xticklabels=names, yticklabels=names)
    axes[0].set_title(f'{title} (Counts)')
    axes[0].set_xlabel('Predicted'); axes[0].set_ylabel('True')
    axes[0].tick_params(axis='x', rotation=45)
    sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues', ax=axes[1],
                xticklabels=names, yticklabels=names)
    axes[1].set_title(f'{title} (Normalized)')
    axes[1].set_xlabel('Predicted'); axes[1].set_ylabel('True')
    axes[1].tick_params(axis='x', rotation=45)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")


def plot_confusion_matrix_pct(y_true, y_pred, output_path, title="Confusion Matrix"):
    """Plot a single normalized (percentage) confusion matrix."""
    labels = sorted(set(y_true) | set(y_pred))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    cm_pct = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
    names = [GESTURE_NAMES.get(l, f"G{l}") for l in labels]
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm_pct, annot=True, fmt='.1f', cmap='Blues', ax=ax,
                xticklabels=names, yticklabels=names,
                vmin=0, vmax=100, cbar_kws={'label': '%'})
    ax.set_title(title)
    ax.set_xlabel('Predicted'); ax.set_ylabel('True')
    ax.tick_params(axis='x', rotation=45)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")


def plot_per_class_accuracy(y_true, y_pred, output_path, title=""):
    labels = sorted(set(y_true))
    accuracies = [accuracy_score(y_true[y_true == g], y_pred[y_true == g]) * 100 for g in labels]
    names = [GESTURE_NAMES.get(g, f"G{g}") for g in labels]
    fig, ax = plt.subplots(figsize=(12, 5))
    bars = ax.bar(range(len(labels)), accuracies, color='steelblue', edgecolor='black')
    for bar, acc in zip(bars, accuracies):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{acc:.1f}%', ha='center', va='bottom', fontsize=9)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(names, rotation=45, ha='right')
    ax.set_ylabel('Accuracy (%)'); ax.set_title(title); ax.set_ylim(0, 105)
    ax.axhline(y=np.mean(accuracies), color='red', linestyle='--',
               label=f'Mean: {np.mean(accuracies):.1f}%')
    ax.legend(); ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_per_subject_accuracy(y_true, y_pred, subj_arr, output_path, title=""):
    subjects = sorted(set(subj_arr))
    accuracies = [accuracy_score(y_true[subj_arr == s], y_pred[subj_arr == s]) * 100
                  for s in subjects]
    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(subjects, accuracies, color='teal', edgecolor='black')
    for bar, acc in zip(bars, accuracies):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{acc:.1f}%', ha='center', va='bottom', fontsize=10)
    ax.set_ylabel('Accuracy (%)'); ax.set_xlabel('Subject'); ax.set_title(title)
    ax.set_ylim(0, 105)
    ax.axhline(y=np.mean(accuracies), color='red', linestyle='--',
               label=f'Mean: {np.mean(accuracies):.1f}%')
    ax.legend(); ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_intra_summary(subject_results, output_path, title=""):
    subjects = [r['subject'] for r in subject_results]
    accs = [r['test_acc'] for r in subject_results]
    mean_acc, std_acc = np.mean(accs), np.std(accs)
    fig, ax = plt.subplots(figsize=(max(8, len(subjects)*0.8), 5))
    bars = ax.bar(range(len(subjects)), accs, color='teal', edgecolor='black')
    for bar, acc in zip(bars, accs):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{acc:.1f}%', ha='center', va='bottom', fontsize=9)
    ax.set_xticks(range(len(subjects)))
    ax.set_xticklabels(subjects, rotation=45, ha='right')
    ax.set_ylabel('Test Accuracy (%)'); ax.set_ylim(0, 105)
    ax.set_title(f'{title}\nMean: {mean_acc:.1f}% +/- {std_acc:.1f}%')
    ax.axhline(y=mean_acc, color='red', linestyle='--',
               label=f'Mean: {mean_acc:.1f}% +/- {std_acc:.1f}%')
    ax.legend(); ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_training_curves(train_losses, val_losses, train_accs, val_accs, output_path):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    epochs = range(1, len(train_losses) + 1)
    ax1.plot(epochs, train_losses, 'b-', label='Train', lw=1.5)
    ax1.plot(epochs, val_losses, 'r-', label='Val', lw=1.5)
    ax1.set_xlabel('Epoch'); ax1.set_ylabel('Loss'); ax1.set_title('Loss')
    ax1.legend(); ax1.grid(alpha=0.3)
    ax2.plot(epochs, [a*100 for a in train_accs], 'b-', label='Train', lw=1.5)
    ax2.plot(epochs, [a*100 for a in val_accs], 'r-', label='Val', lw=1.5)
    best_ep = np.argmax(val_accs) + 1
    ax2.axvline(x=best_ep, color='green', linestyle='--', alpha=0.7,
                label=f'Best: {max(val_accs)*100:.1f}% (ep {best_ep})')
    ax2.set_xlabel('Epoch'); ax2.set_ylabel('Accuracy (%)'); ax2.set_title('Accuracy')
    ax2.legend(); ax2.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


# ==================== Cross-Subject ====================

def run_cross_subject(window_label, args, device, train_subjs, test_subjs, stack_list):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    windowed_dir = os.path.join(script_dir, "windowed_data", window_label)
    data_dir = os.path.join(windowed_dir, "data")
    label_dir = os.path.join(windowed_dir, "label")

    tf_str = args.transform.upper()
    stack_str = '+'.join(stack_list) if stack_list else 'nostack'
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = os.path.join(args.log_dir,
                           f"{window_label}_CNN2d-{tf_str}-{stack_str}_cross_{timestamp}")
    os.makedirs(log_dir, exist_ok=True)

    tee = Tee(os.path.join(log_dir, "console_output.txt"))
    original_stdout = sys.stdout
    sys.stdout = tee

    print(f"\n{'='*70}")
    print(f"SimpleCNN2d {tf_str} - Cross-Subject | Window: {window_label}")
    print(f"Stack: {stack_str}")
    print(f"Train: {train_subjs} | Test: {test_subjs}")
    print(f"{'='*70}")

    print("\n[Loading data...]")
    train_emg, y_train_all, _, _ = load_raw_windowed_data(data_dir, label_dir, train_subjs)
    test_emg, y_test, subj_test, _ = load_raw_windowed_data(data_dir, label_dir, test_subjs)

    if len(train_emg) == 0 or len(test_emg) == 0:
        print("  ERROR: No data.")
        sys.stdout = original_stdout; tee.close()
        return None

    unique_labels = sorted(set(y_train_all) | set(y_test))
    label_to_idx = {lab: idx for idx, lab in enumerate(unique_labels)}
    idx_to_label = {idx: lab for lab, idx in label_to_idx.items()}
    y_train_mapped = np.array([label_to_idx[l] for l in y_train_all])
    y_test_mapped = np.array([label_to_idx[l] for l in y_test])
    num_classes = len(unique_labels)

    train_idx, val_idx = train_test_split(
        np.arange(len(train_emg)), test_size=args.val_split,
        stratify=y_train_mapped, random_state=42)

    y_train = y_train_mapped[train_idx]
    y_val = y_train_mapped[val_idx]
    print(f"  Stratified val: train={len(train_idx)}, val={len(val_idx)}")

    use_amp = device.type == 'cuda'
    nw = args.num_workers
    train_loader = DataLoader(
        EMGDataset([train_emg[i] for i in train_idx], y_train,
                   augment=args.augment, noise_std=args.noise_std),
        batch_size=args.batch_size, shuffle=True,
        num_workers=nw, pin_memory=True, persistent_workers=nw > 0)
    val_loader = DataLoader(
        EMGDataset([train_emg[i] for i in val_idx], y_val),
        batch_size=args.batch_size, shuffle=False,
        num_workers=nw, pin_memory=True, persistent_workers=nw > 0)
    test_loader = DataLoader(
        EMGDataset(test_emg, y_test_mapped),
        batch_size=args.batch_size, shuffle=False,
        num_workers=nw, pin_memory=True, persistent_workers=nw > 0)

    model = SimpleCNN2d(
        in_channels=2, num_classes=num_classes, dropout=args.dropout,
        transform_type=args.transform, n_fft=256, hop_length=64,
        n_scales=128, f_min=20.0, f_max=450.0, fs=FS_EMG, n_mels=128,
        stack_list=stack_list,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n  Model: SimpleCNN2d | Params: {total_params:,}")

    emg_time = train_emg[0].shape[0]
    try:
        sample = torch.randn(2, 2, emg_time).to(device)
        with torch.no_grad():
            out = model(sample)
        print(f"  Forward OK: {list(sample.shape)} -> {list(out.shape)}")
        del sample, out
    except Exception as e:
        print(f"  Forward FAILED: {e}")
        sys.stdout = original_stdout; tee.close()
        return None

    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=15, T_mult=2, eta_min=1e-6)
    scaler = GradScaler('cuda') if use_amp else None
    mixup_alpha = args.mixup_alpha if args.augment else 0.0

    best_val_acc, best_epoch, patience_counter = 0, 0, 0
    train_losses, val_losses, train_accs, val_accs = [], [], [], []
    best_model_path = os.path.join(log_dir, "best_model.pt")
    t_start = perf_counter()

    for epoch in range(1, args.epochs + 1):
        if args.warmup_epochs > 0 and epoch <= args.warmup_epochs:
            warmup_lr = args.lr * epoch / args.warmup_epochs
            for pg in optimizer.param_groups:
                pg['lr'] = warmup_lr

        train_loss, train_acc = train_epoch(model, train_loader, criterion,
                                            optimizer, device, scaler,
                                            mixup_alpha=mixup_alpha)
        val_loss, val_acc, _, _ = evaluate(model, val_loader, criterion, device, use_amp)
        if epoch > args.warmup_epochs:
            scheduler.step()

        train_losses.append(train_loss); val_losses.append(val_loss)
        train_accs.append(train_acc); val_accs.append(val_acc)

        if epoch % 5 == 0 or epoch == 1 or val_acc > best_val_acc:
            print(f"    Ep {epoch:3d}/{args.epochs} | "
                  f"Train: {train_loss:.4f}/{train_acc*100:.1f}% | "
                  f"Val: {val_loss:.4f}/{val_acc*100:.1f}%")

        if val_acc > best_val_acc:
            best_val_acc, best_epoch, patience_counter = val_acc, epoch, 0
            torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(),
                        'val_acc': val_acc}, best_model_path)
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"    Early stop at epoch {epoch} (best: ep {best_epoch})")
                break

    train_time = perf_counter() - t_start
    print(f"  Best val: {best_val_acc*100:.1f}% (ep {best_epoch}) | Time: {train_time/60:.1f}min")

    checkpoint = torch.load(best_model_path, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    test_loss, test_acc, y_pred_s, y_true_s = evaluate(model, test_loader, criterion, device)

    y_pred = np.array([idx_to_label[i] for i in y_pred_s])
    y_true = np.array([idx_to_label[i] for i in y_true_s])

    print(f"\n>>> TEST ACCURACY: {test_acc*100:.1f}%")

    per_subject = {}
    for subj in sorted(set(subj_test)):
        mask = subj_test == subj
        acc = accuracy_score(y_true[mask], y_pred[mask]) * 100
        per_subject[subj] = acc
        print(f"  {subj}: {acc:.1f}%")

    plot_training_curves(train_losses, val_losses, train_accs, val_accs,
                         os.path.join(log_dir, "training_curves.png"))
    plot_confusion_matrix(y_true, y_pred, os.path.join(log_dir, "confusion_matrix.png"),
                          title=f"CNN2d-{tf_str}-{stack_str} Cross ({window_label}) - {test_acc*100:.1f}%")
    plot_per_class_accuracy(y_true, y_pred, os.path.join(log_dir, "per_gesture_accuracy.png"),
                            title=f"CNN2d-{tf_str} Per-Gesture ({window_label}) - {test_acc*100:.1f}%")
    plot_per_subject_accuracy(y_true, y_pred, subj_test,
                              os.path.join(log_dir, "per_subject_accuracy.png"),
                              title=f"CNN2d-{tf_str} Per-Subject ({window_label}) - {test_acc*100:.1f}%")

    log_data = {
        "timestamp": datetime.now().isoformat(),
        "mode": "cross-subject", "transform": args.transform,
        "stack": args.stack, "window": window_label,
        "test_accuracy": test_acc * 100,
        "best_val_accuracy": best_val_acc * 100,
        "per_subject": per_subject, "params": total_params,
    }
    with open(os.path.join(log_dir, "training_log.json"), 'w') as f:
        json.dump(log_data, f, indent=2)

    del model, optimizer, scheduler, criterion
    gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    sys.stdout = original_stdout; tee.close()

    return {"window": window_label, "transform": args.transform, "stack": args.stack,
            "eval_mode": "cross", "test_acc": test_acc * 100,
            "best_val_acc": best_val_acc * 100, "log_dir": log_dir}


# ==================== Intra-Subject (rep-based split) ====================

def run_intra_subject(window_label, args, device, subjects, stack_list):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    windowed_dir = os.path.join(script_dir, "windowed_data", window_label)
    data_dir = os.path.join(windowed_dir, "data")
    label_dir = os.path.join(windowed_dir, "label")

    tf_str = args.transform.upper()
    stack_str = '+'.join(stack_list) if stack_list else 'nostack'
    n_folds = args.n_folds
    use_kfold = n_folds > 1

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_log_dir = os.path.join(args.log_dir,
                                f"{window_label}_CNN2d-{tf_str}-{stack_str}_intra_{timestamp}")
    os.makedirs(base_log_dir, exist_ok=True)

    tee = Tee(os.path.join(base_log_dir, "console_output.txt"))
    original_stdout = sys.stdout
    sys.stdout = tee

    print(f"\n{'='*70}")
    if use_kfold:
        print(f"SimpleCNN2d {tf_str} - Intra-Subject ({n_folds}-Fold CV)")
    else:
        print(f"SimpleCNN2d {tf_str} - Intra-Subject")
    print(f"{'='*70}")
    print(f"Window: {window_label} | Stack: {stack_str}")
    if use_kfold:
        print(f"Split: Rep-based {n_folds}-Fold (Leave-{10//n_folds}-Reps-Out)")
    else:
        print(f"Split: Rep-based {(1-args.test_split)*100:.0f}% train / "
              f"{args.test_split*100:.0f}% test (no leakage)")
    use_amp = device.type == 'cuda'
    print(f"Device: {device}, AMP: {use_amp}")
    if args.augment:
        print(f"Augmentation: ON (noise={args.noise_std}, mixup={args.mixup_alpha}, "
              f"warmup={args.warmup_epochs}ep)")
    else:
        print(f"Augmentation: OFF")
    print(f"{'='*70}")

    subject_results = []
    total_params = 0
    global_y_true = []
    global_y_pred = []

    for subj in subjects:
        print(f"\n{'_'*60}")
        print(f"  Subject: {subj}")
        print(f"{'_'*60}")

        emg_list, labels, _, rep_arr = load_raw_windowed_data(data_dir, label_dir, [subj])
        if len(emg_list) == 0:
            print(f"  No data, skipping."); continue

        unique_labels = sorted(set(labels))
        label_to_idx = {lab: idx for idx, lab in enumerate(unique_labels)}
        idx_to_label = {idx: lab for lab, idx in label_to_idx.items()}
        labels_mapped = np.array([label_to_idx[l] for l in labels])
        num_classes = len(unique_labels)

        # ---- Rep-based splitting ----
        if use_kfold:
            folds = generate_kfold_splits(rep_arr, n_folds=n_folds, seed=42)
        else:
            folds = [rep_based_split(rep_arr, args.test_split, args.val_split, seed=42)]

        all_y_true = []
        all_y_pred = []
        fold_accs = []
        fold_val_accs = []
        fold_epochs = []
        train_losses_last, val_losses_last = [], []
        train_accs_last, val_accs_last = [], []
        t_start = perf_counter()

        for fold_i, (train_idx, val_idx, test_idx,
                      train_reps, val_reps, test_reps) in enumerate(folds):
            y_train = labels_mapped[train_idx]
            y_val = labels_mapped[val_idx]
            y_test = labels_mapped[test_idx]

            if use_kfold:
                print(f"\n  Fold {fold_i+1}/{n_folds}: "
                      f"train={sorted(train_reps)}({len(train_idx)}) "
                      f"val={sorted(val_reps)}({len(val_idx)}) "
                      f"test={sorted(test_reps)}({len(test_idx)})")
            else:
                print(f"  Rep-based split (no leakage):")
                print(f"    Train reps: {sorted(train_reps)} -> {len(train_idx)} clips")
                print(f"    Val   reps: {sorted(val_reps)} -> {len(val_idx)} clips")
                print(f"    Test  reps: {sorted(test_reps)} -> {len(test_idx)} clips")

            nw = args.num_workers
            train_loader = DataLoader(
                EMGDataset([emg_list[i] for i in train_idx], y_train,
                           augment=args.augment, noise_std=args.noise_std),
                batch_size=args.batch_size, shuffle=True,
                num_workers=nw, pin_memory=True, persistent_workers=nw > 0)
            val_loader = DataLoader(
                EMGDataset([emg_list[i] for i in val_idx], y_val),
                batch_size=args.batch_size, shuffle=False,
                num_workers=nw, pin_memory=True, persistent_workers=nw > 0)
            test_loader = DataLoader(
                EMGDataset([emg_list[i] for i in test_idx], y_test),
                batch_size=args.batch_size, shuffle=False,
                num_workers=nw, pin_memory=True, persistent_workers=nw > 0)

            model = SimpleCNN2d(
                in_channels=2, num_classes=num_classes, dropout=args.dropout,
                transform_type=args.transform, n_fft=256, hop_length=64,
                n_scales=128, f_min=20.0, f_max=450.0, fs=FS_EMG, n_mels=128,
                stack_list=stack_list,
            ).to(device)
            total_params = sum(p.numel() for p in model.parameters())

            criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
            optimizer = optim.AdamW(model.parameters(), lr=args.lr,
                                    weight_decay=args.weight_decay)
            scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer, T_0=15, T_mult=2, eta_min=1e-6)
            scaler = GradScaler('cuda') if use_amp else None
            mixup_alpha = args.mixup_alpha if args.augment else 0.0

            best_val_acc, best_epoch, patience_counter = 0, 0, 0
            cur_train_losses, cur_val_losses = [], []
            cur_train_accs, cur_val_accs = [], []
            model_path = os.path.join(base_log_dir,
                f"best_model_{subj}_fold{fold_i}.pt" if use_kfold
                else f"best_model_{subj}.pt")

            for epoch in range(1, args.epochs + 1):
                if args.warmup_epochs > 0 and epoch <= args.warmup_epochs:
                    warmup_lr = args.lr * epoch / args.warmup_epochs
                    for pg in optimizer.param_groups:
                        pg['lr'] = warmup_lr

                train_loss, train_acc = train_epoch(
                    model, train_loader, criterion, optimizer, device, scaler,
                    mixup_alpha=mixup_alpha)
                val_loss, val_acc, _, _ = evaluate(
                    model, val_loader, criterion, device, use_amp)
                if epoch > args.warmup_epochs:
                    scheduler.step()

                cur_train_losses.append(train_loss); cur_val_losses.append(val_loss)
                cur_train_accs.append(train_acc); cur_val_accs.append(val_acc)

                if val_acc > best_val_acc:
                    best_val_acc, best_epoch, patience_counter = val_acc, epoch, 0
                    torch.save({'epoch': epoch,
                                'model_state_dict': model.state_dict(),
                                'val_acc': val_acc}, model_path)
                else:
                    patience_counter += 1

                if not use_kfold and (epoch % 10 == 0 or epoch == 1):
                    print(f"    Ep {epoch:3d} | Train: {train_loss:.4f}/{train_acc*100:.1f}% | "
                          f"Val: {val_loss:.4f}/{val_acc*100:.1f}%")

                if patience_counter >= args.patience:
                    if not use_kfold:
                        print(f"    Early stop at epoch {epoch} (best: ep {best_epoch})")
                    break

            # Evaluate best model on test set
            checkpoint = torch.load(model_path, weights_only=False)
            model.load_state_dict(checkpoint['model_state_dict'])
            _, test_acc, y_pred_s, y_true_s = evaluate(
                model, test_loader, criterion, device)

            fold_accs.append(test_acc * 100)
            fold_val_accs.append(best_val_acc * 100)
            fold_epochs.append(best_epoch)
            train_losses_last = cur_train_losses
            val_losses_last = cur_val_losses
            train_accs_last = cur_train_accs
            val_accs_last = cur_val_accs

            y_pred_orig = [idx_to_label[i] for i in y_pred_s]
            y_true_orig = [idx_to_label[i] for i in y_true_s]
            all_y_pred.extend(y_pred_orig)
            all_y_true.extend(y_true_orig)

            if use_kfold:
                print(f"    -> Fold {fold_i+1}: test={test_acc*100:.1f}% "
                      f"(val={best_val_acc*100:.1f}%, ep {best_epoch})")

            del model, optimizer, scheduler, criterion, scaler
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            if use_kfold and os.path.exists(model_path):
                os.remove(model_path)

        train_time = perf_counter() - t_start

        # Aggregate results
        all_y_true_arr = np.array(all_y_true)
        all_y_pred_arr = np.array(all_y_pred)
        overall_acc = accuracy_score(all_y_true_arr, all_y_pred_arr) * 100

        if use_kfold:
            mean_fold = np.mean(fold_accs)
            std_fold = np.std(fold_accs)
            print(f"\n  >>> {subj} {n_folds}-FOLD CV: {overall_acc:.1f}% "
                  f"(fold mean: {mean_fold:.1f}% +/- {std_fold:.1f}%, "
                  f"{train_time/60:.1f}min)")
        else:
            print(f"\n  >>> {subj} TEST: {overall_acc:.1f}% "
                  f"(val: {fold_val_accs[0]:.1f}%, ep {fold_epochs[0]}, "
                  f"{train_time/60:.1f}min)")

        # Per-subject confusion matrix
        if use_kfold:
            cm_title = f"{subj} CNN2d-{tf_str} ({window_label}) {n_folds}-Fold - {overall_acc:.1f}%"
        else:
            cm_title = f"{subj} CNN2d-{tf_str} ({window_label}) - {overall_acc:.1f}%"
        plot_confusion_matrix(
            all_y_true_arr, all_y_pred_arr,
            os.path.join(base_log_dir, f"confusion_matrix_{subj}.png"),
            title=cm_title)

        # Training curves for single-split only
        if not use_kfold:
            plot_training_curves(train_losses_last, val_losses_last,
                                 train_accs_last, val_accs_last,
                                 os.path.join(base_log_dir, f"training_curves_{subj}.png"))

        per_gesture = {}
        for g in sorted(set(all_y_true_arr)):
            mask = all_y_true_arr == g
            per_gesture[int(g)] = accuracy_score(
                all_y_true_arr[mask], all_y_pred_arr[mask]) * 100

        result_entry = {
            'subject': subj, 'test_acc': overall_acc,
            'train_time_min': train_time / 60,
            'n_total': len(emg_list), 'n_folds': n_folds,
            'num_classes': num_classes, 'per_gesture': per_gesture,
        }
        if use_kfold:
            result_entry.update({
                'fold_accs': fold_accs,
                'fold_mean': np.mean(fold_accs), 'fold_std': np.std(fold_accs),
                'best_val_accs': fold_val_accs, 'best_epochs': fold_epochs,
            })
        else:
            result_entry.update({
                'best_val_acc': fold_val_accs[0], 'best_epoch': fold_epochs[0],
                'epochs_run': len(train_losses_last),
                'n_train': len(folds[0][0]), 'n_val': len(folds[0][1]),
                'n_test': len(folds[0][2]),
            })
        subject_results.append(result_entry)
        global_y_true.extend(all_y_true)
        global_y_pred.extend(all_y_pred)

    if not subject_results:
        print("\nNo subjects completed.")
        sys.stdout = original_stdout; tee.close()
        return None

    accs = [r['test_acc'] for r in subject_results]
    mean_acc, std_acc = np.mean(accs), np.std(accs)

    # ---- Summary table ----
    print(f"\n{'='*70}")
    if use_kfold:
        print(f"INTRA-SUBJECT SUMMARY | CNN2d-{tf_str}-{stack_str} | {window_label} | "
              f"{n_folds}-Fold CV")
        print(f"{'='*70}")
        print(f"{'Subject':<8} {'Total':>6} {'CV Acc':>10} "
              + " ".join([f"{'F'+str(i+1):>6}" for i in range(n_folds)]))
        print("-" * (30 + 7 * n_folds))
        for r in subject_results:
            fold_str = " ".join([f"{a:>5.1f}%" for a in r['fold_accs']])
            print(f"{r['subject']:<8} {r['n_total']:>6} {r['test_acc']:>9.1f}% {fold_str}")
        print(f"\nMean CV Acc: {mean_acc:.1f}% +/- {std_acc:.1f}%")
    else:
        print(f"INTRA-SUBJECT SUMMARY | CNN2d-{tf_str}-{stack_str} | {window_label}")
        print(f"{'='*70}")
        print(f"{'Subject':<8} {'Train':>6} {'Val':>5} {'Test':>5} "
              f"{'Test Acc':>10} {'Val Acc':>10}")
        print("-" * 52)
        for r in subject_results:
            print(f"{r['subject']:<8} {r['n_train']:>6} {r['n_val']:>5} {r['n_test']:>5} "
                  f"{r['test_acc']:>9.1f}% {r['best_val_acc']:>9.1f}%")
        print(f"\nMean: {mean_acc:.1f}% +/- {std_acc:.1f}%")
    print(f"{'='*70}")

    # Summary bar chart
    plot_title = (f"CNN2d-{tf_str}-{stack_str} Intra ({window_label}, {n_folds}-Fold CV)"
                  if use_kfold else f"CNN2d-{tf_str}-{stack_str} Intra ({window_label})")
    plot_intra_summary(subject_results, os.path.join(base_log_dir, "intra_summary.png"),
                       title=plot_title)

    # Aggregated confusion matrix across all subjects (normalized %)
    if len(global_y_true) > 0:
        global_y_true_arr = np.array(global_y_true)
        global_y_pred_arr = np.array(global_y_pred)
        global_acc = accuracy_score(global_y_true_arr, global_y_pred_arr) * 100
        cv_str = f" {n_folds}-Fold CV" if use_kfold else ""
        plot_confusion_matrix_pct(
            global_y_true_arr, global_y_pred_arr,
            os.path.join(base_log_dir, "confusion_matrix_all_subjects.png"),
            title=f"All Subjects CNN2d-{tf_str}{cv_str} ({window_label}) - {global_acc:.1f}%")

    summary = {
        "timestamp": datetime.now().isoformat(),
        "mode": "intra-subject", "transform": args.transform,
        "stack": args.stack, "window": window_label,
        "n_folds": n_folds, "augment": args.augment,
        "mean_accuracy": mean_acc, "std_accuracy": std_acc,
        "per_subject": subject_results,
    }
    with open(os.path.join(base_log_dir, "intra_summary.json"), 'w') as f:
        json.dump(summary, f, indent=2)

    sys.stdout = original_stdout; tee.close()

    return {"window": window_label, "transform": args.transform, "stack": args.stack,
            "eval_mode": "intra", "mean_acc": mean_acc, "std_acc": std_acc,
            "per_subject": subject_results, "log_dir": base_log_dir}


# ==================== MAIN ====================

ALL_WINDOWS = ["2s", "4s", "6s", "8s", "10s", "12s", "14s", "16s"]


def parse_range(s):
    p = s.split('-')
    return [f"S{i:02d}" for i in range(int(p[0][1:]), int(p[1][1:])+1)]


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))

    parser = argparse.ArgumentParser(description='Simple 2D CNN on EMG TF representations')
    parser.add_argument("--window", "-w", default="16s")
    parser.add_argument("--transform", "-t", default="stft", choices=["stft", "cwt", "logmel"])
    parser.add_argument("--stack", "-s", default="none",
                        help="Stack features: none, rms, mav, wl, psd, or comma-separated")
    parser.add_argument("--eval-mode", "-e", default="cross", choices=["cross", "intra"])
    parser.add_argument("--subjects", default=None)
    parser.add_argument("--train", default="S01-S16")
    parser.add_argument("--test", default="S17-S20")
    parser.add_argument("--log-dir", default=os.path.join(script_dir, "results_cnn2d"))

    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--val-split", type=float, default=0.15)
    parser.add_argument("--test-split", type=float, default=0.2)
    parser.add_argument("--label-smoothing", type=float, default=0.05)
    parser.add_argument("--num-workers", type=int, default=0)

    # Augmentation (opt-in, matches ResNet18 script)
    parser.add_argument("--augment", action="store_true",
                        help="Enable data augmentation (noise, scaling, shift, ch-dropout, mixup)")
    parser.add_argument("--no-augment", dest="augment", action="store_false")
    parser.add_argument("--mixup-alpha", type=float, default=0.2,
                        help="Mixup alpha (only used with --augment)")
    parser.add_argument("--noise-std", type=float, default=0.05,
                        help="Max Gaussian noise std for augmentation")
    parser.add_argument("--warmup-epochs", type=int, default=0,
                        help="Linear LR warmup epochs (0=off)")
    parser.add_argument("--n-folds", type=int, default=1,
                        help="Folds for intra-subject CV (1=single rep-split [default], 5=5-fold)")
    parser.set_defaults(augment=False)
    args = parser.parse_args()

    stack_list = parse_stack(args.stack)

    if args.window.lower() == "all":
        window_sizes = ALL_WINDOWS
    elif "," in args.window:
        window_sizes = [w.strip() for w in args.window.split(",")]
    else:
        window_sizes = [args.window]

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    tf_str = args.transform.upper()
    stack_str = '+'.join(stack_list) if stack_list else 'none'
    eval_str = "INTRA" if args.eval_mode == "intra" else "CROSS"

    print(f"\n{'='*70}")
    print(f"SimpleCNN2d | {tf_str} | {eval_str}-SUBJECT")
    print(f"{'='*70}")
    print(f"Windows: {', '.join(window_sizes)}")
    print(f"Transform: {tf_str} | Stack: {stack_str} | Device: {device}")
    if args.eval_mode == "intra":
        print(f"Split: Rep-based (n_folds={args.n_folds})")
        if args.augment:
            print(f"Augmentation: ON | Mixup: {args.mixup_alpha} | "
                  f"Noise: {args.noise_std} | Warmup: {args.warmup_epochs}ep")
    print(f"{'='*70}")

    os.makedirs(args.log_dir, exist_ok=True)
    all_results = []

    for i, wl in enumerate(window_sizes):
        print(f"\n>>> [{i+1}/{len(window_sizes)}] Window: {wl}")

        if args.eval_mode == "intra":
            subjects = [s.strip() for s in args.subjects.split(",")] if args.subjects \
                       else [f"S{i:02d}" for i in range(1, 21)]
            result = run_intra_subject(wl, args, device, subjects, stack_list)
            if result:
                all_results.append(result)
                print(f"    -> Mean: {result['mean_acc']:.1f}% +/- {result['std_acc']:.1f}%")
        else:
            result = run_cross_subject(wl, args, device,
                                       parse_range(args.train),
                                       parse_range(args.test), stack_list)
            if result:
                all_results.append(result)
                print(f"    -> Test: {result['test_acc']:.1f}%")

    print(f"\n{'='*70}")
    print(f"FINAL SUMMARY - SimpleCNN2d {tf_str} {eval_str} (stack={stack_str})")
    print(f"{'='*70}")
    for r in all_results:
        if r['eval_mode'] == 'intra':
            print(f"  {r['window']} -> {r['mean_acc']:.1f}% +/- {r['std_acc']:.1f}%")
        else:
            print(f"  {r['window']} -> {r['test_acc']:.1f}%")
    print(f"{'='*70}")

    summary_path = os.path.join(args.log_dir, "batch_summary.json")
    with open(summary_path, 'w') as f:
        json.dump({"timestamp": datetime.now().isoformat(), "results": all_results},
                   f, indent=2, default=str)
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()