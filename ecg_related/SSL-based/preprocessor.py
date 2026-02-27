# preprocessor.py
import glob
import os
from typing import List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

import wfdb


def _normalize_window(x: np.ndarray) -> np.ndarray:
    """
    x: [C, L]  (channels first)
    Per-channel z-score within the window.
    """
    mean = x.mean(axis=1, keepdims=True)
    std = x.std(axis=1, keepdims=True) + 1e-6
    return (x - mean) / std


def _window_signal(sig: np.ndarray,
                   window_size: int,
                   step_size: int) -> List[np.ndarray]:
    """
    sig: [T, C] (time first), returns list of [C, window_size] windows.
    """
    T, C = sig.shape
    windows = []
    for start in range(0, T - window_size + 1, step_size):
        end = start + window_size
        chunk = sig[start:end, :]    # [window_size, C]
        chunk = chunk.T              # [C, window_size]
        chunk = _normalize_window(chunk)
        windows.append(chunk.astype(np.float32))
    return windows


class PTBWindowedDataset(Dataset):
    """
    Loads PTB Diagnostic ECG database, slices WFDB records into windows.

    Returns tensors of shape [C, L] for each window.
    """
    def __init__(self,
                 root_dir: str,
                 window_size: int = 4096,
                 step_size: int = 2048,
                 use_15_leads: bool = False):
        super().__init__()
        self.samples: List[np.ndarray] = []

        # Collect all .dat paths
        dat_paths = sorted(
            glob.glob(os.path.join(root_dir, "patient*", "*.dat"))
        )

        for dat_path in dat_paths:
            base = os.path.splitext(dat_path)[0]
            try:
                rec = wfdb.rdrecord(base)
            except Exception as e:
                print(f"[WARN] Failed to read {base}: {e}")
                continue

            sig = rec.p_signal  # [T, nsig]
            if sig is None:
                continue

            sig = np.asarray(sig, dtype=np.float32)  # [T, nsig]
            nsig = sig.shape[1]

            if use_15_leads:
                # Use all available signals as channels
                pass
            else:
                # Use the first 12 leads (i, ii, iii, avr, avl, avf, v1–v6)
                if nsig < 12:
                    continue
                sig = sig[:, :12]

            windows = _window_signal(sig,
                                     window_size=window_size,
                                     step_size=step_size)
            self.samples.extend(windows)

        print(f"[PTBWindowedDataset] Loaded {len(self.samples)} windows.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        x = self.samples[idx]  # [C, L]
        return torch.from_numpy(x)


class WECGWindowedDataset(Dataset):
    """
    Loads windows from the new wECG .npy files in `wECG_dataset_npy`.

    File layout example (per dataset id):
      - dataset_001_LA_V3.npy
      - dataset_001_LA_V5.npy
      - dataset_001_LA_A.npy
      - dataset_001_LA_A_self.npy
      - dataset_001_info.npy  (metadata, skipped)

    Each lead .npy is a structured array with fields including:
      - 'reference_12_lead': [T, 12]
      - 'wECG': [T, 2]
      - plus small fields like training/testing splits and metadata

    This dataset extracts either the reference 12-lead (use_reference=True)
    or the 2-lead wearable ECG (use_reference=False), then windows it.

    When random_n_channels > 0 **and** stored windows have more channels than
    random_n_channels, __getitem__ randomly selects random_n_channels channels
    on-the-fly (a form of channel-level data augmentation).  This allows
    pretraining on 12-lead data while keeping the model in_channels=2.
    """
    def __init__(self,
                 npy_dir: str,
                 use_reference: bool = True,
                 window_size: int = 4096,
                 step_size: int = 2048,
                 include_variants: Tuple[str, ...] = ("LA_V3", "LA_V5", "LA_A", "LA_A_self"),
                 random_n_channels: int = 0):
        super().__init__()
        self.samples: List[np.ndarray] = []
        self.random_n_channels = random_n_channels

        # Collect all variant files; skip the *_info.npy files
        patterns = [f"dataset_*_{v}.npy" for v in include_variants]
        npy_paths: List[str] = []
        for pat in patterns:
            npy_paths.extend(glob.glob(os.path.join(npy_dir, pat)))
        npy_paths = sorted(npy_paths)

        if not npy_paths:
            print(f"[WARN] No dataset_*_{{{', '.join(include_variants)}}}.npy files found in {npy_dir}")

        for npy_path in npy_paths:
            # Defensive: skip info files if they slip in
            if npy_path.endswith("_info.npy"):
                continue

            try:
                data = np.load(npy_path, allow_pickle=True)
            except Exception as e:
                print(f"[WARN] Failed to load {npy_path}: {e}")
                continue

            sig = None

            # New structured format: 1x1 object array with named fields
            if isinstance(data, np.ndarray) and data.shape == (1, 1) and data.dtype.names:
                rec = data[0, 0]
                key = "reference_12_lead" if use_reference else "wECG"
                if key not in rec.dtype.names:
                    print(f"[WARN] {npy_path}: key '{key}' not found in structured array, skipping.")
                    continue
                sig = rec[key]

            # Back-compat: if someone saved plain arrays or dicts
            elif isinstance(data, dict):
                key = "reference_12_lead" if use_reference else "wECG"
                if key not in data:
                    print(f"[WARN] {npy_path}: expected key '{key}' not found, skipping.")
                    continue
                sig = data[key]
            elif isinstance(data, np.ndarray):
                if data.ndim != 2:
                    print(f"[WARN] {npy_path}: unexpected array shape {data.shape}, expected [T, C], skipping.")
                    continue
                sig = data
            else:
                print(f"[WARN] {npy_path}: unsupported data type {type(data)}, skipping.")
                continue

            sig = np.asarray(sig, dtype=np.float32)
            if sig.ndim != 2:
                print(f"[WARN] {npy_path}: extracted signal has shape {sig.shape}, expected 2D [T, C].")
                continue

            windows = _window_signal(sig, window_size=window_size, step_size=step_size)
            self.samples.extend(windows)

        n_ch = self.samples[0].shape[0] if self.samples else "?"
        print(f"[WECGWindowedDataset] Loaded {len(self.samples)} windows "
              f"({n_ch}-ch) from {len(npy_paths)} files."
              + (f"  Random {random_n_channels}-ch selection enabled." if random_n_channels > 0 else ""))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        x = self.samples[idx]  # [C, L]
        if self.random_n_channels > 0 and x.shape[0] > self.random_n_channels:
            indices = np.random.choice(x.shape[0], self.random_n_channels, replace=False)
            indices.sort()
            x = x[indices]
        return torch.from_numpy(x)
