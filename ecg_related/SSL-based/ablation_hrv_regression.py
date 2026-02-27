"""
ablation_hrv_regression.py
Within-subject HRV Regression evaluation.

For Neurokit datasets (Free Form / Steady State / Gesture):
  - For EACH subject, train regression head on reps 1-7, test on reps 8-10.
  - Predictions from all subjects are pooled, then accuracy is computed.

For wECG:
  - For each recording (dataset_XXX), temporal split (first 70% train, last 30% test).
  - Pool predictions across all recordings for accuracy.

MAE Encoder: partial fine-tuning (last 2 blocks unfrozen, from 4 ablation configs).

Downstream tasks:
  1. wECG        – 2-channel wearable ECG
  2. Free Form   – REST windows NOT overlapping Static gesture
  3. Steady State – REST windows overlapping Static gesture (gesture_id=0)
  4. Gesture      – gesture windows (gesture_id 1-10)

Accuracy:
  HR  – L1 regression metrics (L1==0, L1<1, L1<2, L1<5, L1<10%*ref)
  HRV – range-based: |pred - true| <= 5% of pooled GT range
  Combined HRV – all 3 (SDNN, RMSSD, pNN50) within threshold
"""
import argparse
import os
import csv
import json
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, mean_absolute_error
from tqdm import tqdm


TRAIN_REPS = set(range(1, 8))    # repetitions 1-7 for training
TEST_REPS  = set(range(8, 11))   # repetitions 8-10 for testing
EXCLUDED_SUBJECTS = {"S06", "S09"}
ALL_SUBJECTS = sorted(
    {f"S{i:02d}" for i in range(1, 21)} - EXCLUDED_SUBJECTS)

WECG_TRAIN_RATIO = 0.7           # temporal split for wECG


# ============================================================================
# Deterministic seeding
# ============================================================================
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================================
# Data loading: Neurokit per-subject data
# ============================================================================
PROCESSED_SR = 500
EVENTS_SR    = 2000
SR_RATIO     = EVENTS_SR // PROCESSED_SR   # 4


def _load_subject_ecg(data_root, subj, channel="chest"):
    """Load raw ECG data for a subject.  Returns {rep_num: 1D array}."""
    subj_dir = os.path.join(data_root, subj, "repetitions")
    if not os.path.exists(subj_dir):
        return None
    ecg_data = {}
    col_name = f"ecg_{channel}"
    for npz_file in os.listdir(subj_dir):
        if not npz_file.endswith('.npz'):
            continue
        try:
            rep_num = int(npz_file.split('_')[1])
        except Exception:
            continue
        npz_path = os.path.join(subj_dir, npz_file)
        data = np.load(npz_path, allow_pickle=True)
        columns = list(data['columns_exg'])
        if col_name not in columns:
            continue
        col_idx = columns.index(col_name)
        ecg_data[rep_num] = data['exg'][:, col_idx]
    return ecg_data if ecg_data else None
    

def _get_static_boundaries(events_root, subj, subj_ecg_data):
    """Parse events.json to get Static gesture boundaries in LOCAL rep coords.

    events.json stores *global* sample indices (across the whole recording at
    2 kHz).  CSV / npz windows use *per-repetition* local indices (starting
    from 0).  We use the REPETITION_START event as offset to convert.

    Returns {rep_num: (local_start_500hz, local_end_500hz)}
    """
    bounds = {}
    events_dir = os.path.join(events_root, subj, "events")
    if not os.path.exists(events_dir):
        return bounds
    for fname in sorted(os.listdir(events_dir)):
        if not fname.endswith('_events.json'):
            continue
        try:
            rep_num = int(fname.split('_')[1])
        except Exception:
            continue
        if rep_num not in subj_ecg_data:
            continue
        with open(os.path.join(events_dir, fname)) as f:
            evt_data = json.load(f)

        rep_start_2k = None
        static_start_2k = static_end_2k = None

        for ev in evt_data.get('events', []):
            if ev['event_type'] == 'REPETITION_START' and rep_start_2k is None:
                rep_start_2k = ev['sample_indices']['exg_idx']
            gid = ev.get('data', {}).get('gesture_id', -1)
            if gid != 0:
                continue
            if ev['event_type'] == 'GESTURE_START':
                static_start_2k = ev['sample_indices']['exg_idx']
            elif ev['event_type'] == 'GESTURE_END':
                static_end_2k = ev['sample_indices']['exg_idx']

        if (static_start_2k is not None and static_end_2k is not None
                and rep_start_2k is not None):
            local_start = (static_start_2k - rep_start_2k) // SR_RATIO
            local_end   = (static_end_2k   - rep_start_2k) // SR_RATIO
            bounds[rep_num] = (local_start, local_end)
    return bounds


def load_neurokit_per_subject(
    subj,
    label_filter,
    data_root="/root/autodl-tmp/processed_data_500hz",
    gt_root="/root/autodl-tmp/neurokit_results",
    events_root="/root/autodl-tmp/raw_20",
    channel="chest",
    input_channel="wrist",
    window_size=5000,
):
    """Load all windows for one subject, organised by repetition.

    Ground truth HRV labels come from *channel* (chest) CSV.
    Input ECG signal comes from *input_channel* (wrist) — duplicated to 2ch.

    Returns dict: {rep_num: list of (ecg_2ch, hr, sdnn, rmssd, pnn50)}
    """
    # Load input ECG from wrist channel
    subj_ecg = _load_subject_ecg(data_root, subj, input_channel)
    if subj_ecg is None:
        return {}

    # Ground truth windows still from chest (more reliable reference)
    windows_path = os.path.join(
        gt_root, "per_subject_windows", f"{subj}_{channel}_windows.csv")
    if not os.path.exists(windows_path):
        return {}

    # Read CSV
    all_rows = []
    with open(windows_path, 'r') as f:
        reader = csv.DictReader(f, delimiter=';')
        for row in reader:
            all_rows.append(row)

    # Static boundaries for FREE_FORM / STEADY_STATE
    static_bounds = {}
    if label_filter in ("FREE_FORM", "STEADY_STATE"):
        static_bounds = _get_static_boundaries(events_root, subj, subj_ecg)

    # Select rows by label filter
    if label_filter == "GESTURE":
        selected = [w for w in all_rows if w['label'] != 'REST']
    elif label_filter == "FREE_FORM":
        selected = []
        for w in all_rows:
            if w['label'] != 'REST':
                continue
            rep = int(w['repetition'])
            ws, we = int(w['start_sample']), int(w['end_sample'])
            if rep in static_bounds:
                ss, se = static_bounds[rep]
                overlap = max(0, min(we, se) - max(ws, ss))
                if overlap > (we - ws) * 0.5:
                    continue   # overlaps Static → skip
            selected.append(w)
    elif label_filter == "STEADY_STATE":
        selected = []
        for w in all_rows:
            if w['label'] != 'REST':
                continue
            rep = int(w['repetition'])
            ws, we = int(w['start_sample']), int(w['end_sample'])
            if rep not in static_bounds:
                continue
            ss, se = static_bounds[rep]
            overlap = max(0, min(we, se) - max(ws, ss))
            if overlap > (we - ws) * 0.5:
                selected.append(w)
    else:
        selected = all_rows

    # Build per-repetition data
    per_rep = {}
    for row in selected:
        if row.get('is_valid_window', '1') == '0':
            continue
        try:
            mean_hr = float(row['mean_hr_bpm'])
            sdnn    = float(row['sdnn_ms'])
            rmssd   = float(row['rmssd_ms'])
            pnn50   = float(row['pnn50_pct'])
        except (ValueError, KeyError):
            continue
        if mean_hr <= 0 or sdnn < 0:
            continue

        rep = int(row['repetition'])
        start_sample = int(row['start_sample'])
        end_sample   = int(row['end_sample'])

        if rep not in subj_ecg:
            continue
        ecg = subj_ecg[rep]
        if end_sample > len(ecg):
            continue
        window = ecg[start_sample:end_sample]
        if len(window) < window_size:
            continue

        ecg_2ch = np.stack([window, window], axis=0).astype(np.float32)
        per_rep.setdefault(rep, []).append((ecg_2ch, mean_hr, sdnn, rmssd, pnn50))

    return per_rep


# ============================================================================
# Data loading: wECG per-recording
# ============================================================================
def load_wecg_all(data_root="/root/autodl-tmp/wECG_dataset_npy",
                  window_size=5000):
    """Load all wECG data using the original **2-channel wrist ECG**, organised
    by dataset ID.

    Each window is stored as [2, window_size].
    HRV ground truth is computed from channel 0.

    Returns dict: {dataset_id: [(ecg_2ch, hr, sdnn, rmssd, pnn50), ...]}
    """
    import neurokit2 as nk

    if not os.path.exists(data_root):
        print(f"[WARN] wECG root not found: {data_root}")
        return {}

    files = sorted([f for f in os.listdir(data_root)
                    if f.endswith('.npy') and 'info' not in f])

    per_id = {}
    for fname in tqdm(files, desc="Loading wECG (2ch wrist)"):
        parts = fname.replace('.npy', '').split('_')
        try:
            dataset_id = parts[1]
        except Exception:
            continue

        try:
            raw = np.load(os.path.join(data_root, fname), allow_pickle=True)
            if raw.dtype.names is not None:
                item = raw[0, 0]
                if item['data_availability'] != 'yes':
                    continue
                if 'wECG' not in item.dtype.names:
                    continue
                data = np.asarray(item['wECG'], dtype=np.float32)  # [T, 2]
            else:
                data = np.asarray(raw, dtype=np.float32)

            # Ensure shape is [C, T] with C=2
            if data.ndim == 1:
                data = np.stack([data, data], axis=0)  # fallback
            elif data.ndim == 2:
                if data.shape[1] == 2 and data.shape[0] != 2:
                    data = data.T  # [T, 2] -> [2, T]
                elif data.shape[0] == 2:
                    pass  # already [2, T]
                else:
                    continue  # skip unexpected shapes

            C, L = data.shape
            if C != 2 or L < window_size:
                continue

            for start in range(0, L - window_size + 1, 500):
                window = data[:, start:start + window_size]  # [2, window_size]
                # Use channel 0 for HRV ground truth computation
                ecg = window[0]
                try:
                    ecg_clean = nk.ecg_clean(ecg, sampling_rate=500)
                    _, rpeaks = nk.ecg_peaks(ecg_clean, sampling_rate=500,
                                             method='neurokit')
                    r_idx = rpeaks['ECG_R_Peaks']
                    if len(r_idx) < 3:
                        continue
                    rr = np.diff(r_idx) / 500.0 * 1000
                    rr = rr[(rr > 300) & (rr < 2000)]
                    if len(rr) < 2:
                        continue
                    mean_hr = 60000.0 / np.mean(rr)
                    sdnn = np.std(rr)
                    rr_diff = np.diff(rr)
                    rmssd = (np.sqrt(np.mean(rr_diff ** 2))
                             if len(rr_diff) > 0 else 0)
                    pnn50 = (100.0 * np.sum(np.abs(rr_diff) > 50) / len(rr_diff)
                             if len(rr_diff) > 0 else 0)
                    if mean_hr <= 0 or sdnn < 0:
                        continue
                    per_id.setdefault(dataset_id, []).append(
                        (window.astype(np.float32), mean_hr, sdnn, rmssd, pnn50))
                except Exception:
                    continue
        except Exception:
            continue

    print(f"[wECG] {sum(len(v) for v in per_id.values())} windows "
          f"from {len(per_id)} recordings (2ch wrist)")
    return per_id


# ============================================================================
# Simple Dataset wrapper
# ============================================================================
class WindowDataset(Dataset):
    """Wraps a list of (ecg, hr, sdnn, rmssd, pnn50) tuples.

    ecg is always [2, L]: either the original 2ch wrist ECG (wECG) or
    1ch chest ECG duplicated to 2ch (Neurokit).
    """
    def __init__(self, windows_list, target_size=1024, stride=512):
        self.windows = windows_list
        self.target_size = target_size
        self.stride = stride
    
    def __len__(self):
        return len(self.windows)
    
    def __getitem__(self, i):
        ecg, hr, sdnn, rmssd, pnn50 = self.windows[i]
        C, L = ecg.shape
        
        sub_wins = []
        for s in range(0, L - self.target_size + 1, self.stride):
            sub_wins.append(ecg[:, s:s + self.target_size])
        if not sub_wins:
            sub_wins = [ecg[:, :self.target_size]]
        
        targets = torch.tensor([hr, sdnn, rmssd, pnn50], dtype=torch.float32)
        return torch.from_numpy(np.stack(sub_wins)).float(), targets


# ============================================================================
# Model
# ============================================================================
def build_model_from_checkpoint(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    config = ckpt.get('config', {})
    from model_overlap_no_peak import MAEConfig, ECGMAE
    cfg = MAEConfig(
        in_channels=2,
        patch_size=config.get('patch_size', 64),
        embed_dim=128, depth=6, num_heads=8, mlp_ratio=4.0,
        decoder_embed_dim=64, decoder_depth=4, decoder_num_heads=4,
        mask_ratio=config.get('mask_ratio', 0.55),
        max_patches=config.get('max_patches', 31),
        overlap_ratio=config.get('overlap_ratio', 0.5),
    )
    model = ECGMAE(cfg)
    model.load_state_dict(ckpt['model_state'])
    return model.to(device), cfg


class RegressionHead(nn.Module):
    """Deeper regression head with residual, dropout, and dual-pooling."""
    def __init__(self, embed_dim, n_outputs=4, dropout=0.1):
        super().__init__()
        # Conv block 1
        self.conv1 = nn.Conv1d(embed_dim, embed_dim, 3, padding=1)
        self.bn1   = nn.BatchNorm1d(embed_dim)
        # Conv block 2
        self.conv2 = nn.Conv1d(embed_dim, embed_dim, 3, padding=1)
        self.bn2   = nn.BatchNorm1d(embed_dim)
        # MLP head  (avg + max pooling → 2 * embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim * 2, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, n_outputs),
        )
    
    def forward(self, x):
        # x: [B, T, D] → [B, D, T]
        h = x.transpose(1, 2)
        h = F.gelu(self.bn1(self.conv1(h))) + h          # residual
        h = F.gelu(self.bn2(self.conv2(h))) + h          # residual
        avg_p = h.mean(dim=2)                             # [B, D]
        max_p = h.max(dim=2)[0]                           # [B, D]
        return self.mlp(torch.cat([avg_p, max_p], dim=1)) # [B, n_outputs]


class HRHRVRegressor(nn.Module):
    """Regressor with partial encoder fine-tuning.

    freeze_mode:
      'full'    – freeze entire encoder (fast, baseline)
      'partial' – freeze all except last `n_unfreeze` encoder blocks + norm
      'none'    – unfreeze everything (slow, may overfit)

    n_encoder_blocks: if set, only use the first *n* encoder blocks during
                      forward (the remaining blocks are ignored).
    """
    def __init__(self, encoder, embed_dim, freeze_mode='partial',
                 n_unfreeze=2, dropout=0.1, n_encoder_blocks=None):
        super().__init__()
        self.encoder = encoder
        self.embed_dim = embed_dim
        self.n_encoder_blocks = n_encoder_blocks

        # ---------- freeze policy ----------
        if freeze_mode == 'full':
            for p in self.encoder.parameters():
                p.requires_grad = False
        elif freeze_mode == 'partial':
            for p in self.encoder.parameters():
                p.requires_grad = False
            # Unfreeze last n encoder blocks + norm
            n_blocks = len(self.encoder.encoder_blocks)
            for blk in self.encoder.encoder_blocks[n_blocks - n_unfreeze:]:
                for p in blk.parameters():
                    p.requires_grad = True
            for p in self.encoder.encoder_norm.parameters():
                p.requires_grad = True
        # 'none' → all params trainable

        self.reg_head = RegressionHead(embed_dim, n_outputs=4, dropout=dropout)
    
    def forward(self, x):
        B, N, C, L = x.shape
        x_flat = x.view(B * N, C, L)
        enc = self.encoder.encode(x_flat, n_blocks=self.n_encoder_blocks)  # [B*N, T, D]
        enc = enc.view(B, N, enc.shape[1], enc.shape[2])
        enc = enc.mean(dim=1)                                  # [B, T, D]
        return self.reg_head(enc)


# ============================================================================
# Per-subject training & prediction
# ============================================================================
def train_and_predict(encoder, cfg, train_windows, test_windows,
                      device, epochs=30, batch_size=32, seed=42,
                      n_encoder_blocks=None):
    """Train regression head on train_windows.

    When n_encoder_blocks is None (default):
        partial fine-tuning – freeze first 4, fine-tune last 2 blocks + norm.
    When n_encoder_blocks is set (e.g. 4):
        only use the first *n* frozen blocks; no encoder fine-tuning.

    Uses:
      • Huber (SmoothL1) loss
      • Per-subject target z-score normalisation
      • Cosine-annealing LR schedule
      • Gradient clipping

    Returns (preds, trues) in ORIGINAL scale, shape (N, 4).
    """
    set_seed(seed)

    # ---- collect raw targets to compute z-score stats ----
    train_targets_np = np.array([w[1:] for w in train_windows])  # (N, 4)
    t_mean = train_targets_np.mean(axis=0)  # (4,)
    t_std  = train_targets_np.std(axis=0) + 1e-8

    train_ds = WindowDataset(train_windows)
    test_ds  = WindowDataset(test_windows)

    g = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=0, generator=g)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False,
                              num_workers=0)

    t_mean_t = torch.tensor(t_mean, dtype=torch.float32, device=device)
    t_std_t  = torch.tensor(t_std,  dtype=torch.float32, device=device)

    if n_encoder_blocks is not None:
        # Only first n blocks, fully frozen encoder
        freeze_mode = 'full'
        n_unfreeze = 0
    else:
        freeze_mode = 'partial'
        n_unfreeze = 2

    regressor = HRHRVRegressor(
        encoder, cfg.embed_dim,
        freeze_mode=freeze_mode, n_unfreeze=n_unfreeze, dropout=0.1,
        n_encoder_blocks=n_encoder_blocks,
    ).to(device)

    # Separate LR: encoder fine-tune layers get smaller LR
    encoder_params, head_params = [], []
    for name, p in regressor.named_parameters():
        if not p.requires_grad:
            continue
        if 'encoder' in name:
            encoder_params.append(p)
        else:
            head_params.append(p)

    optimizer = torch.optim.AdamW([
        {'params': encoder_params, 'lr': 1e-4, 'weight_decay': 1e-4},
        {'params': head_params,    'lr': 5e-4, 'weight_decay': 1e-4},
    ])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=1e-6)

    best_loss = float('inf')
    best_state = None

    for epoch in range(1, epochs + 1):
        regressor.train()
        total_loss = 0
        total = 0
        for x, targets in train_loader:
            x, targets = x.to(device), targets.to(device)
            targets_norm = (targets - t_mean_t) / t_std_t

            optimizer.zero_grad()
            preds_norm = regressor(x)
            loss = F.smooth_l1_loss(preds_norm, targets_norm)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(regressor.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item() * x.size(0)
            total += x.size(0)

        scheduler.step()
        avg_loss = total_loss / total
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_state = {k: v.cpu().clone()
                          for k, v in regressor.state_dict().items()}

    # ---- Restore best & predict (denormalise) ----
    regressor.load_state_dict(best_state)
    regressor.eval()
    all_preds, all_targets = [], []
    with torch.no_grad():
        for x, targets in test_loader:
            x = x.to(device)
            preds_norm = regressor(x)
            # denormalise
            preds = preds_norm.cpu() * t_std_t.cpu() + t_mean_t.cpu()
            all_preds.append(preds.numpy())
            all_targets.append(targets.numpy())
    
    return np.concatenate(all_preds), np.concatenate(all_targets)


# ============================================================================
# Accuracy computation (pooled across subjects)
# ============================================================================
def compute_accuracy(all_preds, all_trues, tolerance=0.05):
    """Compute HR L1 metrics + HRV range-based accuracy on pooled data."""
    hr_pred, sdnn_pred, rmssd_pred, pnn50_pred = all_preds.T
    hr_true, sdnn_true, rmssd_true, pnn50_true = all_trues.T
    
    # HR: pure regression L1 metrics
    hr_l1 = np.abs(hr_pred - hr_true)
    hr_l1_eq0  = np.mean(hr_pred.astype(int) == hr_true.astype(int))  # int(pred)==int(true)
    hr_l1_lt1  = np.mean(hr_l1 < 1.0)                      # L1 < 1 bpm
    hr_l1_lt2  = np.mean(hr_l1 < 2.0)                      # L1 < 2 bpm
    hr_l1_lt5  = np.mean(hr_l1 < 5.0)                      # L1 < 5 bpm
    hr_l1_lt10p = np.mean(hr_l1 < 0.10 * np.abs(hr_true))  # L1 < 10% × ref

    # HRV: range-based
    def _range_acc(true_vals, pred_vals, tol):
        gt_range = np.max(true_vals) - np.min(true_vals)
        threshold = gt_range * tol
        correct = np.abs(pred_vals - true_vals) <= threshold
        return np.mean(correct), gt_range, threshold, correct

    sdnn_acc,  sdnn_range,  sdnn_thr,  sdnn_ok  = _range_acc(sdnn_true,  sdnn_pred,  tolerance)
    rmssd_acc, rmssd_range, rmssd_thr, rmssd_ok = _range_acc(rmssd_true, rmssd_pred, tolerance)
    pnn50_acc, pnn50_range, pnn50_thr, pnn50_ok = _range_acc(pnn50_true, pnn50_pred, tolerance)

    hrv_combined = np.mean(sdnn_ok & rmssd_ok & pnn50_ok)

    return {
        'hr_mae': float(np.mean(hr_l1)),
        'hr_std': float(np.std(hr_l1)),
        'hr_l1_eq0': hr_l1_eq0,
        'hr_l1_lt1': hr_l1_lt1,
        'hr_l1_lt2': hr_l1_lt2,
        'hr_l1_lt5': hr_l1_lt5,
        'hr_l1_lt10p': hr_l1_lt10p,
        'sdnn_acc': sdnn_acc, 'sdnn_range': sdnn_range, 'sdnn_thr': sdnn_thr,
        'rmssd_acc': rmssd_acc, 'rmssd_range': rmssd_range, 'rmssd_thr': rmssd_thr,
        'pnn50_acc': pnn50_acc, 'pnn50_range': pnn50_range, 'pnn50_thr': pnn50_thr,
        'hrv_combined_acc': hrv_combined,
        'sdnn_mae': mean_absolute_error(sdnn_true, sdnn_pred),
        'rmssd_mae': mean_absolute_error(rmssd_true, rmssd_pred),
        'pnn50_mae': mean_absolute_error(pnn50_true, pnn50_pred),
        'n_samples': len(hr_true),
    }
    

# ============================================================================
# Main
# ============================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device",       type=str, default="cuda")
    parser.add_argument("--epochs",       type=int, default=30)
    parser.add_argument("--batch_size",   type=int, default=32)
    parser.add_argument("--ablation_dir", type=str, default="ablation_results")
    parser.add_argument("--only", type=str, default=None,
                        help="Comma-separated config names to run (e.g. pm1_po0,pm1_po1)")
    parser.add_argument("--n_encoder_blocks", type=int, default=None,
                        help="Only use first N encoder blocks (frozen). None=all 6 with partial fine-tune.")
    args = parser.parse_args()
    
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Device: {device}")
    
    all_configs = [
        ("pm0_po0", "{0,0}", f"{args.ablation_dir}/pm0_po0_best.pt"),
        ("pm0_po1", "{0,1}", f"{args.ablation_dir}/pm0_po1_best.pt"),
        ("pm1_po0", "{1,0}", f"{args.ablation_dir}/pm1_po0_best.pt"),
        ("pm1_po1", "{1,1}", f"{args.ablation_dir}/pm1_po1_best.pt"),
    ]
    # Filter configs if --only flag provided
    if hasattr(args, 'only') and args.only:
        only_set = set(args.only.split(','))
        configs = [c for c in all_configs if c[0] in only_set]
    else:
        configs = all_configs
    
    # Load MAE pretrain losses
    loss_data = {}
    summary_path = os.path.join(args.ablation_dir, "ablation_summary.csv")
    if os.path.exists(summary_path):
        with open(summary_path, 'r') as f:
            reader = csv.DictReader(f, delimiter=';')
            for row in reader:
                loss_data[row['config']] = float(row['best_test_loss'])
    
    print("\n" + "=" * 70)
    print("WITHIN-SUBJECT HRV REGRESSION")
    print("=" * 70)
    print(f"Subjects: {ALL_SUBJECTS}")
    print(f"Excluded: {sorted(EXCLUDED_SUBJECTS)}")
    print(f"Neurokit split: rep 1-7 train, rep 8-10 test (per subject)")
    print(f"wECG split: temporal 70/30 (per recording)")
    print("=" * 70)
    
    # ------------------------------------------------------------------
    # 1. Pre-load ALL Neurokit data (once, reused across configs)
    # ------------------------------------------------------------------
    tasks = ["FREE_FORM", "STEADY_STATE", "GESTURE"]
    task_labels = {
        "FREE_FORM": "Free Form",
        "STEADY_STATE": "Steady State",
        "GESTURE": "Gesture",
    }

    # {task: {subj: {rep: [(ecg, hr, sdnn, rmssd, pnn50), ...]}}}
    all_neurokit = {}
    for task in tasks:
        print(f"\n[INFO] Loading {task_labels[task]} data ...")
        task_data = {}
        total_train = total_test = 0
        for subj in ALL_SUBJECTS:
            per_rep = load_neurokit_per_subject(subj, label_filter=task)
            if per_rep:
                task_data[subj] = per_rep
                n_tr = sum(len(v) for r, v in per_rep.items() if r in TRAIN_REPS)
                n_te = sum(len(v) for r, v in per_rep.items() if r in TEST_REPS)
                total_train += n_tr
                total_test  += n_te
        all_neurokit[task] = task_data
        print(f"       → {len(task_data)} subjects, "
              f"{total_train} train windows, {total_test} test windows")

    # ------------------------------------------------------------------
    # 2. Pre-load ALL wECG data (once)
    # ------------------------------------------------------------------
    print("\n[INFO] Loading wECG data ...")
    wecg_per_id = load_wecg_all()

    # ------------------------------------------------------------------
    # 3. Run ablation configs
    # ------------------------------------------------------------------
    all_results = {}
    
    for config_name, config_label, ckpt_path in configs:
        print(f"\n{'=' * 70}")
        print(f"Config: {config_label} ({config_name})")
        print(f"{'=' * 70}")
        
        if not os.path.exists(ckpt_path):
            print(f"[SKIP] {ckpt_path} not found")
            continue
        
        encoder, cfg = build_model_from_checkpoint(ckpt_path, device)
        encoder.eval()
        
        config_results = {
            'test_loss': loss_data.get(config_name, 0),
        }

        # ---- Task 1: wECG (within-recording, temporal split) ----
        print("  [wECG] Training per-recording ...")
        wecg_preds_all, wecg_trues_all = [], []
        wecg_per_rec_hr_mae = []          # ← per-recording HR MAE
        n_recs = 0
        for rec_id in sorted(wecg_per_id.keys()):
            windows = wecg_per_id[rec_id]
            if len(windows) < 5:
                continue  # too few windows
            n_train = max(1, int(len(windows) * WECG_TRAIN_RATIO))
            train_w = windows[:n_train]
            test_w  = windows[n_train:]
            if len(test_w) == 0:
                continue
            preds, trues = train_and_predict(
                encoder, cfg, train_w, test_w, device,
                epochs=args.epochs, batch_size=args.batch_size,
                n_encoder_blocks=args.n_encoder_blocks)
            wecg_preds_all.append(preds)
            wecg_trues_all.append(trues)
            wecg_per_rec_hr_mae.append(np.mean(np.abs(preds[:, 0] - trues[:, 0])))
            n_recs += 1

        if wecg_preds_all:
            wecg_preds = np.concatenate(wecg_preds_all)
            wecg_trues = np.concatenate(wecg_trues_all)
            res = compute_accuracy(wecg_preds, wecg_trues)
            # e_i = HR_hat_i - HR_i
            signed_err = wecg_preds[:, 0] - wecg_trues[:, 0]
            abs_err    = np.abs(signed_err)
            res['hr_mae_subj_mean'] = float(np.mean(abs_err))                   # MAE = mean(|e_i|)
            res['hr_mae_subj_std']  = float(np.std(abs_err))                    # STD of |e_i| (ddof=0)
            res['hr_rmse']          = float(np.sqrt(np.mean(signed_err ** 2)))   # RMSE
            config_results['wECG'] = res
            print(f"  [wECG] {n_recs} recordings, {res['n_samples']} test windows")
            _print_task("wECG", res)

        # ---- Tasks 2-4: Neurokit (within-subject, rep split) ----
        for task in tasks:
            task_name = task_labels[task]
            print(f"  [{task_name}] Training per-subject ...")
            task_preds_all, task_trues_all = [], []
            per_subj_hr_mae = []              # ← per-subject HR MAE
            n_subj = 0

            for subj in ALL_SUBJECTS:
                if subj not in all_neurokit[task]:
                    continue
                per_rep = all_neurokit[task][subj]

                train_w = [w for r, ws in per_rep.items()
                           if r in TRAIN_REPS for w in ws]
                test_w  = [w for r, ws in per_rep.items()
                           if r in TEST_REPS for w in ws]
                if len(train_w) < 2 or len(test_w) < 1:
                    continue

                preds, trues = train_and_predict(
                    encoder, cfg, train_w, test_w, device,
                    epochs=args.epochs, batch_size=args.batch_size,
                    n_encoder_blocks=args.n_encoder_blocks)
                task_preds_all.append(preds)
                task_trues_all.append(trues)
                per_subj_hr_mae.append(np.mean(np.abs(preds[:, 0] - trues[:, 0])))
                n_subj += 1

            if task_preds_all:
                pooled_preds = np.concatenate(task_preds_all)
                pooled_trues = np.concatenate(task_trues_all)
                res = compute_accuracy(pooled_preds, pooled_trues)
                # e_i = HR_hat_i - HR_i
                signed_err = pooled_preds[:, 0] - pooled_trues[:, 0]
                abs_err    = np.abs(signed_err)
                res['hr_mae_subj_mean'] = float(np.mean(abs_err))                   # MAE = mean(|e_i|)
                res['hr_mae_subj_std']  = float(np.std(abs_err))                    # STD of |e_i| (ddof=0)
                res['hr_rmse']          = float(np.sqrt(np.mean(signed_err ** 2)))   # RMSE
                config_results[task] = res
                print(f"  [{task_name}] {n_subj} subjects, "
                      f"{res['n_samples']} test windows")
                _print_task(task_name, res)

        all_results[config_name] = config_results

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    ds_keys   = ['wECG', 'FREE_FORM', 'STEADY_STATE', 'GESTURE']
    ds_labels = {
        'wECG': 'wECG',
        'FREE_FORM': 'Free Form',
        'STEADY_STATE': 'Steady State',
        'GESTURE': 'Gesture',
    }

    print("\n" + "=" * 140)
    print("FINAL RESULTS – WITHIN-SUBJECT HRV REGRESSION")
    print("MAE = mean(|e_i|),  STD = std(|e_i|, ddof=0),  RMSE = sqrt(mean(e_i^2))")
    print("HR: L1 regression metrics  |  HRV: 5% of pooled GT range")
    print("Neurokit: rep 1-7 train, rep 8-10 test  |  wECG: temporal 70/30")
    print("=" * 140)
    
    # Use all_configs for summary (prints all available)
    for config_name, config_label, _ in all_configs:
        if config_name not in all_results:
            continue
        res = all_results[config_name]
        test_loss = res.get('test_loss', 0)
        
        print(f"\n### {config_label}, MAE Pre-train Loss = {test_loss:.4f}")
        print(f"{'Dataset':<14} {'HR MAE±STD':>16} {'RMSE':>8} {'L1<2':>8} {'L1<5':>8} {'L1<10%r':>8} "
              f"{'RMSSD':>8} {'SDNN':>8} {'N':>6}")
        print("-" * 100)
        for dk in ds_keys:
            if dk in res:
                r = res[dk]
                hr_mae  = r.get('hr_mae_subj_mean', r['hr_mae'])
                hr_std  = r.get('hr_mae_subj_std', r['hr_std'])
                hr_rmse = r.get('hr_rmse', 0)
                mae_std_str = f"{hr_mae:.2f}±{hr_std:.2f}"
                print(f"{ds_labels[dk]:<14} "
                      f"{mae_std_str:>16}  "
                      f"{hr_rmse:>7.2f} "
                      f"{r['hr_l1_lt2']*100:>7.2f}% "
                      f"{r['hr_l1_lt5']*100:>7.2f}% "
                      f"{r['hr_l1_lt10p']*100:>7.2f}% "
                      f"{r['rmssd_acc']*100:>7.2f}% "
                      f"{r['sdnn_acc']*100:>7.2f}% "
                      f"{r['n_samples']:>6}")
        print("-" * 100)
    
    # ------------------------------------------------------------------
    # Save CSV
    # ------------------------------------------------------------------
    output_path = os.path.join(args.ablation_dir,
                               "ablation_regression_results.csv")
    with open(output_path, 'w') as f:
        writer = csv.writer(f, delimiter=';')
        header = ['config', 'mae_loss']
        for ds in ds_keys:
            header.extend([f'{ds}_hr_mae', f'{ds}_hr_std',
                           f'{ds}_hr_mae_subj_mean', f'{ds}_hr_mae_subj_std',
                           f'{ds}_hr_rmse',
                           f'{ds}_hr_l1_eq0', f'{ds}_hr_l1_lt1',
                           f'{ds}_hr_l1_lt2', f'{ds}_hr_l1_lt5',
                           f'{ds}_hr_l1_lt10p'])
            for m in ['sdnn', 'rmssd', 'pnn50']:
                header.extend([f'{ds}_{m}_mae', f'{ds}_{m}_acc'])
            header.append(f'{ds}_hrv_combined_acc')
        writer.writerow(header)
        
        for config_name, config_label, _ in all_configs:
            if config_name not in all_results:
                continue
            res = all_results[config_name]
            row = [config_label, res.get('test_loss', 0)]
            for ds in ds_keys:
                if ds in res:
                    r = res[ds]
                    row.extend([r.get('hr_mae', 0), r.get('hr_std', 0),
                                r.get('hr_mae_subj_mean', r.get('hr_mae', 0)),
                                r.get('hr_mae_subj_std', r.get('hr_std', 0)),
                                r.get('hr_rmse', 0),
                                r.get('hr_l1_eq0', 0), r.get('hr_l1_lt1', 0),
                                r.get('hr_l1_lt2', 0), r.get('hr_l1_lt5', 0),
                                r.get('hr_l1_lt10p', 0)])
                    for m in ['sdnn', 'rmssd', 'pnn50']:
                        row.extend([r.get(f'{m}_mae', 0), r.get(f'{m}_acc', 0)])
                    row.append(r.get('hrv_combined_acc', 0))
                else:
                    row.extend([0] * 17)
            writer.writerow(row)
    
    print(f"\nResults saved to: {output_path}")


def _print_task(name, r):
    print(f"    HR: MAE={r['hr_mae']:.2f}±{r['hr_std']:.2f}bpm  L1==0={r['hr_l1_eq0']*100:.2f}%  "
          f"L1<1={r['hr_l1_lt1']*100:.2f}%  "
          f"L1<2={r['hr_l1_lt2']*100:.2f}%  "
          f"L1<5={r['hr_l1_lt5']*100:.2f}%  "
          f"L1<10%ref={r['hr_l1_lt10p']*100:.2f}%")
    print(f"    SDNN={r['sdnn_acc']*100:.2f}%  "
          f"RMSSD={r['rmssd_acc']*100:.2f}%  "
          f"pNN50={r['pnn50_acc']*100:.2f}%  "
          f"HRV(all)={r['hrv_combined_acc']*100:.2f}%")
    print(f"    MAE: SDNN={r['sdnn_mae']:.2f} "
          f"RMSSD={r['rmssd_mae']:.2f} pNN50={r['pnn50_mae']:.2f}")
    print(f"    5% thr: SDNN={r['sdnn_thr']:.2f}ms "
          f"RMSSD={r['rmssd_thr']:.2f}ms pNN50={r['pnn50_thr']:.2f}%")


if __name__ == "__main__":
    main()
