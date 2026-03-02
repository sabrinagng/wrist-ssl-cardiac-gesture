"""
Intra-Subject 2D CNN Downstream with Pretrained Encoder
========================================================
Train a separate 2D CNN model per subject using reps 1-8 as train, 9-10 as test.
Uses pretrained 2D CNN encoder from MAE (STFT) + simple 2D CNN classifier head.
Reports mean ± std accuracy across all subjects.
"""

import gc
import os
import sys
import json
import math
import argparse
import numpy as np
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

sys.path.insert(0, '/root/autodl-fs/emg_pretrain')
from resnet_emg import GESTURE_NAMES, EXCLUDED_GESTURES
from downstream_classifier_2d_cnn import GestureClassifier2DCNN

PRETRAINED = ('/root/autodl-fs/emg_pretrain/pretrain/checkpoints/'
              'cnn_mae_2d_stft/2026-02-13-14-59-13/ckpt_cnn_mae_2d_epoch_50.pth')


# ──────────────────── Dataset ────────────────────

class EMGDataset(Dataset):
    def __init__(self, emg_list, labels):
        if len(emg_list) == 0:
            self.emg = torch.zeros(0, 2, 1)
            self.labels = torch.zeros(0, dtype=torch.long)
        else:
            self.emg = torch.from_numpy(
                np.stack([e.T for e in emg_list], axis=0).astype(np.float32))
            self.labels = torch.from_numpy(labels.astype(np.int64))

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.emg[idx], self.labels[idx]


# ──────────────────── Data Loading ────────────────────

def load_subject(data_dir, label_dir, subj):
    """Load data for a single subject."""
    sd = os.path.join(data_dir, subj)
    sl = os.path.join(label_dir, subj)
    emg_list, labels, reps = [], [], []

    if not os.path.isdir(sd):
        return emg_list, np.array(labels), np.array(reps)

    for f in sorted(os.listdir(sd)):
        if not f.endswith('.npz'):
            continue
        dp = os.path.join(sd, f)
        lp = os.path.join(sl, f)
        if not os.path.exists(lp):
            continue
        d = np.load(dp, allow_pickle=True)
        gid = int(d['gesture_id'])
        rep = int(d['repetition'])
        if gid in EXCLUDED_GESTURES:
            continue
        emg_list.append(d['emg'].astype(np.float32))
        labels.append(gid)
        reps.append(rep)

    return emg_list, np.array(labels), np.array(reps)


# ──────────────────── Per-Subject Training ────────────────────

def run_subject(subj, data_dir, label_dir, args, device):
    """
    Train and evaluate one subject.
    Returns (accuracy, preds_original_labels, true_original_labels) or None.
    """
    emg_list, labels, reps = load_subject(data_dir, label_dir, subj)
    if len(emg_list) == 0:
        print(f"  {subj}: no data")
        return None

    # Split by repetition: reps 1-8 train, reps 9-10 test
    tri = [i for i, r in enumerate(reps) if r in range(1, 9)]
    tei = [i for i, r in enumerate(reps) if r in (9, 10)]

    tr_emg = [emg_list[i] for i in tri]
    tr_lab = labels[tri]
    te_emg = [emg_list[i] for i in tei]
    te_lab = labels[tei]

    if len(tr_emg) == 0 or len(te_emg) == 0:
        print(f"  {subj}: insufficient data")
        return None

    # Keep original labels for confusion matrix
    te_lab_orig = te_lab.copy()

    # Map labels to contiguous 0..N-1
    ul = sorted(set(tr_lab) | set(te_lab))
    lm = {l: i for i, l in enumerate(ul)}
    idx_to_label = {i: l for l, i in lm.items()}
    nc = len(ul)
    tr_lab = np.array([lm[l] for l in tr_lab])
    te_lab = np.array([lm[l] for l in te_lab])

    # Validation split from training data
    if len(tr_emg) > 10:
        tridx, vidx = train_test_split(
            np.arange(len(tr_emg)), test_size=0.15,
            stratify=tr_lab, random_state=42)
        v_emg = [tr_emg[i] for i in vidx]
        v_lab = tr_lab[vidx]
        tr_emg = [tr_emg[i] for i in tridx]
        tr_lab = tr_lab[tridx]
    else:
        v_emg, v_lab = te_emg, te_lab

    # Dataloaders
    trl = DataLoader(
        EMGDataset(tr_emg, tr_lab),
        batch_size=min(args.batch_size, len(tr_emg)),
        shuffle=True, num_workers=0, pin_memory=True,
        drop_last=len(tr_emg) > args.batch_size)
    vl = DataLoader(
        EMGDataset(v_emg, v_lab),
        batch_size=args.batch_size,
        shuffle=False, num_workers=0, pin_memory=True)
    tel = DataLoader(
        EMGDataset(te_emg, te_lab),
        batch_size=args.batch_size,
        shuffle=False, num_workers=0, pin_memory=True)

    # Create model
    emg_channels = tr_emg[0].shape[1]
    model = GestureClassifier2DCNN(
        emg_channels=emg_channels,
        num_classes=nc,
        dropout=args.dropout,
        pretrained_path=PRETRAINED if os.path.exists(PRETRAINED) else None,
        freeze_encoder=args.freeze_encoder,
        transform_type=args.transform,
        n_fft=args.n_fft,
        hop_length=args.hop_length,
    ).to(device)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # Optimizer: lower lr for encoder, higher for head
    enc_p = [p for n, p in model.named_parameters()
             if p.requires_grad and 'encoder' in n]
    head_p = [p for n, p in model.named_parameters()
              if p.requires_grad and 'encoder' not in n]
    pg = [{'params': head_p, 'lr': args.lr}]
    if enc_p:
        pg.append({'params': enc_p, 'lr': args.lr * 0.1})

    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    optimizer = optim.AdamW(pg, weight_decay=args.weight_decay)

    # Cosine schedule with warmup
    total_steps = args.epochs * len(trl)
    warmup = int(0.05 * total_steps)

    def lr_fn(step):
        if step < warmup:
            return step / max(warmup, 1)
        prog = (step - warmup) / max(total_steps - warmup, 1)
        return 0.5 * (1 + math.cos(math.pi * prog))

    sched = optim.lr_scheduler.LambdaLR(optimizer, lr_fn)
    scaler = GradScaler('cuda') if device.type == 'cuda' else None

    best_va, best_ep, pat = 0.0, 0, 0
    best_state = None

    for ep in range(1, args.epochs + 1):
        # ---- Train ----
        model.train()
        for x, y in trl:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            if scaler:
                with autocast('cuda'):
                    logits = model(x)
                    loss = criterion(logits, y)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                logits = model(x)
                loss = criterion(logits, y)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            sched.step()

        # ---- Validate ----
        model.eval()
        ap, ay = [], []
        with torch.no_grad():
            for x, y in vl:
                x = x.to(device)
                if scaler:
                    with autocast('cuda'):
                        lo = model(x)
                else:
                    lo = model(x)
                ap.extend(lo.argmax(1).cpu().numpy())
                ay.extend(y.numpy())
        va = accuracy_score(ay, ap)

        if va > best_va:
            best_va = va
            best_ep = ep
            pat = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            pat += 1
        if pat >= args.patience:
            break

    # ---- Test ----
    if best_state:
        model.load_state_dict(best_state)
    model.eval()
    ap, ay = [], []
    with torch.no_grad():
        for x, y in tel:
            x = x.to(device)
            if scaler:
                with autocast('cuda'):
                    lo = model(x)
            else:
                lo = model(x)
            ap.extend(lo.argmax(1).cpu().numpy())
            ay.extend(y.numpy())
    te_acc = accuracy_score(ay, ap)

    # Map predicted indices back to original gesture IDs
    preds_orig = np.array([idx_to_label[p] for p in ap])
    true_orig = te_lab_orig

    print(f"  {subj}: Test {te_acc:.2%}  Val {best_va:.2%}  ep{best_ep}  "
          f"train={len(tr_emg)} test={len(te_emg)} params={trainable:,}")

    del model, optimizer, best_state
    gc.collect()
    torch.cuda.empty_cache()
    return te_acc, preds_orig, true_orig


# ──────────────────── Plotting ────────────────────

def plot_confusion_matrix(cm, gesture_names, save_path, title_extra=''):
    """Plot dual confusion matrix (counts + normalized)."""
    cm_norm = cm.astype('float') / np.maximum(cm.sum(axis=1, keepdims=True), 1) * 100

    fig, axes = plt.subplots(1, 2, figsize=(18, 7))

    # Left: counts + percentage
    im0 = axes[0].imshow(cm, interpolation='nearest', cmap='Blues', aspect='auto')
    thresh0 = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            color = 'white' if cm[i, j] > thresh0 else 'black'
            axes[0].text(j, i, f'{cm[i, j]}\n({cm_norm[i, j]:.1f}%)',
                         ha='center', va='center', fontsize=8,
                         color=color, fontweight='bold')
    axes[0].set_xticks(range(len(gesture_names)))
    axes[0].set_yticks(range(len(gesture_names)))
    axes[0].set_xticklabels(gesture_names, rotation=45, ha='right', fontsize=9)
    axes[0].set_yticklabels(gesture_names, fontsize=9)
    axes[0].set_xlabel('Predicted', fontsize=11, fontweight='bold')
    axes[0].set_ylabel('True', fontsize=11, fontweight='bold')
    axes[0].set_title('Confusion Matrix — Counts', fontsize=12, fontweight='bold')
    plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    # Right: normalized
    im1 = axes[1].imshow(cm_norm, interpolation='nearest', cmap='Blues',
                         aspect='auto', vmin=0, vmax=100)
    thresh1 = 50.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            color = 'white' if cm_norm[i, j] > thresh1 else 'black'
            axes[1].text(j, i, f'{cm_norm[i, j]:.1f}%\n({cm[i, j]})',
                         ha='center', va='center', fontsize=8,
                         color=color, fontweight='bold')
    axes[1].set_xticks(range(len(gesture_names)))
    axes[1].set_yticks(range(len(gesture_names)))
    axes[1].set_xticklabels(gesture_names, rotation=45, ha='right', fontsize=9)
    axes[1].set_yticklabels(gesture_names, fontsize=9)
    axes[1].set_xlabel('Predicted', fontsize=11, fontweight='bold')
    axes[1].set_ylabel('True', fontsize=11, fontweight='bold')
    axes[1].set_title('Confusion Matrix — Normalized (%)', fontsize=12, fontweight='bold')
    plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04, label='Recall (%)')

    if title_extra:
        fig.suptitle(title_extra, fontsize=13, fontweight='bold', y=1.02)

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Confusion matrix saved: {save_path}")


def plot_per_subject_accuracy(accs, subj_names, save_path, title_extra=''):
    """Bar chart of per-subject accuracy."""
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = ['#e74c3c' if a < 60 else '#f39c12' if a < 75 else '#2ecc71' for a in accs]
    bars = ax.bar(range(len(accs)), accs, color=colors, edgecolor='black', alpha=0.85)
    for bar, a in zip(bars, accs):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                f'{a:.1f}%', ha='center', fontsize=9, fontweight='bold')
    ax.set_xticks(range(len(accs)))
    ax.set_xticklabels(subj_names, rotation=45, ha='right')
    ax.set_ylabel('Test Accuracy (%)')
    if title_extra:
        ax.set_title(title_extra)
    ax.axhline(y=np.mean(accs), color='red', linestyle='--', linewidth=2,
               label=f'Mean: {np.mean(accs):.1f}%')
    ax.set_ylim(0, 105)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Per-subject chart saved: {save_path}")


# ──────────────────── Main ────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Intra-Subject 2D CNN Downstream with Pretrained Encoder')
    parser.add_argument('--window', type=str, default='10s',
                        help='Window size (e.g., 10s, 12s, 14s, 16s)')
    parser.add_argument('--data-dir', type=str,
                        default='/root/autodl-fs/windowed_data',
                        help='Base directory for windowed data')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--patience', type=int, default=20)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight-decay', type=float, default=1e-4)
    parser.add_argument('--dropout', type=float, default=0.3)
    parser.add_argument('--freeze-encoder', action='store_true', default=True,
                        help='Freeze pretrained encoder')
    parser.add_argument('--no-freeze-encoder', dest='freeze_encoder',
                        action='store_false',
                        help='Fine-tune pretrained encoder')
    parser.add_argument('--transform', type=str, default='stft',
                        choices=['stft', 'cwt', 'logmel'],
                        help='Time-frequency transform type')
    parser.add_argument('--n-fft', type=int, default=256)
    parser.add_argument('--hop-length', type=int, default=64)
    parser.add_argument('--n-subjects', type=int, default=20)
    parser.add_argument('--out-dir', type=str,
                        default='/root/autodl-fs/emg_pretrain/results_2d_cnn_intra')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    ws = int(args.window.replace('s', ''))
    data_dir = os.path.join(args.data_dir, f"{ws}s", "data")
    label_dir = os.path.join(args.data_dir, f"{ws}s", "label")
    subjects = [f"S{i:02d}" for i in range(1, args.n_subjects + 1)]

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(args.out_dir, f"{args.window}_2DCNN_intra_{timestamp}")
    os.makedirs(out_dir, exist_ok=True)

    print(f"\n{'='*70}")
    print(f"Intra-Subject 2D CNN Downstream with Pretrained Encoder")
    print(f"{'='*70}")
    print(f"  Window: {args.window}  Subjects: {len(subjects)}")
    print(f"  Transform: {args.transform}  Encoder frozen: {args.freeze_encoder}")
    print(f"  Pretrained: {PRETRAINED}")
    print(f"  lr={args.lr}  dropout={args.dropout}  wd={args.weight_decay}")
    print(f"  epochs={args.epochs}  patience={args.patience}")
    print(f"  Output: {out_dir}")
    print(f"{'='*70}\n")

    accs = []
    all_preds = []
    all_true = []
    subj_names = []
    per_subject_results = []

    t0 = perf_counter()

    for subj in subjects:
        result = run_subject(subj, data_dir, label_dir, args, device)
        if result is not None:
            acc, preds, true = result
            accs.append(acc)
            all_preds.extend(preds)
            all_true.extend(true)
            subj_names.append(subj)
            per_subject_results.append({
                'subject': subj,
                'test_accuracy': round(acc * 100, 2),
            })

    elapsed = perf_counter() - t0
    accs_pct = np.array(accs) * 100
    all_preds = np.array(all_preds)
    all_true = np.array(all_true)

    overall_acc = accuracy_score(all_true, all_preds)

    # ---- Print summary ----
    print(f"\n{'='*70}")
    print(f"RESULTS — {args.window} Intra-Subject 2D CNN")
    print(f"{'='*70}")
    print(f"  Mean ± Std: {accs_pct.mean():.2f} ± {accs_pct.std():.2f}%")
    print(f"  Overall (pooled preds): {overall_acc * 100:.2f}%")
    print(f"  Min: {accs_pct.min():.2f}%  Max: {accs_pct.max():.2f}%")
    print(f"  All: {', '.join(f'{a:.1f}' for a in accs_pct)}")
    print(f"  Time: {elapsed:.1f}s")
    print(f"{'='*70}")

    # Classification report
    gesture_labels = sorted(set(all_true))
    gesture_names = [GESTURE_NAMES.get(int(g), f"G{int(g)}") for g in gesture_labels]
    print("\nClassification Report:")
    print(classification_report(all_true, all_preds,
                                target_names=gesture_names, zero_division=0))

    # ---- Confusion matrix ----
    cm = confusion_matrix(all_true, all_preds, labels=gesture_labels)
    cm_path = os.path.join(out_dir, f'confusion_matrix_intra_2dcnn_{args.window}.png')
    plot_confusion_matrix(
        cm, gesture_names, cm_path,
        title_extra=(f'Intra-Subject Pretrained CNN + 2D CNN Head — {args.window}\n'
                     f'Mean Acc: {accs_pct.mean():.2f}±{accs_pct.std():.2f}%  '
                     f'Overall: {overall_acc * 100:.2f}%'))

    # ---- Per-subject bar chart ----
    subj_path = os.path.join(out_dir,
                             f'per_subject_acc_intra_2dcnn_{args.window}.png')
    plot_per_subject_accuracy(
        accs_pct.tolist(), subj_names, subj_path,
        title_extra=(f'Per-Subject Accuracy — Intra-Subject 2D CNN ({args.window})\n'
                     f'Mean: {accs_pct.mean():.2f}±{accs_pct.std():.2f}%'))

    # ---- Save results JSON ----
    results = {
        'window': args.window,
        'transform': args.transform,
        'freeze_encoder': args.freeze_encoder,
        'mean_accuracy': round(float(accs_pct.mean()), 2),
        'std_accuracy': round(float(accs_pct.std()), 2),
        'overall_accuracy': round(float(overall_acc * 100), 2),
        'min_accuracy': round(float(accs_pct.min()), 2),
        'max_accuracy': round(float(accs_pct.max()), 2),
        'n_subjects': len(accs),
        'elapsed_sec': round(elapsed, 1),
        'per_subject': per_subject_results,
        'args': vars(args),
    }

    results_path = os.path.join(out_dir, 'results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved: {results_path}")


if __name__ == '__main__':
    main()
