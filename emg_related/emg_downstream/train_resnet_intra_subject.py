"""
Intra-Subject ResNet Downstream with Pretrained Encoder
========================================================
Train a separate model per subject using reps 1-8 as train, 9-10 as test.
This avoids the inter-subject variability and distribution shift issues.
Reports mean ± std accuracy across all subjects.
"""

import gc, os, sys, json, argparse, math
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
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(0, '/root/autodl-fs/emg_pretrain')
from resnet_emg import GESTURE_NAMES, EXCLUDED_GESTURES, SpectrogramTransform, BasicBlock2d

PRETRAINED = ('/root/autodl-fs/emg_pretrain/pretrain/checkpoints/'
              'cnn_mae_2d_stft/2026-02-13-14-59-13/ckpt_cnn_mae_2d_epoch_50.pth')


class EMGDataset(Dataset):
    def __init__(self, emg_list, labels):
        if len(emg_list) == 0:
            self.emg = torch.zeros(0, 2, 1)
            self.labels = torch.zeros(0, dtype=torch.long)
        else:
            self.emg = torch.from_numpy(
                np.stack([e.T for e in emg_list], axis=0).astype(np.float32))
            self.labels = torch.from_numpy(labels.astype(np.int64))
    def __len__(self): return len(self.labels)
    def __getitem__(self, idx): return self.emg[idx], self.labels[idx]


class SpecAugment(nn.Module):
    def __init__(self, freq_mask=12, time_mask=15, n_freq=2, n_time=2, noise_std=0.04):
        super().__init__()
        self.fm, self.tm = freq_mask, time_mask
        self.nf, self.nt = n_freq, n_time
        self.noise = noise_std

    def forward(self, x):
        if not self.training: return x
        B, C, F, T = x.shape
        if self.noise > 0:
            x = x + torch.randn_like(x) * self.noise
        for b in range(B):
            for _ in range(self.nf):
                f = torch.randint(1, max(2, min(self.fm, F)), (1,)).item()
                f0 = torch.randint(0, max(1, F - f), (1,)).item()
                x[b, :, f0:f0+f, :] = 0
            for _ in range(self.nt):
                t = torch.randint(1, max(2, min(self.tm, T)), (1,)).item()
                t0 = torch.randint(0, max(1, T - t), (1,)).item()
                x[b, :, :, t0:t0+t] = 0
        return x


class IntraSubjectResNet(nn.Module):
    """Lighter model suitable for per-subject training (fewer samples)."""

    def __init__(self, emg_channels=2, num_classes=9, pretrained_path=None,
                 n_fft=256, hop_length=64,
                 head_channels=256, blocks=2, unfreeze_last_n=1,
                 dropout=0.4, mlp_dims=None):
        super().__init__()
        if mlp_dims is None:
            mlp_dims = [128]

        self.target_tf = (8192 - n_fft) // hop_length + 1

        self.tf = SpectrogramTransform(n_fft=n_fft, hop_length=hop_length, normalized=True)
        self.aug = SpecAugment()

        # Encoder
        enc_cfgs = list(zip([64, 128, 256, 512], [7, 5, 3, 3], [2, 2, 2, 2]))
        self.enc = nn.ModuleList()
        in_c = emg_channels
        for out_c, k, s in enc_cfgs:
            self.enc.append(nn.Sequential(
                nn.Conv2d(in_c, out_c, k, stride=s, padding=k//2),
                nn.BatchNorm2d(out_c),
                nn.ReLU(True)))
            in_c = out_c

        self._load_enc(pretrained_path)

        # Freeze
        for i, g in enumerate(self.enc):
            freeze = i < (4 - unfreeze_last_n)
            for p in g.parameters():
                p.requires_grad = not freeze

        # Head: 1 layer of BasicBlocks
        self._in = 512
        ds = nn.Sequential(
            nn.Conv2d(512, head_channels, 1, stride=2, bias=False),
            nn.BatchNorm2d(head_channels))
        self.head = nn.Sequential(
            BasicBlock2d(512, head_channels, stride=2, downsample=ds),
            *[BasicBlock2d(head_channels, head_channels) for _ in range(blocks - 1)])

        self.gap = nn.AdaptiveAvgPool2d(1)

        layers = [nn.LayerNorm(head_channels)]
        prev = head_channels
        for d in mlp_dims:
            layers += [nn.Linear(prev, d), nn.GELU(), nn.Dropout(dropout)]
            prev = d
        layers.append(nn.Linear(prev, num_classes))
        self.clf = nn.Sequential(*layers)

    def _load_enc(self, path):
        if not path or not os.path.exists(path): return
        ckpt = torch.load(path, map_location='cpu', weights_only=False)
        state = ckpt.get('model_state_dict', ckpt)
        m = {0:(0,0),1:(0,1), 3:(1,0),4:(1,1), 6:(2,0),7:(2,1), 9:(3,0),10:(3,1)}
        ns = {}
        for k, v in state.items():
            if not k.startswith('encoder.'): continue
            r = k[8:]; p = r.split('.', 1)
            fi = int(p[0]); pn = p[1] if len(p)>1 else ''
            if fi not in m: continue
            g, s = m[fi]
            ns[f'enc.{g}.{s}.{pn}'] = v
        self.load_state_dict(ns, strict=False)

    def forward(self, emg):
        tf = self.tf(emg)
        if tf.shape[-1] != self.target_tf:
            tf = F.interpolate(tf, size=(tf.shape[2], self.target_tf),
                               mode='bilinear', align_corners=False)
        tf = self.aug(tf)
        z = tf
        for g in self.enc:
            z = g(z)
        z = self.head(z)
        z = self.gap(z).flatten(1)
        return self.clf(z)


def mixup(x, y, alpha=0.3):
    if alpha <= 0: return x, y, y, 1.0
    lam = max(np.random.beta(alpha, alpha), 0.5)
    idx = torch.randperm(x.size(0), device=x.device)
    return lam * x + (1 - lam) * x[idx], y, y[idx], lam


def load_subject(data_dir, label_dir, subj):
    sd, sl = os.path.join(data_dir, subj), os.path.join(label_dir, subj)
    emg_list, labels, reps = [], [], []
    if not os.path.isdir(sd): return emg_list, np.array(labels), np.array(reps)
    for f in sorted(os.listdir(sd)):
        if not f.endswith('.npz'): continue
        dp, lp = os.path.join(sd, f), os.path.join(sl, f)
        if not os.path.exists(lp): continue
        d = np.load(dp, allow_pickle=True)
        gid, rep = int(d['gesture_id']), int(d['repetition'])
        if gid in EXCLUDED_GESTURES: continue
        emg_list.append(d['emg'].astype(np.float32))
        labels.append(gid); reps.append(rep)
    return emg_list, np.array(labels), np.array(reps)


def run_subject(subj, data_dir, label_dir, args, device, label_map=None):
    """Returns (accuracy, preds_original_labels, true_original_labels) or None."""
    emg_list, labels, reps = load_subject(data_dir, label_dir, subj)
    if len(emg_list) == 0:
        print(f"  {subj}: no data"); return None

    # Split
    tri = [i for i,r in enumerate(reps) if r in range(1,9)]
    tei = [i for i,r in enumerate(reps) if r in (9,10)]
    tr_emg = [emg_list[i] for i in tri]
    tr_lab = labels[tri]
    te_emg = [emg_list[i] for i in tei]
    te_lab = labels[tei]

    if len(tr_emg) == 0 or len(te_emg) == 0:
        print(f"  {subj}: insufficient data"); return None

    # Keep original labels for confusion matrix
    te_lab_orig = te_lab.copy()

    # Map labels
    ul = sorted(set(tr_lab) | set(te_lab))
    lm = {l: i for i, l in enumerate(ul)}
    idx_to_label = {i: l for l, i in lm.items()}
    nc = len(ul)
    tr_lab = np.array([lm[l] for l in tr_lab])
    te_lab = np.array([lm[l] for l in te_lab])

    # Val split
    if len(tr_emg) > 10:
        tridx, vidx = train_test_split(np.arange(len(tr_emg)), test_size=0.15,
                                       stratify=tr_lab, random_state=42)
        v_emg = [tr_emg[i] for i in vidx]
        v_lab = tr_lab[vidx]
        tr_emg = [tr_emg[i] for i in tridx]
        tr_lab = tr_lab[tridx]
    else:
        v_emg, v_lab = te_emg, te_lab

    trl = DataLoader(EMGDataset(tr_emg, tr_lab), batch_size=min(args.batch_size, len(tr_emg)),
                     shuffle=True, num_workers=0, pin_memory=True, drop_last=len(tr_emg) > args.batch_size)
    vl = DataLoader(EMGDataset(v_emg, v_lab), batch_size=args.batch_size,
                    shuffle=False, num_workers=0, pin_memory=True)
    tel = DataLoader(EMGDataset(te_emg, te_lab), batch_size=args.batch_size,
                     shuffle=False, num_workers=0, pin_memory=True)

    model = IntraSubjectResNet(
        emg_channels=tr_emg[0].shape[1], num_classes=nc,
        pretrained_path=PRETRAINED,
        head_channels=args.head_ch, blocks=args.blocks,
        unfreeze_last_n=args.unfreeze, dropout=args.dropout,
        mlp_dims=[args.mlp_dim],
    ).to(device)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # Optimizer
    enc_p = [p for n,p in model.named_parameters() if p.requires_grad and 'enc.' in n]
    head_p = [p for n,p in model.named_parameters() if p.requires_grad and 'enc.' not in n]
    pg = [{'params': head_p, 'lr': args.lr}]
    if enc_p:
        pg.append({'params': enc_p, 'lr': args.lr * 0.1})

    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    optimizer = optim.AdamW(pg, weight_decay=args.wd)

    total_steps = args.epochs * len(trl)
    warmup = int(0.05 * total_steps)
    def lr_fn(step):
        if step < warmup: return step / max(warmup, 1)
        prog = (step - warmup) / max(total_steps - warmup, 1)
        return 0.5 * (1 + math.cos(math.pi * prog))
    sched = optim.lr_scheduler.LambdaLR(optimizer, lr_fn)
    scaler = GradScaler('cuda') if device.type == 'cuda' else None

    best_va, best_ep, pat = 0., 0, 0
    best_state = None

    for ep in range(1, args.epochs + 1):
        model.train()
        for x, y in trl:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            if args.mixup > 0:
                xm, ya, yb, lam = mixup(x, y, args.mixup)
            else:
                xm, ya, yb, lam = x, y, y, 1.0
            if scaler:
                with autocast('cuda'):
                    lo = model(xm)
                    loss = lam * criterion(lo, ya) + (1-lam) * criterion(lo, yb)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer); scaler.update()
            else:
                lo = model(xm)
                loss = lam * criterion(lo, ya) + (1-lam) * criterion(lo, yb)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            sched.step()

        # Val
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
            best_va = va; best_ep = ep; pat = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            pat += 1
        if pat >= args.patience:
            break

    # Test
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
    gc.collect(); torch.cuda.empty_cache()
    return te_acc, preds_orig, true_orig


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--window', default='10s')
    parser.add_argument('--data-dir', default='/root/autodl-fs/windowed_data')
    parser.add_argument('--epochs', type=int, default=120)
    parser.add_argument('--patience', type=int, default=25)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--wd', type=float, default=1e-4)
    parser.add_argument('--head-ch', type=int, default=256)
    parser.add_argument('--blocks', type=int, default=2)
    parser.add_argument('--unfreeze', type=int, default=1)
    parser.add_argument('--dropout', type=float, default=0.4)
    parser.add_argument('--mlp-dim', type=int, default=128)
    parser.add_argument('--mixup', type=float, default=0.3)
    parser.add_argument('--n-subjects', type=int, default=20)
    parser.add_argument('--out-dir', type=str, default='/tmp/resnet_intra_results')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ws = int(args.window.replace('s',''))
    dd = os.path.join(args.data_dir, f"{ws}s", "data")
    ld = os.path.join(args.data_dir, f"{ws}s", "label")

    subjects = [f"S{i:02d}" for i in range(1, args.n_subjects + 1)]

    print(f"{'='*60}")
    print(f"Intra-Subject ResNet Downstream")
    print(f"Window: {args.window}  Subjects: {len(subjects)}")
    print(f"head_ch={args.head_ch} blocks={args.blocks} unfreeze={args.unfreeze}")
    print(f"dropout={args.dropout} mlp={args.mlp_dim} lr={args.lr} mixup={args.mixup}")
    print(f"{'='*60}")

    accs = []
    all_preds = []
    all_true = []
    subj_names = []
    for subj in subjects:
        result = run_subject(subj, dd, ld, args, device)
        if result is not None:
            acc, preds, true = result
            accs.append(acc)
            all_preds.extend(preds)
            all_true.extend(true)
            subj_names.append(subj)

    accs = np.array(accs) * 100
    all_preds = np.array(all_preds)
    all_true = np.array(all_true)

    overall_acc = accuracy_score(all_true, all_preds)

    print(f"\n{'='*60}")
    print(f"RESULTS — {args.window} Intra-Subject ResNet")
    print(f"{'='*60}")
    print(f"  Mean ± Std: {accs.mean():.2f} ± {accs.std():.2f}%")
    print(f"  Overall (pooled preds): {overall_acc*100:.2f}%")
    print(f"  Min: {accs.min():.2f}%  Max: {accs.max():.2f}%")
    print(f"  All: {', '.join(f'{a:.1f}' for a in accs)}")
    print(f"{'='*60}")

    # Classification report
    gesture_labels = sorted(set(all_true))
    gesture_names = [GESTURE_NAMES.get(int(g), f"G{int(g)}") for g in gesture_labels]
    print("\nClassification Report:")
    print(classification_report(all_true, all_preds,
          target_names=gesture_names, zero_division=0))

    # Confusion matrix
    cm = confusion_matrix(all_true, all_preds, labels=gesture_labels)
    cm_norm = cm.astype('float') / cm.sum(axis=1, keepdims=True) * 100

    # Plot
    out_dir = args.out_dir if hasattr(args, 'out_dir') and args.out_dir else '/tmp'
    os.makedirs(out_dir, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(18, 7))

    # Left: counts + percentage
    im0 = axes[0].imshow(cm, interpolation='nearest', cmap='Blues', aspect='auto')
    thresh0 = cm.max() / 2.0
    for i in range(len(gesture_labels)):
        for j in range(len(gesture_labels)):
            color = 'white' if cm[i,j] > thresh0 else 'black'
            axes[0].text(j, i, f'{cm[i,j]}\n({cm_norm[i,j]:.1f}%)',
                        ha='center', va='center', fontsize=8, color=color, fontweight='bold')
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
    for i in range(len(gesture_labels)):
        for j in range(len(gesture_labels)):
            color = 'white' if cm_norm[i,j] > thresh1 else 'black'
            axes[1].text(j, i, f'{cm_norm[i,j]:.1f}%\n({cm[i,j]})',
                        ha='center', va='center', fontsize=8, color=color, fontweight='bold')
    axes[1].set_xticks(range(len(gesture_names)))
    axes[1].set_yticks(range(len(gesture_names)))
    axes[1].set_xticklabels(gesture_names, rotation=45, ha='right', fontsize=9)
    axes[1].set_yticklabels(gesture_names, fontsize=9)
    axes[1].set_xlabel('Predicted', fontsize=11, fontweight='bold')
    axes[1].set_ylabel('True', fontsize=11, fontweight='bold')
    axes[1].set_title('Confusion Matrix — Normalized (%)', fontsize=12, fontweight='bold')
    plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04, label='Recall (%)')

    fig.suptitle(f'Intra-Subject Pretrained CNN + ResNet Head — {args.window}\n'
                 f'Mean Acc: {accs.mean():.2f}±{accs.std():.2f}%  '
                 f'Overall: {overall_acc*100:.2f}%',
                 fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()
    cm_path = os.path.join(out_dir, f'confusion_matrix_intra_resnet_{args.window}.png')
    plt.savefig(cm_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"\nConfusion matrix saved: {cm_path}")

    # Per-subject accuracy bar chart
    fig2, ax2 = plt.subplots(figsize=(10, 5))
    colors = ['#e74c3c' if a < 60 else '#f39c12' if a < 75 else '#2ecc71' for a in accs]
    bars = ax2.bar(range(len(accs)), accs, color=colors, edgecolor='black', alpha=0.85)
    for bar, a in zip(bars, accs):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{a:.1f}%', ha='center', fontsize=9, fontweight='bold')
    ax2.set_xticks(range(len(accs)))
    ax2.set_xticklabels(subj_names, rotation=45, ha='right')
    ax2.set_ylabel('Test Accuracy (%)')
    ax2.set_title(f'Per-Subject Accuracy — Intra-Subject Pretrained CNN + ResNet Head ({args.window})\n'
                  f'Mean: {accs.mean():.2f}±{accs.std():.2f}%')
    ax2.axhline(y=accs.mean(), color='red', linestyle='--', linewidth=2,
                label=f'Mean: {accs.mean():.1f}%')
    ax2.set_ylim(0, 105)
    ax2.legend(); ax2.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    subj_path = os.path.join(out_dir, f'per_subject_acc_intra_resnet_{args.window}.png')
    plt.savefig(subj_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Per-subject chart saved: {subj_path}")


if __name__ == '__main__':
    main()
