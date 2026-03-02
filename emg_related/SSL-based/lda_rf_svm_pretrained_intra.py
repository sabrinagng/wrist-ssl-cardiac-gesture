
import os, sys, json, argparse
from pathlib import Path
from datetime import datetime
from typing import List, Optional

import numpy as np
from tqdm import tqdm
from scipy.signal import stft as scipy_stft

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix

from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import torch
import torch.nn as nn

sys.path.insert(0, '/root/autodl-fs/emg_pretrain')
from model_2d import mean_var_norm_2d

#constants
FS_EMG = 2000
CHUNK_LEN = 8192
EXCLUDED_GESTURES = {6}

GESTURE_NAMES = {
    1: "Hand Close", 2: "Hand Open", 3: "Wrist Flex", 4: "Wrist Ext",
    5: "Point Index", 7: "Little Finger", 8: "Tripod",
    9: "Thumb Flex", 10: "Middle Finger",
}

#encoder loading

def load_pretrained_encoder(pretrained_path: str, device: str = 'cuda') -> nn.Module:
    """Load frozen 2D CNN encoder from MAE checkpoint."""
    ckpt = torch.load(pretrained_path, map_location=device, weights_only=False)
    state = ckpt.get('model_state_dict', ckpt)
    enc_state = {k[8:]: v for k, v in state.items() if k.startswith('encoder.')}

    layers = []
    in_c = 2
    for out_c, k, s in zip([64, 128, 256, 512], [7, 5, 3, 3], [2, 2, 2, 2]):
        layers += [nn.Conv2d(in_c, out_c, k, stride=s, padding=k // 2),
                   nn.BatchNorm2d(out_c), nn.ReLU(inplace=True)]
        in_c = out_c
    encoder = nn.Sequential(*layers)
    encoder.load_state_dict(enc_state, strict=False)
    for p in encoder.parameters():
        p.requires_grad = False
    encoder.to(device).eval()
    print(f"[Encoder] Loaded {len(enc_state)} params, frozen, on {device}")
    return encoder

# feature extraction

def stft_chunk(chunk: np.ndarray, n_fft=256, hop=64) -> np.ndarray:
    """(chunk_len, 2) → (2, F, T) log-magnitude STFT."""
    outs = []
    for ch in range(2):
        f, t, Z = scipy_stft(chunk[:, ch], fs=FS_EMG,
                              nperseg=n_fft, noverlap=n_fft - hop,
                              nfft=n_fft, padded=False, boundary=None)
        mag = np.log1p(np.abs(Z)).astype(np.float32)
        outs.append(mag)
    return np.stack(outs, axis=0)  # (2, F, T)


def split_into_chunks(emg: np.ndarray, chunk_len=CHUNK_LEN,
                      stride: Optional[int] = None) -> List[np.ndarray]:
    if stride is None:
        stride = chunk_len // 2  # 50 % overlap by default
    T = emg.shape[0]
    chunks = []
    start = 0
    while start + chunk_len <= T:
        chunks.append(emg[start:start + chunk_len])
        start += stride
    if len(chunks) == 0 and T > 0:
        pad = np.zeros((chunk_len, 2), dtype=emg.dtype)
        pad[:T] = emg
        chunks.append(pad)
    return chunks


@torch.no_grad()
def extract_feature_vector(encoder: nn.Module, emg: np.ndarray,
                           device='cuda', n_fft=256, hop=64,
                           chunk_stride=None) -> np.ndarray:
    """
    EMG clip (T, 2) → 512-dim feature vector.
    Chunk-and-aggregate: split → STFT → encoder → GAP → mean.
    """
    chunks = split_into_chunks(emg, CHUNK_LEN, chunk_stride)
    feats = []
    for chunk in chunks:
        tf = stft_chunk(chunk, n_fft, hop)                        # (2, F, T)
        x = torch.from_numpy(tf).float().unsqueeze(0).to(device)  # (1,2,F,T)
        x, _, _ = mean_var_norm_2d(x)
        z = encoder(x)                                            # (1,512,H',W')
        z = nn.AdaptiveAvgPool2d(1)(z).squeeze()                  # (512,)
        feats.append(z.cpu().numpy())
    return np.stack(feats).mean(axis=0)  # (512,)

# data loading

def load_subject_data(window_dir: Path, subj: str, excluded: set,
                      encoder: nn.Module, device: str,
                      n_fft: int, hop: int, chunk_stride):
    """
    Returns X_train, y_train, X_test, y_test for one subject.
    Split: reps 1-8 train, reps 9-10 test.
    """
    data_dir = window_dir / "data" / subj
    label_dir = window_dir / "label" / subj
    if not data_dir.exists():
        return None

    Xtr, ytr, Xte, yte = [], [], [], []

    for fname in sorted(data_dir.glob("*.npz")):
        d = np.load(fname, allow_pickle=True)
        gid = int(d['gesture_id'])
        rep = int(d['repetition'])
        if gid in excluded:
            continue

        emg = d['emg'].astype(np.float32)
        if emg.ndim == 1:
            emg = np.stack([emg, emg], axis=1)
        if emg.ndim == 2 and emg.shape[1] != 2:
            continue

        feat = extract_feature_vector(encoder, emg, device, n_fft, hop, chunk_stride)

        if rep in range(1, 9):
            Xtr.append(feat); ytr.append(gid)
        elif rep in (9, 10):
            Xte.append(feat); yte.append(gid)

    if len(Xtr) == 0 or len(Xte) == 0:
        return None
    return (np.stack(Xtr), np.array(ytr, dtype=np.int32),
            np.stack(Xte), np.array(yte, dtype=np.int32))

# classifiers

def train_eval_clf(clf_name, Xtr, ytr, Xte, yte, fast=False, seed=42):
    if clf_name == 'svm':
        pg = ({"C": [1, 10], "gamma": ["scale", 0.01], "kernel": ["rbf"]}
              if fast else
              {"C": [0.1, 1, 10, 100], "gamma": ["scale", 0.001, 0.01, 0.1], "kernel": ["rbf"]})
        gs = GridSearchCV(SVC(), pg, cv=(2 if fast else 3), n_jobs=-1, verbose=0)
    elif clf_name == 'rf':
        pg = ({"n_estimators": [200, 500], "max_depth": [None, 20],
               "min_samples_split": [2, 5], "max_features": ["sqrt"]}
              if fast else
              {"n_estimators": [200, 500, 1000], "max_depth": [None, 10, 20, 40],
               "min_samples_split": [2, 5, 10], "min_samples_leaf": [1, 2, 4],
               "max_features": ["sqrt", "log2"]})
        gs = GridSearchCV(RandomForestClassifier(random_state=seed, n_jobs=-1),
                          pg, cv=(2 if fast else 3), n_jobs=-1, verbose=0)
    elif clf_name == 'lda':
        pg = ([{"solver": ["svd"], "shrinkage": [None]},
               {"solver": ["lsqr"], "shrinkage": [None, "auto"]}]
              if fast else
              [{"solver": ["svd"], "shrinkage": [None]},
               {"solver": ["lsqr", "eigen"], "shrinkage": [None, "auto"]}])
        gs = GridSearchCV(LinearDiscriminantAnalysis(), pg,
                          cv=(2 if fast else 3), n_jobs=-1, verbose=0)
    else:
        raise ValueError(clf_name)

    gs.fit(Xtr, ytr)
    pred = gs.best_estimator_.predict(Xte)
    acc = accuracy_score(yte, pred) * 100.0
    return acc, pred, gs.best_params_, gs.best_score_ * 100.0

#confusion matrix plot

def plot_cm(y_true, y_pred, labels, path, title):
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    cm_n = cm.astype(float) / np.maximum(cm.sum(1, keepdims=True), 1e-12) * 100
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm_n, cmap='Blues', aspect='auto', vmin=0, vmax=100)
    names = [GESTURE_NAMES.get(int(l), str(l)) for l in labels]
    ax.set_xticks(range(len(names))); ax.set_yticks(range(len(names)))
    ax.set_xticklabels(names, rotation=45, ha='right', fontsize=9)
    ax.set_yticklabels(names, fontsize=9)
    ax.set_xlabel('Predicted', fontweight='bold')
    ax.set_ylabel('True', fontweight='bold')
    ax.set_title(title, fontweight='bold')
    for i in range(len(names)):
        for j in range(len(names)):
            color = 'white' if cm_n[i, j] > 50 else 'black'
            ax.text(j, i, f'{cm_n[i,j]:.1f}%', ha='center', va='center',
                    fontsize=8, color=color, fontweight='bold')
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label='Recall %')
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches='tight'); plt.close()

# run one (window, clf)

def run_experiment(window, clf_name, encoder, device, args):
    window_dir = Path(args.data_root) / window
    subjects = [f"S{i:02d}" for i in range(1, 11)]

    per_subj = []
    all_yt, all_yp = [], []

    for subj in tqdm(subjects, desc=f"{window} {clf_name.upper()}"):
        out = load_subject_data(window_dir, subj, EXCLUDED_GESTURES,
                                encoder, device, args.n_fft, args.hop,
                                args.chunk_stride)
        if out is None:
            print(f"  [SKIP] {subj}")
            continue
        Xtr, ytr, Xte, yte = out

        # Standardise per subject
        sc = StandardScaler()
        Xtr_s = sc.fit_transform(Xtr)
        Xte_s = sc.transform(Xte)

        # Optional PCA
        pca_used = None
        if args.pca_dim and args.pca_dim > 0:
            k = min(args.pca_dim, Xtr_s.shape[0], Xtr_s.shape[1])
            if k >= 2:
                pca = PCA(n_components=k, random_state=42)
                Xtr_s = pca.fit_transform(Xtr_s)
                Xte_s = pca.transform(Xte_s)
                pca_used = k

        acc, pred, bp, cv = train_eval_clf(clf_name, Xtr_s, ytr, Xte_s, yte,
                                           fast=args.fast)
        per_subj.append({
            'subject': subj, 'test_acc': acc, 'cv_best': cv,
            'n_train': len(ytr), 'n_test': len(yte),
            'best_params': bp, 'pca_used': pca_used,
        })
        all_yt.extend(yte.tolist())
        all_yp.extend(pred.tolist())
        print(f"  [{subj}] acc={acc:.2f}%  train={len(ytr)} test={len(yte)}")

    if len(per_subj) == 0:
        return None

    accs = np.array([r['test_acc'] for r in per_subj])
    mean_acc = float(accs.mean())
    std_acc = float(accs.std(ddof=1)) if len(accs) > 1 else 0.0
    pooled_acc = accuracy_score(all_yt, all_yp) * 100.0

    return {
        'window': window, 'clf': clf_name,
        'mean_acc': mean_acc, 'std_acc': std_acc,
        'pooled_acc': pooled_acc,
        'n_subjects': len(per_subj),
        'per_subject': per_subj,
        'all_yt': all_yt, 'all_yp': all_yp,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root', default='/root/autodl-fs/windowed_data')
    parser.add_argument('--pretrained', default=(
        '/root/autodl-fs/emg_pretrain/pretrain/checkpoints/'
        'cnn_mae_2d_stft/2026-02-13-14-59-13/ckpt_cnn_mae_2d_epoch_50.pth'))
    parser.add_argument('--out-dir', default='/tmp/pretrained_intra_svm_rf_lda')
    parser.add_argument('--n-fft', type=int, default=256)
    parser.add_argument('--hop', type=int, default=64)
    parser.add_argument('--chunk-stride', type=int, default=None,
                        help='Chunk stride (default: chunk_len//2 = 4096)')
    parser.add_argument('--pca-dim', type=int, default=None)
    parser.add_argument('--fast', action='store_true')
    parser.add_argument('--windows', nargs='+', default=['10s', '16s'])
    parser.add_argument('--clfs', nargs='+', default=['svm', 'rf', 'lda'])
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    encoder = load_pretrained_encoder(args.pretrained, device)

    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_dir = Path(args.out_dir) / ts
    out_dir.mkdir(parents=True, exist_ok=True)

    all_results = []

    for window in args.windows:
        for clf_name in args.clfs:
            res = run_experiment(window, clf_name, encoder, device, args)
            if res is None:
                continue
            all_results.append(res)

            # Save aggregated confusion matrix
            labels = sorted(set(res['all_yt']) | set(res['all_yp']))
            plot_cm(res['all_yt'], res['all_yp'], labels,
                    out_dir / f"cm_{window}_{clf_name}.png",
                    f"Pretrained→{clf_name.upper()} {window} "
                    f"(pooled {res['pooled_acc']:.1f}%)")

            # Print intermediate summary
            print(f"\n  >>> {window} {clf_name.upper()}: "
                  f"Mean={res['mean_acc']:.2f}±{res['std_acc']:.2f}%  "
                  f"Pooled={res['pooled_acc']:.2f}%\n")

    # ── Summary table ──
    print("\n" + "=" * 80)
    print("INTRA-SUBJECT RESULTS: Pretrained 2D CNN Encoder → SVM/RF/LDA")
    print("  Feature: chunk-and-aggregate (chunk=8192, 50% overlap) → encoder → GAP → 512-d")
    print("=" * 80)
    print(f"{'Window':<8} {'CLF':<6} {'Mean Acc':>10} {'± Std':>8} {'Pooled Acc':>12} {'Subjects':>10}")
    print("-" * 80)
    for r in all_results:
        print(f"{r['window']:<8} {r['clf'].upper():<6} "
              f"{r['mean_acc']:>9.2f}% {r['std_acc']:>7.2f}% "
              f"{r['pooled_acc']:>11.2f}% {r['n_subjects']:>10}")
    print("-" * 80)

    # Save JSON (remove large arrays for readability)
    save_results = []
    for r in all_results:
        sr = {k: v for k, v in r.items() if k not in ('all_yt', 'all_yp')}
        save_results.append(sr)
    with open(out_dir / 'results.json', 'w') as f:
        json.dump(save_results, f, indent=2)

    # Save summary table as text
    with open(out_dir / 'summary.txt', 'w') as f:
        f.write("INTRA-SUBJECT: Pretrained 2D CNN Encoder → SVM/RF/LDA\n")
        f.write("Feature: chunk-and-aggregate → encoder → GAP → 512-d\n")
        f.write("=" * 80 + "\n")
        f.write(f"{'Window':<8} {'CLF':<6} {'Mean±Std':>18} {'Pooled':>10}\n")
        f.write("-" * 80 + "\n")
        for r in all_results:
            f.write(f"{r['window']:<8} {r['clf'].upper():<6} "
                    f"{r['mean_acc']:.2f}±{r['std_acc']:.2f}%  "
                    f"{r['pooled_acc']:.2f}%\n")
        f.write("-" * 80 + "\n")

    print(f"\nResults saved to {out_dir}")


if __name__ == '__main__':
    main()
