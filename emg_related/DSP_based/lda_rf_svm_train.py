"""
Intra-subject TF (2 x F x T) Classification Sweep with SVM / RF / LDA
====================================================================
- EMG only
- Each clip -> time-frequency representation (STFT / CWT / LogMel)
- Each clip becomes ONE sample: X = flatten(2*F*T)
- Standardize per subject + optional PCA
- Intra-subject REP-BASED split (reps 1-8 train, 9-10 test) to avoid leakage
- Outputs per-subject results + mean/std across subjects
- Optional confusion matrix (ALL dims) per subject
- Aggregated confusion matrix across all subjects

Examples:
  python lda_rf_svm_train.py --data-root ./EMG_classification/windowed_data --window 16s --subjects S01-S20 \
    --transform stft --clf svm --split 0.8 --pca-dim 256 --cache-tf --save-cm

  python lda_rf_svm_train.py --window 16s --subjects S01-S20 --transform logmel --clf lda --pca-dim 128
"""

import os
import json
import argparse
from pathlib import Path
from datetime import datetime

import numpy as np
from tqdm import tqdm

from scipy.signal import stft

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix

from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

import matplotlib.pyplot as plt
import hashlib

def tf_cache_tag(transform: str, tf_kwargs: dict) -> str:
    s = json.dumps({"transform": transform, "tf_kwargs": tf_kwargs}, sort_keys=True)
    return hashlib.md5(s.encode("utf-8")).hexdigest()[:10]

def pad_or_trim_tf(x: np.ndarray, target_T: int) -> np.ndarray:
    """
    x: (2, F, T) -> (2, F, target_T)
    """
    if x.ndim != 3 or x.shape[0] != 2:
        raise ValueError(f"TF must be (2,F,T), got {x.shape}")
    C, F, T = x.shape
    if T == target_T:
        return x
    if T > target_T:
        # center crop
        start = (T - target_T) // 2
        return x[:, :, start:start + target_T]
    out = np.zeros((C, F, target_T), dtype=x.dtype)
    out[:, :, :T] = x
    return out

def align_tf_lists(Xtr_list, Xte_list, strategy="median"):
    Ts = [x.shape[-1] for x in Xtr_list] + [x.shape[-1] for x in Xte_list]
    if len(Ts) == 0:
        return Xtr_list, Xte_list, None
    Ts = np.array(Ts, dtype=np.int32)
    if strategy == "min":
        target_T = int(Ts.min())
    elif strategy == "max":
        target_T = int(Ts.max())
    else:
        target_T = int(np.median(Ts))  # robust default
    if target_T < 1:
        return Xtr_list, Xte_list, None
    Xtr_list = [pad_or_trim_tf(x, target_T) for x in Xtr_list]
    Xte_list = [pad_or_trim_tf(x, target_T) for x in Xte_list]
    return Xtr_list, Xte_list, target_T


# Optional deps
try:
    import librosa
except Exception:
    librosa = None

try:
    import pywt
except Exception:
    pywt = None


# -------------------- constants --------------------

FS_EMG = 2000
EXCLUDED_GESTURES_DEFAULT = {6}  # Cut Something

GESTURE_NAMES = {
    1: "Hand Close",
    2: "Hand Open",
    3: "Wrist Flex",
    4: "Wrist Ext",
    5: "Point Index",
    7: "Little Finger",
    8: "Tripod",
    9: "Thumb Flex",
    10: "Middle Finger",
}


# -------------------- helpers --------------------

def parse_range(s: str):
    p = s.split("-")
    if len(p) != 2:
        raise ValueError(f"Bad subject range: {s} (expect like S01-S20)")
    a = int(p[0][1:])
    b = int(p[1][1:])
    return [f"S{i:02d}" for i in range(a, b + 1)]


def plot_confusion_matrix(y_true, y_pred, labels, out_path: Path, title: str):
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    cm_norm = cm.astype(np.float64)
    row_sum = cm_norm.sum(axis=1, keepdims=True)
    cm_norm = np.divide(cm_norm, np.maximum(row_sum, 1e-12))

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    im = ax.imshow(cm, aspect="auto")
    ax.set_title(f"{title}\n(Counts)")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels([GESTURE_NAMES.get(int(l), str(l)) for l in labels], rotation=45, ha="right")
    ax.set_yticklabels([GESTURE_NAMES.get(int(l), str(l)) for l in labels])
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax = axes[1]
    im = ax.imshow(cm_norm, aspect="auto")
    ax.set_title(f"{title}\n(Normalized)")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels([GESTURE_NAMES.get(int(l), str(l)) for l in labels], rotation=45, ha="right")
    ax.set_yticklabels([GESTURE_NAMES.get(int(l), str(l)) for l in labels])
    for i in range(cm_norm.shape[0]):
        for j in range(cm_norm.shape[1]):
            ax.text(j, i, f"{cm_norm[i, j]:.2f}", ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_confusion_matrix_pct(y_true, y_pred, labels, out_path: Path, title: str):
    """Plot a single normalized (percentage) confusion matrix with seaborn heatmap."""
    import seaborn as sns
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    cm_pct = cm.astype(np.float64) / np.maximum(cm.sum(axis=1, keepdims=True), 1e-12) * 100.0
    names = [GESTURE_NAMES.get(int(l), str(l)) for l in labels]

    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(cm_pct, annot=True, fmt='.1f', cmap='Blues', ax=ax,
                xticklabels=names, yticklabels=names,
                vmin=0, vmax=100, cbar_kws={'label': '%'})
    ax.set_title(title)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.tick_params(axis='x', rotation=45)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {out_path}")


def _ensure_2ch(emg):
    emg = np.asarray(emg)
    if emg.ndim == 1:
        emg = np.stack([emg, emg], axis=1)
    if emg.ndim == 2 and emg.shape[1] == 1:
        emg = np.concatenate([emg, emg], axis=1)
    if not (emg.ndim == 2 and emg.shape[1] == 2):
        raise ValueError(f"Bad EMG shape {emg.shape}, expect (T,2)")
    return emg.astype(np.float32)


# -------------------- label/clip loading --------------------

def get_gesture_id(npz_path: Path, label_path: Path) -> int:
    """
    Prefer label file if exists (one-hot). Fallback to gesture_id inside data npz.
    """
    if label_path.exists():
        z = np.load(label_path, allow_pickle=True)
        if "label" in z.files:
            label = z["label"]
            # one-hot -> 1-based gesture id
            return int(np.argmax(label) + 1)
    z = np.load(npz_path, allow_pickle=True)
    if "gesture_id" in z.files:
        return int(z["gesture_id"])
    return int(z.get("gesture_id", 0))


def get_repetition(npz_path: Path) -> int:
    """Extract repetition number from data npz file."""
    z = np.load(npz_path, allow_pickle=True)
    if "repetition" in z.files:
        return int(z["repetition"])
    raise ValueError(f"No 'repetition' field in {npz_path}")


def load_subject_clips(window_dir: Path, subj: str, excluded: set, max_clips: int | None):
    """Load clips with gesture id AND repetition number."""
    data_dir = window_dir / "data" / subj
    label_dir = window_dir / "label" / subj
    if not data_dir.exists():
        return []

    files = sorted([p.name for p in data_dir.glob("*.npz")])
    if max_clips is not None:
        files = files[:max_clips]

    usable = []
    for name in files:
        data_path = data_dir / name
        gid = get_gesture_id(data_path, label_dir / name)
        if gid == 0 or gid in excluded:
            continue
        rep = get_repetition(data_path)
        usable.append((name, gid, rep))
    return usable


def rep_based_split(clips_with_labels_reps, train_reps=None, test_reps=None):
    """Split clips by repetition number to prevent data leakage.

    All clips from the same repetition go to the same set.
    Default: reps 1-8 train, reps 9-10 test.
    """
    if train_reps is None:
        train_reps = set(range(1, 9))   # {1,2,3,4,5,6,7,8}
    if test_reps is None:
        test_reps = set(range(9, 11))    # {9, 10}

    train_clips = []
    test_clips = []

    for clip_name, gid, rep in clips_with_labels_reps:
        if rep in train_reps:
            train_clips.append(clip_name)
        elif rep in test_reps:
            test_clips.append(clip_name)
        # clips with reps outside both sets are silently skipped

    return train_clips, test_clips


# -------------------- TF transforms (2 x F x T) --------------------

def tf_stft_2ch(emg_2ch, n_fft=256, hop=64, fmax=450.0):
    emg_2ch = _ensure_2ch(emg_2ch)
    outs = []
    for ch in range(2):
        f, t, Z = stft(
            emg_2ch[:, ch],
            fs=FS_EMG,
            nperseg=n_fft,
            noverlap=n_fft - hop,
            nfft=n_fft,
            padded=False,
            boundary=None
        )
        mag = np.abs(Z)  # (F, T)
        keep = f <= fmax
        mag = mag[keep, :]
        mag = np.log1p(mag).astype(np.float32)
        outs.append(mag)
    return np.stack(outs, axis=0)  # (2, F, T)


def tf_logmel_2ch(emg_2ch, n_fft=256, hop=64, n_mels=128, fmin=20.0, fmax=450.0):
    if librosa is None:
        raise RuntimeError("librosa not installed. Run: pip install librosa")
    emg_2ch = _ensure_2ch(emg_2ch)
    outs = []
    for ch in range(2):
        y = emg_2ch[:, ch].astype(np.float32)
        S = librosa.feature.melspectrogram(
            y=y,
            sr=FS_EMG,
            n_fft=n_fft,
            hop_length=hop,
            n_mels=n_mels,
            fmin=fmin,
            fmax=fmax,
            power=2.0,
        )  # (M, T)
        S = np.log1p(S).astype(np.float32)
        outs.append(S)
    return np.stack(outs, axis=0)  # (2, M, T)


def tf_cwt_2ch(emg_2ch, n_scales=128, fmin=20.0, fmax=450.0, wavelet="morl", hop=64):
    if pywt is None:
        raise RuntimeError("pywavelets not installed. Run: pip install pywavelets")
    emg_2ch = _ensure_2ch(emg_2ch)

    freqs = np.logspace(np.log10(fmin), np.log10(fmax), n_scales)
    w = pywt.ContinuousWavelet(wavelet)
    k = pywt.scale2frequency(w, 1.0) * FS_EMG
    scales = k / freqs

    outs = []
    for ch in range(2):
        coef, _ = pywt.cwt(
            emg_2ch[:, ch],
            scales=scales,
            wavelet=wavelet,
            sampling_period=1 / FS_EMG
        )  # (n_scales, T)
        mag = np.abs(coef).astype(np.float32)
        mag = np.log1p(mag)
        if hop > 1:
            mag = mag[:, ::hop]
        outs.append(mag)
    return np.stack(outs, axis=0)  # (2, n_scales, T_down)


def load_clip_tf(window_dir: Path,
                 subj: str,
                 clip_name: str,
                 excluded: set,
                 transform: str,
                 tf_kwargs: dict,
                 cache_tf: bool):
    data_path = window_dir / "data" / subj / clip_name
    label_path = window_dir / "label" / subj / clip_name
    if not data_path.exists():
        return None

    gid = get_gesture_id(data_path, label_path)
    if gid == 0 or gid in excluded:
        return None

    cache_file = None
    if cache_tf:
        tag = tf_cache_tag(transform, tf_kwargs)
        cache_dir = window_dir / f"tf_cache_{transform}_{tag}" / subj
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_file = cache_dir / clip_name

        if cache_file.exists():
            try:
                z = np.load(cache_file, allow_pickle=True)
                x = z["x"].astype(np.float32)
                y = int(z["y"])
                if x.ndim != 3 or x.shape[0] != 2:
                    raise ValueError(f"Bad cached tf shape: {x.shape}")
                return x, y
            except Exception:
                try:
                    cache_file.unlink()
                except Exception:
                    pass

    z = np.load(data_path, allow_pickle=True)
    if "emg" not in z.files:
        return None
    emg = _ensure_2ch(z["emg"])

    if transform == "stft":
        tf = tf_stft_2ch(emg, **tf_kwargs)
    elif transform == "logmel":
        tf = tf_logmel_2ch(emg, **tf_kwargs)
    elif transform == "cwt":
        tf = tf_cwt_2ch(emg, **tf_kwargs)
    else:
        raise ValueError(transform)

    if cache_tf and cache_file is not None:
        np.savez_compressed(cache_file, x=tf.astype(np.float32), y=np.int32(gid))

    return tf.astype(np.float32), int(gid)



def stack_clips_to_Xy(window_dir: Path,
                      subj: str,
                      clip_list: list[str],
                      excluded: set,
                      transform: str,
                      tf_kwargs: dict,
                      cache_tf: bool):
    X_list, y_list = [], []
    for clip in clip_list:
        out = load_clip_tf(window_dir, subj, clip, excluded, transform, tf_kwargs, cache_tf)
        if out is None:
            continue
        tf, gid = out
        X_list.append(tf)
        y_list.append(gid)
    if len(X_list) == 0:
        return None
    return X_list, np.array(y_list, dtype=np.int32)


# -------------------- trainers: SVM / RF / LDA --------------------

def train_and_eval(clf_name: str, Xtr, ytr, Xte, yte, fast: bool, seed: int):
    """
    Return: acc(%), pred, best_params(dict or None), best_cv(% or None)
    """
    clf_name = clf_name.lower()

    if clf_name == "svm":
        if fast:
            param_grid = {"C": [1, 10], "gamma": ["scale", 0.01], "kernel": ["rbf"]}
            cv_folds = 2
        else:
            param_grid = {"C": [0.1, 1, 10, 100], "gamma": ["scale", 0.001, 0.01, 0.1], "kernel": ["rbf"]}
            cv_folds = 3

        gs = GridSearchCV(SVC(), param_grid, cv=cv_folds, n_jobs=-1, verbose=0)
        gs.fit(Xtr, ytr)
        best = gs.best_estimator_
        pred = best.predict(Xte)
        acc = accuracy_score(yte, pred) * 100.0
        return acc, pred, gs.best_params_, gs.best_score_ * 100.0

    if clf_name == "rf":
        if fast:
            param_grid = {
                "n_estimators": [200, 500],
                "max_depth": [None, 20],
                "min_samples_split": [2, 5],
                "max_features": ["sqrt"],
            }
            cv_folds = 2
        else:
            param_grid = {
                "n_estimators": [200, 500, 1000],
                "max_depth": [None, 10, 20, 40],
                "min_samples_split": [2, 5, 10],
                "min_samples_leaf": [1, 2, 4],
                "max_features": ["sqrt", "log2"],
            }
            cv_folds = 3

        base = RandomForestClassifier(random_state=seed, n_jobs=-1)
        gs = GridSearchCV(base, param_grid, cv=cv_folds, n_jobs=-1, verbose=0)
        gs.fit(Xtr, ytr)
        best = gs.best_estimator_
        pred = best.predict(Xte)
        acc = accuracy_score(yte, pred) * 100.0
        return acc, pred, gs.best_params_, gs.best_score_ * 100.0

    if clf_name == "lda":
        if fast:
            param_grid = [
                {"solver": ["svd"], "shrinkage": [None]},
                {"solver": ["lsqr"], "shrinkage": [None, "auto"]},
            ]
            cv_folds = 2
        else:
            param_grid = [
                {"solver": ["svd"], "shrinkage": [None]},
                {"solver": ["lsqr", "eigen"], "shrinkage": [None, "auto"]},
            ]
            cv_folds = 3

        base = LinearDiscriminantAnalysis()
        gs = GridSearchCV(base, param_grid, cv=cv_folds, n_jobs=-1, verbose=0)
        gs.fit(Xtr, ytr)
        best = gs.best_estimator_
        pred = best.predict(Xte)
        acc = accuracy_score(yte, pred) * 100.0
        return acc, pred, gs.best_params_, gs.best_score_ * 100.0

    raise ValueError(f"Unknown clf: {clf_name} (use svm|rf|lda)")


# -------------------- per-subject experiment --------------------

def run_one_subject(window_dir: Path, subj: str, excluded: set,
                    train_reps: set, test_reps: set,
                    max_clips: int | None,
                    fast: bool, cache_tf: bool,
                    save_cm: bool, cm_dir: Path | None,
                    clf_name: str, seed: int,
                    transform: str, tf_kwargs: dict,
                    pca_dim: int | None):

    clips_with_labels_reps = load_subject_clips(window_dir, subj, excluded, max_clips)
    if len(clips_with_labels_reps) < 2:
        return None

    # ---- Rep-based split (no leakage) ----
    train_clips, test_clips = rep_based_split(clips_with_labels_reps, train_reps, test_reps)
    if len(train_clips) == 0 or len(test_clips) == 0:
        return None

    # Log which reps ended up in which set
    rep_map = {name: rep for name, gid, rep in clips_with_labels_reps}
    actual_train_reps = sorted(set(rep_map[c] for c in train_clips))
    actual_test_reps = sorted(set(rep_map[c] for c in test_clips))

    out_tr = stack_clips_to_Xy(window_dir, subj, train_clips, excluded, transform, tf_kwargs, cache_tf)
    out_te = stack_clips_to_Xy(window_dir, subj, test_clips, excluded, transform, tf_kwargs, cache_tf)
    if out_tr is None or out_te is None:
        return None

    Xtr_list, ytr = out_tr
    Xte_list, yte = out_te
    
    # ---- Align TF shape across clips (fix np.stack crash) ----
    Xtr_list, Xte_list, target_T = align_tf_lists(Xtr_list, Xte_list, strategy="median")
    if target_T is None:
        return None
    # ---- Vectorize: (N, D) ----
    Xtr = np.stack([x.reshape(-1) for x in Xtr_list], axis=0).astype(np.float32)
    Xte = np.stack([x.reshape(-1) for x in Xte_list], axis=0).astype(np.float32)

    # ---- Standardize per subject ----
    scaler = StandardScaler()
    Xtr_s = scaler.fit_transform(Xtr)
    Xte_s = scaler.transform(Xte)

    # ---- Optional PCA ----
    pca_used = None
    if pca_dim is not None:
        max_k = min(Xtr_s.shape[0], Xtr_s.shape[1])
        k = min(pca_dim, max_k)
        if k < 2:
            k = None
        if k is not None:
            pca = PCA(n_components=k, random_state=seed)
            Xtr_s = pca.fit_transform(Xtr_s)
            Xte_s = pca.transform(Xte_s)
            pca_used = k

    labels_sorted = sorted(set(ytr.tolist()) | set(yte.tolist()))

    acc, pred, best_params, best_cv = train_and_eval(
        clf_name, Xtr_s, ytr, Xte_s, yte, fast=fast, seed=seed
    )

    split_info = {
        "n_clips_total": len(clips_with_labels_reps),
        "n_clips_train": len(train_clips),
        "n_clips_test": len(test_clips),
        "train_reps": actual_train_reps,
        "test_reps": actual_test_reps,
        "train_gestures": sorted(set(ytr.tolist())),
        "test_gestures": sorted(set(yte.tolist())),
        "gestures_only_in_train": sorted(set(ytr.tolist()) - set(yte.tolist())),
    }

    if save_cm and cm_dir is not None:
        cm_path = cm_dir / f"{subj}_cm_{transform}_{clf_name}.png"
        plot_confusion_matrix(
            yte, pred, labels_sorted,
            cm_path,
            title=f"{subj} {transform.upper()} {clf_name.upper()} acc={acc:.2f}%"
        )

    # per-gesture accuracy
    per_gesture = {}
    for g in labels_sorted:
        mask = (yte == g)
        if np.any(mask):
            per_gesture[int(g)] = float(accuracy_score(yte[mask], pred[mask]) * 100.0)

    return {
        "subject": subj,
        "clf": clf_name,
        "transform": transform,
        "tf_kwargs": tf_kwargs,
        "pca_dim_requested": pca_dim,
        "pca_dim_used": pca_used,
        "test_acc": float(acc),
        "cv_best": float(best_cv) if best_cv is not None else None,
        "best_params": best_params,
        "n_train_clips": int(len(train_clips)),
        "n_test_clips": int(len(test_clips)),
        "n_train_samples": int(len(ytr)),
        "n_test_samples": int(len(yte)),
        "X_dim_flat": int(Xtr.shape[1]),
        "X_dim_after_pca": int(Xtr_s.shape[1]),
        "split_info": split_info,
        "per_gesture": per_gesture,
        # ---- ADDED: raw predictions for aggregation ----
        "y_true": yte.tolist(),
        "y_pred": pred.tolist(),
    }


# -------------------- main --------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=str, default="./EMG_classification/windowed_data")
    parser.add_argument("--window", type=str, default="16s")
    parser.add_argument("--subjects", type=str, default="S01-S20")

    parser.add_argument("--transform", choices=["stft", "cwt", "logmel"], default="stft")
    parser.add_argument("--clf", choices=["svm", "rf", "lda"], default="svm")

    parser.add_argument("--exclude-gestures", type=int, nargs="*", default=[])

    parser.add_argument("--fast", action="store_true")
    parser.add_argument("--max-clips-per-subj", type=int, default=None)

    parser.add_argument("--cache-tf", action="store_true")
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--pca-dim", type=int, default=256,
                        help="PCA dims after scaling (recommended 128/256/512). Use 0 to disable.")
    parser.add_argument("--save-cm", action="store_true",
                        help="Save confusion matrix per subject")
    parser.add_argument("--out-dir", type=str, default="./EMG_classification/results_intra_all_tf")

    # Rep-based split
    parser.add_argument("--train-reps", type=int, nargs="+", default=[1,2,3,4,5,6,7,8],
                        help="Repetition numbers for training (default: 1-8)")
    parser.add_argument("--test-reps", type=int, nargs="+", default=[9,10],
                        help="Repetition numbers for testing (default: 9-10)")

    # TF params
    parser.add_argument("--n-fft", type=int, default=256)
    parser.add_argument("--hop", type=int, default=64)
    parser.add_argument("--n-mels", type=int, default=128)
    parser.add_argument("--n-scales", type=int, default=128)
    parser.add_argument("--fmin", type=float, default=20.0)
    parser.add_argument("--fmax", type=float, default=450.0)

    args = parser.parse_args()

    window_dir = Path(args.data_root) / args.window
    if not window_dir.exists():
        raise FileNotFoundError(f"Window dir not found: {window_dir}")

    subs = parse_range(args.subjects)
    excluded = set(EXCLUDED_GESTURES_DEFAULT) | set(args.exclude_gestures)

    pca_dim = None if args.pca_dim == 0 else int(args.pca_dim)

    train_reps = set(args.train_reps)
    test_reps = set(args.test_reps)

    # build tf kwargs
    tf_kwargs = {}
    if args.transform == "stft":
        tf_kwargs = {"n_fft": args.n_fft, "hop": args.hop, "fmax": args.fmax}
    elif args.transform == "logmel":
        tf_kwargs = {"n_fft": args.n_fft, "hop": args.hop, "n_mels": args.n_mels,
                     "fmin": args.fmin, "fmax": args.fmax}
    elif args.transform == "cwt":
        tf_kwargs = {"n_scales": args.n_scales, "fmin": args.fmin, "fmax": args.fmax,
                     "wavelet": "morl", "hop": args.hop}

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.out_dir) / (
        f"{args.window}_intra_all_{args.subjects}_{args.transform}_{args.clf}"
        f"_rep{sorted(train_reps)[0]}-{sorted(train_reps)[-1]}v{sorted(test_reps)[0]}-{sorted(test_reps)[-1]}"
        f"_pca{(pca_dim if pca_dim is not None else 0)}_{timestamp}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    cm_dir = None
    if args.save_cm:
        cm_dir = out_dir / "confusion_matrices"
        cm_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 100)
    print("INTRA-ALL subject-wise TF classification (clip-level) + mean/std")
    print("=" * 100)
    print(f"Window: {args.window} | Subjects: {subs[0]}..{subs[-1]} ({len(subs)})")
    print(f"Transform: {args.transform.upper()} | TF kwargs: {tf_kwargs}")
    print(f"CLF: {args.clf.upper()} | Split: rep-based (train reps={sorted(train_reps)}, test reps={sorted(test_reps)})")
    print(f"Excluded gestures: {sorted(excluded)}")
    print(f"PCA: {'OFF' if pca_dim is None else pca_dim}")
    print(f"Fast grid: {args.fast} | cache_tf: {args.cache_tf} | max_clips_per_subj: {args.max_clips_per_subj}")
    print(f"save_cm: {args.save_cm}")
    print(f"Output: {out_dir}")
    print("=" * 100)

    per_subject = []
    global_y_true = []
    global_y_pred = []

    for subj in tqdm(subs, desc="Subjects"):
        out = run_one_subject(
            window_dir=window_dir,
            subj=subj,
            excluded=excluded,
            train_reps=train_reps,
            test_reps=test_reps,
            max_clips=args.max_clips_per_subj,
            fast=args.fast,
            cache_tf=args.cache_tf,
            save_cm=args.save_cm,
            cm_dir=cm_dir,
            clf_name=args.clf,
            seed=args.seed,
            transform=args.transform,
            tf_kwargs=tf_kwargs,
            pca_dim=pca_dim
        )
        if out is None:
            print(f"[SKIP] {subj} (insufficient data after filtering/splitting)")
            continue
        per_subject.append(out)
        global_y_true.extend(out["y_true"])
        global_y_pred.extend(out["y_pred"])

        si = out["split_info"]
        print(f"[{subj}] acc={out['test_acc']:.2f}% | "
              f"clips={si['n_clips_train']}/{si['n_clips_test']} | "
              f"train_reps={si['train_reps']} test_reps={si['test_reps']} | "
              f"dim_flat={out['X_dim_flat']} -> dim={out['X_dim_after_pca']}")

    if len(per_subject) == 0:
        raise RuntimeError("No subjects produced results.")

    accs = np.array([r["test_acc"] for r in per_subject], dtype=np.float64)
    mean_acc = float(np.mean(accs))
    std_acc = float(np.std(accs, ddof=1)) if len(accs) > 1 else 0.0

    # ---- Aggregated confusion matrix across all subjects ----
    if len(global_y_true) > 0:
        global_y_true_arr = np.array(global_y_true, dtype=np.int32)
        global_y_pred_arr = np.array(global_y_pred, dtype=np.int32)
        global_acc = accuracy_score(global_y_true_arr, global_y_pred_arr) * 100.0
        all_labels = sorted(set(global_y_true_arr.tolist()) | set(global_y_pred_arr.tolist()))

        # Counts + Normalized (dual panel)
        plot_confusion_matrix(
            global_y_true_arr, global_y_pred_arr, all_labels,
            out_dir / "confusion_matrix_all_subjects.png",
            title=f"All Subjects {args.transform.upper()} {args.clf.upper()} "
                  f"({args.window}) - {global_acc:.1f}%"
        )

        # Percentage heatmap (single panel, seaborn)
        plot_confusion_matrix_pct(
            global_y_true_arr, global_y_pred_arr, all_labels,
            out_dir / "confusion_matrix_all_subjects_pct.png",
            title=f"All Subjects {args.transform.upper()} {args.clf.upper()} "
                  f"({args.window}) - {global_acc:.1f}%"
        )

        # Save raw predictions for later re-analysis
        np.savez_compressed(
            out_dir / "predictions_all_subjects.npz",
            y_true=global_y_true_arr,
            y_pred=global_y_pred_arr,
            subjects=np.array([r["subject"] for r in per_subject
                               for _ in r["y_true"]]),
        )
        print(f"  Aggregated CM saved: {out_dir / 'confusion_matrix_all_subjects.png'}")
        print(f"  Aggregated CM (%) saved: {out_dir / 'confusion_matrix_all_subjects_pct.png'}")
        print(f"  Predictions saved: {out_dir / 'predictions_all_subjects.npz'}")
        print(f"  Aggregated accuracy (pooled): {global_acc:.1f}%")
    else:
        global_acc = None

    summary = {
        "clf": args.clf,
        "transform": args.transform,
        "tf_kwargs": tf_kwargs,
        "window": args.window,
        "subjects_requested": args.subjects,
        "subjects_used": [r["subject"] for r in per_subject],
        "n_subjects_used": len(per_subject),
        "split_by": "repetition",
        "train_reps": sorted(train_reps),
        "test_reps": sorted(test_reps),
        "excluded_gestures": sorted(excluded),
        "fast": args.fast,
        "cache_tf": args.cache_tf,
        "max_clips_per_subj": args.max_clips_per_subj,
        "seed": args.seed,
        "pca_dim_requested": pca_dim,
        "mean_test_acc": mean_acc,
        "std_test_acc": std_acc,
        "aggregated_accuracy_pooled": global_acc,
    }

    per_path = out_dir / "per_subject_results.json"
    with open(per_path, "w", encoding="utf-8") as f:
        json.dump(per_subject, f, indent=2)

    sum_path = out_dir / "summary_mean_std.json"
    with open(sum_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    csv_path = out_dir / "summary_table.csv"
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("subject,test_acc,cv_best,n_train_clips,n_test_clips,dim_flat,dim_after_pca\n")
        for r in per_subject:
            f.write(f"{r['subject']},{r['test_acc']},{r['cv_best']},"
                    f"{r['n_train_clips']},{r['n_test_clips']},"
                    f"{r['X_dim_flat']},{r['X_dim_after_pca']}\n")
        f.write(f"MEAN,{mean_acc},,,,\n")
        f.write(f"STD,{std_acc},,,,\n")

    print("\n" + "=" * 100)
    print("DONE")
    print(f"Transform: {args.transform.upper()} | CLF: {args.clf.upper()}")
    print(f"Split: rep-based (train={sorted(train_reps)}, test={sorted(test_reps)})")
    print(f"Subjects used: {len(per_subject)}")
    print(f"Mean±std test acc: {mean_acc:.2f} ± {std_acc:.2f}")
    if global_acc is not None:
        print(f"Aggregated (pooled) acc: {global_acc:.1f}%")
    print(f"Saved: {per_path}")
    print(f"Saved: {sum_path}")
    print(f"Saved: {csv_path}")
    if args.save_cm:
        print(f"Confusion matrices: {cm_dir}")
    print("=" * 100)


if __name__ == "__main__":
    main()