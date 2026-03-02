# Wearable EMG Gesture Classifications

9-class hand gesture classification from 2-channel wrist surface EMG (sEMG). 
## Overview

Two forearm sEMG channels (flexor + extensor) are recorded at 2 kHz via BioAmp EXG Pill modules. Three classification pipelines are provided, all operating on time-frequency (TF) representations of the EMG signal:

1. **Traditional ML** — STFT / CWT / Log-Mel spectrograms flattened and classified with SVM, Random Forest, or LDA (with optional PCA).
2. **Mini-ResNet18** — A lightweight 2D ResNet with residual connections, operating on time-frequency representations.
3. **Vanilla 2D CNN** — A 4-layer ConvNet baseline without residual connections, on spectrograms.

## Repository Structure

```
emg_related/
├── DSP_based/
│   ├── lda_rf_svm_train.py      # SVM / RF / LDA on flattened TF features
│   ├── resnet_emg.py            # 2D Mini-ResNet on TF representations
│   └── train_2dcnn.py           # Vanilla 2D CNN on spectrograms
│
├── SSL_based/                   # (Planned)
│
└── README.md
```

---

## Gesture Set (9 Classes)

| ID | Gesture | Description |
|----|---------|-------------|
| 1 | Hand Close | Close all fingers into fist |
| 2 | Hand Open | Extend all fingers fully |
| 3 | Wrist Flexion | Bend wrist downward |
| 4 | Wrist Extension | Bend wrist upward |
| 5 | Index Finger Point | Extend index finger only |
| 7 | Little Finger Flexion | Flex little finger only |
| 8 | Tripod Grasp | Thumb + index + middle pinch |
| 9 | Thumb Flexion | Flex thumb toward palm |
| 10 | Middle Finger Flexion | Flex middle finger only |

Gesture 6 (Cut Something) is excluded by default. Gesture selection is partially aligned with the [Ninapro](http://ninapro.hevs.ch/) dataset.

---

## Data

### Windowed EMG Clips

All pipelines expect pre-windowed EMG clips organized as:

```
EMG_classification/windowed_data/
├── 10s/                        # Window duration
│   ├── data/
│   │   ├── S01/
│   │   │   ├── clip_001.npz    # Contains 'emg' (T, 2) and 'repetition' fields
│   │   │   └── ...
│   │   ├── S02/
│   │   └── ...
│   └── label/
│       ├── S01/
│       │   ├── clip_001.npz    # Contains 'label' (one-hot) or 'gesture_id'
│       │   └── ...
│       └── ...
├── 2s/
└── ...
```

### Evaluation Modes

| Mode | Split Strategy | Default |
|------|---------------|---------|
| **Intra-subject** | Repetition-based per subject (reps 1–8 train, 9–10 test) | Traditional ML, CNN |
| **Cross-subject** | Subject-based (S01–S10 train, S17–S20 test) | ResNet, CNN |

Repetition-based splitting ensures no data leakage between train and test sets.

---

## Time-Frequency Transforms

All three pipelines support the following TF representations as input options:
| Transform | Description | Key Parameters |
|-----------|-------------|----------------|
| **STFT** | Short-Time Fourier Transform | `n_fft=256`, `hop=64`, `fmax=450 Hz` |
| **CWT** | Continuous Wavelet Transform (Morlet) | `n_scales=128`, `f=[20–450] Hz` |
| **Log-Mel** | Log-Mel Spectrogram | `n_mels=128`, `f=[20–450] Hz` |

### Stack Features (Deep Learning Only)

Optional time-domain features can be appended along the frequency axis of the 2D spectrogram:

| Feature | Flag | Description |
|---------|------|-------------|
| RMS | `rms` | Root mean square envelope |
| MAV | `mav` | Mean absolute value |
| WL | `wl` | Waveform length |
| PSD | `psd` | Spectral power density |

Usage: `--stack rms,mav,wl`

---

## Pipeline 1: Traditional ML (SVM / RF / LDA)

> **Script:** `lda_rf_svm_train.py`

Computes TF representations offline, flattens to a feature vector, applies StandardScaler + optional PCA, and trains a classifier with grid search cross-validation.

### Usage

```bash
# SVM with STFT features and PCA
python lda_rf_svm_train.py \
    --data-root ./EMG_classification/windowed_data \
    --window 10s --subjects S01-S20 \
    --transform stft --clf svm --pca-dim 128

# LDA with Log-Mel features
python lda_rf_svm_train.py \
    --window 10s --subjects S01-S20 \
    --transform logmel --clf lda --pca-dim 128

# Random Forest with CWT, save confusion matrices
python lda_rf_svm_train.py \
    --window 10s --transform cwt --clf rf \
    --pca-dim 128 --cache-tf --save-cm
```

### Key Parameters

| Flag | Default | Description |
|------|---------|-------------|
| `--clf` | `svm` | Classifier (`svm`, `rf`, `lda`) |
| `--transform` | `stft` | TF transform (`stft`, `cwt`, `logmel`) |
| `--pca-dim` | `256` | PCA dimensions (0 to disable) |
| `--train-reps` | `1–8` | Repetitions for training |
| `--test-reps` | `9–10` | Repetitions for testing |
| `--fast` | off | Reduced grid search for faster iteration |
| `--cache-tf` | off | Cache TF representations to disk |
| `--save-cm` | off | Save per-subject confusion matrices |

### Outputs

| File | Description |
|------|-------------|
| `per_subject_results.json` | Per-subject accuracy, best params, per-gesture breakdown |
| `summary_mean_std.json` | Mean ± STD accuracy across all subjects |
| `summary_table.csv` | Tabular per-subject results |
| `confusion_matrix_all_subjects.png` | Aggregated confusion matrix (counts + normalized) |
| `predictions_all_subjects.npz` | Raw predictions for offline re-analysis |

---

## Pipeline 2: Mini-ResNet

> **Script:** `resnet_emg.py`

A lightweight ResNet with 4 stages × 1 BasicBlock each (channels: 10→32→64→128), operating on 2D time-frequency representations:

```
Input [B, 2, T]
  → TF Transform (STFT / CWT / LogMel)  [on GPU]
  → [Optional: Stack Features (RMS, MAV, ...)]
  → Conv2d(2, 10, k=3) → BN → ReLU → MaxPool
  → BasicBlock2d × 4 stages (10→32→64→128)
  → AdaptiveAvgPool2d → LayerNorm → FC(128→64) → FC(64→9)
```

### Usage

```bash
# CWT + stacked features (cross-subject)
python resnet_emg.py --window 10s --transform cwt --stack rms,mav

# Intra-subject
python resnet_emg.py --window 2s --eval-mode intra

# All window sizes (cross-subject)
python resnet_emg.py --window all --eval-mode cross

# Intra-subject with K-fold CV and augmentation
python resnet_emg.py --window 10s --eval-mode intra \
    --n-folds 5 --augment --warmup-epochs 5
```

### Key Parameters

| Flag | Default | Description |
|------|---------|-------------|
| `--transform` | `stft` | TF transform (`stft`, `cwt`, `logmel`) |
| `--stack` | `none` | Stack features (`rms,mav,wl,psd`) |
| `--eval-mode` | `cross` | Evaluation mode (`cross`, `intra`) |
| `--epochs` | `80` | Training epochs |
| `--batch-size` | `32` | Batch size |
| `--lr` | `0.001` | Learning rate |
| `--augment` | off | Data augmentation + mixup |
| `--n-folds` | `1` | K-fold CV for intra-subject (1 = single split) |
| `--scheduler` | `cosine` | LR scheduler (`cosine`, `plateau`) |

---

## Pipeline 3: Vanilla 2D CNN

> **Script:** `train_2dcnn.py`

A simple 4-layer ConvNet baseline without residual connections:

```
Input [B, 2, T]
  → TF Transform (STFT / CWT / LogMel)  [on GPU]
  → [Optional: Stack Features]
  → Conv2d(2, 32, 3) → BN → ReLU → MaxPool
  → Conv2d(32, 64, 3) → BN → ReLU → MaxPool
  → Conv2d(64, 128, 3) → BN → ReLU → MaxPool
  → Conv2d(128, 256, 3) → BN → ReLU → MaxPool
  → AdaptiveAvgPool2d(1) → FC(256→128) → ReLU → Dropout → FC(128→9)
```

### Usage

```bash
# CWT, intra-subject
python train_2dcnn.py --window 10s --transform cwt --eval-mode intra

# STFT + stacked features, cross-subject
python train_2dcnn.py --window 10s --transform stft --stack rms,mav,wl

# Intra-subject with 5-fold CV and augmentation
python train_2dcnn.py --window 10s --eval-mode intra --n-folds 5 --augment
```

---

## Installation

```bash
pip install torch torchvision torchaudio
pip install numpy scipy scikit-learn matplotlib seaborn tqdm

# Optional (for CWT / Log-Mel transforms in traditional ML pipeline)
pip install pywavelets librosa
```

