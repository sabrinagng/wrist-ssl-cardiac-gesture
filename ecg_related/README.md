# Wearable ECG Analysis: DSP vs SSL Approaches

Comparing **traditional digital signal processing (DSP)** and **self-supervised learning (SSL)** approaches for heart rate (HR) and heart rate variability (HRV) estimation from low-channel wearable ECG. Part of a multimodal wearable platform study combining EMG and ECG for simultaneous gesture recognition and cardiovascular monitoring.

## Overview

Wrist-worn ECG offers convenient continuous monitoring but suffers from lower signal quality compared to chest-worn devices. This repository provides two complementary pipelines to estimate cardiac parameters (HR, SDNN, RMSSD) from 2-channel wearable ECG and evaluates them against chest ECG ground truth:

1. **DSP-Based Pipeline** — Classical R-peak detection (NeuroKit2, Pan-Tompkins, Kalidas) applied directly to wrist ECG, with windowed HR/HRV comparison against chest ECG ground truth.
2. **SSL-Based Pipeline** — A 1-D Masked Autoencoder (MAE) pretrained on 12-lead ECG via self-supervised reconstruction, then fine-tuned for HR/HRV regression. Includes a 2×2 ablation over peak-adaptive masking and overlapping patch embeddings.


## Repository Structure

```
├── DSP_based/
│   └── compare_l1_2rep.py          # Wrist vs chest ECG comparison pipeline
│
├── SSL_based/
│   ├── model_overlap_peak.py       # MAE with overlap + peak-adaptive masking
│   ├── model_overlap_no_peak.py    # MAE with overlap + random masking only
│   ├── peak_detector_neurokit.py   # NeuroKit2-based R-peak detector wrapper
│   ├── preprocessor.py             # Dataset classes (PTB-XL, wECG .npy)
│   ├── train_ablation_peak.py      # Pretraining script (4 ablation configs)
│   └── ablation_hrv_regression.py  # Downstream HRV regression evaluation
│
└── README.md
```

---

## Installation

```bash
git clone https://github.com/sabrinagng/wearable-ecg-dsp-ssl.git
cd wearable-ecg-dsp-ssl

pip install torch torchvision torchaudio
pip install neurokit2 wfdb numpy scipy scikit-learn tqdm
```

**Tested with:** Python 3.10+, PyTorch 2.x, CUDA 11.8+

---

## Data

### Custom Wearable ECG Dataset

Data was collected from 20 subjects (S01–S20, excluding S06 and S09 due to signal quality issues) performing 10 repetitions of 9 hand gestures plus a static rest posture. Each repetition contains simultaneous chest ECG (Modified Lead I) and wrist ECG, sampled at 500 Hz after downsampling from 2 kHz.

**Processed data layout** (for DSP pipeline):
```
processed_500hz/
├── S01/
│   ├── repetitions/
│   │   ├── rep_01_data.npz    # exg array 
│   │   └── ...
│   └── events/
│       ├── rep_01_events.json  # gesture timing annotations
│       └── ...
├── S02/
└── ...
```

### wECG Dataset (for SSL pretraining)

```
wECG_dataset_npy/
├── dataset_001_LA_V3.npy      # structured array with 'reference_12_lead' [T, 12] and 'wECG' [T, 2]
├── dataset_001_LA_V5.npy
├── dataset_001_LA_A.npy
├── dataset_001_LA_A_self.npy
├── dataset_001_info.npy        # metadata (auto-skipped)
└── ...
```

---

## Part 1: DSP-Based Pipeline

> **Script:** `compare_l1_2rep.py`

Traditional signal processing approach: detect R-peaks independently on wrist and chest ECG using established algorithms, compute HR/HRV in sliding windows, and compare.

### Method

1. **R-peak detection** on both chest (ground truth) and wrist ECG using NeuroKit2 with selectable algorithms (NeuroKit, Pan-Tompkins, Kalidas).
2. **RR interval quality filtering** — adaptive threshold rejecting physiologically implausible intervals (< 300 ms or > 2000 ms).
3. **10-second sliding windows** (1 s stride) classified by condition:
   - **Free Form** — REST windows 
   - **Gesture** — windows during active gesture performance (gesture IDs 1–10)
4. **Per-window metrics**: HR (bpm), SDNN (ms), RMSSD (ms) compared between wrist and chest.


### Usage

**Single R-peak method:**
```bash
python compare_l1_2rep.py \
    -d ./processed_500hz \
    -o ./comparison_results \
    -m neurokit \
    --gt-method neurokit \
```

**All three methods with aggregation:**
```bash
python compare_l1_2rep.py \
    -d ./processed_500hz \
    -o ./comparison_results \
    --all-methods \
    --test-reps-only
```

**Using only test repetitions (reps 8–10, matching SSL split):**
```bash
python compare_l1_2rep.py \
    -d ./processed_500hz \
    -o ./results_test_split \
    -m neurokit \
    --test-reps-only
```

### Outputs

| File | Description |
|------|-------------|
| `hr_accuracy.csv` | Per-subject HR: MAE, RMSE, r, Bias |
| `hrv_accuracy.csv` | Per-subject HRV: SDNN/RMSSD MAE and r |
| `ablation_aligned_baseline.csv` | Condition-split metrics (Free Form / Gesture) |
| `window_comparison_detail.csv` | Every window with chest/wrist metrics and errors |
| `latex_tables.tex` | Publication-ready LaTeX tables |
| `multi_method_aggregated.csv` | Mean ± STD across all 3 R-peak methods (if `--all-methods`) |

### Key Parameters

| Flag | Default | Description |
|------|---------|-------------|
| `-m` | `neurokit` | Wrist R-peak method (`neurokit`, `pantompkins1985`, `kalidas2017`) |
| `--gt-method` | same as `-m` | Chest ground truth R-peak method |
| `--tolerance` | `50` | R-peak matching tolerance (ms) |
| `-w` | `10.0` | Window size (seconds) |
| `-s` | `1.0` | Stride (seconds) |
| `--excluded-subjects` | `S06,S09` | Subjects to exclude |
| `--test-reps-only` | off | Only use reps 8–10 for metrics (matches SSL test split) |

---

## Part 2: SSL-Based Pipeline

Self-supervised pretraining via a 1-D Masked Autoencoder, followed by downstream HR/HRV regression with partial encoder fine-tuning.

### 2.1 MAE Architecture

```
Input [B, 2, 1024]
    │
    ├── InstanceNorm1d (per-channel)
    ├── Conv1d patch embedding (kernel=64, stride=32 or 64)
    │       └── expand → GELU → compress → BatchNorm → GELU
    │
    ├── Sinusoidal 1-D positional encoding
    ├── Masking (55% ratio): peak-adaptive OR random
    │
    ├── Transformer Encoder (6 blocks, dim=128, 8 heads)
    │
    ├── Linear → Decoder (4 blocks, dim=64, 4 heads)
    ├── Prediction head → reconstructed patches [B, N, C×P]
    │
    └── L1 reconstruction loss (masked regions only)
        └── F.fold overlap averaging for overlapping configs
```

### 2.2 Ablation: Peak Masking × Patch Overlap

| Config | Peak Mask | Overlap | File | Description |
|--------|:---------:|:-------:|------|-------------|
| `pm0_po0` | ✗ | ✗ | `model_overlap_no_peak.py` | Baseline: random mask, non-overlapping |
| `pm0_po1` | ✗ | ✓ | `model_overlap_no_peak.py` | Random mask + 50% overlap |
| `pm1_po0` | ✓ | ✗ | `model_overlap_peak.py` | Peak-adaptive mask, non-overlapping |
| `pm1_po1` | ✓ | ✓ | `model_overlap_peak.py` | Full: peak-adaptive + overlap |

**Peak-adaptive masking**: R-peaks are detected via NeuroKit2 during training. Per-patch weights modulate random noise so that peak-containing patches are preferentially *kept*, forcing the model to reconstruct non-peak morphology from cardiac landmark context. At inference, standard random masking is used (or no masking for `encode()`).

**Overlapping patches**: 50% overlap (stride = patch_size / 2) with `F.fold`-based averaging during reconstruction to avoid boundary artifacts.

### 2.3 Pretraining

Pretrain all four configs on 12-lead reference ECG with random 2-channel selection per sample:

```bash
python train_ablation_peak.py \
    --wecg_root /path/to/wECG_dataset_npy/ \
    --window_size 1024 \
    --patch_size 64 \
    --mask_ratio 0.55 \
    --epochs 30 \
    --batch_size 64 \
    --lr 1e-4 \
    --out_dir ablation_results \
    --retrain_all
```

**Output**: `pm0_po0_best.pt`, `pm0_po1_best.pt`, `pm1_po0_best.pt`, `pm1_po1_best.pt` + `ablation_12lead_retrain.json`.

| Hyperparameter | Value | Notes |
|----------------|-------|-------|
| `window_size` | 1024 | ~2 s at 500 Hz |
| `patch_size` | 64 | 128 ms per patch |
| `mask_ratio` | 0.55 | 55% masked |
| `overlap_ratio` | 0.5 | stride = 32 when enabled |
| `embed_dim` | 128 | Encoder dimension |
| `depth` / `decoder_depth` | 6 / 4 | Transformer blocks |
| Train/test split | 90/10 | Fixed seed 42 |

### 2.4 Downstream HRV Regression

> **Script:** `ablation_hrv_regression.py`

Evaluates frozen (or partially fine-tuned) encoder representations on four downstream tasks:

| Task | Source | Split | Input |
|------|--------|-------|-------|
| **wECG** | wECG dataset | Temporal 70/30 per recording | 2-ch wrist ECG |
| **Free Form** | Custom dataset | Rep 1–7 train / 8–10 test per subject | 1-ch wrist ECG (duplicated to 2-ch) |
| **Steady State** | Custom dataset | Rep 1–7 train / 8–10 test per subject | 1-ch wrist ECG (duplicated to 2-ch) |
| **Gesture** | Custom dataset | Rep 1–7 train / 8–10 test per subject | 1-ch wrist ECG (duplicated to 2-ch) |

**Regression head architecture**: Dual Conv1d blocks with residual connections → avg + max pooling → 3-layer MLP. Targets are z-score normalised per subject during training.

**Fine-tuning strategy**: Partial — first 4 encoder blocks frozen, last 2 blocks + LayerNorm unfrozen. Separate learning rates (encoder 1e-4, head 5e-4).

```bash
python ablation_hrv_regression.py \
    --ablation_dir ablation_results \
    --epochs 30 \
    --batch_size 32 \
    --device cuda
```

**Run specific configs only:**
```bash
python ablation_hrv_regression.py \
    --ablation_dir ablation_results \
    --only pm1_po0,pm1_po1
```

**Use only first N encoder blocks (frozen):**
```bash
python ablation_hrv_regression.py \
    --ablation_dir ablation_results \
    --n_encoder_blocks 4
```

### Evaluation Metrics

**HR metrics**: MAE ± STD (bpm), RMSE, L1 < 2 bpm, L1 < 5 bpm, L1 < 10% × reference

**HRV metrics**: SDNN/RMSSD — MAE (ms) and range-based accuracy (|pred − true| ≤ 5% of pooled GT range)

---

## Comparison: DSP vs SSL

Both pipelines are evaluated on the same test split (repetitions 8–10) and the same conditions (Free Form, Gesture) to enable direct comparison. Use `--test-reps-only` in the DSP pipeline to align with the SSL evaluation protocol.

```bash
# DSP baseline (aligned to SSL test split)
python compare_l1_2rep.py -d ./processed_500hz -o ./results_aligned --all-methods --test-reps-only

# SSL downstream
python ablation_hrv_regression.py --ablation_dir ablation_results
```

Results from both pipelines are exported as CSV files suitable for generating unified comparison tables.

---


