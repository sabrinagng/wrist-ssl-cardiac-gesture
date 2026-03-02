# Wrist-Worn Biosignal Platform: Simultaneous Cardiac Monitoring & Gesture Recognition

A wrist-worn wearable platform that  captures **ECG** and **EMG** signals for two parallel tasks:

1. **Cardiac Monitoring** — Heart rate (HR) and heart rate variability (HRV) estimation from wrist ECG, comparing classical DSP and self-supervised learning (SSL) approaches against chest ECG ground truth.
2. **Gesture Recognition** — 9-class hand gesture classification from 2-channel wrist EMG using traditional ML (SVM, RF, LDA) and deep learning (ResNet, 2D CNN) on time-frequency representations.

Data was collected from 20 subjects performing 10 hand gestures  over 10 repetitions (~80 min per session), with simultaneous chest and wrist ECG, and dual-channel forearm EMG.

## Repository Structure

```
wrist-ssl-cardiac-gesture/
│
├── ecg_related/                  # Cardiac monitoring pipelines
│   ├── DSP_based/
│   │   └── compare_l1_2rep.py        # Wrist vs chest ECG: R-peak detection → HR/HRV comparison
│   ├── SSL_based/
│   │   ├── model_overlap_peak.py     # 1-D MAE with peak-adaptive masking + overlap
│   │   ├── model_overlap_no_peak.py  # 1-D MAE with random masking + overlap
│   │   ├── peak_detector_neurokit.py # NeuroKit2 R-peak detector wrapper
│   │   ├── preprocessor.py           # Dataset classes (PTB-XL, wECG)
│   │   ├── train_ablation_peak.py    # SSL pretraining (2×2 ablation)
│   │   └── ablation_hrv_regression.py# Downstream HR/HRV regression
│   └── README.md
│
├── emg_related/                  # Gesture recognition pipelines
│   ├── DSP_based/
│   │   ├── lda_rf_svm_train.py       # SVM / RF / LDA on TF features (STFT, CWT, LogMel)
│   │   ├── resnet_emg.py             # Mini-ResNet (1D raw + 2D TF modes)
│   │   └── train_2dcnn.py            # Vanilla 2D CNN on spectrograms
│   └── README.md
│
├── firmware/                     # ESP32-S3 acquisition firmware
│   ├── main/
│   │   └── exg_integrated_sys.cpp    # Dual-core ADC + USB streaming
│   ├── sensors/
│   │   ├── sensors.cpp               # MPU6050 + MAX32664 drivers
│   │   └── sensors.hpp
│   └── README.md
│
├── stimulus/                     # Data collection stimulus program
│   ├── stimulus.py                   # Tkinter GUI with serial DAQ
│   ├── images/                       # Gesture prompt images
│   └── README.md
│
└── .gitignore
```

## Hardware

The wearable platform is built on an **Arduino Nano ESP32 (ESP32-S3)** with:

| Sensor | Signal | Rate | Interface |
|--------|--------|------|-----------|
| BioAmp EXG Pill × 2 | ECG (chest + wrist) | 2 kHz | ADC |
| BioAmp EXG Pill × 2 | sEMG (flexor + extensor) | 2 kHz | ADC |


All streams are transmitted over USB Serial JTAG using a custom binary frame protocol. See [`firmware/README.md`](firmware/README.md) for details.

## Data Collection Protocol

The stimulus program guides participants through **10 repetitions** of **10 hand gestures** + 1 static rest posture. Each gesture is performed twice per trial with a brief rest in between.

| Parameter | Value |
|-----------|-------|
| Subjects | 20 (S01–S20, excluding S06/S09) |
| Gestures | 9 dynamic + 1 static rest |
| Repetitions | 10 per session |
| Sampling | ECG/EMG @ 2 kHz, IMU @ 1 kHz, PPG @ 200 Hz |
| Session duration | ~80 min |

See [`stimulus/README.md`](stimulus/README.md) for the full experimental protocol.

## ECG: Cardiac Monitoring

### DSP Baseline

Classical R-peak detection (NeuroKit2, Pan-Tompkins, Kalidas) on wrist ECG with 10-second sliding windows. Computes HR, SDNN, and RMSSD and compares against chest ECG ground truth.

```bash
python ecg_related/DSP_based/compare_l1_2rep.py \
    -d ./processed_500hz -o ./results --all-methods --test-reps-only
```

### SSL Pipeline

A **1-D Masked Autoencoder (MAE)** pretrained on 12-lead ECG via self-supervised reconstruction, then fine-tuned for HR/HRV regression. Includes a **2×2 ablation** over:

- **Peak-adaptive masking** — preferentially keeps R-peak patches, forcing reconstruction of non-peak morphology
- **Overlapping patch embeddings** — 50% overlap with fold-based averaging to reduce boundary artifacts

| Config | Peak Mask | Overlap |
|--------|:---------:|:-------:|
| `pm0_po0` | — | — |
| `pm0_po1` | — | Yes |
| `pm1_po0` | Yes | — |
| `pm1_po1` | Yes | Yes |

```bash
# Pretrain (4 ablation configs)
python ecg_related/SSL_based/train_ablation_peak.py --wecg_root ./wECG_dataset_npy/

# Downstream HRV regression
python ecg_related/SSL_based/ablation_hrv_regression.py --ablation_dir ablation_results
```

See [`ecg_related/README.md`](ecg_related/README.md) for architecture details and evaluation metrics.

## EMG: Gesture Recognition

### Traditional ML

Time-frequency representations (STFT, CWT, Log-Mel) flattened and classified with **SVM**, **Random Forest**, or **LDA** (with optional PCA). Intra-subject, repetition-based train/test split to prevent data leakage.

```bash
python emg_related/DSP_based/lda_rf_svm_train.py \
    --window 16s --transform stft --clf svm --pca-dim 256
```

### Deep Learning

- **Mini-ResNet** — 1D (raw time-series) and 2D (spectrogram) modes with optional stacked features (RMS, MAV, WL, PSD)
- **Vanilla 2D CNN** — 4-layer ConvNet on spectrograms with optional multi-fold CV and data augmentation

Both support cross-subject and intra-subject evaluation modes.

```bash
python emg_related/DSP_based/resnet_emg.py --window 16s --mode 2d --transform cwt --stack rms,mav
python emg_related/DSP_based/train_2dcnn.py --window 16s --transform cwt --eval-mode intra
```

See [`emg_related/README.md`](emg_related/README.md) for more details.

## Installation

```bash
git clone https://github.com/sabrinagng/wrist-ssl-cardiac-gesture.git
cd wrist-ssl-cardiac-gesture

# Core dependencies
pip install torch torchvision torchaudio
pip install neurokit2 wfdb numpy scipy scikit-learn tqdm matplotlib seaborn

# Optional (for EMG CWT/LogMel transforms)
pip install pywavelets librosa
```

**Tested with:** Python 3.13.9, PyTorch 2.11.0, CUDA 12.8

For firmware build instructions, see [`firmware/README.md`](firmware/README.md).
