"""
Wrist vs Chest ECG Comparison — Ablation-Aligned Edition
=========================================================
Compare wrist ECG against chest ECG (ground truth) using 10s sliding windows.

Produces:
  1. Original per-subject tables (HR, HRV)
  2. Ablation-aligned metrics split by condition:
       - Free Form:    REST windows (including Static gesture)
       - Gesture:      gesture windows (gesture_id 1-10)
     Metrics:
       HR  — MAE±STD, RMSE, L1<2, L1<5, L1<10%×ref
       HRV — RMSSD MAE±STD, SDNN MAE±STD

Usage:
    python compare_l1_2rep.py -d ./processed_500hz -o ./comparison_results
    python compare_l1_2rep.py -d ./processed_500hz -o ./results_pt -m pantompkins1985 --gt-method neurokit --test-reps-only
"""

import json
import numpy as np
import neurokit2 as nk
import os
import csv
from tqdm import tqdm
from dataclasses import dataclass
from typing import List, Tuple, Optional
from scipy import stats as scipy_stats
import warnings
warnings.filterwarnings('ignore')


# ============== Configuration ==============
SAMPLE_RATE = 500

# Physiological limits (ms)
RR_MIN_MS = 300
RR_MAX_MS = 2000

# Quality threshold
THRESHOLD_MULTIPLIER = 2.5

# Window settings
WINDOW_SIZE_SEC = 10
STRIDE_SEC = 1.0

# Label settings
STATIC_GESTURE_ID = 0
RATIO_THRESHOLD = 0.5

# HRV range-based tolerance
HRV_RANGE_TOLERANCE = 0.05

# ---------- Conditions (no more STEADY_STATE) ----------
CONDITIONS = ['FREE_FORM', 'GESTURE']
CONDITION_LABELS = {'FREE_FORM': 'Free Form', 'GESTURE': 'Gesture'}


@dataclass
class ActionPeriod:
    start_idx: int
    end_idx: int
    gesture_id: int
    gesture_name: str
    action_num: int


# ============== Helper Functions ==============

def load_events_json(json_path: str, data_fs: int = 500) -> Tuple[List[ActionPeriod], int]:
    with open(json_path, 'r') as f:
        data = json.load(f)
    events = data['events']
    original_fs = data['metadata']['sampling_rates']['exg']
    downsample_factor = original_fs // data_fs

    rep_offset = 0
    for event in events:
        if event['event_type'] == 'REPETITION_START':
            rep_offset = event['sample_indices']['exg_idx']
            break

    action_periods = []
    current = None
    gesture_current = None

    for event in events:
        etype = event['event_type']
        exg_abs = event['sample_indices']['exg_idx']
        exg_rel = exg_abs - rep_offset
        idx = exg_rel // downsample_factor

        if etype == 'ACTION_START':
            current = {
                'start': idx,
                'gesture_id': event['data']['gesture_id'],
                'gesture_name': event['data']['gesture_name'],
                'action_num': event['data']['action_num']
            }
        elif etype == 'ACTION_END' and current:
            action_periods.append(ActionPeriod(
                start_idx=current['start'], end_idx=idx,
                gesture_id=current['gesture_id'],
                gesture_name=current['gesture_name'],
                action_num=current['action_num']
            ))
            current = None
        elif etype == 'GESTURE_START':
            gid = event.get('data', {}).get('gesture_id', -1)
            gesture_current = {
                'start': idx,
                'gesture_id': gid,
                'gesture_name': event.get('data', {}).get('gesture_name', 'Static'),
                'action_num': 0
            }
        elif etype == 'GESTURE_END' and gesture_current:
            action_periods.append(ActionPeriod(
                start_idx=gesture_current['start'], end_idx=idx,
                gesture_id=gesture_current['gesture_id'],
                gesture_name=gesture_current['gesture_name'],
                action_num=gesture_current['action_num']
            ))
            gesture_current = None

    return action_periods, rep_offset


def detect_rpeaks(ecg: np.ndarray, fs: int, method: str) -> Optional[np.ndarray]:
    try:
        ecg_clean = nk.ecg_clean(ecg.astype(np.float64), sampling_rate=fs, method='neurokit')
        _, info = nk.ecg_peaks(ecg_clean, sampling_rate=fs, method=method)
        rpeaks = info.get('ECG_R_Peaks', np.array([]))
        return rpeaks if len(rpeaks) >= 2 else None
    except:
        return None


def analyze_rr_quality(rr_ms: np.ndarray, threshold_mult: float = 2.5) -> np.ndarray:
    n = len(rr_ms)
    is_valid = np.ones(n, dtype=bool)
    mask_reasonable = (rr_ms >= RR_MIN_MS) & (rr_ms <= RR_MAX_MS)
    global_median = np.median(rr_ms[mask_reasonable]) if np.sum(mask_reasonable) > 0 else np.median(rr_ms)

    valid_rr_history = []
    for i in range(n):
        rr = rr_ms[i]
        ref_rr = np.mean(valid_rr_history) if len(valid_rr_history) > 0 else global_median
        threshold_ms = threshold_mult * ref_rr
        if rr < RR_MIN_MS or rr > RR_MAX_MS or rr > threshold_ms:
            is_valid[i] = False
        else:
            valid_rr_history.append(rr)
    return is_valid


def calculate_hrv_metrics(rr_ms: np.ndarray) -> dict:
    if len(rr_ms) < 2:
        return {k: np.nan for k in ['mean_hr_bpm', 'sdnn_ms', 'rmssd_ms']}
    hr_bpm = 60000.0 / rr_ms
    rr_diff = np.diff(rr_ms)
    return {
        'mean_hr_bpm': np.mean(hr_bpm),
        'sdnn_ms': np.std(rr_ms, ddof=1),
        'rmssd_ms': np.sqrt(np.mean(rr_diff ** 2)) if len(rr_diff) > 0 else np.nan,
    }


# ============== Window Generation ==============

def build_windows(signal_length: int, fs: int, periods: Optional[List[ActionPeriod]],
                  window_sec: float, stride_sec: float) -> List[dict]:
    window_samples = int(window_sec * fs)
    stride_samples = int(stride_sec * fs)

    boundaries = [0]
    if periods is not None:
        for p in periods:
            if p.gesture_id != STATIC_GESTURE_ID:
                boundaries.append(p.start_idx)
                boundaries.append(p.end_idx)
    boundaries.append(signal_length)
    boundaries = sorted(set(boundaries))

    def get_gesture_info(sample_idx):
        if periods is None:
            return 'REST', -1, 'REST'
        for p in periods:
            if p.start_idx <= sample_idx < p.end_idx:
                if p.gesture_id == STATIC_GESTURE_ID:
                    return 'REST', 0, 'Static'
                else:
                    return 'GESTURE', p.gesture_id, p.gesture_name
        return 'REST', -1, 'REST'

    windows = []
    for i in range(len(boundaries) - 1):
        seg_start = boundaries[i]
        seg_end = boundaries[i + 1]
        seg_duration = seg_end - seg_start
        if seg_duration < window_samples:
            continue
        mid = (seg_start + seg_end) // 2
        label, gesture_id, gesture_name = get_gesture_info(mid)
        n_win = (seg_duration - window_samples) // stride_samples + 1
        for w in range(n_win):
            w_start = seg_start + w * stride_samples
            w_end = w_start + window_samples
            windows.append({
                'start': w_start, 'end': w_end,
                'label': label, 'gesture_id': gesture_id, 'gesture_name': gesture_name
            })
    return windows


def compute_window_metrics(rpeaks: np.ndarray, is_valid: np.ndarray, rr_ms: np.ndarray,
                           window: dict) -> dict:
    w_start, w_end = window['start'], window['end']
    win_rr_ms = []
    n_invalid = 0

    if rpeaks is not None and len(rpeaks) >= 2:
        for j in range(len(rr_ms)):
            if rpeaks[j] >= w_start and rpeaks[j + 1] <= w_end:
                win_rr_ms.append(rr_ms[j])
                if not is_valid[j]:
                    n_invalid += 1

    win_rr_ms = np.array(win_rr_ms)
    n_rr = len(win_rr_ms)
    is_valid_window = (n_invalid == 0 and n_rr >= 2)
    hrv = calculate_hrv_metrics(win_rr_ms) if n_rr >= 2 else {
        k: np.nan for k in ['mean_hr_bpm', 'sdnn_ms', 'rmssd_ms']}

    return {
        'n_rr': n_rr, 'n_invalid': n_invalid, 'is_valid': is_valid_window,
        'rr_ms': win_rr_ms, **hrv
    }


# ============== Main Processing ==============

def process_subject_rep(ecg_chest: np.ndarray, ecg_wrist: np.ndarray, fs: int,
                        method: str, gt_method: str,
                        periods: Optional[List[ActionPeriod]]) -> Optional[dict]:
    rpeaks_chest = detect_rpeaks(ecg_chest, fs, gt_method)
    rpeaks_wrist = detect_rpeaks(ecg_wrist, fs, method)

    if rpeaks_chest is None:
        return None

    rr_ms_chest = np.diff(rpeaks_chest) / fs * 1000
    valid_chest = analyze_rr_quality(rr_ms_chest, THRESHOLD_MULTIPLIER)

    if rpeaks_wrist is not None and len(rpeaks_wrist) >= 2:
        rr_ms_wrist = np.diff(rpeaks_wrist) / fs * 1000
        valid_wrist = analyze_rr_quality(rr_ms_wrist, THRESHOLD_MULTIPLIER)
        has_wrist = True
    else:
        rr_ms_wrist = np.array([])
        valid_wrist = np.array([], dtype=bool)
        rpeaks_wrist = np.array([])
        has_wrist = False

    signal_length = min(len(ecg_chest), len(ecg_wrist))
    windows = build_windows(signal_length, fs, periods, WINDOW_SIZE_SEC, STRIDE_SEC)

    window_pairs = []
    for w in windows:
        chest_m = compute_window_metrics(rpeaks_chest, valid_chest, rr_ms_chest, w)
        wrist_m = compute_window_metrics(rpeaks_wrist, valid_wrist, rr_ms_wrist, w) if has_wrist else {
            'n_rr': 0, 'n_invalid': 0, 'is_valid': False, 'rr_ms': np.array([]),
            'mean_hr_bpm': np.nan, 'sdnn_ms': np.nan, 'rmssd_ms': np.nan
        }
        window_pairs.append({
            'window': w,
            'chest': chest_m,
            'wrist': wrist_m,
            'comparable': chest_m['is_valid'] and wrist_m['n_rr'] >= 2
        })

    return {
        'window_pairs': window_pairs,
        'n_chest_peaks': len(rpeaks_chest),
        'n_wrist_peaks': len(rpeaks_wrist) if has_wrist else 0
    }


# ============== Condition Classification ==============

def classify_condition(window: dict) -> str:
    if window['label'] == 'GESTURE':
        return 'GESTURE'
    return 'FREE_FORM'


# ============== Ablation-Aligned Accuracy (Table 2 metrics) ==============

def compute_ablation_accuracy(preds, trues):
    """Compute Table-2 metrics: HR (MAE,STD,RMSE,L1<2,L1<5,L1<10%) + HRV (RMSSD,SDNN)."""
    if len(preds) == 0:
        return None

    hr_pred, sdnn_pred, rmssd_pred = preds.T
    hr_true, sdnn_true, rmssd_true = trues.T

    # HR metrics
    hr_err = hr_pred - hr_true          # signed error
    hr_l1 = np.abs(hr_err)

    hr_r = np.corrcoef(hr_true, hr_pred)[0, 1] if np.std(hr_true) > 0 and np.std(hr_pred) > 0 else np.nan

    # SDNN / RMSSD signed errors
    sdnn_err = sdnn_pred - sdnn_true
    rmssd_err = rmssd_pred - rmssd_true

    def _corr(t, p):
        if len(t) < 3:
            return np.nan
        return np.corrcoef(t, p)[0, 1] if np.std(t) > 0 and np.std(p) > 0 else np.nan

    return {
        # HR
        'hr_mae':      float(np.mean(hr_l1)),
        'hr_std':      float(np.std(hr_err, ddof=1)),
        'hr_rmse':     float(np.sqrt(np.mean(hr_err ** 2))),
        'hr_r':        float(hr_r) if not np.isnan(hr_r) else np.nan,
        'hr_bias':     float(np.mean(hr_err)),
        'hr_l1_lt2':   float(np.mean(hr_l1 < 2.0)),
        'hr_l1_lt5':   float(np.mean(hr_l1 < 5.0)),
        'hr_l1_lt10p': float(np.mean(hr_l1 < 0.10 * np.abs(hr_true))),
        # SDNN
        'sdnn_mae':  float(np.mean(np.abs(sdnn_err))),
        'sdnn_std':  float(np.std(sdnn_err, ddof=1)),
        'sdnn_rmse': float(np.sqrt(np.mean(sdnn_err ** 2))),
        'sdnn_r':    float(_corr(sdnn_true, sdnn_pred)),
        # RMSSD
        'rmssd_mae':  float(np.mean(np.abs(rmssd_err))),
        'rmssd_std':  float(np.std(rmssd_err, ddof=1)),
        'rmssd_rmse': float(np.sqrt(np.mean(rmssd_err ** 2))),
        'rmssd_r':    float(_corr(rmssd_true, rmssd_pred)),
        # count
        'n_samples': len(hr_true),
    }


# ============== Per-Subject Aggregation ==============

def aggregate_subject(all_rep_results: List[dict]) -> dict:
    hr_chest, hr_wrist = [], []
    sdnn_chest, sdnn_wrist = [], []
    rmssd_chest, rmssd_wrist = [], []

    for rep_result in all_rep_results:
        for wp in rep_result['window_pairs']:
            if not wp['comparable']:
                continue
            c, w = wp['chest'], wp['wrist']
            if not np.isnan(c['mean_hr_bpm']) and not np.isnan(w['mean_hr_bpm']):
                hr_chest.append(c['mean_hr_bpm']); hr_wrist.append(w['mean_hr_bpm'])
            if not np.isnan(c['sdnn_ms']) and not np.isnan(w['sdnn_ms']):
                sdnn_chest.append(c['sdnn_ms']); sdnn_wrist.append(w['sdnn_ms'])
            if not np.isnan(c['rmssd_ms']) and not np.isnan(w['rmssd_ms']):
                rmssd_chest.append(c['rmssd_ms']); rmssd_wrist.append(w['rmssd_ms'])

    hr_chest, hr_wrist = np.array(hr_chest), np.array(hr_wrist)
    if len(hr_chest) >= 3:
        hr_err = hr_wrist - hr_chest
        hr_metrics = {
            'mae': np.mean(np.abs(hr_err)),
            'rmse': np.sqrt(np.mean(hr_err ** 2)),
            'n_windows': len(hr_chest)
        }
    else:
        hr_metrics = {'mae': np.nan, 'rmse': np.nan, 'n_windows': len(hr_chest)}

    def compute_comparison(chest_vals, wrist_vals):
        chest_vals, wrist_vals = np.array(chest_vals), np.array(wrist_vals)
        if len(chest_vals) >= 3:
            err = wrist_vals - chest_vals
            return {'mae': np.mean(np.abs(err)), 'n': len(chest_vals)}
        return {'mae': np.nan, 'n': len(chest_vals)}

    return {
        'hr': hr_metrics,
        'sdnn': compute_comparison(sdnn_chest, sdnn_wrist),
        'rmssd': compute_comparison(rmssd_chest, rmssd_wrist),
    }


# ============== Output Functions ==============

def save_hr_table(results: dict, output_path: str):
    subjects = sorted(results.keys())
    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Subject', 'N_Windows', 'MAE (bpm)', 'RMSE (bpm)'])
        all_mae, all_rmse = [], []
        for s in subjects:
            hr = results[s]['hr']
            writer.writerow([
                s, hr['n_windows'],
                f"{hr['mae']:.2f}" if not np.isnan(hr['mae']) else '-',
                f"{hr['rmse']:.2f}" if not np.isnan(hr['rmse']) else '-',
            ])
            if not np.isnan(hr['mae']):
                all_mae.append(hr['mae']); all_rmse.append(hr['rmse'])
        writer.writerow([
            'Mean ± SD', '',
            f"{np.mean(all_mae):.2f} ± {np.std(all_mae):.2f}",
            f"{np.mean(all_rmse):.2f} ± {np.std(all_rmse):.2f}",
        ])


def save_hrv_table(results: dict, output_path: str):
    subjects = sorted(results.keys())
    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Subject', 'SDNN MAE (ms)', 'RMSSD MAE (ms)'])
        all_sm, all_rm = [], []
        for s in subjects:
            sdnn = results[s]['sdnn']; rmssd = results[s]['rmssd']
            def fmt(v): return f"{v:.2f}" if not np.isnan(v) else '-'
            writer.writerow([s, fmt(sdnn['mae']), fmt(rmssd['mae'])])
            if not np.isnan(sdnn['mae']): all_sm.append(sdnn['mae'])
            if not np.isnan(rmssd['mae']): all_rm.append(rmssd['mae'])
        def fmt_ms(vals): return f"{np.mean(vals):.2f} ± {np.std(vals):.2f}" if vals else '-'
        writer.writerow(['Mean ± SD', fmt_ms(all_sm), fmt_ms(all_rm)])


def save_ablation_aligned_csv(condition_results: dict, output_path: str, method_label: str = 'NeuroKit'):
    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f, delimiter=';')
        header = ['method']
        for ds in CONDITIONS:
            header.extend([
                f'{ds}_hr_mae', f'{ds}_hr_std', f'{ds}_hr_rmse',
                f'{ds}_hr_l1_lt2', f'{ds}_hr_l1_lt5', f'{ds}_hr_l1_lt10p',
                f'{ds}_rmssd_mae', f'{ds}_rmssd_std',
                f'{ds}_sdnn_mae', f'{ds}_sdnn_std',
                f'{ds}_n_samples',
            ])
        writer.writerow(header)

        row = [f'{method_label} (signal processing baseline)']
        for ds in CONDITIONS:
            if ds in condition_results and condition_results[ds] is not None:
                r = condition_results[ds]
                row.extend([
                    r['hr_mae'], r['hr_std'], r['hr_rmse'],
                    r['hr_l1_lt2'], r['hr_l1_lt5'], r['hr_l1_lt10p'],
                    r['rmssd_mae'], r['rmssd_std'],
                    r['sdnn_mae'], r['sdnn_std'],
                    r['n_samples'],
                ])
            else:
                row.extend([0] * 11)
        writer.writerow(row)


def print_ablation_summary(condition_results: dict, method: str, gt_method: str,
                           reps_label: str = "all"):
    print("\n" + "=" * 120)
    print(f"TABLE 2 — SIGNAL PROCESSING BASELINE")
    print(f"Wrist method: {method}  |  Chest GT method: {gt_method}  |  Reps: {reps_label}")
    print("=" * 120)

    print(f"\n{'':_<120}")
    print(f"{'':40}{'Heart Rate Metrics':^40}{'HRV Metrics':^30}")
    print(f"{'Method':<14} {'Dataset':<12} "
          f"{'MAE':>7} {'STD':>7} {'RMSE':>7} {'L1<2':>7} {'L1<5':>7} {'L1<10%':>8} "
          f"{'RMSSD':>8} {'SDNN':>8} {'N':>6}")
    print("-" * 120)

    method_label = METHOD_NAMES.get(method, method)
    for dk in CONDITIONS:
        if dk in condition_results and condition_results[dk] is not None:
            r = condition_results[dk]
            print(f"{method_label:<14} {CONDITION_LABELS[dk]:<12} "
                  f"{r['hr_mae']:>7.2f} {r['hr_std']:>7.2f} {r['hr_rmse']:>7.2f} "
                  f"{r['hr_l1_lt2']*100:>6.1f}% {r['hr_l1_lt5']*100:>6.1f}% "
                  f"{r['hr_l1_lt10p']*100:>7.1f}% "
                  f"{r['rmssd_mae']:>8.2f} {r['sdnn_mae']:>8.2f} "
                  f"{r['n_samples']:>6}")
        else:
            print(f"{method_label:<14} {CONDITION_LABELS[dk]:<12}  (no data)")
    print("-" * 120)


def save_window_detail_csv(all_window_data: dict, output_path: str):
    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'subject', 'rep', 'window_start', 'window_end',
            'label', 'gesture_id', 'gesture_name', 'condition', 'comparable',
            'chest_n_rr', 'chest_valid', 'chest_hr', 'chest_sdnn', 'chest_rmssd',
            'wrist_n_rr', 'wrist_valid', 'wrist_hr', 'wrist_sdnn', 'wrist_rmssd',
            'hr_error', 'sdnn_error', 'rmssd_error'
        ])

        for subject in sorted(all_window_data.keys()):
            for rep, rep_result in all_window_data[subject]:
                for wp in rep_result['window_pairs']:
                    w = wp['window']
                    c, wr = wp['chest'], wp['wrist']
                    condition = classify_condition(w)

                    hr_err = (wr['mean_hr_bpm'] - c['mean_hr_bpm']) if wp['comparable'] else np.nan
                    sdnn_err = (wr['sdnn_ms'] - c['sdnn_ms']) if wp['comparable'] else np.nan
                    rmssd_err = (wr['rmssd_ms'] - c['rmssd_ms']) if wp['comparable'] else np.nan

                    def fmt(v): return f"{v:.4f}" if not np.isnan(v) else ''

                    writer.writerow([
                        subject, rep, w['start'], w['end'],
                        w['label'], w['gesture_id'], w['gesture_name'], condition,
                        1 if wp['comparable'] else 0,
                        c['n_rr'], 1 if c['is_valid'] else 0,
                        fmt(c['mean_hr_bpm']), fmt(c['sdnn_ms']), fmt(c['rmssd_ms']),
                        wr['n_rr'], 1 if wr['is_valid'] else 0,
                        fmt(wr['mean_hr_bpm']), fmt(wr['sdnn_ms']), fmt(wr['rmssd_ms']),
                        fmt(hr_err), fmt(sdnn_err), fmt(rmssd_err)
                    ])



# ============== Single-Method Processing ==============

def process_all_subjects(data_dir: str, subjects: list, method: str, gt_method: str,
                         test_reps_only: bool = False) -> Tuple[dict, dict, dict]:
    test_reps = set(range(8, 11))
    all_results = {}
    all_window_data = {}
    condition_pool = {c: {'preds': [], 'trues': []} for c in CONDITIONS}

    for subject in tqdm(subjects, desc=f"  [{method}]"):
        reps_dir = os.path.join(data_dir, subject, 'repetitions')
        events_dir = os.path.join(data_dir, subject, 'events')

        if not os.path.exists(reps_dir):
            continue

        has_events = os.path.exists(events_dir)
        rep_results = []
        all_window_data[subject] = []

        for rep in range(1, 11):
            data_path = os.path.join(reps_dir, f'rep_{rep:02d}_data.npz')
            events_path = os.path.join(events_dir, f'rep_{rep:02d}_events.json') if has_events else None

            if not os.path.exists(data_path):
                continue

            data = np.load(data_path, allow_pickle=True)
            fs = int(data.get('fs_exg', 500))
            exg = data['exg']
            ecg_chest = exg[:, 0]
            ecg_wrist = exg[:, 1]

            periods = None
            if events_path and os.path.exists(events_path):
                try:
                    periods, _ = load_events_json(events_path, fs)
                except:
                    periods = None

            result = process_subject_rep(ecg_chest, ecg_wrist, fs, method, gt_method, periods)
            if result is not None:
                rep_results.append(result)
                all_window_data[subject].append((rep, result))

                if test_reps_only and rep not in test_reps:
                    continue

                for wp in result['window_pairs']:
                    if not wp['comparable']:
                        continue
                    c, wr = wp['chest'], wp['wrist']
                    chest_vals = [c['mean_hr_bpm'], c['sdnn_ms'], c['rmssd_ms']]
                    wrist_vals = [wr['mean_hr_bpm'], wr['sdnn_ms'], wr['rmssd_ms']]
                    if any(np.isnan(v) for v in chest_vals + wrist_vals):
                        continue
                    condition = classify_condition(wp['window'])
                    condition_pool[condition]['trues'].append(chest_vals)
                    condition_pool[condition]['preds'].append(wrist_vals)

        if rep_results:
            all_results[subject] = aggregate_subject(rep_results)

    return all_results, all_window_data, condition_pool


# ============== Multi-Method Aggregation ==============

ALL_METHODS = ['neurokit', 'pantompkins1985', 'kalidas2017']
METHOD_NAMES = {
    'pantompkins1985': 'Pan-Tompkins',
    'kalidas2017': 'Kalidas',
    'neurokit': 'NeuroKit'
}

AGG_KEYS = [
    'hr_mae', 'hr_std', 'hr_rmse', 'hr_r', 'hr_bias',
    'hr_l1_lt2', 'hr_l1_lt5', 'hr_l1_lt10p',
    'sdnn_mae', 'sdnn_std', 'sdnn_rmse', 'sdnn_r',
    'rmssd_mae', 'rmssd_std', 'rmssd_rmse', 'rmssd_r',
    'n_samples',
]


def aggregate_across_methods(per_method_condition_results: dict) -> dict:
    methods = list(per_method_condition_results.keys())
    agg = {}
    for cond in CONDITIONS:
        cond_agg = {}
        for key in AGG_KEYS:
            vals = []
            for m in methods:
                r = per_method_condition_results[m].get(cond)
                if r is not None and key in r:
                    v = r[key]
                    if not np.isnan(v):
                        vals.append(v)
            if vals:
                cond_agg[key] = {'mean': float(np.mean(vals)), 'std': float(np.std(vals)), 'values': vals}
            else:
                cond_agg[key] = {'mean': np.nan, 'std': np.nan, 'values': []}
        agg[cond] = cond_agg
    return agg


def print_multi_method_summary(agg: dict, per_method_results: dict,
                               gt_method: str, reps_label: str = "all"):
    methods = list(per_method_results.keys())

    print("\n" + "=" * 130)
    print(f"MULTI-METHOD TABLE 2 — SIGNAL PROCESSING BASELINE")
    print(f"Wrist methods: {', '.join(methods)}  |  Chest GT: {gt_method}  |  Reps: {reps_label}")
    print("=" * 130)

    # Per-method
    for m in methods:
        print(f"\n  [{METHOD_NAMES.get(m, m)}]")
        for dk in CONDITIONS:
            r = per_method_results[m].get(dk)
            if r is not None:
                print(f"    {CONDITION_LABELS[dk]:<12}  HR MAE={r['hr_mae']:.2f}  "
                      f"RMSSD={r['rmssd_mae']:.2f}  SDNN={r['sdnn_mae']:.2f}  N={r['n_samples']}")
            else:
                print(f"    {CONDITION_LABELS[dk]:<12}  (no data)")

    # Aggregated Table 2
    print(f"\n{'':=<130}")
    print(f"AGGREGATED (mean ± std across {len(methods)} methods)")
    print(f"{'':=<130}")

    print(f"\n{'':_<130}")
    print(f"{'':40}{'Heart Rate Metrics':^42}{'HRV Metrics':^24}")
    print(f"{'Method':<14} {'Dataset':<12} "
          f"{'MAE':>9} {'STD':>9} {'RMSE':>9} {'L1<2':>9} {'L1<5':>9} {'L1<10%':>9} "
          f"{'RMSSD':>10} {'SDNN':>10} {'N':>6}")
    print("-" * 130)

    for dk in CONDITIONS:
        a = agg[dk]
        def fmt(key):
            v = a[key]
            if np.isnan(v['mean']): return '--'
            return f"{v['mean']:.2f}±{v['std']:.2f}"
        def fmt_pct(key):
            v = a[key]
            if np.isnan(v['mean']): return '--'
            return f"{v['mean']*100:.1f}±{v['std']*100:.1f}%"
        n_str = f"{a['n_samples']['mean']:.0f}" if not np.isnan(a['n_samples']['mean']) else '--'
        print(f"{'Mean±STD':<14} {CONDITION_LABELS[dk]:<12} "
              f"{fmt('hr_mae'):>9} {fmt('hr_std'):>9} {fmt('hr_rmse'):>9} "
              f"{fmt_pct('hr_l1_lt2'):>9} {fmt_pct('hr_l1_lt5'):>9} {fmt_pct('hr_l1_lt10p'):>9} "
              f"{fmt('rmssd_mae'):>10} {fmt('sdnn_mae'):>10} {n_str:>6}")
    print("-" * 130)


def save_multi_method_csv(agg: dict, per_method_results: dict, output_path: str):
    methods = list(per_method_results.keys())
    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        header = ['method', 'condition',
                  'hr_mae', 'hr_std', 'hr_rmse', 'hr_r', 'hr_bias',
                  'hr_l1_lt2', 'hr_l1_lt5', 'hr_l1_lt10p',
                  'sdnn_mae', 'sdnn_r',
                  'rmssd_mae', 'rmssd_r',
                  'n_samples']
        writer.writerow(header)

        for m in methods:
            for dk in CONDITIONS:
                r = per_method_results[m].get(dk)
                if r is not None:
                    writer.writerow([
                        METHOD_NAMES.get(m, m), dk,
                        r['hr_mae'], r['hr_std'], r['hr_rmse'], r['hr_r'], r['hr_bias'],
                        r['hr_l1_lt2'], r['hr_l1_lt5'], r['hr_l1_lt10p'],
                        r['sdnn_mae'], r['sdnn_r'],
                        r['rmssd_mae'], r['rmssd_r'],
                        r['n_samples']
                    ])

        for dk in CONDITIONS:
            a = agg[dk]
            def gm(k): return a[k]['mean']
            def gs(k): return a[k]['std']
            writer.writerow([
                f'Mean±STD ({len(methods)} methods)', dk,
                f"{gm('hr_mae'):.2f}±{gs('hr_mae'):.2f}",
                f"{gm('hr_std'):.2f}±{gs('hr_std'):.2f}",
                f"{gm('hr_rmse'):.2f}±{gs('hr_rmse'):.2f}",
                f"{gm('hr_r'):.3f}±{gs('hr_r'):.3f}",
                f"{gm('hr_bias'):.2f}±{gs('hr_bias'):.2f}",
                f"{gm('hr_l1_lt2')*100:.1f}±{gs('hr_l1_lt2')*100:.1f}%",
                f"{gm('hr_l1_lt5')*100:.1f}±{gs('hr_l1_lt5')*100:.1f}%",
                f"{gm('hr_l1_lt10p')*100:.1f}±{gs('hr_l1_lt10p')*100:.1f}%",
                f"{gm('sdnn_mae'):.2f}±{gs('sdnn_mae'):.2f}",
                f"{gm('sdnn_r'):.3f}±{gs('sdnn_r'):.3f}",
                f"{gm('rmssd_mae'):.2f}±{gs('rmssd_mae'):.2f}",
                f"{gm('rmssd_r'):.3f}±{gs('rmssd_r'):.3f}",
                f"{gm('n_samples'):.0f}",
            ])



# ============== Main ==============

def main():
    import argparse

    parser = argparse.ArgumentParser(description='Wrist vs Chest ECG Comparison (Ablation-Aligned)')
    parser.add_argument('-d', '--data-dir', default='./processed_500hz',
                        help='Data directory (500Hz)')
    parser.add_argument('-o', '--output', default='./comparison_results',
                        help='Output directory')
    parser.add_argument('-m', '--method', default='neurokit',
                        choices=['pantompkins1985', 'kalidas2017', 'neurokit'],
                        help='R-peak detection method for WRIST ECG (default: neurokit)')
    parser.add_argument('--gt-method', default=None,
                        choices=['pantompkins1985', 'kalidas2017', 'neurokit'],
                        help='R-peak detection method for CHEST ECG ground truth (default: same as -m)')
    parser.add_argument('-t', '--threshold', type=float, default=2.5,
                        help='RR quality threshold multiplier (default: 2.5)')
    parser.add_argument('-w', '--window-size', type=float, default=10.0,
                        help='Window size in seconds (default: 10)')
    parser.add_argument('-s', '--stride', type=float, default=1.0,
                        help='Stride in seconds (default: 1)')
    parser.add_argument('--test-reps-only', action='store_true',
                        help='Only use reps 8-10 for ablation-aligned metrics (match ablation split)')
    parser.add_argument('--excluded-subjects', type=str, default='S06,S09',
                        help='Comma-separated subjects to exclude (default: S06,S09)')
    parser.add_argument('--all-methods', action='store_true',
                        help='Run all 3 R-peak methods and report MAE±STD across methods')
    args = parser.parse_args()

    if args.gt_method is None:
        args.gt_method = args.method

    global THRESHOLD_MULTIPLIER, WINDOW_SIZE_SEC, STRIDE_SEC
    THRESHOLD_MULTIPLIER = args.threshold
    WINDOW_SIZE_SEC = args.window_size
    STRIDE_SEC = args.stride

    excluded = set(args.excluded_subjects.split(',')) if args.excluded_subjects else set()

    subjects = sorted([
        d for d in os.listdir(args.data_dir)
        if os.path.isdir(os.path.join(args.data_dir, d)) and d.startswith('S')
        and d not in excluded
    ])

    if not subjects:
        print(f"No subjects found in {args.data_dir}")
        return

    print("=" * 70)
    print("Wrist vs Chest ECG Comparison (Table 2 format)")
    print("=" * 70)
    print(f"Data directory:  {args.data_dir}")
    print(f"Wrist method:    {args.method}")
    print(f"Chest GT method: {args.gt_method}")
    print(f"Window:          {WINDOW_SIZE_SEC}s, Stride: {STRIDE_SEC}s")
    print(f"RR threshold:    {THRESHOLD_MULTIPLIER}x")
    print(f"Subjects:        {len(subjects)}")
    print(f"Excluded:        {sorted(excluded) if excluded else 'none'}")
    print(f"Conditions:      Free Form, Gesture (no Steady State)")
    print(f"Ablation pool:   {'reps 8-10 only' if args.test_reps_only else 'all reps'}")
    print("=" * 70)

    os.makedirs(args.output, exist_ok=True)

    # ================================================================
    # ALL-METHODS MODE
    # ================================================================
    if args.all_methods:
        print(f"\n>>> ALL-METHODS MODE: running {', '.join(ALL_METHODS)}")
        print(f">>> GT method (chest): {args.gt_method}\n")

        per_method_condition_results = {}

        for wrist_method in ALL_METHODS:
            print(f"\n--- Running wrist method: {wrist_method} ---")
            all_results_m, all_window_data_m, condition_pool_m = process_all_subjects(
                args.data_dir, subjects, wrist_method, args.gt_method,
                test_reps_only=args.test_reps_only
            )

            cond_results_m = {}
            for cond in CONDITIONS:
                pool = condition_pool_m[cond]
                if len(pool['preds']) > 0:
                    preds = np.array(pool['preds'])
                    trues = np.array(pool['trues'])
                    cond_results_m[cond] = compute_ablation_accuracy(preds, trues)
                else:
                    cond_results_m[cond] = None

            per_method_condition_results[wrist_method] = cond_results_m

            method_label = METHOD_NAMES.get(wrist_method, wrist_method)
            method_dir = os.path.join(args.output, wrist_method)
            os.makedirs(method_dir, exist_ok=True)

            if all_results_m:
                save_hr_table(all_results_m, os.path.join(method_dir, 'hr_accuracy.csv'))
                save_hrv_table(all_results_m, os.path.join(method_dir, 'hrv_accuracy.csv'))
                save_window_detail_csv(all_window_data_m, os.path.join(method_dir, 'window_comparison_detail.csv'))
                save_ablation_aligned_csv(cond_results_m, os.path.join(method_dir, 'ablation_aligned_baseline.csv'),
                                          method_label=method_label)

            print_ablation_summary(cond_results_m, wrist_method, args.gt_method,
                                   reps_label='8-10 only' if args.test_reps_only else 'all')

        # Aggregate
        agg = aggregate_across_methods(per_method_condition_results)
        print_multi_method_summary(agg, per_method_condition_results, args.gt_method,
                                   reps_label='8-10 only' if args.test_reps_only else 'all')

        save_multi_method_csv(agg, per_method_condition_results,
                              os.path.join(args.output, 'multi_method_aggregated.csv'))

        print(f"\nOutput saved to: {args.output}/")
        return

    # ================================================================
    # SINGLE-METHOD MODE
    # ================================================================
    all_results, all_window_data, condition_pool = process_all_subjects(
        args.data_dir, subjects, args.method, args.gt_method,
        test_reps_only=args.test_reps_only
    )

    if not all_results:
        print("No results to report.")
        return

    condition_results = {}
    for cond in CONDITIONS:
        pool = condition_pool[cond]
        if len(pool['preds']) > 0:
            preds = np.array(pool['preds'])
            trues = np.array(pool['trues'])
            condition_results[cond] = compute_ablation_accuracy(preds, trues)
        else:
            condition_results[cond] = None

    method_label = METHOD_NAMES.get(args.method, args.method)

    save_hr_table(all_results, os.path.join(args.output, 'hr_accuracy.csv'))
    save_hrv_table(all_results, os.path.join(args.output, 'hrv_accuracy.csv'))
    save_window_detail_csv(all_window_data, os.path.join(args.output, 'window_comparison_detail.csv'))
    save_ablation_aligned_csv(condition_results, os.path.join(args.output, 'ablation_aligned_baseline.csv'),
                              method_label=method_label)

    # Print summary
    print("\n" + "=" * 70)
    print(f"HR ACCURACY (Wrist [{args.method}] vs Chest [{args.gt_method}]) — Per Subject")
    print("=" * 70)
    print(f"{'Subject':<8} {'N':>6} {'MAE':>10} {'RMSE':>10}")
    print("-" * 40)
    all_mae = []
    for s in sorted(all_results.keys()):
        hr = all_results[s]['hr']
        print(f"{s:<8} {hr['n_windows']:>6} {hr['mae']:>10.2f} {hr['rmse']:>10.2f}")
        all_mae.append(hr['mae'])
    print("-" * 40)
    print(f"{'Mean±SD':<8} {'':>6} {np.mean(all_mae):>7.2f}±{np.std(all_mae):<5.2f}")

    print_ablation_summary(condition_results, args.method, args.gt_method,
                           reps_label='8-10 only' if args.test_reps_only else 'all')

    print(f"\nWindow counts by condition:")
    for cond in CONDITIONS:
        n = len(condition_pool[cond]['preds'])
        print(f"  {cond:<15}: {n} comparable windows")

    print(f"\nOutput saved to: {args.output}/")


if __name__ == '__main__':
    main()