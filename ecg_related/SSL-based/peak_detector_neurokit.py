"""
peak_detector_neurokit.py
ECG peak detector using NeuroKit2 for R-peak detection.
Compatible interface with original ECGPeakDetector.
"""
import numpy as np
import neurokit2 as nk
import warnings

warnings.filterwarnings('ignore')


class ECGPeakDetectorNeurokit:
    """ECG peak detector using NeuroKit2's neurokit method."""

    def __init__(self, sampling_rate=500.0, method='neurokit'):
        """
        Initialize the peak detector.

        Args:
            sampling_rate: Sampling frequency in Hz
            method: NeuroKit2 R-peak detection method
                    ('neurokit', 'pantompkins1985', 'kalidas2017', etc.)
        """
        self.sampling_rate = sampling_rate
        self.method = method

    def detect_r_peaks(self, signal, return_indices=True, auto_correct_inversion=True):
        """
        Detect R-peaks in ECG signal using NeuroKit2.

        Args:
            signal: 1D ECG signal
            return_indices: If True, return peak indices; if False, return binary mask
            auto_correct_inversion: Automatically correct inverted signals

        Returns:
            If return_indices=True: (peak_indices, properties_dict)
            If return_indices=False: (binary_mask, properties_dict)
        """
        signal = np.asarray(signal).flatten().astype(np.float64)

        if len(signal) < 10:
            if return_indices:
                return np.array([]), {}
            else:
                return np.zeros_like(signal, dtype=np.float32), {}

        try:
            cleaned = nk.ecg_clean(signal, sampling_rate=int(self.sampling_rate),
                                   method=self.method)
            _, info = nk.ecg_peaks(cleaned, sampling_rate=int(self.sampling_rate))
            peak_indices = info.get('ECG_R_Peaks', np.array([]))
        except Exception:
            peak_indices = np.array([])

        if return_indices:
            return np.asarray(peak_indices), {}
        else:
            mask = np.zeros(len(signal), dtype=np.float32)
            if len(peak_indices) > 0:
                valid = peak_indices[peak_indices < len(signal)]
                mask[valid.astype(int)] = 1.0
            return mask, {}


# Alias for compatibility
ECGPeakDetector = ECGPeakDetectorNeurokit
