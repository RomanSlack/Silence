from __future__ import annotations

import numpy as np
from scipy.signal import butter, iirnotch, sosfiltfilt, tf2sos


def bandpass(signal: np.ndarray, sample_rate: float, low: float = 20.0, high: float = 500.0,
             order: int = 4) -> np.ndarray:
    """Zero-phase Butterworth bandpass. signal shape (channels, samples)."""
    nyq = 0.5 * sample_rate
    high = min(high, nyq - 1.0)
    sos = butter(order, [low / nyq, high / nyq], btype="band", output="sos")
    return sosfiltfilt(sos, signal, axis=-1).astype(np.float32)


def notch(signal: np.ndarray, sample_rate: float, freq: float = 60.0, q: float = 30.0) -> np.ndarray:
    """Zero-phase notch to kill mains hum (60 Hz US, 50 Hz EU)."""
    b, a = iirnotch(freq, q, sample_rate)
    sos = tf2sos(b, a)
    return sosfiltfilt(sos, signal, axis=-1).astype(np.float32)


def zscore(signal: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Per-channel z-score across time."""
    mean = signal.mean(axis=-1, keepdims=True)
    std = signal.std(axis=-1, keepdims=True)
    return ((signal - mean) / (std + eps)).astype(np.float32)
