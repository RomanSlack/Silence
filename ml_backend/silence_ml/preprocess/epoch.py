from __future__ import annotations

import re
from typing import Iterable

import numpy as np

from silence_ml.data.base import EMGRecording
from silence_ml.preprocess.filters import bandpass, notch, zscore


def window_signal(signal: np.ndarray, window_samples: int, hop_samples: int) -> np.ndarray:
    """Slice a (channels, samples) signal into (n_windows, channels, window_samples)."""
    _, n = signal.shape
    if n < window_samples:
        return np.empty((0, signal.shape[0], window_samples), dtype=np.float32)
    starts = range(0, n - window_samples + 1, hop_samples)
    return np.stack([signal[:, s:s + window_samples] for s in starts]).astype(np.float32)


_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


def build_windows_from_recordings(
    recs: Iterable[EMGRecording],
    vocab: list[str],
    window_sec: float = 1.0,
    hop_sec: float = 0.5,
    apply_bandpass: bool = True,
    apply_notch: bool = True,
    notch_freq: float = 60.0,
    match_mode: str = "word_spotting",
) -> tuple[list[np.ndarray], list[int]]:
    """Turn full-utterance recordings into fixed-length labeled windows.

    match_mode:
      "exact"         — keep utterance only if its transcript equals one vocab word.
      "word_spotting" — keep utterance if ANY vocab word appears in its transcript.
                        All windows from the utterance get that word's label.
                        This is a weak-label approximation — adequate for Gaddy
                        pipeline bring-up, replaced with forced-aligned windows
                        once Roman records his own isolated-word data.
    """
    vocab_lc = [w.lower() for w in vocab]
    vocab_set = set(vocab_lc)
    vocab_idx = {w: i for i, w in enumerate(vocab_lc)}

    windows: list[np.ndarray] = []
    labels: list[int] = []

    for rec in recs:
        if rec.transcript is None:
            continue
        tokens = _tokenize(rec.transcript)
        if not tokens:
            continue

        if match_mode == "exact":
            if len(tokens) != 1 or tokens[0] not in vocab_set:
                continue
            label = vocab_idx[tokens[0]]
        elif match_mode == "word_spotting":
            hits = [t for t in tokens if t in vocab_set]
            if not hits:
                continue
            label = vocab_idx[hits[0]]
        else:
            raise ValueError(f"unknown match_mode: {match_mode}")

        sig = rec.signal
        if apply_bandpass:
            sig = bandpass(sig, rec.sample_rate)
        if apply_notch:
            sig = notch(sig, rec.sample_rate, freq=notch_freq)
        sig = zscore(sig)

        win = int(round(window_sec * rec.sample_rate))
        hop = int(round(hop_sec * rec.sample_rate))
        w = window_signal(sig, win, hop)
        for i in range(w.shape[0]):
            windows.append(w[i])
            labels.append(label)

    return windows, labels
