"""End-to-end smoke test on synthetic EMG. Runs with no dataset required.

Validates that filters -> epoching -> EMGDataset -> 1D CNN -> train loop all
wire together and the model can fit easy synthetic class structure above chance.
If this passes, real-data bugs are isolated to data loading / labels.
"""

from __future__ import annotations

import numpy as np
import torch

from silence_ml.data.base import EMGDataset, EMGRecording
from silence_ml.data.gaddy import GADDY_N_CHANNELS, GADDY_SAMPLE_RATE
from silence_ml.preprocess import build_windows_from_recordings
from silence_ml.train import TrainConfig, train


def synth_utterance(label_word: str, n_channels: int, sample_rate: float,
                    duration_sec: float, rng: np.random.Generator) -> EMGRecording:
    """Fake EMG where each label has a distinct per-channel carrier frequency."""
    label_seed = sum(ord(c) for c in label_word)
    rng2 = np.random.default_rng(label_seed)
    freqs = rng2.uniform(30, 200, size=n_channels)
    t = np.arange(int(duration_sec * sample_rate)) / sample_rate
    signal = np.stack([
        np.sin(2 * np.pi * f * t) * (0.5 + 0.5 * rng.normal())
        + rng.normal(0, 0.3, size=t.shape)
        for f in freqs
    ]).astype(np.float32)
    return EMGRecording(
        signal=signal,
        sample_rate=sample_rate,
        transcript=label_word,
        label=-1,
        source=f"synth:{label_word}",
    )


def main() -> int:
    vocab = ["yes", "no", "stop", "go", "left", "right"]
    rng = np.random.default_rng(0)
    recs = [synth_utterance(w, GADDY_N_CHANNELS, GADDY_SAMPLE_RATE,
                            duration_sec=2.0, rng=rng)
            for w in vocab for _ in range(20)]

    windows, labels = build_windows_from_recordings(
        recs, vocab, window_sec=1.0, hop_sec=0.5, match_mode="exact"
    )
    assert len(windows) > 0, "no windows built"
    print(f"synthetic dataset: {len(windows)} windows, {len(set(labels))} classes")

    ds = EMGDataset(windows, labels)
    cfg = TrainConfig(epochs=8, batch_size=32, val_split=0.2,
                      device="cuda" if torch.cuda.is_available() else "cpu")
    result = train(ds, n_channels=GADDY_N_CHANNELS, n_classes=len(vocab), cfg=cfg)
    print(result)
    assert result["best_val_acc"] > 1.5 / len(vocab), "model did not beat chance"
    print("SMOKE TEST OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
