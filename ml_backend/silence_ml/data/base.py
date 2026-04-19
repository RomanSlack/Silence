from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass
class EMGRecording:
    """A single contiguous EMG recording with optional label/transcript.

    Shape convention: signal is (n_channels, n_samples), float32, volts or
    z-scored units. Sample rate is Hz. Transcript is the raw text if known;
    label is the integer class id for closed-vocab tasks (-1 if unlabeled).
    """

    signal: np.ndarray
    sample_rate: float
    transcript: str | None = None
    label: int = -1
    source: str = ""
    meta: dict | None = None

    def __post_init__(self) -> None:
        if self.signal.ndim != 2:
            raise ValueError(f"signal must be (channels, samples); got {self.signal.shape}")
        if self.signal.dtype != np.float32:
            self.signal = self.signal.astype(np.float32)


class EMGSource(Protocol):
    """Anything that yields EMGRecording objects: a dataset file, a live board, a saved session."""

    def __iter__(self): ...
    def __len__(self) -> int: ...


class EMGDataset(Dataset):
    """Thin torch Dataset wrapping a list of pre-epoched windows.

    Each item is (signal_tensor, label). Signal is (channels, samples).
    """

    def __init__(self, windows: list[np.ndarray], labels: list[int]):
        assert len(windows) == len(labels)
        self.windows = windows
        self.labels = labels

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        x = torch.from_numpy(self.windows[idx])
        y = int(self.labels[idx])
        return x, y


def load_vocab(path: str | Path) -> list[str]:
    return [w.strip() for w in Path(path).read_text().splitlines() if w.strip()]
