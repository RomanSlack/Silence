"""Session = one contiguous recording run. Persists trials to disk as .npz and
a live manifest.json, so you can resume if you close the tab or the app crashes.
"""

from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path

import numpy as np


@dataclass
class Trial:
    word: str
    label: int
    trial_idx: int                 # per-word index (0, 1, 2, ...)
    path: str                      # relative to session dir
    duration_sec: float
    sample_rate: float
    n_channels: int
    timestamp: str


class Session:
    def __init__(self, root: Path, name: str, vocab: list[str],
                 sample_rate: float, n_channels: int):
        self.root = Path(root)
        self.name = name
        self.dir = self.root / name
        self.dir.mkdir(parents=True, exist_ok=True)
        self.vocab = list(vocab)
        self.sample_rate = float(sample_rate)
        self.n_channels = int(n_channels)
        self.trials: list[Trial] = []
        self._lock = threading.Lock()
        self._load_or_init()

    @property
    def manifest_path(self) -> Path:
        return self.dir / "manifest.json"

    def _load_or_init(self) -> None:
        if self.manifest_path.exists():
            data = json.loads(self.manifest_path.read_text())
            self.vocab = data.get("vocab", self.vocab)
            self.sample_rate = data.get("sample_rate", self.sample_rate)
            self.n_channels = data.get("n_channels", self.n_channels)
            self.trials = [Trial(**t) for t in data.get("trials", [])]
        else:
            self._save_manifest()

    def _save_manifest(self) -> None:
        payload = {
            "name": self.name,
            "created": datetime.utcnow().isoformat() + "Z",
            "vocab": self.vocab,
            "sample_rate": self.sample_rate,
            "n_channels": self.n_channels,
            "trials": [asdict(t) for t in self.trials],
        }
        tmp = self.manifest_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(payload, indent=2))
        tmp.replace(self.manifest_path)

    def counts(self) -> dict[str, int]:
        out = {w: 0 for w in self.vocab}
        for t in self.trials:
            out[t.word] = out.get(t.word, 0) + 1
        return out

    def total(self) -> int:
        return len(self.trials)

    def save_trial(self, word: str, signal: np.ndarray, duration_sec: float) -> Trial:
        if word not in self.vocab:
            raise ValueError(f"word not in vocab: {word!r}")
        label = self.vocab.index(word)
        with self._lock:
            trial_idx = sum(1 for t in self.trials if t.word == word)
            fname = f"{word}_{trial_idx:04d}.npz"
            path = self.dir / fname
            np.savez(
                path,
                signal=signal.astype(np.float32),
                sample_rate=np.float32(self.sample_rate),
                word=word,
                label=np.int32(label),
                trial_idx=np.int32(trial_idx),
                duration_sec=np.float32(duration_sec),
                timestamp=datetime.utcnow().isoformat() + "Z",
            )
            trial = Trial(
                word=word,
                label=label,
                trial_idx=trial_idx,
                path=fname,
                duration_sec=duration_sec,
                sample_rate=self.sample_rate,
                n_channels=int(signal.shape[0]),
                timestamp=datetime.utcnow().isoformat() + "Z",
            )
            self.trials.append(trial)
            self._save_manifest()
        return trial

    def undo_last(self) -> Trial | None:
        with self._lock:
            if not self.trials:
                return None
            trial = self.trials.pop()
            p = self.dir / trial.path
            if p.exists():
                p.unlink()
            self._save_manifest()
        return trial


def new_session_name() -> str:
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
