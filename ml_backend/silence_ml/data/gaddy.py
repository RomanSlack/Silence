"""Loader for the Gaddy & Klein 2020 silent-speech EMG dataset.

Dataset: https://zenodo.org/records/4064409  (DOI 10.5281/zenodo.4064408)

Extracted layout (after `tar -xf emg_data.tar.gz`):
    emg_data/
      nonparallel_data/{session}/{i}_emg.npy
      silent_parallel_data/{session}/{i}_emg.npy
      voiced_parallel_data/{session}/{i}_emg.npy
      closed_vocab/{session}/{i}_emg.npy                  (may or may not exist)

Per utterance i:
    {i}_emg.npy           raw EMG, shape (n_samples, 8) at 1000 Hz
    {i}_audio.flac        paired audio
    {i}_audio_clean.flac  denoised audio
    {i}_info.json         {"text": "...", "book": "...", "sentence_index": ...}
    {i}_button.npy        button state (trial boundaries)

Samples with `sentence_index == -1` are reference/calibration EMG, not real
utterances — we skip them.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Iterator

import numpy as np

from silence_ml.data.base import EMGRecording

GADDY_SAMPLE_RATE = 1000.0  # Hz
GADDY_N_CHANNELS = 8

MODES = ("voiced_parallel_data", "silent_parallel_data", "nonparallel_data", "closed_vocab")

_INFO_RE = re.compile(r"(\d+)_info\.json$")


def _iter_utterance_ids(session_dir: Path) -> Iterator[int]:
    ids: list[int] = []
    for fname in session_dir.iterdir():
        m = _INFO_RE.match(fname.name)
        if m:
            ids.append(int(m.group(1)))
    yield from sorted(ids)


def iter_gaddy(root: str | Path, mode: str | None = None,
               include_silent: bool = True) -> Iterator[EMGRecording]:
    """Yield every utterance under `root`.

    `root` is the directory containing the mode subdirs (e.g. `data/raw/gaddy/emg_data`).
    `mode`, if given, restricts to one of MODES. If None, all modes are yielded.
    `include_silent` lets you drop silent_parallel_data for Model-1 (overt) training.
    """
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(f"Gaddy root does not exist: {root}")

    if mode is None:
        mode_dirs = [root / m for m in MODES if (root / m).exists()]
    else:
        mode_dirs = [root / mode]

    for mode_dir in mode_dirs:
        if not mode_dir.exists():
            continue
        if not include_silent and mode_dir.name == "silent_parallel_data":
            continue

        # closed_vocab has an extra {voiced,silent}/ level before the session dir.
        if mode_dir.name == "closed_vocab":
            session_dirs: list[Path] = []
            for sub in sorted(p for p in mode_dir.iterdir() if p.is_dir()):
                if not include_silent and sub.name == "silent":
                    continue
                session_dirs.extend(sorted(p for p in sub.iterdir() if p.is_dir()))
        else:
            session_dirs = sorted(p for p in mode_dir.iterdir() if p.is_dir())

        for session_dir in session_dirs:
            for i in _iter_utterance_ids(session_dir):
                info_path = session_dir / f"{i}_info.json"
                emg_path = session_dir / f"{i}_emg.npy"
                if not emg_path.exists():
                    continue
                try:
                    info = json.loads(info_path.read_text())
                except Exception:
                    continue
                if info.get("sentence_index", 0) == -1:
                    continue

                emg = np.load(emg_path)
                if emg.ndim != 2 or emg.shape[1] != GADDY_N_CHANNELS:
                    continue

                yield EMGRecording(
                    signal=emg.T.astype(np.float32),  # (8, T)
                    sample_rate=GADDY_SAMPLE_RATE,
                    transcript=info.get("text", "").strip() or None,
                    label=-1,
                    source=f"gaddy:{mode_dir.name}/{session_dir.name}/{i}",
                    meta={
                        "mode": mode_dir.name,
                        "session": session_dir.name,
                        "utterance_id": i,
                        **{k: v for k, v in info.items() if k != "text"},
                    },
                )


def count_utterances(root: str | Path, mode: str | None = None) -> int:
    return sum(1 for _ in iter_gaddy(root, mode=mode))
