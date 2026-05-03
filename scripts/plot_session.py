#!/usr/bin/env python3
"""
Plot filtered EMG (CH1 masseter, CH2 orbicularis) for every trial in a session.
One row per word, two subplots per row (CH1 left, CH2 right).

Usage:
    ../ml_backend/.venv/bin/python scripts/plot_session.py [session_dir]

If session_dir is omitted, uses the most recent session under recorder/data/sessions/.
Saves plot to <session_dir>/emg_overview.png and opens it.
"""

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import butter, sosfiltfilt

HERE = Path(__file__).resolve().parent
SESSIONS_DIR = HERE.parent / "recorder" / "data" / "sessions"

FS = 250.0
CH1, CH2 = 0, 1
CH_LABELS = ["CH1 masseter", "CH2 orbicularis"]
CH_COLORS = ["#ff4444", "#ff8800"]


def bandpass(signal: np.ndarray) -> np.ndarray:
    sos = butter(4, [20.0, 115.0], btype="band", fs=FS, output="sos")
    return sosfiltfilt(sos, signal)


def find_latest_session() -> Path:
    sessions = sorted(SESSIONS_DIR.iterdir(), key=lambda p: p.name)
    if not sessions:
        raise RuntimeError(f"No sessions found under {SESSIONS_DIR}")
    return sessions[-1]


def load_session(session_dir: Path):
    manifest = json.loads((session_dir / "manifest.json").read_text())
    trials = []
    for t in manifest["trials"]:
        npz = np.load(session_dir / t["path"])
        trials.append({"word": t["word"], "signal": npz["signal"]})
    return manifest, trials


def main():
    if len(sys.argv) > 1:
        session_dir = Path(sys.argv[1])
    else:
        session_dir = find_latest_session()

    print(f"Session: {session_dir.name}")
    manifest, trials = load_session(session_dir)

    words = [t["word"] for t in trials]
    n = len(trials)
    t_axis = np.arange(trials[0]["signal"].shape[1]) / FS

    fig, axes = plt.subplots(n, 2, figsize=(12, max(3, n * 1.6)),
                             sharex=True, sharey=False)
    if n == 1:
        axes = axes[np.newaxis, :]

    fig.suptitle(f"Session {session_dir.name} — EMG overview ({n} trials)",
                 fontsize=11, y=1.01)

    for row, trial in enumerate(trials):
        sig = trial["signal"]  # (8, n_samples)
        for col, ch in enumerate([CH1, CH2]):
            ax = axes[row, col]
            raw = sig[ch].astype(np.float64)
            raw -= raw.mean()
            filt = bandpass(raw)
            rms = float(np.sqrt(np.mean(filt ** 2)))
            ax.plot(t_axis, filt, color=CH_COLORS[col], linewidth=0.7)
            ax.axhline(0, color="#333", linewidth=0.4)
            ax.set_ylabel(trial["word"], fontsize=9, rotation=0,
                          labelpad=50, va="center")
            if row == 0:
                ax.set_title(f"{CH_LABELS[col]}\n(rms={rms:.0f} µV)",
                             fontsize=9, color=CH_COLORS[col])
            else:
                ax.set_title(f"rms={rms:.0f} µV", fontsize=8,
                             color="#666", pad=2)
            ax.tick_params(labelsize=7)
            ax.spines[["top", "right"]].set_visible(False)

    axes[-1, 0].set_xlabel("time (s)", fontsize=8)
    axes[-1, 1].set_xlabel("time (s)", fontsize=8)

    fig.tight_layout()
    out = session_dir / "emg_overview.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    print(f"Saved: {out}")

    import subprocess, os
    opener = "xdg-open"
    subprocess.Popen([opener, str(out)], stdout=subprocess.DEVNULL,
                     stderr=subprocess.DEVNULL)


if __name__ == "__main__":
    main()
