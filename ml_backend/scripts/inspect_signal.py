"""Plot raw vs filtered EMG for one Gaddy utterance. Sanity-check for the preprocessing pipeline.

Usage:
    python scripts/inspect_signal.py --gaddy-root data/raw/gaddy/emg_data --mode voiced_parallel_data
"""

from __future__ import annotations

import argparse
from itertools import islice
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from silence_ml.data.gaddy import iter_gaddy
from silence_ml.preprocess.filters import bandpass, notch, zscore


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gaddy-root", type=Path, default=Path("data/raw/gaddy/emg_data"))
    ap.add_argument("--mode", default="voiced_parallel_data")
    ap.add_argument("--index", type=int, default=0, help="which utterance to plot")
    ap.add_argument("--out", type=Path, default=Path("runs/inspect_signal.png"))
    args = ap.parse_args()

    rec = next(islice(iter_gaddy(args.gaddy_root, mode=args.mode), args.index, args.index + 1), None)
    if rec is None:
        print("No utterance found at that index.")
        return 1

    print(f"utterance : {rec.source}")
    print(f"transcript: {rec.transcript!r}")
    print(f"shape     : {rec.signal.shape} @ {rec.sample_rate} Hz "
          f"({rec.signal.shape[1] / rec.sample_rate:.2f} s)")

    raw = rec.signal
    filt = bandpass(raw, rec.sample_rate)
    filt = notch(filt, rec.sample_rate, freq=60.0)
    normed = zscore(filt)

    t = np.arange(raw.shape[1]) / rec.sample_rate
    n_ch = raw.shape[0]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(n_ch, 3, figsize=(14, 2 * n_ch), sharex=True)
    for c in range(n_ch):
        axes[c, 0].plot(t, raw[c], lw=0.5)
        axes[c, 1].plot(t, filt[c], lw=0.5)
        axes[c, 2].plot(t, normed[c], lw=0.5)
        axes[c, 0].set_ylabel(f"ch{c}")
    axes[0, 0].set_title("raw")
    axes[0, 1].set_title("bandpass 20-500 Hz + 60 Hz notch")
    axes[0, 2].set_title("z-scored")
    axes[-1, 1].set_xlabel("time (s)")
    fig.suptitle(f"{rec.source}  —  {rec.transcript[:80] if rec.transcript else ''}", fontsize=9)
    fig.tight_layout()
    fig.savefig(args.out, dpi=120)
    print(f"saved -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
