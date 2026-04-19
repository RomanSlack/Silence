"""End-to-end baseline: Gaddy vocalized recordings -> filters -> windows -> 1D CNN.

Usage:
    python scripts/train_baseline.py --gaddy-root data/raw/gaddy --mode voiced_parallel
"""

from __future__ import annotations

import argparse
from pathlib import Path

from silence_ml.data.base import EMGDataset, load_vocab
from silence_ml.data.gaddy import iter_gaddy, GADDY_N_CHANNELS
from silence_ml.preprocess import build_windows_from_recordings
from silence_ml.train import TrainConfig, train


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gaddy-root", type=Path, default=Path("data/raw/gaddy/emg_data"))
    ap.add_argument("--mode", default="voiced_parallel_data",
                    help="Gaddy subdir: voiced_parallel_data | silent_parallel_data | nonparallel_data | closed_vocab")
    ap.add_argument("--match", default="word_spotting", choices=["word_spotting", "exact"])
    ap.add_argument("--vocab", type=Path, default=Path("vocab/mvp_20words.txt"))
    ap.add_argument("--window-sec", type=float, default=1.0)
    ap.add_argument("--hop-sec", type=float, default=0.5)
    ap.add_argument("--epochs", type=int, default=30)
    args = ap.parse_args()

    vocab = load_vocab(args.vocab)
    print(f"vocab: {len(vocab)} words")

    recs = iter_gaddy(args.gaddy_root, mode=args.mode)
    windows, labels = build_windows_from_recordings(
        recs, vocab, window_sec=args.window_sec, hop_sec=args.hop_sec,
        match_mode=args.match,
    )
    print(f"windows: {len(windows)}  (from utterances whose transcript matched vocab)")
    if len(windows) == 0:
        print("No vocab matches in this Gaddy subset. Try --mode closed_vocab, "
              "or broaden the vocab to match Gaddy transcripts.")
        return 1

    ds = EMGDataset(windows, labels)
    cfg = TrainConfig(epochs=args.epochs)
    result = train(ds, n_channels=GADDY_N_CHANNELS, n_classes=len(vocab), cfg=cfg)
    print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
