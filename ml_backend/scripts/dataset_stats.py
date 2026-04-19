"""Report Gaddy dataset stats + how much of our 20-word vocab it covers.

Run after extracting emg_data.tar.gz. Tells us before training whether
word-spotting against the MVP vocab will have enough matches to train on.
"""

from __future__ import annotations

import argparse
import re
from collections import Counter
from pathlib import Path

from silence_ml.data.base import load_vocab
from silence_ml.data.gaddy import iter_gaddy, MODES

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gaddy-root", type=Path, default=Path("data/raw/gaddy/emg_data"))
    ap.add_argument("--vocab", type=Path, default=Path("vocab/mvp_20words.txt"))
    args = ap.parse_args()

    vocab = [w.lower() for w in load_vocab(args.vocab)]
    vocab_set = set(vocab)
    print(f"vocab ({len(vocab)}): {vocab}")
    print()

    for mode in MODES:
        if not (args.gaddy_root / mode).exists():
            continue
        n_utts = 0
        n_matched_utts = 0
        token_counts = Counter()
        hit_counts = Counter()
        durations_sec = 0.0
        for rec in iter_gaddy(args.gaddy_root, mode=mode):
            n_utts += 1
            durations_sec += rec.signal.shape[1] / rec.sample_rate
            if rec.transcript:
                toks = _TOKEN_RE.findall(rec.transcript.lower())
                token_counts.update(toks)
                hits = [t for t in toks if t in vocab_set]
                if hits:
                    n_matched_utts += 1
                    hit_counts.update(hits)
        print(f"[{mode}]")
        print(f"  utterances : {n_utts}")
        print(f"  duration   : {durations_sec/60:.1f} min")
        print(f"  vocab-hit utterances: {n_matched_utts} "
              f"({100*n_matched_utts/max(1,n_utts):.1f}%)")
        print(f"  per-word hit counts:")
        for w in vocab:
            print(f"    {w:10s} {hit_counts.get(w, 0)}")
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
