#!/usr/bin/env python3
"""
Leave-one-out SVM classifier on a small set of words from a session.
Use this to sanity-check whether the EMG signal is discriminable
before investing in more recording.

Usage:
    ../ml_backend/.venv/bin/python scripts/quick_classify.py [session_dir] [word1 word2 ...]

Defaults to most recent session, words = first 5 in vocab with >= 3 trials.
"""

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.signal import butter, sosfiltfilt
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

HERE = Path(__file__).resolve().parent
SESSIONS_DIR = HERE.parent / "recorder" / "data" / "sessions"
FS = 250.0


def bandpass(sig):
    sos = butter(4, [20.0, 115.0], btype="band", fs=FS, output="sos")
    return sosfiltfilt(sos, sig)


def extract_features(signal):
    """signal: (8, n_samples). Returns 1D feature vector."""
    feats = []
    for ch in [0, 1]:  # CH1 masseter, CH2 orbicularis
        raw = signal[ch].astype(np.float64)
        raw -= raw.mean()
        f = bandpass(raw)
        n = len(f)
        # split into 4 quarters, compute RMS per quarter (captures timing)
        quarters = np.array_split(f, 4)
        for q in quarters:
            feats.append(float(np.sqrt(np.mean(q**2))))
        # global features
        feats.append(float(np.sqrt(np.mean(f**2))))           # RMS
        feats.append(float(np.mean(np.abs(f))))               # MAV
        feats.append(float(np.sum(np.abs(np.diff(f)))))       # waveform length
        feats.append(float(np.sum(np.diff(np.sign(f)) != 0))) # zero crossings
    return np.array(feats, dtype=np.float32)


def main():
    args = sys.argv[1:]

    # parse session dir vs word list
    session_dir = None
    words_filter = []
    for a in args:
        p = Path(a)
        if p.exists() and p.is_dir():
            session_dir = p
        else:
            words_filter.append(a.lower())

    if session_dir is None:
        sessions = sorted(SESSIONS_DIR.iterdir(), key=lambda p: p.name)
        if not sessions:
            print("No sessions found."); return
        session_dir = sessions[-1]

    print(f"Session: {session_dir.name}")
    manifest = json.loads((session_dir / "manifest.json").read_text())

    by_word = defaultdict(list)
    for t in manifest["trials"]:
        sig = np.load(session_dir / t["path"])["signal"]
        by_word[t["word"]].append(sig)

    # pick words
    if words_filter:
        words = [w for w in words_filter if len(by_word.get(w, [])) >= 2]
    else:
        # auto: first words with >= 3 trials, up to 5
        words = [w for w in manifest["vocab"]
                 if len(by_word.get(w, [])) >= 3][:5]

    if len(words) < 2:
        print("Need at least 2 words with enough trials."); return

    print(f"Words: {words}")
    print(f"Trials per word: { {w: len(by_word[w]) for w in words} }\n")

    # build feature matrix
    X, y = [], []
    for label, word in enumerate(words):
        for sig in by_word[word]:
            X.append(extract_features(sig))
            y.append(label)
    X = np.array(X)
    y = np.array(y)

    # leave-one-out
    correct = 0
    confusion = np.zeros((len(words), len(words)), dtype=int)
    for i in range(len(X)):
        X_train = np.delete(X, i, axis=0)
        y_train = np.delete(y, i)
        X_test  = X[i:i+1]
        y_test  = y[i]

        clf = make_pipeline(StandardScaler(), SVC(kernel="rbf", C=1.0))
        clf.fit(X_train, y_train)
        pred = clf.predict(X_test)[0]
        confusion[y_test, pred] += 1
        if pred == y_test:
            correct += 1

    acc = correct / len(X)
    chance = 1.0 / len(words)

    print(f"LOO accuracy : {correct}/{len(X)}  =  {acc:.0%}")
    print(f"Chance level : {chance:.0%}  ({len(words)}-class)")
    print(f"Ratio vs chance: {acc/chance:.1f}x\n")

    # confusion matrix
    print("Confusion matrix (rows=true, cols=pred):")
    header = f"{'':>10}" + "".join(f"{w[:6]:>8}" for w in words)
    print(header)
    for i, word in enumerate(words):
        row = f"{word:>10}" + "".join(f"{confusion[i,j]:>8}" for j in range(len(words)))
        print(row)


if __name__ == "__main__":
    main()
