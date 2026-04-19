# Silence — ML Backend

Machine-learning pipeline for the Silence silent-speech device.

## Mission

Build a wearable silent-speech recognition system worn on the face/jaw. Surface EMG of speech muscles (mentalis, masseter, mastoid reference) is classified into words **silently mouthed or subvocalized** by the user. Same fundamental technique as MIT AlterEgo (2018) — not EEG.

**MVP target**: recognize **≥20 words**, trained on one user (Roman). Single-speaker system; no generalization required for v1.

**Long term**: silent universal translation and an invisible AI-assistant interface.

## Why this sub-module exists separately

The hardware track (OpenBCI V3 clone) is currently blocked on custom firmware from the seller. The ML track does not wait. We build the full pipeline against a **public sEMG silent-speech dataset** (Gaddy & Klein 2020) so that when the board is streaming, it is a data-source swap — not a rewrite.

## Approach

Two-model cascade (the potentially novel contribution):

1. **Model 1 — overt mouthing.** Trained on clearly mouthed speech. Stronger EMG signal, easier supervision. Acts as bootstrap.
2. **Model 2 — pure subvocalization.** Trained on signals with zero visible movement, using Model 1's predictions on paired data as soft labels (self-supervised distillation).

At inference: ensemble both, switch by signal amplitude.

The MVP classifier (this repo's first deliverable) only implements Model 1 + a closed 20-word vocabulary. Model 2 + self-distillation comes after the baseline lands.

## Stack

- **Python** 3.12
- **PyTorch** 2.11 (raw — no Lightning, no Hydra)
- **NumPy / SciPy** for signal processing
- **MNE-Python** for bandpass, notch, epoching utilities
- **BrainFlow** for live OpenBCI streaming (wired up later)
- **soundfile / librosa** only for loading Gaddy's paired audio (not required at inference)

## Repo layout

```
ml_backend/
├── silence_ml/
│   ├── data/
│   │   ├── base.py          # EMGRecording abstraction
│   │   ├── gaddy.py         # Gaddy 2020 Zenodo dataset loader
│   │   └── openbci.py       # live OpenBCI / saved-session loader (stub)
│   ├── preprocess/
│   │   ├── filters.py       # bandpass, notch, z-score
│   │   └── epoch.py         # windowing + label alignment
│   ├── models/
│   │   └── cnn.py           # baseline 1D CNN classifier
│   ├── train.py
│   └── eval.py
├── scripts/
│   ├── download_gaddy.py    # pulls Zenodo tarball
│   └── train_baseline.py    # end-to-end: data → features → CNN → eval
├── vocab/
│   └── mvp_20words.txt
├── data/                    # gitignored; raw + processed EMG
├── checkpoints/             # gitignored
├── pyproject.toml
└── README.md
```

## Dataset — Gaddy & Klein 2020

- 8-channel facial sEMG from a single speaker
- ~20 hours, both silent and vocalized modes
- Zenodo DOI: `10.5281/zenodo.4064408` ([link](https://zenodo.org/records/4064409))
- Original paper: [Digital Voicing of Silent Speech](https://aclanthology.org/2020.emnlp-main.445/) (EMNLP 2020)
- Reference repo: [dgaddy/silent_speech](https://github.com/dgaddy/silent_speech)

We use Gaddy for pipeline bring-up. The original paper does seq2seq decoding to audio/text; we use the same data as a **closed-vocabulary classification** task for the MVP.

## State of the art (context)

- **2025, MDPI Sensors** — single-channel EMG + EEG fusion, Siamese few-shot, 95.25% on 10-word military sentences.
- **sEMG (face/neck), 2200-word vocab** — 8.9% WER (91.1% recognition).

91–95% at small vocab is achievable with modern architectures on single-user data. Our ~92% target is realistic.

## Getting started

```bash
cd ml_backend
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

# Pull the Gaddy dataset (~5 GB)
python scripts/download_gaddy.py

# Train the baseline classifier
python scripts/train_baseline.py
```

## MVP 20-word vocabulary

See `vocab/mvp_20words.txt`. Chosen for a useful command surface (digits + navigation + meta-commands) and phonetic diversity so the classifier sees varied EMG patterns.

## Roadmap

- [x] Repo scaffold
- [ ] Gaddy dataset loader + preprocessing sanity plots
- [ ] Baseline 1D CNN classifier on Gaddy vocalized subset
- [ ] Closed 20-word vocab evaluation harness
- [ ] OpenBCI live loader (blocked on hardware)
- [ ] First Roman-personal recording session + Model 1 training
- [ ] Model 2: subvocalization self-distillation
- [ ] Two-model cascade inference + switching heuristic
- [ ] Real-time demo loop w/ bone-conduction audio feedback
