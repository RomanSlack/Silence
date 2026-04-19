# Data Collection Strategy

How much data to record, when, and how, to hit MVP and eventually AlterEgo-level accuracy.

## How much

Single-user, 20-word closed vocabulary, sEMG on jaw/face muscles.

| Tier | Reps × Words | Total | Recording time | Expected accuracy (overt) |
|---|---|---|---|---|
| First-signal | 25 × 20 | 500 | ~25 min | 40–70% — confirms the pipeline learns *something* on your muscles |
| **MVP** | **50 × 20** | **1000** | **~45 min** | **70–85%** |
| Solid | 100 × 20 | 2000 | ~1.5 hr | 85–92% |
| AlterEgo-level | 200 × 20 | 4000 | ~3 hr | 90–95% |

Numbers assume clean electrode prep and the round-robin recording mode. Bad electrodes can drop you two tiers.

Subvocal (Model 2) adds **another ~100/word of silent mouthing paired with the overt recordings** (same prompts in the same order), so the overt twin's Model-1 prediction can serve as a soft label for the silent twin.

## Session structure

**Cap each recording block at ~20–25 min of active trials.** Fatigue and electrode drift degrade the signal past that — you'll be training on noise.

A good MVP plan:
- **Day 1, session A (25 min):** 25 × 20 = 500 overt trials
- **Day 1, session B (25 min, later same day):** another 25 × 20 = 500 overt trials
- Train Model 1 on the 1000. Evaluate.
- **Day 2, session C (25 min):** 50 × 20 = 1000 more overt trials if Model 1 looks promising
- **Day 3+, paired silent sessions** for Model 2

**Spread across ≥2 days.** Variance from re-prepping electrodes, different skin conductance, different muscle tone is what teaches the model *you the person*, not *you at 2pm on Tuesday*.

## Hygiene

Order these by ROI:

1. **Electrode prep**: alcohol wipe, let dry, fresh gel if using wet electrodes, firm tape. Repeat at the start of *every* session.
2. **Posture consistency**: sit the same way each session. Jaw/neck angle shifts the muscle geometry.
3. **Round-robin mode**: cycle through all 20 words, don't block (50 "yes" in a row then 50 "no" lets the model learn a time-of-session bias that won't hold at inference).
4. **Watch the signal readout** (rms/peak) after each trial. If channels go flat mid-session, re-prep.
5. **Mark bad trials immediately** — `U` to undo and re-record. Don't let a bad trial into the dataset planning to "clean it later."

## Overt vs subvocal

**Overt (Model 1):** visibly mouthing the word, lips + tongue moving, no voice. Stronger EMG signal. Train this first. It's the bootstrap.

**Subvocal (Model 2):** zero visible movement. Deliberately slow down and *hyperarticulate internally* — this is counterintuitive but increases articulatory effort and gives the EMG enough to work with. Record the same prompts as Model 1 in the same order so paired distillation works.

## When to stop

Stopping rules that aren't "keep going until you hit 4000":

- Train after every 500–1000 new trials. Plot val-acc over data size. When the curve goes flat, more of the same data won't help — the next bottleneck is model, features, or electrode placement, not count.
- If accuracy is stuck at chance after 500 trials, stop recording and fix electrodes/placement. More bad data is worse than less good data.
- If you're fatigued, stop. A fatigued 30-min block is *net negative* — it adds noise and the model learns "tired Roman" as a class signal.

## On-disk

Every trial the recorder writes:

```
data/sessions/{session_name}/
  manifest.json
  yes_0000.npz     # signal (C, T) float32, sample_rate, word, label, trial_idx, timestamp
  yes_0001.npz
  ...
```

This format is what `silence_ml/data/openbci.py::iter_saved_sessions` already loads. Training on your data once recorded is a one-line data-source swap in `train_baseline.py`.

## MVP checklist

- [ ] Firmware working, clean signal visible in the recorder's rms/peak readout on jaw clench
- [ ] Session 1: 500 overt trials (round-robin)
- [ ] Session 2 (same day or next): 500 more overt trials
- [ ] Train Model 1 on 1000 trials; check val-acc vs data-size curve
- [ ] Decision point: push to 2000 overt, or start paired silent recordings for Model 2
