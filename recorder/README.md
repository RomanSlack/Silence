# Silence — Recording Harness

Flask UI for recording labeled EMG trials. Designed to be running on day-1 of
working hardware: the Cyton firmware is blocked, so we use a `MockBoard` now
and swap to a real `CytonBoard` later without touching the UI.

## Why a harness?

MVP target is 20 words × ~200 reps = 4000 labeled trials on Roman. That is
hours of focused recording. A clean UI (shows the target word, one-key record,
auto-advances, tracks per-word progress, resumable) makes this bearable and
produces clean on-disk labels ready for the ML backend.

## Run

```bash
cd recorder
../ml_backend/.venv/bin/python app.py
# open http://127.0.0.1:5001/
```

Flags:
- `--vocab <path>` — word list, one per line. Defaults to `ml_backend/vocab/mvp_20words.txt`.
- `--session <name>` — session dir name. Default: timestamp. Resumable — re-run with the same name.
- `--duration <sec>` — seconds per trial. Default 2.0.
- `--sample-rate`, `--channels` — defaults 1000 Hz, 8 ch (matches Gaddy / planned Cyton).
- `--board mock|cyton` — `mock` for synthetic EMG, `cyton` once firmware works.

## Keys

- **Space** — record one trial on the current word
- **U** — undo last trial (deletes the file)
- **← / →** — change current word
- **R** — toggle round-robin ↔ focus mode (focus stays on current word; round-robin advances)

## On-disk layout

```
recorder/data/sessions/{session_name}/
  manifest.json                       # live index, updated every trial
  yes_0000.npz                        # signal, sample_rate, word, label, trial_idx, duration_sec, timestamp
  yes_0001.npz
  no_0000.npz
  ...
```

`signal` shape is `(n_channels, n_samples)`, `float32`.

The saved `.npz` format is the same one `silence_ml/data/openbci.py::iter_saved_sessions`
expects, so training on your own data is:

```bash
../ml_backend/.venv/bin/python -c "
from silence_ml.data.openbci import iter_saved_sessions
for rec in iter_saved_sessions('data/sessions/2026-04-20_10-00-00'):
    print(rec.source, rec.transcript, rec.signal.shape)
"
```

## Wiring the real board (day-1 when firmware lands)

In `board.py`, add a `CytonBoard` class with the same three methods as `MockBoard`:

```python
class CytonBoard:
    sample_rate = 250.0   # stock Cyton
    n_channels = 8

    def __init__(self, serial_port="/dev/ttyUSB0"):
        from brainflow.board_shim import BoardShim, BoardIds, BrainFlowInputParams
        params = BrainFlowInputParams()
        params.serial_port = serial_port
        self._board = BoardShim(BoardIds.CYTON_BOARD.value, params)
        self._board.prepare_session()

    def start(self): self._board.start_stream()
    def stop(self):  self._board.stop_stream(); self._board.release_session()
    def capture(self, duration_sec):
        import time, numpy as np
        time.sleep(duration_sec)
        data = self._board.get_current_board_data(int(duration_sec * self.sample_rate))
        # brainflow returns (n_rows, n_samples); EMG rows are BoardShim.get_emg_channels(board_id)
        ...
```

Then update `make_board()` in `board.py` and pass `--board cyton` on the command line.

## Not doing (intentionally)

- **Live channel scope** — nice-to-have but would need WebSocket streaming. Skip until the real board arrives so we can calibrate the visualization against real signal, not synthetic.
- **Auto-threshold trial start** — AlterEgo-style "detect EMG burst, auto-crop window" is a training-time preprocessing step, not a capture-time one. We record fixed 2 s windows and crop later.
- **Hold-to-record** — clean idea but adds a state machine; spacebar+countdown is simpler and works fine for 20 × 200 trials.
