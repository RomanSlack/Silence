"""Live + saved-session loader for OpenBCI Cyton-family boards.

STUB — hardware is blocked on custom firmware from the eBay seller as of
2026-04-19. Wire this up once the board streams. Expected sample rate: 250 Hz
for stock Cyton firmware, 8 channels.

Design notes for when the board works:
- Use BrainFlow's BoardShim with BoardIds.CYTON_BOARD on /dev/ttyUSB0.
- Stream into a ring buffer, mark trial boundaries via a keypress or button.
- Save raw sessions as .npz {signal: (8, n_samples), sample_rate, transcript,
  label, timestamp} so they load through the same EMGRecording interface as
  Gaddy.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterator

import numpy as np

from silence_ml.data.base import EMGRecording

OPENBCI_SAMPLE_RATE = 250.0
OPENBCI_N_CHANNELS = 8


def iter_saved_sessions(root: str | Path) -> Iterator[EMGRecording]:
    """Load .npz sessions previously captured from the board."""
    root = Path(root)
    for path in sorted(root.glob("*.npz")):
        d = np.load(path, allow_pickle=True)
        yield EMGRecording(
            signal=d["signal"].astype(np.float32),
            sample_rate=float(d["sample_rate"]),
            transcript=str(d.get("transcript", "")) or None,
            label=int(d.get("label", -1)),
            source=f"openbci:{path.name}",
            meta={"timestamp": str(d.get("timestamp", ""))},
        )


def stream_live(serial_port: str = "/dev/ttyUSB0"):
    raise NotImplementedError(
        "Live OpenBCI streaming is blocked on hardware firmware. "
        "Once the seller provides working firmware, implement with "
        "brainflow.board_shim.BoardShim(BoardIds.CYTON_BOARD, params)."
    )
