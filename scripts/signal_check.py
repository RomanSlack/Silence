#!/usr/bin/env python3
"""
Live RMS per channel — run this before recording to confirm electrode contact.
Clench jaw hard, you should see channels 1 and/or 2 spike.

Usage: sudo python scripts/signal_check.py
"""

import time
import sys
import numpy as np

try:
    from brainflow.board_shim import BoardShim, BoardIds, BrainFlowInputParams
except ImportError:
    print("ERROR: run with venv: sudo ../ml_backend/.venv/bin/python signal_check.py")
    sys.exit(1)

PORT = "/dev/ttyUSB0"
WINDOW_SEC = 0.5
N_ACTIVE = 8

BoardShim.disable_board_logger()
params = BrainFlowInputParams()
params.serial_port = PORT
board = BoardShim(BoardIds.CYTON_BOARD.value, params)

print("Connecting...")
board.prepare_session()
board.start_stream()
print("Streaming. Watch channels 1+2 — clench your jaw hard to test.\n")
print("Ctrl+C to stop.\n")

try:
    while True:
        time.sleep(WINDOW_SEC)
        data = board.get_current_board_data(int(WINDOW_SEC * 250))
        if data.shape[1] == 0:
            continue
        rms = np.sqrt(np.mean(data[:N_ACTIVE] ** 2, axis=1))
        peak = np.max(np.abs(data[:N_ACTIVE]), axis=1)

        bars = []
        for ch in range(N_ACTIVE):
            r = rms[ch]
            bar_len = min(40, int(r / 5))
            bar = "#" * bar_len
            marker = " <-- JAW" if ch in (0, 1) and r > 50 else ""
            bars.append(f"  ch{ch+1}  rms={r:7.1f}  peak={peak[ch]:7.1f}  |{bar}{marker}")

        print("\033[2J\033[H", end="")  # clear screen
        print("=== Signal Check — clench jaw to test ===\n")
        print("\n".join(bars))
        print("\nch1=mentalis  ch2=masseter  (SRB+BIAS on mastoid)")
        print("Good signal: rms>100 on clench, rms<30 at rest")

except KeyboardInterrupt:
    pass
finally:
    board.stop_stream()
    board.release_session()
    print("\nDone.")
