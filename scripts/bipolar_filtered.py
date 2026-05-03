#!/usr/bin/env python3
"""
Bipolar CH1 with 20-500 Hz bandpass. Prints RMS once per 0.5 s.
Rest -> RMS ~1-30 uV. Clench -> RMS ~100-1000+ uV.

Wire: electrode A -> N1P, electrode B -> N1N, ~2 cm apart on target muscle.

Usage: sudo ../ml_backend/.venv/bin/python scripts/bipolar_filtered.py
"""

import time
import numpy as np
from scipy.signal import butter, sosfiltfilt
from brainflow.board_shim import BoardShim, BoardIds, BrainFlowInputParams

PORT = "/dev/ttyUSB0"
FS = 250.0
CMD_CH1_BIPOLAR = 'x1060000X'

sos = butter(4, [20.0, 115.0], btype='band', fs=FS, output='sos')

BoardShim.disable_board_logger()
params = BrainFlowInputParams(); params.serial_port = PORT
board = BoardShim(BoardIds.CYTON_BOARD.value, params)
board.prepare_session()
board.config_board(CMD_CH1_BIPOLAR); time.sleep(1)

emg_rows = BoardShim.get_emg_channels(BoardIds.CYTON_BOARD.value)

print("Streaming. Ctrl-C to stop.\n")
print(f"{'rms (uV)':>10} | {'peak (uV)':>10} | state")
print("-" * 40)

board.start_stream()
try:
    while True:
        time.sleep(0.5)
        data = board.get_current_board_data(int(0.5 * FS))
        if data.shape[1] < 20:
            continue
        ch1 = data[emg_rows[0]].astype(np.float64)
        ch1 -= ch1.mean()
        filt = sosfiltfilt(sos, ch1)
        rms = float(np.sqrt(np.mean(filt**2)))
        peak = float(np.max(np.abs(filt)))
        state = "CLENCH" if rms > 80 else ("?" if rms > 30 else "rest")
        print(f"{rms:10.1f} | {peak:10.1f} | {state}")
except KeyboardInterrupt:
    pass

board.stop_stream()
board.release_session()
