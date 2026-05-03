#!/usr/bin/env python3
"""
Bipolar-differential CH1 test.

Reconfigures CH1 so SRB2 is disconnected and the channel reads
(N1P - N1N) directly. Also disables BIAS contribution from CH1.

Wire two electrodes on the same muscle (e.g. forearm flexor):
  - Electrode A -> N1P (top row, second from right on the top header)
  - Electrode B -> N1N (bottom row, corresponding position)
  - Space them ~2 cm along the muscle belly.
  - BIAS and SRB cups can be left dangling or removed — not used here.

Streams for 10 s and prints CH1 stats + a rolling live view.
Clench the muscle every ~2 s. Look for amplitude bursts in the "max" column.

Usage: sudo ../ml_backend/.venv/bin/python scripts/bipolar_test.py
"""

import time
import numpy as np
from brainflow.board_shim import BoardShim, BoardIds, BrainFlowInputParams

PORT = "/dev/ttyUSB0"

# x C P G T I B1 B2 X
# C=1 ch1, P=0 on, G=6 gain24, T=0 normal, I=0 no bias, B1=0 no SRB2, B2=0 no SRB1
CMD_CH1_BIPOLAR = 'x1060000X'

BoardShim.disable_board_logger()
params = BrainFlowInputParams()
params.serial_port = PORT
board = BoardShim(BoardIds.CYTON_BOARD.value, params)
board.prepare_session()

print("Configuring CH1 as bipolar (N1P - N1N), no SRB, no BIAS...")
board.config_board(CMD_CH1_BIPOLAR)
time.sleep(1)

emg_rows = BoardShim.get_emg_channels(BoardIds.CYTON_BOARD.value)
print("\nStreaming 10 s. Clench the target muscle every ~2 s.\n")
print(f"{'t(s)':>5} | {'min':>10} | {'max':>10} | {'mean':>10} | {'std':>10}")
print("-" * 60)

board.start_stream()
t_start = time.time()
last_print = 0
while time.time() - t_start < 10:
    time.sleep(0.5)
    now = time.time() - t_start
    if now - last_print >= 0.5:
        data = board.get_current_board_data(125)  # ~0.5 s at 250 Hz
        if data.shape[1] > 0:
            ch1 = data[emg_rows[0]]
            print(f"{now:5.1f} | {ch1.min():10.1f} | {ch1.max():10.1f} | {ch1.mean():10.1f} | {ch1.std():10.1f}")
            last_print = now

board.stop_stream()
board.release_session()

print("\nInterpretation:")
print("  * std ~ 1-10 uV at rest + bursts of std > 100 uV on clench = WORKING.")
print("  * all rows at ~ +/- 187500 = still railed. Body common-mode is too far")
print("    from board GND. Tie a third electrode on skin to CP2102 GND wire")
print("    as a crude ground and rerun.")
