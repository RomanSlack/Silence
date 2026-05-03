#!/usr/bin/env python3
"""
ADS1299 shorted-input self-test.

Sets CH1 input MUX to SHORTED (internally shorts the PGA inputs to each other,
bypassing all electrode/body/BIAS/SRB concerns), streams via BrainFlow,
and prints CH1 min/max/mean.

- If CH1 reads ~0 uV with small noise -> amp+ADC are healthy; fault is in the
  external electrode/BIAS/SRB path (probably clone mis-wires BIAS or SRB pin).
- If CH1 still rails at 187,500 -> chip or analog supply is broken.

Usage: sudo ../ml_backend/.venv/bin/python scripts/ch_selftest.py
"""

import time
import numpy as np
from brainflow.board_shim import BoardShim, BoardIds, BrainFlowInputParams

PORT = "/dev/ttyUSB0"

# Channel setting string for OpenBCI:
#   x C P G T I B1 B2 X
# C=channel, P=power(0=on), G=gain(6=24x), T=input_type(1=SHORTED),
# I=bias(0=off), B1=SRB2(0=off), B2=SRB1(0=off)
CMD_CH1_SHORTED = 'x1061000X'

BoardShim.disable_board_logger()
params = BrainFlowInputParams()
params.serial_port = PORT
board = BoardShim(BoardIds.CYTON_BOARD.value, params)
board.prepare_session()

print("Configuring CH1 -> SHORTED input...")
board.config_board(CMD_CH1_SHORTED)
time.sleep(1)

print("Streaming for 5 s...")
board.start_stream()
time.sleep(5)
data = board.get_board_data()
board.stop_stream()
board.release_session()

emg_rows = BoardShim.get_emg_channels(BoardIds.CYTON_BOARD.value)
ch1 = data[emg_rows[0]]

print(f"\nCH1 SHORTED-input result (n={len(ch1)} samples):")
print(f"  min  = {ch1.min():.1f} uV")
print(f"  max  = {ch1.max():.1f} uV")
print(f"  mean = {ch1.mean():.1f} uV")
print(f"  std  = {ch1.std():.1f} uV")

if abs(ch1.mean()) > 100000 or ch1.std() < 0.01:
    print("\n=> RAILED. Amp/ADC or analog supply is broken on this clone.")
elif abs(ch1.mean()) < 100 and ch1.std() < 50:
    print("\n=> CLEAN. Amp is healthy. The fault is in external path")
    print("   (clone's BIAS/SRB pin likely not wired through to ADS1299).")
else:
    print("\n=> INTERMEDIATE. Not railed but noisier than expected. Reduce gain and retest.")
