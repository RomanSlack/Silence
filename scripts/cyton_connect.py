#!/usr/bin/env python3
"""
Test BrainFlow connection to the Cyton board over /dev/ttyUSB0.

Run this after plugging the 3-pin cable into the BLE1 header and
switching the board to PC mode:

    /home/roman/Silence/ml_backend/.venv/bin/python scripts/cyton_connect.py

If it fails, flip the 3-pin connector 180 degrees and run again.
"""

import sys
import time

PORT = "/dev/ttyUSB0"

try:
    from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds
except ImportError:
    print("ERROR: brainflow not found. Run with the venv:")
    print("  /home/roman/Silence/ml_backend/.venv/bin/python scripts/cyton_connect.py")
    sys.exit(1)

import serial.tools.list_ports

def check_port():
    ports = [p.device for p in serial.tools.list_ports.comports()]
    if PORT not in ports:
        print(f"ERROR: {PORT} not found. Available ports: {ports or 'none'}")
        print("Check that the USB cable is plugged into the PC.")
        sys.exit(1)
    print(f"OK: {PORT} found.")

def connect():
    print(f"\nConnecting to Cyton on {PORT}...")
    BoardShim.disable_board_logger()
    params = BrainFlowInputParams()
    params.serial_port = PORT
    board = BoardShim(BoardIds.CYTON_BOARD, params)

    board.prepare_session()
    print("Session ready. Starting stream (5 seconds)...")
    board.start_stream()
    time.sleep(5)
    data = board.get_board_data()
    board.stop_stream()
    board.release_session()

    n_samples = data.shape[1]
    n_channels = data.shape[0]
    rate = n_samples / 5
    print(f"\nSUCCESS")
    print(f"  Channels : {n_channels}")
    print(f"  Samples  : {n_samples} ({rate:.0f} Hz, expected ~250)")
    if rate < 50:
        print("  WARNING: very low sample rate — possible wiring/orientation issue")
    else:
        print("  Board is alive and streaming. Wiring orientation is correct.")

if __name__ == "__main__":
    check_port()
    try:
        connect()
    except Exception as e:
        print(f"\nFAILED: {e}")
        print("\nWhat to try:")
        print("  1. Board switch is in PC mode (not BLE or OFF)?")
        print("  2. Battery connected and charged?")
        print("  3. Flip the 3-pin connector 180 degrees on the BLE1 header and run again.")
