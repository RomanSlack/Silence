#!/usr/bin/env python3
"""
Dump ADS1299 register state from Cyton firmware via `?` command.
Tells us if BIAS_DRV is actually enabled and what each channel is referenced to.

Usage: sudo ../ml_backend/.venv/bin/python scripts/dump_registers.py
"""

import serial
import time

PORT = "/dev/ttyUSB0"
BAUD = 115200

print(f"Opening {PORT}...")
ser = serial.Serial(PORT, BAUD, timeout=3)
time.sleep(0.5)
ser.reset_input_buffer()

print("Sending 'v' (soft reset / version)...")
ser.write(b'v')
time.sleep(2)
_ = ser.read(ser.in_waiting or 512)

print("Sending '?' (register query)...\n")
ser.write(b'?')
time.sleep(3)
resp = ser.read(ser.in_waiting or 4096)
print(resp.decode(errors='replace'))

ser.close()
