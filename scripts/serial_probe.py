#!/usr/bin/env python3
"""
Raw serial probe for OpenBCI Cyton firmware 3.1.1.
Sends wakeup commands and listens for a response.

Usage: sudo python serial_probe.py
"""

import serial
import time

PORT = "/dev/ttyUSB0"
BAUD = 115200

print(f"Opening {PORT} at {BAUD} baud...")
ser = serial.Serial(PORT, BAUD, timeout=3)
time.sleep(0.5)

ser.reset_input_buffer()

print("\n--- Sending 'v' (version / soft reset) ---")
ser.write(b'v')
time.sleep(2)
resp = ser.read(ser.in_waiting or 512)
if resp:
    print(f"Got {len(resp)} bytes: {resp}")
else:
    print("No response to 'v'")

print("\n--- Sending 'b' (start streaming) ---")
ser.write(b'b')
time.sleep(3)
resp = ser.read(ser.in_waiting or 512)
if resp:
    print(f"Got {len(resp)} bytes: {resp[:200]}")
    print("\nBoard is alive and streaming. Wiring is correct.")
else:
    print("No response to 'b' either.")
    print("\nPossible causes:")
    print("  - TX/RX swapped: GND is correct (north, LED on) but signal wires need flipping")
    print("  - Board not powered / switch not in PC mode")
    print("  - Wrong baud rate (unlikely, 115200 is standard Cyton)")

print("\n--- Sending 's' (stop streaming) ---")
ser.write(b's')
time.sleep(0.5)
ser.close()
print("Done.")
