import serial
import time

port = '/dev/ttyUSB0'

for baud in [115200, 9600, 57600]:
    print(f"\nTrying {baud} baud - just listening for 4 seconds...")
    ser = serial.Serial(port, baud, timeout=4)
    ser.setDTR(False)
    ser.setRTS(False)
    time.sleep(0.5)
    ser.setDTR(True)
    time.sleep(3)
    response = ser.read(ser.in_waiting or 256)
    print(f"Got ({len(response)} bytes): {response}")
    ser.close()
    if response:
        print(f"FOUND response at {baud} baud!")
        break
