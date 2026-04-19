import asyncio
from bleak import BleakScanner

async def scan():
    print("Scanning 15 seconds - put board in BLE mode and close to PC...")
    devices = await BleakScanner.discover(timeout=15, return_adv=True)
    for addr, (device, adv) in devices.items():
        print(f"\n{addr} | name={device.name} | rssi={adv.rssi}dB")
        if adv.manufacturer_data:
            print(f"  mfr_data: {dict((k, v.hex()) for k,v in adv.manufacturer_data.items())}")
        if adv.service_uuids:
            print(f"  services: {adv.service_uuids}")

asyncio.run(scan())
