import asyncio
from bleak import BleakClient

MAC = "48:31:B7:81:50:56"

async def probe():
    print(f"Connecting to {MAC}...")
    async with BleakClient(MAC, timeout=15) as client:
        print(f"Connected: {client.is_connected}")
        for service in client.services:
            print(f"\nService: {service.uuid}")
            for char in service.characteristics:
                print(f"  Char: {char.uuid} | props: {char.properties}")

asyncio.run(probe())
