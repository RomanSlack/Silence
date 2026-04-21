import asyncio
from bleak import BleakClient

MAC = "48:31:B7:81:50:56"

received = []

def handler(sender, data):
    received.append(data)
    print(f"[{sender}] ({len(data)}B): {data.hex()} | {data}")

async def stream():
    async with BleakClient(MAC, timeout=15) as client:
        print("Connected")

        # Read static characteristics
        for uuid in ["0000fff1-0000-1000-8000-00805f9b34fb",
                     "f000ffe2-0451-4000-b000-000000000000"]:
            try:
                val = await client.read_gatt_char(uuid)
                print(f"Read {uuid[-8:]}: {val.hex()} | {val}")
            except Exception as e:
                print(f"Read {uuid[-8:]} failed: {e}")

        # Subscribe to all notify characteristics
        for uuid in ["0000fff1-0000-1000-8000-00805f9b34fb",
                     "f000ffe2-0451-4000-b000-000000000000"]:
            await client.start_notify(uuid, handler)
            print(f"Subscribed to {uuid[-8:]}")

        # Try commands on both write characteristics
        for write_uuid in ["0000fff2-0000-1000-8000-00805f9b34fb",
                           "f000ffe1-0451-4000-b000-000000000000"]:
            for cmd in [b'v', b'b', b'\x62']:
                try:
                    await client.write_gatt_char(write_uuid, cmd, response=False)
                    print(f"Sent {cmd} to {write_uuid[-8:]}")
                    await asyncio.sleep(1)
                except Exception as e:
                    print(f"Write failed: {e}")

        print("Listening 8 more seconds...")
        await asyncio.sleep(8)
        print(f"\nTotal packets: {len(received)}")

asyncio.run(stream())
