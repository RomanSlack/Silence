import time
from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds

BoardShim.enable_dev_board_logger()

params = BrainFlowInputParams()
params.serial_port = '/dev/ttyUSB0'

board = BoardShim(BoardIds.CYTON_BOARD, params)

try:
    board.prepare_session()
    print("Connected! Starting stream...")
    board.start_stream()
    time.sleep(5)
    data = board.get_board_data()
    print(f"Got {data.shape[1]} samples across {data.shape[0]} channels")
    print(f"Sample rate: ~{data.shape[1]/5:.0f} samples/sec (expected ~250)")
    board.stop_stream()
    board.release_session()
    print("Done. Board is working.")
except Exception as e:
    print(f"ERROR: {e}")
