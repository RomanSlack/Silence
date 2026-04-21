"""Board abstraction.

Any board exposes a running ring buffer of EMG samples. `capture(duration_sec)`
blocks for that long and returns the most recent window.

`MockBoard` produces plausible-looking synthetic EMG so the recording UI works
end-to-end with no hardware attached. When the OpenBCI firmware issue is
resolved, add a BrainFlow-backed `CytonBoard` that implements the same three
methods — nothing else in the app needs to change.
"""

from __future__ import annotations

import threading
import time
from typing import Protocol

import numpy as np


class Board(Protocol):
    sample_rate: float
    n_channels: int

    def start(self) -> None: ...
    def stop(self) -> None: ...
    def capture(self, duration_sec: float) -> np.ndarray: ...


class MockBoard:
    """Synthetic 8-channel EMG — bandpass-like pink-ish noise + occasional bursts."""

    def __init__(self, sample_rate: float = 1000.0, n_channels: int = 8,
                 buffer_sec: float = 10.0, seed: int = 0):
        self.sample_rate = float(sample_rate)
        self.n_channels = int(n_channels)
        self._buf_len = int(buffer_sec * sample_rate)
        self._buf = np.zeros((n_channels, self._buf_len), dtype=np.float32)
        self._write_idx = 0
        self._lock = threading.Lock()
        self._rng = np.random.default_rng(seed)
        self._running = False
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._producer, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=0.5)
            self._thread = None

    def _producer(self) -> None:
        chunk = max(1, int(self.sample_rate * 0.05))  # 50 ms chunks
        dt = chunk / self.sample_rate
        while self._running:
            noise = self._rng.standard_normal((self.n_channels, chunk)).astype(np.float32) * 15.0
            if self._rng.random() < 0.05:
                burst_len = self._rng.integers(20, min(chunk, 120))
                start = self._rng.integers(0, chunk - burst_len + 1)
                amp = self._rng.uniform(40, 120)
                ch = self._rng.integers(0, self.n_channels)
                noise[ch, start:start + burst_len] += amp * np.sin(
                    2 * np.pi * self._rng.uniform(80, 200) *
                    np.arange(burst_len) / self.sample_rate
                )
            with self._lock:
                end = self._write_idx + chunk
                if end <= self._buf_len:
                    self._buf[:, self._write_idx:end] = noise
                else:
                    first = self._buf_len - self._write_idx
                    self._buf[:, self._write_idx:] = noise[:, :first]
                    self._buf[:, :chunk - first] = noise[:, first:]
                self._write_idx = end % self._buf_len
            time.sleep(dt)

    def capture(self, duration_sec: float) -> np.ndarray:
        """Block for `duration_sec`, then return the most recent window.

        Returns shape (n_channels, n_samples), dtype float32.
        """
        time.sleep(duration_sec)
        n = int(duration_sec * self.sample_rate)
        n = min(n, self._buf_len)
        with self._lock:
            end = self._write_idx
            start = end - n
            if start >= 0:
                out = self._buf[:, start:end].copy()
            else:
                out = np.concatenate(
                    [self._buf[:, self._buf_len + start:], self._buf[:, :end]], axis=1
                )
        return out.astype(np.float32)


class CytonBoard:
    """Live OpenBCI Cyton board via BrainFlow. CP2102 wired serial connection."""

    sample_rate = 250.0
    n_channels = 8

    def __init__(self, serial_port: str = "/dev/ttyUSB0"):
        from brainflow.board_shim import BoardShim, BoardIds, BrainFlowInputParams
        BoardShim.disable_board_logger()
        params = BrainFlowInputParams()
        params.serial_port = serial_port
        self._board = BoardShim(BoardIds.CYTON_BOARD.value, params)
        self._emg_channels = list(range(8))  # Cyton channels 0-7 are the 8 EXG channels

    def start(self) -> None:
        self._board.prepare_session()
        self._board.start_stream()

    def stop(self) -> None:
        self._board.stop_stream()
        self._board.release_session()

    def capture(self, duration_sec: float) -> np.ndarray:
        """Block for duration_sec, return (8, n_samples) float32 in µV."""
        time.sleep(duration_sec)
        data = self._board.get_current_board_data(int(duration_sec * self.sample_rate))
        return data[self._emg_channels, :].astype(np.float32)


def make_board(kind: str = "mock", **kw) -> Board:
    if kind == "mock":
        return MockBoard(**kw)
    if kind == "cyton":
        serial_port = kw.pop("serial_port", "/dev/ttyUSB0")
        return CytonBoard(serial_port=serial_port)
    raise ValueError(f"unknown board kind: {kind}")
