"""
Shared data models and system state definitions.

This module defines immutable data contracts used across
ingestion, analytics, and storage layers.
"""

from dataclasses import dataclass
from typing import Deque, Dict, List
from collections import deque
import threading


@dataclass(frozen=True)
class Tick:
    symbol: str
    event_time: int      # exchange timestamp (ms)
    price: float
    quantity: float
    receipt_time: float  # local timestamp (s)


class TickBuffer:
    """
    Thread-safe in-memory ring buffers for tick storage.

    Each symbol has its own bounded buffer.
    Old data is dropped automatically.
    """

    def __init__(self, maxlen: int = 10_000):
        self.buffers: Dict[str, Deque[Tick]] = {}
        self.maxlen = maxlen
        self._lock = threading.RLock()

    def add(self, tick: Tick):
        """Add a tick to the appropriate symbol buffer."""
        with self._lock:
            if tick.symbol not in self.buffers:
                self.buffers[tick.symbol] = deque(maxlen=self.maxlen)
            self.buffers[tick.symbol].append(tick)

    def get(self, symbol: str) -> Deque[Tick]:
        """Get buffer for a symbol (returns empty deque if not found)."""
        with self._lock:
            return self.buffers.get(symbol, deque())

    def clear(self):
        """Clear all buffers."""
        with self._lock:
            self.buffers.clear()

    def clear_symbol(self, symbol: str):
        """Clear buffer for a specific symbol."""
        with self._lock:
            if symbol in self.buffers:
                self.buffers[symbol].clear()

    def get_symbols(self) -> List[str]:
        """Get list of symbols currently in buffer."""
        with self._lock:
            return list(self.buffers.keys())

    def get_count(self, symbol: str) -> int:
        """Get count of ticks for a symbol."""
        with self._lock:
            if symbol in self.buffers:
                return len(self.buffers[symbol])
            return 0
