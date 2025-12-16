"""
Real-time quantitative analytics engine.

Responsibilities:
- Compute rolling statistical-arbitrage analytics
- Operate on in-memory tick buffers
- Emit analytics snapshots at controlled intervals

This module must never:
- Connect to WebSockets
- Talk to Streamlit
- Control persistence logic
"""

import time
import numpy as np
from typing import Dict, Optional

from utils import TickBuffer


class AnalyticsEngine:
    def __init__(
        self,
        buffer: TickBuffer,
        symbol_x: str,
        symbol_y: str,
        window_size: int = 200
    ):
        self.buffer = buffer
        self.symbol_x = symbol_x
        self.symbol_y = symbol_y
        self.window_size = window_size

    def _extract_log_prices(self, symbol: str) -> Optional[np.ndarray]:
        ticks = list(self.buffer.get(symbol))
        if len(ticks) < self.window_size:
            return None

        # Extract last N prices
        prices = np.array(
            [t.price for t in ticks[-self.window_size:]],
            dtype=np.float64
        )
        
        # Filter out zero/negative prices (prevents log errors)
        prices = prices[prices > 0]
        if len(prices) < self.window_size:
            return None

        return np.log(prices)

    def _rolling_ols(self, x: np.ndarray, y: np.ndarray):
        """
        Solve y = alpha + beta * x using least squares.
        """
        A = np.vstack([x, np.ones(len(x))]).T
        beta, alpha = np.linalg.lstsq(A, y, rcond=None)[0]
        return alpha, beta

    def compute(self) -> Optional[Dict]:
        """
        Compute analytics snapshot.
        Returns None if insufficient data.
        """
        x = self._extract_log_prices(self.symbol_x)
        y = self._extract_log_prices(self.symbol_y)

        if x is None or y is None:
            return None

        alpha, beta = self._rolling_ols(x, y)

        spread = y - (alpha + beta * x)
        mean = spread.mean()
        std = spread.std()

        if std == 0:
            return None

        zscore = (spread[-1] - mean) / std
        correlation = np.corrcoef(x, y)[0, 1]

        snapshot = {
            "timestamp": time.time(),
            "pair": f"{self.symbol_y}/{self.symbol_x}",
            "alpha": float(alpha),
            "beta": float(beta),
            "spread": float(spread[-1]),
            "zscore": float(zscore),
            "correlation": float(correlation),
            "window": self.window_size
        }

        return snapshot
