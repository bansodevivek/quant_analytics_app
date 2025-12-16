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
from typing import Dict, Optional, Union

from utils import TickBuffer


# ============================================================
# PURE HELPER FUNCTIONS (no I/O, no side effects)
# ============================================================

# DIAGNOSTICS_WINDOW: Fixed window for local stationarity check
# 30 points minimum for statistically valid ADF test
DIAGNOSTICS_WINDOW = 30


def run_adf_test(spread: np.ndarray) -> Dict:
    """
    Run Augmented Dickey-Fuller test for LOCAL stationarity.
    
    Uses exactly DIAGNOSTICS_WINDOW points (last 30) for responsive,
    trader-relevant stationarity detection.
    
    ALWAYS returns a dict (never None) with full contract:
        {
            "valid": bool,
            "points_used": int,
            "reason": str | None,
            "p_value": float | None,
            "is_stationary": bool | None,
            "critical_5pct": float | None
        }
    """
    # Base response structure
    result = {
        "valid": False,
        "points_used": 0,
        "reason": None,
        "p_value": None,
        "is_stationary": None,
        "critical_5pct": None
    }
    
    try:
        if spread is None:
            result["reason"] = "No spread data provided"
            return result
        
        # Clean NaN values
        spread = spread[~np.isnan(spread)]
        spread = spread[np.isfinite(spread)]
        
        result["points_used"] = len(spread)
        
        if len(spread) < DIAGNOSTICS_WINDOW:
            result["reason"] = f"Insufficient data: {len(spread)}/{DIAGNOSTICS_WINDOW} points"
            return result
        
        # ENFORCE: Use exactly last DIAGNOSTICS_WINDOW points
        spread = spread[-DIAGNOSTICS_WINDOW:]
        result["points_used"] = DIAGNOSTICS_WINDOW
        
        from statsmodels.tsa.stattools import adfuller
        
        adf_result = adfuller(spread, autolag='AIC')
        
        p_value = float(adf_result[1])
        critical_5pct = float(adf_result[4]['5%'])
        adf_stat = float(adf_result[0])
        
        result["valid"] = True
        result["reason"] = None
        result["p_value"] = p_value
        result["is_stationary"] = adf_stat < critical_5pct
        result["critical_5pct"] = critical_5pct
        
        return result
        
    except Exception as e:
        result["reason"] = f"ADF computation error: {str(e)}"
        return result


def compute_half_life(spread: np.ndarray) -> Optional[float]:
    """
    Compute mean-reversion half-life using AR(1) model.
    
    Uses exactly DIAGNOSTICS_WINDOW points for local estimation.
    
    Returns:
        Half-life in ticks, or None if unstable/invalid.
    """
    try:
        if spread is None:
            return None
        
        # Clean NaN values
        spread = spread[~np.isnan(spread)]
        spread = spread[np.isfinite(spread)]
        
        if len(spread) < DIAGNOSTICS_WINDOW:
            return None
        
        # ENFORCE: Use exactly last DIAGNOSTICS_WINDOW points
        spread = spread[-DIAGNOSTICS_WINDOW:]
        
        # AR(1) regression: spread[t] = phi * spread[t-1] + epsilon
        lagged = spread[:-1]
        delta = spread[1:] - lagged
        
        if np.std(lagged) == 0:
            return None
        
        # Fit: delta = phi * lagged + intercept
        phi = np.polyfit(lagged, delta, 1)[0]
        
        # Guard: phi must indicate mean-reversion (-1 < phi < 0)
        if phi >= 0 or phi <= -1:
            return None
        
        # Half-life = -ln(2) / ln(1 + phi)
        half_life = -np.log(2) / np.log(1 + phi)
        
        if not np.isfinite(half_life) or half_life <= 0:
            return None
        
        return float(half_life)
    except Exception:
        return None



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
