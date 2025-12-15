"""
Persistence and resampling layer using DuckDB.

Responsibilities:
- Persist raw tick data
- Resample ticks into OHLCV bars (1s, 1m, 5m)
- Serve analytics-ready queries

This module must never:
- Connect to WebSockets
- Perform analytics logic
- Know about Streamlit or UI
"""

import duckdb
import threading
import time
from typing import List

from utils import Tick


class DuckDBStorage:
    def __init__(self, db_path: str = ":memory:"):
        # Use in-memory DB to avoid file locking between runs
        # For persistence, use "data/market.duckdb" but ensure single writer
        self.conn = duckdb.connect(db_path)
        self._lock = threading.Lock()
        self._init_schema()

    def _init_schema(self):
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS ticks (
                event_time BIGINT,
                symbol TEXT,
                price DOUBLE,
                quantity DOUBLE,
                receipt_time DOUBLE
            )
        """)

    def insert_ticks(self, ticks: List[Tick]):
        if not ticks:
            return

        with self._lock:
            self.conn.executemany(
                "INSERT INTO ticks VALUES (?, ?, ?, ?, ?)",
                [
                    (
                        t.event_time,
                        t.symbol,
                        t.price,
                        t.quantity,
                        t.receipt_time
                    )
                    for t in ticks
                ]
            )

    def resample_ohlcv(self, symbol: str, timeframe: str):
        """
        timeframe: '1s', '1m', '5m'
        """
        bucket_map = {
            "1s": "1 second",
            "1m": "1 minute",
            "5m": "5 minute"
        }

        if timeframe not in bucket_map:
            raise ValueError("Unsupported timeframe")

        interval = bucket_map[timeframe]

        query = f"""
        SELECT
            time_bucket(INTERVAL '{interval}', to_timestamp(event_time / 1000)) AS ts,
            FIRST(price) AS open,
            MAX(price) AS high,
            MIN(price) AS low,
            LAST(price) AS close,
            SUM(quantity) AS volume
        FROM ticks
        WHERE symbol = ?
        GROUP BY ts
        ORDER BY ts
        """

        return self.conn.execute(query, [symbol]).fetchdf()
