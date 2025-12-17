"""
Asynchronous real-time data ingestion from Binance WebSocket streams.

Responsibilities:
- Maintain resilient WebSocket connections
- Parse tick-level trade data
- Push normalized ticks into in-memory buffers
- Support graceful stop and restart for symbol changes
- Never block downstream systems
"""

import asyncio
import json
import time
import logging
import threading
from typing import List, Optional

import websockets

from utils import Tick, TickBuffer


logger = logging.getLogger("ingest")


class BinanceIngestor:
    """
    Async WebSocket ingestor with graceful shutdown support.
    """
    
    def __init__(self, symbols: List[str], buffer: TickBuffer):
        self.symbols = [s.lower() for s in symbols]
        self.buffer = buffer
        self._running = False
        self._tasks: List[asyncio.Task] = []
        self._loop: Optional[asyncio.AbstractEventLoop] = None

    async def _connect_symbol(self, symbol: str):
        # Use Binance Futures endpoint (not Spot)
        url = f"wss://fstream.binance.com/ws/{symbol}@trade"
        backoff = 1
        max_retries = 5
        retry_count = 0

        while self._running:
            try:
                logger.info(f"[INGEST] Connecting to {symbol}")
                async with websockets.connect(
                    url,
                    ping_interval=20,
                    ping_timeout=10,
                    close_timeout=5
                ) as ws:
                    logger.info(f"[INGEST] ✓ Connected to {symbol}")
                    backoff = 1
                    retry_count = 0

                    async for msg in ws:
                        if not self._running:
                            break

                        data = json.loads(msg)

                        tick = Tick(
                            symbol=symbol,
                            event_time=data["E"],
                            price=float(data["p"]),
                            quantity=float(data["q"]),
                            receipt_time=time.time()
                        )

                        self.buffer.add(tick)

            except asyncio.CancelledError:
                logger.info(f"[INGEST] {symbol} task cancelled")
                break
            except websockets.exceptions.InvalidStatusCode as e:
                retry_count += 1
                if e.status_code == 400:
                    logger.error(f"[INGEST] ❌ {symbol.upper()} - Invalid symbol (HTTP 400)")
                    if retry_count >= max_retries:
                        logger.error(f"[INGEST] ❌ {symbol.upper()} - Giving up after {max_retries} failures")
                        break
                elif e.status_code == 451:
                    logger.error(f"[INGEST] ❌ {symbol.upper()} - Unavailable in your region (HTTP 451)")
                    break
                else:
                    logger.warning(f"[INGEST] {symbol} HTTP error: {e.status_code}")
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, 30)
            except Exception as e:
                if not self._running:
                    break
                retry_count += 1
                logger.warning(f"[INGEST] {symbol} error: {e}")
                if retry_count >= max_retries:
                    logger.error(f"[INGEST] ❌ {symbol.upper()} - Max retries exceeded")
                    break
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, 30)

    async def run(self):
        """Start ingestion for all symbols."""
        self._running = True
        self._loop = asyncio.get_event_loop()
        self._tasks = [
            asyncio.create_task(self._connect_symbol(sym))
            for sym in self.symbols
        ]
        
        try:
            await asyncio.gather(*self._tasks, return_exceptions=True)
        except asyncio.CancelledError:
            pass

    def stop(self):
        """Signal all tasks to stop."""
        logger.info("[INGEST] Stop requested...")
        self._running = False
        
        # Cancel all tasks
        for task in self._tasks:
            if not task.done():
                task.cancel()


class IngestionManager:
    """
    Manages ingestion lifecycle with support for symbol changes.
    
    This manager:
    - Runs ingestion in a separate thread
    - Supports stopping and restarting with new symbols
    - Cleans up resources properly
    """
    
    def __init__(self, buffer: TickBuffer):
        self.buffer = buffer
        self._ingestor: Optional[BinanceIngestor] = None
        self._thread: Optional[asyncio.AbstractEventLoop] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._current_symbols: List[str] = []
        self._lock = asyncio.Lock() if hasattr(asyncio, 'Lock') else None
        self._thread_lock = threading.Lock()

    def _run_async_loop(self, symbols: List[str]):
        """Run async event loop in thread."""
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        
        self._ingestor = BinanceIngestor(symbols=symbols, buffer=self.buffer)
        
        try:
            self._loop.run_until_complete(self._ingestor.run())
        except Exception as e:
            logger.error(f"[INGEST] Loop error: {e}")
        finally:
            self._loop.close()
            self._loop = None

    def start(self, symbols: List[str]):
        """Start ingestion for given symbols."""
        
        with self._thread_lock:
            self._current_symbols = [s.lower() for s in symbols]
            
            self._thread = threading.Thread(
                target=self._run_async_loop,
                args=(self._current_symbols,),
                daemon=True
            )
            self._thread.start()
            logger.info(f"[INGEST] Started for: {', '.join(s.upper() for s in self._current_symbols)}")

    def stop(self):
        """Stop current ingestion."""
        with self._thread_lock:
            if self._ingestor:
                self._ingestor.stop()
            
            if self._thread and self._thread.is_alive():
                self._thread.join(timeout=3.0)
            
            self._ingestor = None
            self._thread = None
            logger.info("[INGEST] Stopped")

    def restart(self, new_symbols: List[str]):
        """Stop current ingestion and restart with new symbols."""
        new_symbols = [s.lower() for s in new_symbols]
        
        # Check if symbols actually changed
        if set(new_symbols) == set(self._current_symbols):
            logger.info("[INGEST] Symbols unchanged, skipping restart")
            return False
        
        logger.info(f"[INGEST] Restarting: {self._current_symbols} → {new_symbols}")
        
        self.stop()
        
        # Give time for cleanup
        import time
        time.sleep(0.5)
        
        self.start(new_symbols)
        return True

    @property
    def current_symbols(self) -> List[str]:
        return self._current_symbols.copy()
