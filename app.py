"""
Application entry point.

Responsibilities:
- Read control signals from UI (START/STOP via control.json)
- Start/stop ingestion based on control signals
- Run analytics on a fixed cadence
- Emit analytics snapshots for UI consumption

Single-command execution:
    python app.py
"""

import time
import logging
import csv
import json
import os
from pathlib import Path
from collections import deque

from utils import TickBuffer
from ingest import IngestionManager
from storage import DuckDBStorage
from analytics import AnalyticsEngine


# ---------------- Logging ----------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger("app")


# ---------------- Configuration ----------------
DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)

CONTROL_PATH = DATA_DIR / "control.json"
STATE_PATH = DATA_DIR / "analytics_state.csv"
HISTORY_PATH = DATA_DIR / "analytics_history.json"

# Defaults
DEFAULT_SYMBOL_X = "btcusdt"
DEFAULT_SYMBOL_Y = "ethusdt"
DEFAULT_WINDOW_SIZE = 200

ANALYTICS_INTERVAL = 0.5     # seconds
STORAGE_FLUSH_INTERVAL = 1.0 # seconds
CONTROL_CHECK_INTERVAL = 0.5 # seconds (fast response to UI)
HISTORY_MAX_LEN = 500


def load_control():
    """Load control signal from UI."""
    if CONTROL_PATH.exists():
        try:
            with open(CONTROL_PATH) as f:
                return json.load(f)
        except Exception:
            pass
    return None


def clear_control():
    """Clear control file after processing."""
    if CONTROL_PATH.exists():
        try:
            CONTROL_PATH.unlink()
        except Exception:
            pass


def persist_snapshot(snapshot: dict):
    """Persist analytics snapshot atomically."""
    temp_path = STATE_PATH.with_suffix(".csv.tmp")
    
    with open(temp_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=snapshot.keys())
        writer.writeheader()
        writer.writerow(snapshot)
    
    os.replace(temp_path, STATE_PATH)


def load_history():
    """Load existing history."""
    if HISTORY_PATH.exists():
        try:
            with open(HISTORY_PATH) as f:
                return deque(json.load(f), maxlen=HISTORY_MAX_LEN)
        except Exception:
            pass
    return deque(maxlen=HISTORY_MAX_LEN)


def save_history(history: deque):
    """Save history to disk."""
    with open(HISTORY_PATH, "w") as f:
        json.dump(list(history), f)


def clear_state():
    """Clear all state files."""
    if STATE_PATH.exists():
        STATE_PATH.unlink()
    with open(HISTORY_PATH, "w") as f:
        json.dump([], f)


def main():
    logger.info("=" * 60)
    logger.info("🚀 QUANT ANALYTICS APP - Waiting for START signal")
    logger.info("=" * 60)
    
    # CRITICAL: Clear any stale control file from previous session
    # This prevents auto-starting from old START signals
    if CONTROL_PATH.exists():
        CONTROL_PATH.unlink()
        logger.info("🧹 Cleared stale control.json")
    
    logger.info("Open Streamlit UI and click START to begin.")
    logger.info("-" * 60)
    
    # State variables
    is_running = False
    buffer = None
    ingestion_manager = None
    analytics = None
    storage = DuckDBStorage()
    history = deque(maxlen=HISTORY_MAX_LEN)
    
    symbol_x = DEFAULT_SYMBOL_X
    symbol_y = DEFAULT_SYMBOL_Y
    window_size = DEFAULT_WINDOW_SIZE
    
    last_control_check = 0
    last_storage_flush = 0
    last_analytics_run = 0
    last_control_timestamp = 0
    
    try:
        while True:
            now = time.time()
            
            # ======== CONTROL CHECK ========
            if now - last_control_check >= CONTROL_CHECK_INTERVAL:
                control = load_control()
                
                if control and control.get("timestamp", 0) > last_control_timestamp:
                    action = control.get("action", "").upper()
                    last_control_timestamp = control.get("timestamp", 0)
                    
                    # -------- START ACTION --------
                    if action == "START":
                        new_x = control.get("symbol_x", DEFAULT_SYMBOL_X).lower()
                        new_y = control.get("symbol_y", DEFAULT_SYMBOL_Y).lower()
                        new_window = control.get("window_size", DEFAULT_WINDOW_SIZE)
                        
                        logger.info("=" * 60)
                        logger.info("▶️  START SIGNAL RECEIVED")
                        logger.info(f"   Pair: {new_y.upper()}/{new_x.upper()}")
                        logger.info(f"   Window: {new_window}")
                        
                        # Stop existing pipeline if running
                        if is_running and ingestion_manager:
                            logger.info("🛑 Stopping existing pipeline...")
                            ingestion_manager.stop()
                            del buffer
                            del analytics
                            del ingestion_manager
                        
                        # Update config
                        symbol_x, symbol_y, window_size = new_x, new_y, new_window
                        
                        # Build new pipeline
                        logger.info("🔧 Building new pipeline...")
                        buffer = TickBuffer(maxlen=50_000)
                        ingestion_manager = IngestionManager(buffer=buffer)
                        analytics = AnalyticsEngine(
                            buffer=buffer,
                            symbol_x=symbol_x,
                            symbol_y=symbol_y,
                            window_size=window_size
                        )
                        
                        # Clear state
                        history.clear()
                        clear_state()
                        
                        # Start ingestion
                        ingestion_manager.start([symbol_x, symbol_y])
                        is_running = True
                        
                        logger.info("✅ Pipeline started successfully")
                        logger.info("=" * 60)
                    
                    # -------- STOP ACTION --------
                    elif action == "STOP":
                        if is_running and ingestion_manager:
                            logger.info("=" * 60)
                            logger.info("⏹️  STOP SIGNAL RECEIVED")
                            
                            ingestion_manager.stop()
                            is_running = False
                            
                            logger.info("✅ Pipeline stopped")
                            logger.info(f"📊 Final history: {len(history)} points")
                            logger.info("=" * 60)
                            logger.info("Waiting for next START signal...")
                
                last_control_check = now
            
            # ======== ONLY RUN IF STARTED ========
            if not is_running:
                time.sleep(0.1)
                continue
            
            # ======== STORAGE FLUSH ========
            if now - last_storage_flush >= STORAGE_FLUSH_INTERVAL:
                total_ticks = 0
                buffer_snapshot = list(buffer.buffers.items())
                for sym, ticks in buffer_snapshot:
                    tick_list = list(ticks)
                    if tick_list:
                        storage.insert_ticks(tick_list)
                        total_ticks += len(tick_list)
                
                if total_ticks > 0:
                    logger.debug(f"💾 Flushed {total_ticks} ticks")
                
                last_storage_flush = now
            
            # ======== ANALYTICS ========
            if now - last_analytics_run >= ANALYTICS_INTERVAL:
                snapshot = analytics.compute()
                
                if snapshot:
                    persist_snapshot(snapshot)
                    
                    history.append(snapshot)
                    save_history(history)
                    
                    z = snapshot['zscore']
                    z_indicator = "🚨" if abs(z) >= 2.0 else "📊"
                    
                    logger.info(
                        f"{z_indicator} [{snapshot['pair']}] "
                        f"z={z:+.2f} | β={snapshot['beta']:.4f} | "
                        f"ρ={snapshot['correlation']:.2f}"
                    )
                else:
                    # Get actual buffer lengths (capped by deque maxlen)
                    x_len = min(buffer.get_count(symbol_x), window_size)
                    y_len = min(buffer.get_count(symbol_y), window_size)
                    
                    if x_len == 0 and y_len == 0:
                        logger.info(f"⏳ Waiting for data... {symbol_x.upper()}, {symbol_y.upper()}")
                    else:
                        # Show status for each symbol
                        x_status = "READY" if x_len >= window_size else f"{x_len}/{window_size} ({int(x_len/window_size*100)}%)"
                        y_status = "READY" if y_len >= window_size else f"{y_len}/{window_size} ({int(y_len/window_size*100)}%)"
                        
                        logger.info(
                            f"⏳ Buffering: {symbol_x.upper()} {x_status} | "
                            f"{symbol_y.upper()} {y_status}"
                        )
                
                last_analytics_run = now
            
            time.sleep(0.05)
    
    except KeyboardInterrupt:
        logger.info("")
        logger.info("=" * 60)
        logger.info("🛑 Shutting down...")
        if is_running and ingestion_manager:
            ingestion_manager.stop()
        logger.info("=" * 60)


if __name__ == "__main__":
    main()
