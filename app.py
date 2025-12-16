"""
Application entry point - Multi-Pair Support.

Responsibilities:
- Read control signals from UI (START/STOP via control.json)
- Start/stop ingestion for unique symbols
- Run analytics for EACH declared pair
- Emit per-pair analytics snapshots for UI consumption

Single-command execution:
    python app.py
"""

import time
import logging
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
STATE_PATH = DATA_DIR / "analytics_state.json"
HISTORY_PATH = DATA_DIR / "analytics_history.json"

# Defaults
DEFAULT_WINDOW_SIZE = 200

ANALYTICS_INTERVAL = 0.5     # seconds
STORAGE_FLUSH_INTERVAL = 1.0 # seconds
CONTROL_CHECK_INTERVAL = 0.5 # seconds
HISTORY_MAX_LEN = 500
MAX_PAIRS = 10  # Hard limit - never trust frontend


def validate_config(base_symbol: str, compare_symbols: list, pairs: list) -> tuple:
    """
    Backend validation - never trust frontend.
    Returns (valid: bool, error_message: str or None)
    """
    # 1. Base symbol must be a single non-empty string
    if not isinstance(base_symbol, str) or not base_symbol.strip():
        return False, "Base symbol must be a single non-empty string"
    
    # 2. Compare symbols must be a non-empty list
    if not compare_symbols or not isinstance(compare_symbols, list):
        return False, "Compare symbols must be a non-empty list"
    
    # 3. Base symbol cannot be in compare symbols
    base_lower = base_symbol.lower()
    if base_lower in [s.lower() for s in compare_symbols]:
        return False, "Base symbol cannot appear in compare symbols"
    
    # 4. Max pairs limit
    if len(pairs) > MAX_PAIRS:
        return False, f"Too many pairs ({len(pairs)} > {MAX_PAIRS})"
    
    # 5. Pairs must be valid tuples
    for p in pairs:
        if not isinstance(p, (list, tuple)) or len(p) != 2:
            return False, f"Invalid pair format: {p}"
    
    return True, None


def load_control():
    """Load control signal from UI."""
    if CONTROL_PATH.exists():
        try:
            with open(CONTROL_PATH) as f:
                return json.load(f)
        except Exception:
            pass
    return None


def persist_state(all_snapshots: dict, version: int):
    """Persist all pair snapshots atomically with Windows lock handling."""
    temp_path = STATE_PATH.with_suffix(".json.tmp")
    
    # Add version to state for UI change detection
    output = {
        "version": version,
        "pairs": all_snapshots
    }
    
    with open(temp_path, "w") as f:
        json.dump(output, f)
    
    # Windows file lock handling - retry up to 3 times
    for attempt in range(3):
        try:
            os.replace(temp_path, STATE_PATH)
            return
        except PermissionError:
            if attempt < 2:
                time.sleep(0.1)  # Wait for file lock release
            else:
                # Final attempt - just overwrite directly
                try:
                    with open(STATE_PATH, "w") as f:
                        json.dump(all_snapshots, f)
                except Exception:
                    pass  # Silently fail rather than crash


def load_history() -> dict:
    """Load per-pair history."""
    if HISTORY_PATH.exists():
        try:
            with open(HISTORY_PATH) as f:
                data = json.load(f)
                # Convert to deques
                return {
                    pair_id: deque(hist, maxlen=HISTORY_MAX_LEN)
                    for pair_id, hist in data.items()
                }
        except Exception:
            pass
    return {}


def save_history(all_history: dict):
    """Save per-pair history."""
    serializable = {
        pair_id: list(hist)
        for pair_id, hist in all_history.items()
    }
    with open(HISTORY_PATH, "w") as f:
        json.dump(serializable, f)


def clear_state():
    """Clear all state files."""
    if STATE_PATH.exists():
        STATE_PATH.unlink()
    with open(HISTORY_PATH, "w") as f:
        json.dump({}, f)


def main():
    logger.info("=" * 60)
    logger.info("🚀 QUANT ANALYTICS APP - Multi-Pair Support")
    logger.info("=" * 60)
    
    # Clear stale control file
    if CONTROL_PATH.exists():
        CONTROL_PATH.unlink()
        logger.info("🧹 Cleared stale control.json")
    
    logger.info("Open Streamlit UI and click START to begin.")
    logger.info("-" * 60)
    
    # State variables
    is_running = False
    buffer = None
    ingestion_manager = None
    analytics_engines = {}  # Dict of pair_id → AnalyticsEngine
    storage = DuckDBStorage()
    all_history = {}  # Dict of pair_id → deque
    pair_ever_ready = {}  # STICKY READINESS: once True, never reverts
    
    pairs = []
    unique_symbols = []
    window_size = DEFAULT_WINDOW_SIZE
    
    last_control_check = 0
    last_storage_flush = 0
    last_analytics_run = 0
    last_control_timestamp = 0
    snapshot_version = 0  # Increment on each state write for UI change detection
    
    MAX_PAIRS = 10  # Backend enforcement (don't trust frontend)
    
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
                        new_pairs = control.get("pairs", [])
                        new_symbols = control.get("unique_symbols", [])
                        new_window = control.get("window_size", DEFAULT_WINDOW_SIZE)
                        
                        # ====== BACKEND VALIDATION ======
                        # 1. Enforce max pairs (don't trust frontend)
                        if len(new_pairs) > MAX_PAIRS:
                            logger.error(f"❌ REJECTED: Too many pairs ({len(new_pairs)} > {MAX_PAIRS})")
                            continue
                        
                        # 2. Validate pairs are tuples of 2
                        valid_pairs = []
                        for p in new_pairs:
                            if isinstance(p, (list, tuple)) and len(p) == 2:
                                valid_pairs.append((p[0].lower(), p[1].lower()))
                            else:
                                logger.warning(f"⚠️ Invalid pair format: {p}")
                        
                        if not valid_pairs:
                            logger.error("❌ REJECTED: No valid pairs")
                            continue
                        
                        new_pairs = valid_pairs
                        
                        # 3. Validate symbols
                        valid_symbols = [s.lower().strip() for s in new_symbols if s.isalnum()]
                        if not valid_symbols:
                            logger.error("❌ REJECTED: No valid symbols")
                            continue
                        
                        new_symbols = valid_symbols
                        
                        logger.info("=" * 60)
                        logger.info("▶️  START SIGNAL RECEIVED")
                        logger.info(f"   Pairs: {len(new_pairs)}")
                        for p in new_pairs:
                            logger.info(f"     • {p[0].upper()}/{p[1].upper()}")
                        logger.info(f"   Symbols: {', '.join(s.upper() for s in new_symbols)}")
                        logger.info(f"   Window: {new_window}")
                        
                        # ====== EXPLICIT PIPELINE TEARDOWN ======
                        if is_running and ingestion_manager:
                            logger.info("🛑 Stopping existing pipeline...")
                            ingestion_manager.stop()
                            logger.info("   ✓ Ingestion stopped")
                            del buffer
                            logger.info("   ✓ Buffers cleared")
                            del ingestion_manager
                            logger.info("   ✓ Manager destroyed")
                            analytics_engines.clear()
                            logger.info("   ✓ Analytics engines cleared")
                            all_history.clear()
                            logger.info("   ✓ History cleared")
                        
                        # Update config
                        pairs = new_pairs
                        unique_symbols = [s.lower() for s in new_symbols]
                        window_size = new_window
                        
                        # Build new pipeline
                        logger.info("🔧 Building new pipeline...")
                        buffer = TickBuffer(maxlen=50_000)
                        ingestion_manager = IngestionManager(buffer=buffer)
                        
                        # Create analytics engine for EACH pair
                        analytics_engines = {}
                        all_history = {}
                        pair_ever_ready = {}  # Reset sticky readiness
                        for pair in pairs:
                            quote, base = pair[0].lower(), pair[1].lower()
                            pair_id = f"{quote}/{base}"
                            analytics_engines[pair_id] = AnalyticsEngine(
                                buffer=buffer,
                                symbol_x=base,
                                symbol_y=quote,
                                window_size=window_size
                            )
                            all_history[pair_id] = deque(maxlen=HISTORY_MAX_LEN)
                            logger.info(f"   📊 Created engine for {pair_id.upper()}")
                        
                        # Clear state
                        clear_state()
                        
                        # Start ingestion for all unique symbols
                        ingestion_manager.start(unique_symbols)
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
                            
                            total_history = sum(len(h) for h in all_history.values())
                            logger.info(f"✅ Pipeline stopped")
                            logger.info(f"📊 Final history: {total_history} total points across {len(pairs)} pairs")
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
            
            # ======== ANALYTICS (PER PAIR) ========
            if now - last_analytics_run >= ANALYTICS_INTERVAL:
                all_snapshots = {}
                any_computed = False
                buffering_pairs = []
                
                for pair_id, engine in sorted(analytics_engines.items()):  # Deterministic order
                    snapshot = engine.compute()
                    
                    if snapshot:
                        # Mark as ever-ready (sticky)
                        pair_ever_ready[pair_id] = True
                        snapshot["ready"] = True
                        all_snapshots[pair_id] = snapshot
                        all_history[pair_id].append(snapshot)
                        any_computed = True
                        
                        z = snapshot['zscore']
                        z_indicator = "🚨" if abs(z) >= 2.0 else "📊"
                        
                        logger.info(
                            f"{z_indicator} [{pair_id.upper()}] "
                            f"z={z:+.2f} | β={snapshot['beta']:.4f} | "
                            f"ρ={snapshot['correlation']:.2f}"
                        )
                    elif pair_ever_ready.get(pair_id, False):
                        # STICKY READINESS: Pair was ready before, use last snapshot
                        # This happens when analytics returns None temporarily (zero prices, etc)
                        if all_history[pair_id]:
                            last_good = dict(all_history[pair_id][-1])
                            last_good["ready"] = True  # Keep ready=True
                            all_snapshots[pair_id] = last_good
                            any_computed = True
                        # Don't log buffering - pair is still considered ready
                    else:
                        # Pair has NEVER been ready - show buffering status
                        # FROZEN SCHEMA: All keys present, analytics values as None
                        quote, base = pair_id.split("/")
                        x_count = min(buffer.get_count(base), window_size)
                        y_count = min(buffer.get_count(quote), window_size)
                        
                        all_snapshots[pair_id] = {
                            "ready": False,
                            "pair": pair_id,
                            # Analytics keys (all present, None when buffering)
                            "alpha": None,
                            "beta": None,
                            "spread": None,
                            "zscore": None,
                            "correlation": None,
                            # Buffering status
                            "x_symbol": base.upper(),
                            "y_symbol": quote.upper(),
                            "x_count": x_count,
                            "y_count": y_count,
                            "window": window_size,
                            "x_pct": int(x_count / window_size * 100),
                            "y_pct": int(y_count / window_size * 100),
                            "timestamp": time.time()
                        }
                        buffering_pairs.append(f"{pair_id.upper()} ({min(x_count, y_count)}/{window_size})")
                
                # Always persist state (including buffering status)
                snapshot_version += 1
                persist_state(all_snapshots, snapshot_version)
                if any_computed:
                    save_history(all_history)
                
                if buffering_pairs:
                    logger.info(f"⏳ Buffering: {', '.join(buffering_pairs)}")
                
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
