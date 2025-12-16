"""
Streamlit Cloud Deployment Entry Point (Single-File Backend + Frontend)

This script integrates the backend logic (ingestion -> analytics) directly into the Streamlit process
using a background thread. This allows the application to run on Streamlit Cloud without a separate backend service.

Architecture:
- BackendService (Singleton, @st.cache_resource): Runs IngestionManager and AnalyticsEngine in a background thread.
- Frontend: Renders UI using data fetched directly from BackendService state (no file I/O).
"""

import time
import json
import datetime
import numpy as np
import streamlit as st
import plotly.graph_objects as go
import pandas as pd
from pathlib import Path
import threading
import logging
from collections import deque

# Backend Imports
from utils import TickBuffer
from ingest import IngestionManager
from analytics import AnalyticsEngine, run_adf_test, compute_half_life

# ---------------- LOGGING SETUP ----------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("deploy")

# ---------------- CONSTANTS ----------------
HISTORY_MAX_LEN = 500
ANALYTICS_INTERVAL = 0.5
MAX_PAIRS = 10

AVAILABLE_SYMBOLS = [
    "btcusdt", "ethusdt", "bnbusdt", "xrpusdt", "adausdt",
    "dogeusdt", "solusdt", "dotusdt", "maticusdt", "ltcusdt",
    "avaxusdt", "linkusdt", "atomusdt", "uniusdt", "etcusdt"
]

# ---------------- BACKEND SERVICE (SINGLETON) ----------------
@st.cache_resource
class BackendService:
    def __init__(self):
        self.lock = threading.Lock()
        self.running = False
        self.thread = None
        
        # Pipeline Components
        self.buffer = TickBuffer(maxlen=50_000)
        self.ingestion_manager = IngestionManager(buffer=self.buffer)
        self.analytics_engines = {}
        
        # State Storage
        self.snapshot_version = 0
        self.state_snapshot = {}  # Corresponds to analytics_state.json
        self.history = {}         # Corresponds to analytics_history.json
        self.pair_ever_ready = {}
        
        # Configuration
        self.pairs = []
        self.window_size = 200
        self.unique_symbols = []

    def start(self, pairs, window_size):
        """Start the backend pipeline with new configuration."""
        with self.lock:
            # Stop existing
            if self.running:
                self.stop()
            
            self.pairs = pairs
            self.window_size = window_size
            self.unique_symbols = list(set([p[0] for p in pairs] + [p[1] for p in pairs]))
            
            # Reset State
            self.snapshot_version = 0
            self.state_snapshot = {}
            self.history = {f"{p[0]}/{p[1]}": deque(maxlen=HISTORY_MAX_LEN) for p in pairs}
            self.pair_ever_ready = {}
            self.analytics_engines = {}
            self.buffer = TickBuffer(maxlen=50_000)
            self.ingestion_manager = IngestionManager(buffer=self.buffer)
            
            # Init Engines
            for pair in self.pairs:
                quote, base = pair[0].lower(), pair[1].lower()
                pair_id = f"{quote}/{base}"
                self.analytics_engines[pair_id] = AnalyticsEngine(
                    buffer=self.buffer,
                    symbol_x=base,
                    symbol_y=quote,
                    window_size=self.window_size
                )
            
            # Start Ingestion
            self.ingestion_manager.start(self.unique_symbols)
            
            # Start Analytics Loop Thread
            self.running = True
            self.thread = threading.Thread(target=self._run_analytics_loop, daemon=True)
            self.thread.start()
            logger.info("✅ BackendService Started")

    def stop(self):
        """Stop the backend pipeline."""
        with self.lock:
            self.running = False
            if self.ingestion_manager:
                self.ingestion_manager.stop()
            logger.info("🛑 BackendService Stopped")

    def _run_analytics_loop(self):
        """Main loop running in background thread."""
        logger.info("🚀 Analytics Loop Thread Started")
        last_run = 0
        
        while self.running:
            now = time.time()
            
            # Analytics Throttle
            if now - last_run < ANALYTICS_INTERVAL:
                time.sleep(0.05)
                continue
            
            snapshot_update = {}
            any_computed = False
            
            # 1. Compute Analytics
            # Access self.analytics_engines under lock if structure changes, but here concurrent compute is safe-ish
            # Strictly speaking we should lock, but buffer is thread-safe.
            
            for pair_id, engine in self.analytics_engines.items():
                res = engine.compute()
                
                if res:
                    self.pair_ever_ready[pair_id] = True
                    res["ready"] = True
                    snapshot_update[pair_id] = res
                    
                    with self.lock:
                        self.history[pair_id].append(res)
                    any_computed = True
                
                elif self.pair_ever_ready.get(pair_id, False):
                    # Sticky readiness logic
                    with self.lock:
                        if self.history[pair_id]:
                            last_good = dict(self.history[pair_id][-1])
                            last_good["ready"] = True
                            snapshot_update[pair_id] = last_good
                
                else:
                    # Buffering logic
                    quote, base = pair_id.split("/")
                    x_count = min(self.buffer.get_count(base), self.window_size)
                    y_count = min(self.buffer.get_count(quote), self.window_size)
                    
                    snapshot_update[pair_id] = {
                        "ready": False,
                        "pair": pair_id,
                        "x_symbol": base.upper(),
                        "y_symbol": quote.upper(),
                        "x_count": x_count,
                        "y_count": y_count,
                        "window": self.window_size,
                        "x_pct": int(x_count / self.window_size * 100),
                        "y_pct": int(y_count / self.window_size * 100),
                        "timestamp": time.time(),
                        # Placeholders
                        "spread": None, "zscore": None, "beta": None, "correlation": None
                    }
            
            # 2. Update Shared State
            with self.lock:
                self.state_snapshot = snapshot_update
                self.snapshot_version += 1
            
            last_run = now

    def get_snapshot(self):
        """Thread-safe access to latest snapshot."""
        with self.lock:
            return self.snapshot_version, self.state_snapshot.copy()

    def get_history(self):
        """Thread-safe access to history."""
        with self.lock:
            # Convert deques to lists for frontend
            return {pid: list(hist) for pid, hist in self.history.items()}


# Initialize Backend Singleton
backend = BackendService()


# ---------------- FRONTEND LOGIC (Copied from streamlit_app.py) ----------------

# Page Config
st.set_page_config(
    page_title="Quant Analytics Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .stMetric {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        padding: 12px;
        border-radius: 10px;
        border: 1px solid #0f3460;
    }
    .stMetric label { color: #e94560 !important; font-size: 0.85rem; }
    .stMetric [data-testid="stMetricValue"] { font-size: 1.5rem; }
    .main-header {
        background: linear-gradient(135deg, #0f3460 0%, #16213e 100%);
        padding: 25px;
        border-radius: 12px;
        margin-bottom: 20px;
        border: 1px solid #e94560;
    }
    .status-running {
        background: linear-gradient(90deg, #2d6a4f, #40916c);
        padding: 10px 15px;
        border-radius: 8px;
        margin: 10px 0;
    }
    .status-stopped {
        background: linear-gradient(90deg, #6c757d, #495057);
        padding: 10px 15px;
        border-radius: 8px;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

# Session State Init
if "running" not in st.session_state:
    st.session_state.running = False
if "pairs" not in st.session_state:
    st.session_state.pairs = []
if "history" not in st.session_state:
    st.session_state.history = {}
if "diagnostics" not in st.session_state:
    st.session_state.diagnostics = {}
if "pair_ready_once" not in st.session_state:
    st.session_state.pair_ready_once = {}
if "render_ready" not in st.session_state:
    st.session_state.render_ready = set()
if "cached_df" not in st.session_state:
    st.session_state.cached_df = {}


# ================== SIDEBAR ==================
with st.sidebar:
    st.header("⚙️ Configuration")
    
    is_running = st.session_state.running
    
    st.subheader("📊 Symbol Selection")
    
    base_symbol = st.selectbox("Base Symbol (X)", AVAILABLE_SYMBOLS, index=0, disabled=is_running)
    
    available_compare = [s for s in AVAILABLE_SYMBOLS if s != base_symbol]
    compare_symbols = st.multiselect("Compare Symbols (Y)", available_compare, default=[available_compare[0]] if available_compare else [], disabled=is_running)
    
    custom_base = st.text_input("Custom Base Symbol", placeholder="e.g., arbusdt", disabled=is_running)
    custom_compare = st.text_input("Custom Compare Symbols", placeholder="e.g., opusdt", disabled=is_running)
    
    # Validation Logic
    def clean_symbol(s):
        s = s.lower().strip()
        return s if s and s.isalnum() else None

    final_base = clean_symbol(custom_base) or base_symbol
    
    all_compare = list(compare_symbols)
    if custom_compare:
        custom_list = [clean_symbol(s) for s in custom_compare.split(",")]
        all_compare.extend([s for s in custom_list if s and s != final_base])
    
    all_compare = list(dict.fromkeys(all_compare))
    all_compare = [s for s in all_compare if s and s != final_base]
    
    pairs = [(quote.lower(), final_base.lower()) for quote in all_compare if quote]
    
    st.divider()
    
    st.subheader("📈 Parameters")
    window_size = st.slider("Rolling Window", 50, 500, 200, 25, disabled=is_running)
    z_threshold = st.slider("Z-Score Alert Threshold", 1.0, 3.0, 2.0, 0.1)
    refresh_rate = st.selectbox("Refresh Rate", [0.5, 1.0, 2.0, 5.0], index=1, format_func=lambda x: f"{x}s")
    
    st.divider()
    
    # Start/Stop Buttons
    st.subheader("🎮 Control")
    col1, col2 = st.columns(2)
    start_clicked = col1.button("▶️ Start", disabled=is_running or not pairs, use_container_width=True, type="primary")
    stop_clicked = col2.button("⏹️ Stop", disabled=not is_running, use_container_width=True)
    
    # Action Handling
    if start_clicked:
        st.session_state.running = True
        st.session_state.pairs = pairs
        st.session_state.history = {f"{p[0]}/{p[1]}": [] for p in pairs}
        st.session_state.render_ready = set()
        st.session_state.cached_df = {}
        
        # START BACKEND
        backend.start(pairs, window_size)
        st.rerun()
        
    if stop_clicked:
        st.session_state.running = False
        # STOP BACKEND
        backend.stop()
        st.rerun()

    # Status
    st.divider()
    if is_running:
        st.markdown(f'<div class="status-running">🟢 <strong>RUNNING</strong><br><small>{len(st.session_state.pairs)} pairs active</small></div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="status-stopped">⚫ <strong>STOPPED</strong><br><small>Configure and press Start</small></div>', unsafe_allow_html=True)


# ================== MAIN CONTENT ==================
st.markdown("""
<div class="main-header">
    <h1 style="margin:0; color: #e94560;">Quant Analytics Dashboard</h1>
    <p style="margin:5px 0 0 0; color: #a0a0a0;">Real-Time Statistical Arbitrage Monitor • Multi-Pair Support</p>
</div>
""", unsafe_allow_html=True)

if not st.session_state.running:
    st.info("👈 **Configure symbols and press Start** to begin live analytics.")
    st.stop()

# ================== DATA SYNC (VS BACKEND) ==================

# 1. Fetch State
state_version, state = backend.get_snapshot()
history_snapshot = backend.get_history()

# 2. Version Lock Skip Logic
if "last_version" not in st.session_state:
    st.session_state.last_version = -1

skip_render = state_version is not None and state_version == st.session_state.last_version

if state_version is not None:
    st.session_state.last_version = state_version

if skip_render:
    # Never sleep if data is ready to keep UI snappy
    any_ready = any(s.get("ready", False) for s in state.values()) if state else False
    if not any_ready:
        time.sleep(refresh_rate)
    st.rerun()

# 3. Apply History to Session State (Monotonic Merge)
for pair_id, new_hist in history_snapshot.items():
    if pair_id not in st.session_state.history:
        st.session_state.history[pair_id] = []
        
    # Simple replace is safer for local-memory integration
    # (Since we get full list from backend memory anyway)
    st.session_state.history[pair_id] = new_hist

# 4. Diagnostics Processing
diag_store = st.session_state.setdefault("diagnostics", {})
diag_intents = {k: v for k, v in st.session_state.items() if k.startswith("diag_intent_") and v}

for intent_key in diag_intents:
    pair_id = intent_key.replace("diag_intent_", "")
    h = st.session_state.history.get(pair_id, [])
    if len(h) >= 30:
        arr = np.array([x.get("spread") or 0 for x in h[-500:]], dtype=float)
        arr = arr[np.isfinite(arr)]
        if len(arr) >= 30:
            diag_store[pair_id] = {
                "adf": run_adf_test(arr),
                "halflife": compute_half_life(arr),
                "computed_at": time.time()
            }
    st.session_state[intent_key] = False


# ================== RENDERING ==================

if not state:
    st.warning("⏳ **Waiting for analytics data...**")
    st.caption("Backend is buffering ticks for configured pairs.")
    st.progress(0.0, text="Starting ingestion...")
    time.sleep(refresh_rate)
    st.rerun()

active_pairs = len(state)
st.success(f"🟢 **LIVE** • {active_pairs} pair{'s' if active_pairs > 1 else ''} • Window: {window_size} • Refresh: {refresh_rate}s")

# Render Tabs
sorted_pair_ids = sorted(state.keys())
if len(sorted_pair_ids) > 0:
    tabs = st.tabs(sorted_pair_ids)
    
    for i, pair_id in enumerate(sorted_pair_ids):
        with tabs[i]:
            snapshot = state[pair_id]
            pair_history = st.session_state.history.get(pair_id, [])
            is_ready = snapshot.get("ready", False)
            
            # --- RENDER ANALYTICS FUNCTION INLINE ---
            
            # 1. Header
            if is_ready:
                st.session_state.pair_ready_once[pair_id] = True
            
            display_ready = st.session_state.pair_ready_once.get(pair_id, False)
            
            if display_ready:
                z = snapshot.get('zscore') or 0
                b = snapshot.get('beta') or 0
                c = snapshot.get('correlation') or 0
                st.markdown(f"""
                <div style="background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%); padding: 15px; border-radius: 10px; border-left: 4px solid #40916c; margin-bottom: 10px;">
                    <h3 style="margin:0; color: #e94560;">{pair_id.upper()}</h3>
                    <p style="margin:0; color:#a0a0a0;">Z={z:+.2f} | β={b:.4f} | ρ={c:.2f}</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.info(f"⏳ **{pair_id.upper()} — Buffering...**")
            
            # 2. Metrics
            c1, c2, c3, c4 = st.columns(4)
            c1.metric(f"β ({pair_id})", f"{snapshot.get('beta',0):.4f}" if is_ready else "—")
            delta = "🚨 ALERT" if abs(snapshot.get('zscore',0)) >= z_threshold else "✅ OK"
            c2.metric(f"Z ({pair_id})", f"{snapshot.get('zscore',0):.2f}" if is_ready else "—", delta=delta if is_ready else None)
            c3.metric(f"Spread", f"{snapshot.get('spread',0):.6f}" if is_ready else "—")
            c4.metric(f"ρ ({pair_id})", f"{snapshot.get('correlation',0):.2%}" if is_ready else "—")
            
            # 3. Diagnostics Button
            d1, d2 = st.columns([1, 3])
            with d1:
                if st.button(f"Run Diagnostics", key=f"diag_{pair_id}", disabled=not is_ready):
                    st.session_state[f"diag_intent_{pair_id}"] = True
            
            with d2:
                if pair_id in diag_store:
                    res = diag_store[pair_id]
                    st.caption(f"ADF p={res['adf'].get('p_value',0):.4f} | HL={res['halflife']:.1f}")

            # 4. Charts (Optimized DataFrame)
            cache_key = f"df_{pair_id}_{len(pair_history)}"
            if cache_key in st.session_state.cached_df:
                df = st.session_state.cached_df[cache_key]
            else:
                if is_ready and len(pair_history) > 1:
                    df = pd.DataFrame(pair_history)
                    df["t"] = range(len(df))
                else:
                    df = pd.DataFrame()
                
                if len(st.session_state.cached_df) > 50:
                    st.session_state.cached_df = {}
                st.session_state.cached_df[cache_key] = df
            
            if not df.empty:
                r1_1, r1_2 = st.columns(2)
                with r1_1:
                    fig_spread = go.Figure()
                    fig_spread.add_trace(go.Scatter(x=df["t"], y=df["spread"], mode="lines", line=dict(color="#00d4ff")))
                    fig_spread.update_layout(title=f"Spread", height=250, margin=dict(l=10,r=10,t=30,b=30), template="plotly_dark")
                    st.plotly_chart(fig_spread, use_container_width=True, key=f"spread_{pair_id}")
                
                with r1_2:
                    fig_z = go.Figure()
                    fig_z.add_trace(go.Scatter(x=df["t"], y=df["zscore"], mode="lines", line=dict(color="#e94560")))
                    fig_z.add_hline(y=z_threshold, line_dash="dash", line_color="yellow")
                    fig_z.add_hline(y=-z_threshold, line_dash="dash", line_color="yellow")
                    fig_z.update_layout(title=f"Z-Score", height=250, margin=dict(l=10,r=10,t=30,b=30), template="plotly_dark")
                    st.plotly_chart(fig_z, use_container_width=True, key=f"zscore_{pair_id}")

# ================== EXPORTS ==================
st.divider()
st.subheader("📤 Exports & Logs")
ec1, ec2, ec3 = st.columns(3)

with ec1:
    if state:
        st.download_button("Download State (JSON)", json.dumps(state, indent=2), "state.json", "application/json", key="dl_state")

with ec2:
    if any(len(h) > 0 for h in st.session_state.history.values()):
        rows = []
        for pid, h in st.session_state.history.items():
            for x in h:
                rows.append({**x, "pair": pid})
        if rows:
            st.download_button("Download History (CSV)", pd.DataFrame(rows).to_csv(index=False), "history.csv", "text/csv", key="dl_csv")

with ec3:
    total_pts = sum(len(h) for h in st.session_state.history.values())
    st.metric("Total History", f"{total_pts} pts", help="In-memory points")


# ================== FOOTER ==================
st.markdown("---")
st.caption(f"🔒 Single-File Deployment Mode | {active_pairs} pairs | Window: {window_size} | Refresh: {refresh_rate}s")
st.caption("ℹ️ Backend running in background thread via @st.cache_resource")

# Auto-refresh
time.sleep(refresh_rate)
st.rerun()
