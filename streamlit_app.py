"""
Streamlit dashboard for real-time quantitative analytics.

Multi-Pair Support:
- Base Symbol: Single symbol (X in regression)
- Compare Symbols: Multiple symbols (Y in regression)
- Each pair gets isolated analytics, history, and charts
- Tabbed view: one tab per pair

UX Design:
- Start button: locks config, starts backend, shows charts
- Stop button: stops backend, hides charts, unlocks config
- Single running flag controls all visibility
"""

import time
import json
import datetime
import numpy as np
import streamlit as st
import plotly.graph_objects as go
import pandas as pd
from pathlib import Path

from analytics import run_adf_test, compute_half_life


# ---------------- Page Config ----------------
st.set_page_config(
    page_title="Quant Analytics Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------- Session State Initialization ----------------
if "running" not in st.session_state:
    st.session_state.running = False
if "pairs" not in st.session_state:
    st.session_state.pairs = []
if "history" not in st.session_state:
    st.session_state.history = {}  # Dict of pair_id → list of snapshots
if "diagnostics" not in st.session_state:
    st.session_state.diagnostics = {}  # Dict of pair_id → {adf, half_life}
if "pair_ready_once" not in st.session_state:
    st.session_state.pair_ready_once = {}  # Sticky ready: once True, never reverts in UI
if "render_ready" not in st.session_state:
    st.session_state.render_ready = set()  # UI-side latch for instant chart render
if "cached_df" not in st.session_state:
    st.session_state.cached_df = {}  # Cache: (pair_id, version) → DataFrame



# ---------------- Constants ----------------
DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)

CONTROL_PATH = DATA_DIR / "control.json"
STATE_PATH = DATA_DIR / "analytics_state.json"
HISTORY_PATH = DATA_DIR / "analytics_history.json"

MAX_PAIRS = 10

AVAILABLE_SYMBOLS = [
    "btcusdt", "ethusdt", "bnbusdt", "xrpusdt", "adausdt",
    "dogeusdt", "solusdt", "dotusdt", "maticusdt", "ltcusdt",
    "avaxusdt", "linkusdt", "atomusdt", "uniusdt", "etcusdt"
]


# ---------------- Custom CSS ----------------
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


# ================== SIDEBAR - CONFIGURATION ==================
with st.sidebar:
    st.header("⚙️ Configuration")
    
    is_running = st.session_state.running
    
    # ========== SYMBOL SELECTION ==========
    st.subheader("📊 Symbol Selection")
    
    # Base Symbol (X) - Single select
    base_symbol = st.selectbox(
        "Base Symbol (X)",
        options=AVAILABLE_SYMBOLS,
        index=0,
        disabled=is_running,
        help="Independent variable in regression"
    )
    
    # Compare Symbols (Y) - Multi-select
    available_compare = [s for s in AVAILABLE_SYMBOLS if s != base_symbol]
    compare_symbols = st.multiselect(
        "Compare Symbols (Y)",
        options=available_compare,
        default=[available_compare[0]] if available_compare else [],
        disabled=is_running,
        help="Dependent variables - one pair per symbol"
    )
    
    # Custom symbols
    custom_base = st.text_input(
        "Custom Base Symbol",
        placeholder="e.g., arbusdt",
        disabled=is_running,
        help="Single symbol only"
    )
    
    custom_compare = st.text_input(
        "Custom Compare Symbols",
        placeholder="e.g., opusdt, wldusdt",
        disabled=is_running,
        help="Comma-separated symbols"
    )
    
    # Clean and validate symbols
    def clean_symbol(s):
        s = s.lower().strip()
        if not s or not s.isalnum():
            return None
        return s
    
    # Process base symbol - use custom if valid, else use dropdown
    cleaned_custom_base = clean_symbol(custom_base) if custom_base else None
    final_base = cleaned_custom_base if cleaned_custom_base else base_symbol
    
    # Process compare symbols (combine dropdown + custom)
    all_compare = list(compare_symbols)
    if custom_compare:
        custom_list = [clean_symbol(s) for s in custom_compare.split(",")]
        custom_list = [s for s in custom_list if s and s != final_base]
        all_compare.extend(custom_list)
    
    # Remove duplicates and base symbol
    all_compare = list(dict.fromkeys(all_compare))  # Preserve order, remove dupes
    all_compare = [s for s in all_compare if s and s != final_base]
    
    # Generate pairs (ensure no None values)
    pairs = [(quote.lower(), final_base.lower()) for quote in all_compare if quote]
    
    st.divider()
    
    # ========== PARAMETERS ==========
    st.subheader("📈 Parameters")
    
    window_size = st.slider(
        "Rolling Window",
        min_value=50,
        max_value=500,
        value=200,
        step=25,
        disabled=is_running
    )
    
    z_threshold = st.slider(
        "Z-Score Alert Threshold",
        min_value=1.0,
        max_value=3.0,
        value=2.0,
        step=0.1
    )
    
    refresh_rate = st.selectbox(
        "Refresh Rate",
        options=[0.5, 1.0, 2.0, 5.0],
        index=1,
        format_func=lambda x: f"{x}s"
    )
    
    st.divider()
    
    # ========== VALIDATION (FAIL-FAST) ==========
    validation_errors = []
    
    # 1. Base symbol must exist (enforced by selectbox but check anyway)
    if not final_base:
        validation_errors.append("You must select exactly ONE base symbol")
    
    # 2. At least one compare symbol
    if len(all_compare) == 0:
        validation_errors.append("Select at least one compare symbol")
    
    # 3. Base symbol not in compare list (should be filtered but double-check)
    if final_base in all_compare:
        validation_errors.append("Base symbol cannot appear in compare symbols")
    
    # 4. Max pairs limit
    if len(pairs) > MAX_PAIRS:
        validation_errors.append(f"Maximum {MAX_PAIRS} compare symbols allowed (got {len(pairs)})")
    
    # 5. Show all errors
    if validation_errors:
        for error in validation_errors:
            st.error(f"❌ {error}")
    
    # ========== START / STOP BUTTONS ==========
    st.subheader("🎮 Control")
    
    # ========== CALLBACKS ==========
    def start_callback():
        if len(validation_errors) == 0:
            # Get unique symbols for ingestion
            unique_symbols = list(set([final_base] + [p[0] for p in pairs]))
            
            # Reset state
            st.session_state.running = True
            st.session_state.pairs = pairs
            st.session_state.history = {f"{p[0]}/{p[1]}": [] for p in pairs}
            
            # Clear old state files
            if STATE_PATH.exists():
                STATE_PATH.unlink()
            with open(HISTORY_PATH, "w") as f:
                json.dump({}, f)
                
            # Write control signal
            with open(CONTROL_PATH, "w") as f:
                json.dump({
                    "action": "START",
                    "base": final_base,
                    "compare": all_compare,
                    "pairs": pairs,
                    "unique_symbols": unique_symbols,
                    "window_size": window_size,
                    "timestamp": time.time()
                }, f)

    def stop_callback():
        st.session_state.running = False
        with open(CONTROL_PATH, "w") as f:
            json.dump({
                "action": "STOP",
                "timestamp": time.time()
            }, f)

    col1, col2 = st.columns(2)
    
    with col1:
        st.button(
            "▶️ Start",
            disabled=is_running or len(validation_errors) > 0,
            use_container_width=True,
            type="primary",
            on_click=start_callback,
            key="start_btn"
        )
    
    with col2:
        st.button(
            "⏹️ Stop",
            disabled=not is_running,
            use_container_width=True,
            on_click=stop_callback,
            key="stop_btn"
        )
    
    # ========== STATUS INDICATOR ==========
    st.divider()
    if is_running:
        pair_count = len(st.session_state.pairs)
        st.markdown(f"""
        <div class="status-running">
        🟢 <strong>RUNNING</strong><br>
        <small>{pair_count} pair{'s' if pair_count > 1 else ''} active</small>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="status-stopped">
        ⚫ <strong>STOPPED</strong><br>
        <small>Configure and press Start</small>
        </div>
        """, unsafe_allow_html=True)
    
    # Show configured pairs
    if pairs and not is_running:
        st.caption(f"**Configured pairs ({len(pairs)}):**")
        for p in pairs[:5]:
            st.caption(f"  • {p[0].upper()}/{p[1].upper()}")
        if len(pairs) > 5:
            st.caption(f"  ... and {len(pairs) - 5} more")


# ================== MAIN CONTENT ==================

# Header
st.markdown("""
<div class="main-header">
    <h1 style="margin:0; color: #e94560;">Quant Analytics Dashboard</h1>
    <p style="margin:5px 0 0 0; color: #a0a0a0;">Real-Time Statistical Arbitrage Monitor • Multi-Pair Support</p>
</div>
""", unsafe_allow_html=True)


# ========== GATE: Show content only when running ==========
if not st.session_state.running:
    st.info("👈 **Configure symbols and press Start** to begin live analytics.")
    st.caption("Select a base symbol and one or more compare symbols, then click Start.")
    
    # Show pair preview
    if pairs:
        st.markdown("### 📋 Pair Preview")
        cols = st.columns(min(len(pairs), 4))
        for i, (quote, base) in enumerate(pairs[:4]):
            with cols[i]:
                st.metric(f"Pair {i+1}", f"{quote.upper()}/{base.upper()}")
    
    st.stop()


# ================== LOAD STATE ==========

def load_state():
    """
    Load multi-pair analytics state.
    Returns (version, pairs_dict) or (None, {}) if file missing/invalid.
    """
    if not STATE_PATH.exists():
        return (None, {})
    try:
        with open(STATE_PATH) as f:
            content = f.read()
            if not content.strip():
                return (None, {})  # Empty file
            data = json.loads(content)
            
            # Handle versioned format
            if isinstance(data, dict) and "version" in data:
                return (data.get("version"), data.get("pairs", {}))
            else:
                # Legacy format (just pairs dict)
                return (0, data)
    except json.JSONDecodeError:
        # Partial write in progress - skip this frame
        return (None, None)  # Signal to skip
    except (FileNotFoundError, PermissionError):
        return (None, {})
    except Exception:
        return (None, {})


def load_history():
    """
    Load per-pair history.
    Returns {} if file missing or None if read fails (skip frame).
    """
    if not HISTORY_PATH.exists():
        return {}
    try:
        with open(HISTORY_PATH) as f:
            content = f.read()
            if not content.strip():
                return {}
            return json.loads(content)
    except json.JSONDecodeError:
        return None  # Skip frame
    except Exception:
        return {}



# Load current state
# Load versioned state
state_version, state = load_state()
history = load_history()

# FRAME SKIP: If JSON read failed (partial write), show waiting state
if state is None or history is None:
    st.info("⏳ Waiting for stable snapshot...")
    # Don't block - let natural refresh handle it

# ========== DIAGNOSTICS INTENT PROCESSING (BEFORE VERSION-LOCK) ==========
# CRITICAL: Process ALL pending diagnostics intents BEFORE version-lock skip
# This ensures button clicks are NEVER dropped by version-lock logic
diag_store = st.session_state.setdefault("diagnostics", {})
diag_intents = {k: v for k, v in st.session_state.items() if k.startswith("diag_intent_") and v}

for intent_key in diag_intents:
    pair_id = intent_key.replace("diag_intent_", "")
    pair_history = st.session_state.history.get(pair_id, [])
    
    # Extract spread data
    if pair_history and len(pair_history) >= 30:
        arr = np.array([h.get("spread") or 0 for h in pair_history[-500:]], dtype=np.float64)
        arr = arr[np.isfinite(arr)]
        
        if len(arr) >= 30:
            adf_result = run_adf_test(arr)
            hl_result = compute_half_life(arr)
            
            diag_store[pair_id] = {
                "adf": adf_result,
                "halflife": hl_result,
                "computed_at": time.time()
            }
            st.session_state["diagnostics"] = diag_store
    
    # Clear latch (ALWAYS - even if computation failed)
    st.session_state[intent_key] = False

# VERSION-LOCKED RENDERING: Skip if state unchanged
if "last_version" not in st.session_state:
    st.session_state.last_version = -1

# Check if we should skip rendering (version unchanged)
skip_render = state_version is not None and state_version == st.session_state.last_version

# Update version tracker
if state_version is not None:
    st.session_state.last_version = state_version

# If version unchanged, just wait for next refresh cycle
# BUT never sleep if any pair is ready (instant chart render)
if skip_render:
    any_ready = any(s.get("ready", False) for s in state.values()) if state else False
    if not any_ready:
        time.sleep(refresh_rate)
    st.rerun()


# Sync history to session state (MERGE with MONOTONICITY enforcement)
HISTORY_MAX_LEN = 500  # Must match backend cap

if isinstance(history, dict):
    for pair_id, pair_history in history.items():
        if pair_id not in st.session_state.history:
            st.session_state.history[pair_id] = []
        
        # Get last timestamp in session history (monotonicity anchor)
        last_ts = 0
        if st.session_state.history[pair_id]:
            last_ts = st.session_state.history[pair_id][-1].get("timestamp", 0)
        
        # MONOTONICITY: Only add entries with strictly newer timestamps
        for entry in pair_history:
            entry_ts = entry.get("timestamp", 0)
            if entry_ts > last_ts:
                st.session_state.history[pair_id].append(entry)
                last_ts = entry_ts  # Update anchor
        
        # CAP: Enforce max length to prevent unbounded growth
        if len(st.session_state.history[pair_id]) > HISTORY_MAX_LEN:
            st.session_state.history[pair_id] = st.session_state.history[pair_id][-HISTORY_MAX_LEN:]
else:
    # Old format was list - ignore it
    history = {}


# ========== WAITING FOR DATA ==========
if not state:
    st.warning("⏳ **Waiting for analytics data...**")
    st.caption("Backend is buffering ticks for configured pairs.")
    st.progress(0.0, text="Waiting for first analytics snapshot...")
    time.sleep(refresh_rate)
    st.rerun()  # Exit here - don't render rest of UI


# ========== CONNECTION STATUS ==========
active_pairs = len(state)
st.success(f"🟢 **LIVE** • {active_pairs} pair{'s' if active_pairs > 1 else ''} • Window: {window_size} • Refresh: {refresh_rate}s")


# ================== HELPER: RENDER PAIR ANALYTICS ==================
def render_pair_analytics(pair_id: str, snapshot: dict, pair_history: list, z_threshold: float, window_size: int):
    """
    Render analytics for a single pair.
    
    CRITICAL: 
    - Every visual element explicitly identifies its pair.
    - ALWAYS renders the same structure (stable widget tree).
    - Uses placeholders when data not ready.
    """
    
    # Check if this pair has analytics data or is still buffering
    is_ready = snapshot.get("ready", False)
    
    # STICKY READY: Once a pair is seen as ready, UI never shows buffering again
    if is_ready:
        st.session_state.pair_ready_once[pair_id] = True
    display_ready = st.session_state.pair_ready_once.get(pair_id, False)
    
    # ====== PAIR IDENTITY HEADER (ALWAYS RENDERED) ======
    # Use display_ready (sticky) to prevent buffering flicker
    if display_ready:
        zscore_display = snapshot.get('zscore') or 0
        beta_display = snapshot.get('beta') or 0
        corr_display = snapshot.get('correlation') or 0
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%); 
                    padding: 15px 20px; border-radius: 10px; margin-bottom: 15px;
                    border-left: 4px solid #40916c;">
            <h3 style="margin: 0; color: #e94560;">{pair_id.upper()}</h3>
            <p style="margin: 5px 0 0 0; color: #a0a0a0; font-size: 0.9rem;">
                Z={zscore_display:+.2f} | β={beta_display:.4f} | ρ={corr_display:.2f}
            </p>
        </div>
        """, unsafe_allow_html=True)
    else:
        # Buffering state - show placeholder header
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%); 
                    padding: 15px 20px; border-radius: 10px; margin-bottom: 15px;
                    border-left: 4px solid #ffd60a;">
            <h3 style="margin: 0; color: #ffd60a;">⏳ {pair_id.upper()}</h3>
            <p style="margin: 5px 0 0 0; color: #a0a0a0; font-size: 0.9rem;">
                Buffering data...
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    # ====== PAIR QUALITY BADGE ======
    # Handle None from frozen schema (buffering state)
    corr = abs(snapshot.get("correlation") or 0)
    beta = abs(snapshot.get("beta") or 0)
    
    if corr >= 0.7 and beta < 5:
        st.markdown(f"""
        <div style="background: linear-gradient(90deg, #2d6a4f, #40916c); padding: 8px 12px; border-radius: 6px; margin: 8px 0; display: inline-block;">
        🟢 <strong>VALID PAIR</strong> — {pair_id.upper()}
        </div>
        """, unsafe_allow_html=True)
    elif corr >= 0.5:
        st.markdown(f"""
        <div style="background: linear-gradient(90deg, #b8860b, #daa520); padding: 8px 12px; border-radius: 6px; margin: 8px 0; display: inline-block;">
        🟡 <strong>WEAK</strong> — {pair_id.upper()}
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div style="background: linear-gradient(90deg, #8b0000, #dc143c); padding: 8px 12px; border-radius: 6px; margin: 8px 0; display: inline-block;">
        🔴 <strong>UNSUITABLE</strong> — {pair_id.upper()}
        </div>
        """, unsafe_allow_html=True)
    
    # ====== KPI METRICS (PAIR-LABELED) - SAFE ACCESS ======
    col1, col2, col3, col4 = st.columns(4)
    
    # Handle None from frozen schema (buffering state)
    beta_val = snapshot.get('beta') or 0
    zscore_val = snapshot.get('zscore') or 0
    spread_val = snapshot.get('spread') or 0
    corr_val = snapshot.get('correlation') or 0
    ts_val = snapshot.get('timestamp') or time.time()
    
    with col1:
        st.metric(f"β ({pair_id})", f"{beta_val:.4f}" if is_ready else "—")
    
    with col2:
        delta = "🚨 ALERT" if abs(zscore_val) >= z_threshold else "✅ OK"
        st.metric(f"Z ({pair_id})", f"{zscore_val:.2f}" if is_ready else "—", delta=delta if is_ready else None)
    
    with col3:
        st.metric(f"Spread ({pair_id})", f"{spread_val:.6f}" if is_ready else "—")
    
    with col4:
        st.metric(f"ρ ({pair_id})", f"{corr_val:.2%}" if is_ready else "—")
    
    # ====== ALERT BANNER (PAIR-SPECIFIC) ======
    if is_ready and abs(zscore_val) >= z_threshold:
        st.error(f"⚠️ **{pair_id.upper()} ALERT**: Z-Score = {zscore_val:.2f} exceeds ±{z_threshold}")
    
    # ====== STATISTICAL DIAGNOSTICS (ON-DEMAND, PRODUCTION-SAFE) ======
    st.markdown("**Statistical Diagnostics (On-Demand)**")
    
    # Constants - MUST match analytics.py DIAGNOSTICS_WINDOW
    MIN_ADF_LEN = 30  # Minimum for valid ADF test
    MAX_ADF_LEN = 500
    
    # Session state keys for latch pattern
    diag_store = st.session_state.setdefault("diagnostics", {})
    intent_key = f"diag_intent_{pair_id}"
    
    # Helper to extract cleaned spread from history
    def _clean_spread_from_history(history, max_len=MAX_ADF_LEN):
        if not history or len(history) == 0:
            return np.array([], dtype=float)
        arr = np.array([h.get("spread") or 0 for h in history[-max_len:]], dtype=np.float64)
        arr = arr[np.isfinite(arr)]
        return arr
    
    spread_arr = _clean_spread_from_history(pair_history)
    has_enough_data = spread_arr.size >= MIN_ADF_LEN
    
    # Readiness and data sufficiency checks
    diag_col1, diag_col2 = st.columns([1, 3])
    
    with diag_col1:
        btn_label = f"Run Stationarity Check (Last {MIN_ADF_LEN} pts)"
        btn_disabled = not is_ready or not has_enough_data
        
        # Button sets intent latch - will be processed on next render cycle
        if st.button(btn_label, key=f"diag_btn_{pair_id}", disabled=btn_disabled):
            st.session_state[intent_key] = True
            # No st.rerun() - let natural refresh handle it
    
    # Display status/results
    with diag_col2:
        if not is_ready:
            st.caption("⏳ Waiting for pair readiness")
        elif not has_enough_data:
            st.caption(f"⏳ Need {MIN_ADF_LEN} spread points — currently {spread_arr.size}")
        elif pair_id in diag_store:
            pd_diag = diag_store[pair_id]
            adf = pd_diag.get("adf", {})
            
            if not adf.get("valid", False):
                # Show reason for invalid result
                reason = adf.get("reason", "Unknown error")
                st.warning(f"⚠️ {reason}")
            else:
                pvalue = adf.get("p_value")
                is_stat = adf.get("is_stationary")
                points_used = adf.get("points_used", MIN_ADF_LEN)
                hl = pd_diag.get("halflife")
                
                verdict = "✅ Mean-Reverting" if is_stat else "❌ Not Stationary"
                pval_str = f"{pvalue:.4f}" if pvalue is not None else "—"
                hl_str = f"{hl:.1f} ticks" if hl is not None else "N/A"
                
                st.caption(f"p-value: {pval_str} | {verdict} | Half-life: {hl_str} | Points: {points_used}")
        else:
            st.caption("Click button to run diagnostics")
    
    # ====== PREPARE PAIR-SPECIFIC DATAFRAME (CACHED) ======
    # Cache key: (pair_id, history_length) - avoids expensive rebuild each render
    RENDER_WINDOW = 100  # Max points to display
    cache_key = f"df_{pair_id}_{len(pair_history)}"
    
    if cache_key in st.session_state.cached_df:
        pair_df = st.session_state.cached_df[cache_key]
    else:
        # Build DataFrame only when cache miss
        if is_ready and len(pair_history) > 1:
            pair_df = pd.DataFrame(pair_history)
            pair_df["time"] = pd.to_datetime(pair_df["timestamp"], unit="s").dt.strftime("%H:%M:%S")
        elif is_ready:
            pair_df = pd.DataFrame([snapshot])
            pair_df["time"] = [time.strftime('%H:%M:%S', time.localtime(ts_val))]
        else:
            pair_df = pd.DataFrame({"time": [], "spread": [], "zscore": [], "beta": [], "correlation": []})
        
        # Process: deduplicate and limit
        if len(pair_df) > 0:
            pair_df = pair_df.drop_duplicates(subset=["time"], keep="last")
            pair_df = pair_df.tail(RENDER_WINDOW)
            pair_df = pair_df.reset_index(drop=True)
            pair_df["t"] = range(len(pair_df))
        
        # Cache it (limit cache size to prevent memory bloat)
        if len(st.session_state.cached_df) > 50:
            # Clear old entries
            st.session_state.cached_df = {}
        st.session_state.cached_df[cache_key] = pair_df
    
    
    # ====== ROW 1: SPREAD & Z-SCORE (ALWAYS RENDER) ======
    row1_col1, row1_col2 = st.columns(2)
    
    with row1_col1:
        # SPREAD CHART
        if len(pair_df) >= 1:
            fig_spread = go.Figure()
            fig_spread.add_trace(go.Scatter(
                x=pair_df["t"], 
                y=pair_df["spread"],
                mode="lines+markers",
                name=f"Spread ({pair_id})",
                line=dict(color="#00d4ff", width=2),
                marker=dict(size=3)
            ))
            fig_spread.update_layout(
                title=f"📉 Spread — {pair_id.upper()}",
                xaxis_title="Tick",
                yaxis_title="Spread",
                height=250,
                template="plotly_dark",
                margin=dict(l=10, r=10, t=40, b=40),
                showlegend=False,
                uirevision=f"spread_{pair_id}"  # Lock axes during reruns
            )
            st.plotly_chart(fig_spread, use_container_width=True, key=f"spread_{pair_id}")
        else:
            # PLACEHOLDER CHART (keeps widget tree stable)
            fig_spread = go.Figure()
            fig_spread.update_layout(
                title=f"📉 Spread — {pair_id.upper()} (buffering...)",
                height=250, template="plotly_dark",
                margin=dict(l=10, r=10, t=40, b=40),
                uirevision=f"spread_{pair_id}",
                annotations=[dict(text="Waiting for data...", showarrow=False, font=dict(size=14, color="gray"))]
            )
            st.plotly_chart(fig_spread, use_container_width=True, key=f"spread_{pair_id}")
    
    with row1_col2:
        # Z-SCORE CHART
        if len(pair_df) >= 1:
            fig_z = go.Figure()
            fig_z.add_trace(go.Scatter(
                x=pair_df["t"], 
                y=pair_df["zscore"],
                mode="lines+markers",
                name=f"Z-Score ({pair_id})",
                line=dict(color="#e94560", width=2),
                marker=dict(size=3)
            ))
            fig_z.add_hline(y=z_threshold, line_dash="dash", line_color="#ffd60a", line_width=2,
                           annotation_text=f"+{z_threshold}")
            fig_z.add_hline(y=-z_threshold, line_dash="dash", line_color="#ffd60a", line_width=2,
                           annotation_text=f"-{z_threshold}")
            fig_z.add_hline(y=0, line_dash="dot", line_color="gray")
            fig_z.update_layout(
                title=f"📊 Z-Score — {pair_id.upper()}",
                xaxis_title="Tick",
                yaxis_title="Z-Score (σ)",
                height=250,
                template="plotly_dark",
                margin=dict(l=10, r=10, t=40, b=40),
                showlegend=False,
                uirevision=f"zscore_{pair_id}"
            )
            st.plotly_chart(fig_z, use_container_width=True, key=f"zscore_{pair_id}")
        else:
            fig_z = go.Figure()
            fig_z.add_hline(y=z_threshold, line_dash="dash", line_color="#ffd60a", line_width=2)
            fig_z.add_hline(y=-z_threshold, line_dash="dash", line_color="#ffd60a", line_width=2)
            fig_z.add_hline(y=0, line_dash="dot", line_color="gray")
            fig_z.update_layout(
                title=f"📊 Z-Score — {pair_id.upper()} (buffering...)",
                height=250, template="plotly_dark",
                margin=dict(l=10, r=10, t=40, b=40),
                uirevision=f"zscore_{pair_id}",
                annotations=[dict(text="Waiting for data...", showarrow=False, font=dict(size=14, color="gray"))]
            )
            st.plotly_chart(fig_z, use_container_width=True, key=f"zscore_{pair_id}")
    
    # ====== ROW 2: BETA & CORRELATION (ALWAYS RENDER) ======
    row2_col1, row2_col2 = st.columns(2)
    
    with row2_col1:
        # BETA CHART
        if len(pair_df) >= 1:
            fig_beta = go.Figure()
            fig_beta.add_trace(go.Scatter(
                x=pair_df["t"], 
                y=pair_df["beta"],
                mode="lines+markers",
                name=f"Beta ({pair_id})",
                line=dict(color="#40916c", width=2),
                marker=dict(size=3)
            ))
            fig_beta.add_hline(y=1.0, line_dash="dash", line_color="#ffd60a", line_width=1,
                              annotation_text="β=1")
            fig_beta.update_layout(
                title=f"📈 Beta (β) — {pair_id.upper()}",
                xaxis_title="Tick",
                yaxis_title="Hedge Ratio",
                height=250,
                template="plotly_dark",
                margin=dict(l=10, r=10, t=40, b=40),
                showlegend=False,
                uirevision=f"beta_{pair_id}"
            )
            st.plotly_chart(fig_beta, use_container_width=True, key=f"beta_{pair_id}")
        else:
            fig_beta = go.Figure()
            fig_beta.add_hline(y=1.0, line_dash="dash", line_color="#ffd60a", line_width=1)
            fig_beta.update_layout(
                title=f"📈 Beta (β) — {pair_id.upper()} (buffering...)",
                height=250, template="plotly_dark",
                margin=dict(l=10, r=10, t=40, b=40),
                uirevision=f"beta_{pair_id}",
                annotations=[dict(text="Waiting for data...", showarrow=False, font=dict(size=14, color="gray"))]
            )
            st.plotly_chart(fig_beta, use_container_width=True, key=f"beta_{pair_id}")
    
    with row2_col2:
        # CORRELATION CHART
        if len(pair_df) >= 1:
            fig_corr = go.Figure()
            fig_corr.add_trace(go.Scatter(
                x=pair_df["t"], 
                y=pair_df["correlation"],
                mode="lines+markers",
                name=f"Correlation ({pair_id})",
                line=dict(color="#9d4edd", width=2),
                marker=dict(size=3)
            ))
            fig_corr.add_hline(y=0.7, line_dash="dash", line_color="#40916c", line_width=1,
                              annotation_text="ρ=0.7 (valid)")
            fig_corr.add_hline(y=-0.7, line_dash="dash", line_color="#40916c", line_width=1)
            fig_corr.add_hline(y=0, line_dash="dot", line_color="gray")
            fig_corr.update_layout(
                title=f"🔗 Correlation (ρ) — {pair_id.upper()}",
                xaxis_title="Tick",
                yaxis_title="Pearson ρ",
                height=250,
                template="plotly_dark",
                margin=dict(l=10, r=10, t=40, b=40),
                yaxis=dict(range=[-1.1, 1.1]),
                showlegend=False,
                uirevision=f"corr_{pair_id}"
            )
            st.plotly_chart(fig_corr, use_container_width=True, key=f"corr_{pair_id}")
        else:
            fig_corr = go.Figure()
            fig_corr.add_hline(y=0.7, line_dash="dash", line_color="#40916c", line_width=1)
            fig_corr.add_hline(y=-0.7, line_dash="dash", line_color="#40916c", line_width=1)
            fig_corr.add_hline(y=0, line_dash="dot", line_color="gray")
            fig_corr.update_layout(
                title=f"🔗 Correlation (ρ) — {pair_id.upper()} (buffering...)",
                height=250, template="plotly_dark",
                margin=dict(l=10, r=10, t=40, b=40),
                yaxis=dict(range=[-1.1, 1.1]),
                uirevision=f"corr_{pair_id}",
                annotations=[dict(text="Waiting for data...", showarrow=False, font=dict(size=14, color="gray"))]
            )
            st.plotly_chart(fig_corr, use_container_width=True, key=f"corr_{pair_id}")
    
    # ====== PAIR-SPECIFIC FOOTER ======
    last_time = time.strftime('%H:%M:%S', time.localtime(ts_val)) if is_ready else "—"
    st.caption(f"📍 {pair_id.upper()} | History: {len(pair_history)} pts | Last: {last_time}")


# ================== TABBED VIEW PER PAIR ==================
st.markdown("---")

# Sort pairs deterministically
sorted_pair_ids = sorted(state.keys())
active_pairs = len(sorted_pair_ids)

# RULE: Always render tabs, even for 0 or 1 pair (stable structure)
if active_pairs == 0:
    # Still render a placeholder container
    with st.container():
        st.warning("No analytics data available yet")
else:
    # ALWAYS use tabs (even for single pair) - stable structure
    tabs = st.tabs(sorted_pair_ids if active_pairs > 0 else ["Waiting..."])
    
    for i, pair_id in enumerate(sorted_pair_ids):
        with tabs[i]:
            snapshot = state[pair_id]
            pair_history = st.session_state.history.get(pair_id, [])
            
            # ALWAYS render the same structure - readiness is handled INSIDE
            is_ready = snapshot.get("ready", False)
            
            # Show buffering status FIRST if not ready (but don't skip the graphs)
            if not is_ready:
                st.info(f"""
                ⏳ **{pair_id.upper()} — Buffering**
                
                | Symbol | Progress |
                |--------|----------|
                | {snapshot.get('x_symbol', '?')} | {snapshot.get('x_count', 0)}/{snapshot.get('window', 200)} ({snapshot.get('x_pct', 0)}%) |
                | {snapshot.get('y_symbol', '?')} | {snapshot.get('y_count', 0)}/{snapshot.get('window', 200)} ({snapshot.get('y_pct', 0)}%) |
                """)
            
            # ALWAYS render the pair analytics (function handles empty state internally)
            render_pair_analytics(pair_id, snapshot, pair_history, z_threshold, window_size)


# ================== GLOBAL: EXPORTS & LOGS (OUTSIDE PAIR LOOP) ==================
st.subheader("📤 Exports & Logs")

# Check if all pairs are ready
all_ready = all(s.get("ready", False) for s in state.values()) if state else False
if state and not all_ready:
    st.warning("⚠️ **Some pairs still buffering** — exports may be incomplete")

# Compute data once
total_history = sum(len(h) for h in st.session_state.history.values())

exp_col1, exp_col2, exp_col3 = st.columns(3)

with exp_col1:
    # JSON STATE DOWNLOAD (IN-MEMORY BYTES)
    state_ready = state is not None and len(state) > 0
    
    if state_ready:
        json_bytes = json.dumps(state, indent=2).encode("utf-8")
        
        st.download_button(
            label="📊 Download State (JSON)",
            data=json_bytes,
            file_name="analytics_state.json",
            mime="application/json",
            use_container_width=True,
            key="download_state_json"
        )
    else:
        st.info("⏳ No state yet")

with exp_col2:
    # CSV HISTORY DOWNLOAD (IN-MEMORY BYTES)
    if total_history > 0:
        rows = []
        for pair_id, hist in st.session_state.history.items():
            for h in hist:
                rows.append({
                    "timestamp": h.get("timestamp", 0),
                    "pair": pair_id,
                    "spread": h.get("spread", 0),
                    "zscore": h.get("zscore", 0),
                    "beta": h.get("beta", 0),
                    "correlation": h.get("correlation", 0)
                })
        
        if rows:
            history_df = pd.DataFrame(rows)
            csv_bytes = history_df.to_csv(index=False).encode("utf-8")
            
            st.download_button(
                label="📈 Download History (CSV)",
                data=csv_bytes,
                file_name="analytics_history.csv",
                mime="text/csv",
                use_container_width=True,
                key="download_history_csv"
            )
    else:
        st.info("⏳ No history yet")

with exp_col3:
    st.metric("Total History", f"{total_history} pts", help="Combined across all pairs")

# ================== GLOBAL: RECENT ANALYTICS LOG (SCROLLABLE, STABLE) ==================
# Wrap in single column to create stable structure (prevents duplicates on reruns)
# The dataframe and info widgets inside have keys to ensure single rendering
_log_col, = st.columns([1])
with _log_col:
    with st.expander("📋 Recent Analytics Log (All Pairs)", expanded=False):
        # Combine all histories for log table - cap at 100, global not per-pair
        LOG_MAX_LEN = 100
        all_recent = []
        for pair_id, hist in st.session_state.history.items():
            for h in hist:
                all_recent.append({
                    "ts": h.get("timestamp", 0),
                    "pair": pair_id.upper(),
                    "beta": h.get("beta") or 0,
                    "zscore": h.get("zscore") or 0,
                    "spread": h.get("spread") or 0,
                    "correlation": h.get("correlation") or 0
                })

        if len(all_recent) > 0:
            log_df = pd.DataFrame(all_recent)
            log_df["time"] = pd.to_datetime(log_df["ts"], unit="s").dt.strftime("%H:%M:%S")
            log_df = log_df.sort_values("ts", ascending=False).head(LOG_MAX_LEN)
            log_df = log_df[["time", "pair", "beta", "zscore", "spread", "correlation"]]
            log_df.columns = ["Time", "Pair", "Beta", "Z-Score", "Spread", "Correlation"]
            
            # Fixed height = no layout jump, scrollable
            st.dataframe(
                log_df,
                height=420,
                use_container_width=True,
                hide_index=True,
                key="analytics_log_dataframe"
            )
        else:
            st.info("No analytics logs yet.")


# ================== FOOTER ==========
st.markdown("---")
st.caption(f"🔒 Read-only dashboard | {len(state)} pairs active | Window: {window_size} | Refresh: {refresh_rate}s")
st.caption("ℹ️ β and ρ are rolling tick-aligned estimates — rapid variation is expected under microstructure noise.")

# ================== AUTO REFRESH ==========
time.sleep(refresh_rate)
st.rerun()
