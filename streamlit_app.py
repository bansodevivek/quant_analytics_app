"""
Streamlit dashboard for real-time quantitative analytics.

UX Design:
- Start button: locks config, starts backend, shows charts
- Stop button: stops backend, hides charts, unlocks config
- Single running flag controls all visibility
- No Apply/Confirm - just Start/Stop
"""

import time
import os
import csv
import json
import streamlit as st
import plotly.graph_objects as go
import numpy as np
import pandas as pd
from pathlib import Path


# ---------------- Page Config ----------------
st.set_page_config(
    page_title="Quant Analytics Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------- Session State Initialization ----------------
if "running" not in st.session_state:
    st.session_state.running = False
if "pair" not in st.session_state:
    st.session_state.pair = None
if "history" not in st.session_state:
    st.session_state.history = []
if "logs" not in st.session_state:
    st.session_state.logs = []


# ---------------- Constants ----------------
DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)

CONTROL_PATH = DATA_DIR / "control.json"
STATE_PATH = DATA_DIR / "analytics_state.csv"
HISTORY_PATH = DATA_DIR / "analytics_history.json"

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
    
    # Symbol Selection (disabled when running)
    st.subheader("📊 Symbol Selection")
    
    base_symbol = st.selectbox(
        "Base Symbol (X)",
        options=AVAILABLE_SYMBOLS,
        index=0,
        disabled=is_running,
        help="Independent variable in regression"
    )
    
    quote_symbol = st.selectbox(
        "Quote Symbol (Y)",
        options=AVAILABLE_SYMBOLS,
        index=1,
        disabled=is_running,
        help="Dependent variable in regression"
    )
    
    custom_base = st.text_input(
        "Custom Base Symbol",
        placeholder="e.g., arbusdt",
        disabled=is_running
    )
    
    custom_quote = st.text_input(
        "Custom Quote Symbol",
        placeholder="e.g., opusdt",
        disabled=is_running
    )
    
    # Use custom if provided
    final_base = custom_base.lower().strip() if custom_base else base_symbol
    final_quote = custom_quote.lower().strip() if custom_quote else quote_symbol
    
    st.divider()
    
    # Analytics Parameters
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
    
    # ========== START / STOP BUTTONS ==========
    st.subheader("🎮 Control")
    
    col1, col2 = st.columns(2)
    
    with col1:
        start_clicked = st.button(
            "▶️ Start",
            disabled=is_running,
            use_container_width=True,
            type="primary"
        )
    
    with col2:
        stop_clicked = st.button(
            "⏹️ Stop",
            disabled=not is_running,
            use_container_width=True
        )
    
    # ========== START LOGIC ==========
    if start_clicked:
        # -------- SYMBOL VALIDATION (STRICT) --------
        validation_error = None
        
        if final_base == final_quote:
            validation_error = "❌ Base and Quote symbols must be different!"
        elif final_base not in AVAILABLE_SYMBOLS and not custom_base:
            validation_error = f"❌ No data stream available for {final_base.upper()}"
        elif final_quote not in AVAILABLE_SYMBOLS and not custom_quote:
            validation_error = f"❌ No data stream available for {final_quote.upper()}"
        
        if validation_error:
            st.error(validation_error)
            st.stop()
        
        # -------- HARD RESET UI STATE --------
        st.session_state.running = True
        st.session_state.pair = f"{final_quote.upper()}/{final_base.upper()}"
        st.session_state.history = []
        st.session_state.logs = []
        
        # Clear old state files
        if STATE_PATH.exists():
            STATE_PATH.unlink()
        if HISTORY_PATH.exists():
            with open(HISTORY_PATH, "w") as f:
                json.dump([], f)
        
        # -------- WRITE CONTROL SIGNAL --------
        with open(CONTROL_PATH, "w") as f:
            json.dump({
                "action": "START",
                "symbol_x": final_base,
                "symbol_y": final_quote,
                "window_size": window_size,
                "timestamp": time.time()
            }, f)
        
        st.rerun()
    
    # ========== STOP LOGIC ==========
    if stop_clicked:
        st.session_state.running = False
        
        # Write stop signal for backend
        with open(CONTROL_PATH, "w") as f:
            json.dump({
                "action": "STOP",
                "timestamp": time.time()
            }, f)
        
        st.rerun()
    
    # Status indicator
    st.divider()
    if is_running:
        st.markdown(f"""
        <div class="status-running">
        🟢 <strong>RUNNING</strong><br>
        <small>{st.session_state.pair}</small>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="status-stopped">
        ⚫ <strong>STOPPED</strong><br>
        <small>Press Start to begin</small>
        </div>
        """, unsafe_allow_html=True)


# ================== MAIN CONTENT ==================

# Header
st.markdown("""
<div class="main-header">
    <h1 style="margin:0; color: #e94560;">Quant Analytics Dashboard</h1>
    <p style="margin:5px 0 0 0; color: #a0a0a0;">Real-Time Statistical Arbitrage Monitor • Binance Futures</p>
</div>
""", unsafe_allow_html=True)


# ========== GATE: Show content only when running ==========
if not st.session_state.running:
    st.info("👈 **Configure symbols and press Start** to begin live analytics.")
    st.caption("Select your trading pair, set parameters, then click the Start button in the sidebar.")
    st.stop()


# ========== RUNNING STATE - Load and Display Data ==========

def load_state():
    """Load latest analytics snapshot."""
    if not STATE_PATH.exists():
        return None
    try:
        with open(STATE_PATH, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                return {
                    "timestamp": float(row["timestamp"]),
                    "pair": row["pair"],
                    "alpha": float(row["alpha"]),
                    "beta": float(row["beta"]),
                    "spread": float(row["spread"]),
                    "zscore": float(row["zscore"]),
                    "correlation": float(row["correlation"]),
                }
    except Exception:
        return None
    return None


def load_history():
    """Load history from backend."""
    if not HISTORY_PATH.exists():
        return []
    try:
        with open(HISTORY_PATH) as f:
            return json.load(f)
    except Exception:
        return []


# Load current state
state = load_state()
history = load_history()

# Sync history to session state
if history:
    st.session_state.history = history

# ========== STATE-AWARE STATUS MESSAGING ==========
requested_pair = f"{final_quote.upper()}/{final_base.upper()}"

# Case 1: No analytics data yet (buffering phase)
if state is None:
    st.markdown("""
    <div style="background: linear-gradient(90deg, #1a5276, #2980b9); padding: 15px; border-radius: 10px; margin: 10px 0;">
    ⏳ <strong>BUFFERING</strong> — Collecting ticks for {pair}<br>
    <small>Window needs to fill before analytics can compute. This may take 30-60 seconds.</small>
    </div>
    """.format(pair=requested_pair), unsafe_allow_html=True)
    
    # Show a progress indicator
    st.progress(0.0, text="Waiting for first analytics snapshot...")
    time.sleep(2)
    st.rerun()

# Case 2: Backend pair doesn't match requested pair (transitioning)
if state["pair"].lower().replace("/", "") != f"{final_quote}{final_base}".lower():
    st.markdown("""
    <div style="background: linear-gradient(90deg, #b8860b, #daa520); padding: 15px; border-radius: 10px; margin: 10px 0;">
    🔄 <strong>TRANSITIONING</strong> — Backend switching from {old} to {new}<br>
    <small>Ingestion restarting for new symbols. Please wait...</small>
    </div>
    """.format(old=state["pair"], new=requested_pair), unsafe_allow_html=True)
    time.sleep(1)
    st.rerun()


# ========== CONNECTION STATUS ==========
col_status, col_time = st.columns([4, 1])
with col_status:
    st.success(f"🟢 **LIVE** • Pair: `{state['pair']}` • Window: `{window_size}` • History: `{len(st.session_state.history)} pts`")
with col_time:
    st.caption(f"Updated: {time.strftime('%H:%M:%S', time.localtime(state['timestamp']))}")


# ========== PAIR QUALITY BADGE ==========
corr = abs(state["correlation"])
beta = abs(state["beta"])

if corr >= 0.7 and beta < 5:
    st.markdown("""
    <div style="background: linear-gradient(90deg, #2d6a4f, #40916c); padding: 10px 15px; border-radius: 8px; margin: 10px 0;">
    🟢 <strong>VALID PAIR</strong> — Strong correlation (ρ = {:.2f}), stable hedge ratio
    </div>
    """.format(state["correlation"]), unsafe_allow_html=True)
elif corr >= 0.5:
    st.markdown("""
    <div style="background: linear-gradient(90deg, #b8860b, #daa520); padding: 10px 15px; border-radius: 8px; margin: 10px 0;">
    🟡 <strong>WEAK RELATIONSHIP</strong> — Moderate correlation (ρ = {:.2f}), use caution
    </div>
    """.format(state["correlation"]), unsafe_allow_html=True)
else:
    st.markdown("""
    <div style="background: linear-gradient(90deg, #8b0000, #dc143c); padding: 10px 15px; border-radius: 8px; margin: 10px 0;">
    🔴 <strong>UNSUITABLE FOR STAT-ARB</strong> — Weak correlation (ρ = {:.2f}), hedge unreliable
    </div>
    """.format(state["correlation"]), unsafe_allow_html=True)


# ========== KPI METRICS ==========
st.markdown("### 📊 Live Metrics")

col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.metric("Pair", state["pair"])

with col2:
    st.metric("Beta (β)", f"{state['beta']:.4f}")

with col3:
    z = state["zscore"]
    delta = "🚨 ALERT" if abs(z) >= z_threshold else "✅ OK"
    st.metric("Z-Score", f"{z:.2f}", delta=delta)

with col4:
    st.metric("Spread", f"{state['spread']:.6f}")

with col5:
    st.metric("Correlation (ρ)", f"{state['correlation']:.2%}")


# ========== ALERT BANNER ==========
if abs(state["zscore"]) >= z_threshold:
    st.error(f"⚠️ **ALERT**: Z-Score = {state['zscore']:.2f} exceeds ±{z_threshold} — Mean-reversion signal!")
else:
    st.success("✅ Z-Score within normal bounds")

st.markdown("---")


# ================== ROLLING CHARTS ==================
st.markdown("### 📈 Rolling Time Series")

# Prepare history data
if len(st.session_state.history) > 1:
    df = pd.DataFrame(st.session_state.history)
    df["time"] = pd.to_datetime(df["timestamp"], unit="s").dt.strftime("%H:%M:%S")
else:
    # Single point fallback
    df = pd.DataFrame([state])
    df["time"] = [time.strftime('%H:%M:%S', time.localtime(state["timestamp"]))]


# Row 1: Spread and Z-Score
chart_col1, chart_col2 = st.columns(2)

with chart_col1:
    fig_spread = go.Figure()
    fig_spread.add_trace(go.Scatter(
        x=df["time"], y=df["spread"],
        mode="lines+markers",
        name="Spread",
        line=dict(color="#00d4ff", width=2),
        marker=dict(size=3)
    ))
    if len(df) > 1:
        mean_spread = df["spread"].mean()
        fig_spread.add_hline(y=mean_spread, line_dash="dot", line_color="yellow",
                            annotation_text=f"μ={mean_spread:.6f}")
    fig_spread.update_layout(
        title="📉 Spread Time Series",
        xaxis_title="Time",
        yaxis_title="Spread",
        height=350,
        template="plotly_dark",
        margin=dict(l=10, r=10, t=50, b=40)
    )
    st.plotly_chart(fig_spread, use_container_width=True)

with chart_col2:
    fig_z = go.Figure()
    fig_z.add_trace(go.Scatter(
        x=df["time"], y=df["zscore"],
        mode="lines+markers",
        name="Z-Score",
        line=dict(color="#e94560", width=2),
        marker=dict(size=3)
    ))
    # Threshold bands
    fig_z.add_hline(y=z_threshold, line_dash="dash", line_color="#ffd60a", line_width=2,
                    annotation_text=f"+{z_threshold}")
    fig_z.add_hline(y=-z_threshold, line_dash="dash", line_color="#ffd60a", line_width=2,
                    annotation_text=f"-{z_threshold}")
    fig_z.add_hline(y=0, line_dash="dot", line_color="gray")
    fig_z.update_layout(
        title="📊 Z-Score with Alert Bands",
        xaxis_title="Time",
        yaxis_title="Z-Score (σ)",
        height=350,
        template="plotly_dark",
        margin=dict(l=10, r=10, t=50, b=40)
    )
    st.plotly_chart(fig_z, use_container_width=True)


# Row 2: Beta and Correlation
chart_col3, chart_col4 = st.columns(2)

with chart_col3:
    fig_beta = go.Figure()
    fig_beta.add_trace(go.Scatter(
        x=df["time"], y=df["beta"],
        mode="lines+markers",
        name="Beta",
        line=dict(color="#7b2cbf", width=2),
        marker=dict(size=3),
        fill="tozeroy",
        fillcolor="rgba(123, 44, 191, 0.15)"
    ))
    fig_beta.update_layout(
        title="📐 Rolling Hedge Ratio (β)",
        xaxis_title="Time",
        yaxis_title="Beta",
        height=300,
        template="plotly_dark",
        margin=dict(l=10, r=10, t=50, b=40)
    )
    st.plotly_chart(fig_beta, use_container_width=True)

with chart_col4:
    fig_corr = go.Figure()
    fig_corr.add_trace(go.Scatter(
        x=df["time"], y=df["correlation"],
        mode="lines+markers",
        name="Correlation",
        line=dict(color="#3a86ff", width=2),
        marker=dict(size=3),
        fill="tozeroy",
        fillcolor="rgba(58, 134, 255, 0.15)"
    ))
    fig_corr.update_layout(
        title="🔗 Rolling Correlation (ρ)",
        xaxis_title="Time",
        yaxis_title="Correlation",
        height=300,
        template="plotly_dark",
        margin=dict(l=10, r=10, t=50, b=40),
        yaxis=dict(range=[-1.1, 1.1])
    )
    st.plotly_chart(fig_corr, use_container_width=True)


st.markdown("---")


# ================== DATA EXPORT ==========
st.markdown("### 📥 Data Export")

exp_col1, exp_col2, exp_col3 = st.columns(3)

with exp_col1:
    csv_data = f"timestamp,pair,alpha,beta,spread,zscore,correlation\n"
    csv_data += f"{state['timestamp']},{state['pair']},{state['alpha']},{state['beta']},{state['spread']},{state['zscore']},{state['correlation']}"
    st.download_button(
        "💾 Download Snapshot",
        data=csv_data,
        file_name=f"snapshot_{state['pair'].replace('/', '_')}_{int(state['timestamp'])}.csv",
        mime="text/csv",
        use_container_width=True
    )

with exp_col2:
    if st.session_state.history:
        st.download_button(
            "📊 Download History",
            data=json.dumps(st.session_state.history, indent=2),
            file_name=f"history_{state['pair'].replace('/', '_')}_{int(time.time())}.json",
            mime="application/json",
            use_container_width=True
        )

with exp_col3:
    st.metric("History Points", len(st.session_state.history))


# ================== RECENT LOG TABLE ==========
if len(st.session_state.history) > 5:
    st.markdown("### 📋 Recent Analytics Log")
    
    log_df = pd.DataFrame(st.session_state.history[-25:])
    log_df["time"] = pd.to_datetime(log_df["timestamp"], unit="s").dt.strftime("%H:%M:%S")
    log_df = log_df[["time", "pair", "beta", "zscore", "spread", "correlation"]]
    log_df.columns = ["Time", "Pair", "Beta", "Z-Score", "Spread", "Correlation"]
    
    def highlight_zscore(val):
        if abs(val) >= z_threshold:
            return 'background-color: rgba(233, 69, 96, 0.3)'
        return ''
    
    styled_df = log_df.style.format({
        "Beta": "{:.4f}",
        "Z-Score": "{:.2f}",
        "Spread": "{:.6f}",
        "Correlation": "{:.2%}"
    }).applymap(highlight_zscore, subset=["Z-Score"])
    
    st.dataframe(styled_df, use_container_width=True)


# ================== FOOTER ==========
st.markdown("---")
st.caption(f"🔒 Read-only dashboard | Active: {state['pair']} | Window: {window_size}")


# ================== AUTO REFRESH ==========
time.sleep(refresh_rate)
st.rerun()
