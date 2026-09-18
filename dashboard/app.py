"""
dashboard/app.py
----------------
Zero-Egress Edge-to-Cloud Drift Detection — Hackathon Demo Dashboard.

Split-screen Streamlit UI:
  LEFT  — Edge node simulation (inference stream + local drift detection)
  RIGHT — Cloud orchestration panel (alerts, egress savings, event log)

Run:
    cd dashboard
    pip install -r requirements.txt
    streamlit run app.py

Prerequisites:
    Run `python edge_daemon/baseline_generator.py` first.
"""

import json
import os
import sys
import time
from collections import deque

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# Allow importing from edge_daemon when running from project root
DAEMON_PATH = os.path.join(os.path.dirname(__file__), "..", "edge_daemon")
if DAEMON_PATH not in sys.path:
    sys.path.insert(0, DAEMON_PATH)

# ---------------------------------------------------------------------------
# Page config — must be first Streamlit call
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Zero-Egress Drift Detection",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ---------------------------------------------------------------------------
# Custom CSS — dark glassmorphism theme
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');

    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    .stApp { background: linear-gradient(135deg, #0a0e1a 0%, #0f1929 50%, #0a0e1a 100%); }

    .panel {
        background: rgba(255,255,255,0.04);
        border: 1px solid rgba(255,255,255,0.1);
        border-radius: 16px;
        padding: 20px;
        margin-bottom: 16px;
        backdrop-filter: blur(10px);
    }

    .metric-card {
        background: rgba(99, 179, 237, 0.08);
        border: 1px solid rgba(99, 179, 237, 0.3);
        border-radius: 12px;
        padding: 14px;
        text-align: center;
    }

    .alert-critical {
        background: rgba(239,68,68,0.15);
        border: 1px solid rgba(239,68,68,0.5);
        border-radius: 10px;
        padding: 12px;
        color: #fca5a5;
        font-weight: 600;
    }

    .alert-ok {
        background: rgba(52,211,153,0.1);
        border: 1px solid rgba(52,211,153,0.3);
        border-radius: 10px;
        padding: 12px;
        color: #6ee7b7;
    }

    .log-entry {
        font-family: 'Courier New', monospace;
        font-size: 12px;
        color: #94a3b8;
        padding: 2px 0;
    }

    div[data-testid="metric-container"] {
        background: rgba(255,255,255,0.04);
        border: 1px solid rgba(255,255,255,0.1);
        border-radius: 10px;
        padding: 10px 16px;
    }

    .stButton>button {
        border-radius: 8px;
        font-weight: 600;
        transition: all 0.2s ease;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Load baseline profile
# ---------------------------------------------------------------------------
BASELINE_PATH = os.path.join(os.path.dirname(__file__), "..", "edge_daemon", "baseline_profile.json")

@st.cache_resource
def load_baseline():
    if not os.path.exists(BASELINE_PATH):
        return None
    with open(BASELINE_PATH) as f:
        return json.load(f)

baseline = load_baseline()

# ---------------------------------------------------------------------------
# Session state initialisation
# ---------------------------------------------------------------------------
DEFAULTS = {
    "running": False,
    "drift_injected": False,
    "step": 0,
    "stream_data": [],          # list of (step, value) for live chart
    "psi_history": [],          # list of (step, psi_value)
    "events": [],               # cloud event log
    "raw_bytes_total": 0,
    "actual_bytes_sent": 0,
    "alerts_sent": 0,
    "drift_confirmed_at": None,
}

for key, val in DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = val


# ---------------------------------------------------------------------------
# Helper: simulate PSI calculation
# ---------------------------------------------------------------------------
def simulate_psi(feature_data: list, baseline_stats: dict, bin_edges_key: str = "HouseAge") -> float:
    """Compute PSI against baseline for a single feature."""
    if baseline is None or len(feature_data) < 10:
        return 0.0
    feat_stats = baseline["features"].get(bin_edges_key, list(baseline["features"].values())[0])
    bin_edges = np.array(feat_stats["bin_edges"])
    expected = np.array(feat_stats["probabilities"])

    counts, _ = np.histogram(feature_data[-200:], bins=bin_edges)
    actual = counts.astype(float)
    actual = np.where(actual == 0, 1e-4, actual)
    actual /= actual.sum()

    psi = float(np.sum((actual - expected) * np.log(actual / expected)))
    return max(0.0, psi)


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.markdown(
    "<h1 style='text-align:center; color:#e2e8f0; font-size:2rem; margin-bottom:4px;'>"
    "⚡ Zero-Egress Edge-to-Cloud Drift Detection"
    "</h1>",
    unsafe_allow_html=True,
)
st.markdown(
    "<p style='text-align:center; color:#64748b; margin-bottom:24px;'>"
    "Real-time statistical drift monitoring | PSI + KS-Test | AWS Serverless MLOps"
    "</p>",
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Control bar
# ---------------------------------------------------------------------------
ctrl_col1, ctrl_col2, ctrl_col3, ctrl_col4 = st.columns([2, 2, 2, 2])

with ctrl_col1:
    if not st.session_state.running:
        if st.button("▶ Start Stream", use_container_width=True, type="primary"):
            st.session_state.running = True
            st.rerun()
    else:
        if st.button("⏹ Stop", use_container_width=True):
            st.session_state.running = False
            st.rerun()

with ctrl_col2:
    if st.button("💥 Inject Covariate Shift", use_container_width=True, type="secondary"):
        st.session_state.drift_injected = True
        st.session_state.events.append(
            f"[{st.session_state.step:04d}] ⚠️  Covariate shift injected at edge node"
        )

with ctrl_col3:
    if st.button("🔄 Reset", use_container_width=True):
        for key, val in DEFAULTS.items():
            st.session_state[key] = val if not isinstance(val, list) else []
        st.rerun()

with ctrl_col4:
    if baseline is None:
        st.error("⚠ baseline_profile.json not found")
    else:
        st.success(f"✅ Baseline loaded ({len(baseline['features'])} features)")

st.divider()

# ---------------------------------------------------------------------------
# Main split layout
# ---------------------------------------------------------------------------
left, right = st.columns([1, 1], gap="large")

with left:
    st.markdown("### 🖥️ Edge Node")
    st.caption("Local inference stream — no raw data sent to cloud")

    # Live inference chart
    stream_chart = st.empty()
    psi_gauge = st.empty()
    edge_log = st.empty()

with right:
    st.markdown("### ☁️ Cloud Orchestration (AWS)")
    st.caption("EventBridge → Step Functions → SageMaker (mock)")

    m1, m2, m3 = st.columns(3)
    raw_metric = m1.empty()
    sent_metric = m2.empty()
    alert_metric = m3.empty()

    savings_bar = st.empty()
    cloud_status = st.empty()
    cloud_log = st.empty()


# ---------------------------------------------------------------------------
# Render static state (when not running)
# ---------------------------------------------------------------------------
def render_ui():
    data = st.session_state.stream_data
    psi_hist = st.session_state.psi_history

    # ── Stream chart ────────────────────────────────────────────────────────
    if data:
        df = pd.DataFrame(data[-150:], columns=["step", "value"])
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df["step"], y=df["value"],
            mode="lines",
            line=dict(color="#63b3ed", width=2),
            fill="tozeroy",
            fillcolor="rgba(99,179,237,0.07)",
            name="Feature Value",
        ))
        if st.session_state.drift_confirmed_at:
            fig.add_vline(
                x=st.session_state.drift_confirmed_at,
                line_dash="dash",
                line_color="#f56565",
                annotation_text="DRIFT",
                annotation_font_color="#f56565",
            )
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            height=200,
            margin=dict(l=0, r=0, t=10, b=0),
            xaxis=dict(showgrid=False, color="#475569"),
            yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.05)", color="#475569"),
            font=dict(color="#cbd5e1"),
            showlegend=False,
        )
        stream_chart.plotly_chart(fig, use_container_width=True)

    # ── PSI gauge ───────────────────────────────────────────────────────────
    if psi_hist:
        current_psi = psi_hist[-1][1]
        psi_color = "#f56565" if current_psi >= 0.2 else ("#f6ad55" if current_psi >= 0.1 else "#68d391")
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=current_psi,
            number={"suffix": "", "font": {"size": 28, "color": "#e2e8f0"}},
            title={"text": "PSI (Population Stability Index)", "font": {"color": "#94a3b8", "size": 13}},
            gauge={
                "axis": {"range": [0, 0.6], "tickcolor": "#475569"},
                "bar": {"color": psi_color},
                "bgcolor": "rgba(0,0,0,0)",
                "bordercolor": "rgba(255,255,255,0.1)",
                "steps": [
                    {"range": [0, 0.1], "color": "rgba(52,211,153,0.15)"},
                    {"range": [0.1, 0.2], "color": "rgba(246,173,85,0.15)"},
                    {"range": [0.2, 0.6], "color": "rgba(239,68,68,0.15)"},
                ],
                "threshold": {
                    "line": {"color": "#f56565", "width": 2},
                    "thickness": 0.75,
                    "value": 0.2,
                },
            },
        ))
        fig_gauge.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            height=200,
            margin=dict(l=20, r=20, t=20, b=10),
            font=dict(color="#e2e8f0"),
        )
        psi_gauge.plotly_chart(fig_gauge, use_container_width=True)

    # ── Edge log ─────────────────────────────────────────────────────────────
    log_lines = st.session_state.events[-8:] if st.session_state.events else ["[----] Waiting for stream..."]
    edge_log.markdown(
        "<div class='panel'>" +
        "".join(f"<div class='log-entry'>{e}</div>" for e in log_lines) +
        "</div>",
        unsafe_allow_html=True,
    )

    # ── Cloud metrics ─────────────────────────────────────────────────────────
    raw_bytes = st.session_state.raw_bytes_total
    sent_bytes = st.session_state.actual_bytes_sent
    saved_pct = (1 - sent_bytes / max(raw_bytes, 1)) * 100

    raw_metric.metric("Raw Bytes (not sent)", f"{raw_bytes:,}")
    sent_metric.metric("Bytes Sent to AWS", f"{sent_bytes:,}")
    alert_metric.metric("Drift Alerts Fired", st.session_state.alerts_sent)

    # savings bar
    savings_bar.markdown(
        f"<div class='panel'>"
        f"<div style='color:#94a3b8;font-size:12px;margin-bottom:6px;'>Network Egress Savings</div>"
        f"<div style='background:rgba(255,255,255,0.07);border-radius:8px;height:18px;overflow:hidden;'>"
        f"<div style='width:{min(saved_pct,100):.1f}%;height:18px;background:linear-gradient(90deg,#48bb78,#38a169);'></div>"
        f"</div>"
        f"<div style='text-align:right;color:#68d391;font-weight:700;margin-top:4px;'>{saved_pct:.1f}% saved</div>"
        f"</div>",
        unsafe_allow_html=True,
    )

    # cloud status
    if st.session_state.drift_injected and st.session_state.alerts_sent > 0:
        cloud_status.markdown(
            "<div class='alert-critical'>"
            "🚨 EventBridge: ModelDriftDetected → Step Functions → SageMaker Retraining Triggered"
            "</div>",
            unsafe_allow_html=True,
        )
    else:
        cloud_status.markdown(
            "<div class='alert-ok'>✅ Status: Healthy — No cloud alerts</div>",
            unsafe_allow_html=True,
        )

    cloud_log.markdown(
        "<div class='panel'>" +
        "".join(f"<div class='log-entry'>{e}</div>" for e in st.session_state.events[-8:]) +
        "</div>",
        unsafe_allow_html=True,
    )


render_ui()

# ---------------------------------------------------------------------------
# Simulation loop (runs one step per Streamlit rerun)
# ---------------------------------------------------------------------------
if st.session_state.running:
    step = st.session_state.step

    # Choose feature to simulate (use first feature from baseline as proxy)
    if baseline:
        feat_key = list(baseline["features"].keys())[0]
        base_mean = baseline["features"][feat_key]["mean"]
        base_std = float(np.sqrt(baseline["features"][feat_key]["variance"]))
    else:
        base_mean, base_std, feat_key = 0.0, 1.0, "feature"

    # Generate new data point
    if st.session_state.drift_injected:
        new_val = float(np.random.normal(base_mean + base_std * 4, base_std * 2))
    else:
        new_val = float(np.random.normal(base_mean, base_std))

    st.session_state.stream_data.append((step, new_val))
    st.session_state.raw_bytes_total += 64  # ~64 bytes per raw record

    # Compute rolling PSI every 20 steps
    if step % 20 == 0 and step > 0:
        vals = [v for _, v in st.session_state.stream_data[-200:]]
        psi = simulate_psi(vals, {}, feat_key)
        st.session_state.psi_history.append((step, psi))

        if psi >= 0.2 and st.session_state.drift_confirmed_at is None and st.session_state.drift_injected:
            st.session_state.drift_confirmed_at = step
            st.session_state.actual_bytes_sent += 1420  # ~1.4KB payload
            st.session_state.alerts_sent += 1
            st.session_state.events.append(
                f"[{step:04d}] 🚨 DRIFT CONFIRMED: PSI={psi:.3f} ≥ 0.2 — Payload sent (1.4 KB)"
            )
            st.session_state.events.append(
                f"[{step:04d}] 📡 API Gateway → Lambda → S3 + EventBridge"
            )
            st.session_state.events.append(
                f"[{step:04d}] ⚙️  Step Functions execution started"
            )
            st.session_state.events.append(
                f"[{step:04d}] 📲 SNS: 'Model Drift on {feat_key} — Retraining Triggered'"
            )
        elif step % 100 == 0:
            st.session_state.events.append(
                f"[{step:04d}] ✅ PSI={psi:.4f} — No drift. 0 bytes sent."
            )

    st.session_state.step += 1

    # Rerun to produce next frame
    time.sleep(0.08)
    st.rerun()
