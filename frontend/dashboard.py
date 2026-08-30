# Assumption: Streamlit >=1.40; st.rerun() replaces deprecated st.experimental_rerun().
"""
frontend/dashboard.py
======================
TrueSync AI — Cyberpunk / dark-mode Streamlit dashboard.

Connects to the backend WebSocket and streams BiometricFrame JSON at ~30 Hz.
Falls back to an in-process mock generator (no backend imports) when the
connection fails or the user enables "Offline Demo Mode" via the sidebar toggle.
"""
from __future__ import annotations

import math
import os
import random
import time
from collections import deque
from typing import Any, Deque, Optional

import cv2
import numpy as np
import plotly.graph_objects as go
import streamlit as st
import websocket
from dotenv import load_dotenv

load_dotenv()

# ── Page config (must be first Streamlit call) ────────────────────────────────
st.set_page_config(
    page_title="TrueSync AI — INNO-CREW",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Constants ─────────────────────────────────────────────────────────────────
_DEFAULT_WS_URL: str = os.getenv("WEBSOCKET_URL", "ws://localhost:8000/ws/session")
_DEFAULT_CAM: str = "0"
_WAVEFORM_LEN: int = 90
_RECONNECT_WAIT: float = 2.0

# ── CSS injection ─────────────────────────────────────────────────────────────
_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;400;600;700&family=Orbitron:wght@700;900&display=swap');

html, body, [class*="css"] {
    background-color: #0a0e14 !important;
    color: #c8d6e5 !important;
    font-family: 'JetBrains Mono', 'Courier New', monospace !important;
}

/* Header */
.ts-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 1rem 0 0.5rem 0;
    border-bottom: 1px solid #00f0ff33;
    margin-bottom: 1.2rem;
}
.ts-title {
    font-family: 'Orbitron', sans-serif;
    font-size: 1.9rem;
    font-weight: 900;
    background: linear-gradient(90deg, #00f0ff, #ff2fd0);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    letter-spacing: 0.05em;
    margin: 0;
    line-height: 1.1;
}
.ts-subtitle {
    font-size: 0.72rem;
    color: #00f0ff99;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    margin-top: 0.25rem;
}
.ts-logo-fallback {
    display: flex;
    align-items: center;
    justify-content: center;
    width: 120px;
    height: 48px;
    border: 1px solid #00f0ff55;
    border-radius: 6px;
    font-family: 'Orbitron', sans-serif;
    font-size: 0.65rem;
    color: #00f0ff;
    letter-spacing: 0.1em;
    background: #0d1520;
    box-shadow: 0 0 8px #00f0ff22;
}

/* Metric cards */
.ts-card-row {
    display: flex;
    gap: 1rem;
    margin-bottom: 1rem;
    flex-wrap: wrap;
}
.ts-card {
    flex: 1;
    min-width: 140px;
    background: #0d1520;
    border: 1px solid #00f0ff22;
    border-radius: 10px;
    padding: 0.85rem 1.1rem;
    box-shadow: 0 0 12px #00f0ff0d;
    transition: box-shadow 0.25s ease;
}
.ts-card:hover { box-shadow: 0 0 24px #00f0ff28; }
.ts-card-label {
    font-size: 0.62rem;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: #5a7fa0;
    margin-bottom: 0.35rem;
}
.ts-card-value {
    font-family: 'JetBrains Mono', monospace;
    font-size: 1.55rem;
    font-weight: 700;
    color: #00f0ff;
    text-shadow: 0 0 10px #00f0ff66;
}
.ts-card-unit {
    font-size: 0.72rem;
    color: #5a7fa0;
    margin-left: 0.2rem;
}

/* Status badge */
.ts-badge {
    display: inline-block;
    padding: 0.22rem 0.7rem;
    border-radius: 20px;
    font-size: 0.65rem;
    font-weight: 700;
    letter-spacing: 0.1em;
    text-transform: uppercase;
}
.ts-badge-nominal    { background:#00ff8822; color:#00ff88; border:1px solid #00ff8855; }
.ts-badge-calibrating{ background:#ffcc0022; color:#ffcc00; border:1px solid #ffcc0055; }
.ts-badge-low_confidence { background:#ff990022; color:#ff9900; border:1px solid #ff990055; }
.ts-badge-spoof_suspected{ background:#ff2f2f22; color:#ff4444; border:1px solid #ff444455;
    animation: blink 1s step-start infinite; }
@keyframes blink { 50% { opacity: 0.4; } }

/* Offline mode banner */
.ts-offline-banner {
    background: #1a0a00;
    border: 1px solid #ff6600;
    border-radius: 8px;
    padding: 0.5rem 1rem;
    color: #ff9955;
    font-size: 0.75rem;
    letter-spacing: 0.08em;
    text-align: center;
    margin-bottom: 0.8rem;
    box-shadow: 0 0 12px #ff660033;
}

/* Section label */
.ts-section {
    font-size: 0.62rem;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color: #00f0ff66;
    border-bottom: 1px solid #00f0ff1a;
    padding-bottom: 0.3rem;
    margin: 0.8rem 0 0.6rem 0;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background-color: #080c12 !important;
    border-right: 1px solid #00f0ff1a;
}
</style>
"""
st.markdown(_CSS, unsafe_allow_html=True)


# ── Session state init ────────────────────────────────────────────────────────
def _init_state() -> None:
    defaults: dict[str, Any] = {
        "offline_mode": False,
        "ws_error": None,
        "waveform": deque([0.0] * _WAVEFORM_LEN, maxlen=_WAVEFORM_LEN),
        "frame": None,
        # Offline mock state
        "off_phase": 0.0,
        "off_bpm": 72.0,
        "off_trust": 0.90,
        "off_liveness": 0.91,
        "off_audio": 0.88,
        "off_sync": 0.86,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init_state()


# ── Offline mock generator (no backend imports) ───────────────────────────────
class _OfflineMock:
    """
    Lightweight in-process replica of MockEngine math.
    Intentionally self-contained — never imports backend modules.
    """
    _rng = random.Random()

    @classmethod
    def tick(cls) -> dict[str, Any]:
        fps = 30
        delta_bpm = cls._rng.gauss(0.0, 0.15)
        st.session_state["off_bpm"] = max(60.0, min(90.0, st.session_state["off_bpm"] + delta_bpm))
        bpm = st.session_state["off_bpm"]

        omega = 2.0 * math.pi * (bpm / 60.0) / fps
        st.session_state["off_phase"] = (st.session_state["off_phase"] + omega) % (2.0 * math.pi)
        phase = st.session_state["off_phase"]

        bvp = (
            0.70 * math.sin(phase)
            + 0.20 * math.sin(2.0 * phase - 0.3)
            + cls._rng.gauss(0.0, 0.04)
        )
        wf: Deque[float] = st.session_state["waveform"]
        wf.append(bvp)

        dt = cls._rng.gauss(0.0, 0.005)
        st.session_state["off_trust"] = max(0.82, min(0.96, st.session_state["off_trust"] + dt))
        trust = st.session_state["off_trust"]

        def _sub(bias: float) -> float:
            return max(0.0, min(1.0, trust + bias + cls._rng.gauss(0.0, 0.008)))

        return {
            "timestamp": time.time(),
            "session_id": "offline-demo",
            "engine_mode": "mock",
            "bpm": round(bpm, 1),
            "rppg_waveform": list(wf),
            "rppg_liveness": round(_sub(0.02), 4),
            "audio_trust": round(_sub(-0.01), 4),
            "sync_score": round(_sub(-0.03), 4),
            "overall_trust": round(trust * 100.0, 2),
            "active_rois": ["forehead", "left_cheek", "right_cheek", "nose_bridge"],
            "status_flag": "nominal",
        }


# ── WebSocket fetch (one frame) ───────────────────────────────────────────────
def _fetch_ws_frame(ws_url: str) -> Optional[dict[str, Any]]:
    """Open a WebSocket, receive one JSON frame, close. Returns None on error."""
    try:
        ws = websocket.create_connection(ws_url, timeout=2)
        try:
            raw = ws.recv()
            import json
            data = json.loads(raw)
            wf: Deque[float] = st.session_state["waveform"]
            for v in data.get("rppg_waveform", []):
                wf.append(float(v))
            st.session_state["ws_error"] = None
            return data
        finally:
            ws.close()
    except Exception as exc:
        st.session_state["ws_error"] = str(exc)
        st.session_state["offline_mode"] = True
        return None


# ── Chart builders ────────────────────────────────────────────────────────────
_BG = "#0a0e14"
_PAPER = "#0d1520"
_CYAN = "#00f0ff"
_MAGENTA = "#ff2fd0"
_GRID = "#1a2535"


def _waveform_chart(waveform: list[float]) -> go.Figure:
    xs = list(range(len(waveform)))
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=xs, y=waveform,
        mode="lines",
        line=dict(color=_CYAN, width=1.8, shape="spline", smoothing=0.8),
        fill="tozeroy",
        fillcolor="rgba(0,240,255,0.06)",
        name="BVP",
        hoverinfo="skip",
    ))
    fig.update_layout(
        paper_bgcolor=_PAPER, plot_bgcolor=_BG,
        margin=dict(l=8, r=8, t=8, b=8),
        height=160,
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=True, gridcolor=_GRID, zeroline=True,
                   zerolinecolor="#1e3040", showticklabels=False,
                   range=[-1.4, 1.4]),
        showlegend=False,
    )
    return fig


def _trust_gauge(trust: float) -> go.Figure:
    color = _CYAN if trust >= 85 else ("#ffcc00" if trust >= 70 else _MAGENTA)
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=trust,
        number=dict(font=dict(family="JetBrains Mono", size=36, color=color), suffix=""),
        gauge=dict(
            axis=dict(range=[0, 100], tickfont=dict(color="#5a7fa0", size=10),
                      tickcolor="#1a2535", nticks=6),
            bar=dict(color=color, thickness=0.28),
            bgcolor=_BG,
            borderwidth=0,
            steps=[
                dict(range=[0, 70], color="#0d1520"),
                dict(range=[70, 85], color="#111a26"),
                dict(range=[85, 100], color="#0d1f2d"),
            ],
            threshold=dict(line=dict(color=color, width=3), thickness=0.8, value=trust),
        ),
        title=dict(text="OVERALL TRUST", font=dict(family="Orbitron", size=11, color="#5a7fa0")),
        domain=dict(x=[0, 1], y=[0, 1]),
    ))
    fig.update_layout(
        paper_bgcolor=_PAPER, plot_bgcolor=_BG,
        margin=dict(l=20, r=20, t=30, b=10),
        height=240,
        font=dict(color="#c8d6e5"),
    )
    return fig


def _mini_bar(label: str, value: float, color: str = _CYAN) -> str:
    pct = int(value * 100)
    w = max(2, pct)
    return f"""
<div class="ts-card">
  <div class="ts-card-label">{label}</div>
  <div style="background:#0a0e14;border-radius:4px;height:6px;margin-bottom:0.5rem;overflow:hidden;">
    <div style="width:{w}%;height:100%;background:{color};border-radius:4px;
         box-shadow:0 0 6px {color}88;transition:width 0.4s ease;"></div>
  </div>
  <div class="ts-card-value" style="font-size:1.1rem;color:{color};">{pct}<span class="ts-card-unit">%</span></div>
</div>"""


def _badge_html(flag: str) -> str:
    cls = f"ts-badge ts-badge-{flag}"
    labels = {
        "nominal": "● NOMINAL",
        "calibrating": "◌ CALIBRATING",
        "low_confidence": "▲ LOW CONFIDENCE",
        "spoof_suspected": "✕ SPOOF SUSPECTED",
    }
    return f'<span class="{cls}">{labels.get(flag, flag)}</span>'


# ── Webcam frame capture ──────────────────────────────────────────────────────
def _capture_frame(cam_src: str) -> Optional[np.ndarray]:
    try:
        src: Any = int(cam_src) if cam_src.strip().isdigit() else cam_src
        cap = cv2.VideoCapture(src)
        try:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
            ok, frame = cap.read()
            if ok:
                return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            return None
        finally:
            cap.release()
    except Exception:
        return None


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<p class="ts-section">⚙ CONNECTION</p>', unsafe_allow_html=True)
    ws_url = st.text_input("WebSocket URL", value=_DEFAULT_WS_URL, key="ws_url_input")
    cam_src = st.text_input("Camera Source (index or RTSP URL)", value=_DEFAULT_CAM, key="cam_input")

    st.markdown('<p class="ts-section">⚡ ENGINE</p>', unsafe_allow_html=True)
    engine_label = os.getenv("ENGINE_MODE", "mock").upper()
    st.markdown(
        f'<div style="font-size:0.7rem;color:#5a7fa0;">MODE</div>'
        f'<div style="font-family:Orbitron,sans-serif;font-size:1rem;color:#00f0ff;">{engine_label}</div>',
        unsafe_allow_html=True,
    )

    st.markdown('<p class="ts-section">🔌 DEMO SAFETY</p>', unsafe_allow_html=True)
    force_offline = st.toggle("Force Offline Demo Mode", value=False, key="force_offline_toggle")
    if force_offline:
        st.session_state["offline_mode"] = True
    elif not force_offline and st.session_state.get("force_offline_toggle_prev", True):
        st.session_state["offline_mode"] = False
    st.session_state["force_offline_toggle_prev"] = force_offline

    st.markdown(
        '<p style="font-size:0.58rem;color:#2a3d50;margin-top:2rem;">'
        'TrueSync AI · INNO-CREW · IEEE WIE ILS 2026</p>',
        unsafe_allow_html=True,
    )


# ── Header ────────────────────────────────────────────────────────────────────
col_logo, col_title = st.columns([1, 4])
with col_logo:
    try:
        st.image("frontend/assets/logo.png", width=110)
    except (FileNotFoundError, Exception):
        st.markdown(
            '<div class="ts-logo-fallback">INNO‑CREW</div>',
            unsafe_allow_html=True,
        )
with col_title:
    st.markdown(
        '<div class="ts-header">'
        '  <div>'
        '    <div class="ts-title">TrueSync AI</div>'
        '    <div class="ts-subtitle">Zero-Trust Liveness &amp; Anti-Deepfake Engine'
        '    &nbsp;·&nbsp; IEEE WIE ILS 2026 · Track 4</div>'
        '  </div>'
        '</div>',
        unsafe_allow_html=True,
    )


# ── Offline banner ────────────────────────────────────────────────────────────
if st.session_state["offline_mode"]:
    err_detail = f" ({st.session_state['ws_error']})" if st.session_state.get("ws_error") else ""
    st.markdown(
        f'<div class="ts-offline-banner">'
        f'  📡 OFFLINE DEMO MODE — backend unavailable{err_detail}. '
        f'  Rendering local mock telemetry.'
        f'</div>',
        unsafe_allow_html=True,
    )


# ── Data fetch ────────────────────────────────────────────────────────────────
if st.session_state["offline_mode"]:
    frame_data = _OfflineMock.tick()
else:
    frame_data = _fetch_ws_frame(ws_url)
    if frame_data is None:
        frame_data = _OfflineMock.tick()


# ── Main layout ───────────────────────────────────────────────────────────────
left_col, right_col = st.columns([3, 2], gap="medium")

with left_col:
    # Metric cards row
    st.markdown('<p class="ts-section">📊 LIVE METRICS</p>', unsafe_allow_html=True)
    bpm_val = frame_data.get("bpm") or 0.0
    overall = frame_data.get("overall_trust") or 0.0
    flag = frame_data.get("status_flag", "calibrating")
    ts_val = frame_data.get("timestamp", time.time())

    cards_html = f"""
<div class="ts-card-row">
  <div class="ts-card">
    <div class="ts-card-label">❤ Heart Rate</div>
    <div class="ts-card-value">{bpm_val:.1f}<span class="ts-card-unit">BPM</span></div>
  </div>
  <div class="ts-card">
    <div class="ts-card-label">🛡 Overall Trust</div>
    <div class="ts-card-value">{overall:.1f}<span class="ts-card-unit">/ 100</span></div>
  </div>
  <div class="ts-card">
    <div class="ts-card-label">🚦 Status</div>
    <div style="margin-top:0.4rem;">{_badge_html(flag)}</div>
  </div>
  <div class="ts-card">
    <div class="ts-card-label">🕐 Timestamp</div>
    <div class="ts-card-value" style="font-size:0.85rem;color:#5a9abf;">
      {time.strftime('%H:%M:%S', time.localtime(ts_val))}
    </div>
  </div>
</div>"""
    st.markdown(cards_html, unsafe_allow_html=True)

    # Sub-score bars
    st.markdown('<p class="ts-section">🔬 SUB-SCORES</p>', unsafe_allow_html=True)
    liveness = frame_data.get("rppg_liveness") or 0.0
    audio = frame_data.get("audio_trust") or 0.0
    sync = frame_data.get("sync_score") or 0.0
    sub_html = (
        '<div class="ts-card-row">'
        + _mini_bar("rPPG Liveness", liveness, _CYAN)
        + _mini_bar("Audio Trust", audio, "#b388ff")
        + _mini_bar("Viseme Sync", sync, _MAGENTA)
        + "</div>"
    )
    st.markdown(sub_html, unsafe_allow_html=True)

    # Waveform chart
    st.markdown('<p class="ts-section">〰 rPPG PULSE WAVEFORM</p>', unsafe_allow_html=True)
    waveform_data = list(st.session_state["waveform"])
    st.plotly_chart(_waveform_chart(waveform_data), use_container_width=True, config={"displayModeBar": False})

    # Active ROIs
    rois = frame_data.get("active_rois", [])
    roi_html = " &nbsp; ".join(
        f'<span style="background:#0d1f2d;border:1px solid #00f0ff33;border-radius:4px;'
        f'padding:0.15rem 0.5rem;font-size:0.65rem;color:#00f0ff99;">{r}</span>'
        for r in rois
    )
    st.markdown(
        f'<p class="ts-section">🎯 ACTIVE ROIs</p><div style="margin-bottom:0.8rem;">{roi_html}</div>',
        unsafe_allow_html=True,
    )

with right_col:
    # Trust gauge
    st.markdown('<p class="ts-section">⚡ TRUST GAUGE</p>', unsafe_allow_html=True)
    st.plotly_chart(_trust_gauge(overall), use_container_width=True, config={"displayModeBar": False})

    # Webcam preview
    st.markdown('<p class="ts-section">📷 LIVE PREVIEW</p>', unsafe_allow_html=True)
    cam_frame = _capture_frame(cam_src)
    if cam_frame is not None:
        st.image(cam_frame, channels="RGB", use_container_width=True)
    else:
        st.markdown(
            '<div style="background:#0d1520;border:1px dashed #1a2e40;border-radius:8px;'
            'height:200px;display:flex;align-items:center;justify-content:center;'
            'color:#2a4060;font-size:0.75rem;letter-spacing:0.1em;">'
            'CAMERA UNAVAILABLE</div>',
            unsafe_allow_html=True,
        )

    # Engine info
    mode_color = "#00f0ff" if frame_data.get("engine_mode") == "mock" else "#ff2fd0"
    session_short = str(frame_data.get("session_id", "—"))[:16]
    st.markdown(
        f'<div style="background:#0d1520;border:1px solid #1a2e40;border-radius:8px;'
        f'padding:0.8rem 1rem;margin-top:0.5rem;font-size:0.68rem;">'
        f'  <div style="color:#5a7fa0;letter-spacing:0.1em;text-transform:uppercase;margin-bottom:0.4rem;">ENGINE INFO</div>'
        f'  <div>Mode: <span style="color:{mode_color};">{frame_data.get("engine_mode","—").upper()}</span></div>'
        f'  <div style="margin-top:0.25rem;">Session: <span style="color:#5a9abf;">{session_short}…</span></div>'
        f'</div>',
        unsafe_allow_html=True,
    )


# ── Auto-refresh ──────────────────────────────────────────────────────────────
time.sleep(0.034)   # ~30 Hz
st.rerun()
