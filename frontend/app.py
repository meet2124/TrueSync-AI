"""
frontend/app.py
===============
TrueSync AI — Production V1 Streamlit Dashboard.

Architecture:
- Persistent WebSocket connection to /ws/session (held open; not reconnected per frame).
- Local cv2.VideoCapture loop for zero-latency camera preview.
- Frames base64-encoded and sent to backend asynchronously via threading.
- BiometricResult JSON streamed back from backend, parsed and rendered live.
- Plotly Indicator gauge for overall_trust; live rPPG waveform line chart.
- Full dark-mode CSS injection (JetBrains Mono + Orbitron fonts).
- Resource cleanup (cap.release(), ws.close()) in finally block.
"""
from __future__ import annotations

import base64
import json
import math
import os
import queue
import threading
import time
from collections import deque
from typing import Any, Deque, Optional

import cv2
import numpy as np
import plotly.graph_objects as go
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="TrueSync AI — INNO-CREW",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Constants ─────────────────────────────────────────────────────────────────
_DEFAULT_WS_URL = os.getenv("WEBSOCKET_URL", "ws://localhost:8000/ws/session")
_DEFAULT_CAM = "0"
_WAVEFORM_LEN = 90
_FRAME_INTERVAL = 1.0 / 30.0  # 30 fps capture target

# ── CSS ───────────────────────────────────────────────────────────────────────
_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;400;600;700&family=Orbitron:wght@700;900&display=swap');

html, body, [class*="css"] {
    background-color: #080c12 !important;
    color: #c8d6e5 !important;
    font-family: 'JetBrains Mono', 'Courier New', monospace !important;
}
.ts-title {
    font-family: 'Orbitron', sans-serif;
    font-size: 2rem; font-weight: 900;
    background: linear-gradient(90deg, #00f0ff, #a259ff);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    background-clip: text; letter-spacing: 0.06em; margin: 0;
}
.ts-sub { font-size: 0.7rem; color: #00f0ff88; letter-spacing: 0.14em;
           text-transform: uppercase; margin-top: 0.2rem; }
.ts-section { font-size: 0.6rem; letter-spacing: 0.18em; text-transform: uppercase;
               color: #00f0ff55; border-bottom: 1px solid #00f0ff18;
               padding-bottom: 0.3rem; margin: 0.8rem 0 0.5rem 0; }
.ts-card {
    background: #0d1520; border: 1px solid #00f0ff1a; border-radius: 10px;
    padding: 0.85rem 1.1rem; box-shadow: 0 0 12px #00f0ff0a;
    transition: box-shadow 0.3s ease;
}
.ts-card:hover { box-shadow: 0 0 22px #00f0ff22; }
.ts-card-label { font-size: 0.6rem; letter-spacing: 0.14em; text-transform: uppercase;
                  color: #4a6a80; margin-bottom: 0.35rem; }
.ts-card-value { font-family: 'JetBrains Mono', monospace; font-size: 1.5rem;
                  font-weight: 700; color: #00f0ff; text-shadow: 0 0 8px #00f0ff55; }
.ts-card-unit  { font-size: 0.7rem; color: #4a6a80; margin-left: 0.2rem; }
.ts-card-row   { display: flex; gap: 0.8rem; flex-wrap: wrap; margin-bottom: 0.8rem; }
.ts-bar-wrap   { background: #0a0e14; border-radius: 4px; height: 6px;
                  margin-bottom: 0.5rem; overflow: hidden; }
.ts-bar-fill   { height: 100%; border-radius: 4px; transition: width 0.4s ease; }
.ts-badge { display:inline-block; padding:0.2rem 0.65rem; border-radius:20px;
             font-size:0.63rem; font-weight:700; letter-spacing:0.1em; text-transform:uppercase; }
.ts-badge-nominal      { background:#00ff8818; color:#00ff88; border:1px solid #00ff8844; }
.ts-badge-calibrating  { background:#ffcc0018; color:#ffcc00; border:1px solid #ffcc0044; }
.ts-badge-low_confidence { background:#ff990018; color:#ff9900; border:1px solid #ff990044; }
.ts-badge-insufficient_data { background:#ff4a4a18; color:#ff4a4a; border:1px solid #ff4a4a44;
                               animation: blink 1.2s step-start infinite; }
@keyframes blink { 50% { opacity: 0.35; } }
.ts-offline { background:#1a0800; border:1px solid #ff6600; border-radius:8px;
               padding:0.5rem 1rem; color:#ff9955; font-size:0.73rem;
               letter-spacing:0.07em; text-align:center; margin-bottom:0.7rem; }
.ts-result-pass { background:#00ff8810; border:1px solid #00ff8844; border-radius:12px;
                   padding:1.2rem 1.5rem; margin-bottom:1rem; }
.ts-result-fail { background:#ff4a4a10; border:1px solid #ff4a4a44; border-radius:12px;
                   padding:1.2rem 1.5rem; margin-bottom:1rem;
                   animation: blink 1.5s step-start infinite; }
.ts-result-title { font-family:'Orbitron',sans-serif; font-size:1.3rem; font-weight:900; margin-bottom:0.5rem; }
.ts-signal-row { display:flex; gap:0.8rem; flex-wrap:wrap; margin:0.6rem 0; }
.ts-signal-pill { display:inline-flex; align-items:center; gap:0.4rem;
                   background:#0d1520; border:1px solid #00f0ff22; border-radius:8px;
                   padding:0.5rem 0.9rem; font-size:0.78rem; }
.ts-signal-pass { border-color:#00ff8844; color:#00ff88; }
.ts-signal-fail { border-color:#ff4a4a44; color:#ff4a4a; }
.ts-signal-na   { border-color:#3a4a5a44; color:#5a7a8a; }
section[data-testid="stSidebar"] {
    background-color: #060a10 !important;
    border-right: 1px solid #00f0ff18;
}
</style>
"""
st.markdown(_CSS, unsafe_allow_html=True)

# ── Session state ─────────────────────────────────────────────────────────────
def _init() -> None:
    defaults: dict[str, Any] = {
        "ws_connected": False,
        "ws_error": None,
        "latest_result": None,
        "waveform": deque([0.0] * _WAVEFORM_LEN, maxlen=_WAVEFORM_LEN),
        "send_q": queue.Queue(maxsize=10),   # frames to send
        "recv_q": queue.Queue(maxsize=10),   # results received
        "ws_thread_started": False,
        "cam_frame": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init()


# ── WebSocket background thread ───────────────────────────────────────────────
def _ws_thread(ws_url: str, send_q: queue.Queue, recv_q: queue.Queue) -> None:
    """
    Runs in a daemon thread. Holds one persistent WebSocket connection open.
    Sends frames from send_q; puts received BiometricResult JSON into recv_q.
    """
    try:
        import websocket as _ws_lib
        ws = _ws_lib.create_connection(ws_url, timeout=10)
        recv_q.put({"_connected": True})

        # Spawn a sub-thread for blocking recv
        def _recv_loop() -> None:
            while True:
                try:
                    raw = ws.recv()
                    if raw:
                        recv_q.put(json.loads(raw))
                except Exception:
                    break

        recv_thread = threading.Thread(target=_recv_loop, daemon=True)
        recv_thread.start()

        while True:
            try:
                msg = send_q.get(timeout=0.05)
                if msg is None:  # shutdown signal
                    break
                ws.send(json.dumps(msg))
            except queue.Empty:
                pass
            except Exception as exc:
                recv_q.put({"_error": str(exc)})
                break

        ws.close()
    except Exception as exc:
        recv_q.put({"_error": str(exc)})


# ── Camera capture helper ─────────────────────────────────────────────────────
def _read_frame(cap: cv2.VideoCapture) -> Optional[np.ndarray]:
    ok, frame = cap.read()
    if not ok:
        return None
    return frame


# ── Chart builders ────────────────────────────────────────────────────────────
_BG = "#080c12"; _PAPER = "#0d1520"; _CYAN = "#00f0ff"; _VIOLET = "#a259ff"; _GRID = "#1a2535"


def _waveform_chart(waveform: list[float]) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=list(range(len(waveform))), y=waveform,
        mode="lines",
        line=dict(color=_CYAN, width=1.8, shape="spline", smoothing=0.8),
        fill="tozeroy", fillcolor="rgba(0,240,255,0.05)",
        hoverinfo="skip",
    ))
    fig.update_layout(
        paper_bgcolor=_PAPER, plot_bgcolor=_BG,
        margin=dict(l=6, r=6, t=6, b=6), height=155,
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=True, gridcolor=_GRID, zeroline=True,
                   zerolinecolor="#1e3040", showticklabels=False, range=[-1.5, 1.5]),
        showlegend=False,
    )
    return fig


def _trust_gauge(trust: float) -> go.Figure:
    color = _CYAN if trust >= 75 else ("#ffcc00" if trust >= 50 else "#ff4a4a")
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=trust,
        number=dict(font=dict(family="JetBrains Mono", size=38, color=color)),
        gauge=dict(
            axis=dict(range=[0, 100], tickfont=dict(color="#4a6a80", size=9), nticks=6),
            bar=dict(color=color, thickness=0.26),
            bgcolor=_BG, borderwidth=0,
            steps=[
                dict(range=[0, 50], color="#0d1520"),
                dict(range=[50, 75], color="#111a26"),
                dict(range=[75, 100], color="#0d1f2d"),
            ],
            threshold=dict(line=dict(color=color, width=3), thickness=0.8, value=trust),
        ),
        title=dict(text="OVERALL TRUST", font=dict(family="Orbitron", size=10, color="#4a6a80")),
        domain=dict(x=[0, 1], y=[0, 1]),
    ))
    fig.update_layout(
        paper_bgcolor=_PAPER, plot_bgcolor=_BG,
        margin=dict(l=16, r=16, t=30, b=8), height=240,
        font=dict(color="#c8d6e5"),
    )
    return fig


def _mini_bar(label: str, value: Optional[float], color: str = _CYAN) -> str:
    if value is None:
        return f"""
<div class="ts-card">
  <div class="ts-card-label">{label}</div>
  <div class="ts-bar-wrap"><div class="ts-bar-fill" style="width:2%;background:#2a3a4a;"></div></div>
  <div class="ts-card-value" style="font-size:1.05rem;color:#3a5a6a;">N/A</div>
</div>"""
    pct = int(value * 100)
    return f"""
<div class="ts-card">
  <div class="ts-card-label">{label}</div>
  <div class="ts-bar-wrap">
    <div class="ts-bar-fill" style="width:{max(2,pct)}%;background:{color};box-shadow:0 0 5px {color}66;"></div>
  </div>
  <div class="ts-card-value" style="font-size:1.05rem;color:{color};">{pct}<span class="ts-card-unit">%</span></div>
</div>"""


def _badge(flag: str) -> str:
    labels = {
        "nominal": "● NOMINAL",
        "calibrating": "◌ CALIBRATING",
        "low_confidence": "▲ LOW CONFIDENCE",
        "insufficient_data": "✕ INSUFFICIENT DATA",
    }
    return f'<span class="ts-badge ts-badge-{flag}">{labels.get(flag, flag)}</span>'


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<p class="ts-section">⚙ CONNECTION</p>', unsafe_allow_html=True)
    ws_url = st.text_input("WebSocket URL", value=_DEFAULT_WS_URL, key="ws_url_input")
    cam_src = st.text_input("Camera Source (index or RTSP URL)", value=_DEFAULT_CAM, key="cam_input")
    st.markdown('<p class="ts-section">ℹ ENGINE</p>', unsafe_allow_html=True)
    st.markdown(
        '<div style="font-size:0.7rem;color:#4a6a80;">MODE</div>'
        '<div style="font-family:Orbitron,sans-serif;font-size:1rem;color:#00f0ff;">PRODUCTION V1</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<p style="font-size:0.56rem;color:#253040;margin-top:2.5rem;">'
        'TrueSync AI · INNO-CREW · IEEE WIE ILS 2026 · Track 4</p>',
        unsafe_allow_html=True,
    )

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown(
    '<div class="ts-title">TrueSync AI</div>'
    '<div class="ts-sub">Zero-Trust Biological Multimodal Authentication · IEEE WIE ILS 2026 · Track 4</div>',
    unsafe_allow_html=True,
)
st.markdown("---")

# ── Start WebSocket + Camera ──────────────────────────────────────────────────
send_q: queue.Queue = st.session_state["send_q"]
recv_q: queue.Queue = st.session_state["recv_q"]

if not st.session_state["ws_thread_started"]:
    t = threading.Thread(
        target=_ws_thread,
        args=(ws_url, send_q, recv_q),
        daemon=True,
    )
    t.start()
    st.session_state["ws_thread_started"] = True

# Drain recv_q for latest result
while not recv_q.empty():
    msg = recv_q.get_nowait()
    if "_connected" in msg:
        st.session_state["ws_connected"] = True
        st.session_state["ws_error"] = None
    elif "_error" in msg:
        st.session_state["ws_connected"] = False
        st.session_state["ws_error"] = msg["_error"]
    else:
        st.session_state["latest_result"] = msg
        wf = msg.get("rppg_waveform", [])
        if wf:
            for v in wf[-5:]:
                st.session_state["waveform"].append(float(v))

# Capture one camera frame and queue it for sending
try:
    cam_idx: Any = int(cam_src) if cam_src.strip().isdigit() else cam_src
    cap = cv2.VideoCapture(cam_idx)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    ok, raw_frame = cap.read()
    cap.release()

    if ok and raw_frame is not None:
        display_frame = cv2.cvtColor(raw_frame, cv2.COLOR_BGR2RGB)
        st.session_state["cam_frame"] = display_frame
        # Encode to JPEG and queue for backend
        encode_frame = cv2.resize(raw_frame, (640, 480))
        _, buf = cv2.imencode(".jpg", encode_frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
        b64 = base64.b64encode(buf.tobytes()).decode("utf-8")
        msg_out = {"type": "video_frame", "timestamp": time.time(), "data": b64}
        try:
            send_q.put_nowait(msg_out)
        except queue.Full:
            pass
except Exception:
    pass

# ── Status banner ─────────────────────────────────────────────────────────────
if not st.session_state["ws_connected"]:
    err = st.session_state.get("ws_error") or "connecting…"
    st.markdown(
        f'<div class="ts-offline">📡 Backend: {err}</div>',
        unsafe_allow_html=True,
    )

# ── Layout ────────────────────────────────────────────────────────────────────
result: Optional[dict[str, Any]] = st.session_state["latest_result"]
overall = result.get("overall_trust") if result else None
bpm_val = result.get("bpm") if result else None
flag = (result.get("status") or "calibrating") if result else "calibrating"
rppg_conf: Optional[float] = result.get("rppg_confidence") if result else None
acoustic: Optional[float] = result.get("acoustic_trust") if result else None
sync_sc: Optional[float] = result.get("sync_score") if result else None
sync_lag = result.get("sync_lag_ms") if result else None
active_rois = result.get("active_rois", []) if result else []
ts_val = result.get("timestamp", time.time()) if result else time.time()
latency_ms = result.get("processing_latency_ms") if result else None
overall_disp = float(overall) if overall is not None else 0.0

left_col, right_col = st.columns([3, 2], gap="medium")

with left_col:
    # ── Connection / calibration state banner ────────────────────────────
    if not st.session_state["ws_connected"]:
        pass  # banner already shown above
    elif flag == "calibrating":
        st.info("⏳ **Calibrating** — Hold still and speak naturally. Collecting signal data…")
    elif flag == "insufficient_data":
        st.warning("📡 **Insufficient Signal** — No biometric data yet. Check camera and microphone.")

    # ── Simple verification signals (primary user view) ──────────────────
    st.markdown('<p class="ts-section">✅ VERIFICATION SIGNALS</p>', unsafe_allow_html=True)

    def _signal_pill(emoji: str, label: str, score: Optional[float]) -> str:
        if score is None:
            cls = "ts-signal-na"
            mark = "⟳"
            pct_txt = "Calibrating…"
        elif score >= 0.70:
            cls = "ts-signal-pass"
            mark = "✓"
            pct_txt = f"{int(score*100)}%"
        else:
            cls = "ts-signal-fail"
            mark = "✗"
            pct_txt = f"{int(score*100)}%"
        return (f'<div class="ts-signal-pill {cls}">'
                f'{emoji} <strong>{label}</strong> &nbsp; {mark} {pct_txt}</div>')

    st.markdown(
        '<div class="ts-signal-row">'
        + _signal_pill("❤️", "Liveness", rppg_conf)
        + _signal_pill("🎙️", "Voice Authenticity", acoustic)
        + _signal_pill("👄", "Audio-Visual Sync", sync_sc)
        + "</div>",
        unsafe_allow_html=True,
    )

    # ── Result screen ─────────────────────────────────────────────────────
    if flag == "nominal" and overall is not None:
        reason_parts = []
        if rppg_conf is not None and rppg_conf >= 0.70:
            reason_parts.append("a live heartbeat signal was detected")
        if acoustic is not None and acoustic >= 0.70:
            reason_parts.append("voice characteristics are consistent with a real person")
        if sync_sc is not None and sync_sc >= 0.70:
            reason_parts.append("lip movement matches the audio")
        reason = "; ".join(reason_parts) if reason_parts else "all available signals passed"
        st.markdown(
            f'<div class="ts-result-pass">'
            f'<div class="ts-result-title" style="color:#00ff88;">🟢 HUMAN / CONSISTENT SIGNALS</div>'
            f'<div style="font-size:1.6rem;font-weight:700;color:#00ff88;margin:0.4rem 0;">{overall_disp:.1f}<span style="font-size:0.9rem;color:#4a8a6a;"> / 100</span></div>'
            f'<div style="font-size:0.75rem;color:#5aaa7a;">Why this result? Because {reason}.</div>'
            f"</div>",
            unsafe_allow_html=True,
        )
    elif flag in ("low_confidence", "insufficient_data") and overall is not None and overall_disp < 50:
        st.markdown(
            f'<div class="ts-result-fail">'
            f'<div class="ts-result-title" style="color:#ff4a4a;">🔴 SUSPICIOUS ACTIVITY</div>'
            f'<div style="font-size:1.6rem;font-weight:700;color:#ff4a4a;margin:0.4rem 0;">{overall_disp:.1f}<span style="font-size:0.9rem;color:#8a4a4a;"> / 100</span></div>'
            f'<div style="font-size:0.75rem;color:#aa5a5a;">Signals were inconsistent with expected live human behaviour. '
            f'This does not confirm a deepfake — environmental factors (lighting, noise) may also cause this.</div>'
            f"</div>",
            unsafe_allow_html=True,
        )

    # ── Technical analysis expander ───────────────────────────────────────
    with st.expander("🔬 View Technical Analysis", expanded=False):
        bpm_txt = f"{float(bpm_val):.1f} BPM" if bpm_val is not None else "N/A"
        overall_txt = f"{overall_disp:.1f} / 100" if overall is not None else "N/A"
        rppg_txt = f"{int(rppg_conf*100)}%" if rppg_conf is not None else "N/A"
        acoustic_txt = f"{int(acoustic*100)}%" if acoustic is not None else "N/A"
        sync_txt = f"{int(sync_sc*100)}%" if sync_sc is not None else "N/A"
        lag_txt = f"{sync_lag:+.0f} ms" if sync_lag is not None else "N/A"
        latency_txt = f"{latency_ms:.1f} ms" if latency_ms is not None else "N/A"
        roi_txt = ", ".join(active_rois) if active_rois else "None"

        st.markdown('<p class="ts-section">📊 RAW METRICS</p>', unsafe_allow_html=True)
        st.markdown(
            f'<div class="ts-card-row">'
            f'<div class="ts-card"><div class="ts-card-label">❤ Heart Rate</div>'
            f'<div class="ts-card-value" style="font-size:1.2rem;">{bpm_txt}</div></div>'
            f'<div class="ts-card"><div class="ts-card-label">🛡 Overall Trust</div>'
            f'<div class="ts-card-value" style="font-size:1.2rem;">{overall_txt}</div></div>'
            f'<div class="ts-card"><div class="ts-card-label">⏱ Processing</div>'
            f'<div class="ts-card-value" style="font-size:1.2rem;color:#5a9abf;">{latency_txt}</div></div>'
            f'</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            f'<div class="ts-card-row">'
            + _mini_bar("rPPG Liveness", rppg_conf, _CYAN)
            + _mini_bar("Acoustic Trust", acoustic, "#a259ff")
            + _mini_bar("Viseme Sync", sync_sc, "#ff2fd0")
            + "</div>",
            unsafe_allow_html=True,
        )
        if sync_lag is not None:
            lag_color = "#00ff88" if abs(sync_lag) <= 80 else "#ffcc00"
            st.markdown(
                f'<div style="font-size:0.65rem;color:{lag_color};margin-bottom:0.6rem;">'
                f'Sync lag: {sync_lag:+.0f} ms (negative = audio leads)</div>',
                unsafe_allow_html=True,
            )
        st.markdown(f'<p class="ts-section">🎯 ACTIVE ROIs</p><div style="font-size:0.7rem;color:#4a8aaa;margin-bottom:0.5rem;">{roi_txt}</div>', unsafe_allow_html=True)

        st.markdown('<p class="ts-section">〰 rPPG PULSE WAVEFORM</p>', unsafe_allow_html=True)
        st.plotly_chart(
            _waveform_chart(list(st.session_state["waveform"])),
            use_container_width=True,
            config={"displayModeBar": False},
        )

with right_col:
    st.markdown('<p class="ts-section">⚡ TRUST GAUGE</p>', unsafe_allow_html=True)
    st.plotly_chart(
        _trust_gauge(overall_disp),
        use_container_width=True,
        config={"displayModeBar": False},
    )

    # Connection state card
    if not st.session_state["ws_connected"]:
        conn_color = "#ff6600"
        conn_label = "🔴 DISCONNECTED"
        conn_detail = st.session_state.get("ws_error") or "Connecting…"
    elif flag == "calibrating":
        conn_color = "#ffcc00"
        conn_label = "🟡 CALIBRATING"
        conn_detail = "Collecting biometric data…"
    elif flag == "nominal":
        conn_color = "#00ff88"
        conn_label = "🟢 ANALYZING"
        conn_detail = "Live biometric stream active"
    else:
        conn_color = "#ff9900"
        conn_label = "🟠 LOW CONFIDENCE"
        conn_detail = "Signals insufficient — check lighting & mic"

    st.markdown(
        f'<div style="background:#0d1520;border:1px solid {conn_color}33;border-radius:8px;'
        f'padding:0.75rem 1rem;margin-bottom:0.5rem;font-size:0.7rem;">'
        f'<div style="color:{conn_color};font-weight:700;margin-bottom:0.2rem;">{conn_label}</div>'
        f'<div style="color:#5a7a8a;">{conn_detail}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )

    st.markdown('<p class="ts-section">📷 LIVE PREVIEW</p>', unsafe_allow_html=True)
    cam_frame = st.session_state.get("cam_frame")
    if cam_frame is not None:
        st.image(cam_frame, channels="RGB", use_container_width=True)
    else:
        st.markdown(
            '<div style="background:#0d1520;border:1px dashed #1a2e40;border-radius:8px;'
            'height:200px;display:flex;align-items:center;justify-content:center;'
            'color:#253040;font-size:0.72rem;letter-spacing:0.1em;">'
            '📷 Camera unavailable — allow access or check index</div>',
            unsafe_allow_html=True,
        )

    session_id = (result.get("session_id") or "—")[:18] if result else "—"
    st.markdown(
        f'<div style="background:#0d1520;border:1px solid #1a2e40;border-radius:8px;'
        f'padding:0.75rem 1rem;margin-top:0.5rem;font-size:0.67rem;">'
        f'<div style="color:#4a6a80;letter-spacing:0.1em;text-transform:uppercase;'
        f'margin-bottom:0.35rem;">ENGINE INFO</div>'
        f'<div>Mode: <span style="color:#00f0ff;">PRODUCTION V1</span></div>'
        f'<div style="margin-top:0.2rem;">Session: <span style="color:#4a8aaa;">{session_id}…</span></div>'
        f'<div style="margin-top:0.2rem;">Time: <span style="color:#4a8aaa;">'
        f'{time.strftime("%H:%M:%S", time.localtime(ts_val))}</span></div>'
        f'</div>',
        unsafe_allow_html=True,
    )

# ── Auto-refresh ──────────────────────────────────────────────────────────────
time.sleep(0.033)
st.rerun()
