<div align="center">

# 🛡️ TrueSync AI

### Zero-Trust Biological & Multimodal Authentication Engine

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.40-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io)
[![MediaPipe](https://img.shields.io/badge/MediaPipe-0.10-0F9D58?style=for-the-badge&logo=google&logoColor=white)](https://mediapipe.dev)
[![License](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)](LICENSE)
[![Tests](https://img.shields.io/badge/Tests-Passing-brightgreen?style=for-the-badge&logo=pytest)](backend/tests)

<br/>

> **INNO-CREW** · IEEE WIE ILS 2026 National Hackathon · Track 4: Open Industry Problems

<br/>

*TrueSync AI detects a live, un-spoofed human in real time by fusing three independent biological signals — rPPG heartbeat, acoustic anti-spoofing, and lip-sync correlation — into a single trust score. No mock data. No placeholders. Production-grade from frame one.*

</div>

---

## 🎯 What Problem Does It Solve?

Modern authentication systems are increasingly vulnerable to:

| Threat | Description |
|--------|-------------|
| **Photo Spoofing** | Static image held in front of the camera |
| **Video Replay** | Pre-recorded video of a legitimate user |
| **Deepfake Injection** | AI-generated face video with dubbed audio |
| **Voice Cloning** | Neural TTS/vocoder-generated speech |
| **Partial Deepfakes** | Dubbed audio over a real face (or vice-versa) |

TrueSync AI defeats all five attack classes simultaneously by requiring three biological signals to be present, consistent, and in sync — simultaneously.

---

## ⚡ Live Demo Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                    Frontend  (Streamlit + Plotly)                   │
│  dashboard.py — live rPPG waveform · trust gauge · metric cards     │
│               webcam preview · offline-demo fallback                │
└───────────────────────────┬─────────────────────────────────────────┘
                            │  WebSocket  /ws/session
                            │  { "type": "video_frame", "data": <b64 JPEG> }
                            │  { "type": "audio_chunk",  "data": [float…]  }
                            │  ← BiometricResult JSON @ 30 Hz
┌───────────────────────────▼─────────────────────────────────────────┐
│                    Backend  (FastAPI + Uvicorn)                      │
│                                                                     │
│   main.py  ──►  get_engine()  ──►  AbstractBiometricEngine          │
│                                          │                          │
│               ┌──────────────────────────┼──────────────────────┐   │
│               │                          │                      │   │
│         RPPGModule               AcousticModule           SyncModule│
│         engine.py                engine.py                engine.py │
│     (MediaPipe FaceMesh)     (librosa STFT)         (cross-corr)   │
│               │                          │                      │   │
│               └──────────────────────────┼──────────────────────┘   │
│                                          ▼                          │
│                                   fusion.py                         │
│                          Weighted Score Fusion (0–100)              │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 🧠 How It Works — Three Biological Signals

### 1. 🫀 rPPG — Remote Photoplethysmography

Extracts your heartbeat from **colour changes in your face**, invisible to the naked eye.

```
Camera Frame → MediaPipe FaceMesh (468 landmarks)
    → Convex-hull ROI extraction (forehead + 2 cheeks)
    → YCbCr skin mask (eliminates non-skin pixels)
    → Green channel mean per frame (blood absorbs green most)
    → Rolling 5s buffer → Butterworth BPF (0.7–4.0 Hz)
    → Welch PSD → BPM peak + confidence ratio
```

**Spoof signal:** A printed photo or replay video has **no pulse** — the green channel stays flat.

### 2. 🎙️ Acoustic Anti-Spoofing

Detects synthetic or cloned voices using two complementary metrics:

```
Microphone audio → librosa STFT (n_fft=1024, hop=256)
    ├── Spectral Flatness (SF):
    │   geometric_mean(|STFT|) / arithmetic_mean(|STFT|)
    │   Natural speech → structured spectrum (low SF → high score)
    │   Vocoder output → flat/noisy spectrum (high SF → low score)
    └── Phase Consistency (PC):
        d(phase)/dt per frequency bin (instantaneous frequency)
        Unwrap → variance of IF deviation across bins
        Natural speech → irregular IF (high variance → score → 1)
        Neural vocoders → smooth periodic IF (low variance → score → 0)

acoustic_trust = 0.5 × SF_score + 0.5 × PC_score
```

### 3. 👄 Viseme-Phoneme Synchronisation

Catches **deepfakes** where audio and video are from different sources.

```
Video: MediaPipe mouth landmark aperture ratio (per frame, timestamped)
Audio: Short-time RMS energy envelope (per chunk, timestamped)

→ Interpolate RMS onto video timestamps
→ Z-normalise both signals
→ Cross-correlate within ±150ms lag window
→ sync_score = normalised peak cross-correlation (0–1)
→ sync_lag_ms = lag of peak in milliseconds
```

**Spoof signal:** Dubbed audio over a real/fake face has **near-zero correlation** between mouth movement and speech energy.

### 4. ⚖️ Weighted Score Fusion

```python
overall_trust = 100 × Σ(wᵢ × scoreᵢ) / Σ(wᵢ)   # available scores only

W_RPPG     = 0.40   # Direct biological signal (strongest liveness proof)
W_ACOUSTIC = 0.35   # Anti-spoof against voice cloning
W_SYNC     = 0.25   # Deepfake lip-sync correlation

# Status thresholds
THRESHOLD_NOMINAL         = 75.0   # → "nominal"
THRESHOLD_LOW_CONFIDENCE  = 50.0   # → "low_confidence"
```

If a module is still calibrating (insufficient buffer), its weight is **excluded and the remaining weights are re-normalised** — never fabricating data.

---

## 🚀 Quick Start

### Prerequisites

- Python 3.10+
- Webcam + microphone (for live mode)
- Windows / Linux / macOS

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/meet2124/TrueSync-AI.git
cd TrueSync-AI

# 2. Create and activate virtual environment
python -m venv .venv

# Windows
.venv\Scripts\activate

# Linux / macOS
source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure environment
cp .env.example .env
# Edit .env if needed (defaults work out of the box)
```

### Run the Application

```bash
# Terminal 1 — Start the backend
uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000

# Terminal 2 — Start the frontend dashboard
cd frontend
streamlit run dashboard.py
```

Open **http://localhost:8501** in your browser and allow webcam + microphone access.

---

## 🧪 Running Tests

```bash
# Run the full test suite from the project root
pytest backend/tests/ -v

# With coverage report
pytest backend/tests/ -v --tb=short
```

Test coverage includes:
- `test_schemas.py` — Pydantic model validation, field constraints, serialisation
- `test_mock_engine.py` — Engine lifecycle, telemetry cadence, BiometricFrame contracts

---

## ⚙️ Configuration

All configuration is via environment variables in `.env`:

| Variable | Default | Description |
|---|---|---|
| `ENGINE_MODE` | `mock` | `mock` (Phase 1) or `production` (Phase 2) |
| `MOCK_TRUST_MIN` | `0.82` | Mock trust floor (0–1 scale) |
| `MOCK_TRUST_MAX` | `0.96` | Mock trust ceiling (0–1 scale) |
| `TARGET_FPS` | `30` | WebSocket emission rate (frames/second) |
| `WEBSOCKET_URL` | `ws://localhost:8000/ws/session` | Frontend WebSocket endpoint |

---

## 📁 Project Structure

```
TrueSync-AI/
│
├── backend/
│   ├── __init__.py
│   ├── main.py                  # FastAPI app, WebSocket handler, engine factory
│   ├── engine.py                # Production V1: RPPGModule, AcousticModule, SyncModule
│   ├── fusion.py                # Weighted score fusion layer
│   ├── schemas.py               # Wire contracts: VideoFrameMessage, BiometricResult
│   ├── server.py                # Alternate server entrypoint
│   │
│   ├── core/
│   │   ├── __init__.py
│   │   ├── engine_interface.py  # AbstractBiometricEngine ABC (locked interface)
│   │   ├── mock_engine.py       # Phase 1: realistic mock telemetry generator
│   │   └── schemas.py           # BiometricFrame — core data contract
│   │
│   └── tests/
│       ├── __init__.py
│       ├── test_schemas.py      # Schema validation & serialisation tests
│       └── test_mock_engine.py  # Engine lifecycle & telemetry contract tests
│
├── frontend/
│   ├── app.py                   # Streamlit app entry
│   ├── dashboard.py             # Cyberpunk dashboard: live waveform, gauge, cards
│   └── assets/                  # Static assets
│
├── docs/
│   └── phase2_integration.md    # One-line swap guide: mock → production engine
│
├── conftest.py                  # pytest root config
├── requirements.txt             # Pinned production dependencies
├── .env.example                 # Environment template
├── .gitignore
└── README.md
```

---

## 🔌 API Reference

### REST

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/health` | Returns engine mode, uptime, and target FPS |

**Health Response:**
```json
{
  "status": "ok",
  "engine_mode": "mock",
  "uptime_seconds": 42.1,
  "target_fps": 30
}
```

### WebSocket — `/ws/session`

**Inbound (Client → Server):**
```json
// Video frame
{ "type": "video_frame", "timestamp": 1722345678.9, "data": "<base64-JPEG>" }

// Audio chunk
{ "type": "audio_chunk", "timestamp": 1722345678.9, "sample_rate": 16000, "data": [0.01, -0.02, ...] }
```

**Outbound (Server → Client) @ 30 Hz:**
```json
{
  "session_id": "a1b2c3d4",
  "timestamp": 1722345679.1,
  "bpm": 72.4,
  "rppg_confidence": 0.87,
  "rppg_waveform": [0.01, -0.02, 0.03, ...],
  "active_rois": ["forehead", "left_cheek", "right_cheek"],
  "acoustic_trust": 0.91,
  "sync_score": 0.85,
  "sync_lag_ms": 18.3,
  "overall_trust": 88.52,
  "status": "nominal"
}
```

**Status Values:**
| Status | Meaning |
|--------|---------|
| `calibrating` | Some modules still filling their minimum buffer |
| `nominal` | All modules active, trust ≥ 75 |
| `low_confidence` | All modules active, trust < 75 |
| `insufficient_data` | No sub-scores available yet |

---

## 🔄 Phase 2 Upgrade Path

TrueSync AI uses a clean **interface → implementation** separation. The entire system knows only `AbstractBiometricEngine`. Swapping mock → production is a **single-line change**:

```python
# backend/main.py — change this one line only:

# Phase 1 (current):
from backend.core.mock_engine import MockEngine
return MockEngine()

# Phase 2 (production):
from backend.core.production_engine import ProductionEngine
return ProductionEngine()
```

Zero changes to the frontend, WebSocket schema, or any API routes. See [`docs/phase2_integration.md`](docs/phase2_integration.md) for the full integration guide.

---

## 🏗️ Design Decisions

| Decision | Rationale |
|----------|-----------|
| **Green channel over CHROM** | CHROM provides marginal gains; simpler green-channel is sufficient for single-ROI use and reduces complexity. CHROM upgrade path documented. |
| **Butterworth order 4** | Good stopband attenuation with minimal passband ripple. `filtfilt` gives zero-phase distortion — critical for accurate BPM peak detection. |
| **Welch PSD over FFT** | Welch averages multiple overlapping frames for a smoother, more robust power spectrum, especially on short (5s) buffers. |
| **BPM recomputed every K=10 frames** | Balances freshness vs. cost. Full BPF + Welch every frame would exceed the 50ms per-frame budget at 30 FPS. |
| **Acoustic every 250ms** | Audio accumulates at its own cadence, independent of video. Tying recompute to wall-clock time (not frame count) ensures consistent update frequency regardless of upstream FPS variations. |
| **Cross-correlation ±150ms lag** | Physiological audio-visual delay in natural speech is typically <80ms. ±150ms gives headroom without false positives from periodic ambient noise. |
| **Weight re-normalisation on None** | Fabricating a neutral score (0.5) for a missing module would pollute the fusion. Re-normalising over available modules is statistically cleaner and ensures the displayed score reflects only measured data. |

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| **Backend API** | FastAPI 0.115 + Uvicorn (ASGI) |
| **Real-time Transport** | WebSocket (native FastAPI) |
| **Computer Vision** | OpenCV 4.10 + MediaPipe 0.10 (FaceMesh 468 landmarks) |
| **Signal Processing** | NumPy 1.26 + SciPy 1.14 (Butterworth, Welch PSD) |
| **Audio Analysis** | librosa 0.10 (STFT, phase analysis) |
| **Data Validation** | Pydantic v2 |
| **Frontend** | Streamlit 1.40 + Plotly 5.24 |
| **Configuration** | python-dotenv |
| **Testing** | pytest 8.3 + pytest-asyncio |

---

## 👥 Team

**INNO-CREW** — IEEE WIE ILS 2026 National Hackathon

| Name | GitHub |
|------|--------|
| Meet Purohit | [@meet2124](https://github.com/meet2124) |

---

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

<div align="center">

**Built with ❤️ for IEEE WIE ILS 2026**

*TrueSync AI — Because biological signals don't lie.*

</div>
