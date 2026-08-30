# Phase 2 Integration Guide

## What to Build

Create two new files:

- **`backend/core/rppg_chrom.py`** — implements real rPPG via the CHROM algorithm on incoming JPEG frames. The `process_video_frame` method should decode the JPEG bytes with OpenCV, extract the forehead/cheek ROI colour channels, and accumulate a sliding BVP signal.
- **`backend/core/audio_cnn.py`** — implements a Mel-spectrogram CNN classifier for synthetic voice detection. The `process_audio_chunk` method should append PCM samples to a ring buffer and run inference when a full window is available.

Both classes must subclass `AbstractBiometricEngine` (from `backend/core/engine_interface.py`) and implement all five async methods: `start`, `process_video_frame`, `process_audio_chunk`, `get_latest_metrics`, and `stop`. The `get_latest_metrics` method must return a `BiometricFrame` (from `backend/core/schemas.py`) with `engine_mode="production"`.

## The Single-Line Swap

Open `backend/main.py` and locate the `get_engine()` factory function. It currently returns `MockEngine()`. Replace the body of the `if ENGINE_MODE == "production":` branch:

```python
# Phase 2: uncomment and adapt these two lines
from backend.core.rppg_chrom import RppgChromEngine
return RppgChromEngine()
```

Then set `ENGINE_MODE=production` in your `.env` file.

## Confirmation

No other file requires modification. The WebSocket handler, the frontend dashboard, and the data schema are all engine-agnostic. The frontend will automatically display real rPPG waveforms and genuine trust scores the moment the engine swap is in place.
