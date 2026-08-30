"""
backend/server.py
=================
TrueSync AI — Production V1 FastAPI WebSocket Server.

Endpoint: WS /ws/session
- One BiometricEngine instance per connection (no shared state).
- CPU-bound work (MediaPipe, scipy, librosa) dispatched via ThreadPoolExecutor.
- Inbound: VideoFrameMessage | AudioChunkMessage (interleaved on one socket).
- Outbound: BiometricResult JSON after each video frame processed.
- Timing instrumentation logged at DEBUG via time.perf_counter().
"""
from __future__ import annotations

import asyncio
import base64
import json
import logging
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import cv2
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from backend.engine import AcousticModule, RPPGModule, SyncModule
from backend.fusion import fuse
from backend.schemas import BiometricResult

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger("truesync.server")

# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="TrueSync AI",
    description="Zero-Trust Biological Multimodal Authentication Engine — INNO-CREW",
    version="1.0.0-production",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_START_TIME = time.time()
_EXECUTOR = ThreadPoolExecutor(max_workers=4, thread_name_prefix="truesync-worker")


@app.get("/health")
async def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "uptime_seconds": round(time.time() - _START_TIME, 1),
        "version": "production-v1",
    }


# ── Per-session engine bundle ─────────────────────────────────────────────────

class SessionEngine:
    """Holds all module instances for one WebSocket session."""

    def __init__(self, session_id: str) -> None:
        self.session_id = session_id
        self.rppg = RPPGModule(fs=30.0)
        self.acoustic = AcousticModule()
        self.sync = SyncModule(video_fps=30.0)
        logger.info("SessionEngine created — session_id=%s", session_id)

    def process_video_frame_sync(self, frame_bgr: np.ndarray, timestamp: float) -> None:
        """Run rPPG update + mouth aperture extraction. Blocking — call in executor."""
        t0 = time.perf_counter()
        self.rppg.update(frame_bgr)
        aperture = self.rppg.get_mouth_aperture(frame_bgr)
        self.sync.update_video(timestamp, aperture)
        logger.debug("process_video_frame_sync: %.1f ms", (time.perf_counter() - t0) * 1000)

    def process_audio_chunk_sync(self, samples: list[float], sample_rate: int, timestamp: float) -> None:
        """Run acoustic + sync audio update. Blocking — call in executor."""
        t0 = time.perf_counter()
        self.acoustic.update(samples, sample_rate, timestamp)
        self.sync.update_audio(samples, sample_rate, timestamp)
        logger.debug("process_audio_chunk_sync: %.1f ms", (time.perf_counter() - t0) * 1000)

    def get_result(self) -> BiometricResult:
        rppg_data = self.rppg.get_score()
        acoustic_data = self.acoustic.get_score()
        sync_data = self.sync.get_score()
        fusion_data = fuse(
            rppg_confidence=rppg_data["rppg_confidence"],
            acoustic_trust=acoustic_data["acoustic_trust"],
            sync_score=sync_data["sync_score"],
        )
        return BiometricResult(
            session_id=self.session_id,
            timestamp=time.time(),
            bpm=rppg_data["bpm"],
            rppg_confidence=rppg_data["rppg_confidence"],
            rppg_waveform=rppg_data["rppg_waveform"],
            active_rois=rppg_data["active_rois"],
            acoustic_trust=acoustic_data["acoustic_trust"],
            sync_score=sync_data["sync_score"],
            sync_lag_ms=sync_data["sync_lag_ms"],
            overall_trust=fusion_data["overall_trust"],
            status=fusion_data["status"],
        )

    def close(self) -> None:
        try:
            self.rppg.close()
        except Exception:
            pass
        logger.info("SessionEngine closed — session_id=%s", self.session_id)


# ── WebSocket endpoint ────────────────────────────────────────────────────────

@app.websocket("/ws/session")
async def websocket_session(websocket: WebSocket) -> None:
    await websocket.accept()
    session_id = str(uuid.uuid4())
    engine = SessionEngine(session_id)
    loop = asyncio.get_event_loop()

    logger.info("WebSocket connected — session_id=%s", session_id)

    try:
        while True:
            t_frame_start = time.perf_counter()

            try:
                raw = await asyncio.wait_for(websocket.receive_text(), timeout=5.0)
            except asyncio.TimeoutError:
                logger.debug("session=%s: no message in 5s, waiting...", session_id)
                continue

            try:
                msg: dict[str, Any] = json.loads(raw)
            except json.JSONDecodeError:
                logger.warning("session=%s: non-JSON message received", session_id)
                continue

            msg_type = msg.get("type", "")
            timestamp = float(msg.get("timestamp", time.time()))

            if msg_type == "video_frame":
                t0 = time.perf_counter()
                try:
                    frame_bytes = base64.b64decode(msg["data"])
                    nparr = np.frombuffer(frame_bytes, dtype=np.uint8)
                    frame_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                    if frame_bgr is None:
                        logger.warning("session=%s: failed to decode JPEG frame", session_id)
                        continue
                except (KeyError, ValueError, Exception) as exc:
                    logger.warning("session=%s: bad video_frame: %s", session_id, exc)
                    continue

                # Dispatch CPU-bound work to thread executor
                await loop.run_in_executor(
                    _EXECUTOR,
                    engine.process_video_frame_sync,
                    frame_bgr,
                    timestamp,
                )
                logger.debug("session=%s: video_frame dispatch+exec %.1f ms",
                             session_id, (time.perf_counter() - t0) * 1000)

                # Emit result back over same socket
                result = engine.get_result()
                await websocket.send_text(result.model_dump_json())
                logger.debug("session=%s: total frame latency %.1f ms",
                             session_id, (time.perf_counter() - t_frame_start) * 1000)

            elif msg_type == "audio_chunk":
                try:
                    samples: list[float] = [float(x) for x in msg["data"]]
                    sample_rate: int = int(msg.get("sample_rate", 16000))
                except (KeyError, ValueError, TypeError) as exc:
                    logger.warning("session=%s: bad audio_chunk: %s", session_id, exc)
                    continue

                # Dispatch audio processing to executor (non-blocking)
                await loop.run_in_executor(
                    _EXECUTOR,
                    engine.process_audio_chunk_sync,
                    samples,
                    sample_rate,
                    timestamp,
                )

            else:
                logger.debug("session=%s: unknown message type '%s'", session_id, msg_type)

    except WebSocketDisconnect:
        logger.info("WebSocket disconnected — session_id=%s", session_id)
    except Exception as exc:
        logger.exception("Unexpected error — session_id=%s: %s", session_id, exc)
    finally:
        engine.close()
        logger.info("Session torn down — session_id=%s", session_id)
