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

Robustness guarantees:
- MAX_PAYLOAD_BYTES enforced before any decode is attempted.
- One bad frame/chunk never crashes the session.
- NaN/Inf values are clamped in get_result() before schema construction.
- Structured error events sent back to client on invalid input.
- All processing exceptions caught and logged; session continues.
"""
from __future__ import annotations

import asyncio
import base64
import binascii
import json
import logging
import math
import os
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Optional

import cv2
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from backend.engine import AcousticModule, RPPGModule, SyncModule
from backend.fusion import fuse
from backend.schemas import BiometricResult

# ── Logging ───────────────────────────────────────────────────────────────────
_LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, _LOG_LEVEL, logging.INFO),
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger("truesync.server")

# ── Configuration ─────────────────────────────────────────────────────────────
# Maximum raw WebSocket message size (bytes). Protects against oversized payloads.
# A 640×480 JPEG at quality 75 is ~30–60 KB. 5 MB is very generous.
MAX_PAYLOAD_BYTES: int = int(os.getenv("MAX_PAYLOAD_BYTES", str(5 * 1024 * 1024)))  # 5 MB

# CORS origins — restrict in production via env var (comma-separated)
_CORS_ORIGINS_RAW = os.getenv("CORS_ORIGINS", "*")
_CORS_ORIGINS = [o.strip() for o in _CORS_ORIGINS_RAW.split(",")] if _CORS_ORIGINS_RAW != "*" else ["*"]

# Backend target frame rate (for future push-mode use)
_TARGET_FPS: int = int(os.getenv("TARGET_FPS", "30"))

# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="TrueSync AI",
    description="Zero-Trust Biological Multimodal Authentication Engine — INNO-CREW",
    version="1.0.0-production",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=_CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_START_TIME = time.time()
_EXECUTOR = ThreadPoolExecutor(
    max_workers=int(os.getenv("WORKER_THREADS", "4")),
    thread_name_prefix="truesync-worker",
)


# ── Health endpoint ───────────────────────────────────────────────────────────

@app.get("/health")
async def health() -> dict[str, Any]:
    """
    Lightweight liveness probe. Does NOT perform any biometric processing.
    Safe to call at high frequency from load balancers (AWS ALB/NLB).
    """
    return {
        "status": "ok",
        "uptime_seconds": round(time.time() - _START_TIME, 1),
        "version": "production-v1",
        "engine": "production",
        "target_fps": _TARGET_FPS,
        "max_payload_bytes": MAX_PAYLOAD_BYTES,
    }


# ── Per-session engine bundle ─────────────────────────────────────────────────

class SessionEngine:
    """
    Holds all module instances for one WebSocket session.
    Each WebSocket connection gets its own SessionEngine — no shared state.
    """

    def __init__(self, session_id: str) -> None:
        self.session_id = session_id
        self.rppg = RPPGModule(fs=30.0)
        self.acoustic = AcousticModule()
        self.sync = SyncModule(video_fps=30.0)
        self._session_start = time.monotonic()
        logger.info(
            "SessionEngine created — session_id=%s",
            session_id,
        )

    def process_video_frame_sync(self, frame_bgr: np.ndarray, timestamp: float) -> None:
        """Run rPPG update + mouth aperture extraction. Blocking — call in executor."""
        t0 = time.perf_counter()
        try:
            self.rppg.update(frame_bgr)
            aperture = self.rppg.get_mouth_aperture(frame_bgr)
            self.sync.update_video(timestamp, aperture)
        except Exception as exc:
            logger.warning(
                "session=%s: video processing error (rPPG/sync): %s",
                self.session_id, exc,
            )
        finally:
            logger.debug(
                "process_video_frame_sync: %.1f ms session=%s",
                (time.perf_counter() - t0) * 1000,
                self.session_id,
            )

    def process_audio_chunk_sync(
        self, samples: list[float], sample_rate: int, timestamp: float
    ) -> None:
        """Run acoustic + sync audio update. Blocking — call in executor."""
        t0 = time.perf_counter()
        try:
            self.acoustic.update(samples, sample_rate, timestamp)
            self.sync.update_audio(samples, sample_rate, timestamp)
        except Exception as exc:
            logger.warning(
                "session=%s: audio processing error (acoustic/sync): %s",
                self.session_id, exc,
            )
        finally:
            logger.debug(
                "process_audio_chunk_sync: %.1f ms session=%s",
                (time.perf_counter() - t0) * 1000,
                self.session_id,
            )

    def _safe_float(self, value: Any, lo: float = 0.0, hi: float = 1.0) -> Optional[float]:
        """Return value clamped to [lo, hi], or None if not a valid finite float."""
        if value is None:
            return None
        try:
            v = float(value)
        except (TypeError, ValueError):
            return None
        if not math.isfinite(v):
            return None
        return max(lo, min(hi, v))

    def get_result(self, processing_latency_ms: Optional[float] = None) -> BiometricResult:
        """
        Gather sub-scores, run fusion, and build a BiometricResult.
        NaN/Inf values are sanitised via _safe_float before schema construction.
        Exceptions inside any sub-module never propagate — they surface as None fields.
        """
        try:
            rppg_data = self.rppg.get_score()
        except Exception as exc:
            logger.warning("session=%s: rppg.get_score() failed: %s", self.session_id, exc)
            rppg_data = {"bpm": None, "rppg_confidence": None, "rppg_waveform": [], "active_rois": []}

        try:
            acoustic_data = self.acoustic.get_score()
        except Exception as exc:
            logger.warning("session=%s: acoustic.get_score() failed: %s", self.session_id, exc)
            acoustic_data = {"acoustic_trust": None}

        try:
            sync_data = self.sync.get_score()
        except Exception as exc:
            logger.warning("session=%s: sync.get_score() failed: %s", self.session_id, exc)
            sync_data = {"sync_score": None, "sync_lag_ms": None}

        # Sanitise all sub-scores before fusion
        rppg_conf = self._safe_float(rppg_data.get("rppg_confidence"))
        acoustic_trust = self._safe_float(acoustic_data.get("acoustic_trust"))
        sync_score = self._safe_float(sync_data.get("sync_score"))

        try:
            fusion_data = fuse(
                rppg_confidence=rppg_conf,
                acoustic_trust=acoustic_trust,
                sync_score=sync_score,
            )
        except Exception as exc:
            logger.warning("session=%s: fuse() failed: %s", self.session_id, exc)
            fusion_data = {"overall_trust": None, "status": "insufficient_data"}

        # Sanitise BPM (physiological range 0–300 BPM)
        bpm = self._safe_float(rppg_data.get("bpm"), lo=0.0, hi=300.0)
        overall_trust = self._safe_float(fusion_data.get("overall_trust"), lo=0.0, hi=100.0)

        # Sanitise waveform — drop any non-finite values
        raw_waveform = rppg_data.get("rppg_waveform", []) or []
        waveform = [v for v in raw_waveform if isinstance(v, (int, float)) and math.isfinite(v)]

        return BiometricResult(
            session_id=self.session_id,
            timestamp=time.time(),
            bpm=bpm,
            rppg_confidence=rppg_conf,
            rppg_waveform=waveform,
            active_rois=rppg_data.get("active_rois", []) or [],
            acoustic_trust=acoustic_trust,
            sync_score=sync_score,
            sync_lag_ms=sync_data.get("sync_lag_ms"),
            overall_trust=overall_trust,
            status=fusion_data.get("status", "calibrating"),
            processing_latency_ms=(
                round(processing_latency_ms, 2)
                if processing_latency_ms is not None and math.isfinite(processing_latency_ms)
                else None
            ),
        )

    def close(self) -> None:
        uptime_s = round(time.monotonic() - self._session_start, 1)
        try:
            self.rppg.close()
        except Exception:
            pass
        logger.info(
            "SessionEngine closed — session_id=%s uptime_s=%.1f",
            self.session_id,
            uptime_s,
        )


# ── Input validation helpers ──────────────────────────────────────────────────

def _validate_payload_size(raw: str, session_id: str) -> bool:
    """Return False and log if payload exceeds MAX_PAYLOAD_BYTES."""
    size = len(raw.encode("utf-8"))
    if size > MAX_PAYLOAD_BYTES:
        logger.warning(
            "session=%s: payload too large (%d bytes > %d limit) — rejected",
            session_id, size, MAX_PAYLOAD_BYTES,
        )
        return False
    return True


def _decode_base64_frame(b64_data: str, session_id: str) -> Optional[bytes]:
    """Safely decode base64 video frame data. Returns None on any error."""
    try:
        return base64.b64decode(b64_data)
    except (binascii.Error, ValueError) as exc:
        logger.warning("session=%s: invalid base64 in video_frame: %s", session_id, exc)
        return None


def _decode_jpeg_frame(frame_bytes: bytes, session_id: str) -> Optional[np.ndarray]:
    """Decode JPEG bytes to BGR numpy array. Returns None on failure."""
    try:
        nparr = np.frombuffer(frame_bytes, dtype=np.uint8)
        frame_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if frame_bgr is None:
            logger.warning("session=%s: cv2.imdecode returned None (corrupt JPEG?)", session_id)
            return None
        if frame_bgr.size == 0:
            logger.warning("session=%s: decoded JPEG is empty", session_id)
            return None
        return frame_bgr
    except Exception as exc:
        logger.warning("session=%s: JPEG decode error: %s", session_id, exc)
        return None


def _parse_audio_samples(
    raw_data: Any, session_id: str
) -> Optional[tuple[list[float], int]]:
    """
    Validate and parse audio chunk data.
    Returns (samples, sample_rate) or None if invalid.
    """
    try:
        samples_raw = raw_data.get("data", [])
        if not isinstance(samples_raw, list):
            logger.warning("session=%s: audio data is not a list", session_id)
            return None
        # Validate and clamp each sample to [-1, 1]
        samples: list[float] = []
        for x in samples_raw:
            try:
                v = float(x)
                if math.isfinite(v):
                    samples.append(max(-1.0, min(1.0, v)))
                # silently drop NaN/Inf samples
            except (TypeError, ValueError):
                pass  # skip non-numeric samples
        sample_rate = int(raw_data.get("sample_rate", 16000))
        if not (8000 <= sample_rate <= 96000):
            logger.warning(
                "session=%s: unsupported sample_rate %d — using 16000",
                session_id, sample_rate,
            )
            sample_rate = 16000
        return samples, sample_rate
    except Exception as exc:
        logger.warning("session=%s: audio_chunk parse error: %s", session_id, exc)
        return None


# ── WebSocket endpoint ────────────────────────────────────────────────────────

@app.websocket("/ws/session")
async def websocket_session(websocket: WebSocket) -> None:
    """
    Bidirectional WebSocket session.

    Accepts video_frame and audio_chunk messages from the client.
    Returns BiometricResult JSON after each video_frame.
    Handles all errors locally — one bad frame never terminates the session.
    """
    await websocket.accept()
    session_id = str(uuid.uuid4())
    engine = SessionEngine(session_id)
    loop = asyncio.get_event_loop()

    logger.info(
        "WebSocket connected — session_id=%s remote=%s",
        session_id,
        websocket.client,
    )

    try:
        while True:
            t_frame_start = time.perf_counter()

            # ── Receive message ───────────────────────────────────────────────
            try:
                raw = await asyncio.wait_for(websocket.receive_text(), timeout=10.0)
            except asyncio.TimeoutError:
                logger.debug("session=%s: no message in 10s, still waiting…", session_id)
                continue

            # ── Payload size guard ────────────────────────────────────────────
            if not _validate_payload_size(raw, session_id):
                try:
                    await websocket.send_text(json.dumps({
                        "error": "payload_too_large",
                        "detail": f"Payload exceeds {MAX_PAYLOAD_BYTES} bytes.",
                    }))
                except Exception:
                    pass
                continue

            # ── JSON parse ────────────────────────────────────────────────────
            try:
                msg: dict[str, Any] = json.loads(raw)
            except json.JSONDecodeError:
                logger.warning("session=%s: non-JSON message received", session_id)
                try:
                    await websocket.send_text(json.dumps({
                        "error": "invalid_json",
                        "detail": "Message must be valid JSON.",
                    }))
                except Exception:
                    pass
                continue

            msg_type = msg.get("type", "")
            timestamp = float(msg.get("timestamp", time.time()))

            # ── Video frame ───────────────────────────────────────────────────
            if msg_type == "video_frame":
                t_decode_start = time.perf_counter()

                b64_data = msg.get("data", "")
                if not b64_data:
                    logger.warning("session=%s: video_frame with empty data field", session_id)
                    continue

                frame_bytes = _decode_base64_frame(b64_data, session_id)
                if frame_bytes is None:
                    continue

                frame_bgr = _decode_jpeg_frame(frame_bytes, session_id)
                if frame_bgr is None:
                    continue

                decode_ms = (time.perf_counter() - t_decode_start) * 1000
                logger.debug("session=%s: JPEG decode %.1f ms", session_id, decode_ms)

                # ── Dispatch CPU-bound work to thread executor ─────────────────
                t_proc_start = time.perf_counter()
                try:
                    await loop.run_in_executor(
                        _EXECUTOR,
                        engine.process_video_frame_sync,
                        frame_bgr,
                        timestamp,
                    )
                except Exception as exc:
                    logger.warning(
                        "session=%s: executor error in video processing: %s",
                        session_id, exc,
                    )
                    continue
                proc_ms = (time.perf_counter() - t_proc_start) * 1000

                # ── Build and emit result ─────────────────────────────────────
                total_latency_ms = (time.perf_counter() - t_frame_start) * 1000
                try:
                    result = engine.get_result(processing_latency_ms=total_latency_ms)
                    await websocket.send_text(result.model_dump_json())
                except Exception as exc:
                    logger.warning(
                        "session=%s: result serialization error: %s",
                        session_id, exc,
                    )
                    continue

                logger.debug(
                    "session=%s: video_frame — decode=%.1fms proc=%.1fms total=%.1fms",
                    session_id, decode_ms, proc_ms, total_latency_ms,
                )

            # ── Audio chunk ───────────────────────────────────────────────────
            elif msg_type == "audio_chunk":
                parsed = _parse_audio_samples(msg, session_id)
                if parsed is None:
                    continue
                samples, sample_rate = parsed

                if not samples:
                    logger.debug("session=%s: audio_chunk with 0 valid samples — skipping", session_id)
                    continue

                try:
                    await loop.run_in_executor(
                        _EXECUTOR,
                        engine.process_audio_chunk_sync,
                        samples,
                        sample_rate,
                        timestamp,
                    )
                except Exception as exc:
                    logger.warning(
                        "session=%s: executor error in audio processing: %s",
                        session_id, exc,
                    )

            # ── Unknown message type ──────────────────────────────────────────
            else:
                logger.debug(
                    "session=%s: unknown message type '%s' — ignored",
                    session_id, msg_type,
                )

    except WebSocketDisconnect:
        logger.info("WebSocket disconnected — session_id=%s", session_id)
    except Exception as exc:
        logger.exception(
            "Unexpected error in WebSocket session — session_id=%s: %s",
            session_id, exc,
        )
    finally:
        engine.close()
        logger.info("Session torn down — session_id=%s", session_id)
