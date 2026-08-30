# Assumption: ENGINE_MODE env var selects the engine; "production" is reserved for Phase 2.
"""
backend/main.py
================
TrueSync AI — FastAPI application entry-point.

Exposes:
  GET  /health               — uptime + engine mode
  WS   /ws/session           — bidirectional biometric telemetry stream

The factory function get_engine() is the ONLY place in the codebase that knows
which concrete engine class to instantiate.  All other code uses the
AbstractBiometricEngine interface.

Inbound WebSocket message envelope (from frontend):
  { "type": "video_frame", "data": "<base64-encoded JPEG>" }
  { "type": "audio_chunk",  "data": [<float>, ...] }

Outbound messages are BiometricFrame.model_dump() JSON objects.
"""
from __future__ import annotations

import asyncio
import base64
import json
import logging
import os
import time
import uuid
from typing import Any

from dotenv import load_dotenv
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from backend.core.engine_interface import AbstractBiometricEngine
from backend.core.mock_engine import MockEngine

load_dotenv()

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger("truesync.main")

# ── Configuration ─────────────────────────────────────────────────────────────
ENGINE_MODE: str = os.getenv("ENGINE_MODE", "mock").lower()
TARGET_FPS: int = int(os.getenv("TARGET_FPS", "30"))
_FRAME_INTERVAL: float = 1.0 / TARGET_FPS

_START_TIME: float = time.time()

# ── FastAPI app ───────────────────────────────────────────────────────────────
app = FastAPI(
    title="TrueSync AI",
    description="Zero-Trust Liveness & Anti-Deepfake Authentication Engine — INNO-CREW",
    version="1.0.0-phase1",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # tighten in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Engine factory — the ONLY place that knows about concrete engine types ─────
def get_engine() -> AbstractBiometricEngine:
    """
    Factory function.  To upgrade to Phase 2, change the import at the top of
    this file and swap MockEngine() for your real engine class here.
    No other file requires modification.  See docs/phase2_integration.md.
    """
    if ENGINE_MODE == "production":
        # Phase 2: import and instantiate the real engine here.
        # e.g. from backend.core.production_engine import ProductionEngine
        # return ProductionEngine()
        raise NotImplementedError(
            "Production engine not yet implemented. "
            "See docs/phase2_integration.md."
        )
    logger.info("Engine factory: instantiating MockEngine (ENGINE_MODE=%s)", ENGINE_MODE)
    return MockEngine()


# ── REST endpoints ────────────────────────────────────────────────────────────
@app.get("/health", tags=["ops"])
async def health() -> dict[str, Any]:
    """Return uptime seconds and current engine mode."""
    return {
        "status": "ok",
        "engine_mode": ENGINE_MODE,
        "uptime_seconds": round(time.time() - _START_TIME, 1),
        "target_fps": TARGET_FPS,
    }


# ── WebSocket session endpoint ────────────────────────────────────────────────
@app.websocket("/ws/session")
async def websocket_session(websocket: WebSocket) -> None:
    """
    Bidirectional WebSocket session.

    * Accepts optional inbound video/audio messages from the client.
    * Emits BiometricFrame JSON at TARGET_FPS cadence.
    * Calls engine.stop() on any disconnect or error.
    """
    await websocket.accept()
    client_id = str(uuid.uuid4())[:8]
    logger.info("WebSocket connected — client_id=%s", client_id)

    engine: AbstractBiometricEngine = get_engine()
    await engine.start()

    try:
        while True:
            loop_start = asyncio.get_event_loop().time()

            # ── Non-blocking inbound message poll ────────────────────────────
            # We use asyncio.wait_for with a short timeout so we don't block
            # the emit cadence waiting for client messages that may never come.
            try:
                raw = await asyncio.wait_for(
                    websocket.receive_text(), timeout=0.001
                )
                await _handle_inbound(engine, raw, client_id)
            except asyncio.TimeoutError:
                pass  # No message arrived in the poll window — that's fine.

            # ── Emit latest metrics ───────────────────────────────────────────
            frame = await engine.get_latest_metrics()
            await websocket.send_json(frame.model_dump())
            logger.debug("Frame emitted — session=%s bpm=%.1f trust=%.2f",
                         frame.session_id, frame.bpm or 0, frame.overall_trust or 0)

            # ── Pace to TARGET_FPS ────────────────────────────────────────────
            elapsed = asyncio.get_event_loop().time() - loop_start
            sleep_for = max(0.0, _FRAME_INTERVAL - elapsed)
            await asyncio.sleep(sleep_for)

    except WebSocketDisconnect:
        logger.info("WebSocket disconnected — client_id=%s", client_id)
    except Exception as exc:
        logger.exception("Unexpected error in WebSocket session client_id=%s: %s", client_id, exc)
    finally:
        await engine.stop()
        logger.info("Engine stopped — client_id=%s", client_id)


async def _handle_inbound(
    engine: AbstractBiometricEngine,
    raw: str,
    client_id: str,
) -> None:
    """
    Parse and dispatch an inbound client message to the appropriate engine method.
    Unknown message types are logged and silently dropped.
    """
    try:
        msg: dict[str, Any] = json.loads(raw)
    except json.JSONDecodeError:
        logger.warning("client_id=%s sent non-JSON message; ignoring.", client_id)
        return

    msg_type = msg.get("type", "")

    if msg_type == "video_frame":
        try:
            frame_bytes: bytes = base64.b64decode(msg.get("data", ""))
            await engine.process_video_frame(frame_bytes)
        except (ValueError, KeyError) as exc:
            logger.warning("Bad video_frame message from client_id=%s: %s", client_id, exc)

    elif msg_type == "audio_chunk":
        try:
            samples: list[float] = list(msg.get("data", []))
            await engine.process_audio_chunk(samples)
        except (TypeError, ValueError) as exc:
            logger.warning("Bad audio_chunk message from client_id=%s: %s", client_id, exc)

    else:
        logger.debug("Unknown message type '%s' from client_id=%s", msg_type, client_id)
