"""
backend/schemas.py — TrueSync AI Production V1
Wire contracts for all WebSocket messages (inbound + outbound).
"""
from __future__ import annotations
from typing import List, Literal, Optional
from pydantic import BaseModel, Field


# ── Inbound ───────────────────────────────────────────────────────────────────

class VideoFrameMessage(BaseModel):
    type: Literal["video_frame"]
    timestamp: float = Field(..., description="Wall-clock UTC epoch seconds at capture.")
    data: str = Field(..., description="Base64-encoded JPEG bytes.")


class AudioChunkMessage(BaseModel):
    type: Literal["audio_chunk"]
    timestamp: float = Field(..., description="Wall-clock UTC epoch seconds at capture.")
    sample_rate: int = Field(..., ge=8000, le=96000)
    data: List[float] = Field(..., description="Normalised PCM samples, -1.0 to 1.0.")


# ── Outbound ──────────────────────────────────────────────────────────────────

class BiometricResult(BaseModel):
    """
    Timestamped telemetry snapshot streamed back to the frontend.
    Fields may be null while a sub-module accumulates its minimum buffer.
    """
    session_id: str
    timestamp: float

    # rPPG
    bpm: Optional[float] = Field(None, ge=0.0, le=300.0)
    rppg_confidence: Optional[float] = Field(None, ge=0.0, le=1.0)
    rppg_waveform: List[float] = Field(default_factory=list)
    active_rois: List[str] = Field(default_factory=list)

    # Acoustic
    acoustic_trust: Optional[float] = Field(None, ge=0.0, le=1.0)

    # Viseme-Phoneme Sync
    sync_score: Optional[float] = Field(None, ge=0.0, le=1.0)
    sync_lag_ms: Optional[float] = None

    # Fused
    overall_trust: Optional[float] = Field(None, ge=0.0, le=100.0)

    # Status
    status: Literal["nominal", "calibrating", "insufficient_data", "low_confidence"] = "calibrating"
