# Assumption: Pydantic v2 semantics are used throughout (model_dump, model_validate).
"""
backend/core/schemas.py
=======================
Locked data contract for TrueSync AI.

BiometricFrame is the *only* shape ever transmitted over the WebSocket,
from either MockEngine (Phase 1) or the real ML engine (Phase 2).
Do NOT add fields without updating engine_interface.py and both engines.
"""
from __future__ import annotations

from typing import List, Literal, Optional

from pydantic import BaseModel, Field


class BiometricFrame(BaseModel):
    """Single timestamped telemetry snapshot emitted by any BiometricEngine."""

    timestamp: float = Field(..., description="Unix epoch seconds (UTC).")
    session_id: str = Field(..., description="Unique session UUID string.")
    engine_mode: Literal["mock", "production"] = Field(
        ..., description="Which engine produced this frame."
    )

    # ── Vital-sign metrics ────────────────────────────────────────────────────
    bpm: Optional[float] = Field(
        None, ge=0.0, description="Heart rate in beats-per-minute, if available."
    )
    rppg_waveform: List[float] = Field(
        default_factory=list,
        description="Last ~90 samples of the BVP signal for live charting.",
    )
    rppg_liveness: Optional[float] = Field(
        None, ge=0.0, le=1.0, description="rPPG liveness confidence (0–1)."
    )

    # ── Anti-deepfake scores ─────────────────────────────────────────────────
    audio_trust: Optional[float] = Field(
        None, ge=0.0, le=1.0, description="Audio anti-spoof confidence (0–1)."
    )
    sync_score: Optional[float] = Field(
        None, ge=0.0, le=1.0, description="Viseme-phoneme lip-sync confidence (0–1)."
    )

    # ── Fused output ─────────────────────────────────────────────────────────
    overall_trust: Optional[float] = Field(
        None,
        ge=0.0,
        le=100.0,
        description="Fused liveness + anti-deepfake trust score (0–100).",
    )

    # ── Diagnostics ──────────────────────────────────────────────────────────
    active_rois: List[str] = Field(
        default_factory=list,
        description='Active region-of-interest labels, e.g. ["forehead","left_cheek"].',
    )
    status_flag: Literal[
        "nominal", "calibrating", "low_confidence", "spoof_suspected"
    ] = Field("calibrating", description="Engine-reported operational status.")
