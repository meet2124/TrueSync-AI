# Assumption: TARGET_FPS env var is read by main.py; mock_engine.py uses it only for waveform buffer sizing.
"""
backend/core/mock_engine.py
============================
Phase 1 — MockEngine: mathematically realistic, continuously-varying telemetry
generator.  Powers the full TrueSync AI stack (WebSocket, dashboard) without
requiring any camera, microphone, or trained ML model.

Phase 2 swap
------------
See docs/phase2_integration.md for the single-line change required to replace
this engine with the real rPPG + audio CNN pipeline.

Waveform design
---------------
* BPM drifts slowly via a bounded random walk (±0.15 BPM/tick) within [60, 90].
* The BVP signal is a sine wave whose frequency tracks the current BPM.
  A small Gaussian jitter term is added so the trace looks physiological rather
  than perfectly synthetic.
* overall_trust (internal 0.0–1.0) random-walks with ±0.005/tick, clipped to
  [MOCK_TRUST_MIN, MOCK_TRUST_MAX], then linearly scaled to 0–100 for the schema.
* sub-scores (rppg_liveness, audio_trust, sync_score) are derived from
  overall_trust with small individual offsets so they each vary independently.
"""
from __future__ import annotations

import asyncio
import math
import os
import random
import time
import uuid
from collections import deque
from typing import Deque, List

from dotenv import load_dotenv

from backend.core.engine_interface import AbstractBiometricEngine
from backend.core.schemas import BiometricFrame

load_dotenv()

# ── Configuration ─────────────────────────────────────────────────────────────
_TRUST_MIN: float = float(os.getenv("MOCK_TRUST_MIN", "0.82"))
_TRUST_MAX: float = float(os.getenv("MOCK_TRUST_MAX", "0.96"))
_TARGET_FPS: int = int(os.getenv("TARGET_FPS", "30"))
_WAVEFORM_BUFFER: int = 90          # ~3 s at 30 Hz
_BPM_MIN: float = 60.0
_BPM_MAX: float = 90.0
_BPM_BASELINE: float = 72.0         # starting BPM
_BPM_WALK_SIGMA: float = 0.15       # max BPM drift per tick
_TRUST_WALK_SIGMA: float = 0.005    # max trust drift per tick
_JITTER_SIGMA: float = 0.04         # BVP amplitude noise σ
_ROI_LABELS: List[str] = ["forehead", "left_cheek", "right_cheek", "nose_bridge"]


class MockEngine(AbstractBiometricEngine):
    """
    Concrete mock implementation of AbstractBiometricEngine.

    Produces smooth, physiologically plausible synthetic telemetry.
    process_video_frame and process_audio_chunk are intentional no-ops
    in Phase 1 but maintain the correct async signatures for transparent
    engine swapping in Phase 2.
    """

    def __init__(self) -> None:
        self._session_id: str = str(uuid.uuid4())
        self._running: bool = False

        # BVP / pulse state
        self._bpm: float = _BPM_BASELINE
        self._phase: float = 0.0                       # phase accumulator (radians)

        # Waveform ring buffer
        self._waveform: Deque[float] = deque(maxlen=_WAVEFORM_BUFFER)

        # Trust state
        self._trust: float = (_TRUST_MIN + _TRUST_MAX) / 2.0  # start at midpoint

        # Sub-score offsets (fixed small per-session deltas so they read as independent sensors)
        self._rng = random.Random()
        self._liveness_bias: float = self._rng.uniform(-0.05, 0.05)
        self._audio_bias: float = self._rng.uniform(-0.05, 0.05)
        self._sync_bias: float = self._rng.uniform(-0.05, 0.05)

        # Tick counter for status flag transitions (simulate calibration phase)
        self._tick: int = 0

        # Last frame (cached so get_latest_metrics is O(1))
        self._latest_frame: BiometricFrame | None = None

    # ── AbstractBiometricEngine interface ─────────────────────────────────────

    async def start(self) -> None:
        """Initialise engine state and seed the waveform buffer."""
        if self._running:
            return
        self._running = True
        self._tick = 0
        # Pre-fill buffer so the chart doesn't start empty
        for _ in range(_WAVEFORM_BUFFER):
            self._advance_state()

    async def process_video_frame(self, frame_bytes: bytes) -> None:
        """No-op in Phase 1.  Real rPPG processing goes here in Phase 2."""
        pass  # noqa: PIE790 — intentional mock no-op; signature must match interface

    async def process_audio_chunk(self, samples: list[float], sample_rate: int = 16000) -> None:
        """No-op in Phase 1.  Real audio CNN processing goes here in Phase 2."""
        pass  # noqa: PIE790 — intentional mock no-op; signature must match interface

    async def get_latest_metrics(self) -> BiometricFrame:
        """Advance internal state by one tick and return the current frame."""
        self._advance_state()
        return self._build_frame()

    async def stop(self) -> None:
        """Mark the engine as stopped.  No resources to release in Phase 1."""
        self._running = False

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _advance_state(self) -> None:
        """Advance BVP phase, BPM, and trust by one tick."""
        # ── BPM random walk ──────────────────────────────────────────────────
        delta_bpm = self._rng.gauss(0.0, _BPM_WALK_SIGMA)
        self._bpm = max(_BPM_MIN, min(_BPM_MAX, self._bpm + delta_bpm))

        # ── Phase accumulator ────────────────────────────────────────────────
        # ω = 2π × (bpm/60) / fps  →  radians per tick
        omega = 2.0 * math.pi * (self._bpm / 60.0) / _TARGET_FPS
        self._phase = (self._phase + omega) % (2.0 * math.pi)

        # ── BVP sample ───────────────────────────────────────────────────────
        # Fundamental + weak 2nd harmonic (physiological shape) + Gaussian noise
        bvp = (
            0.70 * math.sin(self._phase)
            + 0.20 * math.sin(2.0 * self._phase - 0.3)
            + self._rng.gauss(0.0, _JITTER_SIGMA)
        )
        self._waveform.append(bvp)

        # ── Trust random walk ────────────────────────────────────────────────
        delta_trust = self._rng.gauss(0.0, _TRUST_WALK_SIGMA)
        self._trust = max(_TRUST_MIN, min(_TRUST_MAX, self._trust + delta_trust))

        self._tick += 1

    def _sub_score(self, bias: float) -> float:
        """Derive a sub-score from current trust + a fixed bias, clamped to [0, 1]."""
        raw = self._trust + bias + self._rng.gauss(0.0, 0.008)
        return max(0.0, min(1.0, raw))

    def _status_flag(self) -> str:
        """Return status based on tick count and trust level."""
        if self._tick < _TARGET_FPS * 2:          # first 2 s → calibrating
            return "calibrating"
        if self._trust < _TRUST_MIN + 0.02:
            return "low_confidence"
        return "nominal"

    def _build_frame(self) -> BiometricFrame:
        """Construct and cache a BiometricFrame from current state."""
        overall_trust_100 = round(self._trust * 100.0, 2)  # scale 0–1 → 0–100

        frame = BiometricFrame(
            timestamp=time.time(),
            session_id=self._session_id,
            engine_mode="mock",
            bpm=round(self._bpm, 1),
            rppg_waveform=list(self._waveform),
            rppg_liveness=round(self._sub_score(self._liveness_bias), 4),
            audio_trust=round(self._sub_score(self._audio_bias), 4),
            sync_score=round(self._sub_score(self._sync_bias), 4),
            overall_trust=overall_trust_100,
            active_rois=_ROI_LABELS,
            status_flag=self._status_flag(),  # type: ignore[arg-type]
        )
        self._latest_frame = frame
        return frame
