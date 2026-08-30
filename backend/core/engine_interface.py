# Assumption: Python 3.11+ is the runtime target (asyncio.TaskGroup available if needed).
"""
backend/core/engine_interface.py
=================================
Abstract base class for all TrueSync AI biometric engines.

Rules for implementors
----------------------
1.  Subclass AbstractBiometricEngine and implement every abstract method.
2.  Never modify these method signatures — doing so would break main.py and
    the WebSocket layer.
3.  See docs/phase2_integration.md for Phase 2 swap instructions.
"""
from __future__ import annotations

from abc import ABC, abstractmethod


class AbstractBiometricEngine(ABC):
    """
    Contract satisfied by every concrete engine (mock or production).

    All five methods are async so that real ML implementations can use
    non-blocking I/O without requiring call-site changes.
    """

    @abstractmethod
    async def start(self) -> None:
        """
        Initialise all internal state, open device handles, load models, etc.
        Must be idempotent — calling start() on an already-running engine
        must not raise.
        """

    @abstractmethod
    async def process_video_frame(self, frame_bytes: bytes) -> None:
        """
        Accept a raw video frame (JPEG-encoded bytes) from the WebSocket
        client and update internal state.  Phase 1 mock may treat this as
        a no-op, but the method MUST exist with this exact signature.

        Parameters
        ----------
        frame_bytes : bytes
            JPEG-compressed BGR image from the frontend webcam capture.
        """

    @abstractmethod
    async def process_audio_chunk(self, samples: list[float]) -> None:
        """
        Accept a chunk of normalised audio samples (PCM, –1.0–1.0) captured
        from the client microphone and update internal state.  Phase 1 mock
        may treat this as a no-op, but the method MUST exist with this exact
        signature.

        Parameters
        ----------
        samples : list[float]
            Normalised PCM audio samples, typically 512–2048 samples per chunk.
        """

    @abstractmethod
    async def get_latest_metrics(self) -> "BiometricFrame":  # noqa: F821 — avoids circular import
        """
        Return the most recent telemetry snapshot as a BiometricFrame.
        Called by main.py at TARGET_FPS cadence; must never block for long.

        Returns
        -------
        BiometricFrame
            The current snapshot.  Import from backend.core.schemas.
        """

    @abstractmethod
    async def stop(self) -> None:
        """
        Release all resources: device handles, model sessions, threads, etc.
        Called on WebSocket disconnect.  Must be safe to call even if start()
        was never invoked.
        """
