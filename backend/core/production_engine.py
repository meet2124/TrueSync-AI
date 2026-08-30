"""
backend/core/production_engine.py
=================================
Production adapter for the biometric engine.
Connects real biometric modules (RPPG, Acoustic, Sync) to the
AbstractBiometricEngine interface used by main.py.
"""
from __future__ import annotations

import asyncio
import logging
import math
import time
import uuid
from typing import Any, List, Optional

import cv2
import numpy as np

from backend.core.engine_interface import AbstractBiometricEngine
from backend.core.schemas import BiometricFrame
from backend.engine import AcousticModule, RPPGModule, SyncModule
from backend.fusion import fuse

logger = logging.getLogger("truesync.production_engine")


class ProductionEngine(AbstractBiometricEngine):
    def __init__(self) -> None:
        self._session_id: str = str(uuid.uuid4())
        self._running: bool = False
        
        # We instantiate modules on start() to ensure clean lifecycle
        self._rppg: Optional[RPPGModule] = None
        self._acoustic: Optional[AcousticModule] = None
        self._sync: Optional[SyncModule] = None

    async def start(self) -> None:
        if self._running:
            return
        logger.info("Starting ProductionEngine session=%s", self._session_id)
        # MediaPipe initialization happens synchronously here
        # It takes some CPU, but start() is called once per session setup.
        # We offload it to thread just in case it takes >100ms
        await asyncio.to_thread(self._init_modules)
        self._running = True

    def _init_modules(self) -> None:
        self._rppg = RPPGModule(fs=30.0)
        self._acoustic = AcousticModule()
        self._sync = SyncModule(video_fps=30.0)

    async def stop(self) -> None:
        if not self._running:
            return
        logger.info("Stopping ProductionEngine session=%s", self._session_id)
        self._running = False
        if self._rppg:
            await asyncio.to_thread(self._rppg.close)

    async def process_video_frame(self, frame_bytes: bytes) -> None:
        if not self._running or not frame_bytes:
            return
            
        timestamp = time.time()
        # Decode and process in a background thread to prevent event loop blocking
        await asyncio.to_thread(self._process_video_sync, frame_bytes, timestamp)

    def _process_video_sync(self, frame_bytes: bytes, timestamp: float) -> None:
        try:
            t0 = time.perf_counter()
            np_arr = np.frombuffer(frame_bytes, np.uint8)
            frame_bgr = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            t_decode = time.perf_counter()
            
            if frame_bgr is None:
                logger.warning("session=%s: Failed to decode JPEG frame", self._session_id)
                return

            if self._rppg:
                self._rppg.update(frame_bgr)
                aperture = self._rppg.get_mouth_aperture(frame_bgr)
                if self._sync:
                    self._sync.update_video(timestamp, aperture)
                    
            t_total = time.perf_counter()
            logger.info("Perf [session=%s]: video decode=%.2fms process=%.2fms total=%.2fms", 
                         self._session_id, (t_decode - t0)*1000, (t_total - t_decode)*1000, (t_total - t0)*1000)
        except Exception as exc:
            logger.warning("session=%s: Video processing error: %s", self._session_id, exc)

    async def process_audio_chunk(self, samples: list[float], sample_rate: int = 16000) -> None:
        if not self._running or not samples:
            return
            
        timestamp = time.time()
        
        # Offload to thread to prevent blocking
        await asyncio.to_thread(self._process_audio_sync, samples, sample_rate, timestamp)

    def _process_audio_sync(self, samples: list[float], sample_rate: int, timestamp: float) -> None:
        try:
            t0 = time.perf_counter()
            # Validate numeric bounds
            valid_samples = [s for s in samples if isinstance(s, (int, float)) and math.isfinite(s)]
            if not valid_samples:
                return

            # Clamp to [-1.0, 1.0]
            valid_samples = [max(-1.0, min(1.0, float(s))) for s in valid_samples]

            if sample_rate != 16000:
                import scipy.signal
                arr = np.array(valid_samples, dtype=np.float32)
                # resample to 16kHz
                new_len = int(len(arr) * 16000 / sample_rate)
                resampled = scipy.signal.resample(arr, new_len)
                valid_samples = resampled.tolist()
                sample_rate = 16000

            if self._acoustic:
                self._acoustic.update(valid_samples, sample_rate, timestamp)
            if self._sync:
                self._sync.update_audio(valid_samples, sample_rate, timestamp)
                
            t_total = time.perf_counter()
            logger.info("Perf [session=%s]: audio process=%.2fms", self._session_id, (t_total - t0)*1000)
        except Exception as exc:
            logger.warning("session=%s: Audio processing error: %s", self._session_id, exc)

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

    async def get_latest_metrics(self) -> BiometricFrame:
        """
        Gather sub-scores, run fusion, and build a BiometricFrame.
        NaN/Inf values are sanitised before schema construction.
        Exceptions inside any sub-module surface as None fields.
        """
        # Execute the sync reading off-thread just in case they trigger anything,
        # though get_score() is O(1).
        return await asyncio.to_thread(self._get_metrics_sync)

    def _get_metrics_sync(self) -> BiometricFrame:
        t0 = time.perf_counter()
        try:
            rppg_data = self._rppg.get_score() if self._rppg else {}
        except Exception as exc:
            logger.warning("session=%s: rppg.get_score failed: %s", self._session_id, exc)
            rppg_data = {}

        try:
            acoustic_data = self._acoustic.get_score() if self._acoustic else {}
        except Exception as exc:
            logger.warning("session=%s: acoustic.get_score failed: %s", self._session_id, exc)
            acoustic_data = {}

        try:
            sync_data = self._sync.get_score() if self._sync else {}
        except Exception as exc:
            logger.warning("session=%s: sync.get_score failed: %s", self._session_id, exc)
            sync_data = {}

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
            logger.warning("session=%s: fuse() failed: %s", self._session_id, exc)
            fusion_data = {"overall_trust": None, "status": "insufficient_data"}

        # Map internal status string to Schema's status_flag
        # Fusion statuses: 'nominal', 'calibrating', 'insufficient_data', 'low_confidence'
        # BiometricFrame statuses: "nominal", "calibrating", "low_confidence", "spoof_suspected"
        internal_status = fusion_data.get("status", "calibrating")
        if internal_status == "insufficient_data":
            # For BiometricFrame schema, "calibrating" is often used for insufficient data initially
            # But the schema has "nominal", "calibrating", "low_confidence", "spoof_suspected"
            # If it's literally "insufficient_data", fallback to "calibrating".
            status_flag = "calibrating"
        elif internal_status == "low_confidence":
            status_flag = "low_confidence"
        elif internal_status == "nominal":
            status_flag = "nominal"
        else:
            status_flag = "calibrating"

        # Overall trust bounds for BiometricFrame: [0, 100]
        overall_trust = self._safe_float(fusion_data.get("overall_trust"), lo=0.0, hi=100.0)
        
        # BPM bounds: [0, 300]
        bpm = self._safe_float(rppg_data.get("bpm"), lo=0.0, hi=300.0)

        # Sanitise waveform
        raw_waveform = rppg_data.get("rppg_waveform", []) or []
        waveform = [float(v) for v in raw_waveform if isinstance(v, (int, float)) and math.isfinite(v)]

        t_total = time.perf_counter()
        logger.info("Perf [session=%s]: metric generation=%.2fms", self._session_id, (t_total - t0)*1000)

        return BiometricFrame(
            timestamp=time.time(),
            session_id=self._session_id,
            engine_mode="production",
            bpm=bpm,
            rppg_waveform=waveform,
            rppg_liveness=rppg_conf,
            audio_trust=acoustic_trust,
            sync_score=sync_score,
            overall_trust=overall_trust,
            active_rois=rppg_data.get("active_rois", []) or [],
            status_flag=status_flag,
        )
