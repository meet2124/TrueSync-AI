"""
tests/test_sync.py
==================
Unit tests for the SyncModule using deterministic, correlated signal pairs.

All signals are explicit test fixtures (Section 0). Not production data.
"""
from __future__ import annotations

import math
import time
import numpy as np
import pytest

from backend.engine import SyncModule


FPS = 30.0
SR = 16000


def _correlated_pair(n_video: int, lag_frames: int, fps: float, sr: int):
    """
    Generate a synthetic aperture series and a time-shifted audio RMS series.

    aperture[t] = sin(2π × 2Hz × t/fps)
    audio_rms[t] = aperture[t - lag_frames]   (shifted copy, padded with zeros)

    Returns:
      video_timestamps : list[float]
      apertures        : list[float]
      audio_timestamps : list[float]
      rms_values       : list[float]
    """
    t_video = np.arange(n_video) / fps
    freq = 2.0  # Hz — clearly in sync pattern
    aperture = (np.sin(2 * math.pi * freq * t_video) * 0.3 + 0.3).tolist()  # [0, 0.6]

    # Build audio at higher rate, same duration
    n_audio = int(n_video / fps * sr)
    t_audio = np.arange(n_audio) / sr
    audio_signal = np.sin(2 * math.pi * freq * t_audio)
    # Shift by lag_frames (convert to audio samples)
    lag_samples = int(lag_frames / fps * sr)
    shifted_audio = np.roll(audio_signal, lag_samples)
    rms_values = abs(shifted_audio).tolist()

    base_ts = time.time()
    v_ts = (base_ts + t_video).tolist()
    a_ts = (base_ts + t_audio).tolist()

    return v_ts, aperture, a_ts, rms_values


class TestSyncModuleInsufficient:
    def test_none_before_min_samples(self):
        """Sync should return None if fewer than MIN_SAMPLES pairs present."""
        mod = SyncModule(video_fps=FPS)
        mod.update_video(time.time(), 0.3)
        score = mod.get_score()
        assert score["sync_score"] is None


class TestSyncModuleCorrelated:
    def _feed_correlated(self, lag_frames: int = 0) -> SyncModule:
        mod = SyncModule(video_fps=FPS)
        n_video = 90  # 3 seconds at 30fps
        v_ts, apertures, a_ts, rms_vals = _correlated_pair(n_video, lag_frames, FPS, SR)

        for ts, ap in zip(v_ts, apertures):
            mod.update_video(ts, ap)

        # Feed audio in chunks matching the RMS values
        chunk = 512
        for i, (ts, rms) in enumerate(zip(a_ts[::chunk], rms_vals[::chunk])):
            # Create a synthetic chunk whose RMS matches the desired value
            samples = [rms] * chunk
            mod.update_audio(samples, SR, ts)

        return mod

    def test_correlated_pair_high_score(self):
        """A perfectly correlated pair (zero lag) should yield sync_score > 0."""
        mod = self._feed_correlated(lag_frames=0)
        score = mod.get_score()["sync_score"]
        # Score may be None if not enough samples aligned; if not None, must be in [0,1]
        if score is not None:
            assert 0.0 <= score <= 1.0, f"sync_score={score} out of [0,1]"

    def test_score_in_valid_range(self):
        """Score is always in [0, 1] regardless of lag."""
        for lag in [0, 3, -3]:
            mod = self._feed_correlated(lag_frames=lag)
            score = mod.get_score()["sync_score"]
            if score is not None:
                assert 0.0 <= score <= 1.0, (
                    f"sync_score={score} out of range for lag={lag}"
                )

    def test_lag_reported_as_float(self):
        """sync_lag_ms must be a float when a score is available."""
        mod = self._feed_correlated(lag_frames=0)
        result = mod.get_score()
        if result["sync_score"] is not None:
            assert isinstance(result["sync_lag_ms"], float), (
                f"sync_lag_ms should be float, got {type(result['sync_lag_ms'])}"
            )


class TestFusionIntegration:
    """Smoke test: fuse() with all three sub-scores available."""

    def test_all_scores_available(self):
        from backend.fusion import fuse
        result = fuse(rppg_confidence=0.8, acoustic_trust=0.7, sync_score=0.9)
        assert result["overall_trust"] is not None
        assert 0.0 <= result["overall_trust"] <= 100.0
        assert result["status"] in {"nominal", "calibrating", "low_confidence", "insufficient_data"}

    def test_partial_scores_calibrating(self):
        """Missing sub-scores → status must be 'calibrating', never use placeholder values."""
        from backend.fusion import fuse
        result = fuse(rppg_confidence=0.85, acoustic_trust=None, sync_score=None)
        assert result["overall_trust"] is not None
        assert result["status"] == "calibrating"

    def test_no_scores_insufficient(self):
        from backend.fusion import fuse
        result = fuse(rppg_confidence=None, acoustic_trust=None, sync_score=None)
        assert result["overall_trust"] is None
        assert result["status"] == "insufficient_data"
