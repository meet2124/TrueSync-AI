"""
tests/test_acoustic.py
======================
Unit tests for the AcousticModule using deterministic synthetic audio.

All signals here are explicit test fixtures (Section 0). Not production data.
"""
from __future__ import annotations

import math
import numpy as np
import pytest

from backend.engine import AcousticModule


SR = 16000  # Hz


def _sine_tone(freq_hz: float, duration_s: float, sr: int = SR) -> list[float]:
    """Generate a pure sine tone (test fixture — not production data)."""
    t = np.linspace(0, duration_s, int(sr * duration_s), endpoint=False)
    return (np.sin(2 * math.pi * freq_hz * t) * 0.5).tolist()


def _white_noise(duration_s: float, sr: int = SR, seed: int = 42) -> list[float]:
    """Generate white noise (deterministic seed — test fixture)."""
    rng = np.random.default_rng(seed)
    return (rng.uniform(-0.5, 0.5, int(sr * duration_s))).tolist()


class TestAcousticModuleInsufficient:
    def test_no_score_before_min_buffer(self):
        """Module should return None if < 250ms audio accumulated."""
        mod = AcousticModule()
        tiny = _sine_tone(440.0, 0.1)  # only 100ms
        mod.update(tiny, SR, timestamp=0.0)
        score = mod.get_score()
        assert score["acoustic_trust"] is None, (
            "Should return None before 250ms minimum buffer"
        )


class TestAcousticModuleScoring:
    def _feed(self, mod: AcousticModule, samples: list[float]) -> None:
        """Feed samples in one batch past the recompute threshold."""
        mod.update(samples, SR, timestamp=0.0)
        # Force a recompute regardless of timer
        mod._total_audio_s = 1.0
        mod._last_compute_time = 0.0
        mod.update([], SR, timestamp=1.0)

    def test_score_in_valid_range(self):
        """Score must always be in [0, 1]."""
        mod = AcousticModule()
        samples = _sine_tone(440.0, 1.0) + _white_noise(0.5)
        self._feed(mod, samples)
        score = mod.get_score()["acoustic_trust"]
        if score is not None:
            assert 0.0 <= score <= 1.0, f"acoustic_trust={score} out of [0,1]"

    def test_white_noise_produces_score(self):
        """White noise (flat spectrum, irregular phase) should yield a computable score."""
        mod = AcousticModule()
        samples = _white_noise(1.0)
        self._feed(mod, samples)
        score = mod.get_score()["acoustic_trust"]
        assert score is not None, "Should produce a score for 1s of white noise"

    def test_score_is_deterministic(self):
        """Same deterministic input should produce same score on two independent instances."""
        samples = _white_noise(1.0, seed=99)

        mod1 = AcousticModule()
        mod1._samples.extend(samples)
        mod1._total_audio_s = 1.0
        mod1._recompute()

        mod2 = AcousticModule()
        mod2._samples.extend(samples)
        mod2._total_audio_s = 1.0
        mod2._recompute()

        s1 = mod1.get_score()["acoustic_trust"]
        s2 = mod2.get_score()["acoustic_trust"]
        assert s1 is not None and s2 is not None
        assert abs(s1 - s2) < 1e-6, f"Same input gave different scores: {s1} vs {s2}"
