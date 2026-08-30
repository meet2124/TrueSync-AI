# Assumption: pytest-asyncio >=0.21 with asyncio_mode="auto" or per-test @pytest.mark.asyncio decorators.
"""
backend/tests/test_mock_engine.py
==================================
Integration tests for MockEngine.

Key assertions:
  * overall_trust stays within the configured [MIN, MAX] range (scaled to 0–100).
  * bpm stays within [60, 90] across many ticks.
  * Consecutive overall_trust values never jump by more than MAX_DELTA,
    proving the bounded random-walk is "smooth" and not slot-machine random.
  * All five abstract methods exist and are awaitable.
  * get_latest_metrics() always returns a valid BiometricFrame.
"""
from __future__ import annotations

import asyncio
import math

import pytest
import pytest_asyncio

from backend.core.mock_engine import MockEngine, _TRUST_MIN, _TRUST_MAX, _BPM_MIN, _BPM_MAX
from backend.core.schemas import BiometricFrame

# Maximum allowed single-tick jump in overall_trust (0–100 scale).
# We permit up to 3× the per-tick sigma × scaling factor with generous headroom.
_MAX_TRUST_DELTA_100: float = 5.0   # 5 out of 100 — generous but meaningful guard


@pytest.fixture
def engine() -> MockEngine:
    return MockEngine()


@pytest.fixture
def started_engine(engine: MockEngine) -> MockEngine:
    """Run start() synchronously via asyncio.run so we can use it in sync tests."""
    asyncio.get_event_loop().run_until_complete(engine.start())
    return engine


# ── Interface completeness ────────────────────────────────────────────────────

class TestMockEngineInterface:
    def test_has_start(self, engine: MockEngine):
        assert callable(engine.start)

    def test_has_process_video_frame(self, engine: MockEngine):
        assert callable(engine.process_video_frame)

    def test_has_process_audio_chunk(self, engine: MockEngine):
        assert callable(engine.process_audio_chunk)

    def test_has_get_latest_metrics(self, engine: MockEngine):
        assert callable(engine.get_latest_metrics)

    def test_has_stop(self, engine: MockEngine):
        assert callable(engine.stop)

    @pytest.mark.asyncio
    async def test_start_is_idempotent(self, engine: MockEngine):
        """Calling start() twice must not raise."""
        await engine.start()
        await engine.start()

    @pytest.mark.asyncio
    async def test_stop_before_start_is_safe(self, engine: MockEngine):
        """stop() must not raise even if start() was never called."""
        await engine.stop()

    @pytest.mark.asyncio
    async def test_process_video_frame_is_noop(self, started_engine: MockEngine):
        """process_video_frame must not raise; it's a no-op in Phase 1."""
        await started_engine.process_video_frame(b"\xff\xd8\xff")   # fake JPEG header

    @pytest.mark.asyncio
    async def test_process_audio_chunk_is_noop(self, started_engine: MockEngine):
        """process_audio_chunk must not raise; it's a no-op in Phase 1."""
        await started_engine.process_audio_chunk([0.0, 0.1, -0.1, 0.05])


# ── Schema correctness ────────────────────────────────────────────────────────

class TestMockEngineSchema:
    @pytest.mark.asyncio
    async def test_returns_biometric_frame(self, started_engine: MockEngine):
        frame = await started_engine.get_latest_metrics()
        assert isinstance(frame, BiometricFrame)

    @pytest.mark.asyncio
    async def test_engine_mode_is_mock(self, started_engine: MockEngine):
        frame = await started_engine.get_latest_metrics()
        assert frame.engine_mode == "mock"

    @pytest.mark.asyncio
    async def test_rppg_waveform_has_samples(self, started_engine: MockEngine):
        frame = await started_engine.get_latest_metrics()
        assert len(frame.rppg_waveform) > 0

    @pytest.mark.asyncio
    async def test_active_rois_populated(self, started_engine: MockEngine):
        frame = await started_engine.get_latest_metrics()
        assert len(frame.active_rois) > 0


# ── Boundary / smoothness tests ───────────────────────────────────────────────

class TestMockEngineBounds:
    @pytest.mark.asyncio
    async def test_bpm_stays_in_range(self, started_engine: MockEngine):
        """BPM must stay within [60, 90] across 300 ticks (10 seconds at 30 Hz)."""
        for _ in range(300):
            frame = await started_engine.get_latest_metrics()
            assert _BPM_MIN <= frame.bpm <= _BPM_MAX, (
                f"BPM {frame.bpm} out of range [{_BPM_MIN}, {_BPM_MAX}]"
            )

    @pytest.mark.asyncio
    async def test_overall_trust_stays_in_range(self, started_engine: MockEngine):
        """overall_trust (0–100) must stay within [TRUST_MIN*100, TRUST_MAX*100]."""
        lo = _TRUST_MIN * 100.0
        hi = _TRUST_MAX * 100.0
        for _ in range(300):
            frame = await started_engine.get_latest_metrics()
            assert lo <= frame.overall_trust <= hi, (
                f"overall_trust {frame.overall_trust} out of range [{lo}, {hi}]"
            )

    @pytest.mark.asyncio
    async def test_overall_trust_is_smooth(self, started_engine: MockEngine):
        """No single tick may cause a jump larger than _MAX_TRUST_DELTA_100."""
        prev_trust: float | None = None
        for _ in range(300):
            frame = await started_engine.get_latest_metrics()
            trust = frame.overall_trust
            if prev_trust is not None:
                delta = abs(trust - prev_trust)
                assert delta <= _MAX_TRUST_DELTA_100, (
                    f"overall_trust jumped {delta:.4f} in one tick (max {_MAX_TRUST_DELTA_100})"
                )
            prev_trust = trust

    @pytest.mark.asyncio
    async def test_rppg_liveness_in_range(self, started_engine: MockEngine):
        for _ in range(100):
            frame = await started_engine.get_latest_metrics()
            if frame.rppg_liveness is not None:
                assert 0.0 <= frame.rppg_liveness <= 1.0

    @pytest.mark.asyncio
    async def test_audio_trust_in_range(self, started_engine: MockEngine):
        for _ in range(100):
            frame = await started_engine.get_latest_metrics()
            if frame.audio_trust is not None:
                assert 0.0 <= frame.audio_trust <= 1.0

    @pytest.mark.asyncio
    async def test_sync_score_in_range(self, started_engine: MockEngine):
        for _ in range(100):
            frame = await started_engine.get_latest_metrics()
            if frame.sync_score is not None:
                assert 0.0 <= frame.sync_score <= 1.0

    @pytest.mark.asyncio
    async def test_trust_is_not_constant(self, started_engine: MockEngine):
        """Trust must not be static — a living random walk should show variation."""
        values = []
        for _ in range(100):
            frame = await started_engine.get_latest_metrics()
            values.append(frame.overall_trust)
        # Standard deviation should be meaningfully non-zero
        mean = sum(values) / len(values)
        variance = sum((v - mean) ** 2 for v in values) / len(values)
        std = math.sqrt(variance)
        assert std > 0.01, f"overall_trust appears static (std={std:.6f})"
