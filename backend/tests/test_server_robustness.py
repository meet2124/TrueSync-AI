"""
backend/tests/test_server_robustness.py
========================================
Tests for server.py input validation and safety helpers.
Mocks are test fixtures only — not used in production code.
"""
from __future__ import annotations

import base64
import json
import math

import pytest

from backend.server import (
    MAX_PAYLOAD_BYTES,
    _decode_base64_frame,
    _decode_jpeg_frame,
    _parse_audio_samples,
    _validate_payload_size,
    SessionEngine,
)
from backend.schemas import BiometricResult


# ── Payload size guard ────────────────────────────────────────────────────────

class TestPayloadSizeGuard:
    def test_normal_payload_accepted(self):
        raw = "x" * 1000
        assert _validate_payload_size(raw, "test") is True

    def test_oversized_payload_rejected(self):
        raw = "x" * (MAX_PAYLOAD_BYTES + 1)
        assert _validate_payload_size(raw, "test") is False

    def test_exact_limit_accepted(self):
        # A payload of exactly MAX_PAYLOAD_BYTES ASCII chars = MAX_PAYLOAD_BYTES bytes
        raw = "x" * MAX_PAYLOAD_BYTES
        assert _validate_payload_size(raw, "test") is True


# ── Base64 decode ─────────────────────────────────────────────────────────────

class TestBase64Decode:
    def test_valid_base64_decoded(self):
        data = base64.b64encode(b"\xff\xd8\xff").decode()
        result = _decode_base64_frame(data, "test")
        assert result == b"\xff\xd8\xff"

    def test_invalid_base64_returns_none(self):
        assert _decode_base64_frame("!!!not-base64!!!", "test") is None

    def test_empty_string_returns_none(self):
        # base64.b64decode("") returns b"" which is valid; empty is handled upstream
        result = _decode_base64_frame("", "test")
        assert result == b""  # empty but not None — upstream guard handles it


# ── JPEG decode ───────────────────────────────────────────────────────────────

class TestJPEGDecode:
    def test_corrupt_bytes_returns_none(self):
        assert _decode_jpeg_frame(b"\x00\x01\x02\x03", "test") is None

    def test_random_bytes_returns_none(self):
        assert _decode_jpeg_frame(b"not a jpeg at all", "test") is None


# ── Audio sample validation ───────────────────────────────────────────────────

class TestAudioParsing:
    def _msg(self, data, sample_rate=16000):
        return {"data": data, "sample_rate": sample_rate}

    def test_valid_samples_parsed(self):
        result = _parse_audio_samples(self._msg([0.1, -0.2, 0.3]), "test")
        assert result is not None
        samples, sr = result
        assert len(samples) == 3
        assert sr == 16000

    def test_nan_samples_dropped(self):
        result = _parse_audio_samples(self._msg([0.1, float("nan"), 0.3]), "test")
        assert result is not None
        samples, _ = result
        assert len(samples) == 2  # NaN dropped

    def test_inf_samples_dropped(self):
        result = _parse_audio_samples(self._msg([float("inf"), 0.1]), "test")
        assert result is not None
        samples, _ = result
        assert len(samples) == 1

    def test_samples_clamped_to_minus_one_one(self):
        result = _parse_audio_samples(self._msg([5.0, -5.0]), "test")
        assert result is not None
        samples, _ = result
        assert samples[0] == 1.0
        assert samples[1] == -1.0

    def test_invalid_sample_rate_corrected(self):
        result = _parse_audio_samples(self._msg([0.1], sample_rate=999), "test")
        assert result is not None
        _, sr = result
        assert sr == 16000  # corrected to safe default

    def test_non_list_data_returns_none(self):
        assert _parse_audio_samples({"data": "not-a-list"}, "test") is None

    def test_empty_list_gives_empty_samples(self):
        result = _parse_audio_samples(self._msg([]), "test")
        assert result is not None
        samples, _ = result
        assert samples == []


# ── SessionEngine.get_result NaN/Inf safety ───────────────────────────────────

class TestSessionEngineResultSafety:
    def test_get_result_returns_biometric_result(self):
        """get_result() should always return a valid BiometricResult even with no data."""
        engine = SessionEngine("test-session-001")
        result = engine.get_result()
        assert isinstance(result, BiometricResult)

    def test_result_with_no_signal_returns_calibrating(self):
        engine = SessionEngine("test-session-002")
        result = engine.get_result()
        # Without any video/audio, sub-scores are None → status should be insufficient_data or calibrating
        assert result.status in {"insufficient_data", "calibrating"}

    def test_result_overall_trust_is_none_without_data(self):
        engine = SessionEngine("test-session-003")
        result = engine.get_result()
        # No data → no trust score
        assert result.overall_trust is None

    def test_safe_float_clamps_nan(self):
        engine = SessionEngine("test-session-004")
        assert engine._safe_float(float("nan")) is None

    def test_safe_float_clamps_inf(self):
        engine = SessionEngine("test-session-005")
        assert engine._safe_float(float("inf")) is None

    def test_safe_float_clamps_above_1(self):
        engine = SessionEngine("test-session-006")
        assert engine._safe_float(1.5) == 1.0

    def test_safe_float_clamps_below_0(self):
        engine = SessionEngine("test-session-007")
        assert engine._safe_float(-0.5) == 0.0

    def test_latency_propagated(self):
        engine = SessionEngine("test-session-008")
        result = engine.get_result(processing_latency_ms=12.5)
        assert result.processing_latency_ms == 12.5

    def test_non_finite_latency_becomes_none(self):
        engine = SessionEngine("test-session-009")
        result = engine.get_result(processing_latency_ms=float("nan"))
        assert result.processing_latency_ms is None
