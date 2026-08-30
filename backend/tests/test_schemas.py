# Assumption: pytest-asyncio is used for async test cases; asyncio_mode = "auto" is set in pyproject.toml or via decorator.
"""
backend/tests/test_schemas.py
==============================
Unit tests for the BiometricFrame Pydantic schema.

Tests validate both the happy path (valid data accepted) and edge cases
(missing required fields, out-of-range values rejected).
"""
from __future__ import annotations

import time

import pytest
from pydantic import ValidationError

from backend.core.schemas import BiometricFrame

# ── Helpers ───────────────────────────────────────────────────────────────────

def _valid_payload(**overrides) -> dict:
    """Return a minimally valid BiometricFrame payload, with optional overrides."""
    base = {
        "timestamp": time.time(),
        "session_id": "test-session-001",
        "engine_mode": "mock",
        "bpm": 72.0,
        "rppg_waveform": [0.1, -0.2, 0.3],
        "rppg_liveness": 0.91,
        "audio_trust": 0.88,
        "sync_score": 0.85,
        "overall_trust": 90.0,
        "active_rois": ["forehead", "left_cheek"],
        "status_flag": "nominal",
    }
    base.update(overrides)
    return base


# ── Happy-path tests ──────────────────────────────────────────────────────────

class TestBiometricFrameValid:
    def test_full_valid_payload(self):
        frame = BiometricFrame(**_valid_payload())
        assert frame.session_id == "test-session-001"
        assert frame.engine_mode == "mock"
        assert frame.overall_trust == 90.0

    def test_optional_fields_can_be_none(self):
        """bpm, rppg_liveness, audio_trust, sync_score, overall_trust are all Optional."""
        frame = BiometricFrame(**_valid_payload(
            bpm=None,
            rppg_liveness=None,
            audio_trust=None,
            sync_score=None,
            overall_trust=None,
        ))
        assert frame.bpm is None
        assert frame.overall_trust is None

    def test_empty_waveform_and_rois_accepted(self):
        frame = BiometricFrame(**_valid_payload(rppg_waveform=[], active_rois=[]))
        assert frame.rppg_waveform == []
        assert frame.active_rois == []

    def test_engine_mode_production(self):
        frame = BiometricFrame(**_valid_payload(engine_mode="production"))
        assert frame.engine_mode == "production"

    def test_all_status_flags(self):
        for flag in ("nominal", "calibrating", "low_confidence", "spoof_suspected"):
            frame = BiometricFrame(**_valid_payload(status_flag=flag))
            assert frame.status_flag == flag

    def test_overall_trust_boundary_values(self):
        for val in (0.0, 50.0, 100.0):
            frame = BiometricFrame(**_valid_payload(overall_trust=val))
            assert frame.overall_trust == val

    def test_sub_score_boundary_values(self):
        for val in (0.0, 0.5, 1.0):
            frame = BiometricFrame(**_valid_payload(rppg_liveness=val, audio_trust=val, sync_score=val))
            assert frame.rppg_liveness == val


# ── Rejection tests ───────────────────────────────────────────────────────────

class TestBiometricFrameInvalid:
    def test_missing_timestamp_raises(self):
        payload = _valid_payload()
        del payload["timestamp"]
        with pytest.raises(ValidationError) as exc_info:
            BiometricFrame(**payload)
        assert "timestamp" in str(exc_info.value)

    def test_missing_session_id_raises(self):
        payload = _valid_payload()
        del payload["session_id"]
        with pytest.raises(ValidationError):
            BiometricFrame(**payload)

    def test_invalid_engine_mode_raises(self):
        with pytest.raises(ValidationError):
            BiometricFrame(**_valid_payload(engine_mode="turbo"))

    def test_invalid_status_flag_raises(self):
        with pytest.raises(ValidationError):
            BiometricFrame(**_valid_payload(status_flag="unknown_status"))

    def test_overall_trust_above_100_raises(self):
        with pytest.raises(ValidationError):
            BiometricFrame(**_valid_payload(overall_trust=100.1))

    def test_overall_trust_below_0_raises(self):
        with pytest.raises(ValidationError):
            BiometricFrame(**_valid_payload(overall_trust=-0.1))

    def test_rppg_liveness_above_1_raises(self):
        with pytest.raises(ValidationError):
            BiometricFrame(**_valid_payload(rppg_liveness=1.01))

    def test_audio_trust_below_0_raises(self):
        with pytest.raises(ValidationError):
            BiometricFrame(**_valid_payload(audio_trust=-0.01))

    def test_sync_score_above_1_raises(self):
        with pytest.raises(ValidationError):
            BiometricFrame(**_valid_payload(sync_score=1.5))

    def test_bpm_below_0_raises(self):
        with pytest.raises(ValidationError):
            BiometricFrame(**_valid_payload(bpm=-1.0))
