"""
backend/tests/test_fusion_safety.py
=====================================
Tests for fusion.py NaN/Inf safety, signal-state distinction,
and output bound guarantees.
"""
from __future__ import annotations

import math
import pytest

from backend.fusion import fuse, _clamp_score, THRESHOLD_NOMINAL, THRESHOLD_LOW_CONFIDENCE


class TestClampScore:
    def test_none_passes_through(self):
        assert _clamp_score(None) is None

    def test_nan_becomes_none(self):
        assert _clamp_score(float("nan")) is None

    def test_inf_becomes_none(self):
        assert _clamp_score(float("inf")) is None

    def test_neg_inf_becomes_none(self):
        assert _clamp_score(float("-inf")) is None

    def test_above_one_clamped(self):
        assert _clamp_score(1.5) == 1.0

    def test_below_zero_clamped(self):
        assert _clamp_score(-0.1) == 0.0

    def test_valid_value_unchanged(self):
        assert _clamp_score(0.75) == 0.75


class TestFusionOutputBounds:
    def test_all_available_trust_in_range(self):
        result = fuse(0.8, 0.7, 0.9)
        assert result["overall_trust"] is not None
        assert 0.0 <= result["overall_trust"] <= 100.0

    def test_all_none_returns_insufficient_data(self):
        result = fuse(None, None, None)
        assert result["overall_trust"] is None
        assert result["status"] == "insufficient_data"

    def test_partial_scores_return_calibrating(self):
        result = fuse(0.85, None, None)
        assert result["status"] == "calibrating"
        assert result["overall_trust"] is not None

    def test_high_scores_return_nominal(self):
        # All scores at 1.0 → overall_trust = 100 → nominal
        result = fuse(1.0, 1.0, 1.0)
        assert result["status"] == "nominal"
        assert result["overall_trust"] >= THRESHOLD_NOMINAL

    def test_low_scores_return_low_confidence(self):
        # All scores at 0.0 → overall_trust = 0 → low_confidence
        result = fuse(0.0, 0.0, 0.0)
        assert result["status"] == "low_confidence"

    def test_nan_input_treated_as_unavailable(self):
        # NaN sub-score should be treated as None (unavailable), not as 0
        result = fuse(float("nan"), 0.8, 0.7)
        # rppg treated as unavailable → only acoustic+sync fused → calibrating
        assert result["status"] == "calibrating"
        assert result["overall_trust"] is not None

    def test_inf_input_treated_as_unavailable(self):
        result = fuse(0.8, float("inf"), 0.7)
        assert result["status"] == "calibrating"

    def test_output_never_exceeds_100(self):
        # Even with floating-point arithmetic, must stay ≤ 100
        result = fuse(1.0, 1.0, 1.0)
        assert result["overall_trust"] <= 100.0

    def test_output_never_below_0(self):
        result = fuse(0.0, 0.0, 0.0)
        assert result["overall_trust"] >= 0.0

    def test_weight_renormalisation_correct(self):
        # Only rPPG available: weight should be 1.0 (W_RPPG / W_RPPG = 1.0)
        result = fuse(0.5, None, None)
        # 0.5 × 1.0 × 100 = 50
        assert result["overall_trust"] is not None
        assert abs(result["overall_trust"] - 50.0) < 0.01
