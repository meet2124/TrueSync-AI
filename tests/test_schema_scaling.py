import pytest
from backend.core.schemas import BiometricFrame
from backend.fusion import fuse
from pydantic import ValidationError

def test_fusion_bounds():
    # Test that fusion strictly outputs between 0 and 100
    result_max = fuse(1.0, 1.0, 1.0)
    assert result_max["overall_trust"] == 100.0

    result_min = fuse(0.0, 0.0, 0.0)
    assert result_min["overall_trust"] == 0.0
    
    # Test that normal values produce bounded results
    result_mid = fuse(0.5, 0.8, 0.2)
    assert 0 <= result_mid["overall_trust"] <= 100.0

def test_fusion_negative_input():
    # Test that negative sub-scores are clamped to 0 internally
    result = fuse(-1.0, -5.0, -100.0)
    assert result["overall_trust"] == 0.0

def test_fusion_excessive_input():
    # Test that excessive sub-scores > 1.0 are clamped to 1.0 internally
    result = fuse(1.5, 5.0, 100.0)
    assert result["overall_trust"] == 100.0

def test_biometric_frame_schema_validation():
    # Verify that BiometricFrame accepts correct bounds
    frame = BiometricFrame(
        timestamp=123.0,
        session_id="test",
        engine_mode="production",
        bpm=60.0,
        rppg_waveform=[],
        rppg_liveness=0.8,    # 0 to 1
        audio_trust=0.9,      # 0 to 1
        sync_score=0.7,       # 0 to 1
        overall_trust=85.5,   # 0 to 100
        active_rois=[],
        status_flag="nominal"
    )
    
    # This just ensures we don't accidentally multiply by 100 in the schema itself.
    assert frame.overall_trust == 85.5
    assert frame.rppg_liveness == 0.8

    # The schema doesn't currently strictly reject >100 values via validators 
    # (it relies on fusion bounds), but if strict Pydantic bounds were added, 
    # this test would ensure they are correct.
