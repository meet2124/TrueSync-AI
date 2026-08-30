import pytest
import asyncio
import math

from backend.core.production_engine import ProductionEngine
from backend.core.schemas import BiometricFrame


@pytest.mark.asyncio
async def test_nan_inf_never_reaches_schema():
    engine = ProductionEngine()
    await engine.start()
    
    # Inject NaN / Inf artificially into the modules to see if ProductionEngine catches them
    engine._rppg._bpm = float('nan')
    engine._rppg._confidence = float('inf')
    engine._rppg._filtered_signal = [float('nan'), float('inf'), 0.5]
    
    engine._acoustic._acoustic_trust = float('nan')
    engine._sync._sync_score = float('nan')

    metrics = await engine.get_latest_metrics()
    
    # Assert NaN / Inf were caught and nullified / sanitized
    assert metrics.bpm is None
    assert metrics.rppg_liveness is None
    assert metrics.audio_trust is None
    assert metrics.sync_score is None
    
    # Assert waveform only contains finite numbers
    for val in metrics.rppg_waveform:
        assert math.isfinite(val)
        
    await engine.stop()


@pytest.mark.asyncio
async def test_get_latest_metrics_returns_biometric_frame():
    engine = ProductionEngine()
    await engine.start()
    metrics = await engine.get_latest_metrics()
    
    assert isinstance(metrics, BiometricFrame)
    assert metrics.engine_mode == "production"
    assert metrics.status_flag in ["calibrating", "low_confidence", "nominal", "spoof_suspected"]
    assert hasattr(metrics, "timestamp")
    
    await engine.stop()
