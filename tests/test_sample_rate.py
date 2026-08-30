import pytest
import asyncio
from backend.core.production_engine import ProductionEngine

@pytest.mark.asyncio
async def test_sample_rate_resampling():
    engine = ProductionEngine()
    await engine.start()
    
    # 44100 Hz input with 1024 samples
    samples_44k = [0.0] * 1024
    await engine.process_audio_chunk(samples_44k, sample_rate=44100)
    
    # Check that it didn't crash and we can fetch metrics
    metrics = await engine.get_latest_metrics()
    assert metrics is not None
    
    # 48000 Hz input
    samples_48k = [0.0] * 1024
    await engine.process_audio_chunk(samples_48k, sample_rate=48000)
    
    metrics = await engine.get_latest_metrics()
    assert metrics is not None
    
    await engine.stop()
