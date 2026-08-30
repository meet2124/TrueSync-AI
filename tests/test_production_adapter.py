import pytest
import asyncio
import cv2
import numpy as np

from backend.core.production_engine import ProductionEngine
from backend.core.schemas import BiometricFrame


@pytest.mark.asyncio
async def test_production_adapter_implements_interface():
    engine = ProductionEngine()
    # Ensure all required abstract methods are present and runnable
    await engine.start()
    
    # Test valid image bytes
    blank_img = np.zeros((100, 100, 3), dtype=np.uint8)
    _, buf = cv2.imencode('.jpg', blank_img)
    frame_bytes = buf.tobytes()
    
    await engine.process_video_frame(frame_bytes)
    
    # Test valid audio chunk
    samples = [0.0] * 512
    await engine.process_audio_chunk(samples)
    
    metrics = await engine.get_latest_metrics()
    assert isinstance(metrics, BiometricFrame)
    assert metrics.engine_mode == "production"
    
    await engine.stop()


@pytest.mark.asyncio
async def test_production_adapter_invalid_video():
    engine = ProductionEngine()
    await engine.start()
    
    # Send garbage bytes, it should not crash the async execution
    await engine.process_video_frame(b"invalid_garbage")
    
    metrics = await engine.get_latest_metrics()
    assert metrics.status_flag in ["calibrating", "low_confidence", "nominal", "spoof_suspected"]
    
    await engine.stop()


@pytest.mark.asyncio
async def test_production_adapter_invalid_audio():
    engine = ProductionEngine()
    await engine.start()
    
    # Send strings and NaNs
    samples = ["not", "float", float("nan"), float("inf")]
    await engine.process_audio_chunk(samples)
    
    metrics = await engine.get_latest_metrics()
    assert isinstance(metrics, BiometricFrame)
    
    await engine.stop()
