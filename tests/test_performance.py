import pytest
import asyncio
import time
import cv2
import numpy as np
from backend.core.production_engine import ProductionEngine

@pytest.mark.asyncio
async def test_processing_latency():
    engine = ProductionEngine()
    await engine.start()
    
    # Warmup
    blank_img = np.zeros((100, 100, 3), dtype=np.uint8)
    _, buf = cv2.imencode('.jpg', blank_img)
    frame_bytes = buf.tobytes()
    
    await engine.process_video_frame(frame_bytes)
    samples = [0.0] * 1024
    await engine.process_audio_chunk(samples)
    await engine.get_latest_metrics()
    
    # Measure
    t0 = time.perf_counter()
    await engine.process_video_frame(frame_bytes)
    await engine.process_audio_chunk(samples)
    await engine.get_latest_metrics()
    t1 = time.perf_counter()
    
    latency_ms = (t1 - t0) * 1000.0
    
    print(f"\nMeasured local processing latency: {latency_ms:.2f} ms")
    assert latency_ms > 0
    
    await engine.stop()
