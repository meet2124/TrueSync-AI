import pytest
import asyncio
import json
import base64
from backend.main import _handle_inbound
from backend.core.production_engine import ProductionEngine

@pytest.fixture
def mock_engine():
    engine = ProductionEngine()
    engine._running = True  # Mock running state
    return engine

@pytest.mark.asyncio
async def test_handle_video_message(mock_engine):
    b64_data = base64.b64encode(b"fake_jpeg_bytes").decode("utf-8")
    msg = json.dumps({"type": "video_frame", "timestamp": 123.45, "data": b64_data})
    
    # Should not crash
    await _handle_inbound(mock_engine, msg, "test_client")

@pytest.mark.asyncio
async def test_handle_audio_message(mock_engine):
    msg = json.dumps({"type": "audio_chunk", "timestamp": 123.45, "data": [0.1, 0.2, 0.3]})
    # Should not crash
    await _handle_inbound(mock_engine, msg, "test_client")

@pytest.mark.asyncio
async def test_handle_malformed_json(mock_engine):
    msg = "{bad_json"
    # Should gracefully ignore
    await _handle_inbound(mock_engine, msg, "test_client")

@pytest.mark.asyncio
async def test_invalid_audio_data(mock_engine):
    # Sends strings instead of floats
    msg = json.dumps({"type": "audio_chunk", "timestamp": 123.45, "data": ["a", "b"]})
    await _handle_inbound(mock_engine, msg, "test_client")

@pytest.mark.asyncio
async def test_nan_inf_audio(mock_engine):
    msg = json.dumps({"type": "audio_chunk", "timestamp": 123.45, "data": [float('nan'), float('inf')]})
    await _handle_inbound(mock_engine, msg, "test_client")

@pytest.mark.asyncio
async def test_empty_payload(mock_engine):
    msg = json.dumps({"type": "video_frame", "data": ""})
    await _handle_inbound(mock_engine, msg, "test_client")
    
    msg2 = json.dumps({"type": "audio_chunk", "data": []})
    await _handle_inbound(mock_engine, msg2, "test_client")

@pytest.mark.asyncio
async def test_oversized_payload(mock_engine):
    # Technically FastAPI WebSocket handles the byte limit, but we test the adapter's robustness
    b64_data = base64.b64encode(b"0" * 10_000_000).decode("utf-8")
    msg = json.dumps({"type": "video_frame", "data": b64_data})
    await _handle_inbound(mock_engine, msg, "test_client")

@pytest.mark.asyncio
async def test_unknown_message_type(mock_engine):
    msg = json.dumps({"type": "unknown_magic", "data": "yes"})
    await _handle_inbound(mock_engine, msg, "test_client")

@pytest.mark.asyncio
async def test_session_isolation():
    engine1 = ProductionEngine()
    engine2 = ProductionEngine()
    await engine1.start()
    await engine2.start()
    assert engine1 is not engine2
    # Ensure they don't share underlying modules
    assert engine1._rppg is not engine2._rppg
    await engine1.stop()
    await engine2.stop()
