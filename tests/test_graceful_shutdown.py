import pytest
import asyncio
from backend.core.production_engine import ProductionEngine

@pytest.mark.asyncio
async def test_engine_graceful_shutdown():
    engine = ProductionEngine()
    
    # Stop before start should be safe
    await engine.stop()
    assert not engine._running
    
    # Start the engine
    await engine.start()
    assert engine._running
    assert engine._rppg is not None
    assert engine._acoustic is not None
    assert engine._sync is not None
    
    # Stop the engine
    await engine.stop()
    assert not engine._running
    
    # Second stop should be safe (idempotent)
    await engine.stop()
    assert not engine._running
    
@pytest.mark.asyncio
async def test_engine_start_idempotent():
    engine = ProductionEngine()
    await engine.start()
    assert engine._running
    
    # Second start should return safely
    await engine.start()
    assert engine._running
    
    await engine.stop()
