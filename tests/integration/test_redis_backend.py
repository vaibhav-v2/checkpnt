"""
Integration tests for the Redis backend.

Requires a running Redis instance. Skipped automatically if Redis is unavailable.

Run with:
    pytest tests/integration/test_redis_backend.py -v

Or against a specific URL:
    CHECKPNT_TEST_REDIS_URL=redis://myhost:6379 pytest tests/integration/
"""

import os
import pytest
import asyncio
from checkpnt.backends.redis import RedisBackend
from checkpnt.schemas.checkpoint import CheckpointBuilder, Framework
from checkpnt.exceptions import CheckpointConflictError, BackendConnectionError

REDIS_URL = os.environ.get("CHECKPNT_TEST_REDIS_URL", "redis://localhost:6379")


def make_cp(agent_id="agent-r", session_id="sess-r", step=0, parent_id=None, ttl=None):
    b = (
        CheckpointBuilder(agent_id=agent_id, framework=Framework.LANGGRAPH)
        .session(session_id)
        .step(step, name=f"step_{step}")
        .execution_state({"step": step, "data": f"payload_{step}"})
        .context({"processed": step})
    )
    if parent_id:
        b = b.parent(parent_id)
    if ttl:
        b = b.ttl(ttl)
    return b.build()


@pytest.fixture(scope="module")
def event_loop():
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(scope="module")
async def redis_backend():
    """Skip all tests if Redis is not available."""
    backend = RedisBackend(REDIS_URL)
    try:
        await backend._conn()
    except BackendConnectionError:
        pytest.skip(f"Redis not available at {REDIS_URL}")
    yield backend
    await backend.close()


@pytest.fixture(autouse=True)
async def cleanup(redis_backend):
    """Flush test keys between tests."""
    yield
    # Clean up test keys
    r = await redis_backend._conn()
    keys = await r.keys("checkpnt:*")
    if keys:
        await r.delete(*keys)


# ── Save & Load ───────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_redis_save_and_load(redis_backend):
    cp = make_cp()
    cid = await redis_backend.save(cp)
    assert cid == cp.checkpoint_id

    loaded = await redis_backend.load(cid)
    assert loaded is not None
    assert loaded.checkpoint_id == cp.checkpoint_id
    assert loaded.execution_state == cp.execution_state
    assert loaded.agent_context == cp.agent_context


@pytest.mark.asyncio
async def test_redis_load_nonexistent(redis_backend):
    result = await redis_backend.load("ghost-id-xyz")
    assert result is None


@pytest.mark.asyncio
async def test_redis_immutability(redis_backend):
    cp = make_cp()
    await redis_backend.save(cp)
    with pytest.raises(CheckpointConflictError):
        await redis_backend.save(cp)


# ── Latest ────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_redis_latest(redis_backend):
    cp1 = make_cp(step=0)
    cp2 = make_cp(step=1, parent_id=cp1.checkpoint_id)
    cp3 = make_cp(step=2, parent_id=cp2.checkpoint_id)

    for cp in [cp1, cp2, cp3]:
        await redis_backend.save(cp)

    latest = await redis_backend.latest("agent-r", "sess-r")
    assert latest is not None
    assert latest.checkpoint_id == cp3.checkpoint_id


@pytest.mark.asyncio
async def test_redis_latest_empty(redis_backend):
    result = await redis_backend.latest("nobody", "no-session")
    assert result is None


# ── History ───────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_redis_history_order(redis_backend):
    parent_id = None
    for i in range(5):
        cp = make_cp(step=i, parent_id=parent_id)
        await redis_backend.save(cp)
        parent_id = cp.checkpoint_id

    history = await redis_backend.history("agent-r", "sess-r")
    assert len(history) == 5
    assert history[0].step_index == 4   # newest first
    assert history[-1].step_index == 0


@pytest.mark.asyncio
async def test_redis_history_limit(redis_backend):
    for i in range(8):
        await redis_backend.save(make_cp(step=i))
    history = await redis_backend.history("agent-r", "sess-r", limit=3)
    assert len(history) == 3


# ── Delete ────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_redis_delete(redis_backend):
    cp = make_cp()
    await redis_backend.save(cp)
    deleted = await redis_backend.delete(cp.checkpoint_id)
    assert deleted is True
    assert await redis_backend.load(cp.checkpoint_id) is None


@pytest.mark.asyncio
async def test_redis_delete_nonexistent(redis_backend):
    result = await redis_backend.delete("never-saved")
    assert result is False


# ── TTL ───────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_redis_ttl_expires(redis_backend):
    import time
    cp = make_cp(ttl=1)
    await redis_backend.save(cp)

    loaded = await redis_backend.load(cp.checkpoint_id)
    assert loaded is not None

    time.sleep(1.2)

    loaded = await redis_backend.load(cp.checkpoint_id)
    assert loaded is None


# ── Integrity ─────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_redis_integrity_on_load(redis_backend):
    cp = make_cp()
    await redis_backend.save(cp)
    loaded = await redis_backend.load(cp.checkpoint_id)
    assert loaded.verify_integrity() is True


# ── Session isolation ─────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_redis_session_isolation(redis_backend):
    cp_a = make_cp(agent_id="agent-a", session_id="sess-a", step=0)
    cp_b = make_cp(agent_id="agent-b", session_id="sess-b", step=0)
    await redis_backend.save(cp_a)
    await redis_backend.save(cp_b)

    hist_a = await redis_backend.history("agent-a", "sess-a")
    hist_b = await redis_backend.history("agent-b", "sess-b")
    assert len(hist_a) == 1
    assert len(hist_b) == 1
    assert hist_a[0].checkpoint_id != hist_b[0].checkpoint_id


# ── Connection error ──────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_redis_bad_url_raises():
    backend = RedisBackend("redis://127.0.0.1:19999")  # Nothing listening here
    with pytest.raises(BackendConnectionError):
        await backend._conn()
    await backend.close()
