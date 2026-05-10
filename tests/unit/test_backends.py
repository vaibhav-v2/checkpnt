"""
Tests for the SQLite backend — save, load, latest, history, delete, TTL, immutability.
"""

import asyncio
import pytest
import tempfile
import os
from checkpnt.backends.sqlite import SQLiteBackend
from checkpnt.schemas.checkpoint import CheckpointBuilder, Framework
from checkpnt.exceptions import CheckpointConflictError


def make_cp(agent_id="agent-1", session_id="sess-1", step=0, parent_id=None, ttl=None):
    b = (
        CheckpointBuilder(agent_id=agent_id, framework=Framework.LANGGRAPH)
        .session(session_id)
        .step(step, name=f"step_{step}")
        .execution_state({"step": step, "messages": [f"msg_{step}"]})
        .context({"processed": step})
    )
    if parent_id:
        b = b.parent(parent_id)
    if ttl:
        b = b.ttl(ttl)
    return b.build()


@pytest.fixture
def db_path(tmp_path):
    return str(tmp_path / "test.db")


@pytest.fixture
async def backend(db_path):
    b = SQLiteBackend(db_path)
    yield b
    await b.close()


# ── Save & Load ───────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_save_and_load(backend):
    cp = make_cp()
    cid = await backend.save(cp)
    assert cid == cp.checkpoint_id

    loaded = await backend.load(cid)
    assert loaded is not None
    assert loaded.checkpoint_id == cp.checkpoint_id
    assert loaded.step_index == cp.step_index
    assert loaded.execution_state == cp.execution_state
    assert loaded.agent_context == cp.agent_context


@pytest.mark.asyncio
async def test_load_nonexistent_returns_none(backend):
    result = await backend.load("nonexistent-id")
    assert result is None


@pytest.mark.asyncio
async def test_immutability_prevents_overwrite(backend):
    cp = make_cp()
    await backend.save(cp)
    with pytest.raises(CheckpointConflictError):
        await backend.save(cp)  # Same checkpoint_id — must fail


# ── Latest ────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_latest_returns_most_recent(backend):
    cp1 = make_cp(step=0)
    cp2 = make_cp(step=1, parent_id=cp1.checkpoint_id)
    cp3 = make_cp(step=2, parent_id=cp2.checkpoint_id)

    await backend.save(cp1)
    await backend.save(cp2)
    await backend.save(cp3)

    latest = await backend.latest("agent-1", "sess-1")
    assert latest is not None
    assert latest.checkpoint_id == cp3.checkpoint_id


@pytest.mark.asyncio
async def test_latest_on_empty_session_returns_none(backend):
    result = await backend.latest("nobody", "no-session")
    assert result is None


# ── History ───────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_history_returns_in_reverse_order(backend):
    cps = []
    parent_id = None
    for i in range(5):
        cp = make_cp(step=i, parent_id=parent_id)
        await backend.save(cp)
        parent_id = cp.checkpoint_id
        cps.append(cp)

    history = await backend.history("agent-1", "sess-1")
    assert len(history) == 5
    # Newest first
    assert history[0].step_index == 4
    assert history[-1].step_index == 0


@pytest.mark.asyncio
async def test_history_respects_limit(backend):
    for i in range(10):
        await backend.save(make_cp(step=i))

    history = await backend.history("agent-1", "sess-1", limit=3)
    assert len(history) == 3


# ── Delete ────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_delete_removes_checkpoint(backend):
    cp = make_cp()
    await backend.save(cp)
    deleted = await backend.delete(cp.checkpoint_id)
    assert deleted is True
    assert await backend.load(cp.checkpoint_id) is None


@pytest.mark.asyncio
async def test_delete_nonexistent_returns_false(backend):
    result = await backend.delete("does-not-exist")
    assert result is False


# ── TTL ───────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_expired_checkpoint_not_returned(backend):
    """A checkpoint with TTL=1 second should disappear after expiry."""
    import time
    cp = make_cp(ttl=1)
    await backend.save(cp)

    # Immediately loadable
    loaded = await backend.load(cp.checkpoint_id)
    assert loaded is not None

    # Wait for expiry
    time.sleep(1.1)

    # Now gone
    loaded = await backend.load(cp.checkpoint_id)
    assert loaded is None


# ── Integrity ─────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_loaded_checkpoint_passes_integrity(backend):
    cp = make_cp()
    await backend.save(cp)
    loaded = await backend.load(cp.checkpoint_id)
    assert loaded.verify_integrity() is True


# ── Multi-session isolation ───────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_sessions_are_isolated(backend):
    cp_a = make_cp(agent_id="agent-a", session_id="sess-a")
    cp_b = make_cp(agent_id="agent-b", session_id="sess-b")
    await backend.save(cp_a)
    await backend.save(cp_b)

    history_a = await backend.history("agent-a", "sess-a")
    history_b = await backend.history("agent-b", "sess-b")

    assert len(history_a) == 1
    assert len(history_b) == 1
    assert history_a[0].checkpoint_id != history_b[0].checkpoint_id
