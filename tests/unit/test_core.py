"""
Tests for the Checkpnt Client — the public API.
Uses SQLite backend with a temp DB.
"""

import pytest
import tempfile
from checkpnt.core import Client
from checkpnt.schemas.checkpoint import Framework
from checkpnt.exceptions import CheckpointNotFoundError


@pytest.fixture
async def client(tmp_path):
    db = str(tmp_path / "test_client.db")
    async with Client.sqlite(db) as c:
        yield c


# ── save ──────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_save_returns_checkpoint_id(client):
    cid = await client.save(
        agent_id="test-agent",
        framework=Framework.LANGGRAPH,
        execution_state={"step": 1},
    )
    assert isinstance(cid, str)
    assert len(cid) > 0


@pytest.mark.asyncio
async def test_save_with_context(client):
    cid = await client.save(
        agent_id="test-agent",
        framework=Framework.LANGGRAPH,
        execution_state={"step": 1},
        context={"invoice_id": "INV-001"},
        step_index=3,
        step_name="validate",
    )
    cp = await client.restore(cid)
    assert cp.agent_context["invoice_id"] == "INV-001"
    assert cp.step_index == 3
    assert cp.step_name == "validate"


# ── restore ───────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_restore_returns_correct_state(client):
    state = {"messages": ["hello", "world"], "next": ["tool"]}
    cid = await client.save(
        agent_id="a", framework=Framework.LANGGRAPH, execution_state=state
    )
    cp = await client.restore(cid)
    assert cp.execution_state == state


@pytest.mark.asyncio
async def test_restore_nonexistent_raises(client):
    with pytest.raises(CheckpointNotFoundError):
        await client.restore("does-not-exist")


@pytest.mark.asyncio
async def test_restore_verifies_integrity_by_default(client):
    cid = await client.save(
        agent_id="a", framework=Framework.CUSTOM, execution_state={}
    )
    cp = await client.restore(cid, verify=True)
    assert cp.verify_integrity() is True


# ── handoff ───────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_handoff_creates_new_checkpoint(client):
    cid = await client.save(
        agent_id="sender",
        framework=Framework.LANGGRAPH,
        execution_state={"result": "processed"},
        context={"invoice_id": "INV-999"},
    )
    handoff_id = await client.handoff(cid, target_agent_id="receiver")
    assert handoff_id != cid

    handoff_cp = await client.restore(handoff_id)
    assert handoff_cp.handoff_target == "receiver"
    assert handoff_cp.parent_id == cid
    assert handoff_cp.agent_context["invoice_id"] == "INV-999"


# ── timeline ──────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_timeline_returns_execution_history(client):
    session_id = "audit-session"
    parent_id = None

    for i in range(4):
        cid = await client.save(
            agent_id="audit-agent",
            framework=Framework.CUSTOM,
            execution_state={"step": i},
            session_id=session_id,
            parent_id=parent_id,
            step_index=i,
        )
        parent_id = cid

    history = await client.timeline("audit-agent", session_id)
    assert len(history) == 4
    assert history[0].step_index == 3  # newest first


# ── expire ────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_expire_deletes_checkpoint(client):
    cid = await client.save(
        agent_id="a", framework=Framework.CUSTOM, execution_state={}
    )
    deleted = await client.expire(cid)
    assert deleted is True

    with pytest.raises(CheckpointNotFoundError):
        await client.restore(cid)


@pytest.mark.asyncio
async def test_expire_nonexistent_returns_false(client):
    result = await client.expire("ghost-id")
    assert result is False
