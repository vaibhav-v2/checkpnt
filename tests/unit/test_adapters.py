"""
Tests for the LangGraph adapter — extract, reconstruct, round-trip, edge cases.
No LangGraph installation required: we test against plain dicts that mirror
the StateSnapshot structure, which is exactly what the adapter receives in practice.
"""

import pytest
from types import SimpleNamespace
from checkpnt.adapters.langgraph import LangGraphAdapter, CheckpntSaver
from checkpnt.schemas.checkpoint import Framework, Checkpoint


# ── Fixtures ──────────────────────────────────────────────────────────────────

def make_snapshot(values=None, next_nodes=None, thread_id="t-001", step=3, source="loop"):
    """Build a dict that mirrors a LangGraph StateSnapshot."""
    return {
        "values": values if values is not None else {"messages": ["hello", "world"], "count": 3},
        "next": next_nodes if next_nodes is not None else ["tool_node"],
        "config": {"configurable": {"thread_id": thread_id}},
        "metadata": {"step": step, "source": source, "parents": {}},
        "parent_config": None,
    }


def make_snapshot_obj(**kwargs):
    """Build a SimpleNamespace that mimics a LangGraph StateSnapshot object."""
    snap = make_snapshot(**kwargs)
    return SimpleNamespace(
        values=snap["values"],
        next=snap["next"],
        config=snap["config"],
        metadata=snap["metadata"],
        parent_config=snap["parent_config"],
    )


@pytest.fixture
def adapter():
    return LangGraphAdapter()


# ── extract — dict input ──────────────────────────────────────────────────────

def test_extract_from_dict(adapter):
    snapshot = make_snapshot()
    cp = adapter.extract(snapshot, agent_id="agent-1", session_id="sess-1")

    assert cp.framework == Framework.LANGGRAPH
    assert cp.agent_id == "agent-1"
    assert cp.session_id == "sess-1"
    assert cp.execution_state["values"] == {"messages": ["hello", "world"], "count": 3}
    assert cp.execution_state["next"] == ["tool_node"]
    assert cp.execution_state["thread_id"] == "t-001"


def test_extract_from_snapshot_object(adapter):
    snapshot = make_snapshot_obj()
    cp = adapter.extract(snapshot, agent_id="agent-1")

    assert cp.framework == Framework.LANGGRAPH
    assert cp.execution_state["values"]["messages"] == ["hello", "world"]


def test_extract_infers_step_from_metadata(adapter):
    snapshot = make_snapshot(step=7)
    cp = adapter.extract(snapshot, agent_id="a")
    assert cp.step_index == 7


def test_extract_infers_session_from_thread_id(adapter):
    snapshot = make_snapshot(thread_id="my-thread-xyz")
    cp = adapter.extract(snapshot, agent_id="a")
    assert cp.session_id == "my-thread-xyz"


def test_extract_infers_step_name_from_next(adapter):
    snapshot = make_snapshot(next_nodes=["validate_node"])
    cp = adapter.extract(snapshot, agent_id="a")
    assert "validate_node" in cp.step_name


def test_extract_step_name_override(adapter):
    snapshot = make_snapshot()
    cp = adapter.extract(snapshot, agent_id="a", step_name="my_custom_step")
    assert cp.step_name == "my_custom_step"


def test_extract_with_context(adapter):
    snapshot = make_snapshot()
    cp = adapter.extract(
        snapshot, agent_id="a",
        context={"invoice_id": "INV-001", "vendor": "Acme"}
    )
    assert cp.agent_context["invoice_id"] == "INV-001"
    assert cp.agent_context["vendor"] == "Acme"


def test_extract_with_parent_id(adapter):
    snapshot = make_snapshot()
    cp = adapter.extract(snapshot, agent_id="a", parent_id="parent-123")
    assert cp.parent_id == "parent-123"


def test_extract_with_ttl(adapter):
    snapshot = make_snapshot()
    cp = adapter.extract(snapshot, agent_id="a", ttl_seconds=1800)
    assert cp.ttl_seconds == 1800


def test_extract_with_metadata_tags(adapter):
    snapshot = make_snapshot()
    cp = adapter.extract(snapshot, agent_id="a", env="production", run_id="run-42")
    assert cp.metadata["env"] == "production"
    assert cp.metadata["run_id"] == "run-42"


# ── reconstruct ───────────────────────────────────────────────────────────────

def test_reconstruct_returns_values(adapter):
    snapshot = make_snapshot()
    cp = adapter.extract(snapshot, agent_id="a")
    restored = adapter.reconstruct(cp)

    assert restored["values"] == snapshot["values"]
    assert restored["next"] == snapshot["next"]
    assert restored["thread_id"] == "t-001"
    assert restored["checkpoint_id"] == cp.checkpoint_id


def test_reconstruct_wrong_framework_raises(adapter):
    from checkpnt.schemas.checkpoint import CheckpointBuilder
    from checkpnt.exceptions import AdapterError

    cp = CheckpointBuilder(agent_id="x", framework=Framework.CREWAI).build()
    with pytest.raises(AdapterError, match="crewai"):
        adapter.reconstruct(cp)


# ── Round-trip ────────────────────────────────────────────────────────────────

def test_round_trip_preserves_values(adapter):
    original_values = {"messages": ["a", "b", "c"], "step": 5, "result": {"key": "val"}}
    snapshot = make_snapshot(values=original_values)

    cp = adapter.extract(snapshot, agent_id="a", context={"ctx_key": "ctx_val"})
    restored = adapter.reconstruct(cp)

    assert restored["values"] == original_values
    assert restored["agent_context"]["ctx_key"] == "ctx_val"


def test_round_trip_via_serialisation(adapter):
    """Checkpoints that survive as_dict → from_dict round-trip."""
    snapshot = make_snapshot()
    cp = adapter.extract(snapshot, agent_id="a")

    # Simulate backend storage: serialise and deserialise
    cp2 = Checkpoint.from_dict(cp.as_dict())
    restored = adapter.reconstruct(cp2)

    assert restored["values"] == snapshot["values"]
    assert restored["checkpoint_id"] == cp.checkpoint_id


# ── Integrity ─────────────────────────────────────────────────────────────────

def test_extracted_checkpoint_passes_integrity(adapter):
    snapshot = make_snapshot()
    cp = adapter.extract(snapshot, agent_id="a")
    assert cp.verify_integrity() is True


# ── Edge cases ────────────────────────────────────────────────────────────────

def test_extract_empty_values(adapter):
    snapshot = make_snapshot(values={})
    cp = adapter.extract(snapshot, agent_id="a")
    assert cp.execution_state["values"] == {}


def test_extract_empty_next(adapter):
    snapshot = make_snapshot(next_nodes=[])
    cp = adapter.extract(snapshot, agent_id="a")
    assert cp.execution_state["next"] == []


def test_extract_complex_nested_values(adapter):
    values = {
        "messages": [{"role": "user", "content": "hello"}],
        "tool_results": [{"tool": "search", "result": {"hits": 10}}],
        "metadata": {"run_id": "abc", "tags": ["prod", "v2"]},
    }
    snapshot = make_snapshot(values=values)
    cp = adapter.extract(snapshot, agent_id="a")
    restored = adapter.reconstruct(cp)
    assert restored["values"]["tool_results"][0]["result"]["hits"] == 10


# ── CheckpntSaver ─────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_checkpnt_saver_aput_and_aget(tmp_path):
    """CheckpntSaver saves and restores state via aput/aget_tuple."""
    from checkpnt.core import Client

    db = str(tmp_path / "saver_test.db")
    client = Client.sqlite(db)
    saver = CheckpntSaver(client)

    config = {"configurable": {"thread_id": "saver-thread-001"}}
    checkpoint = {
        "v": 1,
        "id": "chk-001",
        "channel_values": {"messages": ["hello"], "step": 1},
        "channel_versions": {},
        "versions_seen": {},
        "pending_sends": [],
    }
    metadata = {"step": 1, "source": "loop", "parents": {}}

    # Save
    new_config = await saver.aput(config, checkpoint, metadata, {})
    assert "checkpoint_id" in new_config["configurable"]

    # Restore
    tuple_result = await saver.aget_tuple(config)
    assert tuple_result is not None
    assert tuple_result.checkpoint["channel_values"]["step"] == 1

    await client.close()


@pytest.mark.asyncio
async def test_checkpnt_saver_alist(tmp_path):
    """CheckpntSaver.alist yields checkpoint tuples in history order."""
    from checkpnt.core import Client

    db = str(tmp_path / "saver_list_test.db")
    client = Client.sqlite(db)
    saver = CheckpntSaver(client)

    config = {"configurable": {"thread_id": "saver-list-thread"}}

    for i in range(3):
        await saver.aput(
            config,
            {"channel_values": {"step": i}, "v": 1, "id": f"id-{i}",
             "channel_versions": {}, "versions_seen": {}, "pending_sends": []},
            {"step": i, "source": "loop", "parents": {}},
            {},
        )

    results = []
    async for t in saver.alist(config):
        results.append(t)

    assert len(results) == 3
    await client.close()
