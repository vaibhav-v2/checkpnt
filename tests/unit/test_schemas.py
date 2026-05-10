"""
Tests for Checkpoint schema — immutability, checksum, versioning, builder.
"""

import pytest
from datetime import datetime, timezone
from checkpnt.schemas.checkpoint import (
    Checkpoint, CheckpointBuilder, Framework,
    CURRENT_SCHEMA_VERSION, _compute_checksum, _new_checkpoint_id,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

def make_checkpoint(**overrides) -> Checkpoint:
    builder = CheckpointBuilder(agent_id="test-agent", framework=Framework.LANGGRAPH)
    c = builder.session("sess-001").step(1, "test_step").execution_state(
        {"messages": ["hello"], "next": ["tool_node"]}
    ).context({"key": "value"}).build()
    # Apply overrides via dict reconstruction if needed
    return c


# ── Immutability ──────────────────────────────────────────────────────────────

def test_checkpoint_is_frozen():
    c = make_checkpoint()
    with pytest.raises((AttributeError, TypeError)):
        c.step_index = 99  # type: ignore


# ── Builder ───────────────────────────────────────────────────────────────────

def test_builder_defaults():
    c = CheckpointBuilder(agent_id="a", framework=Framework.CUSTOM).build()
    assert c.agent_id == "a"
    assert c.framework == Framework.CUSTOM
    assert c.step_index == 0
    assert c.parent_id is None
    assert c.ttl_seconds is None
    assert c.handoff_target is None
    assert c.schema_version == CURRENT_SCHEMA_VERSION


def test_builder_fluent_chain():
    c = (
        CheckpointBuilder(agent_id="agent-x", framework=Framework.LANGGRAPH)
        .session("my-session")
        .parent("parent-id-123")
        .step(5, name="validate")
        .execution_state({"messages": []})
        .context({"result": 42})
        .ttl(3600)
        .tag(env="test", run_id="run-001")
        .build()
    )
    assert c.session_id == "my-session"
    assert c.parent_id == "parent-id-123"
    assert c.step_index == 5
    assert c.step_name == "validate"
    assert c.ttl_seconds == 3600
    assert c.metadata["env"] == "test"
    assert c.agent_context["result"] == 42


def test_builder_handoff():
    c = (
        CheckpointBuilder(agent_id="sender", framework=Framework.LANGGRAPH)
        .handoff("receiver-agent", schema="1.0")
        .build()
    )
    assert c.handoff_target == "receiver-agent"
    assert c.handoff_schema == "1.0"


# ── Checksum ──────────────────────────────────────────────────────────────────

def test_checksum_is_deterministic():
    state = {"messages": ["a", "b"], "step": 3}
    ctx = {"result": "done"}
    assert _compute_checksum(state, ctx) == _compute_checksum(state, ctx)


def test_checksum_changes_on_state_change():
    state1 = {"step": 1}
    state2 = {"step": 2}
    assert _compute_checksum(state1, {}) != _compute_checksum(state2, {})


def test_integrity_passes_on_valid_checkpoint():
    c = make_checkpoint()
    assert c.verify_integrity() is True


def test_integrity_detects_corruption():
    c = make_checkpoint()
    # Tamper: build a new frozen object with a bad checksum
    import dataclasses
    tampered = dataclasses.replace(c, checksum="aaaaaaaaaaaaaaaa")
    assert tampered.verify_integrity() is False


# ── Serialization ─────────────────────────────────────────────────────────────

def test_roundtrip_as_dict():
    c = make_checkpoint()
    d = c.as_dict()
    c2 = Checkpoint.from_dict(d)
    assert c2.checkpoint_id == c.checkpoint_id
    assert c2.agent_id == c.agent_id
    assert c2.checksum == c.checksum
    assert c2.execution_state == c.execution_state
    assert c2.agent_context == c.agent_context


def test_from_dict_sets_correct_framework():
    c = make_checkpoint()
    d = c.as_dict()
    c2 = Checkpoint.from_dict(d)
    assert c2.framework == Framework.LANGGRAPH


# ── TTL ───────────────────────────────────────────────────────────────────────

def test_ttl_not_expired():
    c = CheckpointBuilder(agent_id="a", framework=Framework.CUSTOM).ttl(3600).build()
    assert c.is_expired() is False


def test_no_ttl_never_expires():
    c = CheckpointBuilder(agent_id="a", framework=Framework.CUSTOM).build()
    assert c.is_expired() is False


# ── ID format ─────────────────────────────────────────────────────────────────

def test_checkpoint_ids_are_unique():
    ids = {_new_checkpoint_id() for _ in range(100)}
    assert len(ids) == 100


def test_checkpoint_ids_are_time_ordered():
    import time
    id1 = _new_checkpoint_id()
    time.sleep(0.01)
    id2 = _new_checkpoint_id()
    # Time-ordered UUIDs: later ID should sort after earlier ID
    assert id1 < id2
