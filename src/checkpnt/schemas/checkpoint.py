"""
checkpnt.schemas.checkpoint
----------------------------
The Checkpoint is the atomic unit of Checkpnt.
It is immutable once created. It is versioned from birth.
It carries everything needed to reconstruct or resume agent execution.
"""

from __future__ import annotations

import hashlib
import json
import uuid
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Any


class Framework(str, Enum):
    LANGGRAPH = "langgraph"
    CREWAI = "crewai"
    AUTOGEN = "autogen"
    CUSTOM = "custom"


CURRENT_SCHEMA_VERSION = "1.0"


@dataclass(frozen=True)
class Checkpoint:
    """
    An immutable snapshot of agent execution state at a moment in time.

    Design decisions:
    - frozen=True: checkpoints cannot be modified after creation
    - parent_id: forms an append-only execution tree (enables time travel, Layer 3)
    - schema_version: every checkpoint knows its own format (enables migration)
    - handoff_target: coordination primitive for multi-agent systems (Layer 5)
    - checksum: integrity verification and state diffing (Layer 4)
    - agent_context vs execution_state: developer-owned dict vs framework-specific state
    """

    # Identity
    checkpoint_id: str
    agent_id: str
    session_id: str
    parent_id: str | None  # None = root of new session

    # Execution Position
    framework: Framework
    schema_version: str
    step_index: int
    step_name: str | None

    # State Payload
    execution_state: dict[str, Any]   # Framework-specific internal state (LangGraph channel_values etc).
    agent_context: dict[str, Any]     # Developer-owned unstructured dict. No schema enforced.
    metadata: dict[str, Any]

    # Provenance
    created_at: datetime
    ttl_seconds: int | None
    checksum: str

    # Coordination — populated in Layer 5, reserved from day one
    handoff_target: str | None
    handoff_schema: str | None

    def is_expired(self) -> bool:
        if self.ttl_seconds is None:
            return False
        age = (datetime.now(timezone.utc) - self.created_at).total_seconds()
        return age > self.ttl_seconds

    def verify_integrity(self) -> bool:
        expected = _compute_checksum(self.execution_state, self.agent_context)
        return self.checksum == expected

    def as_dict(self) -> dict[str, Any]:
        return {
            "checkpoint_id": self.checkpoint_id,
            "agent_id": self.agent_id,
            "session_id": self.session_id,
            "parent_id": self.parent_id,
            "framework": self.framework.value,
            "schema_version": self.schema_version,
            "step_index": self.step_index,
            "step_name": self.step_name,
            "execution_state": self.execution_state,
            "agent_context": self.agent_context,
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat(),
            "ttl_seconds": self.ttl_seconds,
            "checksum": self.checksum,
            "handoff_target": self.handoff_target,
            "handoff_schema": self.handoff_schema,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Checkpoint":
        from checkpnt.schemas.versions import migrate
        data = migrate(data, from_version=data.get("schema_version", "1.0"))
        return cls(
            checkpoint_id=data["checkpoint_id"],
            agent_id=data["agent_id"],
            session_id=data["session_id"],
            parent_id=data.get("parent_id"),
            framework=Framework(data["framework"]),
            schema_version=data["schema_version"],
            step_index=data["step_index"],
            step_name=data.get("step_name"),
            execution_state=data["execution_state"],
            agent_context=data["agent_context"],
            metadata=data.get("metadata", {}),
            created_at=datetime.fromisoformat(data["created_at"]),
            ttl_seconds=data.get("ttl_seconds"),
            checksum=data["checksum"],
            handoff_target=data.get("handoff_target"),
            handoff_schema=data.get("handoff_schema"),
        )


class CheckpointBuilder:
    """
    Fluent builder for creating Checkpoints.

    Usage:
        checkpoint = (
            CheckpointBuilder(agent_id="invoice-agent", framework=Framework.LANGGRAPH)
            .session("run-2026-03-06")
            .parent("prev-checkpoint-id")
            .step(7, name="validate_line_items")
            .execution_state({"messages": [...], "next": ["tool_node"]})
            .context({"invoice_total": 42000, "line_items": [...]})
            .ttl(3600)
            .build()
        )
    """

    def __init__(self, agent_id: str, framework: Framework):
        self._agent_id = agent_id
        self._framework = framework
        self._session_id: str = str(uuid.uuid4())
        self._parent_id: str | None = None
        self._step_index: int = 0
        self._step_name: str | None = None
        self._execution_state: dict = {}
        self._agent_context: dict = {}
        self._metadata: dict = {}
        self._ttl_seconds: int | None = None
        self._handoff_target: str | None = None
        self._handoff_schema: str | None = None

    def session(self, session_id: str) -> "CheckpointBuilder":
        self._session_id = session_id
        return self

    def parent(self, parent_id: str) -> "CheckpointBuilder":
        self._parent_id = parent_id
        return self

    def step(self, index: int, name: str | None = None) -> "CheckpointBuilder":
        self._step_index = index
        self._step_name = name
        return self

    def execution_state(self, state: dict) -> "CheckpointBuilder":
        self._execution_state = state
        return self

    def context(self, context: dict) -> "CheckpointBuilder":
        self._agent_context = context
        return self

    def tag(self, **kwargs) -> "CheckpointBuilder":
        self._metadata.update(kwargs)
        return self

    def ttl(self, seconds: int) -> "CheckpointBuilder":
        self._ttl_seconds = seconds
        return self

    def handoff(self, target_agent_id: str, schema: str | None = None) -> "CheckpointBuilder":
        self._handoff_target = target_agent_id
        self._handoff_schema = schema or CURRENT_SCHEMA_VERSION
        return self

    def build(self) -> Checkpoint:
        checksum = _compute_checksum(self._execution_state, self._agent_context)
        return Checkpoint(
            checkpoint_id=_new_checkpoint_id(),
            agent_id=self._agent_id,
            session_id=self._session_id,
            parent_id=self._parent_id,
            framework=self._framework,
            schema_version=CURRENT_SCHEMA_VERSION,
            step_index=self._step_index,
            step_name=self._step_name,
            execution_state=self._execution_state,
            agent_context=self._agent_context,
            metadata=self._metadata,
            created_at=datetime.now(timezone.utc),
            ttl_seconds=self._ttl_seconds,
            checksum=checksum,
            handoff_target=self._handoff_target,
            handoff_schema=self._handoff_schema,
        )


def _new_checkpoint_id() -> str:
    """
    Time-ordered UUID (UUID v7 pattern).
    Enables efficient range queries on checkpoint history without secondary indexes.
    Replace with uuid.uuid7() when Python 3.13 is baseline.
    """
    timestamp_ms = int(time.time() * 1000)
    time_hex = format(timestamp_ms, '012x')
    random_part = uuid.uuid4().hex[12:]
    raw = time_hex + random_part
    return f"{raw[0:8]}-{raw[8:12]}-7{raw[13:16]}-{raw[16:20]}-{raw[20:32]}"


def _compute_checksum(execution_state: dict, agent_context: dict) -> str:
    """SHA-256 of canonical JSON. Deterministic regardless of dict insertion order."""
    canonical = json.dumps(
        {"execution_state": execution_state, "agent_context": agent_context},
        sort_keys=True,
        default=str,
    )
    return hashlib.sha256(canonical.encode()).hexdigest()
