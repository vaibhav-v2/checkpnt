"""
checkpnt.core
--------------
The Checkpnt Client. The only public interface developers interact with.
Five operations. That is all.

Usage:
    from checkpnt import Client, Framework

    client = Client.sqlite("./checkpnt_local.db")  # local dev
    # client = Client.redis("redis://localhost:6379")  # production

    # Save state
    checkpoint_id = await client.save(
        agent_id="invoice-processor",
        framework=Framework.LANGGRAPH,
        session_id="run-001",
        execution_state=graph_state,
        context={"invoice_id": "INV-4821", "step": "validate"},
    )

    # Restore state
    checkpoint = await client.restore(checkpoint_id)
    graph_state = checkpoint.execution_state

    # Transfer state to another agent
    await client.handoff(checkpoint_id, target_agent_id="approval-agent")

    # Audit execution history
    history = await client.timeline(agent_id="invoice-processor", session_id="run-001")

    # Clean up
    await client.expire(checkpoint_id)
"""

from __future__ import annotations

from typing import Any, TYPE_CHECKING

from checkpnt.schemas.checkpoint import Checkpoint, CheckpointBuilder, Framework
from checkpnt.exceptions import CheckpointNotFoundError, CheckpointIntegrityError

if TYPE_CHECKING:
    from checkpnt.backends.base import Backend


class Client:
    """
    The Checkpnt public API.

    All methods are async. Checkpnt is designed for async agent frameworks.
    For sync usage, wrap calls in asyncio.run().
    """

    def __init__(self, backend: "Backend"):
        self._backend = backend

    # ── Factory methods ──────────────────────────────────────────────────────

    @classmethod
    def sqlite(cls, path: str = "./checkpnt_local.db") -> "Client":
        """Create a Client backed by SQLite. For local development."""
        from checkpnt.backends.sqlite import SQLiteBackend
        return cls(SQLiteBackend(path))

    @classmethod
    def redis(cls, url: str = "redis://localhost:6379") -> "Client":
        """Create a Client backed by Redis. For production."""
        from checkpnt.backends.redis import RedisBackend
        return cls(RedisBackend(url))

    # ── The Five Operations ──────────────────────────────────────────────────

    async def save(
        self,
        agent_id: str,
        framework: Framework,
        execution_state: dict[str, Any],
        session_id: str | None = None,
        parent_id: str | None = None,
        context: dict[str, Any] | None = None,
        step_index: int = 0,
        step_name: str | None = None,
        ttl_seconds: int | None = None,
        **metadata,
    ) -> str:
        """
        Save agent execution state. Returns checkpoint_id.

        Args:
            agent_id:        Stable identifier for this agent.
            framework:       Which framework this agent runs on.
            execution_state: Framework-specific graph/flow state.
            session_id:      Groups checkpoints within one run. Auto-generated if None.
            parent_id:       Previous checkpoint in this execution chain.
            context:         Framework-agnostic portable context.
            step_index:      Position in execution sequence (0-based).
            step_name:       Human-readable label for this step.
            ttl_seconds:     Auto-expire after N seconds. None = keep forever.
            **metadata:      Arbitrary tags for filtering and annotation.

        Returns:
            checkpoint_id: Use this to restore, handoff, or expire this checkpoint.
        """
        import uuid
        checkpoint = (
            CheckpointBuilder(agent_id=agent_id, framework=framework)
            .session(session_id or str(uuid.uuid4()))
            .step(step_index, name=step_name)
            .execution_state(execution_state)
            .context(context or {})
            .tag(**metadata)
        )

        if parent_id:
            checkpoint = checkpoint.parent(parent_id)

        if ttl_seconds:
            checkpoint = checkpoint.ttl(ttl_seconds)

        built = checkpoint.build()
        return await self._backend.save(built)

    async def restore(self, checkpoint_id: str, verify: bool = True) -> Checkpoint:
        """
        Load and return a checkpoint by ID.

        Args:
            checkpoint_id: ID returned by a previous save() or handoff() call.
            verify:        If True, verify checksum integrity before returning.

        Raises:
            CheckpointNotFoundError:  If the checkpoint does not exist.
            CheckpointIntegrityError: If verify=True and checksum fails.
        """
        checkpoint = await self._backend.load(checkpoint_id)
        if checkpoint is None:
            raise CheckpointNotFoundError(checkpoint_id)

        if verify and not checkpoint.verify_integrity():
            raise CheckpointIntegrityError(checkpoint_id)

        return checkpoint

    async def handoff(
        self,
        checkpoint_id: str,
        target_agent_id: str,
        schema: str | None = None,
    ) -> str:
        """
        Transfer execution state to another agent.
        Creates a new checkpoint tagged for the target agent.

        Args:
            checkpoint_id:   Source checkpoint to transfer.
            target_agent_id: agent_id of the receiving agent.
            schema:          Schema version the target expects. Defaults to current.

        Returns:
            New checkpoint_id that the target agent should restore from.
        """
        source = await self.restore(checkpoint_id)
        new_checkpoint = (
            CheckpointBuilder(agent_id=source.agent_id, framework=source.framework)
            .session(source.session_id)
            .parent(checkpoint_id)
            .step(source.step_index + 1, name=f"handoff_to_{target_agent_id}")
            .execution_state(source.execution_state)
            .context(source.agent_context)
            .handoff(target_agent_id, schema=schema)
            .build()
        )
        return await self._backend.save(new_checkpoint)

    async def timeline(
        self,
        agent_id: str,
        session_id: str,
        limit: int = 50,
    ) -> list[Checkpoint]:
        """
        Return execution history for a session in reverse-chronological order.
        The full execution tree is reconstructable from checkpoint parent_id links.

        Args:
            agent_id:   Agent to query.
            session_id: Session to query.
            limit:      Maximum checkpoints to return.

        Returns:
            List of Checkpoints, newest first.
        """
        return await self._backend.history(agent_id, session_id, limit=limit)

    async def expire(self, checkpoint_id: str) -> bool:
        """
        Delete a checkpoint immediately.
        Checkpoints also expire automatically if ttl_seconds was set.

        Returns:
            True if deleted, False if checkpoint was not found.
        """
        return await self._backend.delete(checkpoint_id)

    # ── Lifecycle ────────────────────────────────────────────────────────────

    async def close(self) -> None:
        """Release backend resources."""
        await self._backend.close()

    async def __aenter__(self) -> "Client":
        return self

    async def __aexit__(self, *args) -> None:
        await self.close()
