"""
Checkpnt — Framework-agnostic state persistence for AI agents.
Resume any agent from exactly where it stopped.

Quick start:
    from checkpnt import Client, Framework

    async with Client.sqlite() as client:
        # Save
        checkpoint_id = await client.save(
            agent_id="my-agent",
            framework=Framework.LANGGRAPH,
            execution_state=my_graph_state,
        )

        # Restore
        checkpoint = await client.restore(checkpoint_id)
"""

from checkpnt.core import Client
from checkpnt.schemas.checkpoint import Checkpoint, CheckpointBuilder, Framework
from checkpnt.exceptions import (
    CheckpntError,
    CheckpointNotFoundError,
    CheckpointConflictError,
    CheckpointIntegrityError,
    SchemaMigrationError,
    BackendConnectionError,
    AdapterError,
)

__version__ = "0.1.0"
__all__ = [
    "Client",
    "Checkpoint",
    "CheckpointBuilder",
    "Framework",
    "CheckpntError",
    "CheckpointNotFoundError",
    "CheckpointConflictError",
    "CheckpointIntegrityError",
    "SchemaMigrationError",
    "BackendConnectionError",
    "AdapterError",
]
