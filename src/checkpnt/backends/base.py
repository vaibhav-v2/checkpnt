"""
checkpnt.backends.base
-----------------------
Abstract backend interface. All backends implement exactly these four operations.
Backends know nothing about agents, frameworks, or the Client.
They store and retrieve Checkpoints.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from checkpnt.schemas.checkpoint import Checkpoint


class Backend(ABC):

    @abstractmethod
    async def save(self, checkpoint: "Checkpoint") -> str:
        """
        Persist a checkpoint. Returns checkpoint_id.
        Raises CheckpointConflictError if checkpoint_id already exists.
        Checkpoints are immutable — overwrite is not permitted.
        """

    @abstractmethod
    async def load(self, checkpoint_id: str) -> "Checkpoint | None":
        """Load exact checkpoint by ID. Returns None if not found."""

    @abstractmethod
    async def latest(self, agent_id: str, session_id: str) -> "Checkpoint | None":
        """Load most recent checkpoint for a session. Returns None if session is empty."""

    @abstractmethod
    async def history(
        self,
        agent_id: str,
        session_id: str,
        limit: int = 50,
    ) -> list["Checkpoint"]:
        """
        Load checkpoint chain in reverse-chronological order (newest first).
        Walks parent_id links to reconstruct the execution tree.
        """

    @abstractmethod
    async def delete(self, checkpoint_id: str) -> bool:
        """
        Hard delete. Used by TTL expiry only.
        Returns True if deleted, False if not found.
        """

    async def close(self) -> None:
        """Release resources. Override if backend holds connections."""
        pass
