"""
checkpnt.adapters.base
-----------------------
Abstract adapter interface. Every framework adapter implements exactly two methods:
  extract()     — framework state → Checkpoint
  reconstruct() — Checkpoint → framework state

Adapters know nothing about backends. They only translate between formats.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from checkpnt.schemas.checkpoint import Checkpoint


class Adapter(ABC):

    @abstractmethod
    def extract(
        self,
        framework_state: Any,
        agent_id: str,
        session_id: str | None = None,
        parent_id: str | None = None,
        step_index: int = 0,
        step_name: str | None = None,
        context: dict | None = None,
        ttl_seconds: int | None = None,
        **metadata,
    ) -> "Checkpoint":
        """
        Translate framework-native state into a Checkpoint.

        Args:
            framework_state: Whatever the framework gives you at a save point.
            agent_id:        Stable agent identifier.
            session_id:      Execution session. Auto-generated if None.
            parent_id:       Previous checkpoint in this chain.
            step_index:      Position in execution sequence.
            step_name:       Human-readable label for this step.
            context:         Framework-agnostic portable context (travels cross-framework).
            ttl_seconds:     Auto-expire after N seconds.
            **metadata:      Arbitrary tags.

        Returns:
            An immutable Checkpoint ready to be saved to any backend.
        """

    @abstractmethod
    def reconstruct(self, checkpoint: "Checkpoint") -> Any:
        """
        Translate a Checkpoint back into framework-native state.

        Args:
            checkpoint: A Checkpoint previously created by extract().

        Returns:
            Framework-native state that the agent can resume from.
        """
