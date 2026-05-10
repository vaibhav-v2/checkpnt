"""
checkpnt.exceptions
--------------------
All Checkpnt exceptions. Flat hierarchy — no deep inheritance chains.
"""


class CheckpntError(Exception):
    """Base class for all Checkpnt errors."""


class CheckpointNotFoundError(CheckpntError):
    """Raised when a checkpoint_id does not exist in the backend."""
    def __init__(self, checkpoint_id: str):
        self.checkpoint_id = checkpoint_id
        super().__init__(f"Checkpoint not found: {checkpoint_id}")


class CheckpointConflictError(CheckpntError):
    """Raised when attempting to overwrite an existing checkpoint (immutability violation)."""
    def __init__(self, checkpoint_id: str):
        self.checkpoint_id = checkpoint_id
        super().__init__(
            f"Checkpoint '{checkpoint_id}' already exists. "
            "Checkpoints are immutable — create a new one with a new ID."
        )


class CheckpointIntegrityError(CheckpntError):
    """Raised when a checkpoint's checksum does not match its payload."""
    def __init__(self, checkpoint_id: str):
        self.checkpoint_id = checkpoint_id
        super().__init__(
            f"Integrity check failed for checkpoint '{checkpoint_id}'. "
            "The stored payload may have been tampered with."
        )


class SchemaMigrationError(CheckpntError):
    """Raised when no migration path exists from an old schema version."""


class BackendConnectionError(CheckpntError):
    """Raised when the backend cannot be reached."""


class AdapterError(CheckpntError):
    """Raised when a framework adapter fails to extract or reconstruct state."""
