"""
checkpnt.backends.sqlite
-------------------------
SQLite backend for local development.

Design decisions:
- aiosqlite for non-blocking async I/O (agents are async by nature)
- msgpack encoding — faster and smaller than JSON, handles bytes natively
- WAL mode — allows concurrent readers without blocking writes
- Two tables: checkpoints (immutable records) + checkpoint_index (fast lookups)
- Insert-only: no UPDATE ever. Immutability is enforced at the DB layer.
- TTL expiry: checked on load, cleaned up lazily on history queries
"""

from __future__ import annotations

import json
import msgpack
from datetime import datetime, timezone
from typing import TYPE_CHECKING

try:
    import aiosqlite
except ImportError:
    raise ImportError(
        "aiosqlite is required for the SQLite backend. "
        "Install it with: pip install aiosqlite"
    )

from checkpnt.backends.base import Backend
from checkpnt.exceptions import CheckpointConflictError

if TYPE_CHECKING:
    from checkpnt.schemas.checkpoint import Checkpoint


# SQL schema — created on first connection
_SCHEMA = """
CREATE TABLE IF NOT EXISTS checkpoints (
    checkpoint_id       TEXT PRIMARY KEY,
    agent_id            TEXT NOT NULL,
    session_id          TEXT NOT NULL,
    parent_id           TEXT,
    framework           TEXT NOT NULL,
    schema_version      TEXT NOT NULL,
    step_index          INTEGER NOT NULL,
    step_name           TEXT,
    payload             BLOB NOT NULL,
    checksum            TEXT NOT NULL,
    created_at_ms       INTEGER NOT NULL,
    expires_at_ms       INTEGER,
    handoff_target      TEXT,
    handoff_schema      TEXT
);

CREATE INDEX IF NOT EXISTS idx_checkpoints_session
    ON checkpoints (agent_id, session_id, created_at_ms DESC);

CREATE INDEX IF NOT EXISTS idx_checkpoints_handoff
    ON checkpoints (handoff_target)
    WHERE handoff_target IS NOT NULL;
"""


class SQLiteBackend(Backend):

    def __init__(self, path: str = "./checkpnt_local.db"):
        self._path = path
        self._db: aiosqlite.Connection | None = None

    async def _conn(self) -> aiosqlite.Connection:
        if self._db is None:
            self._db = await aiosqlite.connect(self._path)
            self._db.row_factory = aiosqlite.Row
            # WAL mode: concurrent reads don't block writes
            await self._db.execute("PRAGMA journal_mode=WAL")
            await self._db.execute("PRAGMA foreign_keys=ON")
            await self._db.executescript(_SCHEMA)
            await self._db.commit()
        return self._db

    # ── Backend interface ────────────────────────────────────────────────────

    async def save(self, checkpoint: "Checkpoint") -> str:
        db = await self._conn()

        # Enforce immutability — no overwrite permitted
        async with db.execute(
            "SELECT 1 FROM checkpoints WHERE checkpoint_id = ?",
            (checkpoint.checkpoint_id,)
        ) as cur:
            if await cur.fetchone():
                raise CheckpointConflictError(checkpoint.checkpoint_id)

        payload = _encode(checkpoint)
        created_ms = int(checkpoint.created_at.timestamp() * 1000)
        expires_ms = (
            created_ms + (checkpoint.ttl_seconds * 1000)
            if checkpoint.ttl_seconds else None
        )

        await db.execute(
            """
            INSERT INTO checkpoints (
                checkpoint_id, agent_id, session_id, parent_id,
                framework, schema_version, step_index, step_name,
                payload, checksum, created_at_ms, expires_at_ms,
                handoff_target, handoff_schema
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                checkpoint.checkpoint_id,
                checkpoint.agent_id,
                checkpoint.session_id,
                checkpoint.parent_id,
                checkpoint.framework.value,
                checkpoint.schema_version,
                checkpoint.step_index,
                checkpoint.step_name,
                payload,
                checkpoint.checksum,
                created_ms,
                expires_ms,
                checkpoint.handoff_target,
                checkpoint.handoff_schema,
            )
        )
        await db.commit()
        return checkpoint.checkpoint_id

    async def load(self, checkpoint_id: str) -> "Checkpoint | None":
        db = await self._conn()
        async with db.execute(
            "SELECT payload, expires_at_ms FROM checkpoints WHERE checkpoint_id = ?",
            (checkpoint_id,)
        ) as cur:
            row = await cur.fetchone()

        if row is None:
            return None

        # Respect TTL
        if row["expires_at_ms"] is not None:
            now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
            if now_ms > row["expires_at_ms"]:
                await self.delete(checkpoint_id)
                return None

        return _decode(row["payload"])

    async def latest(self, agent_id: str, session_id: str) -> "Checkpoint | None":
        db = await self._conn()
        now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)

        async with db.execute(
            """
            SELECT payload FROM checkpoints
            WHERE agent_id = ? AND session_id = ?
              AND (expires_at_ms IS NULL OR expires_at_ms > ?)
            ORDER BY created_at_ms DESC, step_index DESC
            LIMIT 1
            """,
            (agent_id, session_id, now_ms)
        ) as cur:
            row = await cur.fetchone()

        return _decode(row["payload"]) if row else None

    async def history(
        self,
        agent_id: str,
        session_id: str,
        limit: int = 50,
    ) -> list["Checkpoint"]:
        db = await self._conn()
        now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)

        async with db.execute(
            """
            SELECT payload FROM checkpoints
            WHERE agent_id = ? AND session_id = ?
              AND (expires_at_ms IS NULL OR expires_at_ms > ?)
            ORDER BY step_index DESC, created_at_ms DESC
            LIMIT ?
            """,
            (agent_id, session_id, now_ms, limit)
        ) as cur:
            rows = await cur.fetchall()

        return [_decode(row["payload"]) for row in rows]

    async def delete(self, checkpoint_id: str) -> bool:
        db = await self._conn()
        async with db.execute(
            "DELETE FROM checkpoints WHERE checkpoint_id = ?",
            (checkpoint_id,)
        ) as cur:
            deleted = cur.rowcount > 0
        await db.commit()
        return deleted

    async def close(self) -> None:
        if self._db is not None:
            await self._db.close()
            self._db = None

    # ── Context manager support ──────────────────────────────────────────────

    async def __aenter__(self) -> "SQLiteBackend":
        await self._conn()
        return self

    async def __aexit__(self, *args) -> None:
        await self.close()


# ── Encoding / Decoding ──────────────────────────────────────────────────────

def _encode(checkpoint: "Checkpoint") -> bytes:
    """Serialize Checkpoint to msgpack bytes for storage."""
    return msgpack.packb(checkpoint.as_dict(), default=_msgpack_default, use_bin_type=True)


def _decode(payload: bytes) -> "Checkpoint":
    """Deserialize msgpack bytes back to a Checkpoint."""
    from checkpnt.schemas.checkpoint import Checkpoint
    data = msgpack.unpackb(payload, raw=False, timestamp=3)
    return Checkpoint.from_dict(data)


def _msgpack_default(obj):
    """Handle types msgpack cannot serialize natively."""
    if isinstance(obj, datetime):
        return obj.isoformat()
    raise TypeError(f"Unknown type for msgpack: {type(obj)}")
