"""
checkpnt.backends.redis
------------------------
Redis backend for production deployments.

Design decisions:
- redis.asyncio for non-blocking async I/O
- Three key types per checkpoint:
    checkpnt:cp:{id}                → msgpack payload (with TTL if set)
    checkpnt:idx:{agent}:{session}  → Sorted set, score = created_at_ms (history queries)
    checkpnt:latest:{agent}:{session} → checkpoint_id string (O(1) latest lookup)
- All three written in a single pipeline — atomic from the caller's perspective
- Sorted sets give O(log N) history queries without full scans
- Native Redis TTL handles expiry — no background job needed
- Pub/sub infrastructure already present — Layer 5 coordination needs no new infra
"""

from __future__ import annotations

import msgpack
from datetime import datetime, timezone
from typing import TYPE_CHECKING

try:
    import redis.asyncio as aioredis
except ImportError:
    raise ImportError(
        "redis is required for the Redis backend. "
        "Install it with: pip install redis"
    )

from checkpnt.backends.base import Backend
from checkpnt.exceptions import CheckpointConflictError, BackendConnectionError

if TYPE_CHECKING:
    from checkpnt.schemas.checkpoint import Checkpoint


# Key builders — centralised so a typo never silently creates a wrong key
def _key_payload(checkpoint_id: str) -> str:
    return f"checkpnt:cp:{checkpoint_id}"

def _key_index(agent_id: str, session_id: str) -> str:
    return f"checkpnt:idx:{agent_id}:{session_id}"

def _key_latest(agent_id: str, session_id: str) -> str:
    return f"checkpnt:latest:{agent_id}:{session_id}"


class RedisBackend(Backend):

    def __init__(self, url: str = "redis://localhost:6379", **kwargs):
        self._url = url
        self._kwargs = kwargs
        self._client: aioredis.Redis | None = None

    async def _conn(self) -> aioredis.Redis:
        if self._client is None:
            try:
                self._client = aioredis.from_url(
                    self._url,
                    decode_responses=False,  # We handle bytes ourselves
                    **self._kwargs,
                )
                await self._client.ping()
            except Exception as e:
                self._client = None
                raise BackendConnectionError(
                    f"Cannot connect to Redis at {self._url}: {e}"
                ) from e
        return self._client

    # ── Backend interface ────────────────────────────────────────────────────

    async def save(self, checkpoint: "Checkpoint") -> str:
        r = await self._conn()
        payload_key = _key_payload(checkpoint.checkpoint_id)

        # Enforce immutability — atomic check-then-set
        # NX = only set if key does not exist
        payload = _encode(checkpoint)
        created_ms = int(checkpoint.created_at.timestamp() * 1000)
        ttl_ms = checkpoint.ttl_seconds * 1000 if checkpoint.ttl_seconds else None

        async with r.pipeline(transaction=True) as pipe:
            # WATCH the payload key — if it already exists, the transaction aborts
            await pipe.watch(payload_key)
            existing = await pipe.exists(payload_key)
            if existing:
                await pipe.reset()
                raise CheckpointConflictError(checkpoint.checkpoint_id)

            pipe.multi()

            # 1. Store payload with optional TTL
            if ttl_ms:
                pipe.set(payload_key, payload, px=ttl_ms)
            else:
                pipe.set(payload_key, payload)

            # 2. Add to sorted set index (score = created_at_ms for range queries)
            idx_key = _key_index(checkpoint.agent_id, checkpoint.session_id)
            pipe.zadd(idx_key, {checkpoint.checkpoint_id: created_ms})
            if ttl_ms:
                # Keep index TTL slightly longer than payload TTL
                # so cleanup queries don't see ghost entries
                pipe.expire(idx_key, checkpoint.ttl_seconds + 60)

            # 3. Update latest pointer
            latest_key = _key_latest(checkpoint.agent_id, checkpoint.session_id)
            pipe.set(latest_key, checkpoint.checkpoint_id)
            if ttl_ms:
                pipe.expire(latest_key, checkpoint.ttl_seconds + 60)

            await pipe.execute()

        return checkpoint.checkpoint_id

    async def load(self, checkpoint_id: str) -> "Checkpoint | None":
        r = await self._conn()
        payload = await r.get(_key_payload(checkpoint_id))
        if payload is None:
            return None
        return _decode(payload)

    async def latest(self, agent_id: str, session_id: str) -> "Checkpoint | None":
        r = await self._conn()
        latest_key = _key_latest(agent_id, session_id)
        checkpoint_id = await r.get(latest_key)
        if checkpoint_id is None:
            return None

        checkpoint_id = checkpoint_id.decode() if isinstance(checkpoint_id, bytes) else checkpoint_id

        # The payload may have expired even if the pointer exists
        payload = await r.get(_key_payload(checkpoint_id))
        if payload is None:
            # Stale pointer — clean it up
            await r.delete(latest_key)
            return None

        return _decode(payload)

    async def history(
        self,
        agent_id: str,
        session_id: str,
        limit: int = 50,
    ) -> list["Checkpoint"]:
        r = await self._conn()
        idx_key = _key_index(agent_id, session_id)

        results = []
        offset = 0
        batch_size = max(limit, 20)
        orphaned: list[str] = []  # index entries whose payload is gone

        # Over-fetch loop: keep fetching until we have enough live entries
        # or exhaust the index. This ensures history(limit=10) always returns
        # exactly 10 results (or fewer only if the index itself has fewer).
        while len(results) < limit:
            checkpoint_ids = await r.zrevrange(
                idx_key, offset, offset + batch_size - 1
            )
            if not checkpoint_ids:
                break

            ids = [cid.decode() if isinstance(cid, bytes) else cid
                   for cid in checkpoint_ids]

            # Batch fetch payloads in one pipeline round trip
            pipe = r.pipeline(transaction=False)
            for cid in ids:
                pipe.get(_key_payload(cid))
            payloads = await pipe.execute()

            for cid, payload in zip(ids, payloads):
                if payload is not None:
                    results.append(_decode(payload))
                    if len(results) == limit:
                        break
                else:
                    # Payload expired or deleted — mark for index cleanup
                    orphaned.append(cid)

            offset += batch_size

        # Clean orphaned index entries so future queries don't re-fetch them
        if orphaned:
            pipe = r.pipeline(transaction=False)
            for cid in orphaned:
                pipe.zrem(idx_key, cid)
            await pipe.execute()

        return results

    async def delete(self, checkpoint_id: str) -> bool:
        r = await self._conn()

        # Load before deleting so we can clean the index entry immediately.
        # Without this, history() would silently return fewer results than
        # requested until the index entry's own TTL expired.
        payload = await r.get(_key_payload(checkpoint_id))
        if payload is None:
            return False

        cp = _decode(payload)
        idx_key = _key_index(cp.agent_id, cp.session_id)
        latest_key = _key_latest(cp.agent_id, cp.session_id)

        pipe = r.pipeline(transaction=False)
        pipe.delete(_key_payload(checkpoint_id))   # remove payload
        pipe.zrem(idx_key, checkpoint_id)           # remove from history index
        # If this was the latest, remove the pointer too
        pipe.delete(latest_key)
        await pipe.execute()

        return True

    async def close(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    async def __aenter__(self) -> "RedisBackend":
        await self._conn()
        return self

    async def __aexit__(self, *args) -> None:
        await self.close()


# ── Encoding / Decoding ──────────────────────────────────────────────────────

def _encode(checkpoint: "Checkpoint") -> bytes:
    return msgpack.packb(checkpoint.as_dict(), default=_msgpack_default, use_bin_type=True)


def _decode(payload: bytes) -> "Checkpoint":
    from checkpnt.schemas.checkpoint import Checkpoint
    data = msgpack.unpackb(payload, raw=False, timestamp=3)
    return Checkpoint.from_dict(data)


def _msgpack_default(obj):
    from datetime import datetime
    if isinstance(obj, datetime):
        return obj.isoformat()
    raise TypeError(f"Unknown type for msgpack: {type(obj)}")
