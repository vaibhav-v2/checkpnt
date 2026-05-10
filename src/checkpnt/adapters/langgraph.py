"""
checkpnt.adapters.langgraph
----------------------------
Adapter between LangGraph's StateSnapshot and Checkpnt's Checkpoint schema.
Also provides CheckpntSaver — a proper BaseCheckpointSaver subclass that
is a drop-in replacement for LangGraph's SqliteSaver / MemorySaver.

This is the direct answer to LangGraph Issue #5790:
https://github.com/langchain-ai/langgraph/issues/5790

LangGraph's `langgraph dev` intentionally replaces any configured checkpointer
with in-memory storage. This is deliberate platform lock-in. CheckpntSaver
bypasses that entirely — it is a proper BaseCheckpointSaver that LangGraph
accepts, backed by any Checkpnt backend.
"""

from __future__ import annotations

import asyncio
import threading
import uuid
from typing import Any, AsyncIterator, Iterator, Sequence, TYPE_CHECKING

from checkpnt.adapters.base import Adapter
from checkpnt.schemas.checkpoint import Checkpoint, CheckpointBuilder, Framework
from checkpnt.exceptions import AdapterError

if TYPE_CHECKING:
    from checkpnt.core import Client


# ── LangGraphAdapter ─────────────────────────────────────────────────────────

class LangGraphAdapter(Adapter):
    """
    Translates between LangGraph StateSnapshot and Checkpnt Checkpoint.
    """

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
    ) -> Checkpoint:
        try:
            execution_state = self.extract_state(framework_state)
        except Exception as e:
            raise AdapterError(
                f"Failed to extract LangGraph state: {e}. "
                "Ensure you are passing a StateSnapshot from app.get_state(config)."
            ) from e

        if isinstance(framework_state, dict):
            lg_meta = framework_state.get("metadata", {}) or {}
        else:
            lg_meta = getattr(framework_state, "metadata", {}) or {}

        inferred_step = lg_meta.get("step", step_index)
        inferred_name = step_name or _infer_step_name(framework_state)
        inferred_session = session_id or _infer_session(framework_state)

        builder = (
            CheckpointBuilder(agent_id=agent_id, framework=Framework.LANGGRAPH)
            .session(inferred_session)
            .step(inferred_step, name=inferred_name)
            .execution_state(execution_state)
            .context(context or {})
            .tag(**metadata)
        )

        if parent_id:
            builder = builder.parent(parent_id)
        if ttl_seconds:
            builder = builder.ttl(ttl_seconds)

        return builder.build()

    def reconstruct(self, checkpoint: Checkpoint) -> dict[str, Any]:
        if checkpoint.framework != Framework.LANGGRAPH:
            raise AdapterError(
                f"Cannot reconstruct LangGraph state from a "
                f"{checkpoint.framework.value} checkpoint."
            )
        state = checkpoint.execution_state
        return {
            "values": state.get("values", {}),
            "next": state.get("next", []),
            "config": state.get("config", {}),
            "metadata": state.get("metadata", {}),
            "thread_id": state.get("thread_id"),
            "checkpoint_id": checkpoint.checkpoint_id,
            "agent_context": checkpoint.agent_context,
        }

    def extract_state(self, snapshot: Any) -> dict[str, Any]:
        if isinstance(snapshot, dict):
            return {
                "values": snapshot.get("values", {}),
                "next": list(snapshot.get("next", []) or []),
                "config": snapshot.get("config", {}),
                "metadata": snapshot.get("metadata", {}),
                "thread_id": _extract_thread_id(snapshot),
                "parent_config": snapshot.get("parent_config"),
            }
        return {
            "values": dict(getattr(snapshot, "values", {}) or {}),
            "next": list(getattr(snapshot, "next", []) or []),
            "config": _serialise_config(getattr(snapshot, "config", {}) or {}),
            "metadata": dict(getattr(snapshot, "metadata", {}) or {}),
            "thread_id": _extract_thread_id(snapshot),
            "parent_config": _serialise_config(getattr(snapshot, "parent_config", None)),
        }


# ── CheckpntSaver — proper BaseCheckpointSaver subclass ─────────────────────

def _get_base_class():
    """
    Lazily import BaseCheckpointSaver so langgraph is an optional dependency.
    Returns BaseCheckpointSaver if available, otherwise a no-op base.
    """
    try:
        from langgraph.checkpoint.base import BaseCheckpointSaver
        return BaseCheckpointSaver
    except ImportError:
        raise ImportError(
            "langgraph is required for CheckpntSaver. "
            "Install it with: pip install langgraph"
        )


class CheckpntSaver:
    """
    Drop-in replacement for LangGraph's SqliteSaver / MemorySaver.

    Inherits from BaseCheckpointSaver at instantiation time so langgraph
    is an optional dependency. The class is constructed dynamically to
    avoid import-time failures when langgraph is not installed.

    Usage:
        async with CheckpntSaver.from_sqlite("./checkpnt_local.db") as saver:
            app = graph.compile(checkpointer=saver)
            result = app.invoke(input, config=config)

    This fixes LangGraph Issue #5790 — your configured checkpointer is no longer
    silently replaced by langgraph dev's in-memory store.
    """

    def __new__(cls, client: "Client"):
        """
        Dynamically create a class that inherits from BaseCheckpointSaver.
        This defers the langgraph import until CheckpntSaver is actually instantiated.
        """
        Base = _get_base_class()

        class _CheckpntSaverImpl(Base):

            def __init__(self, client: "Client"):
                super().__init__()
                self._client = client
                self._thread_latest: dict[str, str] = {}
                # Dedicated event loop in a background daemon thread.
                # Bridges sync callers (LangGraph's BackgroundExecutor /
                # ThreadPoolExecutor) with our async backends safely.
                # Works from any context — no-loop threads, running-loop
                # threads (FastAPI/Uvicorn), or the main thread.
                self._bg_loop = asyncio.new_event_loop()
                self._bg_thread = threading.Thread(
                    target=self._bg_loop.run_forever,
                    daemon=True,
                    name="checkpnt-bg-loop",
                )
                self._bg_thread.start()

            @classmethod
            def from_sqlite(cls, path: str = "./checkpnt_local.db") -> "_CheckpntSaverImpl":
                from checkpnt.core import Client as CheckpntClient
                return cls(CheckpntClient.sqlite(path))

            @classmethod
            def from_redis(cls, url: str = "redis://localhost:6379") -> "_CheckpntSaverImpl":
                from checkpnt.core import Client as CheckpntClient
                return cls(CheckpntClient.redis(url))

            def get_next_version(self, current, channel=None):
                if current is None:
                    return 1
                if isinstance(current, int):
                    return current + 1
                return current

            # ── Core sync interface (LangGraph calls these) ──────────────────

            def _run_sync(self, coro):
                """
                Submit a coroutine to the background event loop and block
                until it completes. Safe to call from any context:
                  - threads with no event loop (LangGraph's ThreadPoolExecutor)
                  - threads with a running event loop (FastAPI / Uvicorn)
                  - the main thread
                Never creates a new event loop. Never calls asyncio.run().
                """
                future = asyncio.run_coroutine_threadsafe(coro, self._bg_loop)
                return future.result()

            def put(self, config, checkpoint, metadata, new_versions):
                return self._run_sync(
                    self._aput_impl(config, checkpoint, metadata, new_versions)
                )

            def get_tuple(self, config):
                return self._run_sync(self._aget_tuple_impl(config))

            def list(self, config, *, filter=None, before=None, limit=None):
                return iter(self._run_sync(self._alist_impl(config, limit=limit)))

            def put_writes(self, config, writes, task_id, task_path=""):
                pass  # Intermediate writes — not persisted

            # ── Async interface ──────────────────────────────────────────────

            async def aput(self, config, checkpoint, metadata, new_versions):
                return await self._aput_impl(config, checkpoint, metadata, new_versions)

            async def aget_tuple(self, config):
                return await self._aget_tuple_impl(config)

            async def alist(self, config, *, filter=None, before=None, limit=None):
                results = await self._alist_impl(config, limit=limit)
                for r in results:
                    yield r

            async def aput_writes(self, config, writes, task_id, task_path=""):
                pass

            # ── Implementation ───────────────────────────────────────────────

            async def _aput_impl(self, config, checkpoint, metadata, new_versions):
                thread_id = config["configurable"]["thread_id"]
                checkpoint_ns = config["configurable"].get("checkpoint_ns", "")

                agent_id = f"checkpnt:{thread_id}"
                session_id = f"{thread_id}:{checkpoint_ns}" if checkpoint_ns else thread_id

                parent_id = self._thread_latest.get(session_id)

                # Use LangGraph's own serde to serialize checkpoint + metadata.
                # This correctly handles HumanMessage, AIMessage, and all
                # LangChain types that plain msgpack/json cannot serialize.
                cp_type, cp_bytes = self.serde.dumps_typed(checkpoint)
                meta_type, meta_bytes = self.serde.dumps_typed(metadata)

                execution_state = {
                    "lg_cp_type": cp_type,
                    "lg_cp_bytes": cp_bytes.hex(),   # bytes → hex string for storage
                    "lg_meta_type": meta_type,
                    "lg_meta_bytes": meta_bytes.hex(),
                    "thread_id": thread_id,
                    "checkpoint_ns": checkpoint_ns,
                    "lg_checkpoint_id": checkpoint.get("id", ""),
                }

                step = metadata.get("step", 0) if isinstance(metadata, dict) else 0
                source = metadata.get("source", "loop") if isinstance(metadata, dict) else "loop"

                builder = (
                    CheckpointBuilder(agent_id=agent_id, framework=Framework.LANGGRAPH)
                    .session(session_id)
                    .step(step, name=source)
                    .execution_state(execution_state)
                    .context({})
                )
                if parent_id:
                    builder = builder.parent(parent_id)

                cp_obj = builder.build()
                saved_id = await self._client._backend.save(cp_obj)
                self._thread_latest[session_id] = saved_id

                return {
                    "configurable": {
                        "thread_id": thread_id,
                        "checkpoint_ns": checkpoint_ns,
                        "checkpoint_id": checkpoint.get("id", saved_id),
                    }
                }

            async def _aget_tuple_impl(self, config):
                from langgraph.checkpoint.base import CheckpointTuple

                thread_id = config["configurable"]["thread_id"]
                checkpoint_ns = config["configurable"].get("checkpoint_ns", "")
                requested_lg_id = config["configurable"].get("checkpoint_id")

                agent_id = f"checkpnt:{thread_id}"
                session_id = f"{thread_id}:{checkpoint_ns}" if checkpoint_ns else thread_id

                if requested_lg_id:
                    # Find checkpoint with matching lg_checkpoint_id
                    history = await self._client._backend.history(agent_id, session_id, limit=100)
                    cp = next(
                        (c for c in history
                         if c.execution_state.get("lg_checkpoint_id") == requested_lg_id),
                        None
                    )
                else:
                    cp = await self._client._backend.latest(agent_id, session_id)

                if cp is None:
                    return None

                return self._to_checkpoint_tuple(cp, config)

            async def _alist_impl(self, config, limit=None):
                thread_id = config["configurable"]["thread_id"]
                checkpoint_ns = config["configurable"].get("checkpoint_ns", "")
                agent_id = f"checkpnt:{thread_id}"
                session_id = f"{thread_id}:{checkpoint_ns}" if checkpoint_ns else thread_id

                history = await self._client._backend.history(
                    agent_id, session_id, limit=limit or 50
                )
                return [self._to_checkpoint_tuple(cp, config) for cp in history]

            def _to_checkpoint_tuple(self, cp: Checkpoint, config: dict):
                from langgraph.checkpoint.base import CheckpointTuple

                state = cp.execution_state
                thread_id = state.get("thread_id", config["configurable"]["thread_id"])
                checkpoint_ns = state.get("checkpoint_ns", "")
                lg_checkpoint_id = state.get("lg_checkpoint_id", cp.checkpoint_id)

                # Deserialise using LangGraph's own serde — restores LangChain
                # message objects (HumanMessage, AIMessage, etc.) correctly.
                if "lg_cp_bytes" in state:
                    cp_type = state["lg_cp_type"]
                    cp_bytes = bytes.fromhex(state["lg_cp_bytes"])
                    lg_checkpoint = self.serde.loads_typed((cp_type, cp_bytes))
                    
                    meta_type = state["lg_meta_type"]
                    meta_bytes = bytes.fromhex(state["lg_meta_bytes"])
                    lg_metadata = self.serde.loads_typed((meta_type, meta_bytes))
                else:
                    # Legacy format fallback
                    lg_checkpoint = {
                        "v": 1,
                        "id": lg_checkpoint_id,
                        "ts": cp.created_at.isoformat(),
                        "channel_values": {},
                        "channel_versions": {},
                        "versions_seen": {},
                        "pending_sends": [],
                    }
                    lg_metadata = {
                        "step": cp.step_index,
                        "source": cp.step_name or "loop",
                        "parents": {},
                    }

                current_config = {
                    "configurable": {
                        "thread_id": thread_id,
                        "checkpoint_ns": checkpoint_ns,
                        "checkpoint_id": lg_checkpoint_id,
                    }
                }

                parent_config = None
                if cp.parent_id:
                    parent_config = {
                        "configurable": {
                            "thread_id": thread_id,
                            "checkpoint_ns": checkpoint_ns,
                            "checkpoint_id": state.get("lg_checkpoint_id", cp.parent_id),
                        }
                    }

                return CheckpointTuple(
                    config=current_config,
                    checkpoint=lg_checkpoint,
                    metadata=lg_metadata,
                    parent_config=parent_config,
                    pending_writes=[],
                )

            # ── Context manager ──────────────────────────────────────────────

            async def __aenter__(self):
                return self

            async def __aexit__(self, *args):
                # Stop the background loop cleanly before closing the client
                self._bg_loop.call_soon_threadsafe(self._bg_loop.stop)
                self._bg_thread.join(timeout=5)
                await self._client.close()

        # Create an instance of the dynamically-built class
        instance = object.__new__(_CheckpntSaverImpl)
        _CheckpntSaverImpl.__init__(instance, client)
        return instance

    @classmethod
    def from_sqlite(cls, path: str = "./checkpnt_local.db") -> "CheckpntSaver":
        from checkpnt.core import Client as CheckpntClient
        return cls(CheckpntClient.sqlite(path))

    @classmethod
    def from_redis(cls, url: str = "redis://localhost:6379") -> "CheckpntSaver":
        from checkpnt.core import Client as CheckpntClient
        return cls(CheckpntClient.redis(url))


# ── Helpers ───────────────────────────────────────────────────────────────────

def _safe_dict(obj: Any) -> dict:
    """Convert any object to a JSON-safe dict."""
    if obj is None:
        return {}
    if isinstance(obj, dict):
        result = {}
        for k, v in obj.items():
            try:
                import json
                json.dumps(v, default=str)
                result[k] = v
            except Exception:
                result[k] = str(v)
        return result
    try:
        return dict(obj)
    except Exception:
        return {"_raw": str(obj)}


def _infer_step_name(snapshot: Any) -> str | None:
    if isinstance(snapshot, dict):
        next_nodes = snapshot.get("next", None)
        meta = snapshot.get("metadata", {}) or {}
    else:
        next_nodes = getattr(snapshot, "next", None)
        meta = getattr(snapshot, "metadata", {}) or {}

    if next_nodes:
        return f"before:{','.join(str(n) for n in next_nodes)}"
    source = meta.get("source")
    if source:
        return source
    return None


def _infer_session(snapshot: Any) -> str:
    thread_id = _extract_thread_id(snapshot)
    return thread_id or str(uuid.uuid4())


def _extract_thread_id(snapshot: Any) -> str | None:
    if isinstance(snapshot, dict):
        config = snapshot.get("config", {}) or {}
    else:
        config = getattr(snapshot, "config", {}) or {}

    if isinstance(config, dict):
        return config.get("configurable", {}).get("thread_id")
    configurable = getattr(config, "configurable", {}) or {}
    return configurable.get("thread_id")


def _serialise_config(config: Any) -> dict | None:
    if config is None:
        return None
    if isinstance(config, dict):
        return config
    return {
        "tags": getattr(config, "tags", []),
        "metadata": getattr(config, "metadata", {}),
        "callbacks": None,
        "recursion_limit": getattr(config, "recursion_limit", 25),
        "configurable": dict(getattr(config, "configurable", {}) or {}),
    }
