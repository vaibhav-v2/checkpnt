# Checkpnt Architecture

> **Design philosophy:** State is not a feature. State is the coordination protocol of multi-agent AI systems. Every decision in this architecture reflects that belief.

---

## Table of Contents

1. [The Problem This Architecture Solves](#1-the-problem-this-architecture-solves)
2. [Core Abstractions](#2-core-abstractions)
3. [Data Model](#3-data-model)
4. [Layer Architecture](#4-layer-architecture)
5. [Backend Design](#5-backend-design)
6. [Adapter Design](#6-adapter-design)
7. [State Versioning](#7-state-versioning)
8. [Multi-Agent Coordination](#8-multi-agent-coordination)
9. [Immutability Guarantees](#9-immutability-guarantees)
10. [Roadmap Decisions Baked Into v1](#10-roadmap-decisions-baked-into-v1)

---

## 1. The Problem This Architecture Solves

### The Surface Problem (what most people see)

AI agents crash. When they crash, state is lost. Developers restart from zero.

### The Deeper Problem (what this architecture is designed for)

As AI systems mature, agents will not run alone. They will run in graphs — hierarchies of coordinating agents, parallel execution branches, supervisor-worker patterns, handoff chains. In that world:

- **State is not just crash recovery** — it is the contract between agents
- **State is not just data** — it is execution provenance, decision history, coordination context
- **State is not agent-specific** — `execution_state` is framework-specific; `agent_context` is developer-owned

A naive implementation saves a dict to a database. This architecture treats state as a first-class, versioned, immutable, auditable primitive that can outlive any single agent, framework, or deployment.

---

## 2. Core Abstractions

Four abstractions, deliberately minimal:

```
Checkpoint
  │  The atomic unit. An immutable snapshot of agent state at a moment in time.
  │  Contains: what the agent knew, where it was, what it had done.
  │
  ├── Backend
  │     The storage layer. SQLite for local dev, Redis for production.
  │     Swappable without changing agent code.
  │
  ├── Adapter
  │     The framework bridge. Translates LangGraph / CrewAI / AutoGen
  │     execution state into the Checkpoint schema and back.
  │
  └── Client
        The public API. Five operations. That is all.
```

Agents interact only with the Client. Everything else is an implementation detail.

---

## 3. Data Model

### 3.1 Checkpoint Schema (v1)

```python
class Checkpoint:
    # Identity
    checkpoint_id: str          # UUID v7 — time-ordered for efficient range queries
    agent_id: str               # Stable identifier for the agent across runs
    session_id: str             # Groups checkpoints within a single execution run
    parent_id: str | None       # Forms an immutable tree — never updated, only extended

    # Execution Position
    framework: Framework        # langgraph | crewai | autogen | custom
    schema_version: str         # "1.0" — every checkpoint knows its schema version
    step_index: int             # Monotonically increasing within a session
    step_name: str | None       # Human-readable label for this execution point

    # State Payload
    execution_state: dict       # Framework-specific graph/flow state
    agent_context: dict         # What the agent knows (tool results, decisions made)
    metadata: dict              # Caller-supplied tags, annotations, run identifiers

    # Provenance
    created_at: datetime        # UTC — always UTC
    ttl_seconds: int | None     # None = persist forever
    checksum: str               # SHA-256 of the canonical payload — integrity verification

    # Coordination (for multi-agent use — populated in Layer 4+)
    handoff_target: str | None  # agent_id this state is intended for
    handoff_schema: str | None  # Schema version the target agent expects
```

### 3.2 Why UUID v7, not UUID v4

UUID v7 is time-ordered. This matters because:
- `list_checkpoints(agent_id)` is a range scan, not a full table scan
- Checkpoints naturally sort chronologically without a secondary index
- Distributed writes across backend shards sort correctly without coordination

### 3.3 Why `parent_id` forms a tree

Checkpoints are never updated. A new checkpoint is always a child of its predecessor. This gives us:
- **Free time travel**: Walk the parent chain backward to any previous state
- **Branching**: Two parallel sub-agents can each be children of the same parent
- **Auditability**: The full execution tree is reconstructable from the data alone

---

## 4. Layer Architecture

```
┌─────────────────────────────────────────────────────────┐
│  Layer 5 — Coordination Protocol  (Month 4–6)           │
│  Typed state contracts between agents. Pub/sub.          │
├─────────────────────────────────────────────────────────┤
│  Layer 4 — State Diffing          (Month 3)             │
│  Compare execution runs. Why did run A succeed, B fail?  │
├─────────────────────────────────────────────────────────┤
│  Layer 3 — Time Travel            (Month 2)             │
│  Replay any agent from any checkpoint. Step backward.    │
├─────────────────────────────────────────────────────────┤
│  Layer 2 — Cross-Framework Portability  (Weeks 8–10)    │
│  Agent starts in LangGraph, resumes in CrewAI.           │
├─────────────────────────────────────────────────────────┤
│  Layer 1 — Survival               (Launch)              │
│  Save state. Restore state. Any framework. Five calls.   │
└─────────────────────────────────────────────────────────┘
```

**Critical design decision**: Every Layer 1 data model decision must not block Layers 2–5. This is why:
- `parent_id` exists in v1 (enables time travel in Layer 3)
- `handoff_target` exists in v1 (enables coordination in Layer 5)
- `schema_version` exists in v1 (enables cross-framework portability in Layer 2)
- `checksum` exists in v1 (enables diff in Layer 4)

None of these fields are used at launch. All of them prevent painful migrations later.

---

## 5. Backend Design

### 5.1 Abstract Backend Interface

All backends implement the same four primitives:

```python
class Backend(ABC):
    async def save(self, checkpoint: Checkpoint) -> str:
        """Persist a checkpoint. Returns checkpoint_id."""

    async def load(self, checkpoint_id: str) -> Checkpoint | None:
        """Load exact checkpoint by ID."""

    async def latest(self, agent_id: str, session_id: str) -> Checkpoint | None:
        """Load most recent checkpoint for a session."""

    async def history(self, agent_id: str, session_id: str, limit: int = 50) -> list[Checkpoint]:
        """Load checkpoint chain in reverse-chronological order."""

    async def delete(self, checkpoint_id: str) -> bool:
        """Hard delete. Used by TTL expiry."""
```

No backend knows about agents, frameworks, or the Client. Backends store and retrieve bytes.

### 5.2 SQLite Backend (Local Development)

```
checkpnt_local.db
  └── checkpoints
        ├── id          TEXT PRIMARY KEY    -- checkpoint_id (UUID v7)
        ├── agent_id    TEXT NOT NULL
        ├── session_id  TEXT NOT NULL
        ├── parent_id   TEXT
        ├── payload     BLOB NOT NULL       -- msgpack-encoded Checkpoint
        ├── checksum    TEXT NOT NULL
        ├── created_at  INTEGER NOT NULL    -- Unix timestamp, milliseconds
        └── expires_at  INTEGER             -- NULL = no expiry
```

Indexes on `(agent_id, session_id, created_at)` for efficient history queries.

### 5.3 Redis Backend (Production)

```
Key structure:
  checkpnt:{checkpoint_id}              → msgpack payload (with TTL if set)
  checkpnt:idx:{agent_id}:{session_id} → Sorted set, score = created_at timestamp
  checkpnt:latest:{agent_id}:{session_id} → checkpoint_id string

Operations:
  SAVE:    SET + ZADD + SET (pipeline — atomic)
  LOAD:    GET
  LATEST:  GET checkpnt:latest:...
  HISTORY: ZREVRANGEBYSCORE + pipeline GET
  DELETE:  DEL + ZREM + conditional DEL latest
```

**Why Redis for production?**
- Sub-millisecond reads for restore operations (agents should not wait)
- Native TTL support — no background cleanup job needed
- Sorted sets give O(log N) history queries
- Pub/sub already in Redis — enables Layer 5 without new infrastructure

---

## 6. Adapter Design

### 6.1 What an Adapter Does

An Adapter has exactly two responsibilities:
1. Extract execution state from a framework's native format into a `Checkpoint`
2. Reconstruct a framework's native format from a `Checkpoint`

```python
class Adapter(ABC):
    def extract(self, framework_state: Any, agent_id: str, **kwargs) -> Checkpoint:
        """Framework state → Checkpoint"""

    def reconstruct(self, checkpoint: Checkpoint) -> Any:
        """Checkpoint → Framework state"""
```

### 6.2 LangGraph Adapter

LangGraph state is a `StateSnapshot` containing:
- `values` — the current graph values dict
- `next` — which nodes are next to execute
- `config` — the RunnableConfig (thread_id, etc.)
- `metadata` — step count, run source

The adapter extracts these into `execution_state` and reconstructs them on restore. The developer's agent sees an identical `StateSnapshot` — it does not know a checkpoint was involved.

### 6.3 Cross-Framework Portability (Layer 2)

When an agent transitions from LangGraph to CrewAI:
1. LangGraph adapter extracts state into canonical `Checkpoint`
2. `Checkpoint.handoff_schema` specifies the target schema version
3. CrewAI adapter's `reconstruct()` reads from `agent_context` (framework-agnostic fields)
4. `execution_state` is framework-specific and not portable. `agent_context` is a developer-owned dict — useful for carrying your own keys (task IDs, decisions, tool results) but has no enforced schema.

The split exists so developers have a clear place to store their own data separately from framework internals.

---

## 7. State Versioning

Every checkpoint carries `schema_version: "1.0"`. This is not optional.

### 7.1 Why This Matters

As Checkpnt evolves, the checkpoint schema will change. Without versioning:
- A checkpoint saved by SDK v1 may be unreadable by SDK v2
- Migrations are impossible — you don't know what format old data is in

### 7.2 Migration Strategy

```
Checkpoint saved with schema v1.0
        │
        ▼
SDK v2.0 calls load()
        │
        ▼
Backend returns raw bytes
        │
        ▼
Deserializer reads schema_version = "1.0"
        │
        ▼
MigrationChain runs: v1.0 → v1.1 → v2.0
        │
        ▼
Agent receives v2.0 Checkpoint
```

Migration functions are pure functions: `migrate_1_0_to_1_1(raw: dict) -> dict`. They live in `schemas/versions.py`. They are tested independently of the rest of the system.

---

## 8. Multi-Agent Coordination

This is Layer 5 — not built at launch. But the data model supports it from day one.

### 8.1 The Pattern

```
Supervisor Agent
  ├── saves checkpoint with handoff_target = "worker_agent_A"
  ├── saves checkpoint with handoff_target = "worker_agent_B"
  │
Worker Agent A                    Worker Agent B
  ├── calls restore(target="me")    ├── calls restore(target="me")
  ├── executes                      ├── executes
  ├── saves result checkpoint       ├── saves result checkpoint
  │
Supervisor Agent
  └── calls restore(source="worker_agent_A")
  └── calls restore(source="worker_agent_B")
  └── merges and continues
```

### 8.2 Why Redis Pub/Sub, Not Polling

In Layer 5, supervisor agents will not poll for worker completion. They will subscribe:

```python
# Worker publishes on completion
await client.publish(checkpoint_id, channel="agent:supervisor:inbox")

# Supervisor subscribes
async for event in client.subscribe("agent:supervisor:inbox"):
    result = await client.restore(event.checkpoint_id)
```

This is why Redis is the production backend choice — it already has the pub/sub infrastructure Layer 5 requires.

---

## 9. Immutability Guarantees

**Checkpoints are never modified after creation.**

This is a hard invariant. The backend's `save()` method is insert-only. There is no `update()`. Clients that attempt to overwrite a `checkpoint_id` receive an error.

Why this matters:
- **Auditability**: Every state the agent was in is permanently reconstructable
- **Debugging**: "What was the agent thinking at step 23?" is always answerable
- **Branching**: Two executions can fork from the same parent without conflict
- **Trust**: Downstream systems can cache checkpoint data knowing it will not change

The only mutation operation is `delete()`, which is only called by TTL expiry. Deletion is final.

---

## 10. Roadmap Decisions Baked Into v1

| Decision | Why it exists in v1 | What it enables |
|---|---|---|
| `parent_id` on every checkpoint | Append-only execution tree | Time travel (Layer 3), branching |
| `schema_version` on every checkpoint | Schema evolution from day one | Cross-framework portability (Layer 2), migrations |
| `handoff_target` field | Coordination primitive | Multi-agent handoffs (Layer 5) |
| `checksum` on every checkpoint | Integrity from day one | State diffing (Layer 4), tamper detection |
| UUID v7 checkpoint IDs | Time-ordered keys | Efficient history queries at scale |
| `agent_context` vs `execution_state` split | Developer-owned dict vs framework internals | Clean separation, reserved for future schema enforcement (Layer 2) |
| Redis pub/sub as production backend | Coordination infrastructure | Agent event subscriptions (Layer 5) |
| Abstract Backend interface | Backend is an implementation detail | Postgres, DynamoDB, S3 backends without API changes |
| BSL license | Commercial protection | Prevents SaaS forks while keeping source readable |

---

## Appendix: What We Deliberately Did Not Build

**Not in scope — ever:**
- Semantic memory (that is Mem0's domain)
- LLM model management
- Agent orchestration / scheduling
- Log aggregation or metrics collection
- A UI (the dashboard is a cloud feature, not a core feature)

**Not in scope — yet:**
- Postgres backend (after SQLite + Redis are proven)
- gRPC API (after REST API is proven)
- State encryption at rest (roadmap item, not v1)
- Cross-cloud replication (enterprise tier)

The temptation to expand scope is constant. Every item on the "not in scope" list was explicitly discussed and explicitly deferred. Scope discipline is how infrastructure companies ship.

---

*Last updated: March 2026*
*Architecture owner: Yasha, Tech4Biz Solutions Pvt Ltd*
