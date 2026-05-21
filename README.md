# checkpnt

**Framework-agnostic state persistence for AI agents.**  
Resume any agent from exactly where it stopped.

```python
from checkpnt import Client, Framework

async with Client.sqlite() as client:
    # Save agent state mid-execution
    checkpoint_id = await client.save(
        agent_id="invoice-processor",
        framework=Framework.LANGGRAPH,
        execution_state=graph_state,
        context={"invoice_id": "INV-4821", "step": "validate"},
    )

    # Crash. Restart. Resume from exact position.
    checkpoint = await client.restore(checkpoint_id)
    graph_state = checkpoint.execution_state
```

---

## The Problem

AI agents run multi-step processes. Every step, they accumulate state — tool results, decisions, intermediate data, coordination context. That state lives in memory.

The moment anything interrupts the agent — a crash, a restart, a hot reload during development, a deployment — the state is gone. The agent has no memory of where it was. It must start over.

**This is not a corner case. It is the daily experience of every team building production agents.**

From [LangGraph Issue #5790](https://github.com/langchain-ai/langgraph/issues/5790), filed August 2025:

> *"Every code change triggering hot reload forces recreating conversation state from scratch. This severely impacts development efficiency."*

> *"After restarting, all conversation state is lost. A checkpoint.db file is created but remains empty — it never receives data."*

> *"Users must choose between Send objects OR checkpointing. You cannot use both."*

LangGraph maintainers closed Issue #5790 as **"by design"**. The `langgraph dev` command intentionally replaces any configured checkpointer with in-memory storage — it forces you onto LangGraph Platform. Every developer who refuses that lock-in is who Checkpnt is built for.

---

## Quick Start

```bash
pip install checkpnt
```

📖 **New to Checkpnt?** Start with the [LangGraph Quickstart](QUICKSTART.md) — working crash recovery in 5 minutes.

**Save and restore state — any framework, 5 lines:**

```python
import asyncio
from checkpnt import Client, Framework

async def main():
    async with Client.sqlite() as client:

        # Your agent runs some steps...
        graph_state = {"messages": ["result from step 3"], "step": 3}

        # Save state
        checkpoint_id = await client.save(
            agent_id="my-agent",
            framework=Framework.LANGGRAPH,
            execution_state=graph_state,
            context={"run_id": "run-001"},
            step_index=3,
            step_name="after_tool_call",
        )
        print(f"Saved: {checkpoint_id}")

        # Simulate crash — restart from here
        checkpoint = await client.restore(checkpoint_id)
        print(f"Restored step: {checkpoint.execution_state['step']}")
        # → Restored step: 3

asyncio.run(main())
```

**LangGraph drop-in — one line change:**

```python
# Before (state lost on every langgraph dev restart):
from langgraph.checkpoint.sqlite import SqliteSaver
with SqliteSaver.from_conn_string("checkpoints.db") as saver:
    app = graph.compile(checkpointer=saver)

# After (state survives everything, no platform lock-in):
from checkpnt.adapters.langgraph import CheckpntSaver
async with CheckpntSaver.from_sqlite("./checkpnt_local.db") as saver:
    app = graph.compile(checkpointer=saver)
```

---

## The Five Operations

That is all there is.

| Operation | What it does |
|---|---|
| `save()` | Persist agent state at any point in execution |
| `restore()` | Resume from exact saved position — integrity verified |
| `handoff()` | Transfer state between agents with parent chain preserved |
| `timeline()` | Full execution history, newest first |
| `expire()` | Delete immediately, or set TTL on save for auto-expiry |

```python
async with Client.sqlite() as client:

    # 1. Save
    checkpoint_id = await client.save(
        agent_id="processor",
        framework=Framework.LANGGRAPH,
        execution_state=state,
        session_id="run-001",
        step_index=7,
        ttl_seconds=3600,          # auto-expire after 1 hour
    )

    # 2. Restore
    checkpoint = await client.restore(checkpoint_id)

    # 3. Handoff — Agent A → Agent B, state travels with it
    handoff_id = await client.handoff(checkpoint_id, target_agent_id="approval-agent")

    # 4. Timeline — full audit trail
    history = await client.timeline(agent_id="processor", session_id="run-001")
    for cp in history:
        print(f"  step {cp.step_index}: {cp.step_name}")

    # 5. Expire
    await client.expire(checkpoint_id)
```

---

## LangGraph Adapter

The `LangGraphAdapter` and `CheckpntSaver` are the direct answer to Issue #5790.

```python
from checkpnt.adapters.langgraph import LangGraphAdapter, CheckpntSaver

# Pattern 1: Drop-in checkpointer (zero code change to your graph)
async with CheckpntSaver.from_sqlite("./checkpnt_local.db") as saver:
    app = graph.compile(checkpointer=saver)
    result = app.invoke(input, config=config)
    # State now persists across langgraph dev restarts, crashes, deployments.

# Pattern 2: Manual — full control over checkpoint granularity
adapter = LangGraphAdapter()

state_snapshot = app.get_state(config)
checkpoint = adapter.extract(
    state_snapshot,
    agent_id="my-agent",
    context={"invoice_id": "INV-001"},   # your own keys — stored separately from framework state
)
checkpoint_id = await client._backend.save(checkpoint)

# Restore
checkpoint = await client.restore(checkpoint_id)
restored = adapter.reconstruct(checkpoint)
# restored["values"]  → your graph state dict
# restored["next"]    → pending nodes
# restored["config"]  → RunnableConfig to resume with
```

**Proof this is a real problem:** The `examples/langgraph/` directory contains:
- `issue_5790_reproduced.py` — the exact code that exposed the bug  
- `checkpoints.db` — the real SQLite database from that session, with 7 checkpoints in an unbroken parent chain, timestamped March 4 2026

Run `python issue_5790_reproduced.py`. It works. Then run `langgraph dev`. Watch it break. That is why this library exists.

---

## Backends

### SQLite — local development

```python
client = Client.sqlite("./checkpnt_local.db")   # default: ./checkpnt_local.db
```

Zero dependencies beyond `aiosqlite`. WAL mode enabled. Full history queries indexed. Works offline. Commits to a file you can inspect directly.

### Redis — production

```python
client = Client.redis("redis://localhost:6379")
```

Sub-millisecond restores. Native TTL expiry. Sorted set history queries — O(log N), not O(N). Single-pipeline atomic writes. The pub/sub infrastructure is already there for multi-agent coordination (coming in v0.2).

```bash
# Spin up Redis locally
docker run -p 6379:6379 redis:7-alpine
```

### Switching backends

```python
# Same API, swap one line
client = Client.sqlite("./dev.db")          # local
client = Client.redis("redis://prod:6379")  # production
```

Your agent code never changes.

---

## State Schema

Every checkpoint carries:

```python
checkpoint.checkpoint_id      # time-ordered UUID — efficient range queries
checkpoint.agent_id           # stable identifier across restarts
checkpoint.session_id         # groups checkpoints within one run
checkpoint.parent_id          # forms an append-only execution tree
checkpoint.framework          # langgraph | crewai | autogen | custom
checkpoint.schema_version     # "1.0" — versioned for forward migrations
checkpoint.step_index         # monotonically increasing within a session
checkpoint.execution_state    # framework-specific state (LangGraph, CrewAI, etc.)
checkpoint.agent_context      # your developer-owned dict (unstructured, no schema enforced)
checkpoint.checksum           # SHA-256 — integrity verified on every restore
checkpoint.created_at         # UTC timestamp
checkpoint.ttl_seconds        # auto-expiry
checkpoint.handoff_target     # for multi-agent coordination (v0.2)
```

**Checkpoints are immutable.** There is no `update()`. Every new state creates a new checkpoint with a `parent_id` link. This gives you:

- Free time travel — walk the parent chain to any previous state
- Free branching — two agents can fork from the same parent
- Free auditability — every state the agent was ever in is reconstructable

---

## Installation

```bash
# Core (SQLite backend)
pip install checkpnt

# With Redis backend
pip install "checkpnt[redis]"

# With LangGraph adapter
pip install "checkpnt[langgraph]"

# Everything
pip install "checkpnt[all]"
```

Requires Python 3.11+.

---

## Architecture

Checkpnt is built in layers. v0.1 ships Layer 1. The data model is designed to never require a breaking migration to reach Layers 2–5.

```
Layer 5 — Coordination Protocol     typed state contracts between agents, pub/sub
Layer 4 — State Diffing             compare runs, diagnose failures
Layer 3 — Time Travel               replay any agent from any checkpoint
Layer 2 — Cross-Framework           agent starts in LangGraph, resumes in CrewAI
Layer 1 — Survival          ← v0.1  crash recovery, five operations, two backends
```

Read [`ARCHITECTURE.md`](ARCHITECTURE.md) for the full design — why every v0.1 field anticipates Layers 2–5, and what was deliberately left out.

---

## Supported Frameworks

| Framework | Adapter | Status |
|---|---|---|
| LangGraph | `checkpnt.adapters.langgraph` | ✅ v0.1 |
| CrewAI | `checkpnt.adapters.crewai` | 🔜 v0.2 |
| AutoGen | `checkpnt.adapters.autogen` | 🔜 v0.2 |
| Custom | `Framework.CUSTOM` | ✅ v0.1 |

For custom frameworks, pass your state as a plain dict to `execution_state`. No adapter needed.

---

## Compared to Alternatives

| | Checkpnt | Mem0 | LangGraph checkpointer | DIY Redis |
|---|---|---|---|---|
| **What it persists** | Execution state | Semantic memory | Execution state | Whatever you build |
| **Framework-agnostic** | ✅ | ✅ | ❌ LangGraph only | ✅ |
| **Works in `langgraph dev`** | ✅ | — | ❌ by design | ✅ |
| **Production backend** | Redis | Managed | LangGraph Platform | Redis |
| **Open source** | ✅ BSL | ✅ Apache | ✅ MIT | — |
| **Schema versioning** | ✅ | — | ❌ | — |
| **Integrity checks** | ✅ | — | ❌ | — |
| **Multi-agent handoff** | 🔜 v0.2 | ❌ | ❌ | Build it yourself |

**Mem0 vs Checkpnt:** These are different primitives. Mem0 persists what your agent *knows* (semantic memory). Checkpnt persists where your agent *was* (execution state). Most production systems need both.

---

## Development

```bash
git clone https://github.com/vaibhav-v2/checkpnt
cd checkpnt
pip install -e ".[dev]"
pytest tests/unit/          # 55 tests, no external dependencies
pytest tests/integration/   # Redis tests — requires Redis running
```

---

## Contributing

See [`CONTRIBUTING.md`](CONTRIBUTING.md). Issues and PRs welcome.

If you are hitting a state persistence problem that Checkpnt does not solve, open an issue with the framework, the error, and what you expected to happen. That is how this library gets better.

---

## License

Business Source License 1.1 — source-available, not open for competing SaaS forks.  
Converts to Apache 2.0 on January 1, 2029.

---

*Built by [Tech4Biz Solutions](https://tech4biz.in) · [github.com/vaibhav-v2/checkpnt](https://github.com/vaibhav-v2/checkpnt)*
