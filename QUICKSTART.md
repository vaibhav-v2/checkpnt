# Quickstart — LangGraph Crash Recovery

You have a LangGraph agent. It loses state on every restart.
This fixes it in two lines.

---

## Install

```bash
pip install checkpnt
```

---

## Before Checkpnt

```python
from langgraph.checkpoint.sqlite import SqliteSaver

with SqliteSaver.from_conn_string("checkpoints.db") as saver:
    app = graph.compile(checkpointer=saver)
    result = app.invoke(input, config=config)
    # State lives in memory during langgraph dev.
    # Process restarts. State gone. Start over.
```

---

## After Checkpnt

```python
from checkpnt.adapters.langgraph import CheckpntSaver

async with CheckpntSaver.from_sqlite("./checkpnt.db") as saver:
    app = graph.compile(checkpointer=saver)
    result = app.invoke(input, config=config)
    # State written to disk on every step.
    # Process restarts. Agent resumes from exact position.
```

Two lines changed. No platform account. No lock-in. Your database, your machine.

---

## Prove It Works

Copy this and run it:

```python
import asyncio
from langgraph.graph import StateGraph, END
from checkpnt.adapters.langgraph import CheckpntSaver
from typing import TypedDict, Annotated
import operator

class State(TypedDict):
    messages: Annotated[list, operator.add]
    step: int

def step_one(state):
    print("  → Running step_one")
    return {"messages": ["step 1 done"], "step": 1}

def step_two(state):
    print("  → Running step_two")
    return {"messages": ["step 2 done"], "step": 2}

graph = StateGraph(State)
graph.add_node("step_one", step_one)
graph.add_node("step_two", step_two)
graph.set_entry_point("step_one")
graph.add_edge("step_one", "step_two")
graph.add_edge("step_two", END)

config = {"configurable": {"thread_id": "my-agent-001"}}

async def main():
    async with CheckpntSaver.from_sqlite("./checkpnt.db") as saver:
        app = graph.compile(checkpointer=saver)
        result = app.invoke({"messages": [], "step": 0}, config=config)
        print(f"\nCompleted: step={result['step']}, messages={result['messages']}")

asyncio.run(main())
```

**Run it once.** Both steps execute. Output:

```
  → Running step_one
  → Running step_two

Completed: step=2, messages=['step 1 done', 'step 2 done']
```

**Run it again immediately.** Neither step re-executes. The graph is at END and Checkpnt knows it:

```
Completed: step=2, messages=['step 1 done', 'step 2 done']
```

**Now simulate a crash.** Interrupt mid-run with Ctrl+C, then run again.
Checkpnt finds the last saved checkpoint and resumes from the next node.
`step_one` does not re-execute.

---

## Use Redis in Production

SQLite is for development. Redis is for production — same interface:

```python
from checkpnt.adapters.langgraph import CheckpntSaver

async with CheckpntSaver.from_redis("redis://your-server:6379") as saver:
    app = graph.compile(checkpointer=saver)
    result = app.invoke(input, config=config)
```

```bash
pip install checkpnt redis
```

---

## What Gets Stored

Every step your graph executes, Checkpnt writes:

- The full graph state at that step
- A parent reference to the previous checkpoint
- A SHA-256 checksum — verified on restore
- The LangGraph metadata (step index, source, channel versions)

Checkpoints are **immutable and append-only**. Nothing gets overwritten.
The full execution history survives process restarts, crashes, and deployments.

---

## Full Example

See `examples/langgraph/crash_recovery.py` for a complete working example
with a 5-step agent that demonstrates crash recovery end-to-end.

```bash
git clone https://github.com/Tech4Biz-Solutions-Pvt-Ltd/checkpnt
cd checkpnt
pip install checkpnt langgraph
python examples/langgraph/crash_recovery.py
```

---

## Next Steps

- [Full README](README.md) — all five operations, Redis backend, architecture
- [Examples](examples/langgraph/) — crash recovery, basic resume, Issue #5790 reproduction
- [GitHub Issues](https://github.com/Tech4Biz-Solutions-Pvt-Ltd/checkpnt/issues) — report problems, request adapters
