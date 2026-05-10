"""
examples/langgraph/basic_resume.py
------------------------------------
The simplest possible Checkpnt + LangGraph integration.

Two patterns shown:
  1. CheckpntSaver — drop-in replacement for SqliteSaver (zero code change)
  2. Manual save/restore — full control over when and what gets saved

Run this, interrupt with Ctrl+C, run again. State resumes.
"""

import asyncio
import sys
from typing import TypedDict, Annotated
import operator

# ── Pattern 1: CheckpntSaver (drop-in replacement) ───────────────────────────

async def pattern_one_drop_in():
    """
    Replace SqliteSaver with CheckpntSaver.
    Literally one line change. Everything else stays the same.
    """
    try:
        from langgraph.graph import StateGraph, END
        from langchain_groq import ChatGroq
    except ImportError:
        print("⚠ LangGraph/LangChain not installed. Showing code only.")
        print("""
    # Before (Issue #5790 — state lost on langgraph dev):
    from langgraph.checkpoint.sqlite import SqliteSaver
    with SqliteSaver.from_conn_string("checkpoints.db") as saver:
        app = graph.compile(checkpointer=saver)

    # After (Checkpnt — state survives everything):
    from checkpnt.adapters.langgraph import CheckpntSaver
    async with CheckpntSaver.from_sqlite("./checkpnt_local.db") as saver:
        app = graph.compile(checkpointer=saver)
        """)
        return

    from checkpnt.adapters.langgraph import CheckpntSaver

    class State(TypedDict):
        messages: Annotated[list, operator.add]
        step: int

    llm = ChatGroq(model="llama-3.1-8b-instant")

    def step_one(state):
        print("  → Step 1 running...")
        return {"messages": ["Step 1 complete"], "step": 1}

    def step_two(state):
        print("  → Step 2 running...")
        return {"messages": ["Step 2 complete"], "step": 2}

    graph = StateGraph(State)
    graph.add_node("step_one", step_one)
    graph.add_node("step_two", step_two)
    graph.set_entry_point("step_one")
    graph.add_edge("step_one", "step_two")
    graph.add_edge("step_two", END)

    config = {"configurable": {"thread_id": "demo-thread-001"}}

    # One line change from SqliteSaver:
    async with CheckpntSaver.from_sqlite("./checkpnt_demo.db") as saver:
        app = graph.compile(checkpointer=saver)
        result = app.invoke({"messages": [], "step": 0}, config=config)
        print(f"  ✓ Completed at step {result['step']}")
        print("  ✓ State saved via Checkpnt — survives langgraph dev restarts")


# ── Pattern 2: Manual save/restore (full control) ────────────────────────────

async def pattern_two_manual():
    """
    Manually save state at each step and restore on restart.
    Use this when you want explicit control over checkpoint granularity.
    """
    from checkpnt import Client, Framework
    from checkpnt.adapters.langgraph import LangGraphAdapter

    adapter = LangGraphAdapter()

    async with Client.sqlite("./checkpnt_manual.db") as client:
        agent_id = "research-agent"
        session_id = "research-session-v1"

        # Check for existing session
        history = await client.timeline(agent_id=agent_id, session_id=session_id)

        if history:
            last = history[0]
            print(f"  ▶ Resuming from step {last.step_index}: '{last.step_name}'")
            restored = adapter.reconstruct(last)
            start_step = last.step_index + 1
            accumulated = restored["values"].get("results", [])
            parent_id = last.checkpoint_id
        else:
            print("  ▶ Starting fresh session")
            start_step = 0
            accumulated = []
            parent_id = None

        # Simulated multi-step agent
        steps = [
            ("search",   {"query": "AI agent frameworks"}),
            ("extract",  {"sources": 12, "relevant": 8}),
            ("analyse",  {"insight": "LangGraph leads with 25k stars"}),
            ("report",   {"format": "markdown", "words": 1200}),
        ]

        for i, (name, data) in enumerate(steps):
            if i < start_step:
                print(f"  ✓ [{i}] {name} — already done, skipping")
                continue

            print(f"  → [{i}] {name}...")
            await asyncio.sleep(0.1)  # Simulate work
            accumulated.append({name: data})

            # Build a snapshot-like dict for the adapter
            snapshot = {
                "values": {"results": accumulated, "current_step": name},
                "next": [steps[i + 1][0]] if i + 1 < len(steps) else [],
                "config": {"configurable": {"thread_id": session_id}},
                "metadata": {"step": i, "source": "loop"},
            }

            cp = adapter.extract(
                framework_state=snapshot,
                agent_id=agent_id,
                session_id=session_id,
                parent_id=parent_id,
                step_index=i,
                step_name=name,
            )
            parent_id = await client._backend.save(cp)
            print(f"  ✓ [{i}] {name} — checkpoint saved")

        print(f"\n  ✅ All steps complete. {len(accumulated)} results collected.")


# ── Main ──────────────────────────────────────────────────────────────────────

async def main():
    print("\n=== Pattern 2: Manual Save/Restore ===")
    await pattern_two_manual()


if __name__ == "__main__":
    print("Checkpnt + LangGraph — interrupt with Ctrl+C, run again to resume")
    print("=" * 60)
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\n⚡ Interrupted — run again to resume from last checkpoint")
        sys.exit(0)
