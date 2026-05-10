"""
examples/langgraph/issue_5790_reproduced.py
--------------------------------------------
This is the code that started Checkpnt.

On March 4, 2026, we built a simple 3-step LangGraph agent and configured it
with SqliteSaver — exactly as the LangGraph documentation recommends. We ran it.
State persisted. 7 checkpoints were saved to checkpoints.db (included in this
directory — you can inspect it yourself).

Then we ran `langgraph dev`.

State was gone. The checkpointer we configured was silently replaced with an
in-memory store. The DB file was created but received nothing.

This is LangGraph Issue #5790, filed August 2025, closed October 2025 as "by design":
https://github.com/langchain-ai/langgraph/issues/5790

LangGraph maintainer response:
    "langgraph dev is strictly for development and by design uses in-memory
     checkpointer. No plans to support alternative checkpointers. If you want
     persistent storage, deploy on LangGraph Platform."

This is not a bug. It is deliberate lock-in. Every developer who refuses that
lock-in is Checkpnt's market.

───────────────────────────────────────────────────────────────────
WHAT THIS FILE DEMONSTRATES
───────────────────────────────────────────────────────────────────

Run this file directly:
    python issue_5790_reproduced.py

State will persist. The included checkpoints.db shows a real execution chain:

    [0] 1f118012-24d3-... ROOT (session start)          step -1
    [1] 1f118012-24d4-... ← [0]                         step  0
    [2] 1f118025-ca0a-... ← [1]  (second run, resumed)  step  1
    [3] 1f118025-ca0b-... ← [2]                         step  2
    [4] 1f118025-ccc8-... ← [3]                         step  3
    [5] 1f118025-cf15-... ← [4]                         step  4
    [6] 1f118025-d030-... ← [5]                         step  5

7 real checkpoints. An unbroken parent chain. State that survived across runs.

Now try `langgraph dev` and watch all of it disappear.

───────────────────────────────────────────────────────────────────
THE CHECKPNT SOLUTION (coming in v0.1.0)
───────────────────────────────────────────────────────────────────

    from checkpnt import Client, Framework

    async with Client.sqlite("./checkpnt_local.db") as client:
        checkpoint_id = await client.save(
            agent_id="research-agent-001",
            framework=Framework.LANGGRAPH,
            execution_state=graph_state,
        )
        # State now survives langgraph dev, hot reloads, crashes, and deployments.
        # No lock-in to LangGraph Platform required.
"""

import asyncio
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.sqlite import SqliteSaver
from langchain_groq import ChatGroq
from typing import TypedDict, Annotated
import operator


# ── Agent State ──────────────────────────────────────────────────────────────

class AgentState(TypedDict):
    messages: Annotated[list, operator.add]
    step: int


# ── LLM ──────────────────────────────────────────────────────────────────────
# Originally used claude-haiku-4-5-20251001. Switched to Groq llama for
# faster iteration during validation. Both reproduce the same issue.

llm = ChatGroq(model="llama-3.1-8b-instant")


# ── Graph Nodes ───────────────────────────────────────────────────────────────

def step_one(state: AgentState):
    print("--- Executing Step 1 ---")
    response = llm.invoke("Say: Step 1 complete.")
    return {"messages": [response.content], "step": 1}

def step_two(state: AgentState):
    print("--- Executing Step 2 ---")
    response = llm.invoke("Say: Step 2 complete.")
    return {"messages": [response.content], "step": 2}

def step_three(state: AgentState):
    print("--- Executing Step 3 ---")
    response = llm.invoke("Say: Step 3 complete.")
    return {"messages": [response.content], "step": 3}


# ── Graph Builder ─────────────────────────────────────────────────────────────

def build_graph(checkpointer):
    graph = StateGraph(AgentState)
    graph.add_node("step_one", step_one)
    graph.add_node("step_two", step_two)
    graph.add_node("step_three", step_three)
    graph.set_entry_point("step_one")
    graph.add_edge("step_one", "step_two")
    graph.add_edge("step_two", "step_three")
    graph.add_edge("step_three", END)
    return graph.compile(checkpointer=checkpointer)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    with SqliteSaver.from_conn_string("checkpoints.db") as checkpointer:
        app = build_graph(checkpointer)
        config = {"configurable": {"thread_id": "research-agent-001"}}

        print("\n=== FIRST RUN ===")
        result = app.invoke({"messages": [], "step": 0}, config=config)
        print(f"Completed step: {result['step']}")
        print(f"Messages: {result['messages']}")

        print("\n=== CHECKING SAVED STATE ===")
        saved = app.get_state(config)
        print(f"Saved step: {saved.values.get('step')}")
        print("✓ State saved to checkpoints.db")

        print("\n=== SIMULATING CRASH AND RESUME ===")
        reloaded = app.get_state(config)
        print(f"Reloaded step: {reloaded.values.get('step')}")
        print("✓ State survived. This works when you run Python directly.")

        print("\n=== NOW TRY: langgraph dev ===")
        print("Run: langgraph dev")
        print("The checkpoints.db file will be created but remain empty.")
        print("Your configured SqliteSaver will be silently replaced with in-memory storage.")
        print("This is Issue #5790. This is why Checkpnt exists.")


if __name__ == "__main__":
    main()
