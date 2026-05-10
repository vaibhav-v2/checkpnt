"""
Integration tests for the core Checkpnt value proposition:
AI agents resume from exactly where they stopped.

These tests require langgraph to be installed.
They are skipped automatically if langgraph is not available.

Run with:
    pytest tests/integration/test_langgraph_resume.py -v
"""

import asyncio
import os
import pytest

try:
    from langgraph.graph import StateGraph, END
    LANGGRAPH_AVAILABLE = True
except ImportError:
    LANGGRAPH_AVAILABLE = False

pytestmark = pytest.mark.skipif(
    not LANGGRAPH_AVAILABLE,
    reason="langgraph not installed"
)


# ── Shared fixtures ───────────────────────────────────────────────────────────

from typing import TypedDict, Annotated
import operator

class State(TypedDict):
    messages: Annotated[list, operator.add]
    step: int

def build_graph(execution_log: list):
    """3-step graph that records which nodes actually executed."""
    def step_one(state):
        execution_log.append("step_one")
        return {"messages": ["s1"], "step": 1}

    def step_two(state):
        execution_log.append("step_two")
        return {"messages": ["s2"], "step": 2}

    def step_three(state):
        execution_log.append("step_three")
        return {"messages": ["s3"], "step": 3}

    graph = StateGraph(State)
    graph.add_node("step_one", step_one)
    graph.add_node("step_two", step_two)
    graph.add_node("step_three", step_three)
    graph.set_entry_point("step_one")
    graph.add_edge("step_one", "step_two")
    graph.add_edge("step_two", "step_three")
    graph.add_edge("step_three", END)
    return graph


# ── Scenario 1: Full run → resume at END ─────────────────────────────────────

@pytest.mark.asyncio
async def test_resume_at_end_does_not_rerun_nodes(tmp_path):
    """
    After a complete run, invoke(None, config) should not re-execute any nodes.
    The graph is at END — there is nothing to resume.
    """
    from checkpnt.adapters.langgraph import CheckpntSaver

    db = str(tmp_path / "resume_end.db")
    execution_log = []
    config = {"configurable": {"thread_id": "end-test"}}

    async with CheckpntSaver.from_sqlite(db) as saver:
        app = build_graph(execution_log).compile(checkpointer=saver)

        result = app.invoke({"messages": [], "step": 0}, config=config)
        assert result["step"] == 3
        assert execution_log == ["step_one", "step_two", "step_three"]

        execution_log.clear()

        # Resume — graph is at END, nothing should re-execute
        result2 = app.invoke(None, config=config)
        assert result2["step"] == 3
        assert execution_log == [], f"Nodes re-executed on resume: {execution_log}"


# ── Scenario 2: Mid-graph crash → resume from correct node ───────────────────

@pytest.mark.asyncio
async def test_resume_after_crash_continues_from_correct_node(tmp_path):
    """
    The core value proposition of Checkpnt.

    Agent crashes after step_one. On restart, it resumes from step_two.
    step_one must NOT be re-executed.

    This is exactly the failure mode in LangGraph Issue #5790.
    """
    from checkpnt.adapters.langgraph import CheckpntSaver

    db = str(tmp_path / "resume_crash.db")
    execution_log = []
    config = {"configurable": {"thread_id": "crash-test"}}

    # === Run 1: crash after step_one ===
    async with CheckpntSaver.from_sqlite(db) as saver:
        app = build_graph(execution_log).compile(checkpointer=saver)
        # Inject state as if step_one completed, then "crash"
        app.update_state(config, {"messages": ["s1"], "step": 1}, as_node="step_one")
        saved = app.get_state(config)
        assert saved.values["step"] == 1
        assert "step_two" in saved.next

    execution_log.clear()

    # === Run 2: new saver instance (simulates app restart) ===
    async with CheckpntSaver.from_sqlite(db) as saver2:
        app2 = build_graph(execution_log).compile(checkpointer=saver2)
        result = app2.invoke(None, config=config)

        assert result["step"] == 3, f"Expected step=3, got {result['step']}"
        assert "step_one" not in execution_log, \
            f"step_one re-executed after crash (parent chain broken): {execution_log}"
        assert "step_two" in execution_log
        assert "step_three" in execution_log


# ── Scenario 3: LangChain message objects survive serialization ───────────────

@pytest.mark.asyncio
async def test_langchain_messages_survive_serialization(tmp_path):
    """
    HumanMessage and AIMessage objects must survive the Checkpnt
    serialization round-trip intact — not silently corrupted to strings.

    This was bug #2 in our critical review.
    """
    from checkpnt.adapters.langgraph import CheckpntSaver

    try:
        from langchain_core.messages import HumanMessage, AIMessage
    except ImportError:
        pytest.skip("langchain_core not installed")

    class ChatState(TypedDict):
        messages: Annotated[list, operator.add]

    def chat_node(state):
        return {"messages": [AIMessage(content="Hello back!")]}

    chat_graph = StateGraph(ChatState)
    chat_graph.add_node("chat_node", chat_node)
    chat_graph.set_entry_point("chat_node")
    chat_graph.add_edge("chat_node", END)

    db = str(tmp_path / "messages.db")
    config = {"configurable": {"thread_id": "msg-test"}}

    async with CheckpntSaver.from_sqlite(db) as saver:
        app = chat_graph.compile(checkpointer=saver)
        result = app.invoke(
            {"messages": [HumanMessage(content="Hello")]},
            config=config,
        )

        saved = app.get_state(config)
        msgs = saved.values.get("messages", [])
        types = [type(m).__name__ for m in msgs]

        assert "HumanMessage" in types, \
            f"HumanMessage corrupted. Got types: {types}"
        assert "AIMessage" in types, \
            f"AIMessage corrupted. Got types: {types}"


# ── Scenario 4: State survives across saver instances ────────────────────────

@pytest.mark.asyncio
async def test_full_state_survives_saver_restart(tmp_path):
    """
    Complete execution state — not just step index — must survive
    creating a new CheckpntSaver instance pointing at the same DB.
    """
    from checkpnt.adapters.langgraph import CheckpntSaver

    db = str(tmp_path / "full_state.db")
    config = {"configurable": {"thread_id": "state-test"}}
    execution_log = []

    # Run to completion
    async with CheckpntSaver.from_sqlite(db) as saver:
        app = build_graph(execution_log).compile(checkpointer=saver)
        result = app.invoke({"messages": [], "step": 0}, config=config)
        assert result["step"] == 3
        assert result["messages"] == ["s1", "s2", "s3"]

    # New saver — check state is fully readable
    async with CheckpntSaver.from_sqlite(db) as saver2:
        app2 = build_graph([]).compile(checkpointer=saver2)
        saved = app2.get_state(config)

        assert saved.values["step"] == 3, \
            f"Step not preserved. Got: {saved.values.get('step')}"
        assert saved.values["messages"] == ["s1", "s2", "s3"], \
            f"Messages not preserved. Got: {saved.values.get('messages')}"
        assert saved.next == (), \
            f"Expected graph at END. Got next: {saved.next}"
