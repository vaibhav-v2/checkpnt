"""
Example: LangGraph Agent with Crash Recovery

This is the exact problem reported in LangGraph Issue #5790.
Developers lose all agent state on every hot reload or restart.

With Checkpnt, the agent resumes from its last saved position.
No lost work. No manual state reconstruction.

Run this file. Interrupt it with Ctrl+C. Run it again.
The agent will resume from exactly where it was interrupted.
"""

import asyncio
import sys
from checkpnt import Client, Framework


async def run_agent_with_recovery():
    """
    A multi-step agent that survives interruption.

    In a real agent, replace the simulated steps with:
    - graph.invoke(state) calls
    - tool executions
    - API calls
    """
    agent_id = "demo-research-agent"
    session_id = "research-session-001"

    async with Client.sqlite("./checkpnt_demo.db") as client:

        # Check if we have a previous checkpoint for this session
        history = await client.timeline(agent_id=agent_id, session_id=session_id)

        if history:
            last = history[0]
            print(f"▶ Resuming from step {last.step_index}: {last.step_name}")
            start_step = last.step_index + 1
            context = last.agent_context
            parent_id = last.checkpoint_id
        else:
            print("▶ Starting fresh session")
            start_step = 0
            context = {}
            parent_id = None

        # Simulated agent steps (replace with real agent calls)
        steps = [
            ("search_web", {"query": "AI agent frameworks 2025", "results": 10}),
            ("extract_sources", {"sources": ["paper1.pdf", "paper2.pdf"]}),
            ("analyze_content", {"summary": "LangGraph dominates with 25k stars"}),
            ("generate_report", {"format": "markdown", "length": "medium"}),
            ("save_output", {"path": "./research_output.md"}),
        ]

        for i, (step_name, step_data) in enumerate(steps):
            if i < start_step:
                print(f"  ✓ Step {i} ({step_name}) — already completed, skipping")
                continue

            print(f"  → Step {i}: {step_name}...")

            # Simulate work (replace with real agent logic)
            await asyncio.sleep(0.5)
            context[step_name] = step_data

            # Save checkpoint after each step
            checkpoint_id = await client.save(
                agent_id=agent_id,
                framework=Framework.CUSTOM,
                session_id=session_id,
                parent_id=parent_id,
                execution_state={"current_step": step_name, "step_data": step_data},
                context=context,
                step_index=i,
                step_name=step_name,
            )
            parent_id = checkpoint_id
            print(f"  ✓ Step {i} complete — checkpoint saved: {checkpoint_id[:16]}...")

        print("\n✅ Agent completed all steps")
        print(f"   Total context collected: {list(context.keys())}")


if __name__ == "__main__":
    print("Checkpnt Demo — interrupt with Ctrl+C, then run again to resume")
    print("=" * 60)
    try:
        asyncio.run(run_agent_with_recovery())
    except KeyboardInterrupt:
        print("\n\n⚡ Interrupted — run again to resume from last checkpoint")
        sys.exit(0)
