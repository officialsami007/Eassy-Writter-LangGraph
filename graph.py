"""LangGraph assembly for the essay workflow."""

import functools
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_groq import ChatGroq
from state import EssayState
from nodes import plan_node, research_node, draft_node, critique_node, revision_node, final_node



def get_model() -> ChatGroq:
    """Return the default chat model used by the graph."""
    return ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0.7,
        max_tokens=4096,
    )



def should_revise(state: EssayState) -> str:
    """Route the workflow to revision or finalization."""
    current = state.get("revision_num", 1)
    maximum = state.get("max_revisions", 2)

    if current >= maximum:
        print(f"\n   [Router] Max revisions ({maximum}) reached → final polish")
        return "final"
    else:
        print(f"\n   [Router] Revision {current}/{maximum} → revise again")
        return "revision"



def build_graph(db_path: str = "essay_memory.db"):
    """Build and compile the essay workflow graph."""

    model = get_model()

    def make_node(fn):
        return functools.partial(fn, model=model)

    builder = StateGraph(EssayState)

    builder.add_node("plan",     make_node(plan_node))
    builder.add_node("research", make_node(research_node))
    builder.add_node("draft",    make_node(draft_node))
    builder.add_node("critique", make_node(critique_node))
    builder.add_node("revision", make_node(revision_node))
    builder.add_node("final",    make_node(final_node))

    builder.set_entry_point("plan")

    builder.add_edge("plan",     "research")
    builder.add_edge("research", "draft")
    builder.add_edge("draft",    "critique")
    builder.add_edge("revision", "critique")   
    builder.add_edge("final",    END)

    builder.add_conditional_edges(
        "critique",
        should_revise,
        {"revision": "revision", "final": "final"}
    )

    # Prefer SQLite-backed checkpointing when available.
    try:
        from langgraph.checkpoint.sqlite import SqliteSaver
        import sqlite3
        conn = sqlite3.connect(db_path, check_same_thread=False)
        memory = SqliteSaver(conn)
        print("  💾 SQLite persistence active — sessions saved to disk")
    except Exception:
        try:
            from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
            memory = MemorySaver()
            print("  💾 Using in-memory persistence")
        except Exception:
            memory = MemorySaver()
            print("  💾 Using in-memory persistence")

    graph = builder.compile(
        checkpointer=memory,
        interrupt_after=["critique"]
    )

    return graph
