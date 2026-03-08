"""
graph.py — Fixed for langgraph 1.0.x
──────────────────────────────────────
Lessons 1, 3, 5: GRAPH ASSEMBLY — Nodes + Edges + Persistence + HITL
"""

import functools
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_groq import ChatGroq
from state import EssayState
from nodes import plan_node, research_node, draft_node, critique_node, revision_node, final_node


# ─────────────────────────────────────────────────────────────────
#  MODEL — 100% FREE via Groq
#  Sign up: https://console.groq.com  →  API Keys  →  copy to .env
# ─────────────────────────────────────────────────────────────────

def get_model() -> ChatGroq:
    """
    Groq free tier: fast inference on llama-3.3-70b, no credit card.
    If you hit rate limits, switch to: llama-3.1-8b-instant
    """
    return ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0.7,
        max_tokens=4096,
    )


# ─────────────────────────────────────────────────────────────────
#  CONDITIONAL EDGE — Lesson 1
#  Reads state, decides: keep revising OR move to final
# ─────────────────────────────────────────────────────────────────

def should_revise(state: EssayState) -> str:
    """
    Returns node name to route to based on revision count.
    Called automatically by LangGraph after every critique node.
    """
    current = state.get("revision_num", 1)
    maximum = state.get("max_revisions", 2)

    if current >= maximum:
        print(f"\n   [Router] Max revisions ({maximum}) reached → final polish")
        return "final"
    else:
        print(f"\n   [Router] Revision {current}/{maximum} → revise again")
        return "revision"


# ─────────────────────────────────────────────────────────────────
#  GRAPH BUILDER
# ─────────────────────────────────────────────────────────────────

def build_graph(db_path: str = "essay_memory.db"):
    """
    Assembles the full multi-agent LangGraph.

    Lesson 3 — Persistence:
      Tries langgraph-checkpoint-sqlite first (saves to disk).
      Falls back to MemorySaver if not installed (still fully functional,
      sessions just won't survive a terminal restart).

    Lesson 5 — HITL:
      interrupt_after=["critique"] pauses the graph after every critique
      so the human can review and optionally add feedback.
    """

    model = get_model()

    # Wrap each node with the shared model instance
    def make_node(fn):
        return functools.partial(fn, model=model)

    # ── StateGraph (Lesson 1) ────────────────────────────────────────
    builder = StateGraph(EssayState)

    builder.add_node("plan",     make_node(plan_node))
    builder.add_node("research", make_node(research_node))
    builder.add_node("draft",    make_node(draft_node))
    builder.add_node("critique", make_node(critique_node))
    builder.add_node("revision", make_node(revision_node))
    builder.add_node("final",    make_node(final_node))

    builder.set_entry_point("plan")

    # Normal edges
    builder.add_edge("plan",     "research")
    builder.add_edge("research", "draft")
    builder.add_edge("draft",    "critique")
    builder.add_edge("revision", "critique")   # ← the revision loop
    builder.add_edge("final",    END)

    # Conditional edge — the decision after every critique
    builder.add_conditional_edges(
        "critique",
        should_revise,
        {"revision": "revision", "final": "final"}
    )

    # ── Lesson 3: Persistence ────────────────────────────────────────
    # Try to use SQLite (persists to disk), fall back to memory
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

    # ── Lesson 5: Human-in-the-Loop ─────────────────────────────────
    graph = builder.compile(
        checkpointer=memory,
        interrupt_after=["critique"]
    )

    return graph
