"""
main.py
───────
THE ENTRY POINT — run this file to start the Essay Writer.

Brings together ALL course lessons in one interactive CLI:

  Lesson 1  — State, Nodes, Edges (via graph.py + nodes.py)
  Lesson 2  — Agentic Search (research_node calls Tavily multiple times)
  Lesson 3  — Persistence (SQLite checkpointing — resume interrupted runs)
  Lesson 4  — Streaming (watch every node's output appear in real time)
  Lesson 5  — Human-in-the-Loop (pause after critique, get human feedback)
  Lesson 6  — Multi-Agent Essay Writer (all nodes working together)

Usage:
    python main.py
"""

import os
import sys
import uuid
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage

# ── Load .env before anything else ──────────────────────────────────
load_dotenv()
_missing = [k for k in ("GROQ_API_KEY", "TAVILY_API_KEY") if not os.getenv(k)]
if _missing:
    print("\n❌  Missing API keys in .env:", ", ".join(_missing))
    print("    Copy .env.example → .env and fill in your keys.")
    print("    Both keys are FREE — see README.md for signup links.\n")
    sys.exit(1)

from graph import build_graph
from state import EssayState


# ────────────────────────────────────────────────────────────────────
#  COLOURS (works on macOS, Linux, and modern Windows terminals)
# ────────────────────────────────────────────────────────────────────
class C:
    RESET  = "\033[0m"
    BOLD   = "\033[1m"
    CYAN   = "\033[96m"
    GREEN  = "\033[92m"
    YELLOW = "\033[93m"
    RED    = "\033[91m"
    DIM    = "\033[2m"
    MAGENTA= "\033[95m"

def header(text: str):
    print(f"\n{C.BOLD}{C.CYAN}{'━'*60}{C.RESET}")
    print(f"{C.BOLD}{C.CYAN}  {text}{C.RESET}")
    print(f"{C.BOLD}{C.CYAN}{'━'*60}{C.RESET}")

def section(text: str):
    print(f"\n{C.BOLD}{C.YELLOW}▸ {text}{C.RESET}")

def success(text: str):
    print(f"{C.GREEN}✓ {text}{C.RESET}")

def info(text: str):
    print(f"{C.DIM}  {text}{C.RESET}")


# ────────────────────────────────────────────────────────────────────
#  STREAMING HELPER — Lesson 4
#  Watches the graph's stream and prints each node's output live.
# ────────────────────────────────────────────────────────────────────

def stream_graph_updates(graph, input_data, config):
    """
    Lesson 4: STREAMING — iterate over graph.stream() to see every
    state update as it happens, node by node.

    stream_mode="updates" means we receive only the DELTA (what changed)
    after each node — not the full state every time. More efficient.
    """
    for event in graph.stream(input_data, config=config, stream_mode="updates"):
        # event is a dict: {node_name: {field: value, ...}}
        for node_name, updates in event.items():
            if node_name == "__interrupt__":
                # This is the HITL pause signal — handle in the main loop
                return "interrupted"
            # Print the node that just ran
            print(f"\n  {C.MAGENTA}[{node_name}]{C.RESET} {C.DIM}completed{C.RESET}")

    return "done"


# ────────────────────────────────────────────────────────────────────
#  DISPLAY HELPERS
# ────────────────────────────────────────────────────────────────────

def display_essay(state: EssayState, label: str = "CURRENT DRAFT"):
    print(f"\n{'─'*60}")
    print(f"{C.BOLD}{C.GREEN}{label}{C.RESET}")
    print('─'*60)
    print(state.get("draft", "(no draft yet)"))
    print('─'*60)

def display_critique(state: EssayState):
    print(f"\n{'─'*60}")
    print(f"{C.BOLD}{C.YELLOW}CRITIQUE{C.RESET}")
    print('─'*60)
    print(state.get("critique", "(no critique yet)"))
    print('─'*60)

def display_plan(state: EssayState):
    print(f"\n{'─'*60}")
    print(f"{C.BOLD}{C.CYAN}ESSAY PLAN{C.RESET}")
    print('─'*60)
    print(state.get("plan", "(no plan yet)"))
    print('─'*60)


# ────────────────────────────────────────────────────────────────────
#  RESUME / PREVIOUS SESSION HELPER — Lesson 3 (Persistence)
# ────────────────────────────────────────────────────────────────────

def get_thread_id(graph) -> str:
    """
    Lesson 3: PERSISTENCE — let the user resume a previous session
    or start fresh.

    Each session has a unique thread_id. The checkpointer saves all
    state for that thread. Resuming just means passing the same thread_id.
    """
    print("\nDo you want to resume a previous session?")
    print("  [1] Start a brand new essay (recommended)")
    print("  [2] Resume from a saved session ID")
    choice = input("\nChoice (1 or 2): ").strip()

    if choice == "2":
        tid = input("Enter your session ID: ").strip()
        if tid:
            # Check if this thread_id has any saved state
            config = {"configurable": {"thread_id": tid}}
            try:
                saved = graph.get_state(config)
                if saved and saved.values:
                    rev = saved.values.get("revision_num", 0)
                    task = saved.values.get("task", "?")
                    print(f"\n{C.GREEN}✓ Session found!{C.RESET}")
                    print(f"  Topic: {task}")
                    print(f"  Revision: {rev}")
                    return tid
                else:
                    print("  No saved state found for that ID. Starting fresh.")
            except Exception as e:
                print(f"  Could not load session: {e}. Starting fresh.")

    # Generate a new random session ID
    new_tid = str(uuid.uuid4())[:8]
    print(f"\n  {C.DIM}New session ID: {new_tid}  (save this to resume later){C.RESET}")
    return new_tid


# ────────────────────────────────────────────────────────────────────
#  HITL INTERACTION — Lesson 5
#  Called every time the graph pauses after the critique node.
# ────────────────────────────────────────────────────────────────────

def handle_hitl_pause(graph, config: dict, revision_num: int, max_revisions: int):
    """
    Lesson 5: HUMAN-IN-THE-LOOP — the graph has paused after critique.

    The human can:
      A) Accept critique as-is and let the agent revise automatically
      B) Add their own extra feedback (merged into the critique)
      C) Skip revision and jump straight to final polish
      D) View the current draft or critique again
    """
    state = graph.get_state(config).values

    section(f"HUMAN-IN-THE-LOOP PAUSE  (Revision {revision_num}/{max_revisions})")
    print(f"\nThe Critic agent has reviewed the draft. You can now:")
    print("  [1] Show the current draft")
    print("  [2] Show the critique")
    print("  [3] Add your own feedback and continue revising")
    print("  [4] Accept critique as-is (agent revises automatically)")
    print("  [5] Skip revision — go straight to final polish")

    human_feedback = ""
    skip_revision  = False

    while True:
        choice = input(f"\n{C.BOLD}Your choice [1-5]: {C.RESET}").strip()

        if choice == "1":
            display_essay(state, f"DRAFT (Revision {revision_num})")

        elif choice == "2":
            display_critique(state)

        elif choice == "3":
            print(f"\n{C.YELLOW}Type your additional feedback.{C.RESET}")
            print("(This will be added to the critique so the Reviser sees it.)")
            print("Press Enter twice when done.\n")
            lines = []
            while True:
                line = input()
                if line == "" and lines and lines[-1] == "":
                    break
                lines.append(line)
            human_feedback = "\n".join(lines).strip()
            if human_feedback:
                success(f"Feedback captured ({len(human_feedback)} chars)")
                # Lesson 5: update_state() injects changes into the paused graph
                graph.update_state(
                    config,
                    {"human_feedback": human_feedback},
                    as_node="critique"   # act as if critique node emitted this
                )
                success("Feedback added to state. Resuming revision...")
            break

        elif choice == "4":
            info("Accepting critique as-is. Agent will revise automatically.")
            break

        elif choice == "5":
            skip_revision = True
            info("Skipping revision. Jumping to final polish.")
            # Force max_revisions so should_revise() routes to "final"
            graph.update_state(
                config,
                {"max_revisions": revision_num},  # current == max → routes to final
                as_node="critique"
            )
            break

        else:
            print("  Please enter 1, 2, 3, 4, or 5.")

    return skip_revision


# ────────────────────────────────────────────────────────────────────
#  MAIN LOOP
# ────────────────────────────────────────────────────────────────────

def main():
    header("AI ESSAY WRITER  —  Powered by LangGraph + Groq (Free)")
    print(f"\n{C.DIM}Implements: State · Nodes · Edges · Agentic Search ·")
    print(f"Persistence · Streaming · Human-in-the-Loop · Multi-Agent{C.RESET}")

    # ── Build the graph (Lesson 1 + 3) ──────────────────────────────
    print("\nInitialising graph...")
    graph = build_graph(db_path="essay_memory.db")
    success("Graph compiled with SQLite persistence")

    # ── Session management — Lesson 3: Persistence ───────────────────
    thread_id = get_thread_id(graph)
    config    = {"configurable": {"thread_id": thread_id}}

    # ── Check if this is a fresh session or a resume ─────────────────
    saved_state = graph.get_state(config)
    is_resuming = bool(saved_state and saved_state.values.get("task"))

    if is_resuming:
        # Resume from saved checkpoint
        state = saved_state.values
        section(f"Resuming session — Topic: {state['task']}")
        display_essay(state, f"LAST SAVED DRAFT (Revision {state.get('revision_num',0)})")
        max_rev = state.get("max_revisions", 2)
        rev_num = state.get("revision_num", 0)

        if not saved_state.next:
            print(f"\n{C.GREEN}This session is already complete!{C.RESET}")
            display_essay(state, "FINAL ESSAY")
            return

        # Resume — graph was paused at HITL
        print("\nThe graph was paused waiting for your input. Resuming from last checkpoint...")
        handle_hitl_pause(graph, config, rev_num, max_rev)
        # Fall through to the run loop below

    else:
        # ── Fresh session — get topic ────────────────────────────────
        section("ESSAY TOPIC")
        print("\nEnter the topic or question for your essay.")
        print(f"{C.DIM}Example: 'The impact of social media on mental health'{C.RESET}\n")
        task = input(f"{C.BOLD}Essay topic: {C.RESET}").strip()
        if not task:
            task = "The impact of artificial intelligence on the future of work"
            info(f"Using default topic: {task}")

        section("REVISION SETTINGS")
        print(f"\nHow many revision cycles? (Recommended: 2)")
        print(f"{C.DIM}Each cycle: Critique → Human review → Revise{C.RESET}")
        try:
            max_rev = int(input("Max revisions [default 2]: ").strip() or "2")
            max_rev = max(1, min(max_rev, 5))  # clamp 1–5
        except ValueError:
            max_rev = 2

        info(f"Running up to {max_rev} revision cycle(s)")

        # ── Initial state ────────────────────────────────────────────
        initial_state: EssayState = {
            "task":           task,
            "plan":           "",
            "research":       "",
            "draft":          "",
            "critique":       "",
            "revision_num":   0,
            "max_revisions":  max_rev,
            "messages":       [HumanMessage(content=task)],
            "human_feedback": "",
        }

        header("STARTING THE ESSAY PIPELINE")
        print(f"\nTopic      : {C.BOLD}{task}{C.RESET}")
        print(f"Max cycles : {max_rev}")
        print(f"Session ID : {C.DIM}{thread_id}{C.RESET}")
        print(f"\n{C.DIM}Pipeline: Plan → Research → Draft → Critique → [HITL] → Revise (×{max_rev}) → Final{C.RESET}\n")

        # ── Run initial pipeline (Plan → Research → Draft → Critique) ──
        # Lesson 4: STREAMING — we stream updates node by node
        section("Running: Plan → Research → Draft → Critique")
        print(f"{C.DIM}Streaming node updates live...{C.RESET}\n")

        result = stream_graph_updates(graph, initial_state, config)
        rev_num = max_rev  # will be updated from state below

    # ── HITL + REVISION LOOP — Lesson 5 ─────────────────────────────
    while True:
        # Get current saved state
        current = graph.get_state(config)
        if not current or not current.values:
            break

        state   = current.values
        rev_num = state.get("revision_num", 1)
        max_rev = state.get("max_revisions", 2)

        # Check if graph has more steps to run
        if not current.next:
            # Graph finished — no more steps
            break

        # ── HITL pause — Lesson 5 ────────────────────────────────────
        header(f"HUMAN REVIEW — After Critique {rev_num}/{max_rev}")
        handle_hitl_pause(graph, config, rev_num, max_rev)

        # ── Resume after human input ──────────────────────────────────
        section(f"Resuming graph (revision cycle {rev_num})...")

        # Lesson 4: Stream the rest of the run (revision → critique or final)
        for event in graph.stream(None, config=config, stream_mode="updates"):
            for node_name, updates in event.items():
                if node_name == "__interrupt__":
                    break
                print(f"  {C.MAGENTA}[{node_name}]{C.RESET} {C.DIM}completed{C.RESET}")

        # Re-check state to see if we're done or paused again
        updated = graph.get_state(config)
        if not updated or not updated.next:
            break  # graph completed

    # ── DONE — display final essay ────────────────────────────────────
    final = graph.get_state(config)
    if final and final.values:
        state = final.values
        header("🎉  ESSAY COMPLETE!")
        display_essay(state, "✨  FINAL ESSAY")

        # Save to file
        out_file = f"essay_{thread_id}.txt"
        with open(out_file, "w", encoding="utf-8") as f:
            f.write(f"TOPIC: {state.get('task','')}\n")
            f.write(f"REVISIONS: {state.get('revision_num',0)}\n")
            f.write("="*60 + "\n\n")
            f.write(state.get("draft", ""))
        success(f"Essay saved to: {out_file}")
        info(f"Session ID for resuming later: {thread_id}")
    else:
        print(f"\n{C.RED}Something went wrong — no final state found.{C.RESET}")
        print("Try running again with the same session ID.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print(f"\n\n{C.YELLOW}Interrupted. Your progress is saved.{C.RESET}")
        print("Run again and enter your session ID to resume.")
    except Exception as e:
        print(f"\n{C.RED}Error: {e}{C.RESET}")
        import traceback
        traceback.print_exc()
