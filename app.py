"""
app.py — Flask Web Server
─────────────────────────
Serves the frontend UI and exposes a streaming API for the LangGraph pipeline.

Endpoints:
  GET  /                    → serves the HTML frontend
  POST /api/start           → starts an essay run, returns thread_id
  GET  /api/stream/<tid>    → SSE stream of agent progress
  POST /api/feedback/<tid>  → injects human feedback, resumes graph
  POST /api/skip/<tid>      → skips revision, jumps to final
  GET  /api/state/<tid>     → returns current state as JSON
  GET  /api/download/<tid>  → download final essay as .txt

Run with:  python app.py
Then open: http://localhost:5000
"""

import os
import json
import queue
import threading
import uuid
import time

from flask import Flask, request, jsonify, Response, render_template, send_file
from dotenv import load_dotenv

load_dotenv()

# Validate keys before importing heavy libs
_missing = [k for k in ("GROQ_API_KEY", "TAVILY_API_KEY") if not os.getenv(k)]
if _missing:
    print(f"\n❌  Missing API keys: {', '.join(_missing)}")
    print("    Copy .env.example → .env and fill in GROQ_API_KEY + TAVILY_API_KEY\n")
    exit(1)

from langchain_core.messages import HumanMessage
from graph import build_graph

app = Flask(__name__)

# ─────────────────────────────────────────────────────────────────
#  In-memory store: thread_id → {graph, config, event_queue, state}
# ─────────────────────────────────────────────────────────────────
sessions: dict = {}

# One shared graph instance (thread-safe with per-thread configs)
_graph = None

def get_graph():
    global _graph
    if _graph is None:
        _graph = build_graph("essay_memory.db")
    return _graph


# ─────────────────────────────────────────────────────────────────
#  ROUTES
# ─────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/start", methods=["POST"])
def start():
    """Start a new essay. Returns thread_id for tracking."""
    data = request.get_json()
    task = (data.get("task") or "").strip()
    if not task:
        return jsonify({"error": "Essay topic is required"}), 400

    max_rev = max(1, min(int(data.get("max_revisions", 2)), 5))
    thread_id = str(uuid.uuid4())[:8]

    config = {"configurable": {"thread_id": thread_id}}
    q = queue.Queue()

    sessions[thread_id] = {
        "config": config,
        "queue":  q,
        "status": "running",
        "stage":  "starting",
    }

    initial_state = {
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

    # Run the graph in a background thread so Flask stays responsive
    t = threading.Thread(
        target=_run_pipeline,
        args=(thread_id, initial_state, config, q),
        daemon=True
    )
    t.start()

    return jsonify({"thread_id": thread_id, "max_revisions": max_rev})


def _run_pipeline(thread_id: str, initial_state: dict, config: dict, q: queue.Queue):
    """Background thread: runs the LangGraph pipeline, emits events to SSE queue."""
    graph = get_graph()
    sess  = sessions[thread_id]

    try:
        _stream_run(graph, initial_state, config, q, sess)

    except Exception as e:
        q.put({"type": "error", "message": str(e)})
        sess["status"] = "error"


def _stream_run(graph, input_data, config: dict, q: queue.Queue, sess: dict):
    """Streams graph events into the SSE queue until interrupted or done."""
    for event in graph.stream(input_data, config=config, stream_mode="updates"):
        for node_name, updates in event.items():
            if node_name == "__interrupt__":
                # Graph paused — HITL moment
                state = graph.get_state(config).values
                q.put({
                    "type":     "hitl_pause",
                    "stage":    "review",
                    "draft":    state.get("draft", ""),
                    "critique": state.get("critique", ""),
                    "revision": state.get("revision_num", 1),
                    "maximum":  state.get("max_revisions", 2),
                })
                sess["status"] = "paused"
                return  # exit — will be resumed via /api/feedback or /api/skip

            # Normal node completion
            state = graph.get_state(config).values
            payload = {
                "type":     "node_complete",
                "node":     node_name,
                "stage":    node_name,
                "revision": state.get("revision_num", 0),
            }

            # Attach relevant output for each node type
            if node_name == "plan":
                payload["plan"] = state.get("plan", "")
            elif node_name == "research":
                snippet = state.get("research", "")
                payload["research_snippet"] = snippet[:300] + "..." if len(snippet) > 300 else snippet
            elif node_name in ("draft", "revision", "final"):
                payload["draft"] = state.get("draft", "")
            elif node_name == "critique":
                payload["critique"] = state.get("critique", "")

            q.put(payload)

    # Graph finished naturally (hit END node)
    final = graph.get_state(config).values
    task  = final.get("task", "")
    draft = final.get("draft", "")
    rev   = final.get("revision_num", 0)
    q.put({
        "type":  "complete",
        "draft": draft,
        "task":  task,
    })
    sess["status"] = "complete"
    # Save to history file so user can revisit later
    try:
        tid = config.get("configurable", {}).get("thread_id", "unknown")
        save_essay_to_history(tid, task, draft, rev,
            plan=final.get("plan",""),
            research=final.get("research",""),
            critique=final.get("critique",""))
    except Exception as e:
        print(f"   Warning: could not save to history: {e}")


@app.route("/api/stream/<thread_id>")
def stream(thread_id: str):
    """SSE endpoint — browser connects here to receive live updates."""
    if thread_id not in sessions:
        return jsonify({"error": "Session not found"}), 404

    q = sessions[thread_id]["queue"]

    def event_generator():
        yield "data: " + json.dumps({"type": "connected"}) + "\n\n"
        while True:
            try:
                event = q.get(timeout=60)
                yield "data: " + json.dumps(event) + "\n\n"
                if event.get("type") in ("complete", "error", "hitl_pause"):
                    break
            except queue.Empty:
                yield "data: " + json.dumps({"type": "ping"}) + "\n\n"

    return Response(
        event_generator(),
        mimetype="text/event-stream",
        headers={
            "Cache-Control":               "no-cache",
            "X-Accel-Buffering":           "no",
            "Access-Control-Allow-Origin": "*",
        }
    )


@app.route("/api/feedback/<thread_id>", methods=["POST"])
def feedback(thread_id: str):
    """Human submits feedback → inject into state → resume graph."""
    if thread_id not in sessions:
        return jsonify({"error": "Session not found"}), 404

    sess   = sessions[thread_id]
    data   = request.get_json()
    config = sess["config"]
    graph  = get_graph()
    q      = sess["queue"]

    human_text = (data.get("feedback") or "").strip()

    if human_text:
        # Lesson 5: update_state() — inject human feedback into the paused graph
        graph.update_state(
            config,
            {"human_feedback": human_text},
            as_node="critique"
        )

    sess["status"] = "running"
    q.put({"type": "resuming", "message": "Resuming with your feedback..."})

    # Resume the graph in a background thread
    t = threading.Thread(
        target=_resume_pipeline,
        args=(thread_id, config, q, sess),
        daemon=True
    )
    t.start()

    return jsonify({"ok": True})


@app.route("/api/skip/<thread_id>", methods=["POST"])
def skip(thread_id: str):
    """Skip remaining revisions → jump straight to final polish."""
    if thread_id not in sessions:
        return jsonify({"error": "Session not found"}), 404

    sess   = sessions[thread_id]
    config = sess["config"]
    graph  = get_graph()
    q      = sess["queue"]

    # Force max_revisions = current revision so should_revise() routes to "final"
    state = graph.get_state(config).values
    graph.update_state(
        config,
        {"max_revisions": state.get("revision_num", 1)},
        as_node="critique"
    )

    sess["status"] = "running"
    q.put({"type": "resuming", "message": "Skipping to final polish..."})

    t = threading.Thread(
        target=_resume_pipeline,
        args=(thread_id, config, q, sess),
        daemon=True
    )
    t.start()

    return jsonify({"ok": True})


def _resume_pipeline(thread_id: str, config: dict, q: queue.Queue, sess: dict):
    """Resume a paused graph (after HITL)."""
    graph = get_graph()
    try:
        _stream_run(graph, None, config, q, sess)
    except Exception as e:
        q.put({"type": "error", "message": str(e)})
        sess["status"] = "error"


@app.route("/api/state/<thread_id>")
def get_state(thread_id: str):
    """Return current state snapshot as JSON."""
    if thread_id not in sessions:
        return jsonify({"error": "not found"}), 404
    graph  = get_graph()
    config = sessions[thread_id]["config"]
    snap   = graph.get_state(config)
    if snap and snap.values:
        v = snap.values
        return jsonify({
            "task":         v.get("task", ""),
            "plan":         v.get("plan", ""),
            "research":     v.get("research", ""),
            "draft":        v.get("draft", ""),
            "critique":     v.get("critique", ""),
            "revision_num": v.get("revision_num", 0),
            "max_revisions":v.get("max_revisions", 2),
            "status":       sessions[thread_id]["status"],
        })
    return jsonify({"status": sessions[thread_id]["status"]})


@app.route("/api/download/<thread_id>")
def download(thread_id: str):
    """Download the final essay as a .txt file."""
    if thread_id not in sessions:
        return jsonify({"error": "not found"}), 404
    graph  = get_graph()
    config = sessions[thread_id]["config"]
    snap   = graph.get_state(config)
    if snap and snap.values:
        v    = snap.values
        task = v.get("task", "essay")
        text = f"TOPIC: {task}\n{'='*60}\n\n{v.get('draft','')}"
        path = f"/tmp/essay_{thread_id}.txt"
        with open(path, "w", encoding="utf-8") as f:
            f.write(text)
        return send_file(path, as_attachment=True,
                         download_name=f"essay_{thread_id}.txt")
    return jsonify({"error": "No essay yet"}), 404


# ─────────────────────────────────────────────────────────────────
#  ESSAY HISTORY — saved to essays.json on every completion
# ─────────────────────────────────────────────────────────────────

HISTORY_FILE = "essays.json"

def save_essay_to_history(thread_id: str, task: str, draft: str, revision_num: int,
                           plan: str = "", research: str = "", critique: str = ""):
    """Append a completed essay to the local history file."""
    import json as _json
    from datetime import datetime
    history = []
    if os.path.exists(HISTORY_FILE):
        try:
            history = _json.loads(open(HISTORY_FILE).read())
        except Exception:
            history = []
    entry = {
        "thread_id":    thread_id,
        "task":         task,
        "draft":        draft,
        "plan":         plan,
        "research":     research,
        "critique":     critique,
        "revision_num": revision_num,
        "word_count":   len(draft.split()),
        "saved_at":     datetime.now().strftime("%Y-%m-%d %H:%M"),
    }
    # Update if exists, else prepend
    for i, item in enumerate(history):
        if item["thread_id"] == thread_id:
            history[i] = entry
            break
    else:
        history.insert(0, entry)
    # Keep last 50
    history = history[:50]
    open(HISTORY_FILE, "w").write(_json.dumps(history, indent=2))


@app.route("/api/essays")
def list_essays():
    """Return saved essay history."""
    import json as _json
    if not os.path.exists(HISTORY_FILE):
        return jsonify([])
    try:
        history = _json.loads(open(HISTORY_FILE).read())
        # Return summary (no full draft text to keep it light)
        summaries = [{
            "thread_id":    e["thread_id"],
            "task":         e["task"],
            "word_count":   e.get("word_count", 0),
            "revision_num": e.get("revision_num", 0),
            "saved_at":     e.get("saved_at", ""),
        } for e in history]
        return jsonify(summaries)
    except Exception as e:
        return jsonify([])


@app.route("/api/essays/<thread_id>", methods=["DELETE"])
def delete_essay(thread_id: str):
    """Delete a specific essay from history."""
    import json as _json
    if not os.path.exists(HISTORY_FILE):
        return jsonify({"ok": True})
    try:
        history = _json.loads(open(HISTORY_FILE).read())
        history = [e for e in history if e["thread_id"] != thread_id]
        open(HISTORY_FILE, "w").write(_json.dumps(history, indent=2))
        return jsonify({"ok": True})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/essays/<thread_id>")
def get_essay(thread_id: str):
    """Return a specific saved essay by thread_id."""
    import json as _json
    if not os.path.exists(HISTORY_FILE):
        return jsonify({"error": "not found"}), 404
    try:
        history = _json.loads(open(HISTORY_FILE).read())
        for item in history:
            if item["thread_id"] == thread_id:
                return jsonify(item)
        return jsonify({"error": "not found"}), 404
    except Exception:
        return jsonify({"error": "read error"}), 500


# ─────────────────────────────────────────────────────────────────
#  ENTRY POINT
# ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n" + "═"*50)
    print("  📝  AI Essay Writer  —  Web Interface")
    print("═"*50)
    print("  Open in browser: http://localhost:5000")
    print("  Press Ctrl+C to stop")
    print("═"*50 + "\n")
    app.run(debug=False, threaded=True, port=5000)