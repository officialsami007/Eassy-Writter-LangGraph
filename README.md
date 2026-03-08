# 📝 AI Essay Writer — LangGraph Multi-Agent System

A production-grade multi-agent essay writing system that implements **every lesson**
from the DeepLearning.AI course *"AI Agents in LangGraph"*.

**100% FREE to run** — uses Groq (free LLM) + Tavily (free search).

---

## 🚀 Quick Start (5 Steps)

### Step 1 — Get Your Free API Keys

| Service | What it does | Free tier | Sign-up link |
|---------|-------------|-----------|--------------|
| **Groq** | Runs the LLM (writer, critic, etc.) | Unlimited (rate-limited) | https://console.groq.com |
| **Tavily** | Web search for research | 1,000 searches/month | https://app.tavily.com |

For both: Create account → go to API Keys section → create a key → copy it.

---

### Step 2 — Set Up the Project

```bash
# Clone or unzip the project, then enter the folder
cd essay_writer

# Create a Python virtual environment (keeps dependencies isolated)
python -m venv venv

# Activate it:
# On macOS / Linux:
source venv/bin/activate
# On Windows:
venv\Scripts\activate
```

---

### Step 3 — Install Dependencies

```bash
pip install -r requirements.txt
```

> ⏱️ This takes ~1-2 minutes the first time.

---

### Step 4 — Add Your API Keys

```bash
# Copy the example file
cp .env.example .env

# Open .env in any text editor and paste your keys:
# GROQ_API_KEY=gsk_your_key_here
# TAVILY_API_KEY=tvly-your_key_here
```

On Windows: just open `.env.example`, save a copy as `.env`, and edit it.

---

### Step 5 — Run It!

```bash
python main.py
```

The program will ask for your essay topic and walk you through everything.

---

## 🎮 How It Works (What You'll See)

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  AI ESSAY WRITER  —  Powered by LangGraph + Groq (Free)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Essay topic: The impact of AI on employment

▸ Running: Plan → Research → Draft → Critique

  [plan]     completed
  [research] completed   ← Tavily called 3x with different queries
  [draft]    completed
  [critique] completed

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  HUMAN REVIEW — After Critique 1/2
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  [1] Show the current draft
  [2] Show the critique
  [3] Add your own feedback and continue revising   ← Human-in-the-Loop
  [4] Accept critique as-is
  [5] Skip revision → go straight to final polish
```

---

## 🧠 Course Lessons Implemented

| Lesson | Concept | Where in code |
|--------|---------|---------------|
| 0 | Agent from Scratch (ReAct loop) | `main.py` comments + documented in `nodes.py` |
| 1 | State, Nodes, Edges | `state.py`, `nodes.py`, `graph.py` |
| 2 | Agentic Search (Tavily, multiple queries) | `nodes.py → research_node()` |
| 3 | Persistence (SQLite checkpointing, thread IDs) | `graph.py → build_graph()`, `main.py → get_thread_id()` |
| 4 | Streaming (node-by-node live output) | `main.py → stream_graph_updates()` |
| 5 | Human-in-the-Loop (pause, feedback, resume) | `main.py → handle_hitl_pause()`, `graph.py → interrupt_after` |
| 6 | Multi-Agent Essay Writer (all together) | All files |

---

## 🗂️ Project Structure

```
essay_writer/
├── main.py          ← Run this. Entry point, streaming, HITL loop
├── graph.py         ← LangGraph assembly: nodes + edges + persistence + HITL
├── nodes.py         ← 6 specialist agent nodes (Planner, Researcher, Writer, ...)
├── state.py         ← EssayState TypedDict definition
├── tools.py         ← Tavily search tool configuration
├── requirements.txt ← All Python dependencies
├── .env.example     ← Template for API keys (rename to .env)
└── README.md        ← This file
```

---

## 🔄 Agent Pipeline

```
User types essay topic
        │
        ▼
  ┌─────────────┐
  │  plan_node  │  → Creates structured outline
  └──────┬──────┘
         │
         ▼
  ┌──────────────────┐
  │  research_node   │  → Calls Tavily 3x with targeted queries
  └──────┬───────────┘
         │
         ▼
  ┌─────────────┐
  │  draft_node │  → Writes full essay from plan + research
  └──────┬──────┘
         │
         ▼
  ┌──────────────────┐
  │  critique_node   │  → Gives specific numbered critique
  └──────┬───────────┘
         │
    ⏸️ PAUSE (Human-in-the-Loop)
    Human can view draft, add feedback, approve, or skip
         │
         ▼
  ┌────────────────────────────────────────────┐
  │  should_revise()  conditional edge         │
  │  revision_num < max_revisions → revision   │
  │  revision_num >= max_revisions → final     │
  └────────────────────────────────────────────┘
         │                    │
         ▼                    ▼
  ┌──────────────┐    ┌─────────────┐
  │ revision_node│    │  final_node │  → Light polish
  │ (loop back   │    └──────┬──────┘
  │  to critique)│           │
  └──────────────┘           ▼
                           END
                    Essay saved to .txt
```

---

## 💡 Tips

- **First run is slow** — model loading + web searches take ~30-60 seconds total
- **Save your session ID** — shown on startup. Enter it next time to resume
- **Rate limits** — if Groq throws a rate limit error, wait 60 seconds and retry
- **No internet?** — The research step will fail but the rest works fine offline
- **Change the model** — edit `graph.py → get_model()` to use `llama-3.1-8b-instant` (faster but less capable)

---

## 🐛 Common Issues

| Error | Fix |
|-------|-----|
| `GROQ_API_KEY not set` | Make sure `.env` exists (not `.env.example`) |
| `AuthenticationError` | Double-check your API key has no extra spaces |
| `RateLimitError` | Wait 60s and run again. Free tier has rate limits |
| `TavilyError` | Check your TAVILY_API_KEY. If expired, research step skips gracefully |
| `ModuleNotFoundError` | Run `pip install -r requirements.txt` again |
