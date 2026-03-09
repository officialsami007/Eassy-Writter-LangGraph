
# 📝 AI Essay Writer — LangGraph Multi-Agent System

<div align="center">

![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=for-the-badge&logo=python&logoColor=white)
![LangGraph](https://img.shields.io/badge/LangGraph-1.0-FF6B6B?style=for-the-badge)
![Flask](https://img.shields.io/badge/Flask-3.0-000000?style=for-the-badge&logo=flask&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-Containerized-2496ED?style=for-the-badge&logo=docker&logoColor=white)
![Render](https://img.shields.io/badge/Deployed-Render-46E3B7?style=for-the-badge)

**A production-ready, full-stack AI application that uses a multi-agent pipeline to research, write, critique, and revise essays — with a real-time streaming web interface and Human-in-the-Loop review.**

[🚀 Live Demo](https://eassy-writter-langgraph.onrender.com) · [📖 How It Works](#how-it-works) · [⚙️ Local Setup](#local-setup) · [🏗️ Architecture](#architecture)

</div>

---

## Overview

AI Essay Writer is a full-stack AI application that goes beyond a simple prompt-and-response pattern. It breaks the essay writing process into discrete, expert stages — each handled by a dedicated AI agent with its own role, instructions, and responsibility.

The pipeline moves from research to planning, drafting, critical review, revision, and final polish — automatically. At each critique stage, the process pauses and hands control back to the user, who can read the agent's feedback, add their own notes, and decide whether to approve another revision or jump straight to the final version.

Every agent interaction streams live to the browser in real time, so users can follow exactly what is happening at each stage rather than waiting for a single bulk response. Completed essays are saved persistently and can be reloaded, downloaded, or deleted at any time from the history panel.

---

## Live Demo

🌐 **[https://eassy-writter-langgraph.onrender.com](https://eassy-writter-langgraph.onrender.com)**

> The app is hosted on Render's free tier. If the page takes ~30 seconds to load, the server is waking from sleep — this is normal for free-tier hosting.

---

## Features

- **6-Agent Pipeline** — Planner → Researcher → Writer → Critic → Reviser → Finaliser, each a specialist LLM node
- **Real-Time Streaming** — Live updates via SSE; watch each agent complete its task in the browser as it happens
- **Agentic Web Search** — The Researcher agent queries Tavily to gather real, up-to-date sources for every essay
- **Human-in-the-Loop** — Pipeline pauses after every critique cycle; user can approve, add their own feedback, or skip to final
- **Configurable Revision Cycles** — Set 1–5 revision loops; the agent graph enforces the limit via conditional edges
- **Persistent History** — Every completed essay is saved to disk with full Plan, Research, Critique, and Draft — reloadable at any time
- **Essay Management** — View, reload, download (.txt), and delete essays from a slide-out history drawer
- **Rich Content Rendering** — Plan tab renders structured outlines with Roman numeral sections; Critique tab formats numbered issues into styled cards with extracted suggestions
- **Dockerized** — Fully containerized for consistent local development and cloud deployment

---

## How It Works

### The Agent Pipeline

```
User Input (topic)
       │
       ▼
  ┌─────────┐
  │ Planner │  Creates a structured essay outline
  └────┬────┘
       │
       ▼
  ┌────────────┐
  │ Researcher │  Searches the web via Tavily API (3 targeted queries)
  └─────┬──────┘
        │
        ▼
  ┌────────┐
  │ Writer │  Writes a full draft using the plan + research
  └────┬───┘
       │
       ▼
  ┌─────────┐
  │  Critic │  Reviews the draft on 5 dimensions: thesis, evidence,
  └────┬────┘  flow, writing quality, conclusion strength
       │
       ▼
  ⏸ HUMAN-IN-THE-LOOP PAUSE
  User reviews critique → optionally adds feedback → approves or skips
       │
       ▼
  ┌─────────┐
  │ Reviser │  Rewrites based on critique + human feedback
  └────┬────┘
       │
       └──── loops back to Critic (up to max_revisions times)
       │
       ▼
  ┌───────────┐
  │ Finaliser │  Light grammar + flow polish pass
  └─────┬─────┘
        │
        ▼
   Final Essay ✓
```

### LangGraph Concepts Applied

| Concept | Implementation |
|---|---|
| **TypedDict State** | `EssayState` in `state.py` — shared memory between all agents |
| **Nodes** | One Python function per agent in `nodes.py` |
| **Conditional Edges** | `should_revise()` checks `revision_num` vs `max_revisions` to loop or end |
| **Persistence** | `SqliteSaver` checkpointer — full state saved after every node |
| **Human-in-the-Loop** | `interrupt_before` in `compile()` — graph pauses, user injects via `update_state()` |
| **Streaming** | `graph.stream()` with `stream_mode="updates"` piped to SSE |
| **Time Travel** | SQLite checkpoints allow full state history per `thread_id` |

---

## Tech Stack

| Layer | Technology |
|---|---|
| **Agent Framework** | LangGraph 1.x |
| **LLM** | Groq API (`llama-3.3-70b-versatile`) — free tier |
| **Web Search** | Tavily API — free tier (1,000 searches/month) |
| **Backend** | Flask 3.0, Python 3.11 |
| **Streaming** | Server-Sent Events (SSE) |
| **Persistence** | SQLite (LangGraph checkpointer) + JSON (essay history) |
| **Frontend** | Vanilla HTML/CSS/JS — no framework |
| **Containerization** | Docker |
| **Hosting** | Render (free tier) |

---

## Architecture

```
essay_writer/
│
├── app.py              # Flask server — REST API + SSE streaming endpoints
├── graph.py            # LangGraph graph assembly — nodes, edges, HITL, persistence
├── nodes.py            # 6 agent functions with system prompts
├── state.py            # EssayState TypedDict — shared state schema
├── tools.py            # Tavily search tool configuration
│
├── templates/
│   └── index.html      # Single-page frontend — tabs, live log, HITL panel, history
│
├── Dockerfile          # Container definition
├── docker-compose.yml  # Local development shortcut
├── requirements.txt    # Python dependencies
└── .env.example        # API key template
```

### Key API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/api/start` | Start a new essay run, returns `thread_id` |
| `GET` | `/api/stream/<tid>` | SSE stream of live agent events |
| `POST` | `/api/feedback/<tid>` | Submit human feedback, resume graph |
| `POST` | `/api/skip/<tid>` | Skip to final agent |
| `GET` | `/api/state/<tid>` | Fetch full current state snapshot |
| `GET` | `/api/essays` | List all saved essays |
| `GET` | `/api/essays/<tid>` | Load a specific saved essay |
| `DELETE` | `/api/essays/<tid>` | Delete an essay from history |
| `GET` | `/api/download/<tid>` | Download essay as `.txt` |

---

## Local Setup

### Prerequisites
- Python 3.10+
- A free [Groq API key](https://console.groq.com) — for the LLM
- A free [Tavily API key](https://app.tavily.com) — for web search

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/officialsami007/essay-writer.git
cd essay-writer

# 2. Create and activate a virtual environment
python -m venv venv

# Windows
venv\Scripts\activate

# Mac/Linux
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt
pip install langgraph-checkpoint-sqlite

# 4. Add your API keys
cp .env.example .env
# Edit .env and fill in your GROQ_API_KEY and TAVILY_API_KEY

# 5. Run the app
python app.py
```

Open **http://localhost:5000** in your browser.

### Running with Docker

```bash
# Build the image
docker build -t essay-writer .

# Run with your API keys
docker run -p 5000:5000 --env-file .env essay-writer

# Or using docker-compose
docker compose up
```

---

## Deployment

This app is deployed on **Render** using Docker. To deploy your own instance:

1. Fork this repository
2. Create a free account at [render.com](https://render.com)
3. New → Web Service → connect your forked repo
4. Render auto-detects the `Dockerfile`
5. Add environment variables: `GROQ_API_KEY` and `TAVILY_API_KEY`
6. Deploy — your app will be live at a `*.onrender.com` URL

---

## Background — Course Foundation

This project is a practical implementation of concepts from the **DeepLearning.AI short courses**:

- [AI Agents in LangGraph](https://learn.deeplearning.ai/accomplishments/38673e7d-c1c1-4535-aa48-0bc38e9018ae?usp=sharing) — core architecture
- [LangChain for LLM Application Development](https://learn.deeplearning.ai/accomplishments/80b37037-d006-4bbc-b890-00f2e87b2f57?usp=sharing) — chains and prompts
---

<div align="center">
Built with LangGraph · Groq · Tavily · Flask
</div>
