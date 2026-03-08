# ── Base image: lightweight Python 3.11 ──────────────────────────
FROM python:3.11-slim

# ── Set working directory inside the container ───────────────────
WORKDIR /app

# ── Copy requirements first (better Docker layer caching) ────────
COPY requirements.txt .

# ── Install dependencies ──────────────────────────────────────────
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir langgraph-checkpoint-sqlite

# ── Copy all project files ────────────────────────────────────────
COPY . .

# ── Create a folder for persistent data (essays + SQLite DB) ─────
RUN mkdir -p /app/data

# ── Tell Flask to run on port 5000 ───────────────────────────────
EXPOSE 5000

# ── Start the app ─────────────────────────────────────────────────
CMD ["python", "app.py"]