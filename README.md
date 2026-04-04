# Veles implementing the idea of Agent Knowledge System

Personal AI assistant that routes queries to specialized agents, grounded in your Markdown notes. Available as a **CLI** and a **web UI**.

---

## Prerequisites

- **Python 3.12+**
- **[uv](https://docs.astral.sh/uv/getting-started/installation/)** — `curl -LsSf https://astral.sh/uv/install.sh | sh`
- An API key for at least one provider:

| Provider | Key | Notes |
|----------|-----|-------|
| **Cerebras** | `CEREBRAS_API_KEY` | Primary — fast, cheap Llama 3.1 |
| **Gemini** | `GEMINI_API_KEY` | Fallback + embeddings |

---

## Setup

```bash
# 1. Clone and enter the repo
git clone https://github.com/shojchi/aks.git && cd aks

# 2. Install dependencies
uv sync

# 3. Configure API keys
cp .env.example .env
# open .env and fill in CEREBRAS_API_KEY and/or GEMINI_API_KEY
```

---

## Web UI

```bash
# Start the server (default: http://127.0.0.1:8080)
uv run aks serve

# Options
uv run aks serve --port 3000           # custom port
uv run aks serve --reload              # auto-restart on code changes (dev)
uv run aks serve --host 0.0.0.0        # expose on your local network
```

Then open **http://127.0.0.1:8080** in your browser.

**What you can do in the UI:**
- Chat with your knowledge base — responses stream token-by-token
- Import a URL or PDF into the knowledge base via the Import button
- Search notes live from the left sidebar
- Monitor provider, cost, and agent status in the right panel

---

## CLI

```bash
# Install the `aks` shell alias (one-time)
bash scripts/install.sh
source ~/.zshrc   # or restart your terminal

# Ask a question
aks ask "why is my Python code slow?"

# Force a specific agent
aks ask --agent code "explain this decorator pattern"

# Interactive multi-turn chat
aks chat
aks chat --save   # prompt to save the session as a note on exit

# Knowledge base
aks save "Redis Caching" "Use SETEX for TTL-based caching..."
aks import https://example.com/article
aks import ~/Downloads/paper.pdf
aks list
aks search "caching"
aks rm redis-caching

# System
aks status        # show loaded agents, models, and config
aks cost          # today's token usage and spend vs daily cap
aks reindex       # force a full rebuild of the search index
```

If `aks` isn't on your PATH yet, prefix any command with `uv run`:

```bash
uv run aks serve
uv run aks ask "hello"
```

---

## Architecture

```
Query → Orchestrator (router) → Agent (code / pkm / writing / planning) → Response
                ↑
         Knowledge Store (SQLite FTS5 + ChromaDB vector)
```

**Routing:** keyword pre-filter → LLM classification → single agent or two-agent chain (e.g. `pkm → writing`)

**Phases:**
- ✅ Phase 1 — CLI + Code agent + keyword search
- ✅ Phase 2 — Hybrid search (FTS + ChromaDB vector embeddings)
- ✅ Phase 3 — All 4 agents + smart routing + multi-agent chains
- ✅ Phase 4 — Conversation memory, document import, cost tracking
- ✅ Phase 5 — Web UI (FastAPI + Jinja2 + HTMX + SSE streaming)

---

## Project Structure

```
config/              YAML configs (system, models, agent prompts)
src/aks/
  main.py            CLI entry point (Click)
  web/               FastAPI web app + Jinja2 templates
  orchestrator/      Intent classification and agent routing
  agents/            Specialist agents (code, pkm, writing, planning)
  retrieval/         Context assembly from knowledge store
  knowledge/         Note I/O and SQLite FTS5 index
  models/            LLM client (Cerebras + Gemini, streaming, fallback)
  utils/             Config loader, cost ledger
knowledge/
  notes/             Your Markdown notes — drop .md files here
  documents/         Imported PDFs
  conversations/     Archived chat sessions
  .index/            Auto-generated index (gitignored)
tests/
```

---

## Adding Notes

Drop any `.md` file into `knowledge/notes/`. YAML frontmatter is optional:

```markdown
---
title: My Note
tags: [python, performance]
---

Content here...
```

The index rebuilds automatically on the next query or server start.
