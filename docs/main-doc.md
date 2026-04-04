Perfect! I now have a complete understanding of the codebase. Let me create a comprehensive overview report.
---
Show less
## COMPREHENSIVE CODEBASE OVERVIEW: AKS (Agent Knowledge System) with name Veles in next both Veles and AKS could appears through the project and it refers the same entity.
### 1. DIRECTORY STRUCTURE (Complete File Listing)
```
/Users/roman.hordiichuk/Desktop/aks/.claude/worktrees/festive-goldberg/
├── .claude/
│   └── settings.local.json          (Claude Code permissions config)
├── .env.example                     (Environment variable template)
├── .git                             (Git file reference)
├── .gitignore                       (Version control exclusions)
├── README.md                        (Project documentation)
├── pyproject.toml                   (Python package config & dependencies)
│
├── config/                          (YAML configuration files — read at runtime)
│   ├── agents/
│   │   ├── code.yaml               (Code agent system prompt & keywords)
│   │   ├── pkm.yaml                (PKM agent system prompt & keywords)
│   │   ├── planning.yaml           (Planning agent system prompt & keywords)
│   │   └── writing.yaml            (Writing agent system prompt & keywords)
│   ├── models.yaml                 (LLM provider config, pricing)
│   └── system.yaml                 (Knowledge store, retrieval, cost control settings)
│
├── src/aks/                         (Main Python package)
│   ├── __init__.py
│   ├── main.py                     (CLI entry point — 377 lines, Click-based)
│   │
│   ├── agents/                     (Specialist agent classes)
│   │   ├── __init__.py
│   │   ├── base.py                 (BaseAgent abstract class)
│   │   ├── code_agent.py           (Code specialist — inherits from BaseAgent)
│   │   ├── pkm_agent.py            (Knowledge mgmt specialist)
│   │   ├── planning_agent.py       (Planning specialist)
│   │   └── writing_agent.py        (Writing specialist)
│   │
│   ├── orchestrator/               (Intent routing & multi-agent chaining)
│   │   ├── __init__.py
│   │   └── router.py               (Orchestrator class, routing logic)
│   │
│   ├── knowledge/                  (Note storage & retrieval)
│   │   ├── __init__.py
│   │   └── store.py                (KnowledgeStore — SQLite FTS5 + ChromaDB)
│   │
│   ├── retrieval/                  (Context assembly)
│   │   ├── __init__.py
│   │   └── search.py               (FTS + hybrid vector search)
│   │
│   ├── models/                     (LLM client abstraction)
│   │   ├── __init__.py
│   │   └── llm.py                  (Client for Cerebras, Gemini; streaming, fallback)
│   │
│   ├── utils/                      (Utilities)
│   │   ├── __init__.py
│   │   ├── config.py               (YAML config loader, paths)
│   │   └── cost.py                 (SQLite cost ledger)
│   │
│   └── config/                     (Copy of config/ for packaging)
│       ├── agents/
│       │   ├── code.yaml
│       │   ├── pkm.yaml
│       │   ├── planning.yaml
│       │   └── writing.yaml
│       ├── models.yaml
│       └── system.yaml
│
├── knowledge/                       (User knowledge base)
│   ├── notes/                      (Markdown note files)
│   │   └── .gitkeep
│   ├── documents/                  (Imported PDFs)
│   │   └── .gitkeep
│   ├── conversations/              (Archived chat sessions)
│   │   └── .gitkeep
│   └── .index/                     (SQLite + ChromaDB indexes — gitignored)
│
├── scripts/
│   ├── .gitkeep
│   └── install.sh                  (Shell alias setup for the `aks` CLI command)
│
└── tests/                           (Pytest test suite)
    ├── .gitkeep
    ├── test_store.py               (KnowledgeStore list/delete/round-trip)
    ├── test_streaming.py           (LLM streaming + BaseAgent.stream())
    ├── test_chaining.py            (Multi-agent routing & chaining)
    └── test_reindex.py             (Index sync, stale detection, reindex)
```
---
### 2. KEY FILES OVERVIEW
#### **Entry Points**
- **`src/aks/main.py`** — CLI application using Click. Commands:
  - `ask <query>` — route and get response
  - `chat` — interactive multi-turn conversation
  - `search <query>` — search notes
  - `save <title> <body>` — save a note
  - `import <url|file>` — import URL or PDF
  - `list [--filter]` — list all notes
  - `rm <slug>` — delete a note
  - `reindex` — rebuild search indexes
  - `status` — show agent & model config
  - `cost [--history N]` — show token usage & cost
#### **Configuration Files**
- **`pyproject.toml`** — Python project metadata & dependencies:
  - **Core:** google-genai, chromadb, click, pyyaml, python-dotenv, openai, trafilatura, lxml_html_clean, pypdf
  - **Dev:** pytest, pytest-asyncio
  - Entry point: `aks = "aks.main:cli"`
- **`config/system.yaml`** — System behavior:
  - Knowledge store paths
  - Retrieval settings (max chunks, FTS vs vector weights, embeddings enabled)
  - Cost cap ($5/day default)
  - Conversation settings
- **`config/models.yaml`** — LLM configuration:
  - **Provider:** Cerebras (primary) or Gemini (fallback)
  - **Models:** All agents use `llama3.1-8b` (Cerebras)
  - **Embeddings:** `gemini-embedding-001` (always Gemini)
  - **Pricing:** Input/output per million tokens
  - **Fallback chain:** Gemini as secondary provider
- **`config/agents/*.yaml`** — Individual agent configs (code, pkm, writing, planning):
  - System prompt template
  - Keywords for routing
  - Description for orchestrator
#### **No FastAPI Routes or HTML Templates**
- This is **NOT a web application**. It's a pure CLI tool.
- No REST API, no web server, no HTML/CSS files.
- Phase 5 (future) mentions web UI plans, but not yet implemented.
---
### 3. CORE ARCHITECTURE
#### **Agent System (4 specialist agents)**
```
BaseAgent (abstract class)
├── CodeAgent         (code, debugging, architecture)
├── PKMAgent          (knowledge retrieval, synthesis)
├── WritingAgent      (emails, docs, translations)
└── PlanningAgent     (task breakdown, scheduling)
```
Each agent:
- Inherits from `BaseAgent`
- Has a `name` attribute
- Receives `AgentMessage` with query, context, conversation history
- Returns `AgentResponse` with content, model used, sources cited
#### **Orchestrator (Intent Routing)**
- Uses Haiku (fast model) for deterministic intent classification
- Keyword pre-filter for common queries (faster)
- Falls back to LLM routing for complex intents
- Supports **multi-agent chains** (e.g., `code->writing`, `pkm->planning`)
#### **Knowledge Store (Dual-index)**
- **SQLite FTS5** — Fast keyword search with BM25 scoring
- **ChromaDB** — Vector embeddings for semantic search (opt-in)
- Hybrid retrieval via Reciprocal Rank Fusion
- Markdown notes with optional YAML frontmatter
- Auto-sync on init (detects new/edited/deleted files via mtime)
- Reindex command for force full re-scan
#### **LLM Client (Multi-provider with Fallback)**
- **Cerebras** (default) — Fast, cheap Llama 3.1 8B
- **Gemini** (fallback) — Google's multi-modal model
- Auto-fallback on 429 (quota exhausted)
- Cost tracking via SQLite ledger
- Daily cap enforcement ($5 default)
#### **Conversation History**
- Stored in `~/.local/share/aks/chat_history.jsonl`
- Resumes on next `chat` session (up to 40 messages)
- Can optionally save sessions as notes
---
### 4. DEPENDENCIES (from pyproject.toml)
**Runtime:**
- `google-genai>=1.0` — Gemini API client
- `chromadb>=0.6` — Vector database
- `click>=8.0` — CLI framework
- `pyyaml>=6.0` — Config parsing
- `python-dotenv>=1.0` — Environment variables
- `openai>=2.30.0` — OpenAI SDK (used for Cerebras)
- `trafilatura>=1.12` — URL text extraction
- `lxml_html_clean>=0.1` — HTML cleaning
- `pypdf>=4.0` — PDF text extraction
**Development:**
- `pytest>=8.0` — Test framework
- `pytest-asyncio>=0.23` — Async test support
**Build:**
- Hatchling (PEP 517 build backend)
---
### 5. KNOWLEDGE/DATA DIRECTORY
```
knowledge/
├── notes/          ← User's Markdown notes (source of truth)
├── documents/      ← Imported PDFs (Phase 4)
├── conversations/  ← Archived chat sessions (Phase 4)
└── .index/         ← Generated (gitignored)
    ├── fts.db      ← SQLite FTS5 full-text index
    ├── note_meta   ← mtime tracking for stale detection
    └── chroma/     ← ChromaDB vector store (if embeddings enabled)
```
Also stored in `~/.local/share/aks/`:
- `cost.db` — SQLite usage ledger
- `chat_history.jsonl` — Multi-turn conversation history
---
### 6. EXISTING FRONTEND CODE
**None.** This is a CLI-only application at present.
Future plans (Phase 5): FastAPI web UI with HTMX, but not yet implemented.
---
### 7. MAIN ENTRY POINTS
1. **CLI Entry:** `src/aks/main.py:cli`
   - Invoked via `uv run aks <command>` or `aks <command>` (after install.sh setup)
   - Decorated with `@click.group()` for multi-command CLI
2. **Orchestration Flow:**
   ```python
   main.py:ask() 
   → _get_orchestrator() 
   → Orchestrator.stream_chain() 
   → route_chain() → (keyword or LLM routing)
   → Agent.stream()
   ```
3. **Knowledge Store Initialization:**
   ```python
   KnowledgeStore(auto_sync=True)
   → _open_db() → SQLite FTS5
   → _open_chroma() → ChromaDB (if enabled)
   → _sync() → scan notes_dir, update indexes
   ```
---
### 8. PHASE ROADMAP (from README)
- **Phase 1 (current):** CLI + Code agent + keyword search ✓
- **Phase 2:** Hybrid search (FTS + ChromaDB vector) — embeddings config present
- **Phase 3:** All 4 agents + smart routing — agents coded but not yet Phase 1
- **Phase 4:** Conversation memory, document import, auto-capture — infra in place
- **Phase 5:** Web UI (FastAPI + HTMX) — not yet started
---
### 9. TESTING
Located in `/tests/`:
- **test_store.py** — KnowledgeStore operations (list, delete, round-trip, search)
- **test_streaming.py** — LLM streaming behavior, BaseAgent.stream()
- **test_chaining.py** — Multi-agent routing, chain parsing, orchestration
- **test_reindex.py** — Index sync, stale detection, mtime tracking
Run via: `pytest` or `uv run pytest`
---
### 10. CONFIGURATION & ENVIRONMENT
- **.env.example** → Copy to `.env`, add API keys:
  - `CEREBRAS_API_KEY=...`
  - `GEMINI_API_KEY=...`
  - `GROQ_API_KEY=...` (backup)
  - `MISTRAL_API_KEY=...` (backup)
- **AKS_HOME** env var — overrides config/data paths (for custom locations)
---
### 11. NOTABLE FEATURES
- **Cost Control:** Daily spend cap + per-provider usage tracking
- **Fallback:** Automatic provider fallback if quota exceeded
- **Streaming:** Real-time token-by-token output
- **Multi-turn:** Conversation memory across sessions
- **Source Attribution:** Retrieves and cites knowledge base sources
- **Auto-index Sync:** Detects file changes via mtime without full reindex
- **Hybrid Search:** Combines keyword (FTS) + semantic (vector) scoring
---
### SUMMARY
**AKS** is a polished **personal AI assistant CLI** (no web UI yet) that:
1. Routes queries to specialized agents (Code, PKM, Writing, Planning)
2. Grounds responses in your personal Markdown notes
3. Tracks costs with fallback to secondary LLM providers
4. Supports multi-turn conversation with persistent history
5. Uses SQLite + ChromaDB for fast, relevant context retrieval
No FastAPI, no HTML templates, no JavaScript. Pure Python CLI with Click, tested with pytest, configur