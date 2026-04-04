"""FastAPI web application for AKS — Phase 5."""
from __future__ import annotations

import asyncio
import html as html_lib
import ipaddress
import json
import queue
import socket
import threading
import urllib.request
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import AsyncIterator
from urllib.parse import urljoin, urlparse

from fastapi import Cookie, FastAPI, Form, Request, Response
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from sse_starlette.sse import EventSourceResponse

TEMPLATES_DIR = Path(__file__).parent / "templates"

# ---------------------------------------------------------------------------
# SSRF guard
# ---------------------------------------------------------------------------

def _is_public_host(hostname: str) -> bool:
    """Return True only if every resolved IP for *hostname* is a public unicast address."""
    try:
        infos = socket.getaddrinfo(hostname, None)
    except socket.gaierror:
        return False
    if not infos:
        return False
    for info in infos:
        addr = info[4][0]
        try:
            ip = ipaddress.ip_address(addr)
        except ValueError:
            return False
        if (
            ip.is_loopback
            or ip.is_private
            or ip.is_link_local
            or ip.is_multicast
            or ip.is_unspecified
            or ip.is_reserved
        ):
            return False
    return True


def _validate_import_url(raw: str) -> str | None:
    """Validate *raw* for use as an import URL.

    Returns a normalised URL string on success, or None if the URL should be
    rejected.  Checks:
      - scheme must be http or https
      - hostname must resolve exclusively to public IP addresses (SSRF guard)
    """
    try:
        parsed = urlparse(raw)
    except Exception:  # noqa: BLE001
        return None
    if parsed.scheme not in {"http", "https"}:
        return None
    hostname = parsed.hostname
    if not hostname:
        return None
    if not _is_public_host(hostname):
        return None
    return raw


# Suppress urllib's automatic redirect-following so we can validate each hop.
class _NoAutoRedirect(urllib.request.HTTPErrorProcessor):
    def http_response(self, request, response):  # type: ignore[override]
        return response

    https_response = http_response


def _safe_fetch(url: str, max_redirects: int = 5) -> str | None:
    """Fetch *url* following only validated redirects (SSRF-safe).

    Validates the scheme and resolved IPs at every hop before connecting.
    Returns the decoded response body on HTTP 200, or None on any failure.
    """
    current = url
    for _ in range(max_redirects + 1):
        if not _validate_import_url(current):
            return None
        req = urllib.request.Request(
            current,
            headers={"User-Agent": "AKS/1.0 (import)"},
        )
        try:
            opener = urllib.request.build_opener(_NoAutoRedirect)
            with opener.open(req, timeout=15) as resp:
                status = resp.status
                if status in (301, 302, 303, 307, 308):
                    location = resp.headers.get("Location", "")
                    if not location:
                        return None
                    current = urljoin(current, location)
                    continue
                if status == 200:
                    raw = resp.read(5 * 1024 * 1024)  # 5 MB cap
                    charset = resp.headers.get_content_charset() or "utf-8"
                    return raw.decode(charset, errors="replace")
                return None
        except Exception:  # noqa: BLE001
            return None
    return None  # exceeded max_redirects
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

app = FastAPI(title="AKS", docs_url=None, redoc_url=None)

# session_id → conversation history (in-memory; single-worker only)
_sessions: dict[str, list[dict]] = {}

# task_id → {"queue": SimpleQueue, "session_id": str}
_tasks: dict[str, dict] = {}

# session_id → task_id of the one in-flight request (serialises chat per session)
_session_active: dict[str, str] = {}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_orchestrator():
    from aks.models.llm import get_client
    from aks.knowledge.store import KnowledgeStore
    from aks.orchestrator.router import Orchestrator
    from aks.utils.config import get_provider

    client = get_client(get_provider())
    store = KnowledgeStore()
    return Orchestrator(client=client, store=store)


def _note_age(path: Path) -> str:
    try:
        mtime = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
        delta = datetime.now(timezone.utc) - mtime
        if delta.days == 0:
            h = delta.seconds // 3600
            return f"{h}h ago" if h > 0 else "just now"
        if delta.days == 1:
            return "yesterday"
        if delta.days < 30:
            return f"{delta.days}d ago"
        return mtime.strftime("%b %d")
    except OSError:
        return ""


def _cost_context() -> dict:
    from aks.utils.cost import CostLedger
    from aks.utils.config import system_config

    ledger = CostLedger()
    today_cost = ledger.today_usd()
    cap = system_config()["cost"]["daily_cap_usd"]
    cap_pct = min(today_cost / cap * 100, 100) if cap > 0 else 0.0
    return {"today_cost": today_cost, "cap": cap, "cap_pct": cap_pct}


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/", response_class=HTMLResponse)
async def index(request: Request, session_id: str = Cookie(default="")):
    from aks.knowledge.store import KnowledgeStore
    from aks.utils.config import get_provider

    if not session_id:
        session_id = str(uuid.uuid4())

    store = KnowledgeStore()
    raw_notes = store.list_notes()
    notes = [{"note": n, "age": _note_age(n.path)} for n in raw_notes]

    history = _sessions.get(session_id, [])

    ctx = {
        "notes": notes,
        "history": history,
        "provider": get_provider().upper(),
        **_cost_context(),
    }
    resp = templates.TemplateResponse(request, "index.html", ctx)
    resp.set_cookie("session_id", session_id, httponly=True, samesite="lax")
    return resp


@app.get("/notes", response_class=HTMLResponse)
async def notes_list(request: Request, q: str = ""):
    from aks.knowledge.store import KnowledgeStore

    store = KnowledgeStore()
    raw_notes = store.list_notes()
    if q:
        raw_notes = [n for n in raw_notes if q.lower() in n.title.lower()]
    notes = [{"note": n, "age": _note_age(n.path)} for n in raw_notes]
    return templates.TemplateResponse(
        request, "partials/note_list.html", {"notes": notes, "q": q}
    )


@app.post("/chat", response_class=HTMLResponse)
async def chat_post(
    request: Request,
    message: str = Form(...),
    session_id: str = Cookie(default=""),
):
    if not session_id:
        session_id = str(uuid.uuid4())

    # Reject concurrent submits for the same session to prevent history corruption.
    if session_id in _session_active:
        return HTMLResponse(
            '<p class="font-label text-[10px] uppercase tracking-wider text-outline/50 text-center px-4 py-6">'
            "Still thinking — please wait…</p>",
            status_code=429,
        )

    # Snapshot history before adding user message (don't include it in the
    # conversation_history passed to the model — the agent adds it itself).
    history_snapshot = list(_sessions.get(session_id, []))

    # Append user turn immediately so page reload shows it.
    _sessions[session_id] = history_snapshot + [{"role": "user", "content": message}]

    task_id = str(uuid.uuid4())
    q: queue.SimpleQueue = queue.SimpleQueue()
    _tasks[task_id] = {"queue": q, "session_id": session_id}
    _session_active[session_id] = task_id

    def _run() -> None:
        try:
            orch = _make_orchestrator()
            chain, _model, chunks, sources = orch.stream_chain(
                message, conversation_history=history_snapshot
            )
            chain_str = " → ".join(chain)
            full = ""
            for chunk in chunks:
                full += chunk
                q.put(("chunk", full))
            # Persist assistant reply atomically after the full response is known.
            _sessions[session_id].append({"role": "assistant", "content": full})
            q.put(("done", {"content": full, "chain": chain_str, "sources": sources}))
        except Exception as exc:  # noqa: BLE001
            q.put(("error", str(exc)))
        finally:
            _session_active.pop(session_id, None)

    threading.Thread(target=_run, daemon=True).start()

    return templates.TemplateResponse(
        request, "partials/chat_turn.html", {"message": message, "task_id": task_id}
    )


@app.get("/chat/stream/{task_id}")
async def chat_stream(task_id: str, request: Request):
    task = _tasks.get(task_id)
    if not task:
        # Task already finished or unknown — tell the client not to reconnect.
        return Response(
            status_code=200,
            media_type="text/event-stream",
            content="retry: 86400000\nevent: done\ndata: {}\n\n",
        )

    q: queue.SimpleQueue = task["queue"]

    async def generator() -> AsyncIterator[dict]:
        # Push retry far into the future to suppress reconnect storms.
        yield {"retry": 86_400_000}

        while True:
            if await request.is_disconnected():
                _tasks.pop(task_id, None)
                _session_active.pop(task.get("session_id"), None)
                return

            try:
                event_type, data = q.get_nowait()
            except queue.Empty:
                await asyncio.sleep(0.04)
                continue

            if event_type == "chunk":
                # Send accumulated text as escaped HTML inside a <pre> wrapper.
                safe = html_lib.escape(data)
                yield {
                    "event": "chunk",
                    "data": f'<span class="whitespace-pre-wrap">{safe}</span>',
                }

            elif event_type == "done":
                yield {"event": "done", "data": json.dumps(data)}
                _tasks.pop(task_id, None)
                return

            elif event_type == "error":
                yield {"event": "error", "data": html_lib.escape(data)}
                _tasks.pop(task_id, None)
                return

    return EventSourceResponse(generator())


@app.post("/import", response_class=HTMLResponse)
async def import_url(request: Request, url: str = Form(...)):
    from aks.knowledge.store import KnowledgeStore
    import trafilatura

    if not _validate_import_url(url):
        return HTMLResponse(
            '<p class="text-xs text-error font-label uppercase px-3 py-2">Invalid or disallowed URL.</p>',
            status_code=422,
        )

    store = KnowledgeStore()
    # Use _safe_fetch instead of trafilatura.fetch_url — the latter follows
    # redirects internally without validating each hop (SSRF via open redirect).
    downloaded = _safe_fetch(url)
    if not downloaded:
        return HTMLResponse(
            '<p class="text-xs text-error font-label uppercase px-3 py-2">Could not fetch URL.</p>',
            status_code=422,
        )
    body = trafilatura.extract(downloaded, include_comments=False, include_tables=True)
    if not body:
        return HTMLResponse(
            '<p class="text-xs text-error font-label uppercase px-3 py-2">No readable text found.</p>',
            status_code=422,
        )
    meta = trafilatura.extract_metadata(downloaded)
    title = (meta.title if meta and meta.title else url.split("/")[-1] or "Imported")
    store.save_note(title=title, body=body, metadata={"source": url})

    # Return a fresh note list.
    raw_notes = store.list_notes()
    notes = [{"note": n, "age": _note_age(n.path)} for n in raw_notes]
    return templates.TemplateResponse(
        request, "partials/note_list.html", {"notes": notes, "q": ""}
    )


@app.post("/import/file", response_class=HTMLResponse)
async def import_file(request: Request, file: bytes = Form(...)):
    # UploadFile is cleaner — re-declare properly
    from fastapi import UploadFile, File as FastAPIFile
    return HTMLResponse(
        '<p class="text-xs text-outline/50 font-label uppercase px-3 py-2">Use the UploadFile endpoint.</p>',
        status_code=501,
    )


@app.post("/import/pdf", response_class=HTMLResponse)
async def import_pdf_file(request: Request):
    from fastapi import UploadFile, File as FastAPIFile
    from aks.knowledge.store import KnowledgeStore
    from pypdf import PdfReader
    import io

    form = await request.form()
    upload = form.get("file")
    if not upload or not hasattr(upload, "read"):
        return HTMLResponse(
            '<p class="text-xs text-error font-label uppercase px-3 py-2">No file received.</p>',
            status_code=422,
        )

    data = await upload.read()
    reader = PdfReader(io.BytesIO(data))
    base_title = upload.filename.rsplit(".", 1)[0].replace("-", " ").replace("_", " ").title()

    CHUNK_CHARS = 8000
    chunks: list[str] = []
    current = ""
    for page in reader.pages:
        text = page.extract_text() or ""
        if len(current) + len(text) > CHUNK_CHARS and current:
            chunks.append(current.strip())
            current = text
        else:
            current += "\n" + text
    if current.strip():
        chunks.append(current.strip())

    if not chunks:
        return HTMLResponse(
            '<p class="text-xs text-error font-label uppercase px-3 py-2">No text extracted.</p>',
            status_code=422,
        )

    store = KnowledgeStore()
    if len(chunks) == 1:
        store.save_note(title=base_title, body=chunks[0], metadata={"source": upload.filename})
    else:
        for i, chunk in enumerate(chunks):
            store.save_note(
                title=f"{base_title} (part {i + 1})",
                body=chunk,
                metadata={"source": upload.filename},
            )

    raw_notes = store.list_notes()
    notes = [{"note": n, "age": _note_age(n.path)} for n in raw_notes]
    return templates.TemplateResponse(
        request, "partials/note_list.html", {"notes": notes, "q": ""}
    )


@app.get("/note/{slug}", response_class=HTMLResponse)
async def note_detail(request: Request, slug: str):
    from aks.knowledge.store import KnowledgeStore

    store = KnowledgeStore()
    note = next((n for n in store.list_notes() if n.path.stem == slug), None)
    if not note:
        return HTMLResponse(
            '<p class="text-xs text-error font-label uppercase px-3 py-2">Note not found.</p>',
            status_code=404,
        )
    return templates.TemplateResponse(
        request,
        "partials/note_detail.html",
        {"note": note, "age": _note_age(note.path)},
    )


@app.get("/status", response_class=HTMLResponse)
async def status_panel(request: Request):
    from aks.utils.config import get_provider

    return templates.TemplateResponse(
        request,
        "partials/status_panel.html",
        {"provider": get_provider().upper(), **_cost_context()},
    )
