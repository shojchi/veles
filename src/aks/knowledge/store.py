"""Markdown note store with SQLite FTS5 keyword search + ChromaDB vector index."""
from __future__ import annotations

import re
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from aks.utils.config import system_config, DATA_DIR


@dataclass
class Note:
    path: Path
    title: str
    body: str
    metadata: dict


@dataclass
class SearchResult:
    note: Note
    snippet: str
    score: float


@dataclass
class ReindexStats:
    added: int
    updated: int
    removed: int

    def __str__(self) -> str:
        return f"added={self.added}  updated={self.updated}  removed={self.removed}"


def _parse_note(path: Path) -> Note:
    text = path.read_text(encoding="utf-8")
    metadata: dict = {}
    body = text

    # Strip YAML frontmatter
    if text.startswith("---"):
        end = text.find("---", 3)
        if end != -1:
            try:
                metadata = yaml.safe_load(text[3:end]) or {}
            except yaml.YAMLError:
                pass
            body = text[end + 3:].lstrip()

    title = metadata.get("title") or path.stem.replace("-", " ").replace("_", " ").title()
    return Note(path=path, title=title, body=body, metadata=metadata)


class KnowledgeStore:
    def __init__(self, auto_sync: bool = True) -> None:
        cfg = system_config()
        self.notes_dir = DATA_DIR / cfg["notes_dir"]
        self.index_dir = DATA_DIR / cfg["index_dir"]
        self.embeddings_enabled: bool = cfg["retrieval"]["embeddings_enabled"]
        self.index_dir.mkdir(parents=True, exist_ok=True)
        self._db = self._open_db()
        self._chroma: Any = None
        if self.embeddings_enabled:
            self._chroma = self._open_chroma()
        if auto_sync:
            self._sync()

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def _open_db(self) -> sqlite3.Connection:
        db_path = self.index_dir / "fts.db"
        conn = sqlite3.connect(db_path)
        conn.execute(
            "CREATE VIRTUAL TABLE IF NOT EXISTS notes USING fts5("
            "path UNINDEXED, title, body, tokenize='porter unicode61')"
        )
        # Separate metadata table tracks mtime for stale-detection (FTS5 can't ALTER TABLE)
        conn.execute(
            "CREATE TABLE IF NOT EXISTS note_meta("
            "path TEXT PRIMARY KEY, mtime REAL NOT NULL)"
        )
        conn.commit()
        return conn

    def _open_chroma(self) -> Any:
        import chromadb
        client = chromadb.PersistentClient(path=str(self.index_dir / "chroma"))
        return client.get_or_create_collection(
            name="notes",
            metadata={"hnsw:space": "cosine"},
        )

    # ------------------------------------------------------------------
    # Sync: index new notes into FTS + ChromaDB
    # ------------------------------------------------------------------

    def _sync(self) -> ReindexStats:
        """Scan notes_dir and bring FTS + ChromaDB in sync with disk.

        - New files are added.
        - Files whose mtime changed are updated.
        - Index entries whose file no longer exists are removed.
        Returns counts of each action.
        """
        # Load current mtime map, full FTS index, and chroma IDs
        meta: dict[str, float] = {
            row[0]: row[1]
            for row in self._db.execute("SELECT path, mtime FROM note_meta").fetchall()
        }
        fts_indexed: set[str] = {
            row[0]
            for row in self._db.execute("SELECT path FROM notes").fetchall()
        }
        chroma_ids: set[str] = (
            set(self._chroma.get()["ids"]) if self._chroma else set()
        )

        disk_keys: set[str] = set()
        added = updated = 0

        for md_file in self.notes_dir.rglob("*.md"):
            key = str(md_file)
            disk_keys.add(key)
            mtime = md_file.stat().st_mtime
            known_mtime = meta.get(key)

            if known_mtime is None:
                # New file
                note = _parse_note(md_file)
                self._db.execute(
                    "INSERT INTO notes(path, title, body) VALUES (?, ?, ?)",
                    (key, note.title, note.body),
                )
                self._db.execute(
                    "INSERT INTO note_meta(path, mtime) VALUES (?, ?)", (key, mtime)
                )
                if self._chroma:
                    self._chroma.add(
                        ids=[key],
                        embeddings=[self._embed(note.title + "\n\n" + note.body)],
                        documents=[note.body],
                        metadatas=[{"title": note.title, "path": key}],
                    )
                added += 1

            elif mtime != known_mtime:
                # Edited file — update FTS, chroma, and mtime
                note = _parse_note(md_file)
                self._db.execute(
                    "UPDATE notes SET title=?, body=? WHERE path=?",
                    (note.title, note.body, key),
                )
                self._db.execute(
                    "UPDATE note_meta SET mtime=? WHERE path=?", (mtime, key)
                )
                if self._chroma:
                    if key in chroma_ids:
                        self._chroma.update(
                            ids=[key],
                            embeddings=[self._embed(note.title + "\n\n" + note.body)],
                            documents=[note.body],
                            metadatas=[{"title": note.title, "path": key}],
                        )
                    else:
                        self._chroma.add(
                            ids=[key],
                            embeddings=[self._embed(note.title + "\n\n" + note.body)],
                            documents=[note.body],
                            metadatas=[{"title": note.title, "path": key}],
                        )
                updated += 1

        # Remove orphaned index entries (file deleted externally)
        # Use fts_indexed as source of truth — meta may be empty after reindex()
        orphans = fts_indexed - disk_keys
        removed = len(orphans)
        for key in orphans:
            self._db.execute("DELETE FROM notes WHERE path=?", (key,))
            self._db.execute("DELETE FROM note_meta WHERE path=?", (key,))
            if self._chroma and key in chroma_ids:
                self._chroma.delete(ids=[key])

        self._db.commit()
        return ReindexStats(added=added, updated=updated, removed=removed)

    def reindex(self) -> ReindexStats:
        """Force a full re-scan of all notes regardless of mtime.

        Clears the mtime cache so every file is treated as new/changed,
        then delegates to _sync().
        """
        self._db.execute("DELETE FROM note_meta")
        self._db.commit()
        return self._sync()

    def _embed(self, text: str) -> list[float]:
        from aks.models.llm import get_embedding
        from aks.utils.config import models_config
        embed_provider = models_config().get("embeddings", {}).get("provider", "gemini")
        return get_embedding(text, provider=embed_provider)

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search(self, query: str, limit: int = 10) -> list[SearchResult]:
        """Keyword search via SQLite FTS5."""
        safe_query = re.sub(r'[^\w\s]', ' ', query).strip()
        if not safe_query:
            return []
        fts_query = " OR ".join(safe_query.split())

        rows = self._db.execute(
            "SELECT path, snippet(notes, 2, '[', ']', '...', 16), "
            "bm25(notes) AS score "
            "FROM notes WHERE notes MATCH ? "
            "ORDER BY score LIMIT ?",
            (fts_query, limit),
        ).fetchall()

        results = []
        for path_str, snippet, score in rows:
            note = _parse_note(Path(path_str))
            results.append(SearchResult(note=note, snippet=snippet, score=abs(score)))
        return results

    def vector_search(self, query: str, limit: int = 10) -> list[SearchResult]:
        """Semantic search via ChromaDB."""
        if not self._chroma or self._chroma.count() == 0:
            return []
        n = min(limit, self._chroma.count())
        results = self._chroma.query(
            query_embeddings=[self._embed(query)],
            n_results=n,
            include=["documents", "metadatas", "distances"],
        )
        out = []
        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            note = _parse_note(Path(meta["path"]))
            # cosine distance [0, 2] → similarity score [0, 1]
            score = max(0.0, 1.0 - dist / 2.0)
            out.append(SearchResult(note=note, snippet=doc[:300], score=score))
        return out

    # ------------------------------------------------------------------
    # List
    # ------------------------------------------------------------------

    def list_notes(self) -> list[Note]:
        """Return all indexed notes sorted by modification time (newest first)."""
        rows = self._db.execute("SELECT path FROM notes").fetchall()
        notes = []
        for (path_str,) in rows:
            p = Path(path_str)
            if p.exists():
                notes.append(_parse_note(p))
        notes.sort(key=lambda n: n.path.stat().st_mtime, reverse=True)
        return notes

    # ------------------------------------------------------------------
    # Delete
    # ------------------------------------------------------------------

    def delete_note(self, path: Path) -> None:
        """Remove a note from disk, FTS index, and ChromaDB."""
        key = str(path)

        self._db.execute("DELETE FROM notes WHERE path = ?", (key,))
        self._db.commit()

        if self._chroma:
            existing = self._chroma.get(ids=[key])
            if existing["ids"]:
                self._chroma.delete(ids=[key])

        if path.exists():
            path.unlink()

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def save_note(self, title: str, body: str, metadata: dict | None = None) -> Path:
        """Write a note to disk and index it in FTS + ChromaDB."""
        slug = re.sub(r"[^\w\s-]", "", title.lower()).strip()
        slug = re.sub(r"[\s_]+", "-", slug)
        path = self.notes_dir / f"{slug}.md"

        meta = metadata or {}
        meta.setdefault("title", title)
        frontmatter = yaml.dump(meta, allow_unicode=True).strip()
        path.write_text(f"---\n{frontmatter}\n---\n\n{body}\n", encoding="utf-8")

        self._db.execute(
            "INSERT INTO notes(path, title, body) VALUES (?, ?, ?)",
            (str(path), title, body),
        )
        self._db.execute(
            "INSERT OR REPLACE INTO note_meta(path, mtime) VALUES (?, ?)",
            (str(path), path.stat().st_mtime),
        )
        self._db.commit()

        if self._chroma:
            self._chroma.add(
                ids=[str(path)],
                embeddings=[self._embed(title + "\n\n" + body)],
                documents=[body],
                metadatas=[{"title": title, "path": str(path)}],
            )

        return path
