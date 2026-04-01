"""Assemble retrieved context — FTS keyword search or hybrid FTS + vector (Phase 2)."""
from __future__ import annotations

from aks.knowledge.store import KnowledgeStore, SearchResult
from aks.utils.config import system_config


def retrieve_context(query: str, store: KnowledgeStore) -> str:
    """Return a formatted context block from the knowledge store."""
    cfg = system_config()["retrieval"]
    limit = cfg["max_chunks"]

    if cfg["embeddings_enabled"]:
        results = _hybrid_search(query, store, limit, cfg["fts_weight"], cfg["vector_weight"])
    else:
        results = store.search(query, limit=limit)

    if not results:
        return ""

    lines = ["## Retrieved Knowledge\n"]
    for i, r in enumerate(results, 1):
        lines.append(f"### [{i}] {r.note.title}")
        lines.append(f"*Source: {r.note.path.name} | relevance: {r.score:.2f}*\n")
        lines.append(r.snippet)
        lines.append("")

    return "\n".join(lines)


def _hybrid_search(
    query: str,
    store: KnowledgeStore,
    limit: int,
    fts_weight: float,
    vector_weight: float,
) -> list[SearchResult]:
    """Reciprocal Rank Fusion over FTS and vector results."""
    fts_results = store.search(query, limit=limit)
    vector_results = store.vector_search(query, limit=limit)

    scores: dict[str, float] = {}
    result_map: dict[str, SearchResult] = {}

    for rank, r in enumerate(fts_results):
        key = str(r.note.path)
        scores[key] = scores.get(key, 0.0) + fts_weight * (1.0 / (rank + 1))
        result_map[key] = r

    for rank, r in enumerate(vector_results):
        key = str(r.note.path)
        scores[key] = scores.get(key, 0.0) + vector_weight * (1.0 / (rank + 1))
        if key not in result_map:
            result_map[key] = r

    sorted_keys = sorted(scores, key=lambda k: scores[k], reverse=True)[:limit]
    return [
        SearchResult(
            note=result_map[k].note,
            snippet=result_map[k].snippet,
            score=scores[k],
        )
        for k in sorted_keys
    ]
