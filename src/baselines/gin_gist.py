from __future__ import annotations
from collections import defaultdict

from src.core import SearchResult
from src.db.connection import get_conn


def search_fulltext(query: str, k: int = 10) -> list[SearchResult]:
    with get_conn() as conn:
        rows = conn.execute(
            """
            SELECT source_id, id, ts_rank(tsv, plainto_tsquery('english', %s)) AS score
            FROM chunks
            WHERE tsv @@ plainto_tsquery('english', %s)
              AND modality = 'text'
            ORDER BY score DESC
            LIMIT %s
            """,
            (query, query, k),
        ).fetchall()

    return [
        SearchResult(source_id=row[0], score=float(row[2]), chunk_id=str(row[1]))
        for row in rows
    ]


def search_fulltext_aggregated(query: str, k: int = 10) -> list[SearchResult]:
    """Igual que search_fulltext pero agrega por source_id (documento)."""
    with get_conn() as conn:
        rows = conn.execute(
            """
            SELECT source_id, id, ts_rank(tsv, plainto_tsquery('english', %s)) AS score
            FROM chunks
            WHERE tsv @@ plainto_tsquery('english', %s)
              AND modality = 'text'
            ORDER BY score DESC
            """,
            (query, query),
        ).fetchall()

    acc: dict[str, float] = defaultdict(float)
    for row in rows:
        acc[row[0]] = max(acc[row[0]], float(row[2]))

    ranked = sorted(acc.items(), key=lambda x: x[1], reverse=True)
    return [SearchResult(source_id=sid, score=sc) for sid, sc in ranked[:k]]