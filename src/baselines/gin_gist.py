"""Comparativa Lado B (texto): índices nativos GIN/GiST + full-text.
OWNER: Ing. Texto (con apoyo del Tech Lead en el esquema).

Debe correr sobre EL MISMO texto que alimenta a SPIMI para que la comparación
sea válida. Mide latencia y memoria de `to_tsvector` + `@@` con índice GIN.
"""
from __future__ import annotations

from src.core import SearchResult
from src.db.connection import get_conn


# plainto_tsquery une los términos con AND (&): con consultas largas (una estrofa
# entera) solo matchea el documento original. Convertimos a OR (|) para que el
# ranking de ts_rank sea comparable con el TF-IDF del Lado A, que puntúa por
# coincidencia parcial de términos.
_TSQUERY_OR = ("(SELECT replace(plainto_tsquery('english', %s)::text, "
               "'&', '|')::tsquery AS q) tq")


def search_fulltext(query: str, k: int = 10) -> list[SearchResult]:
    with get_conn() as conn:
        rows = conn.execute(
            f"""
            SELECT source_id, id, ts_rank(tsv, tq.q) AS score
            FROM chunks, {_TSQUERY_OR}
            WHERE tsv @@ tq.q
              AND modality = 'text'
            ORDER BY score DESC
            LIMIT %s
            """,
            (query, k),
        ).fetchall()

    return [
        SearchResult(source_id=row[0], score=float(row[2]), chunk_id=str(row[1]))
        for row in rows
    ]


def search_fulltext_aggregated(query: str, k: int = 10) -> list[SearchResult]:
    """Igual que search_fulltext pero agrega por source_id (documento).

    La agregación MAX por canción se hace en SQL: traer todos los chunks
    coincidentes a Python costaba ~10x más a escala 100k.
    """
    with get_conn() as conn:
        rows = conn.execute(
            f"""
            SELECT source_id, MAX(ts_rank(tsv, tq.q)) AS score
            FROM chunks, {_TSQUERY_OR}
            WHERE tsv @@ tq.q
              AND modality = 'text'
            GROUP BY source_id
            ORDER BY score DESC
            LIMIT %s
            """,
            (query, k),
        ).fetchall()

    return [SearchResult(source_id=sid, score=float(sc)) for sid, sc in rows]