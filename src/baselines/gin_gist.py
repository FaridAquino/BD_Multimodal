"""Comparativa Lado B (texto): índices nativos GIN/GiST + full-text.
OWNER: Ing. Texto (con apoyo del Tech Lead en el esquema).

Debe correr sobre EL MISMO texto que alimenta a SPIMI para que la comparación
sea válida. Mide latencia y memoria de `to_tsvector` + `@@` con índice GIN.
"""
from __future__ import annotations

from src.core import SearchResult
from src.db.connection import get_conn

# Índices nativos sobre chunks.tsv (creados en 03_indexes.sql)
INDICES_TSV = {"gin": "idx_chunks_tsv_gin", "gist": "idx_chunks_tsv_gist"}


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


SQL_FULLTEXT_AGGREGATED = f"""
    SELECT source_id, MAX(ts_rank(tsv, tq.q)) AS score
    FROM chunks, {_TSQUERY_OR}
    WHERE tsv @@ tq.q
      AND modality = 'text'
    GROUP BY source_id
    ORDER BY score DESC
    LIMIT %s
"""


class sesion_indice_forzado:
    """Context manager: conexión donde el planner SOLO puede usar el índice
    tsv pedido ('gin' o 'gist').

    Dentro de una transacción se hace DROP del índice rival (DROP INDEX es
    transaccional: el ROLLBACK final lo restaura) y se apaga el seq scan,
    de modo que a cualquier escala la consulta pase por el índice medido.
    Uso:
        with sesion_indice_forzado("gist") as conn:
            rows = conn.execute(SQL_FULLTEXT_AGGREGATED, (q, k)).fetchall()
    """

    def __init__(self, indice: str):
        if indice not in INDICES_TSV:
            raise ValueError(f"Índice desconocido: {indice!r} (usa 'gin' o 'gist')")
        self.indice = indice

    def __enter__(self):
        self._conn_cm = get_conn()
        self._conn = self._conn_cm.__enter__()
        rival = [n for key, n in INDICES_TSV.items() if key != self.indice][0]
        # psycopg abre una transacción implícita en el primer execute; el
        # rollback de __exit__ restaura el índice rival y el enable_seqscan.
        self._conn.execute("SET LOCAL enable_seqscan = off")
        self._conn.execute(f"DROP INDEX IF EXISTS {rival}")
        return self._conn

    def __exit__(self, *exc):
        try:
            self._conn.rollback()
        finally:
            self._conn_cm.__exit__(None, None, None)
        return False