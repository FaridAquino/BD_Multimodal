"""Comparativa Lado B (imagen/audio): pgvector con índice HNSW o IVF.
OWNER: Ing. Imágenes + Ing. Audio (esquema: Tech Lead).

Almacena los MISMOS histogramas como vectores y busca por similitud con `<->`.
"""
from __future__ import annotations

from pgvector.psycopg import register_vector
from src.core import SearchResult
from src.db.connection import get_conn


def search_vector(modality: str, query_vec, k: int = 10) -> list[SearchResult]:
    # Convertimos el diccionario sparse a un vector denso de 512 elementos
    dense_vector = [0.0] * 512
    for cw, count in query_vec.counts.items():
        idx = int(cw)
        if 0 <= idx < 512:
            dense_vector[idx] = float(count)

    # Conectamos a la BD usando el patrón del repositorio
    with get_conn() as conn:
        register_vector(conn)
        table_name = f"embeddings_{modality}"
        
        query = f"""
            SELECT chunk_id, source_id, 1 - (embedding <=> %s::vector) AS score
            FROM {table_name}
            ORDER BY embedding <=> %s::vector
            LIMIT %s
        """
        rows = conn.execute(query, (dense_vector, dense_vector, k)).fetchall()
        
    return [
        SearchResult(chunk_id=str(row[0]), source_id=str(row[1]), score=float(row[2]))
        for row in rows
    ]
