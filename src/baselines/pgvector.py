"""Comparativa Lado B (imagen/audio): pgvector con índice HNSW o IVF.
OWNER: Ing. Imágenes + Ing. Audio (esquema: Tech Lead).

Almacena los MISMOS histogramas como vectores y busca por similitud con `<->`.
"""
from __future__ import annotations

from pgvector.psycopg import register_vector
from src.core import SearchResult
from src.db.connection import get_conn


def search_vector(modality: str, query_vec, k: int = 10) -> list[SearchResult]:
    # 1. Definir la dimensión correcta según la modalidad (= k del codebook)
    if modality == "audio":
        dim = 256
    else:
        dim = 256

    # 2. Convertimos el diccionario sparse a un vector denso usando la variable 'dim'
    dense_vector = [0.0] * dim
    for cw, count in query_vec.counts.items():
        idx = int(cw)
        if 0 <= idx < dim:
            dense_vector[idx] = float(count)

    # Consulta sin codewords: la distancia coseno con vector nulo no está
    # definida (NaN); devolvemos vacío.
    if not query_vec.counts:
        return []

    # Chunks guardados como vector cero (p. ej. patches en blanco) también
    # producen NaN con <=>; se excluyen de la búsqueda.
    zero_vector = [0.0] * dim

    with get_conn() as conn:
        register_vector(conn)
        table_name = f"embeddings_{modality}"

        # Agregamos por fuente (mejor chunk por canción/imagen) para devolver
        # fuentes distintas — mismo criterio que el Lado A y que gin_gist.
        query = f"""
            SELECT source_id, MAX(1 - (embedding <=> %s::vector)) AS score
            FROM {table_name}
            WHERE embedding <> %s::vector
            GROUP BY source_id
            ORDER BY score DESC
            LIMIT %s
        """
        rows = conn.execute(query, (dense_vector, zero_vector, k)).fetchall()

    return [
        SearchResult(source_id=str(row[0]), score=float(row[1]))
        for row in rows
    ]
