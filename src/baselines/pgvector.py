"""Comparativa Lado B (imagen/audio): pgvector con índice HNSW o IVF.
OWNER: Ing. Imágenes + Ing. Audio (esquema: Tech Lead).

Almacena los MISMOS histogramas como vectores y busca por similitud con `<->`.
"""
from __future__ import annotations

from collections import defaultdict

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


def search_vector_voting(
    modality: str, query_vectors, k: int = 10, knn: int = 10
) -> list[SearchResult]:
    """Búsqueda por VOTACIÓN de ventanas (estilo Shazam) para pgvector.

    A diferencia de ``search_vector`` (que fusiona toda la consulta en un solo
    histograma y hace MAX-coseno contra chunks 1-hot, produciendo empates
    degenerados), aquí cada ventana de la consulta hace su propio kNN y VOTA por
    las canciones cuyos chunks son sus vecinos más cercanos. Consulta y BD quedan
    a la misma granularidad (por ventana) — mismo esquema que scripts.probe_query_audio.

    query_vectors: iterable de vectores densos (uno por ventana de la consulta).
    knn: vecinos recuperados por ventana; k: fuentes devueltas.
    """
    puntajes: dict[str, float] = defaultdict(float)
    table_name = f"embeddings_{modality}"
    with get_conn() as conn:
        register_vector(conn)
        for qvec in query_vectors:
            if not getattr(qvec, "any", bool)():
                continue
            rows = conn.execute(
                f"SELECT source_id, embedding <=> %s::vector AS dist "
                f"FROM {table_name} ORDER BY dist LIMIT %s",
                (qvec, knn),
            ).fetchall()
            for source_id, dist in rows:
                puntajes[str(source_id)] += 1.0 - float(dist)

    ranked = sorted(puntajes.items(), key=lambda x: x[1], reverse=True)[:k]
    return [SearchResult(source_id=sid, score=float(sc)) for sid, sc in ranked]
