"""Comparativa Lado B (imagen/audio): pgvector con índice HNSW o IVF.
OWNER: Ing. Imágenes + Ing. Audio (esquema: Tech Lead).

Almacena los MISMOS histogramas como vectores y busca por similitud con `<->`.
"""
from __future__ import annotations

from src.core import SearchResult


def search_vector(modality: str, query_vec, k: int = 10) -> list[SearchResult]:
    # TODO: SELECT ... ORDER BY embedding <-> %s::vector LIMIT k
    raise NotImplementedError
