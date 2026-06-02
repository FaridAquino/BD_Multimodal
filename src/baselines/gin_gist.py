"""Comparativa Lado B (texto): índices nativos GIN/GiST + full-text.
OWNER: Ing. Texto (con apoyo del Tech Lead en el esquema).

Debe correr sobre EL MISMO texto que alimenta a SPIMI para que la comparación
sea válida. Mide latencia y memoria de `to_tsvector` + `@@` con índice GIN.
"""
from __future__ import annotations

from src.core import SearchResult


def search_fulltext(query: str, k: int = 10) -> list[SearchResult]:
    # TODO: SELECT ... WHERE tsv @@ plainto_tsquery(%s) ORDER BY ts_rank(...) LIMIT k
    raise NotImplementedError
