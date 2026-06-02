"""Acceso a datos (codebooks, histogramas, metadatos).  OWNER: Tech Lead.

Provee las operaciones que tanto Lado A (índice propio) como Lado B (GIN/pgvector)
usan sobre las MISMAS tablas, garantizando comparación justa.
"""
from __future__ import annotations

from typing import Iterable

from src.core import Chunk, Histogram


def insert_source(source_id: str, modality: str, uri: str, metadata: dict) -> None:
    # TODO(lead): INSERT ... ON CONFLICT DO NOTHING en tabla `sources`.
    raise NotImplementedError


def insert_chunks(chunks: Iterable[Chunk]) -> None:
    raise NotImplementedError


def insert_histograms(histograms: Iterable[Histogram]) -> None:
    raise NotImplementedError
