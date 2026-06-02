"""Índice invertido por BSBI (OPCIONAL, baseline comparativo).  OWNER: Ing. Texto.

Solo si el equipo decide comparar SPIMI vs BSBI en la Fase 4. Si no, este archivo
queda como NotImplementedError y no se evalúa.
"""
from __future__ import annotations

from typing import Iterable

from src.core import Histogram, InvertedIndex, SearchResult


class BsbiIndex(InvertedIndex):
    def build(self, histograms: Iterable[Histogram]) -> None:
        raise NotImplementedError("Opcional: Blocked Sort-Based Indexing")

    def search(self, query: Histogram, k: int = 10) -> list[SearchResult]:
        raise NotImplementedError
