"""Índice invertido sobre histogramas de acoustic words.  OWNER: Ing. Audio."""
from __future__ import annotations

from typing import Iterable

from src.core import Histogram, InvertedIndex, SearchResult


class AcousticInvertedIndex(InvertedIndex):
    def build(self, histograms: Iterable[Histogram]) -> None:
        raise NotImplementedError("Ing. Audio: índice invertido acústico")

    def search(self, query: Histogram, k: int = 10) -> list[SearchResult]:
        raise NotImplementedError
