"""Índice invertido sobre histogramas de visual words.  OWNER: Ing. Imágenes."""
from __future__ import annotations

from typing import Iterable

from src.core import Histogram, InvertedIndex, SearchResult


class VisualInvertedIndex(InvertedIndex):
    def build(self, histograms: Iterable[Histogram]) -> None:
        raise NotImplementedError("Ing. Imágenes: índice invertido visual")

    def search(self, query: Histogram, k: int = 10) -> list[SearchResult]:
        raise NotImplementedError
