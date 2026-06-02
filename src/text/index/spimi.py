"""Índice invertido por SPIMI (OBLIGATORIO por enunciado).  OWNER: Ing. Texto.

Single-Pass In-Memory Indexing: construye diccionario + posting lists por bloque
en memoria, vuelca a disco al llenarse, y fusiona al final. Implementa
core.interfaces.InvertedIndex.
"""
from __future__ import annotations

from typing import Iterable

from src.core import Histogram, InvertedIndex, SearchResult


class SpimiIndex(InvertedIndex):
    def __init__(self, block_size: int = 100_000):
        self.block_size = block_size

    def build(self, histograms: Iterable[Histogram]) -> None:
        # TODO(texto): construcción SPIMI por bloques + merge.
        raise NotImplementedError("Ing. Texto: implementar SPIMI")

    def search(self, query: Histogram, k: int = 10) -> list[SearchResult]:
        # TODO(texto): recuperar y rankear (p. ej. coseno sobre TF-IDF).
        raise NotImplementedError
