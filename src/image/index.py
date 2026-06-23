from __future__ import annotations

import math
import pickle
from collections import defaultdict
from typing import Iterable

from src.core import Histogram, InvertedIndex, SearchResult


class VisualInvertedIndex(InvertedIndex):
    """Índice invertido de visual words (Lado A) con búsqueda por coseno y persistencia en disco."""

    FORMAT_VERSION = 1

    def __init__(self):
        self._postings: dict[int, list[tuple[int, float]]] = defaultdict(list)
        self._chunk_ids: list[str] = []
        self._source_ids: list[str] = []
        self._norms: list[float] = []

    def build(self, histograms: Iterable[Histogram]) -> None:
        for h in histograms:
            idx = len(self._chunk_ids)
            self._chunk_ids.append(h.chunk_id)
            self._source_ids.append(h.source_id)

            suma_cuadrados = 0.0
            for vw, freq in h.counts.items():
                self._postings[int(vw)].append((idx, float(freq)))
                suma_cuadrados += float(freq) ** 2
            self._norms.append(math.sqrt(suma_cuadrados))

    def search(self, query: Histogram, k: int = 10) -> list[SearchResult]:
        q_norm = math.sqrt(sum(float(f) ** 2 for f in query.counts.values()))
        if q_norm == 0:
            return []

        dots: dict[int, float] = defaultdict(float)
        for vw, qf in query.counts.items():
            for idx, freq in self._postings.get(int(vw), []):
                dots[idx] += float(qf) * freq

        resultados: list[SearchResult] = []
        for idx, dot in dots.items():
            denom = q_norm * self._norms[idx]
            if denom == 0:
                continue
            resultados.append(
                SearchResult(
                    source_id=self._source_ids[idx],
                    score=dot / denom,
                    chunk_id=self._chunk_ids[idx],
                )
            )
        resultados.sort(key=lambda r: r.score, reverse=True)
        return resultados[:k]

    def save(self, path: str) -> None:
        data = {
            "version": self.FORMAT_VERSION,
            "postings": dict(self._postings),
            "chunk_ids": self._chunk_ids,
            "source_ids": self._source_ids,
            "norms": self._norms,
        }
        with open(path, "wb") as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, path: str) -> "VisualInvertedIndex":
        with open(path, "rb") as f:
            data = pickle.load(f)
        if data.get("version") != cls.FORMAT_VERSION:
            raise ValueError(
                f"Versión de índice incompatible: {data.get('version')} "
                f"!= {cls.FORMAT_VERSION}. Reconstruye el índice."
            )
        ix = cls()
        ix._postings = defaultdict(list, data["postings"])
        ix._chunk_ids = data["chunk_ids"]
        ix._source_ids = data["source_ids"]
        ix._norms = data["norms"]
        return ix

    def stats(self) -> dict:
        return {
            "n_chunks": len(self._chunk_ids),
            "n_visual_words": len(self._postings),
            "n_postings": sum(len(v) for v in self._postings.values()),
        }
