"""Codebook lingüístico: top-k palabras más frecuentes."""
from __future__ import annotations

import json
from collections import Counter
from typing import Sequence

from src.core import Codebook, CodebookBuilder, Descriptor, Histogram


class LinguisticCodebook(Codebook):
    def __init__(self, vocab: dict[str, int]):
        self._vocab = vocab

    @property
    def size(self) -> int:
        return len(self._vocab)

    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self._vocab, f, ensure_ascii=False)

    @classmethod
    def load(cls, path: str) -> "LinguisticCodebook":
        with open(path, encoding="utf-8") as f:
            vocab = json.load(f)
        return cls({term: int(cid) for term, cid in vocab.items()})

    def encode(self, descriptor: Descriptor) -> Histogram:
        counts: dict[int, int] = {}
        for token in descriptor.vector:
            cid = self._vocab.get(token)
            if cid is not None:
                counts[cid] = counts.get(cid, 0) + 1
        return Histogram(
            chunk_id=descriptor.chunk.chunk_id or "",
            source_id=descriptor.chunk.source_id,
            counts=counts,
        )


class TopKCodebookBuilder(CodebookBuilder):
    def __init__(self, k: int = 5000):
        self.k = k

    def build(self, descriptors: Sequence[Descriptor]) -> Codebook:
        freq: Counter[str] = Counter()
        for desc in descriptors:
            freq.update(desc.vector)
        top = freq.most_common(self.k)
        vocab = {term: idx for idx, (term, _) in enumerate(top)}
        return LinguisticCodebook(vocab)
