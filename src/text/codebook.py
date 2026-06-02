"""Codebook lingüístico: top-k palabras más frecuentes.  OWNER: Ing. Texto."""
from __future__ import annotations

from typing import Sequence

from src.core import Codebook, CodebookBuilder, Descriptor, Histogram


class LinguisticCodebook(Codebook):
    def __init__(self, vocab: dict[str, int]):
        self._vocab = vocab  # término -> codeword_id

    @property
    def size(self) -> int:
        return len(self._vocab)

    def encode(self, descriptor: Descriptor) -> Histogram:
        # TODO(texto): contar términos del descriptor que están en el top-k.
        raise NotImplementedError


class TopKCodebookBuilder(CodebookBuilder):
    def __init__(self, k: int = 5000):
        self.k = k

    def build(self, descriptors: Sequence[Descriptor]) -> Codebook:
        # TODO(texto): acumular frecuencias globales y quedarse con las k mayores.
        raise NotImplementedError("Ing. Texto: construir vocabulario top-k")
