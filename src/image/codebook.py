"""Visual words por K-Means sobre SIFT.  OWNER: Ing. Imágenes."""
from __future__ import annotations

from typing import Sequence

from src.core import Codebook, CodebookBuilder, Descriptor, Histogram


class VisualCodebook(Codebook):
    def __init__(self, centroids):
        self._centroids = centroids

    @property
    def size(self) -> int:
        return len(self._centroids)

    def encode(self, descriptor: Descriptor) -> Histogram:
        # TODO(imagen): asignar cada SIFT al centroide más cercano y contar.
        raise NotImplementedError


class KMeansVisualBuilder(CodebookBuilder):
    def __init__(self, k: int = 256):
        self.k = k

    def build(self, descriptors: Sequence[Descriptor]) -> Codebook:
        # TODO(imagen): K-Means sobre todos los SIFT -> k centroides.
        raise NotImplementedError("Ing. Imágenes: K-Means visual words")
