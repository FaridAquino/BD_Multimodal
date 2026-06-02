"""Acoustic words por K-Means sobre MFCC.  OWNER: Ing. Audio."""
from __future__ import annotations

from typing import Sequence

from src.core import Codebook, CodebookBuilder, Descriptor, Histogram


class AcousticCodebook(Codebook):
    def __init__(self, centroids):
        self._centroids = centroids

    @property
    def size(self) -> int:
        return len(self._centroids)

    def encode(self, descriptor: Descriptor) -> Histogram:
        raise NotImplementedError


class KMeansAcousticBuilder(CodebookBuilder):
    def __init__(self, k: int = 256):
        self.k = k

    def build(self, descriptors: Sequence[Descriptor]) -> Codebook:
        # TODO(audio): K-Means sobre todos los MFCC -> k centroides.
        raise NotImplementedError("Ing. Audio: K-Means acoustic words")
