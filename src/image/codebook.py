from __future__ import annotations

from typing import Sequence

import numpy as np
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.metrics import pairwise_distances_argmin

from src.core import Codebook, CodebookBuilder, Descriptor, Histogram


class VisualCodebook(Codebook):
    """Vocabulario visual entrenado (k centroides) que codifica patches en histogramas."""

    def __init__(self, centroids: np.ndarray):
        self._centroids = centroids

    @property
    def size(self) -> int:
        return self._centroids.shape[0]

    def encode(self, descriptor: Descriptor) -> Histogram:
        desc = descriptor.vector
        chunk = descriptor.chunk
        counts: dict[int, int] = {}
        if desc is not None and desc.size > 0:
            asignaciones = pairwise_distances_argmin(desc, self._centroids)
            ids, freqs = np.unique(asignaciones, return_counts=True)
            counts = {int(i): int(f) for i, f in zip(ids, freqs)}
        return Histogram(
            chunk_id=str(chunk.chunk_id) if chunk.chunk_id is not None else "",
            source_id=chunk.source_id,
            counts=counts,
        )

    def save(self, path: str) -> None:
        np.save(path, self._centroids)

    @classmethod
    def from_file(cls, path: str) -> "VisualCodebook":
        return cls(np.load(path))


class KMeansVisualBuilder(CodebookBuilder):
    """Entrena el codebook visual con K-Means sobre todos los descriptores SIFT."""

    def __init__(self, k: int = 256, sample: int | None = None,
                 use_minibatch: bool = True, seed: int = 42):
        self.k = k
        self.sample = sample
        self.use_minibatch = use_minibatch
        self.seed = seed

    def build(self, descriptors: Sequence[Descriptor]) -> Codebook:
        matrices = [d.vector for d in descriptors
                    if d.vector is not None and d.vector.size > 0]
        if not matrices:
            raise ValueError("No hay descriptores para entrenar el codebook.")
        todos = np.vstack(matrices).astype(np.float32)

        if self.sample is not None and todos.shape[0] > self.sample:
            rng = np.random.default_rng(self.seed)
            idx = rng.choice(todos.shape[0], size=self.sample, replace=False)
            entrenamiento = todos[idx]
        else:
            entrenamiento = todos

        if self.use_minibatch:
            km = MiniBatchKMeans(n_clusters=self.k, random_state=self.seed,
                                 batch_size=10_000, n_init="auto")
        else:
            km = KMeans(n_clusters=self.k, random_state=self.seed, n_init="auto")
        km.fit(entrenamiento)

        return VisualCodebook(centroids=km.cluster_centers_)
