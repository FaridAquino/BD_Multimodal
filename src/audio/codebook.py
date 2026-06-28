"""Acoustic words por K-Means sobre MFCC.  OWNER: Ing. Audio."""
from __future__ import annotations

import numpy as np
import joblib

from sklearn.cluster import KMeans
from typing import Sequence
from src.core import Codebook, CodebookBuilder, Descriptor, Histogram


class AcousticCodebook(Codebook):
    def __init__(self, centroids):
        self._centroids = centroids

    @property
    def size(self) -> int:
        return len(self._centroids)

    def encode(self, descriptor: Descriptor) -> Histogram:

        distances = np.linalg.norm(self._centroids - descriptor.vector, axis=1)
        closest_index = np.argmin(distances)

        counts = {int(closest_index): 1}

        chunk_id = str(descriptor.chunk.chunk_id) if descriptor.chunk.chunk_id else "temp_id"
        
        return Histogram(
            chunk_id=chunk_id,
            source_id=descriptor.chunk.source_id,
            counts=counts,
            raw_embedding=descriptor.vector.tolist(),
        )


class KMeansAcousticBuilder(CodebookBuilder):
    def __init__(self, k: int = 256):
        self.k = k

    def build(self, descriptors: Sequence[Descriptor]) -> Codebook:
        
        data = np.array([d.vector for d in descriptors])
        kmeans = KMeans(n_clusters=self.k, n_init=10, random_state=42)
        kmeans.fit(data)

        return AcousticCodebook(centroids=kmeans.cluster_centers_)

    def load_from_file(self, file_path: str) -> Codebook:
        """Carga un modelo K-Means universal guardado previamente con joblib."""
        print(f"   -> Cargando modelo K-Means global desde {file_path}")
        kmeans_model = joblib.load(file_path)
        
        # Opcional: Actualizamos el 'k' del builder por si acaso es diferente al del modelo
        self.k = len(kmeans_model.cluster_centers_) 
        
        return AcousticCodebook(centroids=kmeans_model.cluster_centers_)
