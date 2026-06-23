from __future__ import annotations

from typing import Sequence

import cv2
import numpy as np

from src.core import Chunk, Descriptor, Extractor


class SiftExtractor(Extractor):
    """Calcula descriptores SIFT (n_keypoints x 128) por cada patch."""

    def __init__(self, n_features: int = 0):
        self._sift = cv2.SIFT_create(nfeatures=n_features)

    def _to_gray(self, patch: np.ndarray) -> np.ndarray:
        if patch.ndim == 3:
            return cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
        return patch

    def extract(self, chunks: Sequence[Chunk]) -> list[Descriptor]:
        descriptors: list[Descriptor] = []
        for ch in chunks:
            gray = self._to_gray(ch.payload)
            _, desc = self._sift.detectAndCompute(gray, None)
            if desc is None:
                desc = np.empty((0, 128), dtype=np.float32)
            descriptors.append(Descriptor(chunk=ch, vector=desc, kind="dense"))
        return descriptors
