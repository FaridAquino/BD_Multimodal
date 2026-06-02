"""Descriptores locales SIFT.  OWNER: Ing. Imágenes."""
from __future__ import annotations

from typing import Sequence

from src.core import Chunk, Descriptor, Extractor


class SiftExtractor(Extractor):
    def extract(self, chunks: Sequence[Chunk]) -> list[Descriptor]:
        # TODO(imagen): SIFT por patch -> Descriptor(kind="dense", vector=ndarray).
        raise NotImplementedError("Ing. Imágenes: implementar SIFT")
