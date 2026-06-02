"""Características MFCC.  OWNER: Ing. Audio."""
from __future__ import annotations

from typing import Sequence

from src.core import Chunk, Descriptor, Extractor


class MfccExtractor(Extractor):
    def extract(self, chunks: Sequence[Chunk]) -> list[Descriptor]:
        # TODO(audio): MFCC por ventana -> Descriptor(kind="dense", vector=ndarray).
        raise NotImplementedError("Ing. Audio: implementar MFCC")
