"""TF-IDF / tokenización para texto.  OWNER: Ing. Texto."""
from __future__ import annotations

from typing import Sequence

from src.core import Chunk, Descriptor, Extractor


class TfidfExtractor(Extractor):
    def extract(self, chunks: Sequence[Chunk]) -> list[Descriptor]:
        # TODO(texto): tokenizar -> normalizar -> stopwords -> stemming.
        # Devuelve Descriptor(kind="tokens", vector=lista_de_tokens).
        raise NotImplementedError("Ing. Texto: implementar TF-IDF / tokenización")
