"""TF-IDF / tokenización para texto.  OWNER: Ing. Texto."""
from __future__ import annotations

import re
from typing import Sequence

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

from src.core import Chunk, Descriptor, Extractor

_STOPWORDS: set[str] = set(stopwords.words("english"))
_STEMMER = PorterStemmer()
_TOKEN_RE = re.compile(r"[a-zA-ZáéíóúüñÁÉÍÓÚÜÑ']+")


class TfidfExtractor(Extractor):
    def extract(self, chunks: Sequence[Chunk]) -> list[Descriptor]:
        descriptors: list[Descriptor] = []
        for chunk in chunks:
            raw = chunk.payload if isinstance(chunk.payload, str) else str(chunk.payload)
            tokens = _tokenize(raw)
            descriptors.append(Descriptor(
                chunk=chunk,
                kind="tokens",
                vector=tokens,
            ))
        return descriptors


def _tokenize(text: str) -> list[str]:
    tokens = _TOKEN_RE.findall(text.lower())
    tokens = [t for t in tokens if t not in _STOPWORDS and len(t) > 1]
    tokens = [_STEMMER.stem(t) for t in tokens]
    return tokens
