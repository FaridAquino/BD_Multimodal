"""Tipos de datos compartidos del dominio.

Estos tipos son AGNÓSTICOS a la modalidad: los usan por igual texto, imagen y
audio. Si necesitas cambiar uno de estos, abre un issue con label `infra` y
asígnaselo al Tech Lead, porque afecta a los tres módulos a la vez.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class Modality(str, Enum):
    TEXT = "text"
    IMAGE = "image"
    AUDIO = "audio"


@dataclass
class Chunk:
    """Unidad atómica de contenido producida por un Splitter.

    payload es el contenido crudo del chunk:
      - texto  -> str (un párrafo)
      - imagen -> np.ndarray del patch / región
      - audio  -> np.ndarray de la ventana
    """
    source_id: str                       # id del documento / imagen / canción origen
    modality: Modality
    payload: Any
    position: int = 0                    # orden dentro del origen
    chunk_id: str | None = None          # lo asigna el repositorio al persistir
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Descriptor:
    """Características extraídas de un chunk por un Extractor.

    kind distingue cómo se cuantiza después:
      - "dense"  -> vector denso (SIFT, MFCC)  -> codebook por K-Means
      - "tokens" -> lista de tokens (texto)    -> codebook lingüístico top-k
    """
    chunk: Chunk
    vector: Any                          # np.ndarray | list[float] | list[str]
    kind: str = "dense"                  # "dense" | "tokens"


@dataclass
class Histogram:
    """Representación final e indexable de un chunk: vector de frecuencias de
    codewords. Es lo que se persiste y lo que compara el índice invertido."""
    chunk_id: str
    source_id: str
    counts: dict[int, int]               # codeword_id -> frecuencia
    codebook_id: int | None = None


@dataclass
class SearchResult:
    source_id: str
    score: float
    chunk_id: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)
