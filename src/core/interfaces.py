"""Interfaces de la arquitectura unificada.

ESTE ES EL CONTRATO. Texto, imagen y audio implementan estas mismas clases
abstractas. Mientras se respeten estas firmas, las tres modalidades avanzan en
paralelo sin bloquearse y el pipeline las orquesta sin saber de qué modalidad se
trata.

Reglas de revisión (Tech Lead):
  - Ningún PR de modalidad debe cambiar las firmas de este archivo.
  - Cualquier cambio aquí requiere aprobación del Tech Lead (ver CODEOWNERS).
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Iterable, Sequence

from .types import Chunk, Descriptor, Histogram, Modality, SearchResult


class Splitter(ABC):
    """Divide contenido crudo en chunks atómicos."""

    modality: Modality

    @abstractmethod
    def split(self, content, source_id: str) -> list[Chunk]:
        """content: texto crudo / imagen / audio cargado. Devuelve sus chunks."""
        raise NotImplementedError


class Extractor(ABC):
    """Extrae descriptores/features de un conjunto de chunks."""

    @abstractmethod
    def extract(self, chunks: Sequence[Chunk]) -> list[Descriptor]:
        raise NotImplementedError


class Codebook(ABC):
    """Diccionario compartido ya entrenado. Sabe convertir un descriptor en un
    histograma de codewords (cuantización para imagen/audio; conteo de términos
    del top-k para texto)."""

    @property
    @abstractmethod
    def size(self) -> int:
        """Número de codewords (k)."""
        raise NotImplementedError

    @abstractmethod
    def encode(self, descriptor: Descriptor) -> Histogram:
        raise NotImplementedError

    @abstractmethod
    def save(self, path: str) -> None:
        """Persiste el codebook entrenado en disco."""
        raise NotImplementedError


class CodebookBuilder(ABC):
    """Entrena el codebook a partir de TODOS los descriptores de la colección
    (K-Means para imagen/audio; top-k de frecuencias para texto)."""

    @abstractmethod
    def build(self, descriptors: Sequence[Descriptor]) -> Codebook:
        raise NotImplementedError


class InvertedIndex(ABC):
    """Índice invertido propio (Lado A). Para texto se implementa con SPIMI."""

    @abstractmethod
    def build(self, histograms: Iterable[Histogram]) -> None:
        raise NotImplementedError

    @abstractmethod
    def search(self, query: Histogram, k: int = 10) -> list[SearchResult]:
        raise NotImplementedError
