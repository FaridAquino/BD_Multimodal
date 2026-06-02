"""Orquestador agnóstico de la arquitectura unificada.

El pipeline no sabe NADA de la modalidad concreta: recibe un Splitter, un
Extractor, un CodebookBuilder y un InvertedIndex que respetan las interfaces, y
ejecuta siempre la misma secuencia:

    split  ->  extract  ->  build codebook  ->  encode  ->  index

Es lo que conecta el trabajo de los tres ingenieros de modalidad. Cada uno
entrega sus implementaciones; el Tech Lead las ensambla aquí.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from .interfaces import CodebookBuilder, Extractor, InvertedIndex, Splitter
from .interfaces import Codebook
from .types import Chunk, Histogram, SearchResult


@dataclass
class ModalityPipeline:
    splitter: Splitter
    extractor: Extractor
    codebook_builder: CodebookBuilder
    index: InvertedIndex
    codebook: Codebook | None = None      # se entrena en .fit()

    def fit(self, corpus: Iterable[tuple[str, object]]) -> "ModalityPipeline":
        """Entrena el codebook y construye el índice sobre toda la colección.

        corpus: iterable de (source_id, contenido_crudo).
        """
        chunks: list[Chunk] = []
        for source_id, content in corpus:
            chunks.extend(self.splitter.split(content, source_id=source_id))

        descriptors = self.extractor.extract(chunks)
        self.codebook = self.codebook_builder.build(descriptors)

        histograms: list[Histogram] = [self.codebook.encode(d) for d in descriptors]
        self.index.build(histograms)
        return self

    def search(self, content, source_id: str = "query", k: int = 10) -> list[SearchResult]:
        """Procesa una consulta cruda con el MISMO pipeline y busca en el índice."""
        if self.codebook is None:
            raise RuntimeError("El pipeline no ha sido entrenado: llama a .fit() primero.")
        q_chunks = self.splitter.split(content, source_id=source_id)
        q_desc = self.extractor.extract(q_chunks)
        # Para una consulta de un solo chunk; si hay varios, se agregan los conteos.
        merged: dict[int, int] = {}
        for d in q_desc:
            h = self.codebook.encode(d)
            for cw, c in h.counts.items():
                merged[cw] = merged.get(cw, 0) + c
        query_hist = Histogram(chunk_id="query", source_id=source_id, counts=merged)
        return self.index.search(query_hist, k=k)
