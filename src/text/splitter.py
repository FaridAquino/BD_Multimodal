"""Split de texto en párrafos.  OWNER: Ing. Texto.

Implementa core.interfaces.Splitter. NO cambies la firma de split().
"""
from __future__ import annotations

from src.core import Chunk, Modality, Splitter


class ParagraphSplitter(Splitter):
    modality = Modality.TEXT

    def split(self, content: str, source_id: str) -> list[Chunk]:
        # TODO(texto): dividir `content` en párrafos y devolver Chunks.
        raise NotImplementedError("Ing. Texto: implementar split por párrafos")
