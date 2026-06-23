"""Split de texto en párrafos/estrofas."""
from __future__ import annotations

import re

from src.core import Chunk, Modality, Splitter

_PARA_SEP = re.compile(r"\n\s*\n")


class ParagraphSplitter(Splitter):
    modality = Modality.TEXT

    def split(self, content: str, source_id: str) -> list[Chunk]:
        raw = _PARA_SEP.split(content.replace("\r\n", "\n"))
        chunks: list[Chunk] = []
        for pos, paragraph in enumerate(raw):
            paragraph = paragraph.strip()
            if not paragraph:
                continue
            chunks.append(Chunk(
                source_id=source_id,
                modality=self.modality,
                payload=paragraph,
                position=pos,
            ))
        return chunks
