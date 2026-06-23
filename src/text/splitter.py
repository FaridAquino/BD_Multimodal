from __future__ import annotations

from src.core import Chunk, Modality, Splitter


class ParagraphSplitter(Splitter):
    modality = Modality.TEXT

    def split(self, content: str, source_id: str) -> list[Chunk]:
        raw = content.split("\n\n")
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
