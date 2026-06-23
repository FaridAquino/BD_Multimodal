from __future__ import annotations

import cv2

from src.core import Chunk, Modality, Splitter


class PatchSplitter(Splitter):
    """Divide una imagen en una rejilla de rows x cols patches (un Chunk por patch)."""

    modality = Modality.IMAGE

    def __init__(self, rows: int = 3, cols: int = 3, overlap: float = 0.0):
        if not (0.0 <= overlap < 1.0):
            raise ValueError("overlap debe estar en [0.0, 1.0)")
        self.rows = rows
        self.cols = cols
        self.overlap = overlap

    def split(self, content, source_id: str) -> list[Chunk]:
        img = cv2.imread(content) if isinstance(content, str) else content
        if img is None:
            raise ValueError(f"No se pudo cargar la imagen: {source_id}")

        h, w = img.shape[:2]
        cell_h = h // self.rows
        cell_w = w // self.cols
        pad_h = int(cell_h * self.overlap)
        pad_w = int(cell_w * self.overlap)

        chunks: list[Chunk] = []
        position = 0
        for r in range(self.rows):
            for c in range(self.cols):
                y0 = max(r * cell_h - pad_h, 0)
                y1 = min((r + 1) * cell_h + pad_h, h)
                x0 = max(c * cell_w - pad_w, 0)
                x1 = min((c + 1) * cell_w + pad_w, w)

                patch = img[y0:y1, x0:x1]
                if patch.size == 0:
                    continue

                chunks.append(
                    Chunk(
                        source_id=source_id,
                        modality=Modality.IMAGE,
                        payload=patch,
                        position=position,
                        metadata={"row": r, "col": c,
                                  "bbox": [int(x0), int(y0), int(x1), int(y1)]},
                    )
                )
                position += 1
        return chunks
