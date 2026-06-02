"""Split de imagen en patches/regiones.  OWNER: Ing. Imágenes."""
from __future__ import annotations

from src.core import Chunk, Modality, Splitter


class PatchSplitter(Splitter):
    modality = Modality.IMAGE

    def split(self, content, source_id: str) -> list[Chunk]:
        # TODO(imagen): generar patches superpuestos / regiones de la imagen.
        raise NotImplementedError("Ing. Imágenes: implementar split por patches")
