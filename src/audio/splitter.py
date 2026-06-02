"""Split de audio en ventanas deslizantes (100-200 ms).  OWNER: Ing. Audio."""
from __future__ import annotations

from src.core import Chunk, Modality, Splitter


class SlidingWindowSplitter(Splitter):
    modality = Modality.AUDIO

    def __init__(self, window_ms: int = 150, hop_ms: int = 75):
        self.window_ms = window_ms
        self.hop_ms = hop_ms

    def split(self, content, source_id: str) -> list[Chunk]:
        # TODO(audio): ventanas deslizantes sobre la señal.
        raise NotImplementedError("Ing. Audio: implementar ventanas deslizantes")
