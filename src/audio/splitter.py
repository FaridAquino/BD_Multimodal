"""Split de audio en ventanas deslizantes (100-200 ms).  OWNER: Ing. Audio."""
from __future__ import annotations

import librosa
import numpy as np


from src.core import Chunk, Modality, Splitter
from typing import Any


class SlidingWindowSplitter(Splitter):
    modality = Modality.AUDIO

    def __init__(self, window_ms: int = 150, hop_ms: int = 75):
        self.window_ms = window_ms
        self.hop_ms = hop_ms

    def split(self, content: str, source_id: str) -> list[Chunk]:

        y, sr = librosa.load(content, sr=None)  

        window_samples = int(self.window_ms * sr / 1000)
        hop_samples = int(self.hop_ms * sr / 1000)
        
        chunks = []
        
        for i, start in enumerate(range(0, len(y) - window_samples, hop_samples)):
            segment = y[start : start + window_samples]
            chunk = Chunk(
                source_id=source_id,
                modality=Modality.AUDIO,
                payload=segment,
                position=i
            )
            chunks.append(chunk)
            
        return chunks
