"""Características MFCC.  OWNER: Ing. Audio."""
from __future__ import annotations

import librosa
import numpy as np

from typing import Sequence
from src.core import Chunk, Descriptor, Extractor


class MfccExtractor(Extractor):
    def extract(self, chunks: Sequence[Chunk]) -> list[Descriptor]:
        descriptors = []
        
        for chunk in chunks:
            mfcc = librosa.feature.mfcc(y=chunk.payload, sr=22050, n_mfcc=20)
            vector = np.mean(mfcc, axis=1)
            descriptors.append(Descriptor(chunk=chunk, kind="dense", vector=vector))
        
        return descriptors
