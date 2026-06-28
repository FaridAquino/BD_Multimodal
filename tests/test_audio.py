"""Tests unitarios del pipeline de audio (sin BD ni modelo en disco).

Mismo estilo que tests/test_text.py: datos sintéticos + asserts. Cubre el
splitter, el extractor MFCC, el codebook acústico y el índice invertido propio
(Lado A): build / search / stats / save+load.
"""
from __future__ import annotations

import numpy as np

from src.core import Chunk, Descriptor, Histogram, Modality
from src.audio.splitter import SlidingWindowSplitter
from src.audio.extractor import MfccExtractor
from src.audio.codebook import AcousticCodebook
from src.audio.index import AcousticInvertedIndex


# ---------- Splitter ----------

def test_splitter_modality_y_params():
    s = SlidingWindowSplitter(window_ms=150, hop_ms=750)
    assert s.modality == Modality.AUDIO
    assert s.window_ms == 150
    assert s.hop_ms == 750


# ---------- Extractor MFCC ----------

def test_mfcc_extractor_dimension():
    sr = 22050
    n = int(0.15 * sr)                       # ventana de ~150 ms
    t = np.linspace(0, 0.15, n, endpoint=False)
    onda = np.sin(2 * np.pi * 440 * t).astype(np.float32)   # la 440 Hz
    chunk = Chunk(source_id="s1", modality=Modality.AUDIO, payload=onda, position=0)

    descs = MfccExtractor().extract([chunk])
    assert len(descs) == 1
    d = descs[0]
    assert d.kind == "dense"
    assert d.vector.shape == (20,)           # n_mfcc = 20
    assert d.chunk is chunk


# ---------- Codebook acústico ----------

def _descriptor(vector, source_id="s1", chunk_id="c1"):
    chunk = Chunk(source_id=source_id, modality=Modality.AUDIO, payload=None,
                  position=0, chunk_id=chunk_id)
    return Descriptor(chunk=chunk, vector=np.asarray(vector, dtype=np.float32))


def test_codebook_size():
    centroids = np.array([[0.0, 0.0], [10.0, 10.0], [20.0, 20.0]])
    assert AcousticCodebook(centroids).size == 3


def test_codebook_encode_palabra_mas_cercana():
    centroids = np.array([[0.0, 0.0], [10.0, 10.0], [20.0, 20.0]])
    codebook = AcousticCodebook(centroids)

    hist = codebook.encode(_descriptor([9.5, 10.5], source_id="songX", chunk_id="42"))
    assert isinstance(hist, Histogram)
    assert hist.counts == {1: 1}             # centroide 1 es el más cercano
    assert hist.source_id == "songX"
    assert hist.chunk_id == "42"


# ---------- Índice invertido (Lado A) ----------

def _histogramas_demo():
    # songA: dos chunks; songB: un chunk con vocabulario distinto.
    return [
        Histogram(chunk_id="a1", source_id="songA", counts={1: 2, 5: 1}),
        Histogram(chunk_id="a2", source_id="songA", counts={5: 1}),
        Histogram(chunk_id="b1", source_id="songB", counts={9: 3, 1: 1}),
    ]


def test_index_build_y_stats():
    ix = AcousticInvertedIndex()
    ix.build(_histogramas_demo())
    stats = ix.stats()
    assert set(stats) == {"n_chunks", "n_acoustic_words", "n_postings"}
    assert stats["n_chunks"] == 2            # agrega por canción: songA, songB
    assert stats["n_acoustic_words"] == len({1, 5, 9})
    assert stats["n_postings"] > 0


def test_index_search_self_match():
    ix = AcousticInvertedIndex()
    ix.build(_histogramas_demo())
    # Consulta con el vocabulario dominante de songA.
    query = Histogram(chunk_id="q", source_id="query", counts={1: 2, 5: 1})
    res = ix.search(query, k=2)
    assert res, "la búsqueda no devolvió resultados"
    assert res[0].source_id == "songA"       # songA debe rankear primero


def test_index_save_load_roundtrip(tmp_path):
    ix = AcousticInvertedIndex()
    ix.build(_histogramas_demo())
    p = tmp_path / "index_audio.pkl"
    ix.save(str(p))

    ix2 = AcousticInvertedIndex.load(str(p))
    assert ix2.stats() == ix.stats()
    query = Histogram(chunk_id="q", source_id="query", counts={1: 2, 5: 1})
    r1 = ix.search(query, k=3)
    r2 = ix2.search(query, k=3)
    assert [x.source_id for x in r1] == [x.source_id for x in r2]
