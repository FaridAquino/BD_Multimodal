"""Verifica que el contrato de la arquitectura unificada está bien definido.
Sirve de smoke test en CI mientras las modalidades se implementan."""
import inspect

from src.core import (
    Chunk, Descriptor, Histogram, Modality, SearchResult,
    Splitter, Extractor, Codebook, CodebookBuilder, InvertedIndex,
    ModalityPipeline,
)


def test_interfaces_are_abstract():
    for cls in (Splitter, Extractor, Codebook, CodebookBuilder, InvertedIndex):
        assert inspect.isabstract(cls), f"{cls.__name__} debe ser abstracta"


def test_types_construct():
    c = Chunk(source_id="s1", modality=Modality.TEXT, payload="hola")
    assert c.modality == "text"
    h = Histogram(chunk_id="c1", source_id="s1", counts={0: 2, 1: 1})
    assert sum(h.counts.values()) == 3
    r = SearchResult(source_id="s1", score=0.9)
    assert 0 <= r.score <= 1


def test_pipeline_requires_fit_before_search():
    import pytest

    class _Dummy(Splitter):
        modality = Modality.TEXT
        def split(self, content, source_id): return []
    class _DExt(Extractor):
        def extract(self, chunks): return []
    class _DBuild(CodebookBuilder):
        def build(self, descriptors): ...
    class _DIdx(InvertedIndex):
        def build(self, histograms): ...
        def search(self, query, k=10): return []

    pipe = ModalityPipeline(_Dummy(), _DExt(), _DBuild(), _DIdx())
    with pytest.raises(RuntimeError):
        pipe.search("consulta")
