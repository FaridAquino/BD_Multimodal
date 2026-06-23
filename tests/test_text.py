from __future__ import annotations

from src.core import Chunk, Histogram, Modality
from src.text.splitter import ParagraphSplitter
from src.text.extractor import TfidfExtractor
from src.text.codebook import TopKCodebookBuilder
from src.text.index.spimi import SpimiIndex


# Splitter
def test_split_paragraphs_basic():
    s = ParagraphSplitter()
    text = "Primer párrafo.\n\nSegundo párrafo.\n\nTercero."
    chunks = s.split(text, "doc1")
    assert len(chunks) == 3
    assert chunks[0].payload == "Primer párrafo."
    assert chunks[1].payload == "Segundo párrafo."
    assert chunks[2].payload == "Tercero."
    assert all(c.source_id == "doc1" for c in chunks)
    assert chunks[0].position == 0
    assert chunks[1].position == 1
    assert chunks[2].position == 2


def test_split_skips_empty():
    s = ParagraphSplitter()
    text = "Uno.\n\n\n\nDos.\n\n"
    chunks = s.split(text, "doc1")
    assert len(chunks) == 2
    assert chunks[0].payload == "Uno."
    assert chunks[1].payload == "Dos."


def test_split_single_paragraph():
    s = ParagraphSplitter()
    chunks = s.split("Solo un párrafo.", "doc1")
    assert len(chunks) == 1
    assert chunks[0].payload == "Solo un párrafo."


def test_split_empty_text():
    s = ParagraphSplitter()
    chunks = s.split("", "doc1")
    assert chunks == []


def test_split_modality():
    s = ParagraphSplitter()
    assert s.modality == Modality.TEXT
    chunks = s.split("Algo.", "doc1")
    assert chunks[0].modality == Modality.TEXT

# Extractor

def test_extractor_basic():
    ex = TfidfExtractor()
    chunks = [
        Chunk(source_id="d1", modality=Modality.TEXT, payload="The quick brown fox", position=0),
    ]
    descs = ex.extract(chunks)
    assert len(descs) == 1
    d = descs[0]
    assert d.kind == "tokens"
    assert "the" not in d.vector  # stopword eliminada
    assert "quick" in d.vector
    assert "brown" in d.vector


def test_extractor_stemming():
    ex = TfidfExtractor()
    chunks = [
        Chunk(source_id="d1", modality=Modality.TEXT, payload="running dogs are barking loudly", position=0),
    ]
    descs = ex.extract(chunks)
    tokens = descs[0].vector
    assert "run" in tokens
    assert "dog" in tokens
    assert "bark" in tokens
    assert "loudli" in tokens


def test_extractor_short_tokens_filtered():
    ex = TfidfExtractor()
    chunks = [
        Chunk(source_id="d1", modality=Modality.TEXT, payload="a an the in on at", position=0),
    ]
    descs = ex.extract(chunks)
    assert descs[0].vector == []  # sólo stopwords cortas


def test_extractor_multiple_chunks():
    ex = TfidfExtractor()
    chunks = [
        Chunk(source_id="d1", modality=Modality.TEXT, payload="fox runs", position=0),
        Chunk(source_id="d1", modality=Modality.TEXT, payload="dog barks", position=1),
    ]
    descs = ex.extract(chunks)
    assert len(descs) == 2
    assert "fox" in descs[0].vector
    assert "dog" in descs[1].vector


# Codebook

def _make_descriptor(tokens: list[str], source_id: str = "d1", pos: int = 0):
    chunk = Chunk(source_id=source_id, modality=Modality.TEXT, payload=" ".join(tokens), position=pos)
    from src.core import Descriptor
    return Descriptor(chunk=chunk, vector=tokens, kind="tokens")


def test_codebook_builder_topk():
    descs = [
        _make_descriptor(["fox", "dog", "fox"]),
        _make_descriptor(["dog", "cat"]),
        _make_descriptor(["fox", "bird", "bird"]),
    ]
    builder = TopKCodebookBuilder(k=3)
    cb = builder.build(descs)
    assert cb.size == 3
    assert "fox" in cb._vocab  # frecuencia 3


def test_codebook_encode():
    descs = [
        _make_descriptor(["fox", "dog", "fox"]),
        _make_descriptor(["dog", "cat"]),
    ]
    builder = TopKCodebookBuilder(k=10)
    cb = builder.build(descs)
    desc = _make_descriptor(["fox", "dog", "fox"])
    hist = cb.encode(desc)
    assert isinstance(hist, Histogram)
    assert hist.source_id == "d1"
    fox_id = cb._vocab["fox"]
    dog_id = cb._vocab["dog"]
    assert hist.counts.get(fox_id) == 2
    assert hist.counts.get(dog_id) == 1


def test_codebook_encode_unknown_term():
    descs = [_make_descriptor(["fox"])]
    builder = TopKCodebookBuilder(k=10)
    cb = builder.build(descs)
    desc = _make_descriptor(["fox", "unknown_word"])
    hist = cb.encode(desc)
    fox_id = cb._vocab["fox"]
    assert hist.counts.get(fox_id) == 1
    assert "unknown_word" not in cb._vocab


def test_codebook_empty_corpus():
    builder = TopKCodebookBuilder(k=10)
    cb = builder.build([])
    assert cb.size == 0


# SPIMI

def _make_hist(source_id: str, counts: dict[int, int], chunk_id: str | None = None):
    return Histogram(
        chunk_id=chunk_id or source_id,
        source_id=source_id,
        counts=counts,
    )


def test_spimi_build_and_search():
    hists = [
        _make_hist("d1", {0: 2, 1: 1, 2: 1}),
        _make_hist("d2", {0: 1, 1: 2}),
        _make_hist("d3", {2: 1, 3: 1}),
    ]
    index = SpimiIndex(block_size=50000)
    index.build(hists)
    assert len(index._index) == 4  # 4 términos únicos
    assert index._doc_count == 3

    # Query similar a d1 (términos 0 y 1); d3 (solo 2,3) no matchea
    q_hist = _make_hist("q", {0: 1, 1: 1})
    results = index.search(q_hist, k=3)
    assert len(results) == 2  # solo d1 y d2 tienen score > 0
    assert results[0].source_id in ("d1", "d2")
    assert results[1].source_id in ("d1", "d2")


def test_spimi_empty_index():
    index = SpimiIndex()
    index.build([])
    assert index.search(_make_hist("q", {0: 1})) == []


def test_spimi_no_match():
    hists = [_make_hist("d1", {0: 1})]
    index = SpimiIndex()
    index.build(hists)
    results = index.search(_make_hist("q", {99: 1}))
    assert results == []


def test_spimi_block_flush():
    # Forzar el volcado a disco con block_size pequeño.
    hists = [
        _make_hist(f"d{i}", {i: 1, (i + 1) % 5: 1})
        for i in range(10)
    ]
    index = SpimiIndex(block_size=5)  # cada ~5 postings fuerza un bloque
    index.build(hists)
    assert len(index._index) > 0
    assert len(index._block_paths) > 0  # debe haber generado al menos 1 bloque


# BSBI

def test_bsbi_basic():
    from src.text.index.bsbi import BsbiIndex
    hists = [
        _make_hist("d1", {0: 2, 1: 1}),
        _make_hist("d2", {0: 1, 2: 1}),
        _make_hist("d3", {1: 2, 2: 1}),
    ]
    idx = BsbiIndex(block_size=50000)
    idx.build(hists)
    assert len(idx._index) == 3
    assert idx._doc_count == 3
    results = idx.search(_make_hist("q", {0: 1, 1: 1}), k=3)
    assert len(results) == 3
    assert results[0].score > 0


def test_bsbi_same_scores_as_spimi():
    from src.text.index.spimi import SpimiIndex
    from src.text.index.bsbi import BsbiIndex
    hists = [
        _make_hist("d1", {0: 2, 1: 1, 2: 1}),
        _make_hist("d2", {0: 1, 1: 2, 3: 1}),
        _make_hist("d3", {2: 1, 3: 1}),
    ]
    spimi = SpimiIndex(block_size=50000)
    bsbi = BsbiIndex(block_size=50000)
    spimi.build(hists)
    bsbi.build(hists)
    q = _make_hist("q", {0: 1, 1: 1})
    spimi_res = {(r.source_id, round(r.score, 6)) for r in spimi.search(q, k=3)}
    bsbi_res = {(r.source_id, round(r.score, 6)) for r in bsbi.search(q, k=3)}
    assert spimi_res == bsbi_res


def test_bsbi_block_flush():
    from src.text.index.bsbi import BsbiIndex
    hists = [
        _make_hist(f"d{i}", {i: 1, (i + 1) % 5: 1})
        for i in range(20)
    ]
    idx = BsbiIndex(block_size=10)
    idx.build(hists)
    assert len(idx._block_paths) > 1
    assert len(idx._index) > 0


def test_bsbi_empty():
    from src.text.index.bsbi import BsbiIndex
    idx = BsbiIndex()
    idx.build([])
    assert idx.search(_make_hist("q", {0: 1})) == []


# Pipeline end-to-end

def test_text_pipeline_e2e():
    from src.core.pipeline import ModalityPipeline

    corpus = [
        ("d1", "The quick brown fox jumps over the lazy dog."),
        ("d2", "A dog is barking loudly at the moon while foxes run."),
        ("d3", "The forest has many brown foxes and barking dogs."),
        ("d4", "Cats are quiet animals that sleep all day long."),
    ]

    pipeline = ModalityPipeline(
        splitter=ParagraphSplitter(),
        extractor=TfidfExtractor(),
        codebook_builder=TopKCodebookBuilder(k=100),
        index=SpimiIndex(block_size=50000),
    )
    pipeline.fit(corpus)

    # Búsqueda: d1, d2, d3 contienen fox/dog; d4 no
    results = pipeline.search("fox dog", k=4)
    assert len(results) == 3  # d4 no matchea
    assert all(r.score > 0 for r in results)
    assert "d4" not in [r.source_id for r in results]


def test_text_pipeline_search_no_corpus():
    from src.core.pipeline import ModalityPipeline

    pipeline = ModalityPipeline(
        splitter=ParagraphSplitter(),
        extractor=TfidfExtractor(),
        codebook_builder=TopKCodebookBuilder(k=100),
        index=SpimiIndex(),
    )
    pipeline.fit([])
    results = pipeline.search("algo", k=5)
    assert results == []
