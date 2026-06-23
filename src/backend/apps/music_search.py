"""App 2 — Búsqueda musical por letra (lado A SPIMI vs lado B GIN)."""
from __future__ import annotations

from collections import defaultdict
from functools import lru_cache
from pathlib import Path

from fastapi import APIRouter, HTTPException, Query

from src.core import Histogram
from src.text.splitter import ParagraphSplitter
from src.text.extractor import TfidfExtractor
from src.text.codebook import LinguisticCodebook
from src.text.index.spimi import SpimiIndex
from src.baselines import gin_gist
from src.db import repositories as repo

router = APIRouter()

MODELS_DIR = Path("models/text")
_CODEBOOK_PATH = MODELS_DIR / "codebook_text.json"
_INDEX_PATH = MODELS_DIR / "index_text.pkl"

_SPLITTER = ParagraphSplitter()
_EXTRACTOR = TfidfExtractor()


@lru_cache(maxsize=1)
def _load_lado_a() -> tuple[LinguisticCodebook, SpimiIndex]:
    if not _CODEBOOK_PATH.exists() or not _INDEX_PATH.exists():
        raise HTTPException(
            status_code=503,
            detail="Índice de texto no disponible. Corre ingest + build_index --modality text.",
        )
    return LinguisticCodebook.load(str(_CODEBOOK_PATH)), SpimiIndex.load(str(_INDEX_PATH))


def _encode_query(codebook: LinguisticCodebook, q: str) -> Histogram:
    chunks = _SPLITTER.split(q, source_id="query")
    descs = _EXTRACTOR.extract(chunks)
    merged: dict[int, int] = {}
    for d in descs:
        for cw, c in codebook.encode(d).counts.items():
            merged[cw] = merged.get(cw, 0) + c
    return Histogram(chunk_id="query", source_id="query", counts=merged)


def _aggregate_by_source(results) -> list[tuple[str, float]]:
    acc: dict[str, float] = defaultdict(float)
    for r in results:
        acc[r.source_id] = max(acc[r.source_id], r.score)
    return sorted(acc.items(), key=lambda x: x[1], reverse=True)


def _enrich(ranked: list[tuple[str, float]]) -> list[dict]:
    meta = repo.get_sources_metadata([sid for sid, _ in ranked])
    out = []
    for sid, score in ranked:
        m = meta.get(sid, {})
        out.append({
            "source_id": sid,
            "artist": m.get("artist"),
            "song": m.get("song"),
            "score": round(float(score), 6),
        })
    return out


@router.get("/search")
def search_music(
    q: str = Query(..., min_length=1, description="Texto de la consulta (letra)"),
    side: str = Query("A", pattern="^[AB]$", description="A=SPIMI propio, B=GIN nativo"),
    k: int = Query(10, ge=1, le=100),
):
    if side == "A":
        codebook, index = _load_lado_a()
        query_hist = _encode_query(codebook, q)
        if not query_hist.counts:
            return {"side": "A", "query": q, "results": []}
        raw = index.search(query_hist, k=k * 10)
        ranked = _aggregate_by_source(raw)[:k]
        return {"side": "A", "query": q, "results": _enrich(ranked)}

    results = gin_gist.search_fulltext_aggregated(q, k=k)
    ranked = [(r.source_id, r.score) for r in results]
    return {"side": "B", "query": q, "results": _enrich(ranked)}
