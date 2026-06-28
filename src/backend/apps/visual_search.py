"""App 1 — Búsqueda Visual E-commerce.  OWNER: Ing. Backend (+ Ing. Imágenes)."""
from __future__ import annotations

from collections import defaultdict
from functools import lru_cache
from pathlib import Path
import cv2
import numpy as np

from fastapi import APIRouter, HTTPException, Query, UploadFile, File

from src.core import Histogram
from src.image.splitter import PatchSplitter
from src.image.extractor import SiftExtractor
from src.image.codebook import VisualCodebook
from src.image.index import VisualInvertedIndex
from src.baselines import pgvector
from src.db import repositories as repo

router = APIRouter()

MODELS_DIR = Path("models/image")
_CODEBOOK_PATH = MODELS_DIR / "codebook_image.npy"
_INDEX_PATH = MODELS_DIR / "index_image.pkl"

_SPLITTER = PatchSplitter()
_EXTRACTOR = SiftExtractor()


@lru_cache(maxsize=1)
def _load_lado_a() -> tuple[VisualCodebook, VisualInvertedIndex]:
    if not _CODEBOOK_PATH.exists() or not _INDEX_PATH.exists():
        raise HTTPException(
            status_code=503,
            detail="Índice de imágenes no disponible. Corre ingest + build_index --modality image.",
        )
    return VisualCodebook.from_file(str(_CODEBOOK_PATH)), VisualInvertedIndex.load(str(_INDEX_PATH))


def _encode_query(codebook: VisualCodebook, img: np.ndarray) -> Histogram:
    chunks = _SPLITTER.split(img, source_id="query")
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
            "product_name": m.get("product_name", m.get("title", "Desconocido")),
            "price": m.get("price", "N/A"),
            "image_url": m.get("image_url", m.get("url", "")),
            "score": round(float(score), 6),
        })
    return out


@router.post("/search")
async def search_by_image(
    file: UploadFile = File(...),
    side: str = Query("A", pattern="^[AB]$", description="A=Índice Propio, B=pgvector Nativo"),
    k: int = Query(10, ge=1, le=100),
):
    file_bytes = await file.read()
    nparr = np.frombuffer(file_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if img is None:
        raise HTTPException(status_code=400, detail="El archivo subido no es una imagen válida.")

    if side == "A":
        codebook, index = _load_lado_a()
        query_hist = _encode_query(codebook, img)
        if not query_hist.counts:
            return {"side": "A", "query": file.filename, "results": []}
        raw = index.search(query_hist, k=k * 10)
        ranked = _aggregate_by_source(raw)[:k]
        return {"side": "A", "query": file.filename, "results": _enrich(ranked)}

    # Lado B
    codebook, _ = _load_lado_a() 
    query_hist = _encode_query(codebook, img)
    
    try:
        results = pgvector.search_vector("image", query_hist, k=k)
        ranked = [(r.source_id, r.score) for r in results]
    except NotImplementedError:
        ranked = []
        
    return {"side": "B", "query": file.filename, "results": _enrich(ranked)}
