"""App 2 — Búsqueda musical por letra (lado A SPIMI vs lado B GIN) y por Audio."""
from __future__ import annotations

from collections import defaultdict
from functools import lru_cache
from pathlib import Path
import tempfile
import shutil
import os
import joblib

from fastapi import APIRouter, HTTPException, Query, UploadFile, File

from src.core import Histogram
from src.text.splitter import ParagraphSplitter
from src.text.extractor import TfidfExtractor
from src.text.codebook import LinguisticCodebook
from src.text.index.spimi import SpimiIndex
from src.baselines import gin_gist, pgvector
from src.db import repositories as repo

from src.audio.splitter import SlidingWindowSplitter
from src.audio.extractor import MfccExtractor
from src.audio.codebook import AcousticCodebook
from src.audio.index import AcousticInvertedIndex

router = APIRouter()

# --- MODELOS DE TEXTO ---
TEXT_MODELS_DIR = Path("models/text")
_TEXT_CODEBOOK_PATH = TEXT_MODELS_DIR / "codebook_text.json"
_TEXT_INDEX_PATH = TEXT_MODELS_DIR / "index_text.pkl"
_TEXT_SPLITTER = ParagraphSplitter()
_TEXT_EXTRACTOR = TfidfExtractor()

# --- MODELOS DE AUDIO ---
AUDIO_MODELS_DIR = Path("models/audio")
_AUDIO_CODEBOOK_PATH = AUDIO_MODELS_DIR / "kmeans_256_fma.joblib"  # nombre exacto del script de ingesta
_AUDIO_INDEX_PATH = AUDIO_MODELS_DIR / "index_audio.pkl"
_AUDIO_SPLITTER = SlidingWindowSplitter()
_AUDIO_EXTRACTOR = MfccExtractor()

@lru_cache(maxsize=1)
def _load_text_lado_a() -> tuple[LinguisticCodebook, SpimiIndex]:
    if not _TEXT_CODEBOOK_PATH.exists() or not _TEXT_INDEX_PATH.exists():
        raise HTTPException(
            status_code=503,
            detail="Índice de texto no disponible. Corre ingest + build_index --modality text.",
        )
    return LinguisticCodebook.load(str(_TEXT_CODEBOOK_PATH)), SpimiIndex.load(str(_TEXT_INDEX_PATH))

@lru_cache(maxsize=1)
def _load_audio_lado_a() -> tuple[AcousticCodebook, AcousticInvertedIndex]:
    if not _AUDIO_CODEBOOK_PATH.exists() or not _AUDIO_INDEX_PATH.exists():
        missing = []
        if not _AUDIO_CODEBOOK_PATH.exists():
            missing.append(str(_AUDIO_CODEBOOK_PATH))
        if not _AUDIO_INDEX_PATH.exists():
            missing.append(str(_AUDIO_INDEX_PATH))
        raise HTTPException(
            status_code=503,
            detail=f"Modelos de audio no encontrados: {missing}. Corre ingest + build_index --modality audio.",
        )
    kmeans_model = joblib.load(str(_AUDIO_CODEBOOK_PATH))
    codebook = AcousticCodebook(centroids=kmeans_model.cluster_centers_)
    index = AcousticInvertedIndex.load(str(_AUDIO_INDEX_PATH))
    return codebook, index

def _encode_text_query(codebook: LinguisticCodebook, q: str) -> Histogram:
    chunks = _TEXT_SPLITTER.split(q, source_id="query")
    descs = _TEXT_EXTRACTOR.extract(chunks)
    merged: dict[int, int] = {}
    for d in descs:
        for cw, c in codebook.encode(d).counts.items():
            merged[cw] = merged.get(cw, 0) + c
    return Histogram(chunk_id="query", source_id="query", counts=merged)

def _encode_audio_query(codebook: AcousticCodebook, path: str) -> Histogram:
    chunks = _AUDIO_SPLITTER.split(path, source_id="query")
    descs = _AUDIO_EXTRACTOR.extract(chunks)
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

def _clean_source_id(raw_sid: str) -> str:
    """Normaliza un source_id que puede venir con sufijos de chunk/patch o extensión de archivo."""
    sid = str(raw_sid).strip()
    # Eliminar sufijos de troceado: "10000_chunk_3" → "10000"
    for suffix in ("_chunk", "_patch", "_segment"):
        if suffix in sid:
            sid = sid.split(suffix)[0]
    # Eliminar extensiones de audio que puedan estar embebidas en el ID
    for ext in (".wav", ".mp3", ".ogg", ".flac", ".m4a"):
        if sid.endswith(ext):
            sid = sid[: -len(ext)]
    return sid


def _enrich(ranked: list[tuple[str, float]]) -> list[dict]:
    # Normalizamos todos los IDs antes de consultar la BD
    cleaned: list[tuple[str, float]] = [(_clean_source_id(sid), score) for sid, score in ranked]

    meta = repo.get_sources_metadata([sid for sid, _ in cleaned])
    out = []
    for sid, score in cleaned:
        m = meta.get(sid, {})
        out.append({
            "source_id": sid,
            "artist": m.get("artist", "Desconocido"),
            "song": m.get("song", "Desconocida"),
            "score": round(float(score), 6),
        })
    return out

@router.get("/search")
def search_music_text(
    q: str = Query(..., min_length=1, description="Texto de la consulta (letra)"),
    side: str = Query("A", pattern="^[AB]$", description="A=SPIMI propio, B=GIN nativo"),
    k: int = Query(10, ge=1, le=100),
):
    if side == "A":
        codebook, index = _load_text_lado_a()
        query_hist = _encode_text_query(codebook, q)
        if not query_hist.counts:
            return {"side": "A", "query": q, "results": []}
        raw = index.search(query_hist, k=k * 10)
        ranked = _aggregate_by_source(raw)[:k]
        return {"side": "A", "query": q, "results": _enrich(ranked)}

    results = gin_gist.search_fulltext_aggregated(q, k=k)
    ranked = [(r.source_id, r.score) for r in results]
    return {"side": "B", "query": q, "results": _enrich(ranked)}

@router.post("/search_audio")
async def search_music_audio(
    file: UploadFile = File(...),
    side: str = Query("A", pattern="^[AB]$", description="A=Índice Propio, B=pgvector Nativo"),
    k: int = Query(10, ge=1, le=100),
):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    try:
        if side == "A":
            codebook, index = _load_audio_lado_a()
            query_hist = _encode_audio_query(codebook, tmp_path)
            if not query_hist.counts:
                return {"side": "A", "query": file.filename, "results": []}
            raw = index.search(query_hist, k=k * 10)
            ranked = _aggregate_by_source(raw)[:k]
            return {"side": "A", "query": file.filename, "results": _enrich(ranked)}

        codebook, _ = _load_audio_lado_a()
        query_hist = _encode_audio_query(codebook, tmp_path)
        
        try:
            results = pgvector.search_vector("audio", query_hist, k=k)
            ranked = [(r.source_id, r.score) for r in results]
        except NotImplementedError:
            ranked = []
            
        return {"side": "B", "query": file.filename, "results": _enrich(ranked)}
    finally:
        os.remove(tmp_path)
