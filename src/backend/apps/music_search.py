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
_AUDIO_EXTRACTOR = MfccExtractor()


@lru_cache(maxsize=1)
def _audio_splitter() -> SlidingWindowSplitter:
    """Ventaneo idéntico al de la ingesta: lee window/hop del codebook en BD.

    Si el splitter de la consulta no coincide con el usado al ingestar, los
    histogramas quedan a escalas distintas y el ranking se degrada.
    """
    window_ms, hop_ms = 150, 750  # defaults de scripts/ingest.py
    try:
        params = repo.get_latest_codebook_params("audio") or {}
        window_ms = int(params.get("window_ms", window_ms))
        hop_ms = int(params.get("hop_ms", hop_ms))
    except Exception:
        pass  # sin BD: usar defaults
    return SlidingWindowSplitter(window_ms=window_ms, hop_ms=hop_ms)

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

def _audio_window_hists(codebook: AcousticCodebook, path: str) -> list[Histogram]:
    """Un histograma por ventana de la consulta (para votación estilo Shazam).

    A diferencia de _encode_audio_query (que fusiona todo en un histograma),
    aquí cada ventana queda separada para que vote individualmente — mismo
    esquema que scripts.probe_query_audio en ambos lados.
    """
    chunks = _audio_splitter().split(path, source_id="query")
    descs = _AUDIO_EXTRACTOR.extract(chunks)
    return [h for h in (codebook.encode(d) for d in descs) if h.counts]


def _encode_audio_windows(codebook: AcousticCodebook, path: str) -> list:
    """Un vector denso por ventana de la consulta (para votación en Lado B)."""
    import numpy as np
    dim = codebook.size
    vecs = []
    for h in _audio_window_hists(codebook, path):
        vec = np.zeros(dim, dtype=np.float32)
        for cw, c in h.counts.items():
            vec[int(cw)] = c
        vecs.append(vec)
    return vecs

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
        # Tracks FMA no tienen artist/song: usamos género y el id del track.
        fma_id = m.get("fma_id")
        out.append({
            "source_id": sid,
            "artist": m.get("artist") or m.get("genre") or "Desconocido",
            "song": m.get("song") or (f"FMA track {fma_id}" if fma_id else "Desconocida"),
            "genre": m.get("genre"),
            # uri: para audio es la ruta del mp3 (el frontend lo reproduce);
            # para texto es el link a la letra.
            "uri": m.get("uri"),
            "score": round(float(score), 6),
        })
    return out

def _con_letra(results: list[dict]) -> list[dict]:
    """Adjunta la letra (chunks de texto concatenados) a cada resultado."""
    letras = repo.get_lyrics([r["source_id"] for r in results])
    for r in results:
        r["lyrics"] = letras.get(r["source_id"])
    return results


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
        return {"side": "A", "query": q, "results": _con_letra(_enrich(ranked))}

    results = gin_gist.search_fulltext_aggregated(q, k=k)
    ranked = [(r.source_id, r.score) for r in results]
    return {"side": "B", "query": q, "results": _con_letra(_enrich(ranked))}

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
            # Votación por ventana (como scripts.probe_query_audio): cada ventana
            # busca en el índice y suma su coseno a las canciones que devuelve.
            # El puntaje es el acumulado de votos, no un coseno acotado a [0,1].
            hists = _audio_window_hists(codebook, tmp_path)
            if not hists:
                return {"side": "A", "query": file.filename, "results": []}
            puntajes: dict[str, float] = defaultdict(float)
            for h in hists:
                for r in index.search(h, k=10):
                    puntajes[r.source_id] += r.score
            ranked = sorted(puntajes.items(), key=lambda x: x[1], reverse=True)[:k]
            return {"side": "A", "query": file.filename, "results": _enrich(ranked)}

        codebook, _ = _load_audio_lado_a()
        # Lado B por VOTACIÓN de ventanas (como scripts.probe_query_audio): cada
        # ventana de la consulta hace su propio kNN en pgvector y vota. Evita los
        # empates degenerados del MAX-coseno sobre chunks 1-hot.
        qvecs = _encode_audio_windows(codebook, tmp_path)
        if not qvecs:
            return {"side": "B", "query": file.filename, "results": []}
        try:
            results = pgvector.search_vector_voting("audio", qvecs, k=k)
            ranked = [(r.source_id, r.score) for r in results]
        except NotImplementedError:
            ranked = []

        return {"side": "B", "query": file.filename, "results": _enrich(ranked)}
    finally:
        os.remove(tmp_path)
