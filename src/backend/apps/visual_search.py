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


def _patch_hists(codebook: VisualCodebook, img: np.ndarray) -> list[Histogram]:
    """Un histograma por patch de la consulta (para votación por patch).

    Igual que scripts.probe_query_image: cada patch vota individualmente en
    lugar de fusionar toda la imagen en un solo histograma — así la API da
    los mismos puntajes que el probe.
    """
    chunks = _SPLITTER.split(img, source_id="query")
    descs = _EXTRACTOR.extract(chunks)
    return [h for h in (codebook.encode(d) for d in descs) if h.counts]


def _patch_vectors(codebook: VisualCodebook, hists: list[Histogram]) -> list[np.ndarray]:
    """Convierte los histogramas por patch a vectores densos (para pgvector)."""
    dim = codebook.size
    vecs = []
    for h in hists:
        vec = np.zeros(dim, dtype=np.float32)
        for cw, c in h.counts.items():
            vec[int(cw)] = c
        vecs.append(vec)
    return vecs


def _enrich(ranked: list[tuple[str, float]]) -> list[dict]:
    # Extraemos la metadata de la BD (con IDs casteados a string para mayor seguridad)
    sids = [str(sid) for sid, _ in ranked]
    meta = repo.get_sources_metadata(sids)
    
    out = []
    for sid, score in ranked:
        sid_str = str(sid)
        m = meta.get(sid_str, {})
        
        out.append({
            "source_id": sid_str,
            # La ingesta guarda productDisplayName/articleType (styles.csv)
            "product_name": (m.get("productDisplayName")
                             or m.get("product_name")
                             or m.get("title")
                             or "Desconocido"),
            "category": m.get("articleType", ""),
            "price": m.get("price", "N/A"),
            "image_url": m.get("uri", ""),  # Lectura estricta del uri devuelto por BD
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
        # Votación por patch (como scripts.probe_query_image): cada patch busca
        # en el índice y suma su coseno a las imágenes que devuelve. El puntaje
        # es el acumulado de votos, no un coseno acotado a [0,1].
        hists = _patch_hists(codebook, img)
        if not hists:
            return {"side": "A", "query": file.filename, "results": []}
        puntajes: dict[str, float] = defaultdict(float)
        for h in hists:
            for r in index.search(h, k=10):
                puntajes[r.source_id] += r.score
        ranked = sorted(puntajes.items(), key=lambda x: x[1], reverse=True)[:k]
        return {"side": "A", "query": file.filename, "results": _enrich(ranked)}

    # Lado B: votación por patch en pgvector (mismo esquema que el probe).
    codebook, _ = _load_lado_a()
    hists = _patch_hists(codebook, img)
    if not hists:
        return {"side": "B", "query": file.filename, "results": []}
    try:
        results = pgvector.search_vector_voting(
            "image", _patch_vectors(codebook, hists), k=k
        )
        ranked = [(r.source_id, r.score) for r in results]
    except NotImplementedError:
        ranked = []

    return {"side": "B", "query": file.filename, "results": _enrich(ranked)}
