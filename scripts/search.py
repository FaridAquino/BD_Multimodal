"""Búsqueda de audio comparando LADO A (índice propio) vs LADO B (pgvector).
OWNER: Ing. Audio.

Procesa un MP3 de consulta con el MISMO pipeline (split -> MFCC -> encode con el
codebook entrenado) y busca por los dos lados:
  - Lado A: AcousticInvertedIndex (TF-IDF + coseno), cargado/reconstruido desde la BD.
  - Lado B: pgvector (operador <=>, distancia coseno) sobre embeddings_audio.
Ambos agregan por canción (source_id). Muestra los rankings y los tiempos.

Uso (Postgres levantado, .venv activado, desde la raíz):
    python -m scripts.search --query data/raw/fma_small/000/000005.mp3 --top 5
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
from pgvector.psycopg import register_vector  # noqa: E402

from src.core import Histogram                       # noqa: E402
from src.audio.splitter import SlidingWindowSplitter  # noqa: E402
from src.audio.extractor import MfccExtractor          # noqa: E402
from src.audio.codebook import KMeansAcousticBuilder   # noqa: E402
from src.audio.index import AcousticInvertedIndex      # noqa: E402
from src.db import repositories as repo                 # noqa: E402
from src.db.connection import get_conn, close_pool      # noqa: E402

MODELS_DIR = Path("models/audio")
MODEL_PATH = "models/audio/kmeans_256_fma.joblib"


def counts_a_vector(counts: dict, k: int) -> np.ndarray:
    vec = np.zeros(k, dtype=np.float32)
    for cw, freq in counts.items():
        vec[int(cw)] = freq
    return vec


def cargar_o_construir_indice(codebook_id: int) -> AcousticInvertedIndex:
    """Carga el índice del Lado A desde disco; si no existe, lo construye desde la BD."""
    path = MODELS_DIR / "index_audio.pkl"
    if path.exists():
        return AcousticInvertedIndex.load(str(path))
    print("Índice no encontrado; construyéndolo desde la BD...")
    index = AcousticInvertedIndex()
    index.build(repo.iter_histograms(codebook_id))
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    index.save(str(path))
    print(f"   -> índice guardado en {path} | {index.stats()}")
    return index


def buscar_lado_b(qvec: np.ndarray, top: int):
    """Lado B: pgvector. Agrega por canción usando la MIN distancia de sus chunks."""
    with get_conn() as conn:
        register_vector(conn)
        rows = conn.execute(
            "SELECT source_id, MIN(embedding <=> %s) AS dist "
            "FROM embeddings_audio GROUP BY source_id ORDER BY dist LIMIT %s",
            (qvec, top),
        ).fetchall()
    return [(sid, 1.0 - float(dist)) for sid, dist in rows]


def imprimir_ranking(titulo, ranking, top, consulta_id):
    print(f"\n=== {titulo} ===")
    print(f"{'rank':>4}  {'source_id':>16}  {'score':>8}")
    for i, (sid, score) in enumerate(ranking[:top], 1):
        marca = "  <- la consulta" if sid == consulta_id else ""
        print(f"{i:>4}  {sid:>16}  {score:>8.3f}{marca}")
    return [sid for sid, _ in ranking[:top]]


def buscar(args) -> None:
    codebook_id = repo.get_latest_codebook_id("audio")
    if codebook_id is None:
        print("[ERROR] No hay codebook de audio en la BD. ¿Ingestaste primero?")
        return

    builder = KMeansAcousticBuilder()
    codebook = builder.load_from_file(args.model)
    k = codebook.size

    splitter = SlidingWindowSplitter()
    extractor = MfccExtractor()
    consulta_id = f"fma_track_{Path(args.query).stem}"

    chunks = splitter.split(args.query, "__query__")
    descs = extractor.extract(chunks)

    # Histograma global de la consulta (counts agregados sobre todos los chunks).
    query_counts: dict[int, int] = {}
    for d in descs:
        for word_id, count in codebook.encode(d).counts.items():
            query_counts[int(word_id)] = query_counts.get(int(word_id), 0) + count
    query_hist = Histogram(chunk_id="__query__", source_id=consulta_id, counts=query_counts)
    qvec = counts_a_vector(query_counts, k)

    # LADO A (índice propio).
    index = cargar_o_construir_indice(codebook_id)
    t0 = time.perf_counter()
    res_a = index.search(query_hist, k=args.top)
    t_a = time.perf_counter() - t0
    ranking_a = [(r.source_id, r.score) for r in res_a]

    # LADO B (pgvector).
    t0 = time.perf_counter()
    ranking_b = buscar_lado_b(qvec, args.top)
    t_b = time.perf_counter() - t0

    top_a = imprimir_ranking("LADO A — índice invertido propio", ranking_a, args.top, consulta_id)
    top_b = imprimir_ranking("LADO B — pgvector (HNSW)", ranking_b, args.top, consulta_id)

    print("\n--- COMPARACIÓN ---")
    print(f"Tiempo Lado A (índice propio) : {t_a*1000:.1f} ms")
    print(f"Tiempo Lado B (pgvector)      : {t_b*1000:.1f} ms")
    comunes = set(top_a) & set(top_b)
    print(f"Coincidencias en top {args.top}      : {len(comunes)}/{args.top} "
          f"({', '.join(sorted(comunes)) if comunes else 'ninguna'})")


def main() -> None:
    p = argparse.ArgumentParser(description="Búsqueda de audio: Lado A vs Lado B")
    p.add_argument("--query", default="data/raw/fma_small/000/000005.mp3",
                   help="Ruta del MP3 de consulta")
    p.add_argument("--model", default=MODEL_PATH)
    p.add_argument("--top", type=int, default=5)
    args = p.parse_args()
    try:
        buscar(args)
    finally:
        close_pool()


if __name__ == "__main__":
    main()
