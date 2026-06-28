"""Búsqueda de audio: lado A (índice propio) vs lado B (pgvector)."""
from __future__ import annotations

import argparse
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
from pgvector.psycopg import register_vector  # noqa: E402

from src.audio.splitter import SlidingWindowSplitter  # noqa: E402
from src.audio.extractor import MfccExtractor          # noqa: E402
from src.audio.codebook import KMeansAcousticBuilder   # noqa: E402
from src.audio.index import AcousticInvertedIndex      # noqa: E402
from src.db.connection import get_conn, close_pool     # noqa: E402

MODELS_DIR = Path("models/audio")


def counts_a_vector(counts: dict, k: int) -> np.ndarray:
    vec = np.zeros(k, dtype=np.float32)
    for cw, freq in counts.items():
        vec[int(cw)] = freq
    return vec


def buscar_lado_b(descs, codebook, k, knn) -> tuple[dict, dict]:
    puntajes = defaultdict(float)
    aportes = defaultdict(int)
    with get_conn() as conn:
        register_vector(conn)
        for d in descs:
            h = codebook.encode(d)
            qvec = counts_a_vector(h.counts, k)
            if qvec.sum() == 0:
                continue
            res = conn.execute(
                "SELECT source_id, embedding <=> %s AS dist "
                "FROM embeddings_audio ORDER BY dist LIMIT %s",
                (qvec, knn),
            ).fetchall()
            for source_id, dist in res:
                puntajes[source_id] += 1.0 - float(dist)
                aportes[source_id] += 1
    return puntajes, aportes


def buscar_lado_a(descs, codebook, index, knn) -> tuple[dict, dict]:
    puntajes = defaultdict(float)
    aportes = defaultdict(int)
    for d in descs:
        h = codebook.encode(d)
        if not h.counts:
            continue
        resultados = index.search(h, k=knn)
        for r in resultados:
            puntajes[r.source_id] += r.score
            aportes[r.source_id] += 1
    return puntajes, aportes


def imprimir_ranking(titulo, puntajes, aportes, top, consulta_id):
    ranking = sorted(puntajes.items(), key=lambda x: x[1], reverse=True)
    print(f"\n=== {titulo} ===")
    print(f"{'rank':>4}  {'source_id':>16}  {'puntaje':>8}  {'ventanas':>8}")
    for i, (sid, score) in enumerate(ranking[:top], 1):
        marca = "  <- la consulta" if sid == consulta_id else ""
        print(f"{i:>4}  {sid:>16}  {score:>8.3f}  {aportes[sid]:>8}{marca}")
    return [sid for sid, _ in ranking[:top]]


def buscar(args) -> None:
    consulta_id = f"fma_track_{Path(args.query).stem}"

    with get_conn() as conn:
        row = conn.execute(
            "SELECT id, k, params FROM codebooks WHERE modality='audio' "
            "ORDER BY id DESC LIMIT 1"
        ).fetchone()
    if row is None:
        print("[ERROR] No hay codebook de audio. ¿Ingestaste primero?")
        return
    _, k, params = row
    # Mismos parámetros de ventaneo con los que se ingestó (consistencia consulta/índice).
    window_ms = params.get("window_ms", 150)
    hop_ms = params.get("hop_ms", 750)
    print(f"Consulta: {consulta_id} | k={k} | ventana={window_ms}ms hop={hop_ms}ms")

    codebook = KMeansAcousticBuilder().load_from_file(str(MODELS_DIR / "kmeans_256_fma.joblib"))
    index_path = MODELS_DIR / "index_audio.pkl"
    if not index_path.exists():
        print(f"[ERROR] No existe {index_path}. "
              "Corre antes: python -m scripts.build_index --modality audio")
        return
    index = AcousticInvertedIndex.load(str(index_path))

    splitter = SlidingWindowSplitter(window_ms=window_ms, hop_ms=hop_ms)
    extractor = MfccExtractor()
    chunks = splitter.split(args.query, source_id="__query__")
    descs = extractor.extract(chunks)

    t0 = time.perf_counter()
    pa, aa = buscar_lado_a(descs, codebook, index, args.knn)
    t_a = time.perf_counter() - t0

    t0 = time.perf_counter()
    pb, ab = buscar_lado_b(descs, codebook, k, args.knn)
    t_b = time.perf_counter() - t0

    top_a = imprimir_ranking("LADO A — índice invertido propio", pa, aa, args.top, consulta_id)
    top_b = imprimir_ranking("LADO B — pgvector (HNSW)", pb, ab, args.top, consulta_id)

    print("\n--- COMPARACIÓN ---")
    print(f"Tiempo Lado A (índice propio) : {t_a*1000:.1f} ms")
    print(f"Tiempo Lado B (pgvector)      : {t_b*1000:.1f} ms")
    comunes = set(top_a) & set(top_b)
    print(f"Coincidencias en top {args.top}      : {len(comunes)}/{args.top} "
          f"({', '.join(sorted(comunes)) if comunes else 'ninguna'})")


def main() -> None:
    p = argparse.ArgumentParser(description="Búsqueda de audio: Lado A vs Lado B")
    p.add_argument("--query", required=True, help="Ruta del MP3 de consulta")
    p.add_argument("--top", type=int, default=5)
    p.add_argument("--knn", type=int, default=10)
    args = p.parse_args()
    try:
        buscar(args)
    finally:
        close_pool()


if __name__ == "__main__":
    main()
