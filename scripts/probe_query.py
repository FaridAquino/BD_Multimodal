"""Búsqueda visual comparando LADO A (índice propio) vs LADO B (pgvector).
OWNER: Ing. Backend.

Procesa una imagen de consulta con el MISMO pipeline (split -> SIFT -> encode con
el codebook guardado) y busca por los dos lados:
  - Lado A: índice invertido propio cargado de models/image/index_image.pkl
  - Lado B: pgvector (operador <=>, distancia coseno) sobre embeddings_image
Ambos agregan los patches por imagen origen. Muestra los rankings lado a lado y
los tiempos, que es la base de la comparación de la Fase 4.

Uso (Postgres levantado, .venv activado, desde la raíz):
    python -m scripts.probe_query --image data/raw/fashion-dataset/images/10000.jpg --top 5
    python -m scripts.probe_query --image otra.jpg --top 5
"""
from __future__ import annotations

import argparse
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2  # noqa: E402
import numpy as np  # noqa: E402
from pgvector.psycopg import register_vector  # noqa: E402

from src.image.splitter import PatchSplitter        # noqa: E402
from src.image.extractor import SiftExtractor        # noqa: E402
from src.image.codebook import VisualCodebook        # noqa: E402
from src.image.index import VisualInvertedIndex      # noqa: E402
from src.db.connection import get_conn, close_pool   # noqa: E402

MODELS_DIR = Path("models/image")


def counts_a_vector(counts: dict, k: int) -> np.ndarray:
    vec = np.zeros(k, dtype=np.float32)
    for cw, freq in counts.items():
        vec[int(cw)] = freq
    return vec


def buscar_lado_b(descs, codebook, k, knn) -> tuple[dict, dict]:
    """Lado B: pgvector. Devuelve (puntajes, aportes) por imagen."""
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
                "FROM embeddings_image ORDER BY dist LIMIT %s",
                (qvec, knn),
            ).fetchall()
            for source_id, dist in res:
                puntajes[source_id] += 1.0 - float(dist)
                aportes[source_id] += 1
    return puntajes, aportes


def buscar_lado_a(descs, codebook, index, knn) -> tuple[dict, dict]:
    """Lado A: índice invertido propio. Devuelve (puntajes, aportes) por imagen."""
    puntajes = defaultdict(float)
    aportes = defaultdict(int)
    for d in descs:
        h = codebook.encode(d)
        if not h.counts:
            continue
        # El índice usa coseno: mayor score = más parecido.
        resultados = index.search(h, k=knn)
        for r in resultados:
            puntajes[r.source_id] += r.score      # ya es similitud (coseno)
            aportes[r.source_id] += 1
    return puntajes, aportes


def imprimir_ranking(titulo, puntajes, aportes, top, consulta_id):
    ranking = sorted(puntajes.items(), key=lambda x: x[1], reverse=True)
    print(f"\n=== {titulo} ===")
    print(f"{'rank':>4}  {'source_id':>12}  {'puntaje':>8}  {'patches':>7}")
    for i, (sid, score) in enumerate(ranking[:top], 1):
        marca = "  <- la consulta" if sid == consulta_id else ""
        print(f"{i:>4}  {sid:>12}  {score:>8.3f}  {aportes[sid]:>7}{marca}")
    return [sid for sid, _ in ranking[:top]]


def buscar(args) -> None:
    img = cv2.imread(args.image)
    if img is None:
        print(f"[ERROR] No se pudo cargar: {args.image}")
        return
    consulta_id = Path(args.image).stem

    # Recuperar metadata del codebook (k, rejilla).
    with get_conn() as conn:
        row = conn.execute(
            "SELECT id, k, params FROM codebooks WHERE modality='image' "
            "ORDER BY id DESC LIMIT 1"
        ).fetchone()
    if row is None:
        print("[ERROR] No hay codebook de imagen. ¿Ingestaste primero?")
        return
    _, k, params = row
    rows_g = params.get("rows", args.rows)
    cols_g = params.get("cols", args.cols)
    print(f"Consulta: {consulta_id} | k={k} | rejilla={rows_g}x{cols_g}")

    # Cargar codebook (.npy) e índice propio (.pkl).
    codebook = VisualCodebook.from_file(str(MODELS_DIR / "codebook_image.npy"))
    index_path = MODELS_DIR / "index_image.pkl"
    if not index_path.exists():
        print(f"[ERROR] No existe {index_path}. Corre antes: python -m scripts.build_index")
        return
    index = VisualInvertedIndex.load(str(index_path))

    # Procesar la consulta UNA vez (mismo pipeline para ambos lados).
    splitter = PatchSplitter(rows=rows_g, cols=cols_g)
    extractor = SiftExtractor()
    chunks = splitter.split(img, source_id="__query__")
    descs = extractor.extract(chunks)

    # LADO A (índice propio).
    t0 = time.perf_counter()
    pa, aa = buscar_lado_a(descs, codebook, index, args.knn)
    t_a = time.perf_counter() - t0

    # LADO B (pgvector).
    t0 = time.perf_counter()
    pb, ab = buscar_lado_b(descs, codebook, k, args.knn)
    t_b = time.perf_counter() - t0

    top_a = imprimir_ranking("LADO A — índice invertido propio", pa, aa, args.top, consulta_id)
    top_b = imprimir_ranking("LADO B — pgvector (HNSW)", pb, ab, args.top, consulta_id)

    # Comparación.
    print("\n--- COMPARACIÓN ---")
    print(f"Tiempo Lado A (índice propio) : {t_a*1000:.1f} ms")
    print(f"Tiempo Lado B (pgvector)      : {t_b*1000:.1f} ms")
    comunes = set(top_a) & set(top_b)
    print(f"Coincidencias en top {args.top}      : {len(comunes)}/{args.top} "
          f"({', '.join(sorted(comunes)) if comunes else 'ninguna'})")


def main() -> None:
    p = argparse.ArgumentParser(description="Búsqueda visual: Lado A vs Lado B")
    p.add_argument("--image", required=True, help="Ruta de la imagen de consulta")
    p.add_argument("--top", type=int, default=10)
    p.add_argument("--knn", type=int, default=10,
                   help="Vecinos por patch a recuperar de cada lado")
    p.add_argument("--rows", type=int, default=3)
    p.add_argument("--cols", type=int, default=3)
    args = p.parse_args()
    try:
        buscar(args)
    finally:
        close_pool()


if __name__ == "__main__":
    main()