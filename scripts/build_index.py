"""Construye el índice invertido del Lado A y lo guarda en disco.
OWNER: Ing. Imágenes + Tech Lead.

Lee los histogramas de la BD (tabla histograms) para un codebook, construye el
VisualInvertedIndex en memoria y lo VUELCA a models/image/index_image.pkl.
Se corre UNA vez después de la ingesta. Luego la búsqueda del Lado A carga este
archivo en vez de reconstruir.

Uso (Postgres levantado, .venv activado, desde la raíz):
    python -m scripts.build_index                      # codebook de imagen más reciente
    python -m scripts.build_index --codebook-id 3      # un codebook concreto
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.image.index import VisualInvertedIndex
from src.db import repositories as repo
from src.db.connection import close_pool

MODELS_DIR = Path("models/image")


def construir(args) -> None:
    # 1) Determinar qué codebook usar.
    codebook_id = args.codebook_id or repo.get_latest_codebook_id("image")
    if codebook_id is None:
        print("[ERROR] No hay codebook de imagen en la BD. ¿Ingestaste primero?")
        return
    print(f"Construyendo índice del codebook id={codebook_id}...")

    # 2) Cargar histogramas y construir el índice (midiendo el tiempo).
    t0 = time.perf_counter()
    index = VisualInvertedIndex()
    index.build(repo.iter_histograms(codebook_id))
    t_build = time.perf_counter() - t0

    stats = index.stats()
    if stats["n_chunks"] == 0:
        print("[ERROR] No se encontraron histogramas para ese codebook.")
        return

    # 3) Guardar a disco (midiendo el tiempo y el tamaño).
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    out = MODELS_DIR / "index_image.pkl"
    t0 = time.perf_counter()
    index.save(str(out))
    t_save = time.perf_counter() - t0
    tam_mb = out.stat().st_size / (1024 * 1024)

    # 4) Reporte (datos útiles para la Fase 4).
    print("\n--- ÍNDICE CONSTRUIDO ---")
    print(f"Chunks indexados     : {stats['n_chunks']}")
    print(f"Visual words usadas  : {stats['n_visual_words']}")
    print(f"Postings totales     : {stats['n_postings']}")
    print(f"Tiempo de build      : {t_build*1000:.1f} ms")
    print(f"Tiempo de guardado   : {t_save*1000:.1f} ms")
    print(f"Tamaño en disco      : {tam_mb:.2f} MB")
    print(f"Guardado en          : {out}")


def main() -> None:
    p = argparse.ArgumentParser(description="Construir índice invertido (Lado A)")
    p.add_argument("--codebook-id", type=int, default=None,
                   help="Id del codebook (por defecto, el más reciente de imagen)")
    args = p.parse_args()
    try:
        construir(args)
    finally:
        close_pool()


if __name__ == "__main__":
    main()