"""Construye el índice invertido del Lado A (imagen o texto) y lo guarda en disco."""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.image.index import VisualInvertedIndex     # noqa: E402
from src.text.index.spimi import SpimiIndex          # noqa: E402
from src.audio.index import AcousticInvertedIndex    # noqa: E402
from src.db import repositories as repo              # noqa: E402
from src.db.connection import close_pool             # noqa: E402

_CONFIG = {
    "image": (VisualInvertedIndex, Path("models/image/index_image.pkl")),
    "text": (SpimiIndex, Path("models/text/index_text.pkl")),
    "audio": (AcousticInvertedIndex, Path("models/audio/index_audio.pkl")),
}


def construir(args) -> None:
    index_cls, out = _CONFIG[args.modality]

    codebook_id = args.codebook_id or repo.get_latest_codebook_id(args.modality)
    if codebook_id is None:
        print(f"[ERROR] No hay codebook de {args.modality} en la BD. ¿Ingestaste primero?")
        return
    print(f"Construyendo índice ({args.modality}) del codebook id={codebook_id}...")

    t0 = time.perf_counter()
    index = index_cls()
    index.build(repo.iter_histograms(codebook_id))
    t_build = time.perf_counter() - t0

    stats = index.stats()
    if stats["n_chunks"] == 0:
        print("[ERROR] No se encontraron histogramas para ese codebook.")
        return

    out.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    index.save(str(out))
    t_save = time.perf_counter() - t0
    tam_mb = out.stat().st_size / (1024 * 1024)

    print("\n--- ÍNDICE CONSTRUIDO ---")
    print(f"Chunks indexados     : {stats['n_chunks']}")
    print("Términos/visual words: "
          f"{stats.get('n_terms') or stats.get('n_visual_words') or stats.get('n_acoustic_words')}")
    print(f"Postings totales     : {stats['n_postings']}")
    print(f"Tiempo de build      : {t_build*1000:.1f} ms")
    print(f"Tiempo de guardado   : {t_save*1000:.1f} ms")
    print(f"Tamaño en disco      : {tam_mb:.2f} MB")
    print(f"Guardado en          : {out}")


def main() -> None:
    p = argparse.ArgumentParser(description="Construir índice invertido (Lado A)")
    p.add_argument("--modality", choices=["image", "text", "audio"], default="image")
    p.add_argument("--codebook-id", type=int, default=None)
    args = p.parse_args()
    try:
        construir(args)
    finally:
        close_pool()


if __name__ == "__main__":
    main()
