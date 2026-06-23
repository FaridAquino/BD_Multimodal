"""Ingesta de imágenes en PostgreSQL.  OWNER: Ing. Backend + Tech Lead.

Pipeline completo para la modalidad IMAGEN:
  Pasada 1: listar imágenes (orden determinista) -> split -> SIFT -> recolectar
            descriptores.
  Entrenar: K-Means sobre los descriptores -> codebook (visual words) -> guardar
            en models/ y registrar en la tabla `codebooks`.
  Pasada 2: por cada patch -> encode (histograma) -> INSERT en chunks, histograms
            y embeddings_image.

Uso (Postgres levantado, .venv activado, desde la raíz):
    python -m scripts.ingest --limit 5            # prueba con 5 imágenes
    python -m scripts.ingest --limit 112 --truncate   # ~1K chunks, limpia antes
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from src.image.splitter import PatchSplitter        # noqa: E402
from src.image.extractor import SiftExtractor        # noqa: E402
from src.image.codebook import KMeansVisualBuilder   # noqa: E402
from src.db import repositories as repo              # noqa: E402
from src.db.connection import close_pool             # noqa: E402

MODELS_DIR = Path("models/image")


def listar_imagenes(carpeta: Path) -> list[Path]:
    exts = {".jpg", ".jpeg", ".png"}
    # Orden DETERMINISTA por nombre -> cargas anidadas y reproducibles.
    return sorted(p for p in carpeta.rglob("*") if p.suffix.lower() in exts)


def counts_a_vector(counts: dict[int, int], k: int) -> np.ndarray:
    """Convierte el histograma disperso {visual_word: freq} en vector denso de
    tamaño k (Lado B / pgvector). Rellena con 0 las visual words ausentes."""
    vec = np.zeros(k, dtype=np.float32)
    for cw, freq in counts.items():
        vec[int(cw)] = freq
    return vec


def ingestar(args) -> None:
    rutas = listar_imagenes(Path(args.images))[: args.limit]
    if not rutas:
        print(f"[ERROR] No se encontraron imágenes en: {args.images}")
        return
    print(f"Imágenes a ingestar: {len(rutas)} | rejilla {args.rows}x{args.cols} | k={args.k}")

    if args.truncate:
        repo.truncate_all()
        print("Tablas vaciadas (truncate).")

    splitter = PatchSplitter(rows=args.rows, cols=args.cols)
    extractor = SiftExtractor()

    # ---------- PASADA 1: recolectar descriptores ----------
    por_imagen = []          # [(source_id, uri, chunks, descriptors)]
    todos_descriptores = []
    for ruta in rutas:
        img = cv2.imread(str(ruta))
        if img is None:
            print(f"  aviso: no se pudo leer {ruta.name}, se omite")
            continue
        source_id = ruta.stem
        chunks = splitter.split(img, source_id=source_id)
        descs = extractor.extract(chunks)
        por_imagen.append((source_id, str(ruta), chunks, descs))
        todos_descriptores.extend(descs)

    total_sift = sum(d.vector.shape[0] for d in todos_descriptores if d.vector.size)
    print(f"Pasada 1: {len(por_imagen)} imágenes, {total_sift} descriptores SIFT.")

    # ---------- ENTRENAR CODEBOOK ----------
    builder = KMeansVisualBuilder(k=args.k, sample=args.sample,
                                  use_minibatch=True, seed=args.seed)
    codebook = builder.build(todos_descriptores)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    cb_path = MODELS_DIR / "codebook_image.npy"
    codebook.save(str(cb_path))
    codebook_id = repo.register_codebook(
        modality="image", k=args.k,
        params={"rows": args.rows, "cols": args.cols, "sample": args.sample,
                "seed": args.seed, "centroids_path": str(cb_path)},
    )
    print(f"Codebook entrenado (id={codebook_id}, {codebook.size} visual words) "
          f"-> {cb_path}")

    # ---------- PASADA 2: encode + insertar ----------
    total_chunks = 0
    for source_id, uri, chunks, descs in por_imagen:
        repo.insert_source(source_id, "image", uri, metadata={})
        chunk_ids = repo.insert_chunks(chunks)        # ids en el mismo orden

        hist_rows, emb_rows = [], []
        for chunk_id, d in zip(chunk_ids, descs):
            h = codebook.encode(d)
            hist_rows.append((chunk_id, codebook_id, source_id, h.counts))
            vec = counts_a_vector(h.counts, args.k)
            emb_rows.append((chunk_id, codebook_id, source_id, vec))

        repo.insert_histograms(hist_rows)
        repo.insert_embeddings_image(emb_rows)
        total_chunks += len(chunk_ids)

    print(f"Pasada 2: insertados {total_chunks} chunks (+ histogramas + embeddings).")
    print("\nIngesta completada. Verifica en psql:")
    print('  SELECT count(*) FROM chunks;')
    print('  SELECT count(*) FROM embeddings_image;')


def main() -> None:
    p = argparse.ArgumentParser(description="Ingesta de imágenes en Postgres")
    p.add_argument("--images", default="data/raw/fashion-dataset/images")
    p.add_argument("--limit", type=int, default=5, help="Nº de imágenes a procesar")
    p.add_argument("--rows", type=int, default=3)
    p.add_argument("--cols", type=int, default=3)
    p.add_argument("--k", type=int, default=256, help="Tamaño del codebook")
    p.add_argument("--sample", type=int, default=None,
                   help="Muestra de descriptores para K-Means (None = todos)")
    p.add_argument("--truncate", action="store_true",
                   help="Vaciar las tablas antes de ingestar")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    try:
        ingestar(args)
    finally:
        # Cierra el pool ordenadamente para evitar los avisos al salir.
        close_pool()


if __name__ == "__main__":
    main()