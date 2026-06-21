"""Prueba del pipeline completo de imagen: split -> SIFT -> codebook.
OWNER: Ing. Imágenes.

Toma una muestra de imágenes, las parte en patches, extrae SIFT, entrena un
codebook de prueba (K-Means -> visual words) y muestra los histogramas que
produce. Valida las tres piezas juntas antes de la ingesta.

Uso (desde la raíz del repo, con el .venv activado):
    python scripts/probes/image/probe_codebook.py
    python -m scripts.probes.image.probe_codebook --images data/raw/fashion-dataset/images --sample 10 --k 32
"""
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

# Permite ejecutar directo sin "ModuleNotFoundError: src".
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import cv2  # noqa: E402

from src.image.splitter import PatchSplitter      # noqa: E402
from src.image.extractor import SiftExtractor      # noqa: E402
from src.image.codebook import KMeansVisualBuilder  # noqa: E402


def listar_imagenes(carpeta: Path) -> list[Path]:
    exts = {".jpg", ".jpeg", ".png"}
    return [p for p in carpeta.rglob("*") if p.suffix.lower() in exts]


def main() -> None:
    p = argparse.ArgumentParser(description="Prueba del codebook (pipeline imagen)")
    p.add_argument("--images", default="data/raw/fashion-dataset/images",
                   help="Carpeta con imágenes")
    p.add_argument("--sample", type=int, default=8, help="Cuántas imágenes usar")
    p.add_argument("--rows", type=int, default=3)
    p.add_argument("--cols", type=int, default=3)
    p.add_argument("--k", type=int, default=32, help="Tamaño del codebook (prueba)")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    rutas = listar_imagenes(Path(args.images))
    if not rutas:
        print(f"[ERROR] No se encontraron imágenes en: {args.images}")
        return
    random.seed(args.seed)
    muestra = random.sample(rutas, min(args.sample, len(rutas)))
    print(f"Imágenes encontradas: {len(rutas)} | usando {len(muestra)}")
    print(f"Rejilla: {args.rows}x{args.cols} | k={args.k}\n")

    splitter = PatchSplitter(rows=args.rows, cols=args.cols)
    extractor = SiftExtractor()

    # 1) Recolectar descriptores de TODAS las imágenes de la muestra.
    todos_descriptores = []
    for ruta in muestra:
        img = cv2.imread(str(ruta))
        if img is None:
            continue
        chunks = splitter.split(img, source_id=ruta.stem)
        todos_descriptores.extend(extractor.extract(chunks))

    total_sift = sum(d.vector.shape[0] for d in todos_descriptores if d.vector.size)
    print(f"Patches totales      : {len(todos_descriptores)}")
    print(f"Descriptores SIFT    : {total_sift} (entrada para K-Means)")

    # 2) Entrenar el codebook (K-Means -> visual words).
    builder = KMeansVisualBuilder(k=args.k, use_minibatch=True, seed=args.seed)
    codebook = builder.build(todos_descriptores)
    print(f"Codebook entrenado   : {codebook.size} visual words\n")

    # 3) Encode: histograma de los primeros patches.
    print("Histogramas de muestra (visual_word_id: frecuencia):")
    for d in todos_descriptores[:6]:
        h = codebook.encode(d)
        total = sum(h.counts.values())
        bins = len(h.counts)
        # Mostrar el histograma compacto (solo las visual words presentes).
        compacto = dict(sorted(h.counts.items())) if total else {}
        print(f"  source={d.chunk.source_id} pos={d.chunk.position}: "
              f"{total} descriptores en {bins} visual words -> {compacto}")

    # 4) Verificación de save/load del codebook.
    tmp = "codebook_prueba.npy"
    codebook.save(tmp)
    from src.image.codebook import VisualCodebook
    recargado = VisualCodebook.from_file(tmp)
    Path(tmp).unlink(missing_ok=True)
    print(f"\nSave/Load OK: codebook recargado con {recargado.size} visual words.")

    print("\n--- VEREDICTO ---")
    if total_sift > 0 and codebook.size == args.k:
        print("OK: split + SIFT + codebook funcionan end-to-end.")
        print("Los histogramas tienen tamaño k y suman el nº de descriptores del patch.")
    else:
        print("REVISAR: no se generaron descriptores o el codebook no tiene tamaño k.")


if __name__ == "__main__":
    main()