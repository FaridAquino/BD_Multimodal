"""Prueba del SiftExtractor sobre patches reales.

Encadena PatchSplitter -> SiftExtractor sobre una imagen y reporta cuántos
descriptores SIFT (de 128 dim) genera cada patch. Sirve para confirmar que SIFT
rinde sobre los patches de la rejilla elegida (3x3 ~350px) antes de avanzar.

Uso (desde la raíz del repo, con el .venv activado):
    python -m scripts.probes.image.probe_extractor --image data/raw/fashion-dataset/images/1163.jpg --rows 3 --cols 3
"""
from __future__ import annotations

import argparse

import cv2

from src.image.splitter import PatchSplitter
from src.image.extractor import SiftExtractor


def main() -> None:
    p = argparse.ArgumentParser(description="Prueba del SiftExtractor")
    p.add_argument("--image", default="data/raw/fashion-dataset/images/1163.jpg",
                   help="Ruta de la imagen")
    p.add_argument("--rows", type=int, default=3)
    p.add_argument("--cols", type=int, default=3)
    p.add_argument("--n-features", type=int, default=0,
                   help="Límite de keypoints por patch (0 = sin límite)")
    args = p.parse_args()

    img = cv2.imread(args.image)
    if img is None:
        print(f"[ERROR] No se pudo cargar la imagen: {args.image}")
        return
    h, w = img.shape[:2]
    print(f"Imagen: {args.image}  ({w}x{h} px)")
    print(f"Rejilla: {args.rows}x{args.cols}\n")

    # 1) Split en patches.
    splitter = PatchSplitter(rows=args.rows, cols=args.cols)
    chunks = splitter.split(img, source_id="prueba_001")

    # 2) Extraer descriptores SIFT de cada patch.
    extractor = SiftExtractor(n_features=args.n_features)
    descriptors = extractor.extract(chunks)

    # 3) Reportar por patch.
    print("Descriptores SIFT por patch:")
    conteos = []
    for d in descriptors:
        n_desc = d.vector.shape[0]      # filas = nº de keypoints
        dim = d.vector.shape[1] if d.vector.size else 0
        conteos.append(n_desc)
        r, c = d.chunk.metadata["row"], d.chunk.metadata["col"]
        marca = "  <- patch sin keypoints (zona lisa)" if n_desc == 0 else ""
        print(f"  patch r{r}c{c}: {n_desc:4d} descriptores (dim={dim}){marca}")

    # 4) Resumen.
    total = sum(conteos)
    print("\n--- RESUMEN ---")
    print(f"Patches procesados   : {len(descriptors)}")
    print(f"Descriptores totales : {total}")
    print(f"Promedio por patch   : {total/len(descriptors):.1f}")
    print(f"Min / Max por patch  : {min(conteos)} / {max(conteos)}")
    print("Dimensión SIFT       : 128 (estándar)")

    # 5) Veredicto.
    print("\n--- VEREDICTO ---")
    vacios = sum(1 for n in conteos if n == 0)
    if total == 0:
        print("PROBLEMA: SIFT no detectó ningún descriptor. Revisa resolución/imagen.")
    elif vacios > len(conteos) // 2:
        print(f"AVISO: {vacios}/{len(conteos)} patches sin keypoints (muchas zonas lisas).")
        print("Funciona, pero considera que el fondo blanco aporta pocos descriptores.")
    else:
        print("OK: SIFT genera descriptores suficientes sobre estos patches.")
        # Estimación para K-Means.
        print(f"\nEstimación: ~{total} descriptores/imagen. Con N imágenes para")
        print(f"el codebook, K-Means entrenará sobre ~{total}*N vectores de 128 dim.")


if __name__ == "__main__":
    main()