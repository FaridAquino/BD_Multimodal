"""Prueba del PatchSplitter sobre una imagen real.

Verifica que el splitter genera el número correcto de chunks (rows x cols),
que cada chunk tiene su patch y metadatos, e (opcional) guarda algunos patches
recortados para inspección visual.

Uso (desde la raíz del repo, con el .venv activado):
    python -m scripts.probes.image.probe_splitter --image data/raw/fashion-dataset/images/1163.jpg --rows 3 --cols 3 --save-patches
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2

from src.image.splitter import PatchSplitter


def main() -> None:
    p = argparse.ArgumentParser(description="Prueba del PatchSplitter")
    p.add_argument("--image", required=True, help="Ruta de la imagen a partir")
    p.add_argument("--rows", type=int, default=8)
    p.add_argument("--cols", type=int, default=8)
    p.add_argument("--overlap", type=float, default=0.0)
    p.add_argument("--save-patches", action="store_true",
                   help="Guardar los primeros 4 patches como archivos para verlos")
    args = p.parse_args()

    # 1) Cargar la imagen.
    img = cv2.imread(args.image)
    if img is None:
        print(f"[ERROR] No se pudo cargar la imagen: {args.image}")
        return
    h, w = img.shape[:2]
    print(f"Imagen: {args.image}")
    print(f"Dimensiones: {w}x{h} px\n")

    # 2) Crear el splitter y partir la imagen.
    splitter = PatchSplitter(rows=args.rows, cols=args.cols, overlap=args.overlap)
    chunks = splitter.split(img, source_id="prueba_001")

    # 3) Verificaciones.
    esperados = args.rows * args.cols
    print(f"Rejilla: {args.rows}x{args.cols}  ->  esperados: {esperados} chunks")
    print(f"Generados: {len(chunks)} chunks")
    print("OK" if len(chunks) == esperados else "REVISAR: el conteo no coincide")
    print()

    # 4) Inspeccionar los primeros chunks.
    print("Primeros chunks:")
    for ch in chunks[:4]:
        ph, pw = ch.payload.shape[:2]
        print(f"  pos={ch.position:2d}  row={ch.metadata['row']} col={ch.metadata['col']}  "
              f"patch={pw}x{ph}px  bbox={ch.metadata['bbox']}")

    # 5) Tamaño promedio de patch (relevante para SIFT).
    tam = [(c.payload.shape[1], c.payload.shape[0]) for c in chunks]
    avg_w = sum(t[0] for t in tam) / len(tam)
    avg_h = sum(t[1] for t in tam) / len(tam)
    print(f"\nTamaño promedio de patch: ~{avg_w:.0f}x{avg_h:.0f} px")
    if min(avg_w, avg_h) < 64:
        print("AVISO: patches pequeños (<64px); SIFT podría detectar pocos keypoints.")
    else:
        print("Tamaño de patch adecuado para SIFT.")

    # 6) Opcional: guardar algunos patches para verlos.
    if args.save_patches:
        out_dir = Path("patch_preview")
        out_dir.mkdir(exist_ok=True)
        for ch in chunks[:3]:
            fname = out_dir / f"patch_r{ch.metadata['row']}_c{ch.metadata['col']}.jpg"
            cv2.imwrite(str(fname), ch.payload)
        print(f"\nGuardados 3 patches de muestra en: {out_dir}/")


if __name__ == "__main__":
    main()