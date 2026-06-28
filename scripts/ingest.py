"""Ingesta multimodal en PostgreSQL (imagen y texto)."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.db import repositories as repo              # noqa: E402
from src.db.connection import close_pool             # noqa: E402

MODELS_DIR_IMAGE = Path("models/image")
MODELS_DIR_TEXT = Path("models/text")


def listar_imagenes(carpeta: Path) -> list[Path]:
    exts = {".jpg", ".jpeg", ".png"}
    return sorted(p for p in carpeta.rglob("*") if p.suffix.lower() in exts)


def counts_a_vector(counts: dict[int, int], k: int):
    import numpy as np
    vec = np.zeros(k, dtype=np.float32)
    for cw, freq in counts.items():
        vec[int(cw)] = freq
    return vec


def ingestar_imagen(args) -> None:
    import cv2
    from src.image.splitter import PatchSplitter
    from src.image.extractor import SiftExtractor
    from src.image.codebook import KMeansVisualBuilder

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

    por_imagen = []
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

    builder = KMeansVisualBuilder(k=args.k, sample=args.sample,
                                  use_minibatch=True, seed=args.seed)
    codebook = builder.build(todos_descriptores)
    MODELS_DIR_IMAGE.mkdir(parents=True, exist_ok=True)
    cb_path = MODELS_DIR_IMAGE / "codebook_image.npy"
    codebook.save(str(cb_path))
    codebook_id = repo.register_codebook(
        modality="image", k=args.k,
        params={"rows": args.rows, "cols": args.cols, "sample": args.sample,
                "seed": args.seed, "centroids_path": str(cb_path)},
    )
    print(f"Codebook entrenado (id={codebook_id}, {codebook.size} visual words) "
          f"-> {cb_path}")

    total_chunks = 0
    for source_id, uri, chunks, descs in por_imagen:
        repo.insert_source(source_id, "image", uri, metadata={})
        chunk_ids = repo.insert_chunks(chunks)

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


def leer_canciones(csv_path: Path):
    import csv
    csv.field_size_limit(10_000_000)
    with open(csv_path, encoding="utf-8", newline="") as f:
        for i, row in enumerate(csv.DictReader(f)):
            yield (f"song_{i}", row["artist"], row["song"], row["link"], row["text"])


def ingestar_texto(args) -> None:
    import math
    from collections import Counter
    from src.text.splitter import ParagraphSplitter
    from src.text.extractor import TfidfExtractor
    from src.text.codebook import TopKCodebookBuilder

    csv_path = Path(args.csv)
    if not csv_path.exists():
        print(f"[ERROR] No se encontró el CSV: {csv_path}")
        return

    if args.truncate:
        repo.truncate_all()
        print("Tablas vaciadas (truncate).")

    splitter = ParagraphSplitter()
    extractor = TfidfExtractor()

    por_cancion = []
    todos_descriptores = []
    total_chunks = 0
    for source_id, artist, song, link, text in leer_canciones(csv_path):
        chunks = splitter.split(text, source_id=source_id)
        if not chunks:
            continue
        restante = args.max_chunks - total_chunks
        if restante <= 0:
            break
        if len(chunks) > restante:
            chunks = chunks[:restante]
        descs = extractor.extract(chunks)
        metadata = {"artist": artist, "song": song, "link": link}
        por_cancion.append((source_id, metadata, link, chunks, descs))
        todos_descriptores.extend(descs)
        total_chunks += len(chunks)

    if total_chunks == 0:
        print("[ERROR] No se generó ningún chunk. ¿CSV vacío o sin texto?")
        return
    print(f"Pasada 1: {len(por_cancion)} canciones, {total_chunks} chunks (estrofas).")

    builder = TopKCodebookBuilder(k=args.k)
    codebook = builder.build(todos_descriptores)
    MODELS_DIR_TEXT.mkdir(parents=True, exist_ok=True)
    cb_path = MODELS_DIR_TEXT / "codebook_text.json"
    codebook.save(str(cb_path))
    codebook_id = repo.register_codebook(
        modality="text", k=args.k,
        params={"language": "english", "stemmer": "porter", "vocab_path": str(cb_path)},
    )
    print(f"Codebook entrenado (id={codebook_id}, {codebook.size} términos) -> {cb_path}")

    encoded = []
    doc_freq: Counter[int] = Counter()
    n_docs = 0
    for _, _, _, _, descs in por_cancion:
        hs = [codebook.encode(d).counts for d in descs]
        encoded.append(hs)
        for counts in hs:
            n_docs += 1
            for term_id in counts:
                doc_freq[term_id] += 1

    cw_rows = []
    for term, cid in codebook._vocab.items():
        df = doc_freq.get(cid, 0)
        idf = math.log(n_docs / df) if df > 0 else 0.0
        cw_rows.append((cid, term, idf, df))
    repo.insert_codewords_text(codebook_id, cw_rows)
    print(f"codewords_text: {len(cw_rows)} términos con IDF (N={n_docs}).")

    insertados = 0
    for (source_id, metadata, link, chunks, descs), hs in zip(por_cancion, encoded):
        repo.insert_source(source_id, "text", uri=link, metadata=metadata)
        chunk_ids = repo.insert_chunks(chunks)
        hist_rows = [
            (chunk_id, codebook_id, source_id, counts)
            for chunk_id, counts in zip(chunk_ids, hs)
        ]
        repo.insert_histograms(hist_rows)
        insertados += len(chunk_ids)

    print(f"Pasada 2: insertados {insertados} chunks (+ histogramas + tsv).")


def main() -> None:
    p = argparse.ArgumentParser(description="Ingesta multimodal en Postgres")
    p.add_argument("--modality", choices=["image", "text"], default="image")
    p.add_argument("--k", type=int, default=None)
    p.add_argument("--truncate", action="store_true")
    p.add_argument("--images", default="data/raw/fashion-dataset/images")
    p.add_argument("--limit", type=int, default=5)
    p.add_argument("--rows", type=int, default=3)
    p.add_argument("--cols", type=int, default=3)
    p.add_argument("--sample", type=int, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--csv", default="data/raw/spotify_millsongdata.csv")
    p.add_argument("--max-chunks", type=int, default=1000)
    args = p.parse_args()

    if args.k is None:
        args.k = 256 if args.modality == "image" else 5000

    try:
        if args.modality == "image":
            ingestar_imagen(args)
        else:
            ingestar_texto(args)
    finally:
        close_pool()


if __name__ == "__main__":
    main()
