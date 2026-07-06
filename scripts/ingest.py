"""Ingesta multimodal en PostgreSQL (imagen, texto y audio)."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.db import repositories as repo              # noqa: E402
from src.db.connection import close_pool             # noqa: E402

MODELS_DIR_IMAGE = Path("models/image")
MODELS_DIR_TEXT = Path("models/text")
MODELS_DIR_AUDIO = Path("models/audio")   # aquí vive el codebook K-Means y el índice


def listar_imagenes(carpeta: Path) -> list[Path]:
    exts = {".jpg", ".jpeg", ".png"}
    return sorted(p for p in carpeta.rglob("*") if p.suffix.lower() in exts)


def cargar_etiquetas_fashion(images_dir: Path) -> dict[str, dict]:
    """Lee styles.csv (carpeta padre de images/) -> {id: metadata con articleType}.

    La etiqueta articleType es el ground truth de relevancia para
    precision/recall en la evaluación de imagen.
    """
    import csv
    styles = images_dir.parent / "styles.csv"
    if not styles.exists():
        return {}
    etiquetas: dict[str, dict] = {}
    with open(styles, encoding="utf-8", newline="", errors="replace") as f:
        for row in csv.DictReader(f):
            pid = (row.get("id") or "").strip()
            if not pid:
                continue
            etiquetas[pid] = {
                "articleType": (row.get("articleType") or "").strip(),
                "subCategory": (row.get("subCategory") or "").strip(),
                "productDisplayName": (row.get("productDisplayName") or "").strip(),
            }
    return etiquetas


def cargar_generos_fma(metadata_dir: Path) -> dict[str, str]:
    """Lee fma_metadata/tracks.csv -> {track_id_6digitos: genre_top}.

    genre_top es el ground truth de relevancia para precision/recall en audio.
    """
    tracks_csv = Path(metadata_dir) / "tracks.csv"
    if not tracks_csv.exists():
        return {}
    import pandas as pd
    df = pd.read_csv(tracks_csv, index_col=0, header=[0, 1])
    serie = df[("track", "genre_top")]
    generos: dict[str, str] = {}
    for tid, genero in serie.items():
        if not isinstance(genero, str) or not genero:
            continue
        try:
            # tid puede ser Hashable (índice pandas); str() antes de int lo
            # normaliza y el try descarta filas de cabecera no numéricas
            # (p. ej. 'track_id') que pandas deja como primera fila de datos.
            clave = f"{int(str(tid)):06d}"
        except (ValueError, TypeError):
            continue
        generos[clave] = genero
    return generos


def counts_a_vector(counts: dict[int, int], k: int):
    import numpy as np
    vec = np.zeros(k, dtype=np.float32)
    for cw, freq in counts.items():
        vec[int(cw)] = freq
    return vec


def ingestar_imagen(args) -> None:
    """Ingesta de imagen en 2 pasadas streaming (RAM acotada a 1 imagen).

    Pasada 1: acumula descriptores SIFT SOLO para entrenar el codebook
              (se detiene al juntar --train-rows filas).
    Pasada 2: re-lee cada imagen, encode e inserta de inmediato (patrón audio).
              SIFT se calcula 2 veces: costo aceptado para poder llegar a 100k
              chunks sin agotar la memoria.
    """
    import math
    import time

    import cv2
    from src.image.splitter import PatchSplitter
    from src.image.extractor import SiftExtractor
    from src.image.codebook import KMeansVisualBuilder

    rutas = listar_imagenes(Path(args.images))[: args.limit]
    if not rutas:
        print(f"[ERROR] No se encontraron imágenes en: {args.images}")
        return

    chunks_por_imagen = args.rows * args.cols
    max_chunks = args.max_chunks
    if max_chunks:
        # margen del 5% (mín. 50) por imágenes ilegibles que se omiten
        necesarias = math.ceil(max_chunks / chunks_por_imagen)
        rutas = rutas[: necesarias + max(50, necesarias // 20)]
    print(f"Imágenes candidatas: {len(rutas)} | rejilla {args.rows}x{args.cols} "
          f"| k={args.k} | presupuesto de chunks: {max_chunks or 'sin límite'}")

    if args.truncate:
        repo.truncate_all()
        print("Tablas vaciadas (truncate).")

    etiquetas = cargar_etiquetas_fashion(Path(args.images))
    if etiquetas:
        print(f"Etiquetas de styles.csv: {len(etiquetas)} productos (articleType).")
    else:
        print("[AVISO] styles.csv no encontrado: sources sin articleType "
              "(precision/recall por etiqueta no estará disponible).")

    splitter = PatchSplitter(rows=args.rows, cols=args.cols)
    extractor = SiftExtractor()

    # ── Pasada 1: descriptores para el codebook (streaming, con tope) ──────
    t0 = time.time()
    train_descs = []
    filas_sift = 0
    vistas = 0
    for ruta in rutas:
        img = cv2.imread(str(ruta))
        if img is None:
            continue
        descs = extractor.extract(splitter.split(img, source_id=ruta.stem))
        train_descs.extend(descs)
        filas_sift += sum(d.vector.shape[0] for d in descs if d.vector.size)
        vistas += 1
        if filas_sift >= args.train_rows:
            break
    print(f"Pasada 1: {vistas} imágenes, {filas_sift} descriptores SIFT "
          f"para entrenamiento ({time.time() - t0:.1f}s).")

    builder = KMeansVisualBuilder(k=args.k, sample=args.sample,
                                  use_minibatch=True, seed=args.seed)
    codebook = builder.build(train_descs)
    del train_descs  # liberar RAM antes de la pasada 2
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

    # ── Pasada 2: streaming — re-leer, encode e insertar por imagen ────────
    total_chunks, insertadas, errores = 0, 0, 0
    t0 = time.time()
    for ruta in rutas:
        if max_chunks and total_chunks >= max_chunks:
            break
        img = cv2.imread(str(ruta))
        if img is None:
            errores += 1
            continue
        source_id = ruta.stem
        chunks = splitter.split(img, source_id=source_id)
        if max_chunks:
            chunks = chunks[: max_chunks - total_chunks]
        descs = extractor.extract(chunks)

        repo.insert_source(source_id, "image", str(ruta),
                           metadata=etiquetas.get(source_id, {}))
        chunk_ids = repo.insert_chunks(chunks)

        hist_rows, emb_rows = [], []
        for chunk_id, d in zip(chunk_ids, descs):
            h = codebook.encode(d)
            hist_rows.append((chunk_id, codebook_id, source_id, h.counts))
            emb_rows.append((chunk_id, codebook_id, source_id,
                             counts_a_vector(h.counts, args.k)))

        repo.insert_histograms(hist_rows)
        repo.insert_embeddings_image(emb_rows)
        total_chunks += len(chunk_ids)
        insertadas += 1
        if insertadas % 200 == 0:
            tasa = total_chunks / max(time.time() - t0, 1e-6)
            eta_min = ((max_chunks - total_chunks) / tasa / 60) if max_chunks else 0
            print(f"  [{insertadas} imgs] {total_chunks} chunks "
                  f"| {tasa:.0f} chunks/s | ETA {eta_min:.1f} min")

    elapsed = (time.time() - t0) / 60
    print("\n=== REPORTE DE INGESTA (imagen) ===")
    print(f"Tiempo pasada 2       : {elapsed:.2f} min")
    print(f"Imágenes insertadas   : {insertadas}")
    print(f"Imágenes ilegibles    : {errores}")
    print(f"Chunks insertados     : {total_chunks}")
    if max_chunks and total_chunks < max_chunks:
        print(f"[AVISO] Presupuesto no alcanzado ({total_chunks}/{max_chunks}): "
              "sube --limit o revisa imágenes ilegibles.")


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
        if args.max_chunks:
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
        if len(por_cancion) % 2000 == 0:
            print(f"  ... {len(por_cancion)} canciones, {total_chunks} chunks")

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


def ingestar_audio(args) -> None:
    import os
    import time

    from src.audio.splitter import SlidingWindowSplitter
    from src.audio.extractor import MfccExtractor
    from src.audio.codebook import KMeansAcousticBuilder

    if not os.path.exists(args.model):
        print(f"[ERROR] No se encontró el modelo {args.model}. Corre train_kmeans primero.")
        return

    # 1. Codebook universal (joblib, entrenado por train_kmeans) + registro en BD.
    builder = KMeansAcousticBuilder()
    codebook = builder.load_from_file(args.model)
    k = codebook.size

    if args.truncate:
        repo.truncate_all()
        print("Tablas vaciadas (truncate).")

    codebook_id = repo.register_codebook(
        modality="audio", k=k,
        params={"window_ms": args.window_ms, "hop_ms": args.hop_ms, "n_mfcc": 20,
                "centroids_path": args.model},
    )
    print(f"Codebook registrado (id={codebook_id}, {k} acoustic words).")

    splitter = SlidingWindowSplitter(window_ms=args.window_ms, hop_ms=args.hop_ms)
    extractor = MfccExtractor()

    generos = cargar_generos_fma(Path(args.fma_metadata))
    if generos:
        print(f"Géneros FMA cargados: {len(generos)} tracks (genre_top).")
    else:
        print("[AVISO] tracks.csv de FMA no encontrado: sources sin género "
              "(precision/recall por etiqueta no estará disponible).")

    mp3_files = sorted(Path(args.audio).rglob("*.mp3"))[: args.limit]
    total = len(mp3_files)
    max_chunks = args.max_chunks
    print(f"Canciones a ingestar: {total} "
          f"| presupuesto de chunks: {max_chunks or 'sin límite'}\n")

    exitos, errores, total_chunks = 0, 0, 0
    start = time.time()

    for i, file_path in enumerate(mp3_files, 1):
        if max_chunks and total_chunks >= max_chunks:
            print(f"\nPresupuesto de chunks alcanzado ({total_chunks}).")
            break
        track_id = file_path.stem
        source_id = f"fma_track_{track_id}"
        print(f"[{i}/{total}] {track_id}...", end=" ", flush=True)
        try:
            chunks = splitter.split(str(file_path), source_id)
            if not chunks:
                raise ValueError("audio muy corto o vacío (0 chunks)")
            if max_chunks:
                chunks = chunks[: max_chunks - total_chunks]

            metadata = {"fma_id": track_id}
            if track_id in generos:
                metadata["genre"] = generos[track_id]
            repo.insert_source(source_id, "audio", str(file_path),
                               metadata=metadata)
            chunk_ids = repo.insert_chunks(chunks)        # ids en el mismo orden
            descs = extractor.extract(chunks)

            hist_rows, emb_rows = [], []
            for chunk_id, d in zip(chunk_ids, descs):
                h = codebook.encode(d)
                hist_rows.append((chunk_id, codebook_id, source_id, h.counts))
                vec = counts_a_vector(h.counts, k)
                emb_rows.append((chunk_id, codebook_id, source_id, vec))

            repo.insert_histograms(hist_rows)
            repo.insert_embeddings_audio(emb_rows)
            total_chunks += len(chunk_ids)
            exitos += 1
            print("OK")
        except Exception as e:  # noqa: BLE001 — el bucle continúa con la siguiente
            errores += 1
            print(f"ERROR: {e}")

    elapsed = (time.time() - start) / 60
    print("\n=== REPORTE DE INGESTA (audio) ===")
    print(f"Tiempo total          : {elapsed:.2f} min")
    print(f"Canciones con éxito   : {exitos}")
    print(f"Canciones con error   : {errores}")
    print(f"Chunks insertados     : {total_chunks}")
    if max_chunks and total_chunks < max_chunks:
        print(f"[AVISO] Presupuesto no alcanzado ({total_chunks}/{max_chunks}): "
              "sube --limit o revisa errores de decodificación.")


def main() -> None:
    p = argparse.ArgumentParser(description="Ingesta multimodal en Postgres")
    p.add_argument("--modality", choices=["image", "text", "audio"], default="image")
    p.add_argument("--k", type=int, default=None)
    p.add_argument("--truncate", action="store_true")
    p.add_argument("--images", default="data/raw/fashion-dataset/images")
    p.add_argument("--limit", type=int, default=5)
    p.add_argument("--rows", type=int, default=3)
    p.add_argument("--cols", type=int, default=3)
    p.add_argument("--sample", type=int, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--csv", default="data/raw/spotify_millsongdata.csv")
    p.add_argument("--max-chunks", type=int, default=1000,
                   help="Presupuesto de chunks a insertar (todas las modalidades). "
                        "0 = sin límite.")
    p.add_argument("--train-rows", type=int, default=1_000_000,
                   help="Filas SIFT máximas para entrenar el codebook de imagen "
                        "(acota RAM/tiempo de la pasada 1).")
    p.add_argument("--audio", default="data/raw/fma_small",
                   help="Carpeta con los MP3 (modalidad audio)")
    p.add_argument("--fma-metadata", default="data/raw/fma_metadata",
                   help="Carpeta con tracks.csv de FMA (géneros para ground truth)")
    p.add_argument("--model", default=str(MODELS_DIR_AUDIO / "kmeans_256_fma.joblib"),
                   help="Codebook de audio entrenado (joblib)")
    p.add_argument("--window-ms", type=int, default=150,
                   help="Ancho de la ventana de audio en ms (modalidad audio)")
    p.add_argument("--hop-ms", type=int, default=750,
                   help="Salto entre ventanas en ms; mayor = menos chunks (modalidad audio)")
    args = p.parse_args()

    if args.k is None:
        args.k = 256 if args.modality in ("image", "audio") else 5000

    try:
        if args.modality == "image":
            ingestar_imagen(args)
        elif args.modality == "text":
            ingestar_texto(args)
        else:
            ingestar_audio(args)
    finally:
        close_pool()


if __name__ == "__main__":
    main()
