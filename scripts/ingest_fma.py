"""Ingesta masiva de audio FMA en PostgreSQL.  OWNER: Ing. Audio + Tech Lead.

Mismo patrón que scripts/ingest.py (imagen), pero para la modalidad AUDIO:
  - Carga el codebook universal entrenado (joblib) por train_kmeans.py.
  - register_codebook("audio", k, params) -> codebook_id.
  - Por cada MP3: split -> insert_source -> insert_chunks (ids) -> MFCC -> encode
    -> INSERT en histograms (Lado A) y embeddings_audio (Lado B, histograma denso).

Uso (Postgres levantado, .venv activado, desde la raíz):
    python -m scripts.ingest_fma --limit 5
    python -m scripts.ingest_fma --truncate          # limpia antes de ingestar
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402

from src.audio.splitter import SlidingWindowSplitter   # noqa: E402
from src.audio.extractor import MfccExtractor           # noqa: E402
from src.audio.codebook import KMeansAcousticBuilder    # noqa: E402
from src.db import repositories as repo                  # noqa: E402
from src.db.connection import close_pool                 # noqa: E402

FMA_DIR = "data/raw/fma_small"
MODEL_PATH = "models/audio/kmeans_256_fma.joblib"


def counts_a_vector(counts: dict[int, int], k: int) -> np.ndarray:
    """Convierte el histograma disperso {acoustic_word: freq} en vector denso de
    tamaño k (Lado B / pgvector). Rellena con 0 las palabras ausentes."""
    vec = np.zeros(k, dtype=np.float32)
    for cw, freq in counts.items():
        vec[int(cw)] = freq
    return vec


def ingestar(args) -> None:
    if not os.path.exists(args.model):
        print(f"[ERROR] No se encontró el modelo {args.model}. Corre train_kmeans primero.")
        return

    # 1. Cargar el codebook universal (joblib) y registrarlo en la BD.
    builder = KMeansAcousticBuilder()
    codebook = builder.load_from_file(args.model)
    k = codebook.size

    if args.truncate:
        repo.truncate_all()
        print("Tablas vaciadas (truncate).")

    codebook_id = repo.register_codebook(
        modality="audio", k=k,
        params={"window_ms": 150, "hop_ms": 75, "n_mfcc": 20,
                "centroids_path": args.model},
    )
    print(f"Codebook registrado (id={codebook_id}, {k} acoustic words).")

    # 2. Herramientas del pipeline.
    splitter = SlidingWindowSplitter()
    extractor = MfccExtractor()

    mp3_files = sorted(Path(args.audio).rglob("*.mp3"))[: args.limit]
    total = len(mp3_files)
    print(f"Canciones a ingestar: {total}\n")

    exitos, errores, total_chunks = 0, 0, 0
    start = time.time()

    for i, file_path in enumerate(mp3_files, 1):
        track_id = file_path.stem
        source_id = f"fma_track_{track_id}"
        print(f"[{i}/{total}] {track_id}...", end=" ", flush=True)
        try:
            chunks = splitter.split(str(file_path), source_id)
            if not chunks:
                raise ValueError("audio muy corto o vacío (0 chunks)")

            repo.insert_source(source_id, "audio", str(file_path),
                               metadata={"fma_id": track_id})
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
    print("\n=== REPORTE DE INGESTA ===")
    print(f"Tiempo total          : {elapsed:.2f} min")
    print(f"Canciones con éxito   : {exitos}")
    print(f"Canciones con error   : {errores}")
    print(f"Chunks insertados     : {total_chunks}")
    print("\nVerifica en psql:")
    print("  SELECT count(*) FROM histograms;")
    print("  SELECT count(*) FROM embeddings_audio;")


def main() -> None:
    p = argparse.ArgumentParser(description="Ingesta de audio FMA en Postgres")
    p.add_argument("--audio", default=FMA_DIR)
    p.add_argument("--model", default=MODEL_PATH)
    p.add_argument("--limit", type=int, default=None, help="Nº de canciones (None = todas)")
    p.add_argument("--truncate", action="store_true", help="Vaciar las tablas antes de ingestar")
    args = p.parse_args()
    try:
        ingestar(args)
    finally:
        close_pool()


if __name__ == "__main__":
    main()
