"""Demo del pipeline completo de audio sobre una sola pista (smoke test manual).

Mismo patrón que scripts/ingest_fma.py: split -> chunks -> MFCC -> encode ->
histograms (Lado A) + embeddings_audio (Lado B) -> índice invertido en memoria.
"""
import os

import numpy as np

from src.audio.splitter import SlidingWindowSplitter
from src.audio.extractor import MfccExtractor
from src.audio.codebook import KMeansAcousticBuilder
from src.audio.index import AcousticInvertedIndex
from src.db import repositories as repo
from src.db.connection import close_pool


def counts_a_vector(counts: dict, k: int) -> np.ndarray:
    vec = np.zeros(k, dtype=np.float32)
    for cw, freq in counts.items():
        vec[int(cw)] = freq
    return vec


def main():
    print("=== Iniciando Pipeline Completo de Audio ===")

    file_path = "data/raw/fma_small/000/000005.mp3"
    source_id = "test_track_000005"
    ruta_modelo = "models/audio/kmeans_256_fma.joblib"

    if not os.path.exists(file_path):
        print(f"Error: No se encuentra el archivo {file_path}")
        return
    if not os.path.exists(ruta_modelo):
        print(f"ERROR: No se encontró el modelo en {ruta_modelo}. Corre train_kmeans.py primero.")
        return

    # 1. Codebook universal + registro en BD.
    builder = KMeansAcousticBuilder()
    codebook = builder.load_from_file(ruta_modelo)
    k = codebook.size
    print(f"1. Codebook global cargado con {k} centroides.")
    codebook_id = repo.register_codebook(
        modality="audio", k=k,
        params={"window_ms": 150, "hop_ms": 75, "n_mfcc": 20},
    )

    # 2. Split del audio.
    splitter = SlidingWindowSplitter()
    chunks = splitter.split(file_path, source_id)
    print(f"2. {len(chunks)} chunks generados.")

    # 3. Metadatos a BD (chunk_ids reales de Postgres).
    repo.insert_source(source_id, "audio", file_path, metadata={"title": "Test FMA"})
    chunk_ids = repo.insert_chunks(chunks)

    # 4. Extracción + cuantización.
    extractor = MfccExtractor()
    descs = extractor.extract(chunks)

    hist_rows, emb_rows, histograms = [], [], []
    for chunk_id, d in zip(chunk_ids, descs):
        h = codebook.encode(d)
        hist_rows.append((chunk_id, codebook_id, source_id, h.counts))
        emb_rows.append((chunk_id, codebook_id, source_id, counts_a_vector(h.counts, k)))
        h.chunk_id = str(chunk_id)
        histograms.append(h)
    print(f"4. {len(histograms)} histogramas generados.")

    # 5. Inserción Lado A + Lado B.
    repo.insert_histograms(hist_rows)
    repo.insert_embeddings_audio(emb_rows)
    print("5. Inserción exitosa en histograms y embeddings_audio.")

    # 6. Índice invertido propio (Lado A en memoria).
    index = AcousticInvertedIndex()
    index.build(histograms)
    resultados = index.search(histograms[0], k=3)
    top = resultados[0].source_id if resultados else "N/A"
    print(f"6. Búsqueda de prueba completada. Top 1 ID: {top}")

    print("\n=== Validación Exitosa: pipeline conectado ===")
    close_pool()


if __name__ == "__main__":
    main()
