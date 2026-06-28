"""Entrena el codebook K-Means de audio (acoustic words) sobre MFCC.

Por defecto entrena sobre las MUESTRAS del repo (data/samples/audio), para que
cualquiera pueda generar SU propio codebook sin descargar el dataset completo:

    python -m scripts.train_kmeans
    python -m scripts.train_kmeans --audio data/raw/fma_small --sample-size 500

El modelo se guarda en models/audio/kmeans_256_fma.joblib (lo usa la ingesta de
audio).  OWNER: Ing. Audio.
"""
from __future__ import annotations

import argparse
import os
import random

import joblib
import librosa
import numpy as np
from pathlib import Path
from sklearn.cluster import MiniBatchKMeans


def train_global_vocabulary(audio_dir: str, output_path: str, k: int,
                            sample_size: int, n_mfcc: int = 20,
                            batch_size: int = 10000) -> None:
    print("1. Buscando archivos MP3...")
    all_mp3_files = list(Path(audio_dir).rglob("*.mp3"))
    if not all_mp3_files:
        print(f"[ERROR] No se encontraron .mp3 en {audio_dir}.")
        return
    print(f"   {len(all_mp3_files)} pistas encontradas en {audio_dir}.")

    n = min(sample_size, len(all_mp3_files))
    sampled_files = random.sample(all_mp3_files, n)
    print(f"   {n} pistas seleccionadas para entrenar.")

    kmeans = MiniBatchKMeans(n_clusters=k, batch_size=batch_size, random_state=42)
    features_buffer = []
    processed_count = 0
    error_count = 0

    print("\n2. Extrayendo MFCC y entrenando por lotes...")
    for i, file_path in enumerate(sampled_files, 1):
        try:
            y, sr = librosa.load(file_path, sr=None, duration=30.0)
            mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc).T
            features_buffer.append(mfccs)
            processed_count += 1
            if len(features_buffer) >= 50:
                kmeans.partial_fit(np.vstack(features_buffer))
                features_buffer = []
                print(f"   -> Progreso: {i}/{n} canciones...")
        except Exception:
            error_count += 1
            print(f"   aviso: error leyendo {Path(file_path).name} (se ignora)")

    if features_buffer:
        kmeans.partial_fit(np.vstack(features_buffer))

    print("\n3. Entrenamiento completado.")
    print(f"   Pistas exitosas: {processed_count} | corruptas/ignoradas: {error_count}")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    joblib.dump(kmeans, output_path)
    print(f"\nModelo guardado en: {output_path}  ({k} acoustic words)")


def main() -> None:
    p = argparse.ArgumentParser(description="Entrena el codebook K-Means de audio")
    p.add_argument("--audio", default="data/samples/audio",
                   help="Carpeta con los MP3 de entrenamiento (def: muestras del repo)")
    p.add_argument("--output", default="models/audio/kmeans_256_fma.joblib",
                   help="Ruta de salida del modelo joblib")
    p.add_argument("--k", type=int, default=256, help="Nº de clusters (acoustic words)")
    p.add_argument("--sample-size", type=int, default=500,
                   help="Máx. de pistas a usar (se acota al total disponible)")
    p.add_argument("--n-mfcc", type=int, default=20)
    args = p.parse_args()
    train_global_vocabulary(args.audio, args.output, args.k, args.sample_size, args.n_mfcc)


if __name__ == "__main__":
    main()
