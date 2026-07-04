"""Orquestador de pruebas por escala: ingesta -> índices -> evaluación.

Cada escala se carga UNA vez con las tres modalidades juntas y luego se
evalúan las métricas (latencia, throughput, precision/recall, memoria, I/O).
Flujo esperado: correr 1k, revisar resultados, luego 10k y finalmente 100k.

Ejecución desde la raíz del proyecto:
  python -m scripts.run_scale_tests --scale 1k
  python -m scripts.run_scale_tests --scale 10k
  python -m scripts.run_scale_tests --scale 100k

Argumentos útiles:
  --only-modality text|image|audio  repite solo una modalidad (NO trunca la BD)
  --skip-ingest                     salta la ingesta (BD ya cargada a esa escala)
  --no-truncate                     no vacía las tablas antes de ingestar
  --k / --runs / --n-queries        parámetros de la evaluación
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

# Consolas Windows cp1252: no crashear por caracteres especiales en prints.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(errors="replace")

ESCALAS = {"1k": 1_000, "10k": 10_000, "100k": 100_000}
MODALIDADES = ["text", "image", "audio"]
AUDIO_MODEL = Path("models/audio/kmeans_256_fma.joblib")

# --sample acota las filas SIFT usadas por MiniBatchKMeans (si hay menos, usa todas)
IMAGE_KMEANS_SAMPLE = 200_000


def correr(descripcion: str, comando: list[str]) -> None:
    """Ejecuta un comando en subproceso (aísla memoria por fase) y mide tiempo."""
    print(f"\n{'=' * 70}\n>>> {descripcion}\n    {' '.join(comando)}\n{'=' * 70}")
    t0 = time.time()
    resultado = subprocess.run(comando)
    minutos = (time.time() - t0) / 60
    if resultado.returncode != 0:
        print(f"\n[ERROR] Fase falló ({descripcion}) tras {minutos:.1f} min. Abortando.")
        sys.exit(resultado.returncode)
    print(f"\n[OK] {descripcion} completada en {minutos:.1f} min.")


def comandos_ingesta(modality: str, n_chunks: int, truncate: bool) -> list[str]:
    py = [sys.executable, "-m", "scripts.ingest", "--modality", modality,
          "--max-chunks", str(n_chunks)]
    if modality in ("image", "audio"):
        py += ["--limit", "999999"]      # el corte real lo pone --max-chunks
    if modality == "image":
        py += ["--sample", str(IMAGE_KMEANS_SAMPLE)]
    if modality == "audio":
        # hop 1500ms => ~20 chunks/canción => el doble de canciones por
        # presupuesto que el default (750). Más canciones por género eleva
        # el techo de precision@k (verificado: 0.32 -> 0.42 a escala 1k).
        py += ["--hop-ms", "1500"]
    if truncate:
        py += ["--truncate"]
    return py


def main() -> None:
    p = argparse.ArgumentParser(description="Pruebas de escala 1k/10k/100k (3 modalidades)")
    p.add_argument("--scale", required=True, choices=list(ESCALAS))
    p.add_argument("--only-modality", choices=MODALIDADES, default=None,
                   help="Correr solo una modalidad (no trunca la BD).")
    p.add_argument("--skip-ingest", action="store_true",
                   help="Saltar la ingesta (la BD ya está cargada a esta escala).")
    p.add_argument("--no-truncate", action="store_true",
                   help="No vaciar las tablas antes de la primera ingesta.")
    p.add_argument("--k", type=int, default=10)
    p.add_argument("--runs", type=int, default=3)
    p.add_argument("--n-queries", type=int, default=5)
    p.add_argument("--output", default=None,
                   help="Carpeta de salida (default: results/<escala>).")
    args = p.parse_args()

    if args.output is None:
        args.output = f"results/{args.scale}"

    n_chunks = ESCALAS[args.scale]
    modalidades = [args.only_modality] if args.only_modality else MODALIDADES

    # ── Validaciones previas ──────────────────────────────────────────────
    if "audio" in modalidades and not args.skip_ingest and not AUDIO_MODEL.exists():
        print(f"[ERROR] Falta el codebook de audio: {AUDIO_MODEL}\n"
              "        Corre primero: python -m scripts.train_kmeans")
        sys.exit(1)

    truncate = not args.no_truncate and not args.only_modality
    if args.only_modality and not args.skip_ingest:
        print("[AVISO] --only-modality NO trunca la BD: si esta modalidad ya fue "
              "ingestada, habrá chunks duplicados. Usa --skip-ingest o una BD limpia.")

    t_inicio = time.time()
    print(f"\n########## ESCALA {args.scale} ({n_chunks} chunks x "
          f"{len(modalidades)} modalidad(es)) ##########")

    # ── Fase 1: ingesta (la primera modalidad trunca todas las tablas) ────
    if args.skip_ingest:
        print("\n[INFO] --skip-ingest: se asume BD ya cargada a esta escala.")
    else:
        primera = True
        for mod in modalidades:
            correr(f"Ingesta {mod} ({args.scale})",
                   comandos_ingesta(mod, n_chunks, truncate=truncate and primera))
            primera = False

    # ── Fase 2: índices invertidos (Lado A) ───────────────────────────────
    for mod in modalidades:
        correr(f"Índice invertido {mod}",
               [sys.executable, "-m", "scripts.build_index", "--modality", mod])

    # ── Fase 3: evaluación ────────────────────────────────────────────────
    for mod in modalidades:
        correr(f"Evaluación {mod} ({args.scale})",
               [sys.executable, "-m", "scripts.evaluate_performance",
                "--modality", mod, "--scale", args.scale,
                "--k", str(args.k), "--runs", str(args.runs),
                "--n_queries", str(args.n_queries), "--output", args.output])

    total_min = (time.time() - t_inicio) / 60
    print(f"\n########## ESCALA {args.scale} COMPLETA en {total_min:.1f} min ##########")
    print(f"CSVs en: {args.output}/  →  genera gráficos con: "
          f"python -m scripts.plot_results")


if __name__ == "__main__":
    main()
