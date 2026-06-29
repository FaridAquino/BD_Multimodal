"""Evaluación experimental: Lado A (Índice Invertido Propio) vs Lado B (Postgres Nativo).

Métricas capturadas por consulta:
  - latencia_ms      : tiempo de respuesta en milisegundos
  - throughput_qps   : consultas/segundo estimadas (1000 / latencia_ms)
  - ram_delta_mb     : incremento de RAM del proceso durante la búsqueda (psutil)
  - recall_at_k      : fracción del top-k de Lado B encontrada también en Lado A
  - io_shared_hit    : bloques encontrados en cache de Postgres (EXPLAIN ANALYZE BUFFERS)
  - io_shared_read   : bloques leídos del disco por Postgres

Ejecución desde la raíz del proyecto:
  # Texto (letra de canción)
  python -m scripts.evaluate_performance --modality text  --queries "love me tender" "bohemian rhapsody" --n_queries 5 --k 10 --scale 1k --output results/

  # Imagen (ruta a un archivo de imagen)
  python -m scripts.evaluate_performance --modality image --queries data/samples/image/10000.jpg --k 10 --scale 1k --output results/

  # Audio (ruta a un archivo .mp3 / .wav)
  python -m scripts.evaluate_performance --modality audio --queries data/samples/audio/000002.mp3 --k 10 --scale 1k --output results/

Argumentos:
  --modality   text | image | audio
  --queries    lista de consultas (strings para texto, rutas de archivo para imagen/audio)
  --n_queries  si se omite --queries, toma N muestras aleatorias de la BD
  --k          número de resultados (top-k) a recuperar
  --scale      1k | 10k | 100k  — se usa solo como etiqueta en el CSV de salida
  --output     carpeta donde guardar el CSV (se crea si no existe)
  --runs       repeticiones por consulta para promediar latencia (default: 3)
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
import time
import tracemalloc
from collections import defaultdict
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import psutil  # noqa: E402

# ── Importaciones del proyecto ──────────────────────────────────────────────
from src.db.connection import get_conn, close_pool  # noqa: E402
from src.db import repositories as repo             # noqa: E402

# Texto
from src.core import Histogram                          # noqa: E402
from src.text.splitter import ParagraphSplitter         # noqa: E402
from src.text.extractor import TfidfExtractor           # noqa: E402
from src.text.codebook import LinguisticCodebook        # noqa: E402
from src.text.index.spimi import SpimiIndex             # noqa: E402
from src.baselines import gin_gist                      # noqa: E402

# Imagen
from src.image.splitter import PatchSplitter            # noqa: E402
from src.image.extractor import SiftExtractor           # noqa: E402
from src.image.codebook import VisualCodebook           # noqa: E402
from src.image.index import VisualInvertedIndex         # noqa: E402

# Audio
from src.audio.splitter import SlidingWindowSplitter    # noqa: E402
from src.audio.extractor import MfccExtractor           # noqa: E402
from src.audio.codebook import KMeansAcousticBuilder    # noqa: E402
from src.audio.index import AcousticInvertedIndex       # noqa: E402

import numpy as np                                      # noqa: E402
from pgvector.psycopg import register_vector            # noqa: E402

# ── Rutas de modelos ─────────────────────────────────────────────────────────
TEXT_MODELS  = Path("models/text")
IMAGE_MODELS = Path("models/image")
AUDIO_MODELS = Path("models/audio")

# ── Helpers generales ────────────────────────────────────────────────────────

def _proceso_actual():
    return psutil.Process(os.getpid())


def _ram_mb() -> float:
    """Memoria RSS del proceso en MB."""
    return _proceso_actual().memory_info().rss / 1024 / 1024


def _medir(fn, *args, **kwargs):
    """Ejecuta fn(*args, **kwargs) midiendo latencia y delta de RAM."""
    ram_antes = _ram_mb()
    t0 = time.perf_counter()
    resultado = fn(*args, **kwargs)
    latencia_ms = (time.perf_counter() - t0) * 1000
    ram_delta = _ram_mb() - ram_antes
    return resultado, latencia_ms, ram_delta


def _recall_at_k(top_a: list[str], top_b: list[str]) -> float:
    """Fracción de los resultados de Lado B presentes en Lado A (B = ground truth)."""
    if not top_b:
        return 0.0
    comunes = set(top_a) & set(top_b)
    return round(len(comunes) / len(top_b), 4)


def _explain_buffers(sql: str, params: tuple, conn) -> dict[str, int]:
    """
    Ejecuta EXPLAIN (ANALYZE, BUFFERS, FORMAT JSON) sobre la consulta dada y
    extrae 'Shared Hit Blocks' y 'Shared Read Blocks' del plan.
    Retorna {"shared_hit": N, "shared_read": N}.
    """
    plan_sql = f"EXPLAIN (ANALYZE, BUFFERS, FORMAT JSON) {sql}"
    try:
        row = conn.execute(plan_sql, params).fetchone()
        plan = row[0][0]["Plan"]
        return {
            "shared_hit":  plan.get("Shared Hit Blocks", 0),
            "shared_read": plan.get("Shared Read Blocks", 0),
        }
    except Exception:
        return {"shared_hit": -1, "shared_read": -1}


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  MODALIDAD: TEXTO                                                       ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def _encode_text(codebook: LinguisticCodebook, q: str) -> Histogram:
    chunks = ParagraphSplitter().split(q, source_id="query")
    descs = TfidfExtractor().extract(chunks)
    merged: dict[int, int] = {}
    for d in descs:
        for cw, c in codebook.encode(d).counts.items():
            merged[cw] = merged.get(cw, 0) + c
    return Histogram(chunk_id="query", source_id="query", counts=merged)


def _agg_source(results) -> list[str]:
    acc: dict[str, float] = defaultdict(float)
    for r in results:
        acc[r.source_id] = max(acc[r.source_id], r.score)
    return [sid for sid, _ in sorted(acc.items(), key=lambda x: x[1], reverse=True)]


def evaluar_texto(query: str, k: int) -> dict:
    codebook = LinguisticCodebook.load(str(TEXT_MODELS / "codebook_text.json"))
    index    = SpimiIndex.load(str(TEXT_MODELS / "index_text.pkl"))
    qh = _encode_text(codebook, query)

    # ── Lado A ───────────────────────────────────────────────────────────
    def lado_a():
        raw = index.search(qh, k=k * 10)
        return _agg_source(raw)[:k]

    top_a, ms_a, ram_a = _medir(lado_a)

    # ── Lado B (GIN) ─────────────────────────────────────────────────────
    def lado_b():
        res = gin_gist.search_fulltext_aggregated(query, k=k)
        return [r.source_id for r in res]

    top_b, ms_b, ram_b = _medir(lado_b)

    # ── I/O Postgres (GIN fulltext search) ───────────────────────────────
    io_b = {"shared_hit": -1, "shared_read": -1}
    try:
        with get_conn() as conn:
            sql = (
                "SELECT source_id, ts_rank_cd(tsv, query) AS score "
                "FROM chunks, plainto_tsquery('english', %s) query "
                "WHERE tsv @@ query ORDER BY score DESC LIMIT %s"
            )
            io_b = _explain_buffers(sql, (query, k), conn)
    except Exception:
        pass

    return {
        "query":             query,
        "lado_a_latencia_ms": round(ms_a, 3),
        "lado_a_qps":         round(1000 / ms_a, 2) if ms_a > 0 else 0,
        "lado_a_ram_delta_mb": round(ram_a, 4),
        "lado_b_latencia_ms": round(ms_b, 3),
        "lado_b_qps":         round(1000 / ms_b, 2) if ms_b > 0 else 0,
        "lado_b_ram_delta_mb": round(ram_b, 4),
        "lado_b_io_shared_hit":  io_b["shared_hit"],
        "lado_b_io_shared_read": io_b["shared_read"],
        "recall_at_k":        _recall_at_k(top_a, top_b),
        "top_a":              top_a[:k],
        "top_b":              top_b[:k],
    }


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  MODALIDAD: IMAGEN                                                      ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def _encode_image_descs(img_path: str):
    import cv2
    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(f"No se pudo cargar la imagen: {img_path}")
    splitter  = PatchSplitter(rows=3, cols=3)
    extractor = SiftExtractor()
    chunks = splitter.split(img, source_id="query")
    return extractor.extract(chunks)


def _counts_a_vec(counts: dict, dim: int) -> np.ndarray:
    vec = np.zeros(dim, dtype=np.float32)
    for cw, freq in counts.items():
        vec[int(cw)] = freq
    return vec


def evaluar_imagen(img_path: str, k: int) -> dict:
    codebook = VisualCodebook.from_file(str(IMAGE_MODELS / "codebook_image.npy"))
    index    = VisualInvertedIndex.load(str(IMAGE_MODELS / "index_image.pkl"))
    descs    = _encode_image_descs(img_path)

    # ── Lado A ───────────────────────────────────────────────────────────
    def lado_a():
        puntajes: dict[str, float] = defaultdict(float)
        for d in descs:
            h = codebook.encode(d)
            if not h.counts:
                continue
            for r in index.search(h, k=k):
                puntajes[r.source_id] += r.score
        return sorted(puntajes, key=puntajes.__getitem__, reverse=True)[:k]

    top_a, ms_a, ram_a = _medir(lado_a)

    # ── Lado B (pgvector) ─────────────────────────────────────────────────
    with get_conn() as conn:
        row = conn.execute(
            "SELECT k FROM codebooks WHERE modality='image' ORDER BY id DESC LIMIT 1"
        ).fetchone()
    dim = row[0] if row else 512

    def lado_b():
        puntajes: dict[str, float] = defaultdict(float)
        with get_conn() as conn:
            register_vector(conn)
            for d in descs:
                h = codebook.encode(d)
                qvec = _counts_a_vec(h.counts, dim)
                if qvec.sum() == 0:
                    continue
                rows = conn.execute(
                    "SELECT source_id, embedding <=> %s AS dist "
                    "FROM embeddings_image ORDER BY dist LIMIT %s",
                    (qvec, k),
                ).fetchall()
                for sid, dist in rows:
                    puntajes[str(sid)] += 1.0 - float(dist)
        return sorted(puntajes, key=puntajes.__getitem__, reverse=True)[:k]

    top_b, ms_b, ram_b = _medir(lado_b)

    # ── I/O Postgres ──────────────────────────────────────────────────────
    io_b = {"shared_hit": -1, "shared_read": -1}
    if descs:
        h0 = codebook.encode(descs[0])
        qvec0 = _counts_a_vec(h0.counts, dim)
        try:
            with get_conn() as conn:
                register_vector(conn)
                sql = ("SELECT source_id, embedding <=> %s AS dist "
                       "FROM embeddings_image ORDER BY dist LIMIT %s")
                io_b = _explain_buffers(sql, (qvec0, k), conn)
        except Exception:
            pass

    return {
        "query":              img_path,
        "lado_a_latencia_ms": round(ms_a, 3),
        "lado_a_qps":         round(1000 / ms_a, 2) if ms_a > 0 else 0,
        "lado_a_ram_delta_mb": round(ram_a, 4),
        "lado_b_latencia_ms": round(ms_b, 3),
        "lado_b_qps":         round(1000 / ms_b, 2) if ms_b > 0 else 0,
        "lado_b_ram_delta_mb": round(ram_b, 4),
        "lado_b_io_shared_hit":  io_b["shared_hit"],
        "lado_b_io_shared_read": io_b["shared_read"],
        "recall_at_k":         _recall_at_k(top_a, top_b),
        "top_a":               top_a[:k],
        "top_b":               top_b[:k],
    }


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  MODALIDAD: AUDIO                                                       ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def evaluar_audio(audio_path: str, k: int) -> dict:
    with get_conn() as conn:
        row = conn.execute(
            "SELECT id, k, params FROM codebooks WHERE modality='audio' ORDER BY id DESC LIMIT 1"
        ).fetchone()
    if row is None:
        raise RuntimeError("No hay codebook de audio en la BD. Corre el script de ingesta.")
    _, k_cb, params = row
    window_ms = params.get("window_ms", 150)
    hop_ms    = params.get("hop_ms", 75)

    codebook = KMeansAcousticBuilder().load_from_file(
        str(AUDIO_MODELS / "kmeans_256_fma.joblib")
    )
    index    = AcousticInvertedIndex.load(str(AUDIO_MODELS / "index_audio.pkl"))
    splitter = SlidingWindowSplitter(window_ms=window_ms, hop_ms=hop_ms)
    extractor = MfccExtractor()
    chunks   = splitter.split(audio_path, source_id="query")
    descs    = extractor.extract(chunks)

    # ── Lado A ───────────────────────────────────────────────────────────
    def lado_a():
        puntajes: dict[str, float] = defaultdict(float)
        for d in descs:
            h = codebook.encode(d)
            if not h.counts:
                continue
            for r in index.search(h, k=k):
                puntajes[r.source_id] += r.score
        return sorted(puntajes, key=puntajes.__getitem__, reverse=True)[:k]

    top_a, ms_a, ram_a = _medir(lado_a)

    # ── Lado B (pgvector) ─────────────────────────────────────────────────
    def lado_b():
        puntajes: dict[str, float] = defaultdict(float)
        with get_conn() as conn:
            register_vector(conn)
            for d in descs:
                h = codebook.encode(d)
                qvec = _counts_a_vec(h.counts, k_cb)
                if qvec.sum() == 0:
                    continue
                rows = conn.execute(
                    "SELECT source_id, embedding <=> %s AS dist "
                    "FROM embeddings_audio ORDER BY dist LIMIT %s",
                    (qvec, k),
                ).fetchall()
                for sid, dist in rows:
                    puntajes[str(sid)] += 1.0 - float(dist)
        return sorted(puntajes, key=puntajes.__getitem__, reverse=True)[:k]

    top_b, ms_b, ram_b = _medir(lado_b)

    # ── I/O Postgres ──────────────────────────────────────────────────────
    io_b = {"shared_hit": -1, "shared_read": -1}
    if descs:
        h0   = codebook.encode(descs[0])
        qvec0 = _counts_a_vec(h0.counts, k_cb)
        try:
            with get_conn() as conn:
                register_vector(conn)
                sql = ("SELECT source_id, embedding <=> %s AS dist "
                       "FROM embeddings_audio ORDER BY dist LIMIT %s")
                io_b = _explain_buffers(sql, (qvec0, k), conn)
        except Exception:
            pass

    return {
        "query":              audio_path,
        "lado_a_latencia_ms": round(ms_a, 3),
        "lado_a_qps":         round(1000 / ms_a, 2) if ms_a > 0 else 0,
        "lado_a_ram_delta_mb": round(ram_a, 4),
        "lado_b_latencia_ms": round(ms_b, 3),
        "lado_b_qps":         round(1000 / ms_b, 2) if ms_b > 0 else 0,
        "lado_b_ram_delta_mb": round(ram_b, 4),
        "lado_b_io_shared_hit":  io_b["shared_hit"],
        "lado_b_io_shared_read": io_b["shared_read"],
        "recall_at_k":         _recall_at_k(top_a, top_b),
        "top_a":               top_a[:k],
        "top_b":               top_b[:k],
    }


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  HELPERS DE MUESTREO DE CONSULTAS ALEATORIAS                            ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def _sample_text_queries(n: int) -> list[str]:
    """Toma N fragmentos de texto aleatorios de la tabla chunks."""
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT payload FROM chunks WHERE modality='text' "
            "AND payload IS NOT NULL ORDER BY RANDOM() LIMIT %s",
            (n,),
        ).fetchall()
    return [r[0][:200] for r in rows if r[0]]


def _sample_image_paths(n: int) -> list[str]:
    """Toma N rutas de imagen aleatorias de la tabla sources."""
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT uri FROM sources WHERE modality='image' "
            "ORDER BY RANDOM() LIMIT %s",
            (n,),
        ).fetchall()
    return [r[0] for r in rows if r[0] and Path(r[0]).exists()]


def _sample_audio_paths(n: int) -> list[str]:
    """Toma N rutas de audio aleatorias de la tabla sources."""
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT uri FROM sources WHERE modality='audio' "
            "ORDER BY RANDOM() LIMIT %s",
            (n,),
        ).fetchall()
    return [r[0] for r in rows if r[0] and Path(r[0]).exists()]


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  MAIN                                                                   ║
# ╚══════════════════════════════════════════════════════════════════════════╝

EVALUADORES = {
    "text":  evaluar_texto,
    "image": evaluar_imagen,
    "audio": evaluar_audio,
}

SAMPLERS = {
    "text":  _sample_text_queries,
    "image": _sample_image_paths,
    "audio": _sample_audio_paths,
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluación experimental: Lado A vs Lado B",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--modality",  required=True, choices=["text", "image", "audio"])
    parser.add_argument("--queries",   nargs="*", default=[],
                        help="Consultas a evaluar (strings para texto, rutas para imagen/audio). "
                             "Si se omite, se usa --n_queries consultas aleatorias de la BD.")
    parser.add_argument("--n_queries", type=int, default=5,
                        help="Número de consultas aleatorias a tomar de la BD si --queries está vacío.")
    parser.add_argument("--k",         type=int, default=10, help="Top-k de resultados.")
    parser.add_argument("--scale",     default="1k", choices=["1k", "10k", "100k"],
                        help="Etiqueta de escala del experimento (solo afecta el nombre del CSV).")
    parser.add_argument("--output",    default="results",
                        help="Carpeta de salida para el CSV.")
    parser.add_argument("--runs",      type=int, default=3,
                        help="Repeticiones por consulta para promediar latencia.")
    args = parser.parse_args()

    # ── Determinar consultas ──────────────────────────────────────────────
    queries = args.queries
    if not queries:
        print(f"[INFO] --queries vacío. Muestreando {args.n_queries} consultas de la BD...")
        queries = SAMPLERS[args.modality](args.n_queries)
        if not queries:
            print("[ERROR] No se encontraron consultas en la BD. ¿Ingestaste datos?")
            sys.exit(1)
        print(f"[INFO] Consultas seleccionadas: {queries}")

    # ── Preparar salida ───────────────────────────────────────────────────
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = output_dir / f"results_{args.scale}_{args.modality}_{ts}.csv"

    FIELDNAMES = [
        "scale", "modality", "query", "run",
        "lado_a_latencia_ms", "lado_a_qps", "lado_a_ram_delta_mb",
        "lado_b_latencia_ms", "lado_b_qps", "lado_b_ram_delta_mb",
        "lado_b_io_shared_hit", "lado_b_io_shared_read",
        "recall_at_k",
    ]

    evaluador = EVALUADORES[args.modality]
    total = len(queries) * args.runs
    print(f"\n[INFO] Iniciando evaluación: {total} ejecuciones ({len(queries)} consultas × {args.runs} runs)")
    print(f"[INFO] Resultados → {csv_path}\n")

    with open(csv_path, "w", newline="", encoding="utf-8") as fcsv:
        writer = csv.DictWriter(fcsv, fieldnames=FIELDNAMES)
        writer.writeheader()

        for q in queries:
            for run in range(1, args.runs + 1):
                try:
                    print(f"  → [{args.modality.upper()}] run={run} | q={str(q)[:60]!r}")
                    metrics = evaluador(q, args.k)
                    row = {
                        "scale":    args.scale,
                        "modality": args.modality,
                        "query":    str(q)[:120],
                        "run":      run,
                        **{k: v for k, v in metrics.items()
                           if k in FIELDNAMES},
                    }
                    writer.writerow(row)
                    fcsv.flush()  # Escribir de inmediato por si se interrumpe
                    print(f"     Lado A: {metrics['lado_a_latencia_ms']:.1f} ms | "
                          f"Lado B: {metrics['lado_b_latencia_ms']:.1f} ms | "
                          f"Recall@{args.k}: {metrics['recall_at_k']:.2%}")
                except Exception as exc:
                    print(f"  [ERROR] Consulta falló: {exc}")

    print(f"\n[OK] CSV guardado en: {csv_path}")

    # ── Resumen en consola ────────────────────────────────────────────────
    try:
        import statistics
        with open(csv_path, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))

        def _avg(col):
            vals = [float(r[col]) for r in rows if r[col] not in ("", "-1")]
            return statistics.mean(vals) if vals else float("nan")

        print("\n══════════════════════ RESUMEN ══════════════════════")
        print(f"  Escala          : {args.scale}")
        print(f"  Modalidad       : {args.modality}")
        print(f"  Consultas totales: {len(rows)}")
        print(f"\n  Latencia promedio  Lado A : {_avg('lado_a_latencia_ms'):.2f} ms")
        print(f"  Latencia promedio  Lado B : {_avg('lado_b_latencia_ms'):.2f} ms")
        print(f"  Throughput prom.   Lado A : {_avg('lado_a_qps'):.2f} QPS")
        print(f"  Throughput prom.   Lado B : {_avg('lado_b_qps'):.2f} QPS")
        print(f"  RAM delta prom.    Lado A : {_avg('lado_a_ram_delta_mb'):.4f} MB")
        print(f"  RAM delta prom.    Lado B : {_avg('lado_b_ram_delta_mb'):.4f} MB")
        print(f"  IO shared_hit prom Lado B : {_avg('lado_b_io_shared_hit'):.0f} bloques")
        print(f"  IO shared_read pro Lado B : {_avg('lado_b_io_shared_read'):.0f} bloques")
        print(f"  Recall@{args.k} promedio   : {_avg('recall_at_k'):.2%}")
        print("═════════════════════════════════════════════════════")
    except Exception:
        pass

    close_pool()


if __name__ == "__main__":
    main()
