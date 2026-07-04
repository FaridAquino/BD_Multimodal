"""Evaluación experimental: Lado A (Índice Invertido Propio) vs Lado B (Postgres Nativo).

Métricas capturadas por consulta (para Lado A y Lado B):
  - latencia_ms      : tiempo de respuesta en milisegundos
  - throughput_qps   : consultas/segundo estimadas (1000 / latencia_ms)
  - ram_delta_mb     : incremento de RAM del proceso durante la búsqueda (psutil)
  - mem_pico_mb      : pico de memoria Python durante la búsqueda (tracemalloc)
  - precision        : TP / (TP+FP) = TP / k, con relevancia por etiqueta del
                       dataset (articleType / genre_top / artist)
  - recall           : TP / (TP+FN) = TP / total de sources relevantes en la BD
  - overlap_at_k     : fracción del top-k de Lado B encontrada también en Lado A
  - io_shared_hit    : bloques encontrados en cache de Postgres (EXPLAIN ANALYZE BUFFERS)
  - io_shared_read   : bloques leídos del disco por Postgres
  - real_chunks      : chunks reales de la modalidad en la BD (verifica --scale)

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
  --scale      1k | 10k | 100k  — etiqueta del experimento; se verifica contra el
               conteo real de chunks en la BD (advierte si difiere >10%)
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

# Consolas Windows cp1252: no crashear por caracteres como '→' o '═'.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(errors="replace")

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


def _io_snapshot():
    """Contadores de I/O del proceso (syscalls de lectura); None si no hay soporte."""
    proc = _proceso_actual()
    return proc.io_counters() if hasattr(proc, "io_counters") else None


def _io_delta(antes) -> tuple[int, float]:
    """(reads, MB leídos) desde el snapshot 'antes'. (-1, -1) si no hay soporte."""
    if antes is None:
        return -1, -1.0
    ahora = _proceso_actual().io_counters()
    return (ahora.read_count - antes.read_count,
            (ahora.read_bytes - antes.read_bytes) / 1024 / 1024)


def _medir(fn, *args, **kwargs):
    """Ejecuta fn(*args, **kwargs) midiendo latencia, RAM, pico e I/O.

    - El pico (tracemalloc) se mide en una SEGUNDA ejecución para no inflar
      la latencia de la primera (tracemalloc añade overhead considerable).
    - io_reads/io_read_mb: syscalls y MB leídos por ESTE proceso durante la
      búsqueda (psutil io_counters). Para el Lado A refleja disco real; para
      el Lado B refleja la red del cliente, por eso B se mide con EXPLAIN.
    """
    io_antes = _io_snapshot()
    ram_antes = _ram_mb()
    t0 = time.perf_counter()
    resultado = fn(*args, **kwargs)
    latencia_ms = (time.perf_counter() - t0) * 1000
    ram_delta = _ram_mb() - ram_antes
    io_reads, io_read_mb = _io_delta(io_antes)

    tracemalloc.start()
    try:
        fn(*args, **kwargs)
        pico_mb = tracemalloc.get_traced_memory()[1] / 1024 / 1024
    finally:
        tracemalloc.stop()
    return resultado, latencia_ms, ram_delta, pico_mb, io_reads, io_read_mb


def _overlap_at_k(top_a: list[str], top_b: list[str]) -> float:
    """Fracción de los resultados de Lado B presentes también en Lado A."""
    if not top_b:
        return 0.0
    comunes = set(top_a) & set(top_b)
    return round(len(comunes) / len(top_b), 4)


def _precision_recall(top: list[str], query_sid: str | None, query_label: str | None,
                      labels: dict[str, str], total_por_label: dict[str, int],
                      k: int) -> tuple[float | str, float | str]:
    """Precision = TP/(TP+FP) y Recall = TP/(TP+FN) con relevancia por etiqueta.

    TP = resultados del top-k con la misma etiqueta que la consulta
    (excluyendo el propio source de la consulta). Relevantes totales =
    sources con esa etiqueta en la BD menos la consulta misma.
    Devuelve ("", "") si la consulta no tiene etiqueta conocida.
    """
    if not query_label:
        return "", ""
    recuperados = [sid for sid in top[:k] if sid != query_sid]
    if not recuperados:
        return 0.0, 0.0
    tp = sum(1 for sid in recuperados if labels.get(sid) == query_label)
    precision = round(tp / len(recuperados), 4)
    total_rel = total_por_label.get(query_label, 0)
    if query_sid in labels:
        total_rel -= 1  # la consulta misma no cuenta como relevante
    recall = round(tp / total_rel, 4) if total_rel > 0 else ""
    return precision, recall


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
    # I/O de carga del índice propio (el Lado A paga su I/O aquí, una vez;
    # después la búsqueda es 100% en RAM).
    index_path = TEXT_MODELS / "index_text.pkl"
    io0 = _io_snapshot()
    codebook = LinguisticCodebook.load(str(TEXT_MODELS / "codebook_text.json"))
    index    = SpimiIndex.load(str(index_path))
    _, io_load_mb = _io_delta(io0)
    index_mb = index_path.stat().st_size / 1024 / 1024
    qh = _encode_text(codebook, query)

    # ── Lado A ───────────────────────────────────────────────────────────
    def lado_a():
        raw = index.search(qh, k=k * 10)
        return _agg_source(raw)[:k]

    top_a, ms_a, ram_a, pico_a, io_reads_a, io_mb_a = _medir(lado_a)

    # ── Lado B (GIN) ─────────────────────────────────────────────────────
    def lado_b():
        res = gin_gist.search_fulltext_aggregated(query, k=k)
        return [r.source_id for r in res]

    top_b, ms_b, ram_b, pico_b, _, _ = _medir(lado_b)

    # ── I/O Postgres (GIN fulltext search, misma query OR que gin_gist) ──
    io_b = {"shared_hit": -1, "shared_read": -1}
    try:
        with get_conn() as conn:
            sql = (
                "SELECT source_id, ts_rank(tsv, tq.q) AS score "
                "FROM chunks, (SELECT replace(plainto_tsquery('english', %s)::text, "
                "'&', '|')::tsquery AS q) tq "
                "WHERE tsv @@ tq.q AND modality = 'text' "
                "ORDER BY score DESC LIMIT %s"
            )
            io_b = _explain_buffers(sql, (query, k), conn)
    except Exception:
        pass

    return {
        "query":             query,
        "lado_a_latencia_ms": round(ms_a, 3),
        "lado_a_qps":         round(1000 / ms_a, 2) if ms_a > 0 else 0,
        "lado_a_ram_delta_mb": round(ram_a, 4),
        "lado_a_mem_pico_mb": round(pico_a, 4),
        "lado_a_io_reads":    io_reads_a,
        "lado_a_io_read_mb":  round(io_mb_a, 4),
        "lado_a_io_load_mb":  round(io_load_mb, 4),
        "lado_a_index_mb":    round(index_mb, 4),
        "lado_b_latencia_ms": round(ms_b, 3),
        "lado_b_qps":         round(1000 / ms_b, 2) if ms_b > 0 else 0,
        "lado_b_ram_delta_mb": round(ram_b, 4),
        "lado_b_mem_pico_mb": round(pico_b, 4),
        "lado_b_io_shared_hit":  io_b["shared_hit"],
        "lado_b_io_shared_read": io_b["shared_read"],
        "overlap_at_k":       _overlap_at_k(top_a, top_b),
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
    index_path = IMAGE_MODELS / "index_image.pkl"
    io0 = _io_snapshot()
    codebook = VisualCodebook.from_file(str(IMAGE_MODELS / "codebook_image.npy"))
    index    = VisualInvertedIndex.load(str(index_path))
    _, io_load_mb = _io_delta(io0)
    index_mb = index_path.stat().st_size / 1024 / 1024
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

    top_a, ms_a, ram_a, pico_a, io_reads_a, io_mb_a = _medir(lado_a)

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

    top_b, ms_b, ram_b, pico_b, _, _ = _medir(lado_b)

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
        "lado_a_mem_pico_mb": round(pico_a, 4),
        "lado_a_io_reads":    io_reads_a,
        "lado_a_io_read_mb":  round(io_mb_a, 4),
        "lado_a_io_load_mb":  round(io_load_mb, 4),
        "lado_a_index_mb":    round(index_mb, 4),
        "lado_b_latencia_ms": round(ms_b, 3),
        "lado_b_qps":         round(1000 / ms_b, 2) if ms_b > 0 else 0,
        "lado_b_ram_delta_mb": round(ram_b, 4),
        "lado_b_mem_pico_mb": round(pico_b, 4),
        "lado_b_io_shared_hit":  io_b["shared_hit"],
        "lado_b_io_shared_read": io_b["shared_read"],
        "overlap_at_k":        _overlap_at_k(top_a, top_b),
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
    hop_ms    = params.get("hop_ms", 750)  # mismo default que la ingesta

    index_path = AUDIO_MODELS / "index_audio.pkl"
    io0 = _io_snapshot()
    codebook = KMeansAcousticBuilder().load_from_file(
        str(AUDIO_MODELS / "kmeans_256_fma.joblib")
    )
    index    = AcousticInvertedIndex.load(str(index_path))
    _, io_load_mb = _io_delta(io0)
    index_mb = index_path.stat().st_size / 1024 / 1024
    splitter = SlidingWindowSplitter(window_ms=window_ms, hop_ms=hop_ms)
    extractor = MfccExtractor()
    chunks   = splitter.split(audio_path, source_id="query")
    descs    = extractor.extract(chunks)

    # ── Lado A ───────────────────────────────────────────────────────────
    # Pool de candidatos k*3 por ventana antes de agregar por canción, para
    # que el ranking final no dependa solo de los primeros matches.
    def lado_a():
        puntajes: dict[str, float] = defaultdict(float)
        for d in descs:
            h = codebook.encode(d)
            if not h.counts:
                continue
            for r in index.search(h, k=k * 3):
                puntajes[r.source_id] += r.score
        return sorted(puntajes, key=puntajes.__getitem__, reverse=True)[:k]

    top_a, ms_a, ram_a, pico_a, io_reads_a, io_mb_a = _medir(lado_a)

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
                    (qvec, k * 3),
                ).fetchall()
                for sid, dist in rows:
                    puntajes[str(sid)] += 1.0 - float(dist)
        return sorted(puntajes, key=puntajes.__getitem__, reverse=True)[:k]

    top_b, ms_b, ram_b, pico_b, _, _ = _medir(lado_b)

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
        "lado_a_mem_pico_mb": round(pico_a, 4),
        "lado_a_io_reads":    io_reads_a,
        "lado_a_io_read_mb":  round(io_mb_a, 4),
        "lado_a_io_load_mb":  round(io_load_mb, 4),
        "lado_a_index_mb":    round(index_mb, 4),
        "lado_b_latencia_ms": round(ms_b, 3),
        "lado_b_qps":         round(1000 / ms_b, 2) if ms_b > 0 else 0,
        "lado_b_ram_delta_mb": round(ram_b, 4),
        "lado_b_mem_pico_mb": round(pico_b, 4),
        "lado_b_io_shared_hit":  io_b["shared_hit"],
        "lado_b_io_shared_read": io_b["shared_read"],
        "overlap_at_k":        _overlap_at_k(top_a, top_b),
        "top_a":               top_a[:k],
        "top_b":               top_b[:k],
    }


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  HELPERS DE MUESTREO DE CONSULTAS ALEATORIAS                            ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def _sample_text_queries(n: int) -> list[tuple[str, str]]:
    """Toma N fragmentos de texto aleatorios -> [(payload, source_id)]."""
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT payload, source_id FROM chunks WHERE modality='text' "
            "AND payload IS NOT NULL ORDER BY RANDOM() LIMIT %s",
            (n,),
        ).fetchall()
    return [(r[0][:200], str(r[1])) for r in rows if r[0]]


def _sample_image_paths(n: int) -> list[tuple[str, str]]:
    """Toma N rutas de imagen aleatorias -> [(ruta, source_id)]."""
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT uri, id FROM sources WHERE modality='image' "
            "ORDER BY RANDOM() LIMIT %s",
            (n,),
        ).fetchall()
    return [(r[0], str(r[1])) for r in rows if r[0] and Path(r[0]).exists()]


def _sample_audio_paths(n: int) -> list[tuple[str, str]]:
    """Toma N rutas de audio aleatorias -> [(ruta, source_id)]."""
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT uri, id FROM sources WHERE modality='audio' "
            "ORDER BY RANDOM() LIMIT %s",
            (n,),
        ).fetchall()
    return [(r[0], str(r[1])) for r in rows if r[0] and Path(r[0]).exists()]


def _derivar_source_id(modality: str, query: str) -> str | None:
    """Deriva el source_id de una consulta dada por el usuario (ruta de archivo)."""
    if modality == "image":
        return Path(str(query)).stem
    if modality == "audio":
        return f"fma_track_{Path(str(query)).stem}"
    return None  # texto libre: sin source conocido


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
                        help="Escala del experimento; se verifica contra el conteo real "
                             "de chunks en la BD.")
    parser.add_argument("--output",    default="results",
                        help="Carpeta de salida para el CSV.")
    parser.add_argument("--runs",      type=int, default=3,
                        help="Repeticiones por consulta para promediar latencia.")
    args = parser.parse_args()

    # ── Verificar escala real en la BD ────────────────────────────────────
    ESCALAS = {"1k": 1000, "10k": 10000, "100k": 100000}
    with get_conn() as conn:
        real_chunks = conn.execute(
            "SELECT COUNT(*) FROM chunks WHERE modality = %s", (args.modality,)
        ).fetchone()[0]
    esperado = ESCALAS[args.scale]
    print(f"[INFO] Chunks de '{args.modality}' en la BD: {real_chunks} "
          f"(escala declarada: {args.scale} = {esperado})")
    if real_chunks == 0:
        print("[ERROR] No hay chunks de esta modalidad en la BD. Corre la ingesta primero.")
        sys.exit(1)
    if abs(real_chunks - esperado) > esperado * 0.10:
        print(f"[WARN] El conteo real ({real_chunks}) difiere >10% de la etiqueta "
              f"--scale {args.scale} ({esperado}). ¿Ingestaste con --max-chunks {esperado}?")

    # ── Ground truth por etiquetas (precision / recall) ───────────────────
    labels = repo.get_source_labels(args.modality)
    total_por_label: dict[str, int] = defaultdict(int)
    for lbl in labels.values():
        total_por_label[lbl] += 1
    if labels:
        print(f"[INFO] Etiquetas de relevancia: {len(labels)} sources, "
              f"{len(total_por_label)} clases.")
    else:
        print("[WARN] Sin etiquetas en sources.metadata: precision/recall "
              "quedarán vacíos (re-ingesta con la versión actual de ingest.py).")

    # ── Determinar consultas: lista de (query, source_id | None) ─────────
    if args.queries:
        queries = [(q, _derivar_source_id(args.modality, q)) for q in args.queries]
    else:
        print(f"[INFO] --queries vacío. Muestreando {args.n_queries} consultas de la BD...")
        queries = SAMPLERS[args.modality](args.n_queries)
        if not queries:
            print("[ERROR] No se encontraron consultas en la BD. ¿Ingestaste datos?")
            sys.exit(1)
        print(f"[INFO] Consultas seleccionadas: {[q for q, _ in queries]}")

    # ── Preparar salida ───────────────────────────────────────────────────
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = output_dir / f"results_{args.scale}_{args.modality}_{ts}.csv"

    FIELDNAMES = [
        "scale", "real_chunks", "modality", "query", "run",
        "lado_a_latencia_ms", "lado_a_qps",
        "lado_a_ram_delta_mb", "lado_a_mem_pico_mb",
        "lado_a_io_reads", "lado_a_io_read_mb",
        "lado_a_io_load_mb", "lado_a_index_mb",
        "lado_a_precision", "lado_a_recall",
        "lado_b_latencia_ms", "lado_b_qps",
        "lado_b_ram_delta_mb", "lado_b_mem_pico_mb",
        "lado_b_precision", "lado_b_recall",
        "lado_b_io_shared_hit", "lado_b_io_shared_read",
        "overlap_at_k",
    ]

    evaluador = EVALUADORES[args.modality]
    total = len(queries) * args.runs
    print(f"\n[INFO] Iniciando evaluación: {total} ejecuciones ({len(queries)} consultas × {args.runs} runs)")
    print(f"[INFO] Resultados → {csv_path}\n")

    with open(csv_path, "w", newline="", encoding="utf-8") as fcsv:
        writer = csv.DictWriter(fcsv, fieldnames=FIELDNAMES)
        writer.writeheader()

        for q, qsid in queries:
            qlabel = labels.get(qsid) if qsid else None
            for run in range(1, args.runs + 1):
                try:
                    print(f"  → [{args.modality.upper()}] run={run} | q={str(q)[:60]!r}")
                    metrics = evaluador(q, args.k)

                    prec_a, rec_a = _precision_recall(
                        metrics["top_a"], qsid, qlabel, labels, total_por_label, args.k)
                    prec_b, rec_b = _precision_recall(
                        metrics["top_b"], qsid, qlabel, labels, total_por_label, args.k)
                    metrics.update({
                        "lado_a_precision": prec_a, "lado_a_recall": rec_a,
                        "lado_b_precision": prec_b, "lado_b_recall": rec_b,
                    })

                    row = {
                        "scale":       args.scale,
                        "real_chunks": real_chunks,
                        "modality":    args.modality,
                        "query":       str(q)[:120],
                        "run":         run,
                        **{k: v for k, v in metrics.items()
                           if k in FIELDNAMES},
                    }
                    writer.writerow(row)
                    fcsv.flush()  # Escribir de inmediato por si se interrumpe
                    prec_txt = (f"P@{args.k} A/B: {prec_a:.2f}/{prec_b:.2f}"
                                if qlabel else "P@k: sin etiqueta")
                    print(f"     Lado A: {metrics['lado_a_latencia_ms']:.1f} ms | "
                          f"Lado B: {metrics['lado_b_latencia_ms']:.1f} ms | "
                          f"{prec_txt} | overlap: {metrics['overlap_at_k']:.2%}")
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
        print(f"  Escala          : {args.scale} (chunks reales: {real_chunks})")
        print(f"  Modalidad       : {args.modality}")
        print(f"  Consultas totales: {len(rows)}")
        print(f"\n  Latencia promedio  Lado A : {_avg('lado_a_latencia_ms'):.2f} ms")
        print(f"  Latencia promedio  Lado B : {_avg('lado_b_latencia_ms'):.2f} ms")
        print(f"  Throughput prom.   Lado A : {_avg('lado_a_qps'):.2f} QPS")
        print(f"  Throughput prom.   Lado B : {_avg('lado_b_qps'):.2f} QPS")
        print(f"  RAM delta prom.    Lado A : {_avg('lado_a_ram_delta_mb'):.4f} MB")
        print(f"  RAM delta prom.    Lado B : {_avg('lado_b_ram_delta_mb'):.4f} MB")
        print(f"  Mem pico prom.     Lado A : {_avg('lado_a_mem_pico_mb'):.4f} MB")
        print(f"  Mem pico prom.     Lado B : {_avg('lado_b_mem_pico_mb'):.4f} MB")
        print(f"  IO búsqueda        Lado A : {_avg('lado_a_io_read_mb'):.4f} MB "
              f"({_avg('lado_a_io_reads'):.0f} reads) — índice ya en RAM")
        print(f"  IO carga índice    Lado A : {_avg('lado_a_io_load_mb'):.2f} MB "
              f"(índice: {_avg('lado_a_index_mb'):.2f} MB en disco)")
        print(f"  IO shared_hit prom Lado B : {_avg('lado_b_io_shared_hit'):.0f} bloques")
        print(f"  IO shared_read pro Lado B : {_avg('lado_b_io_shared_read'):.0f} bloques")
        print(f"  Precision@{args.k} prom.  Lado A : {_avg('lado_a_precision'):.2%}")
        print(f"  Precision@{args.k} prom.  Lado B : {_avg('lado_b_precision'):.2%}")
        print(f"  Recall@{args.k} prom.     Lado A : {_avg('lado_a_recall'):.2%}")
        print(f"  Recall@{args.k} prom.     Lado B : {_avg('lado_b_recall'):.2%}")
        print(f"  Overlap@{args.k} promedio : {_avg('overlap_at_k'):.2%}")
        print("═════════════════════════════════════════════════════")
    except Exception:
        pass

    close_pool()


if __name__ == "__main__":
    main()
