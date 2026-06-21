"""Acceso a datos: inserción en las tablas de Postgres.  OWNER: Tech Lead.

Capa que sabe CÓMO escribir en cada tabla. El ingest.py decide QUÉ insertar y
llama a estas funciones. Inserción por LOTES (execute_many) porque insertar
100K filas una por una sería lentísimo.

Comparte tablas entre modalidades; las específicas (embeddings_image/audio) tienen
su propia función por la dimensión distinta del vector.
"""
from __future__ import annotations

import json
import numpy as np
from typing import Iterable, Sequence

from pgvector.psycopg import register_vector

from src.core import Chunk, Histogram
from .connection import get_conn


# ---------------------------------------------------------------------------
# CODEBOOK
# ---------------------------------------------------------------------------
def register_codebook(modality: str, k: int, params: dict) -> int:
    """Registra un codebook y devuelve su id (para enlazar histogramas/embeddings)."""
    with get_conn() as conn:
        row = conn.execute(
            "INSERT INTO codebooks (modality, k, params) VALUES (%s, %s, %s) "
            "RETURNING id",
            (modality, k, json.dumps(params)),
        ).fetchone()
        conn.commit()
        return row[0]


# ---------------------------------------------------------------------------
# SOURCES
# ---------------------------------------------------------------------------
def insert_source(source_id: str, modality: str, uri: str, metadata: dict) -> None:
    """Inserta un origen (imagen/canción/doc). ON CONFLICT evita duplicados."""
    with get_conn() as conn:
        conn.execute(
            "INSERT INTO sources (id, modality, uri, metadata) "
            "VALUES (%s, %s, %s, %s) ON CONFLICT (id) DO NOTHING",
            (source_id, modality, uri, json.dumps(metadata)),
        )
        conn.commit()


# ---------------------------------------------------------------------------
# CHUNKS  (devuelve los ids asignados, en el mismo orden)
# ---------------------------------------------------------------------------
def insert_chunks(chunks: Sequence[Chunk]) -> list[int]:
    """Inserta chunks por lote y devuelve los ids generados (BIGSERIAL).

    Para imagen/audio, payload y tsv quedan NULL (solo texto los usa).
    """
    ids: list[int] = []
    with get_conn() as conn:
        with conn.cursor() as cur:
            for ch in chunks:
                cur.execute(
                    "INSERT INTO chunks (source_id, modality, position, payload, metadata) "
                    "VALUES (%s, %s, %s, %s, %s) RETURNING id",
                    (ch.source_id, ch.modality.value, ch.position,
                     ch.payload if ch.modality.value == "text" else None,
                     json.dumps(ch.metadata)),
                )
                ids.append(cur.fetchone()[0])
        conn.commit()
    return ids


# ---------------------------------------------------------------------------
# HISTOGRAMS  (Lado A, común a las 3 modalidades)
# ---------------------------------------------------------------------------
def insert_histograms(rows: Iterable[tuple[int, int, str, dict]]) -> None:
    """rows: iterable de (chunk_id, codebook_id, source_id, counts_dict)."""
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.executemany(
                "INSERT INTO histograms (chunk_id, codebook_id, source_id, counts) "
                "VALUES (%s, %s, %s, %s) "
                "ON CONFLICT (chunk_id, codebook_id) DO UPDATE SET counts = EXCLUDED.counts",
                [(cid, cbid, sid, json.dumps(counts)) for cid, cbid, sid, counts in rows],
            )
        conn.commit()


# ---------------------------------------------------------------------------
# EMBEDDINGS_IMAGE  (Lado B, vector denso para pgvector)
# ---------------------------------------------------------------------------
def insert_embeddings_image(rows: Iterable[tuple[int, int, str, "np.ndarray"]]) -> None:
    """rows: iterable de (chunk_id, codebook_id, source_id, embedding_vector).
    embedding_vector: ndarray de dimensión k (el histograma como vector denso)."""
    with get_conn() as conn:
        register_vector(conn)            # habilita el adaptador de pgvector
        with conn.cursor() as cur:
            cur.executemany(
                "INSERT INTO embeddings_image (chunk_id, codebook_id, source_id, embedding) "
                "VALUES (%s, %s, %s, %s) "
                "ON CONFLICT (chunk_id) DO UPDATE SET embedding = EXCLUDED.embedding",
                [(cid, cbid, sid, emb) for cid, cbid, sid, emb in rows],
            )
        conn.commit()


# ---------------------------------------------------------------------------
# Utilidad: limpiar todo (para reingestar desde cero sin recrear el esquema)
# ---------------------------------------------------------------------------
def truncate_all() -> None:
    with get_conn() as conn:
        conn.execute(
            "TRUNCATE sources, chunks, codebooks, histograms, "
            "embeddings_image, embeddings_audio RESTART IDENTITY CASCADE"
        )
        conn.commit()

# ===========================================================================
# Lee los histogramas de un codebook para construir el índice Lado A.
# ===========================================================================

def get_latest_codebook_id(modality: str) -> int | None:
    """Devuelve el id del codebook más reciente de una modalidad (o None)."""
    with get_conn() as conn:
        row = conn.execute(
            "SELECT id FROM codebooks WHERE modality = %s ORDER BY id DESC LIMIT 1",
            (modality,),
        ).fetchone()
        return row[0] if row else None


def iter_histograms(codebook_id: int):
    """Lee los histogramas de un codebook como objetos Histogram.

    Las claves del JSON `counts` vienen como string desde Postgres; se convierten
    a int para que coincidan con los visual_word_id del índice.
    """
    from src.core import Histogram
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT chunk_id, source_id, counts FROM histograms WHERE codebook_id = %s",
            (codebook_id,),
        ).fetchall()
    for chunk_id, source_id, counts in rows:
        counts_int = {int(k): int(v) for k, v in counts.items()}
        yield Histogram(chunk_id=str(chunk_id), source_id=source_id,
                        counts=counts_int, codebook_id=codebook_id)