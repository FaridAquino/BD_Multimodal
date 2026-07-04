"""Acceso a datos: inserción y lectura en las tablas de Postgres."""
from __future__ import annotations

import json
import numpy as np
from typing import Iterable, Sequence

from pgvector.psycopg import register_vector

from src.core import Chunk, Histogram
from .connection import get_conn


def register_codebook(modality: str, k: int, params: dict) -> int:
    with get_conn() as conn:
        row = conn.execute(
            "INSERT INTO codebooks (modality, k, params) VALUES (%s, %s, %s) "
            "RETURNING id",
            (modality, k, json.dumps(params)),
        ).fetchone()
        conn.commit()
        return row[0]


def insert_source(source_id: str, modality: str, uri: str, metadata: dict) -> None:
    with get_conn() as conn:
        conn.execute(
            "INSERT INTO sources (id, modality, uri, metadata) "
            "VALUES (%s, %s, %s, %s) ON CONFLICT (id) DO NOTHING",
            (source_id, modality, uri, json.dumps(metadata)),
        )
        conn.commit()


def insert_chunks(chunks: Sequence[Chunk]) -> list[int]:
    """Inserta chunks en lote y devuelve sus ids en el mismo orden.

    Usa executemany con returning=True (psycopg3) para evitar un round-trip
    por fila; el tsv solo se calcula para chunks de texto.
    """
    if not chunks:
        return []
    params = [
        (ch.source_id, ch.modality.value, ch.position,
         ch.payload if ch.modality.value == "text" else None,
         ch.modality.value,
         ch.payload if ch.modality.value == "text" else None,
         json.dumps(ch.metadata))
        for ch in chunks
    ]
    ids: list[int] = []
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.executemany(
                "INSERT INTO chunks (source_id, modality, position, payload, tsv, metadata) "
                "VALUES (%s, %s, %s, %s, "
                "        CASE WHEN %s = 'text' THEN to_tsvector('english', %s) END, %s) "
                "RETURNING id",
                params,
                returning=True,
            )
            while True:
                ids.append(cur.fetchone()[0])
                if not cur.nextset():
                    break
        conn.commit()
    return ids


def insert_histograms(rows: Iterable[tuple[int, int, str, dict]]) -> None:
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.executemany(
                "INSERT INTO histograms (chunk_id, codebook_id, source_id, counts) "
                "VALUES (%s, %s, %s, %s) "
                "ON CONFLICT (chunk_id, codebook_id) DO UPDATE SET counts = EXCLUDED.counts",
                [(cid, cbid, sid, json.dumps(counts)) for cid, cbid, sid, counts in rows],
            )
        conn.commit()


def insert_codewords_text(codebook_id: int,
                          rows: Iterable[tuple[int, str, float, int]]) -> None:
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.executemany(
                "INSERT INTO codewords_text (codebook_id, codeword_id, term, idf, doc_freq) "
                "VALUES (%s, %s, %s, %s, %s) "
                "ON CONFLICT (codebook_id, codeword_id) DO UPDATE SET "
                "term = EXCLUDED.term, idf = EXCLUDED.idf, doc_freq = EXCLUDED.doc_freq",
                [(codebook_id, cwid, term, idf, df) for cwid, term, idf, df in rows],
            )
        conn.commit()


def get_sources_metadata(source_ids: Sequence[str]) -> dict[str, dict]:
    if not source_ids:
        return {}
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT id, uri, metadata FROM sources WHERE id = ANY(%s)",
            (list(source_ids),),
        ).fetchall()
    return {str(sid): {"uri": uri, "metadata": meta, **(meta or {})} for sid, uri, meta in rows}


def insert_embeddings_image(rows: Iterable[tuple[int, int, str, "np.ndarray"]]) -> None:
    with get_conn() as conn:
        register_vector(conn)
        with conn.cursor() as cur:
            cur.executemany(
                "INSERT INTO embeddings_image (chunk_id, codebook_id, source_id, embedding) "
                "VALUES (%s, %s, %s, %s) "
                "ON CONFLICT (chunk_id) DO UPDATE SET embedding = EXCLUDED.embedding",
                [(cid, cbid, sid, emb) for cid, cbid, sid, emb in rows],
            )
        conn.commit()


def insert_embeddings_audio(rows: Iterable[tuple[int, int, str, "np.ndarray"]]) -> None:
    """rows: iterable de (chunk_id, codebook_id, source_id, embedding_vector).
    embedding_vector: ndarray de dimensión k (el histograma como vector denso)."""
    with get_conn() as conn:
        register_vector(conn)
        with conn.cursor() as cur:
            cur.executemany(
                "INSERT INTO embeddings_audio (chunk_id, codebook_id, source_id, embedding) "
                "VALUES (%s, %s, %s, %s) "
                "ON CONFLICT (chunk_id) DO UPDATE SET embedding = EXCLUDED.embedding",
                [(cid, cbid, sid, emb) for cid, cbid, sid, emb in rows],
            )
        conn.commit()


def truncate_all() -> None:
    with get_conn() as conn:
        conn.execute(
            "TRUNCATE sources, chunks, codebooks, histograms, "
            "embeddings_image, embeddings_audio RESTART IDENTITY CASCADE"
        )
        conn.commit()


# Etiqueta de relevancia (ground truth) por modalidad, leída de sources.metadata.
_LABEL_KEYS = {"image": "articleType", "audio": "genre", "text": "artist"}


def get_source_labels(modality: str) -> dict[str, str]:
    """Devuelve {source_id: etiqueta} para calcular precision/recall.

    image -> articleType (styles.csv) | audio -> genre (FMA genre_top)
    text  -> artist (spotify CSV). Fuentes sin etiqueta se omiten.
    """
    key = _LABEL_KEYS.get(modality)
    if key is None:
        return {}
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT id, metadata->>%s FROM sources WHERE modality = %s",
            (key, modality),
        ).fetchall()
    return {str(sid): label for sid, label in rows if label}


def get_latest_codebook_params(modality: str) -> dict | None:
    """Params (window_ms, hop_ms, rows, cols, ...) del codebook más reciente."""
    with get_conn() as conn:
        row = conn.execute(
            "SELECT params FROM codebooks WHERE modality = %s ORDER BY id DESC LIMIT 1",
            (modality,),
        ).fetchone()
        return row[0] if row else None


def get_latest_codebook_id(modality: str) -> int | None:
    with get_conn() as conn:
        row = conn.execute(
            "SELECT id FROM codebooks WHERE modality = %s ORDER BY id DESC LIMIT 1",
            (modality,),
        ).fetchone()
        return row[0] if row else None


def iter_histograms(codebook_id: int):
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT chunk_id, source_id, counts FROM histograms WHERE codebook_id = %s",
            (codebook_id,),
        ).fetchall()
    for chunk_id, source_id, counts in rows:
        counts_int = {int(k): int(v) for k, v in counts.items()}
        yield Histogram(chunk_id=str(chunk_id), source_id=source_id,
                        counts=counts_int, codebook_id=codebook_id)
