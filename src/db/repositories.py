"""Acceso a datos: inserción y lectura en las tablas de Postgres."""
from __future__ import annotations

import math
import json
from typing import Iterable
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
    ids: list[int] = []
    with get_conn() as conn:
        with conn.cursor() as cur:
            for ch in chunks:
                if ch.modality.value == "text":
                    cur.execute(
                        "INSERT INTO chunks (source_id, modality, position, payload, tsv, metadata) "
                        "VALUES (%s, %s, %s, %s, to_tsvector('english', %s), %s) RETURNING id",
                        (ch.source_id, ch.modality.value, ch.position,
                         ch.payload, ch.payload, json.dumps(ch.metadata)),
                    )
                else:
                    cur.execute(
                        "INSERT INTO chunks (source_id, modality, position, payload, metadata) "
                        "VALUES (%s, %s, %s, %s, %s) RETURNING id",
                        (ch.source_id, ch.modality.value, ch.position,
                         None, json.dumps(ch.metadata)),
                    )
                ids.append(cur.fetchone()[0])
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
            "SELECT id, metadata FROM sources WHERE id = ANY(%s)",
            (list(source_ids),),
        ).fetchall()
    return {sid: (meta or {}) for sid, meta in rows}


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


def truncate_all() -> None:
    with get_conn() as conn:
        conn.execute(
            "TRUNCATE sources, chunks, codebooks, histograms, "
            "embeddings_image, embeddings_audio RESTART IDENTITY CASCADE"
        )
        conn.commit()


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
