"""Acceso a datos (codebooks, histogramas, metadatos).  OWNER: Tech Lead.

Provee las operaciones que tanto Lado A (índice propio) como Lado B (GIN/pgvector)
usan sobre las MISMAS tablas, garantizando comparación justa.
"""
from __future__ import annotations

import math
import json
from typing import Iterable

from src.core import Chunk, Histogram
from src.db.connection import get_conn

def insert_source(source_id: str, modality: str, uri: str, metadata: dict) -> None:
    """Inserta la pista de audio (o imagen/texto) en la tabla sources."""
    with get_conn() as conn:
        conn.execute("""
            INSERT INTO sources (id, modality, uri, metadata)
            VALUES (%s, %s, %s, %s::jsonb)
            ON CONFLICT (id) DO NOTHING;
        """, (source_id, modality, uri, json.dumps(metadata)))
    
       


def insert_chunks(chunks: Iterable[Chunk]) -> None:
    """Inserta los pedacitos y actualiza el objeto Chunk con el ID de la base de datos."""
    with get_conn() as conn:
        for chunk in chunks:

            row = conn.execute("""
                INSERT INTO chunks (source_id, modality, position)
                VALUES (%s, %s, %s)
                RETURNING id;
            """, (chunk.source_id, chunk.modality.value, chunk.position)).fetchone()
            
            if row:
                chunk.chunk_id = row[0]


def insert_histograms(histograms: Iterable[Histogram]) -> None:
    """Inserta en el Lado A (histograms) y Lado B (embeddings_audio)."""
    
    with get_conn() as conn:
        for hist in histograms:
            # --- 1. LADO A: histograms ---
            counts_json = json.dumps(hist.counts)


            norm_values = math.sqrt(sum(count ** 2 for count in hist.counts.values()))

            conn.execute("""
                INSERT INTO histograms (chunk_id, codebook_id, source_id, counts, norm)
                VALUES (%s, %s, %s, %s::jsonb, %s)
                ON CONFLICT (chunk_id, codebook_id) DO NOTHING;
            """, (hist.chunk_id, hist.codebook_id, hist.source_id, counts_json, norm_values))

            # --- 2. LADO B: embeddings_audio ---
            # Construimos el array denso
            
            vector_str = "[" + ",".join(str(x) for x in hist.raw_embedding) + "]"

            conn.execute("""
                INSERT INTO embeddings_audio (chunk_id, codebook_id, source_id, embedding)
                VALUES (%s, %s, %s, %s)
                ON CONFLICT (chunk_id) DO NOTHING;
            """, (hist.chunk_id, hist.codebook_id, hist.source_id, vector_str))


def search_similar_audio(query_embedding: list[float], limit: int = 5):
    """
    Busca los fragmentos de audio más similares usando pgvector (Lado B).
    Devuelve una lista de tuplas: (chunk_id, similitud_coseno).
    """
    # Convertimos la lista de Python al formato de texto que pgvector entiende: "[0.1, 0.2, ...]"
    vector_str = "[" + ",".join(map(str, query_embedding)) + "]"
    
    with get_conn() as conn:
        # El operador <=> calcula la "distancia" del coseno (0 es idéntico, 2 es opuesto).
        # Para convertir distancia en "similitud" (1 a -1), la restamos de 1: (1 - distancia).
        cursor = conn.execute("""
            SELECT 
                chunk_id, 
                1 - (embedding <=> %s::vector) AS similarity
            FROM embeddings_audio
            ORDER BY embedding <=> %s::vector ASC
            LIMIT %s;
        """, (vector_str, vector_str, limit))
        
        results = cursor.fetchall()
        return results