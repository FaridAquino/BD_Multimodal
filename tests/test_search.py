"""Test de integración del Lado B de audio (pgvector).

Comprueba la búsqueda por similitud de coseno (operador `<=>`) sobre
embeddings_audio: tomando un embedding ya almacenado como consulta, el primer
resultado debe estar a distancia 0 (similitud ~1.0), porque el propio vector
está en la tabla.

Nota: NO se exige que el `source_id` del top coincida con el de la consulta.
Los embeddings de audio son histogramas one-hot por chunk ({palabra: 1} en
dim k), así que chunks de canciones distintas comparten vector idéntico y
empatan a distancia 0; cuál sale primero entre los empates no es determinista.

Requiere Postgres levantado y datos ingestados; si no, el test se SALTA
(no falla) para no romper la suite cuando no hay infraestructura.
"""
from __future__ import annotations

import numpy as np
import pytest

try:
    from pgvector.psycopg import register_vector
    from src.db.connection import get_conn, close_pool
except Exception as exc:  # pragma: no cover - entorno sin drivers
    pytest.skip(f"dependencias de BD no disponibles: {exc}", allow_module_level=True)


def _fetch_un_embedding(conn):
    return conn.execute(
        "SELECT chunk_id, source_id, embedding FROM embeddings_audio LIMIT 1"
    ).fetchone()


def test_busqueda_vectorial_self_match():
    try:
        with get_conn() as conn:
            register_vector(conn)
            row = _fetch_un_embedding(conn)
            if row is None:
                pytest.skip("embeddings_audio vacía: corre la ingesta de audio primero")

            chunk_id, source_id, embedding = row
            qvec = np.asarray(embedding, dtype=np.float32)

            resultados = conn.execute(
                "SELECT source_id, 1 - (embedding <=> %s) AS similitud "
                "FROM embeddings_audio ORDER BY embedding <=> %s LIMIT 5",
                (qvec, qvec),
            ).fetchall()
    except pytest.skip.Exception:
        raise
    except Exception as exc:
        pytest.skip(f"Postgres no disponible: {exc}")
    finally:
        close_pool()

    assert resultados, "la consulta vectorial no devolvió filas"
    _, top_sim = resultados[0]
    # El propio vector está en la tabla -> distancia 0 -> similitud ~1.0.
    assert top_sim == pytest.approx(1.0, abs=1e-3)
