"""Conexión a PostgreSQL.  OWNER: Tech Lead.

Lee TODO de variables de entorno (.env). El cambio entre Opción A (Python local)
y Opción B (Python en Docker) es solo cambiar DB_HOST en el .env, sin tocar código:
  - Opción A:  DB_HOST=localhost
  - Opción B:  DB_HOST=postgres   (nombre del servicio en docker-compose)
"""
from __future__ import annotations

import os
from contextlib import contextmanager

from psycopg_pool import ConnectionPool


def _dsn() -> str:
    return (
        f"host={os.getenv('DB_HOST', 'localhost')} "
        f"port={os.getenv('DB_PORT', '5435')} "
        f"dbname={os.getenv('DB_NAME', 'multimodal')} "
        f"user={os.getenv('DB_USER', 'postgres')} "
        f"password={os.getenv('DB_PASSWORD', 'postgres')}"
    )


_pool: ConnectionPool | None = None


def get_pool() -> ConnectionPool:
    global _pool
    if _pool is None:
        _pool = ConnectionPool(conninfo=_dsn(), min_size=1, max_size=10, open=True)
    return _pool


@contextmanager
def get_conn():
    with get_pool().connection() as conn:
        yield conn


def healthcheck() -> bool:
    """Comprueba conectividad y que la extensión pgvector esté instalada."""
    with get_conn() as conn:
        row = conn.execute(
            "SELECT 1 FROM pg_extension WHERE extname = 'vector'"
        ).fetchone()
        return row is not None

def close_pool() -> None:
    """Cierra el pool de conexiones ordenadamente (llamar al terminar)."""
    global _pool
    if _pool is not None:
        _pool.close()
        _pool = None