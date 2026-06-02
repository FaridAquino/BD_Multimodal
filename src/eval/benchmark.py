"""Marco experimental Fase 4.  OWNER: Ing. Evaluación.

Corre Lado A (índice propio) vs Lado B (GIN/pgvector) sobre cargas de
1K / 10K / 100K chunks y recoge métricas.
"""
from __future__ import annotations

LOADS = (1_000, 10_000, 100_000)


def run(load: int) -> dict:
    # TODO(eval): ejecutar consultas en ambos lados y medir.
    raise NotImplementedError
