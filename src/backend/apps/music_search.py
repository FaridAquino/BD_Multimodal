"""App 2 — Búsqueda Musical Inteligente.  OWNER: Ing. Backend (+ Texto/Audio)."""
from __future__ import annotations

from fastapi import APIRouter

router = APIRouter()


@router.get("/search")
async def search_music(q: str | None = None):
    # TODO(backend): full-text por letra (texto) + similitud acústica (audio).
    raise NotImplementedError
