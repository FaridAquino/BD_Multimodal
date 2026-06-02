"""App 1 — Búsqueda Visual E-commerce.  OWNER: Ing. Backend (+ Ing. Imágenes)."""
from __future__ import annotations

from fastapi import APIRouter, UploadFile

router = APIRouter()


@router.post("/search")
async def search_by_image(file: UploadFile):
    # TODO(backend): correr el pipeline de imagen sobre `file` y devolver top-10.
    raise NotImplementedError
