"""API del motor de búsqueda multimodal.  OWNER: Ing. Backend/Apps."""
from __future__ import annotations

from fastapi import FastAPI

from src.db.connection import healthcheck
from .apps import music_search, visual_search

app = FastAPI(title="Multimodal Search Engine", version="0.1.0")
app.include_router(visual_search.router, prefix="/visual", tags=["App 1: Visual"])
app.include_router(music_search.router, prefix="/music", tags=["App 2: Music"])


@app.get("/health")
def health():
    return {"status": "ok", "pgvector": healthcheck()}
