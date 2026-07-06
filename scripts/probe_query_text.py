"""Búsqueda por letra: lado A (SPIMI propio) vs lado B (GIN nativo)."""
from __future__ import annotations

import argparse
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.core import Histogram                          # noqa: E402
from src.text.splitter import ParagraphSplitter         # noqa: E402
from src.text.extractor import TfidfExtractor           # noqa: E402
from src.text.codebook import LinguisticCodebook        # noqa: E402
from src.text.index.spimi import SpimiIndex             # noqa: E402
from src.baselines import gin_gist                      # noqa: E402
from src.db import repositories as repo                 # noqa: E402
from src.db.connection import close_pool                # noqa: E402

MODELS_DIR = Path("models/text")


def encode_query(codebook: LinguisticCodebook, q: str) -> Histogram:
    chunks = ParagraphSplitter().split(q, source_id="query")
    descs = TfidfExtractor().extract(chunks)
    merged: dict[int, int] = {}
    for d in descs:
        for cw, c in codebook.encode(d).counts.items():
            merged[cw] = merged.get(cw, 0) + c
    return Histogram(chunk_id="query", source_id="query", counts=merged)


def agg_por_cancion(results) -> list[tuple[str, float]]:
    acc: dict[str, float] = defaultdict(float)
    for r in results:
        acc[r.source_id] = max(acc[r.source_id], r.score)
    return sorted(acc.items(), key=lambda x: x[1], reverse=True)


def mostrar(titulo: str, ranked, meta, ms: float, top: int) -> None:
    print(f"\n=== {titulo}  ({ms:.1f} ms) ===")
    if not ranked:
        print("  (sin resultados)")
        return
    for i, (sid, score) in enumerate(ranked[:top], 1):
        m = meta.get(sid, {})
        print(f"  {i:>2}. {score:.4f}  {m.get('artist','?')} — {m.get('song','?')}  [{sid}]")


def main() -> None:
    p = argparse.ArgumentParser(description="Probe de búsqueda por letra (lado A vs B)")
    p.add_argument("--q", required=True)
    p.add_argument("--top", type=int, default=5)
    args = p.parse_args()

    try:
        codebook = LinguisticCodebook.load(str(MODELS_DIR / "codebook_text.json"))
        index = SpimiIndex.load(str(MODELS_DIR / "index_text.pkl"))

        qh = encode_query(codebook, args.q)
        t0 = time.perf_counter()
        raw_a = index.search(qh, k=args.top * 10)
        ranked_a = agg_por_cancion(raw_a)
        ms_a = (time.perf_counter() - t0) * 1000

        t0 = time.perf_counter()
        res_b = gin_gist.search_fulltext_aggregated(args.q, k=args.top)
        ranked_b = [(r.source_id, r.score) for r in res_b]
        ms_b = (time.perf_counter() - t0) * 1000

        sids = {sid for sid, _ in ranked_a[:args.top]} | {sid for sid, _ in ranked_b[:args.top]}
        meta = repo.get_sources_metadata(list(sids))

        print(f"Consulta: {args.q!r}")
        mostrar("LADO A — SPIMI (índice propio)", ranked_a, meta, ms_a, args.top)
        mostrar("LADO B — GIN (Postgres full-text)", ranked_b, meta, ms_b, args.top)
    finally:
        close_pool()


if __name__ == "__main__":
    main()
