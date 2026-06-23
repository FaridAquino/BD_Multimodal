from __future__ import annotations

import heapq
import json
import math
import os
import tempfile
from collections import defaultdict
from typing import Iterable

from src.core import Histogram, InvertedIndex, SearchResult


class BsbiIndex(InvertedIndex):
    def __init__(self, block_size: int = 100_000):
        self.block_size = block_size
        self._block: list[tuple[int, str, int]] = []
        self._doc_count = 0
        self._block_paths: list[str] = []
        self._index: dict[int, list[tuple[str, int]]] = {}
        self._doc_norms: dict[str, float] = {}
        self._work_dir: str | None = None

    def build(self, histograms: Iterable[Histogram]) -> None:
        self._work_dir = tempfile.mkdtemp(prefix="bsbi_")
        for hist in histograms:
            doc_id = hist.chunk_id or hist.source_id
            self._doc_count += 1
            for cw_id, tf in hist.counts.items():
                self._block.append((cw_id, doc_id, tf))
                if len(self._block) >= self.block_size:
                    self._flush_block()
        if self._block:
            self._flush_block()
        self._merge_blocks()
        self._compute_doc_norms()

    def _flush_block(self) -> None:
        self._block.sort(key=lambda x: (x[0], x[1]))
        path = os.path.join(self._work_dir, f"block_{len(self._block_paths):04d}.json")
        with open(path, "w") as f:
            json.dump(self._block, f)
        self._block_paths.append(path)
        self._block = []

    def _merge_blocks(self) -> None:
        if not self._block_paths:
            return
        iters = []
        for path in self._block_paths:
            data = json.load(open(path))
            iters.append(iter(data))
        merged = heapq.merge(*iters, key=lambda x: (x[0], x[1]))
        acc: dict[int, dict[str, int]] = {}
        for term_id, doc_id, tf in merged:
            acc.setdefault(term_id, {}).setdefault(doc_id, 0)
            acc[term_id][doc_id] += tf
        self._index = {t: list(d.items()) for t, d in acc.items()}

    def _compute_doc_norms(self) -> None:
        doc_freq = {t: len(ps) for t, ps in self._index.items()}
        acc: dict[str, float] = defaultdict(float)
        for term, postings in self._index.items():
            df = doc_freq[term]
            idf = math.log(self._doc_count / df) if df > 0 else 0.0
            for doc_id, tf in postings:
                w = (1.0 + math.log(tf)) * idf
                acc[doc_id] += w * w
        self._doc_norms = {d: math.sqrt(n) for d, n in acc.items()}

    def search(self, query: Histogram, k: int = 10) -> list[SearchResult]:
        if not self._index:
            return []
        scores: dict[str, float] = defaultdict(float)
        q_norm_sq = 0.0
        doc_freq = {t: len(ps) for t, ps in self._index.items()}
        for cw_id, q_tf in query.counts.items():
            postings = self._index.get(cw_id)
            if postings is None:
                continue
            df = doc_freq[cw_id]
            idf = math.log(self._doc_count / df) if df > 0 else 0.0
            q_w = (1.0 + math.log(q_tf)) * idf
            q_norm_sq += q_w * q_w
            for doc_id, d_tf in postings:
                d_w = (1.0 + math.log(d_tf)) * idf
                scores[doc_id] += q_w * d_w
        q_norm = math.sqrt(q_norm_sq) if q_norm_sq > 0 else 1.0
        results: list[SearchResult] = []
        for doc_id, raw_score in scores.items():
            d_norm = self._doc_norms.get(doc_id, 1.0)
            cos = raw_score / (q_norm * d_norm) if d_norm > 0 else 0.0
            source_id = doc_id.split(":")[0]
            results.append(SearchResult(source_id=source_id, score=cos, chunk_id=doc_id))
        results.sort(key=lambda r: r.score, reverse=True)
        return results[:k]
