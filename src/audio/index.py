"""Índice invertido sobre histogramas de acoustic words.  OWNER: Ing. Audio."""
from __future__ import annotations

import math

from collections import defaultdict
from typing import Iterable
from src.core import Histogram, InvertedIndex, SearchResult


class AcousticInvertedIndex(InvertedIndex):

    def __init__(self):
    
        self.index: dict[int, list[tuple[str, float]]] = defaultdict(list)
        self.idf: dict[int, float] = {}
        self.doc_norms: dict[str, float] = defaultdict(float)


    def build(self, histograms: Iterable[Histogram]) -> None:

        doc_counts = defaultdict(lambda: defaultdict(int))
        for hist in histograms:
            for codeword_id, count in hist.counts.items():
                doc_counts[hist.source_id][codeword_id] += count

        N = len(doc_counts) 
        if N == 0:
            return
        
        df = defaultdict(int)

        for source_id, counts in doc_counts.items():
            for codeword_id in counts.keys():
                df[codeword_id] += 1

        for cw, freq in df.items():
            self.idf[cw] = math.log((N + 1) / freq)


        for source_id, counts in doc_counts.items():
            norm_sq = 0.0
            for codeword_id, count in counts.items():
               
                tf = 1 + math.log(count)
                weight = tf * self.idf[codeword_id]
                
                self.index[codeword_id].append((source_id, weight))
                norm_sq += weight ** 2
            
           
            self.doc_norms[source_id] = math.sqrt(norm_sq)


    def search(self, query: Histogram, k: int = 10) -> list[SearchResult]:
      
        query_weights = {}
        query_norm_sq = 0.0
        
        for codeword_id, count in query.counts.items():
            tf = 1 + math.log(count)
            weight = tf * self.idf.get(codeword_id, 0.0)
            
            if weight > 0:
                query_weights[codeword_id] = weight
                query_norm_sq += weight ** 2
                
        query_norm = math.sqrt(query_norm_sq)
        if query_norm == 0:
            return [] 

       
        scores = defaultdict(float)
        for codeword_id, q_weight in query_weights.items():
            for source_id, doc_weight in self.index.get(codeword_id, []):
                scores[source_id] += q_weight * doc_weight


        results = []
        for source_id, score in scores.items():
            cosine_sim = score / (query_norm * self.doc_norms[source_id])
            results.append(SearchResult(source_id=source_id, score=cosine_sim))

        results.sort(key=lambda x: x.score, reverse=True)
        return results[:k]
