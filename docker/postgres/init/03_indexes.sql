-- Índices nativos para el Lado B de la comparación. OWNER: Tech Lead.

-- GIN para full-text de texto (App 2: búsqueda por letra).
CREATE INDEX IF NOT EXISTS idx_chunks_tsv ON chunks USING gin (tsv);

-- Alternativa GiST (para comparar GIN vs GiST si el equipo lo desea):
-- CREATE INDEX idx_chunks_tsv_gist ON chunks USING gist (tsv);

-- HNSW para búsqueda vectorial aproximada (imagen/audio).
-- Usa la distancia que corresponda a tu histograma (L2 por defecto).
CREATE INDEX IF NOT EXISTS idx_hist_embedding_hnsw
    ON histograms USING hnsw (embedding vector_l2_ops);

-- Alternativa IVFFlat (para comparar HNSW vs IVF):
-- CREATE INDEX idx_hist_embedding_ivf
--     ON histograms USING ivfflat (embedding vector_l2_ops) WITH (lists = 100);
