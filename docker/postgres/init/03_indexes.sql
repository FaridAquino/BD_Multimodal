-- =====================================================================
-- ÍNDICES para el LADO B (baselines nativos). OWNER: Tech Lead.
-- Se ejecuta DESPUÉS de 02_schema.sql.
--
-- DECISIÓN DE MÉTRICA (pendiente de confirmar por el equipo):
--   Si NO normalizan histogramas -> vector_l2_ops (euclidiana, por defecto).
--   Si SÍ normalizan histogramas -> vector_cosine_ops (coseno).
--   La métrica del Lado B debería coincidir con la del Lado A para una
--   comparación justa en la Fase 4.
-- =====================================================================

-- ---------- TEXTO: full-text con GIN sobre tsvector ----------
CREATE INDEX idx_chunks_tsv_gin ON chunks USING gin (tsv);
-- Comparar GIN vs GiST (lo pide el enunciado):
-- CREATE INDEX idx_chunks_tsv_gist ON chunks USING gist (tsv);

-- ---------- IMAGEN: pgvector HNSW ----------
CREATE INDEX idx_emb_image_hnsw
    ON embeddings_image USING hnsw (embedding vector_cosine_ops);

-- ---------- AUDIO: pgvector HNSW ----------
CREATE INDEX idx_emb_audio_hnsw
    ON embeddings_audio USING hnsw (embedding vector_l2_ops);

-- ---------- Alternativa IVFFlat (comparar HNSW vs IVF) ----------
-- CREATE INDEX idx_emb_image_ivf ON embeddings_image
--     USING ivfflat (embedding vector_l2_ops) WITH (lists = 100);
-- CREATE INDEX idx_emb_audio_ivf ON embeddings_audio
--     USING ivfflat (embedding vector_l2_ops) WITH (lists = 100);