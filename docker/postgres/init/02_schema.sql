-- =====================================================================
-- ESQUEMA DEFINITIVO — Sistema Multimodal de Recuperación y Búsqueda
-- Proyecto 2, BD2 (UTEC).  OWNER: Tech Lead.
-- Se ejecuta después de 01_extensions.sql y antes de 03_indexes.sql.
--
-- ORGANIZACIÓN:
--   A) Tablas COMUNES (agnósticas a la modalidad): sources, chunks,
--      codebooks, histograms.
--   B) Específicas de TEXTO: codewords_text
--   C) Específicas de IMAGEN: embeddings_image.
--   D) Específicas de AUDIO:  embeddings_audio.
-- =====================================================================


-- =====================================================================
-- A) TABLAS COMUNES (las tres modalidades)
-- =====================================================================

-- A.1 SOURCES: un registro por origen (documento, imagen o canción).
CREATE TABLE sources (
    id          TEXT PRIMARY KEY,
    modality    TEXT NOT NULL CHECK (modality IN ('text','image','audio')),
    uri         TEXT,                              -- ruta del archivo original (img/audio en disco)
    metadata    JSONB NOT NULL DEFAULT '{}'::jsonb -- título, categoría, artista, etc.
);

-- A.2 CHUNKS: un registro por unidad atómica tras el split.
--     position SÍ se usa en las 3 (orden de párrafo/patch/ventana).
--     payload y tsv SOLO aplican a texto -> NULL en imagen/audio.
CREATE TABLE chunks (
    id          BIGSERIAL PRIMARY KEY,
    source_id   TEXT NOT NULL REFERENCES sources(id) ON DELETE CASCADE,
    modality    TEXT NOT NULL CHECK (modality IN ('text','image','audio')),
    position    INT  NOT NULL DEFAULT 0,           -- orden dentro del origen
    payload     TEXT,                              -- texto del párrafo (NULL en img/audio)
    tsv         tsvector,                          -- LADO B texto, índice GIN (NULL en img/audio)
    metadata    JSONB NOT NULL DEFAULT '{}'::jsonb
);
CREATE INDEX idx_chunks_source   ON chunks (source_id);
CREATE INDEX idx_chunks_modality ON chunks (modality);

-- A.3 CODEBOOKS: un registro por codebook entrenado. Aquí vive el k.
CREATE TABLE codebooks (
    id          BIGSERIAL PRIMARY KEY,
    modality    TEXT NOT NULL CHECK (modality IN ('text','image','audio')),
    k           INT  NOT NULL,                     -- tamaño del vocabulario
    params      JSONB NOT NULL DEFAULT '{}'::jsonb,-- idioma/stemmer (texto), n_init/seed (K-Means)
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- A.4 HISTOGRAMS: LADO A (índice propio), común a las 3 modalidades.
--     counts (JSONB) = frecuencia de cada codeword en el chunk = el "TF"
--     generalizado (palabras / visual words / acoustic words).
--     JSONB no tiene dimensión fija -> sirve para cualquier k.
CREATE TABLE histograms (
    chunk_id    BIGINT NOT NULL REFERENCES chunks(id) ON DELETE CASCADE,
    codebook_id BIGINT NOT NULL REFERENCES codebooks(id) ON DELETE CASCADE,
    source_id   TEXT   NOT NULL,
    counts      JSONB  NOT NULL,                   -- {"codeword_id": frecuencia}
    norm        REAL,                              -- ||vector|| precalculada (opcional)
    PRIMARY KEY (chunk_id, codebook_id)
);
CREATE INDEX idx_hist_source ON histograms (source_id);


-- =====================================================================
-- B) ESPECÍFICAS DE TEXTO
-- =====================================================================
--    CODEWORDS_TEXT: el vocabulario. Una fila por término. Aquí vive el IDF
--    (es por-término, global; no se repite por chunk).
--    TF-IDF = counts(TF, en histograms) x idf(aquí). Se calcula al consultar.
CREATE TABLE codewords_text (
    codebook_id  BIGINT NOT NULL REFERENCES codebooks(id) ON DELETE CASCADE,
    codeword_id  INT    NOT NULL,                  -- id interno (0..k-1)
    term         TEXT   NOT NULL,                  -- el stem: 'camiset', 'algodon'
    idf          REAL   NOT NULL,                  -- IDF precalculado
    doc_freq     INT,                              -- en cuántos chunks aparece
    PRIMARY KEY (codebook_id, codeword_id)
);
CREATE INDEX idx_codewords_term ON codewords_text (codebook_id, term);

-- =====================================================================
-- C) ESPECÍFICA DE IMAGEN
-- =====================================================================

-- C.1 EMBEDDINGS_IMAGE: LADO B (pgvector). Histograma como vector denso.
--     Dimensión = k_image. Índice HNSW en 03_indexes.sql.
CREATE TABLE embeddings_image (
    chunk_id    BIGINT PRIMARY KEY REFERENCES chunks(id) ON DELETE CASCADE,
    codebook_id BIGINT NOT NULL REFERENCES codebooks(id) ON DELETE CASCADE,
    source_id   TEXT   NOT NULL,
    embedding   vector(256) NOT NULL               -- <-- k_image (sincronizar con codebooks.k)
);


-- =====================================================================
-- D) ESPECÍFICA DE AUDIO
-- =====================================================================

-- D.1 EMBEDDINGS_AUDIO: gemela de embeddings_image; solo cambia la dimensión.
--     Dimensión = k_audio. Índice HNSW en 03_indexes.sql.
CREATE TABLE embeddings_audio (
    chunk_id    BIGINT PRIMARY KEY REFERENCES chunks(id) ON DELETE CASCADE,
    codebook_id BIGINT NOT NULL REFERENCES codebooks(id) ON DELETE CASCADE,
    source_id   TEXT   NOT NULL,
    embedding   vector(256) NOT NULL               -- <-- k_audio (sincronizar con codebooks.k)
);
