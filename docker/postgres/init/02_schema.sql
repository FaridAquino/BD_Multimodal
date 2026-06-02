-- Esquema base. OWNER: Tech Lead.
-- Diseñado para que Lado A (índice propio) y Lado B (GIN/pgvector) compartan
-- las MISMAS tablas de entrada -> comparación justa.

-- Orígenes: un documento, imagen o canción.
CREATE TABLE IF NOT EXISTS sources (
    id          TEXT PRIMARY KEY,
    modality    TEXT NOT NULL CHECK (modality IN ('text','image','audio')),
    uri         TEXT,
    metadata    JSONB DEFAULT '{}'::jsonb
);

-- Chunks: unidades atómicas (párrafo / patch / ventana).
CREATE TABLE IF NOT EXISTS chunks (
    id          BIGSERIAL PRIMARY KEY,
    source_id   TEXT NOT NULL REFERENCES sources(id) ON DELETE CASCADE,
    modality    TEXT NOT NULL,
    position    INT  NOT NULL DEFAULT 0,
    payload     TEXT,                 -- texto del párrafo (para Lado B full-text)
    metadata    JSONB DEFAULT '{}'::jsonb
);

-- Codebooks entrenados.
CREATE TABLE IF NOT EXISTS codebooks (
    id          BIGSERIAL PRIMARY KEY,
    modality    TEXT NOT NULL,
    k           INT  NOT NULL,
    params      JSONB DEFAULT '{}'::jsonb
);

-- LADO A: histogramas de codewords por chunk (índice invertido propio).
CREATE TABLE IF NOT EXISTS histograms (
    chunk_id    BIGINT NOT NULL REFERENCES chunks(id) ON DELETE CASCADE,
    codebook_id BIGINT NOT NULL REFERENCES codebooks(id) ON DELETE CASCADE,
    source_id   TEXT   NOT NULL,
    counts      JSONB  NOT NULL,       -- {codeword_id: frecuencia}
    PRIMARY KEY (chunk_id, codebook_id)
);

-- LADO B (texto): columna full-text para GIN/GiST.
ALTER TABLE chunks ADD COLUMN IF NOT EXISTS tsv tsvector;

-- LADO B (imagen/audio): histograma como vector para pgvector.
-- OJO: la dimensión = tamaño del codebook (k). Ajusta el número al k que elijan
-- (debe coincidir; aquí 256 como ejemplo). Si k difiere por modalidad, usa dos
-- columnas o dos tablas.
ALTER TABLE histograms ADD COLUMN IF NOT EXISTS embedding vector(256);
