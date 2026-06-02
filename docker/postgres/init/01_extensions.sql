-- Extensiones necesarias.
-- pgvector  -> búsqueda vectorial (Lado B: imagen/audio)
-- pg_trgm   -> apoyo a búsqueda de texto / similitud difusa
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pg_trgm;
