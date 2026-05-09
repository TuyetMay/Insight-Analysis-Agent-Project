-- Run this once in Supabase SQL Editor before starting the app.
--
-- Step 1: Enable pgvector extension (requires Dashboard > Database > Extensions,
--         or run here if your role has superuser privileges).
CREATE EXTENSION IF NOT EXISTS vector;

-- Step 2: Create the RAG chunks table.
CREATE TABLE IF NOT EXISTS rag_chunks (
    chunk_id  text    NOT NULL,
    layer     text    NOT NULL,   -- 'static' or 'dynamic'
    text      text    NOT NULL,
    metadata  jsonb   NOT NULL DEFAULT '{}',
    embedding vector(384),
    PRIMARY KEY (chunk_id, layer)
);

-- Step 3: IVFFlat index for cosine similarity search.
-- lists=10 is appropriate for ~100-500 chunks (Superstore dataset size).
-- Increase lists if chunk count grows beyond 1000.
CREATE INDEX IF NOT EXISTS rag_chunks_embedding_idx
    ON rag_chunks
    USING ivfflat (embedding vector_cosine_ops)
    WITH (lists = 10);
