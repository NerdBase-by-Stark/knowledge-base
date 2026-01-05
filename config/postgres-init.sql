-- Enable required extensions
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pg_trgm;

-- Documents table
CREATE TABLE IF NOT EXISTS documents (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    title TEXT NOT NULL,
    source_path TEXT,
    source_url TEXT,
    content TEXT NOT NULL,
    content_hash TEXT UNIQUE,
    doc_type TEXT NOT NULL,  -- pdf, docx, markdown, html, etc.
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Chunks table for processed content
CREATE TABLE IF NOT EXISTS chunks (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    document_id UUID REFERENCES documents(id) ON DELETE CASCADE,
    chunk_index INTEGER NOT NULL,
    content TEXT NOT NULL,
    embedding vector(1024),  -- BGE-M3 dimensions
    token_count INTEGER,
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Collections/Knowledge bases
CREATE TABLE IF NOT EXISTS collections (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name TEXT UNIQUE NOT NULL,
    description TEXT,
    config JSONB DEFAULT '{}',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Document-collection mapping
CREATE TABLE IF NOT EXISTS document_collections (
    document_id UUID REFERENCES documents(id) ON DELETE CASCADE,
    collection_id UUID REFERENCES collections(id) ON DELETE CASCADE,
    PRIMARY KEY (document_id, collection_id)
);

-- Full-text search index (BM25-like)
CREATE INDEX IF NOT EXISTS idx_documents_content_fts
    ON documents USING GIN (to_tsvector('english', content));

CREATE INDEX IF NOT EXISTS idx_chunks_content_fts
    ON chunks USING GIN (to_tsvector('english', content));

-- Vector index for chunks (HNSW)
CREATE INDEX IF NOT EXISTS idx_chunks_embedding
    ON chunks USING hnsw (embedding vector_cosine_ops);

-- Trigram index for fuzzy search
CREATE INDEX IF NOT EXISTS idx_documents_title_trgm
    ON documents USING GIN (title gin_trgm_ops);

-- Metadata indexes
CREATE INDEX IF NOT EXISTS idx_documents_metadata
    ON documents USING GIN (metadata);

CREATE INDEX IF NOT EXISTS idx_chunks_metadata
    ON chunks USING GIN (metadata);

-- Function for hybrid search (vector + BM25)
CREATE OR REPLACE FUNCTION hybrid_search(
    query_text TEXT,
    query_embedding vector(1024),
    match_limit INTEGER DEFAULT 10,
    vector_weight FLOAT DEFAULT 0.5
)
RETURNS TABLE (
    chunk_id UUID,
    document_id UUID,
    content TEXT,
    combined_score FLOAT,
    vector_score FLOAT,
    text_score FLOAT
) AS $$
BEGIN
    RETURN QUERY
    WITH vector_results AS (
        SELECT
            c.id,
            c.document_id,
            c.content,
            1 - (c.embedding <=> query_embedding) as score
        FROM chunks c
        WHERE c.embedding IS NOT NULL
        ORDER BY c.embedding <=> query_embedding
        LIMIT match_limit * 2
    ),
    text_results AS (
        SELECT
            c.id,
            c.document_id,
            c.content,
            ts_rank_cd(to_tsvector('english', c.content), plainto_tsquery('english', query_text)) as score
        FROM chunks c
        WHERE to_tsvector('english', c.content) @@ plainto_tsquery('english', query_text)
        LIMIT match_limit * 2
    ),
    combined AS (
        SELECT
            COALESCE(v.id, t.id) as id,
            COALESCE(v.document_id, t.document_id) as doc_id,
            COALESCE(v.content, t.content) as content,
            COALESCE(v.score, 0) as v_score,
            COALESCE(t.score, 0) as t_score
        FROM vector_results v
        FULL OUTER JOIN text_results t ON v.id = t.id
    )
    SELECT
        c.id as chunk_id,
        c.doc_id as document_id,
        c.content,
        (c.v_score * vector_weight + c.t_score * (1 - vector_weight)) as combined_score,
        c.v_score as vector_score,
        c.t_score as text_score
    FROM combined c
    ORDER BY combined_score DESC
    LIMIT match_limit;
END;
$$ LANGUAGE plpgsql;

-- Insert default collection
INSERT INTO collections (name, description)
VALUES ('default', 'Default knowledge base collection')
ON CONFLICT (name) DO NOTHING;
