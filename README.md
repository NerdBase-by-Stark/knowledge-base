# Knowledge Base System

A local-first RAG system that actually understands your documents. Built for people who need semantic search without the cloud dependency.

## Why I built this

I needed something that could:

1. **Work offline** - No API calls, no token costs, no data leaving my machine
2. **Understand context** - Not just keyword matching, but actual semantic search
3. **Handle any format** - PDFs, docs, slides, whatever I throw at it
4. **Scale reasonably** - Thousands of documents without falling over

Traditional search is frustrating because it only finds exact matches. If you search for "how to fix X" but your document says "repairing X", you're out of luck. This system bridges that gap using embeddings and knowledge graphs.

## What it actually does

- **Ingests documents** in all common formats (PDF, DOCX, PPTX, XLSX, HTML, MD, images with OCR)
- **Builds three storage layers**: Qdrant for vectors, Neo4j for relationships, PostgreSQL for keyword search
- **Lets you query naturally** - ask questions in plain language, get relevant answers
- **Runs everything locally** - Ollama for LLM stuff, BGE-M3 for embeddings

## Quick Start

```bash
# Clone and install
git clone https://github.com/NerdBase-by-Stark/knowledge-base.git
cd knowledge-base
pip install -r requirements.txt

# Start the databases
docker compose up -d

# Initialize and ingest
python cli.py init
python cli.py ingest /path/to/docs -c my_collection

# Search
python cli.py search "your query" --mode hybrid

# Start the dashboard
python cli.py serve --dashboard
# http://localhost:8501
```

## Architecture

```
Documents → Docling → Embeddings (BGE-M3)
                  ↓
    ┌─────────────┼─────────────┐
    ↓             ↓             ↓
Qdrant       Neo4j        PostgreSQL
(vectors)    (graph)      (keywords)
    └─────────────┼─────────────┘
                  ↓
        Hybrid Search + LLM API
```

## Access Points

| Service | URL | Purpose |
|---------|-----|---------|
| API | http://localhost:8000/docs | REST endpoints |
| Dashboard | http://localhost:8501 | Search UI |
| Monitoring | http://localhost:8502 | Ingestion stats |
| Qdrant | http://localhost:6333/dashboard | Vector explorer |
| Neo4j | http://localhost:7474 | Graph browser |

## Python API

```python
from src.knowledge_base import KnowledgeBase

kb = KnowledgeBase()
kb.initialize()

# Ingest documents
doc_id = kb.ingest_file("document.pdf", collection="my_kb")
doc_ids = kb.ingest_directory("/path/to/docs", collection="my_kb")

# Search
results = kb.search("query", limit=10, mode="hybrid")
for r in results:
    print(f"{r.score:.3f} - {r.document_title}: {r.content[:100]}")
```

## Configuration

Environment variables (prefix with `KB_`):

| Variable | Default | Description |
|----------|---------|-------------|
| `KB_QDRANT_HOST` | localhost | Qdrant host |
| `KB_QDRANT_PORT` | 6333 | Qdrant port |
| `KB_NEO4J_URI` | bolt://localhost:7687 | Neo4j connection |
| `KB_NEO4J_USER` | neo4j | Neo4j username |
| `KB_NEO4J_PASSWORD` | agentmemory123 | Neo4j password |
| `KB_EMBEDDING_MODEL` | BAAI/bge-m3 | Embedding model |
| `KB_CHUNK_SIZE` | 512 | Chunk size in tokens |

## Project Structure

```
knowledge-base/
├── docker-compose.yml     # Qdrant + PostgreSQL
├── cli.py                 # CLI interface
├── requirements.txt       # Dependencies
├── config/
│   ├── qdrant.yaml
│   └── postgres-init.sql
├── data/                  # Runtime data (gitignored)
├── scripts/               # Utility scripts
└── src/
    ├── config.py
    ├── embeddings.py      # BGE-M3 service
    ├── document_processor.py  # Docling pipeline
    ├── vector_store.py    # Qdrant operations
    ├── graph_store.py     # Neo4j operations
    ├── knowledge_base.py  # Main interface
    ├── api.py             # FastAPI
    └── dashboard.py       # Streamlit
```

## Tips

- **GPU helps** but isn't required - system auto-detects CUDA
- **Hybrid mode** gives the best search results (vectors + keywords + graph)
- **Use collections** to separate different knowledge domains
- Background jobs via API for large ingestions

## Status

Currently in final beta. Core features work, but expect rough edges.

## License

Copyright (c) 2025 NerdBase-by-Stark. All rights reserved.
