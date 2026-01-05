# Knowledge Base System

A production-ready knowledge base combining vector search (Qdrant), knowledge graphs (Neo4j), and hybrid retrieval powered by local embeddings (BGE-M3).

## Features

- **Multi-Modal Ingestion**: PDF, DOCX, PPTX, XLSX, HTML, Markdown, Images (OCR)
- **Hybrid Search**: Vector similarity + keyword matching (BM25) + graph traversal
- **Triple Storage**: Qdrant (vectors) + Neo4j (graph) + PostgreSQL (BM25)
- **Local LLM Integration**: Works with Ollama for private AI processing
- **REST API**: FastAPI endpoints for all operations
- **Dashboard**: Streamlit visualization interface

## Architecture

```
Document Sources → Docling Processing → Embeddings (BGE-M3)
                                              ↓
                    ┌─────────────────────────┼─────────────────────────┐
                    ↓                         ↓                         ↓
              Qdrant (Vectors)         Neo4j (Graph)           PostgreSQL (BM25)
                    └─────────────────────────┼─────────────────────────┘
                                              ↓
                                    LlamaIndex Orchestration
                                              ↓
                              Streamlit Dashboard / FastAPI
```

## Quick Start

### Prerequisites

- Docker and Docker Compose
- Python 3.10+
- NVIDIA GPU (optional, for acceleration)

### 1. Clone and Install

```bash
cd ~/ai/knowledge-base
pip install -r requirements.txt
```

### 2. Start Services

```bash
docker compose up -d
```

### 3. Initialize Knowledge Base

```bash
python cli.py init
```

### 4. Ingest Documents

```bash
# Single file
python cli.py ingest /path/to/document.pdf -c my_collection

# Directory of markdown files
python cli.py ingest /path/to/docs -c my_collection --ext md

# All supported formats
python cli.py ingest /path/to/docs -c my_collection
```

### 5. Search

```bash
python cli.py search "your query here" --mode hybrid
```

### 6. Start Services

```bash
# API only
python cli.py serve

# API + Dashboard
python cli.py serve --dashboard

# Or use scripts
./scripts/run_api.sh       # API on :8000
./scripts/run_dashboard.sh  # Dashboard on :8501
```

## Access Points

| Service | URL | Description |
|---------|-----|-------------|
| **API Docs** | http://localhost:8000/docs | Swagger UI |
| **Dashboard** | http://localhost:8501 | Streamlit visualization |
| **Qdrant UI** | http://localhost:6333/dashboard | Vector space explorer |
| **Neo4j Browser** | http://localhost:7474 | Graph visualization |

## Supported Formats

- **PDF** - Full OCR and table extraction
- **DOCX** - Microsoft Word documents
- **PPTX** - PowerPoint presentations
- **XLSX** - Excel spreadsheets
- **HTML** - Web pages
- **Markdown** - Optimized fast processing
- **Images** - PNG, JPG, TIFF (with OCR)

## Python API

```python
from src.knowledge_base import KnowledgeBase

# Initialize
kb = KnowledgeBase()
kb.initialize()

# Ingest
doc_id = kb.ingest_file("document.pdf", collection="my_kb")
doc_ids = kb.ingest_directory("/path/to/docs", collection="my_kb")

# Search
results = kb.search("query", limit=10, mode="hybrid")
for r in results:
    print(f"{r.score:.3f} - {r.document_title}: {r.content[:100]}")

# Stats
kb.print_stats()
```

## REST API Examples

```bash
# Search
curl -X POST http://localhost:8000/search \
  -H "Content-Type: application/json" \
  -d '{"query": "how to configure security", "limit": 5}'

# Ingest text
curl -X POST http://localhost:8000/ingest/text \
  -H "Content-Type: application/json" \
  -d '{"text": "Your content here", "title": "My Doc", "collection": "default"}'

# Get stats
curl http://localhost:8000/stats
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
~/ai/knowledge-base/
├── docker-compose.yml     # Qdrant + PostgreSQL
├── cli.py                 # Command-line interface
├── requirements.txt       # Python dependencies
├── config/
│   ├── qdrant.yaml        # Qdrant configuration
│   └── postgres-init.sql  # Database schema
├── data/                  # Persistent storage (gitignored)
│   ├── qdrant/
│   └── postgres/
├── scripts/
│   ├── run_api.sh
│   └── run_dashboard.sh
└── src/
    ├── config.py          # Settings
    ├── embeddings.py      # BGE-M3 service
    ├── document_processor.py  # Docling pipeline
    ├── vector_store.py    # Qdrant operations
    ├── graph_store.py     # Neo4j operations
    ├── knowledge_base.py  # Unified interface
    ├── api.py             # FastAPI endpoints
    └── dashboard.py       # Streamlit UI
```

## Development Status

This project is currently in **Final Beta**. Core features are implemented and tested.

## Tips

1. **GPU Acceleration**: The system auto-detects CUDA. For best performance, ensure GPU is available.

2. **Large Ingestions**: Use background tasks via the API for large directories.

3. **Hybrid Search**: Combines vector similarity with keyword matching for best results.

4. **Collections**: Use collections to organize different knowledge domains.

## License

Copyright (c) 2025 NerdBase-by-Stark. All rights reserved.
