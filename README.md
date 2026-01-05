# Knowledge Base System

A local-first RAG (Retrieval-Augmented Generation) system that **converts your documents into intelligent, queryable formats**.

## What this actually does

**Your documents go in → Three queryable knowledge bases come out**

This system transforms static files (PDFs, docs, web pages) into three different intelligent representations that you can query in ways the original files never allowed:

| Your Original File | Becomes... | What you can now do |
|-------------------|------------|---------------------|
| `manual.pdf` | **Vector embeddings** (Qdrant) | Find content by *meaning*, not just keywords |
| `manual.pdf` | **Knowledge graph** (Neo4j) | Explore relationships between concepts |
| `manual.pdf` | **Full-text index** (PostgreSQL) | Fast exact phrase searches |

**The key insight:** Once your documents are converted, you can query them instantly, discover hidden connections, and get answers with exact source citations. No more reading through hundreds of pages to find one piece of information.

### Before and After

**Before (static files):**
```
manual.pdf
├── You search: "how to configure relay"
├── Result: File contains 500 pages, good luck
└── You read... page 1, page 2, page 47...
```

**After (converted knowledge base):**
```
manual.pdf → Converted → Three databases

You search: "how to configure relay"
├── Vector DB: Found 3 relevant sections (meaning match)
├── Graph DB: Relay connects to GPIO → Core 110f
├── Keyword DB: Exact phrase on page 47
└── Answer: "Section 4.2: Configure GPIO pins 1-4 as..."
           Source: manual.pdf, page 47
```

## Why I built this

I needed something that could:

1. **Work offline** - No API calls, no token costs, no data leaving my machine
2. **Understand context** - Not just keyword matching, but actual semantic search
3. **Handle any format** - PDFs, docs, slides, whatever I throw at it
4. **Scale reasonably** - Thousands of documents without falling over
5. **Be trustworthy** - Answers backed by actual sources, not AI hallucinations

## Why not just use an LLM to search files?

Great question. When you ask ChatGPT or a local LLM to "search my files," here's what actually happens:

| Approach | How it works | Problems |
|----------|--------------|----------|
| **LLM File Search** | LLM reads files one by one, loads them into context window | • Limited by context window (can't read 1000s of files)<br>• Slow - has to read everything each time<br>• Expensive API calls or slow local processing<br>• No memory between searches<br>• Can hallucinate - makes things up |
| **This RAG System** | Documents pre-processed, embedded, stored in specialized databases | • Instant retrieval from indexed data<br>• Scales to 100,000+ documents<br>• One-time cost, then fast forever<br>• Tracks relationships between concepts<br>• Every answer cites its source |

### The key difference: Provenance

When this system gives you an answer, it tells you **exactly** which document and which chunk it came from. You can verify it. An LLM just searching files might confidently tell you something that isn't actually there.

## How it works (the "noob" explanation)

### Step 1: Ingestion (one-time setup)

When you feed documents into the system, it does three things with each chunk of text:

```
Original text: "The GPIO pins on the Q-SYS Core 110f can be configured for relay control"
```

**1. Vector Storage (Qdrant)** - Converts text to numbers representing meaning:
```
[0.234, -0.567, 0.891, ...]  ← 1024 numbers that capture "what this means"
```
This lets you find similar content even with different words. Search "relay output" and it finds this because the *meaning* is close.

**2. Graph Storage (Neo4j)** - Extracts entities and relationships:
```
(:GPIO)-[:CONFIGURED_FOR]->(:RelayControl)
(:GPIO)-[:PART_OF]->(:Core110f)
(:Core110f)-[:MANUFACTURED_BY]->(:QSC)
```
This lets you traverse connections. "Show me all GPIO-related features" finds everything connected to GPIO.

**3. Keyword Storage (PostgreSQL)** - Traditional full-text search:
```
indexed as: gpio, pins, q-sys, core, 110f, configured, relay, control
```
This catches exact matches fast.

### Step 2: Query (fast retrieval)

When you search "how do I use relays with GPIO", the system:

1. **Vector search** finds chunks with similar meaning (even if they say "relay output" not "relay")
2. **Graph traversal** finds related entities (GPIO → Relay → Control → Core)
3. **Keyword search** catches exact phrase matches
4. **Results combined and ranked** - best matches first
5. **LLM synthesizes** an answer citing the specific chunks

## What the graph actually looks like

Fire up the Neo4j browser at http://localhost:7474 and you'll see something like this:

```
        ┌─────────────┐
        │   QSC Core  │
        │   110f      │
        └──────┬──────┘
               │
       ┌───────┼────────┐
       │       │        │
       ↓       ↓        ↓
   ┌──────┐ ┌──────┐ ┌──────┐
   │ GPIO │ │Audio │ │Network│
   └───┬──┘ └───┬──┘ └───┬──┘
       │        │        │
       ↓        ↓        ↓
   ┌──────┐ ┌──────┐ ┌──────┐
   │Relay │ │DSP   │ │Dante │
   └──────┘ └──────┘ └──────┘
```

**Example query:** `MATCH (g:GPIO {name: "Relay Control"})<-[:RELATES_TO*]-(related) RETURN related`

This would find everything connected to GPIO relay control - the Core model, related audio routing, configuration steps, etc. You can't do this with keyword search.

## Example: Why this matters

### Scenario: You're building a Q-SYS audio system

**Question:** "Can I use GPIO for both input and output on the same Core?"

**Traditional LLM search:**
> "Yes, many Cores support bidirectional GPIO. Check your documentation."
> ← Helpful, but vague. Which Cores? Where's the proof?

**This RAG system:**
> "According to the Core 110f Hardware Guide (section 4.2): 'Each GPIO pin can be individually configured as input or output, but not both simultaneously.' For bidirectional control, use two separate pins or the Control Scripting feature."
> ← Specific, sourced, verifiable. You know exactly which document and page it came from.

The difference matters when you're building real systems.

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

## Architecture & Information Flow

### Complete Data Flow Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              INGESTION PHASE                               │
│                    (Run once per document/update)                          │
└─────────────────────────────────────────────────────────────────────────────┘

  Document Files
       │
       ▼
  ┌─────────┐
  │ Docling │  ← Extracts text from PDF, DOCX, PPTX, images (OCR)
  └────┬────┘
       │
       ▼
  ┌─────────────────┐
  │  Chunk & Split  │  ← Breaks into ~500 token chunks with overlap
  └────┬────────────┘
       │
       ├─────────────────────────────────────────────────────────────┐
       │                                                             │
       ▼                                                             ▼
  ┌──────────────┐                                          ┌──────────────┐
  │ BGE-M3       │                                          │ LLM Entity    │
  │ Embeddings   │                                          │ Extraction   │
  │ (local)      │                                          │ (Ollama)      │
  └──────┬───────┘                                          └──────┬───────┘
         │                                                          │
         │  [0.234, -0.567, ...]                                  │ GPIO, Core,
         │                                                          │ QSC, Relay...
         │                                                          │
         ▼                                                          ▼
  ┌──────────────┐                                          ┌──────────────┐
  │   Qdrant     │                                          │    Neo4j     │
  │  (Vectors)   │                                          │   (Graph)    │
  │              │                                          │              │
  │ Chunk 1: [a] │                                          │ (GPIO)-[:FOR]│
  │ Chunk 2: [b] │                                          │     ->(Relay) │
  │ Chunk 3: [c] │                                          │ (Core)-[:MADE]│
  └──────────────┘                                          │    ->(QSC)   │
                                                            └──────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│                               QUERY PHASE                                  │
│                        (Run every search - fast)                           │
└─────────────────────────────────────────────────────────────────────────────┘

  User Query: "How do I configure GPIO for relays?"
       │
       ▼
  ┌─────────────────┐
  │ Query Embedding │  ← Convert question to vector
  └────┬────────────┘
       │
       ├─────────────────────────────────────────────────────────────┐
       │                                                             │
       ▼                                                             ▼
  ┌──────────────┐                                          ┌──────────────┐
  │ Vector Search│                                          │Graph Traversal│
  │ (Qdrant)     │                                          │ (Neo4j)       │
  │              │                                          │              │
  │ Find chunks  │                                          │ Find GPIO →  │
  │ with similar │                                          │ Relay path   │
  │ meaning      │                                          │              │
  └──────┬───────┘                                          └──────┬───────┘
         │                                                          │
         │ Chunk 2 (0.89 match)                                     │ GPIO → Relay
         │ Chunk 7 (0.82 match)                                     │ → Core 110f
         │ Chunk 15 (0.76 match)                                    │
         │                                                          │
         └──────────────────────┬───────────────────────────────────┘
                                ▼
                       ┌──────────────────┐
                       │ PostgreSQL       │
                       │ Keyword Search   │
                       │ (exact matches)  │
                       └────────┬─────────┘
                                │
                                ▼
                       ┌──────────────────┐
                       │  Rank & Combine  │
                       │                  │
                       │ 1. Chunk 2      │
                       │ 2. Chunk 7      │
                       │ 3. Chunk 15     │
                       └────────┬─────────┘
                                │
                                ▼
                       ┌──────────────────┐
                       │    LLM Synthesize│
                       │    (Ollama)      │
                       │                  │
                       │ "According to    │
                       │  Core 110f docs  │
                       │  section 4.2..." │
                       └────────┬─────────┘
                                │
                                ▼
                      ┌─────────────────────┐
                      │  Response + Sources │
                      │                     │
                      │ Answer: [text]      │
                      │ Source: doc.pdf:42  │
                      │ Confidence: 0.89    │
                      └─────────────────────┘
```

### Component Summary

| Component | Purpose | Technology |
|-----------|---------|------------|
| **Docling** | Document parsing (PDF, images, etc.) | IBM Docling |
| **BGE-M3** | Convert text to meaning vectors | FlagEmbedding |
| **Qdrant** | Vector similarity search | Vector DB |
| **Neo4j** | Knowledge graph, relationships | Graph DB |
| **PostgreSQL** | Full-text keyword search | Relational DB |
| **Ollama** | Local LLM for entities + answers | Llama/Qwen/etc |

### Why Three Databases?

Each database excels at different queries:

```
"What documents mention GPIO?"           → PostgreSQL (keyword)
"What's similar to 'relay control'?"    → Qdrant (vector)
"What's related to the Core 110f?"      → Neo4j (graph)
"How do I wire relays to GPIO pins?"    → Hybrid (all three)
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
- **Explore the graph** at http://localhost:7474 to see relationships

## Further Reading

If you're new to RAG and vector databases:

- **Vector Search Explained**: Think of it like a library where books are shelved by "meaning" instead of alphabetically. Books on similar topics end up nearby, even if they use different words.
- **Knowledge Graphs**: Like a mind map of your documents. Concepts are nodes, relationships are edges. You can discover connections you'd never find with search.
- **Why Hybrid?**: Each approach has blind spots. Vectors miss exact matches. Keywords miss synonyms. Graphs miss unconnected context. Combine them = best results.

## Status

Currently in final beta. Core features work, but expect rough edges.

## License

Copyright (c) 2025 NerdBase-by-Stark. All rights reserved.
