# Knowledge Base Project Checkpoint

**Date**: 2025-12-28
**Status**: Ready for full re-ingestion with qwen2.5:32b

## What's New (v2 Pipeline)
- Optimized prompts in `src/prompts.py` for local LLMs
- New `src/ingest_v2.py` with better progress tracking
- Tested with qwen2.5:32b - 13.8s/context generation, high quality

---

## Current State

### Services Running
```bash
# Check status
docker ps --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"

# Expected:
# kb-qdrant     - port 6333 (vectors)
# kb-postgres   - port 5433 (hybrid search)
# agent-memory-neo4j - port 7474 (knowledge graph)
```

### Data Ingested
- **Collection**: `qsys`
- **Vectors**: 2,098 chunks
- **Source**: 698 Q-SYS documentation files
- **Mode Used**: `quick` (no LLM enrichment)

### What's Missing (Quick Mode)
- ❌ Contextual retrieval (prepended context per chunk)
- ❌ Summary extraction
- ❌ Questions answered extraction
- ❌ LLM-based entity extraction

---

## To Resume: Full Re-Ingestion

### Step 1: Start Services
```bash
cd ~/ai/knowledge-base
docker compose up -d
```

### Step 2: Verify Ollama Models
```bash
ollama list
# Should have: qwen2.5:32b (or pull it: ollama pull qwen2.5:32b)
```

### Step 3: Re-ingest with Full Mode (v2 Pipeline)
```bash
cd ~/ai/knowledge-base

# RECOMMENDED: Use v2 pipeline with optimized prompts
python -m src.ingest_v2 ingest ~/qsys/markdown \
    --collection qsys-full \
    --model qwen2.5:32b \
    --ext md

# Estimated time: ~2 min/file × 698 files = ~23 hours
# (can run overnight or in background with nohup)

# To run in background:
nohup python -m src.ingest_v2 ingest ~/qsys/markdown \
    --collection qsys-full --model qwen2.5:32b --ext md \
    > ingestion.log 2>&1 &
```

### Step 4: Monitor Progress
```bash
~/ai/knowledge-base/scripts/check_progress.sh
# Or: watch -n 30 ~/ai/knowledge-base/scripts/check_progress.sh
```

---

## Key Files

| File | Purpose |
|------|---------|
| `~/ai/knowledge-base/docker-compose.yml` | Qdrant + PostgreSQL |
| `~/ai/knowledge-base/cli.py` | Main CLI interface |
| `~/ai/knowledge-base/src/ingest.py` | Enhanced ingestion pipeline |
| `~/ai/knowledge-base/src/knowledge_base.py` | Unified KB interface |
| `~/.claude/skills/kb-ingest/SKILL.md` | Ingestion skill for Claude |

---

## Quick Commands

```bash
# Search
cd ~/ai/knowledge-base
python cli.py search "your query" -c qsys

# Stats
python cli.py stats

# Start API + Dashboard
python cli.py serve --dashboard
# API: http://localhost:8000/docs
# Dashboard: http://localhost:8501

# Check vectors
curl -s http://localhost:6333/collections/knowledge_base | python3 -c "import sys,json; d=json.load(sys.stdin); print(f'Vectors: {d[\"result\"][\"points_count\"]}')"
```

---

## Ingestion Modes Reference

| Mode | LLM Calls | Time/File | Features |
|------|-----------|-----------|----------|
| `quick` | 0 | ~35s | Chunking + embeddings only |
| `standard` | 0 | ~35s | + rule-based entities, keywords |
| `full` | 4-10 | ~2-3min | + contextual, summary, QA, LLM entities |

---

## Model Recommendations

| Model | Size | Quality | Time/Chunk |
|-------|------|---------|------------|
| `qwen2.5:7b` | 4.7GB | Good | ~5s |
| `qwen2.5:32b` | 19GB | Better | ~18s |
| `llama3.3:70b` | ~40GB | Best (GPT-4 class) | ~30-45s |

---

## Architecture Diagram

```
Documents → Docling → Chunking → [LLM Context] → BGE-M3 Embeddings
                                       ↓
                    ┌──────────────────┼──────────────────┐
                    ↓                  ↓                  ↓
              Qdrant:6333        Neo4j:7474        PostgreSQL:5433
              (vectors)          (graph)           (full-text)
                    └──────────────────┼──────────────────┘
                                       ↓
                              LlamaIndex Hybrid Search
                                       ↓
                              API:8000 / Dashboard:8501
```

---

## Next Steps (When Resuming)

1. **Choose model**: `qwen2.5:32b` (balanced) or `llama3.3:70b` (best quality)
2. **Update config**: Edit `src/ingest.py` line 123 to change `llm_model`
3. **Clear old data**: `python cli.py init --recreate` (optional)
4. **Run full ingestion**: See Step 3 above
5. **Validate**: Test search quality after ingestion
