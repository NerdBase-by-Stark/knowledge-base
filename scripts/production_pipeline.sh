#!/bin/bash
#===============================================================================
# PRODUCTION KB INGESTION PIPELINE
# Runs LLAMA + QWEN extraction, merges, cleans, and stores
#===============================================================================

set -e  # Exit on error

BASE_DIR="$HOME/ai/knowledge-base"
SOURCE_DIR="$BASE_DIR/production-ingest"
LOG_DIR="$BASE_DIR/logs"
SCRIPTS_DIR="$BASE_DIR/scripts"

# Config
THRESHOLD=0.20
MAX_RUNS=10
COLLECTION="qsys-lua-production"

# Create log directory
mkdir -p "$LOG_DIR"

# Timestamps for logs
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
MASTER_LOG="$LOG_DIR/pipeline_${TIMESTAMP}.log"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$MASTER_LOG"
}

#===============================================================================
# PHASE 1: LLAMA EXTRACTION
#===============================================================================
phase1_llama() {
    log "=========================================="
    log "PHASE 1: LLAMA3.1 EXTRACTION"
    log "=========================================="

    LLAMA_DIR="$BASE_DIR/prod-llama"
    mkdir -p "$LLAMA_DIR"
    cp "$SOURCE_DIR"/*.md "$LLAMA_DIR/"

    log "Starting LLAMA extraction (threshold: $THRESHOLD)..."
    python3 "$SCRIPTS_DIR/kb_ingest_robust_v2.py" \
        --source "$LLAMA_DIR" \
        --collection "${COLLECTION}-llama" \
        --model "llama3.1:8b" \
        --threshold "$THRESHOLD" \
        --max-runs "$MAX_RUNS" \
        --no-cleanup \
        2>&1 | tee -a "$LOG_DIR/llama_${TIMESTAMP}.log"

    log "LLAMA extraction complete"
}

#===============================================================================
# PHASE 2: QWEN EXTRACTION
#===============================================================================
phase2_qwen() {
    log "=========================================="
    log "PHASE 2: QWEN3 EXTRACTION"
    log "=========================================="

    QWEN_DIR="$BASE_DIR/prod-qwen"
    mkdir -p "$QWEN_DIR"
    cp "$SOURCE_DIR"/*.md "$QWEN_DIR/"

    log "Starting QWEN extraction (threshold: $THRESHOLD)..."
    python3 "$SCRIPTS_DIR/kb_ingest_robust_v2.py" \
        --source "$QWEN_DIR" \
        --collection "${COLLECTION}-qwen" \
        --model "Qwen3:latest" \
        --threshold "$THRESHOLD" \
        --max-runs "$MAX_RUNS" \
        --no-cleanup \
        2>&1 | tee -a "$LOG_DIR/qwen_${TIMESTAMP}.log"

    log "QWEN extraction complete"
}

#===============================================================================
# PHASE 3: MERGE + CLEANUP
#===============================================================================
phase3_merge_cleanup() {
    log "=========================================="
    log "PHASE 3: MERGE + CLEANUP"
    log "=========================================="

    python3 << 'PYTHON'
import json
import re
import ollama
from pathlib import Path

base = Path.home() / "ai/knowledge-base"

# Find latest checkpoints
def find_latest_checkpoint(dir_path):
    checkpoints = sorted(dir_path.glob("checkpoint_run*.json"),
                        key=lambda x: x.stat().st_mtime, reverse=True)
    return checkpoints[0] if checkpoints else None

llama_cp = find_latest_checkpoint(base / "prod-llama")
qwen_cp = find_latest_checkpoint(base / "prod-qwen")

print(f"LLAMA checkpoint: {llama_cp}")
print(f"QWEN checkpoint: {qwen_cp}")

# Load and merge
llama = json.load(open(llama_cp)) if llama_cp else {"known_entities": []}
qwen = json.load(open(qwen_cp)) if qwen_cp else {"known_entities": []}

llama_entities = set(llama.get("known_entities", []))
qwen_entities = set(qwen.get("known_entities", []))
merged = llama_entities | qwen_entities

print(f"LLAMA: {len(llama_entities)}")
print(f"QWEN: {len(qwen_entities)}")
print(f"MERGED: {len(merged)}")

# Rule-based cleanup
noise_patterns = [
    r'^[0-9.]+$', r'^[a-z]$', r'\.png$|\.jpg$|\.gif$|\.svg$',
    r'^(the|a|an|is|are|was|were|be|been|being)\b',
    r'^(this|that|these|those|it|its)\b',
    r'^(and|or|but|if|then|else|when|where|how|why|what)\b',
    r'^(can|could|will|would|shall|should|may|might|must)\b',
    r'^(true|false|null|none|nil|undefined|nan)\s*$',
    r'^default[_\s]', r'^example[_\s]', r'^usual\s',
    r'^\*\*.*\*\*$', r'^\.\.\.$', r'^[%#$&*\[\]()]+[^a-zA-Z]*$',
    r'^!\[', r'^\d+\.\d+\.\d+$',
    r'^(CONCEPT|FUNCTION|PARAMETER|PRODUCT|TECHNOLOGY|RELATIONSHIP|CLASS)$',
    r'.{80,}',
]
compiled = [re.compile(p, re.IGNORECASE) for p in noise_patterns]

clean = []
for e in merged:
    name = e.split(':')[0] if ':' in e else e
    if not any(p.search(name) for p in compiled):
        clean.append(e)

print(f"After rule cleanup: {len(clean)}")

# LLM cleanup
def strip_thinking(text):
    return re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()

batch_size = 30
discards = []
for i in range(0, len(clean), batch_size):
    batch = clean[i:i+batch_size]
    print(f"LLM batch {i//batch_size + 1}/{(len(clean)+batch_size-1)//batch_size}...")

    entity_list = "\n".join(f"  - {e}" for e in batch)
    prompt = f"""Review these entities. Identify ones to DISCARD (incomplete phrases, descriptions, generic terms).
ENTITIES:
{entity_list}
Return JSON: {{"discard": ["entity:TYPE", ...]}}"""

    try:
        response = ollama.chat(model='Qwen3:latest', messages=[{'role': 'user', 'content': prompt}])
        text = strip_thinking(response['message']['content'])
        match = re.search(r'\{.*\}', text, re.DOTALL)
        if match:
            result = json.loads(match.group())
            discards.extend(result.get('discard', []))
    except Exception as e:
        print(f"  Error: {e}")

discard_set = set(discards)
final = [e for e in clean if e not in discard_set]
print(f"After LLM cleanup: {len(final)}")

# Save merged checkpoint
merged_dir = base / "prod-merged"
merged_dir.mkdir(exist_ok=True)

# Use QWEN checkpoint as base (has chunks)
output = qwen.copy() if qwen else llama.copy()
output['known_entities'] = final

with open(merged_dir / "checkpoint_final.json", 'w') as f:
    json.dump(output, f, indent=2)

print(f"Saved to {merged_dir / 'checkpoint_final.json'}")
PYTHON

    log "Merge and cleanup complete"
}

#===============================================================================
# PHASE 4: STORAGE (ChromaDB + Neo4j)
#===============================================================================
phase4_storage() {
    log "=========================================="
    log "PHASE 4: STORAGE"
    log "=========================================="

    python3 << 'PYTHON'
import json
import chromadb
from pathlib import Path
from neo4j import GraphDatabase

base = Path.home() / "ai/knowledge-base"
checkpoint = json.load(open(base / "prod-merged/checkpoint_final.json"))

chunks = checkpoint.get('chunks', [])
entities = checkpoint.get('known_entities', [])

print(f"Storing {len(chunks)} chunks and {len(entities)} entities")

# ChromaDB
client = chromadb.PersistentClient(path=str(base / "chromadb"))
collection = client.get_or_create_collection("qsys-lua-production")

# Add chunks
for chunk in chunks:
    collection.upsert(
        ids=[chunk['chunk_id']],
        documents=[chunk['content']],
        metadatas=[{
            'doc_name': chunk.get('doc_name', ''),
            'doc_title': chunk.get('doc_title', ''),
            'context': chunk.get('context', ''),
            'summary': chunk.get('summary', ''),
        }]
    )

print(f"ChromaDB: stored {len(chunks)} chunks")

# Neo4j
driver = GraphDatabase.driver("bolt://localhost:7687", auth=("neo4j", "agentmemory123"))

with driver.session() as session:
    # Create entity nodes
    for e in entities:
        parts = e.rsplit(':', 1)
        name = parts[0] if len(parts) > 1 else e
        etype = parts[1] if len(parts) > 1 else 'UNKNOWN'

        session.run("""
            MERGE (e:Entity {name: $name})
            SET e.type = $type
        """, name=name, type=etype)

    # Link entities to chunks
    for chunk in chunks:
        chunk_entities = chunk.get('entities', [])
        for ent in chunk_entities:
            name = ent.get('name', '') if isinstance(ent, dict) else str(ent).split(':')[0]
            session.run("""
                MATCH (e:Entity {name: $name})
                MERGE (c:Chunk {id: $chunk_id})
                SET c.doc_name = $doc_name
                MERGE (c)-[:MENTIONS]->(e)
            """, name=name, chunk_id=chunk['chunk_id'], doc_name=chunk.get('doc_name', ''))

driver.close()
print(f"Neo4j: stored {len(entities)} entities with chunk links")
PYTHON

    log "Storage complete"
}

#===============================================================================
# MAIN
#===============================================================================
main() {
    log "=========================================="
    log "PRODUCTION PIPELINE STARTING"
    log "Source: $SOURCE_DIR"
    log "Threshold: $THRESHOLD"
    log "=========================================="

    # Run phases
    phase1_llama
    phase2_qwen
    phase3_merge_cleanup
    phase4_storage

    log "=========================================="
    log "PIPELINE COMPLETE"
    log "=========================================="
}

# Allow running individual phases
case "${1:-all}" in
    llama) phase1_llama ;;
    qwen) phase2_qwen ;;
    merge) phase3_merge_cleanup ;;
    storage) phase4_storage ;;
    all) main ;;
    *) echo "Usage: $0 {llama|qwen|merge|storage|all}" ;;
esac
