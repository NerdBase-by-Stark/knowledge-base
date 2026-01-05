#!/usr/bin/env python3
"""
MULTIPASS 3X RUN
Runs the complete 6-pass extraction 3 more times, tracking NEW entities found
in each complete run vs what was already discovered.
"""
import json
import time
import hashlib
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Set
from pathlib import Path
from datetime import datetime
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import ollama
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct
from rich.console import Console
from rich.table import Table

console = Console()

# Configuration
DOCS_DIR = Path.home() / "ai/knowledge-base/scrape-jobs/qsys-lua/markdown"
RESULTS_DIR = Path.home() / "ai/knowledge-base/multipass_3x_results"
RESULTS_DIR.mkdir(exist_ok=True)
PROGRESS_FILE = Path.home() / "multipass_3x_progress.txt"

MODEL = "qwen2.5:32b"
NUM_RUNS = 3
NUM_PASSES = 6
CHUNK_SIZE = 512
CHUNK_OVERLAP = 75

def log_progress(msg: str):
    """Write progress to file for monitoring"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(PROGRESS_FILE, "a") as f:
        f.write(f"[{timestamp}] {msg}\n")
    print(f"[{timestamp}] {msg}", flush=True)

# Same prompts as before
ENTITY_PROMPTS = [
    """Extract technical entities from this Q-SYS/Lua documentation. Return JSON array only.
Entity types: PRODUCT, TECHNOLOGY, CONCEPT, PARAMETER, FUNCTION, CLASS, EVENT, PROPERTY
Text:
---
{chunk}
---
Return JSON like: [{{"name": "Entity", "type": "TYPE"}}]
Extract up to 10 entities. JSON only:""",

    """Extract ALL function names, method names, and API calls from this text. Return JSON array.
Text:
---
{chunk}
---
Focus on: function names, method calls, callbacks, event handlers, API endpoints.
Return JSON like: [{{"name": "FunctionName", "type": "FUNCTION"}}]
JSON only:""",

    """Extract ALL parameters, properties, configuration options, and settings from this text.
Text:
---
{chunk}
---
Focus on: function parameters, object properties, config options, default values.
Return JSON like: [{{"name": "paramName", "type": "PARAMETER"}}]
JSON only:""",

    """Extract programming concepts, design patterns, and technical terminology from this text.
Text:
---
{chunk}
---
Focus on: programming patterns, concepts, terminology, best practices.
Return JSON like: [{{"name": "Concept", "type": "CONCEPT"}}]
JSON only:""",

    """Extract entities that RELATE to other entities - things that connect, inherit, or depend on each other.
Text:
---
{chunk}
---
Focus on: parent classes, dependencies, related components, protocols.
Return JSON like: [{{"name": "Entity", "type": "RELATIONSHIP"}}]
JSON only:""",

    """Final pass: Extract ANY remaining technical terms, names, or identifiers not yet captured.
Text:
---
{chunk}
---
Look for: variable names, constants, enums, error codes, file types, anything technical.
Return JSON like: [{{"name": "term", "type": "OTHER"}}]
JSON only:"""
]

CONTEXT_PROMPT = """Write a brief context statement (2-3 sentences) for this Q-SYS documentation chunk.
Document: {doc_title}
Chunk:
---
{chunk}
---
Context (2-3 sentences):"""

SUMMARY_PROMPT = """Summarize this technical documentation in ONE sentence:
---
{chunk}
---
One sentence summary:"""

QA_PROMPT = """What questions can this documentation answer? List 3 specific questions.
---
{chunk}
---
Questions (one per line):"""

@dataclass
class ChunkData:
    chunk_id: str
    doc_name: str
    doc_title: str
    chunk_index: int
    content: str
    context: str = ""
    entities: List[Dict] = field(default_factory=list)
    summary: str = ""
    questions: List[str] = field(default_factory=list)

@dataclass
class RunStats:
    run_number: int
    new_entities: int
    total_extracted: int
    cumulative_unique: int
    time_seconds: float
    pass_breakdown: List[Dict] = field(default_factory=list)

def simple_chunk(text: str) -> List[str]:
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = start + CHUNK_SIZE
        chunk = " ".join(words[start:end])
        if chunk.strip():
            chunks.append(chunk)
        start = end - CHUNK_OVERLAP
    return chunks

def load_all_docs() -> List[ChunkData]:
    all_chunks = []
    md_files = sorted(DOCS_DIR.glob("*.md"))

    for doc_path in md_files:
        doc_name = doc_path.name
        doc_title = doc_name.replace(".md", "").replace("_", " ")
        text = doc_path.read_text()
        chunks = simple_chunk(text)

        for i, chunk in enumerate(chunks):
            chunk_id = hashlib.md5(f"{doc_name}:{i}:{chunk[:50]}".encode()).hexdigest()[:12]
            all_chunks.append(ChunkData(
                chunk_id=chunk_id,
                doc_name=doc_name,
                doc_title=doc_title,
                chunk_index=i,
                content=chunk
            ))

    return all_chunks

def call_llm(prompt: str, max_tokens: int = 300) -> str:
    for attempt in range(3):
        try:
            response = ollama.chat(
                model=MODEL,
                messages=[{"role": "user", "content": prompt}],
                options={"temperature": 0.3, "num_predict": max_tokens}
            )
            return response["message"]["content"]
        except Exception as e:
            if attempt < 2:
                time.sleep(2)
            else:
                return f"ERROR: {e}"
    return "ERROR: Max retries"

def parse_entities(text: str) -> List[Dict]:
    try:
        start = text.find("[")
        end = text.rfind("]") + 1
        if start >= 0 and end > start:
            entities = json.loads(text[start:end])
            return [{"name": e.get("name", "").strip(), "type": e.get("type", "OTHER")}
                    for e in entities if e.get("name", "").strip()]
    except:
        pass
    return []

def parse_questions(text: str) -> List[str]:
    lines = text.strip().split("\n")
    questions = []
    for line in lines:
        line = line.strip()
        if line and not line.startswith("#"):
            if len(line) > 1 and line[0].isdigit() and line[1] in ".)":
                line = line[2:].strip()
            if line:
                questions.append(line)
    return questions[:3]

def run_single_complete_multipass(chunks: List[ChunkData], run_num: int,
                                   global_known: Set[str]) -> RunStats:
    """Run a complete 6-pass extraction"""
    log_progress(f"")
    log_progress(f"{'='*60}")
    log_progress(f"RUN {run_num}/{NUM_RUNS} STARTING")
    log_progress(f"{'='*60}")
    log_progress(f"Known entities before this run: {len(global_known)}")

    run_start = time.time()
    run_new_entities = 0
    run_total_extracted = 0
    pass_breakdown = []

    focus_areas = ["Standard", "Functions", "Parameters", "Concepts", "Relationships", "Catch-all"]

    for pass_num in range(1, NUM_PASSES + 1):
        log_progress(f"  Pass {pass_num}/6 ({focus_areas[pass_num-1]})...")
        pass_start = time.time()
        pass_new = 0
        pass_total = 0

        prompt_template = ENTITY_PROMPTS[pass_num - 1]

        for idx, chunk in enumerate(chunks):
            # First pass of first run does context/summary (only once ever)
            if run_num == 1 and pass_num == 1:
                if not chunk.context:
                    resp = call_llm(CONTEXT_PROMPT.format(doc_title=chunk.doc_title, chunk=chunk.content), 150)
                    if not resp.startswith("ERROR"):
                        chunk.context = resp

                if not chunk.summary:
                    resp = call_llm(SUMMARY_PROMPT.format(chunk=chunk.content), 100)
                    if not resp.startswith("ERROR"):
                        chunk.summary = resp

                if not chunk.questions:
                    resp = call_llm(QA_PROMPT.format(chunk=chunk.content), 200)
                    if not resp.startswith("ERROR"):
                        chunk.questions = parse_questions(resp)

            # Entity extraction
            resp = call_llm(prompt_template.format(chunk=chunk.content), 400)
            if not resp.startswith("ERROR"):
                entities = parse_entities(resp)

                for ent in entities:
                    ent_key = f"{ent['name'].lower()}:{ent['type']}"
                    pass_total += 1
                    run_total_extracted += 1

                    if ent_key not in global_known:
                        global_known.add(ent_key)
                        pass_new += 1
                        run_new_entities += 1
                        if ent not in chunk.entities:
                            chunk.entities.append(ent)

            # Progress every 20 chunks
            if (idx + 1) % 20 == 0:
                log_progress(f"    {idx+1}/{len(chunks)} chunks, {pass_new} new this pass")

        pass_time = time.time() - pass_start
        pass_breakdown.append({
            "pass": pass_num,
            "focus": focus_areas[pass_num-1],
            "new_entities": pass_new,
            "total_extracted": pass_total,
            "time_seconds": pass_time
        })
        log_progress(f"  Pass {pass_num} done: {pass_new} new, {pass_total} total, {pass_time:.0f}s")

    run_time = time.time() - run_start

    stats = RunStats(
        run_number=run_num,
        new_entities=run_new_entities,
        total_extracted=run_total_extracted,
        cumulative_unique=len(global_known),
        time_seconds=run_time,
        pass_breakdown=pass_breakdown
    )

    log_progress(f"")
    log_progress(f"RUN {run_num} COMPLETE:")
    log_progress(f"  New entities this run: {run_new_entities}")
    log_progress(f"  Total extracted: {run_total_extracted}")
    log_progress(f"  Cumulative unique: {len(global_known)}")
    log_progress(f"  Time: {run_time/60:.1f} minutes")

    # Save checkpoint
    checkpoint = {
        "run": run_num,
        "stats": asdict(stats),
        "cumulative_unique": len(global_known),
        "timestamp": datetime.now().isoformat()
    }
    with open(RESULTS_DIR / f"checkpoint_run{run_num}.json", "w") as f:
        json.dump(checkpoint, f, indent=2)

    return stats

def store_to_qdrant(chunks: List[ChunkData], collection: str):
    """Store to Qdrant"""
    from FlagEmbedding import BGEM3FlagModel

    log_progress(f"Storing to Qdrant collection: {collection}")

    embedder = BGEM3FlagModel("BAAI/bge-m3", use_fp16=False)
    client = QdrantClient(host="localhost", port=6333)

    try:
        client.delete_collection(collection)
    except:
        pass

    client.create_collection(
        collection_name=collection,
        vectors_config=VectorParams(size=1024, distance=Distance.COSINE)
    )

    batch_size = 10
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i+batch_size]

        contents = [f"{c.context}\n\n{c.content}" if c.context else c.content for c in batch]
        embeddings = embedder.encode(contents)["dense_vecs"]

        points = []
        for j, c in enumerate(batch):
            points.append(PointStruct(
                id=abs(hash(c.chunk_id)) % (2**63),
                vector=embeddings[j].tolist(),
                payload={
                    "chunk_id": c.chunk_id,
                    "doc_name": c.doc_name,
                    "title": c.doc_title,
                    "content": c.content,
                    "context": c.context,
                    "summary": c.summary,
                    "entities": c.entities,
                    "questions": c.questions,
                    "entity_count": len(c.entities)
                }
            ))

        client.upsert(collection_name=collection, points=points)

        if (i + batch_size) % 50 == 0:
            log_progress(f"  Stored {i + batch_size}/{len(chunks)} chunks")

    log_progress(f"Stored {len(chunks)} chunks to Qdrant")

def link_graph(chunks: List[ChunkData]):
    """Link entities in Neo4j"""
    from neo4j import GraphDatabase

    log_progress("Linking entities in Neo4j...")

    driver = GraphDatabase.driver("bolt://localhost:7687", auth=("neo4j", "agentmemory123"))

    entity_counts = {}
    for chunk in chunks:
        for ent in chunk.entities:
            key = (ent["name"], ent["type"])
            entity_counts[key] = entity_counts.get(key, 0) + 1

    with driver.session() as session:
        for (name, etype), count in entity_counts.items():
            session.run("""
                MERGE (e:Entity {name: $name})
                SET e.type = $type, e.count = $count
            """, name=name, type=etype, count=count)

        for chunk in chunks:
            session.run("""
                MERGE (c:Chunk {id: $id})
                SET c.doc = $doc, c.title = $title
            """, id=chunk.chunk_id, doc=chunk.doc_name, title=chunk.doc_title)

            for ent in chunk.entities:
                session.run("""
                    MATCH (c:Chunk {id: $chunk_id})
                    MATCH (e:Entity {name: $name})
                    MERGE (c)-[:CONTAINS]->(e)
                """, chunk_id=chunk.chunk_id, name=ent["name"])

    driver.close()
    log_progress(f"Linked {len(entity_counts)} unique entities in Neo4j")

def main():
    # Clear progress file
    PROGRESS_FILE.write_text("")

    log_progress("="*60)
    log_progress("MULTIPASS 3X RUN")
    log_progress(f"Model: {MODEL}")
    log_progress(f"Complete runs: {NUM_RUNS}")
    log_progress(f"Passes per run: {NUM_PASSES}")
    log_progress("="*60)

    # Load documents
    chunks = load_all_docs()
    log_progress(f"Loaded {len(chunks)} chunks from {len(set(c.doc_name for c in chunks))} documents")

    # Load existing entities from first run
    first_run_results = Path.home() / "ai/knowledge-base/multipass_results/final_results.json"
    global_known: Set[str] = set()

    if first_run_results.exists():
        log_progress("Loading entities from initial multipass run...")
        # We need to reconstruct the known entities from the chunks
        # Load the checkpoint to get entity count
        with open(first_run_results) as f:
            initial_data = json.load(f)
        log_progress(f"Initial run had {initial_data['unique_entities']} unique entities")

        # Load from Qdrant to get actual entities
        client = QdrantClient(host="localhost", port=6333)
        try:
            points = client.scroll(
                collection_name="qsys-lua-multipass-20251231",
                limit=1000,
                with_payload=True
            )[0]
            for point in points:
                for ent in point.payload.get("entities", []):
                    ent_key = f"{ent['name'].lower()}:{ent['type']}"
                    global_known.add(ent_key)
                    # Also add to chunk entities
                    for chunk in chunks:
                        if chunk.chunk_id == point.payload.get("chunk_id"):
                            if ent not in chunk.entities:
                                chunk.entities.append(ent)
            log_progress(f"Loaded {len(global_known)} unique entities from previous run")
        except Exception as e:
            log_progress(f"Could not load from Qdrant: {e}")
            log_progress("Starting fresh...")

    initial_count = len(global_known)

    # Run 3 more complete multipass runs
    all_stats: List[RunStats] = []

    for run_num in range(1, NUM_RUNS + 1):
        stats = run_single_complete_multipass(chunks, run_num, global_known)
        all_stats.append(stats)

    # Final summary
    log_progress("")
    log_progress("="*60)
    log_progress("FINAL SUMMARY - 3X MULTIPASS COMPARISON")
    log_progress("="*60)
    log_progress("")
    log_progress(f"Initial entities (from first multipass): {initial_count}")
    log_progress("")

    for stats in all_stats:
        pct_new = (stats.new_entities / stats.total_extracted * 100) if stats.total_extracted > 0 else 0
        log_progress(f"Run {stats.run_number}:")
        log_progress(f"  New entities: {stats.new_entities}")
        log_progress(f"  Total extracted: {stats.total_extracted}")
        log_progress(f"  % New: {pct_new:.1f}%")
        log_progress(f"  Cumulative unique: {stats.cumulative_unique}")
        log_progress(f"  Time: {stats.time_seconds/60:.1f} min")
        log_progress("")

    # Diminishing returns analysis
    log_progress("DIMINISHING RETURNS:")
    prev = initial_count
    for stats in all_stats:
        added = stats.new_entities
        if prev > 0:
            pct_gain = (added / prev * 100)
        else:
            pct_gain = 100
        log_progress(f"  Run {stats.run_number}: +{added} entities (+{pct_gain:.1f}% over previous)")
        prev = stats.cumulative_unique

    # Store final results
    collection = f"qsys-lua-3x-{datetime.now().strftime('%Y%m%d-%H%M')}"
    store_to_qdrant(chunks, collection)
    link_graph(chunks)

    # Save final results
    total_entities = sum(len(c.entities) for c in chunks)
    results = {
        "model": MODEL,
        "initial_entities": initial_count,
        "total_runs": NUM_RUNS,
        "passes_per_run": NUM_PASSES,
        "final_unique_entities": len(global_known),
        "total_entities_on_chunks": total_entities,
        "collection": collection,
        "run_stats": [asdict(s) for s in all_stats],
        "completed": datetime.now().isoformat()
    }

    with open(RESULTS_DIR / "final_3x_results.json", "w") as f:
        json.dump(results, f, indent=2)

    log_progress("")
    log_progress("="*60)
    log_progress("COMPLETE!")
    log_progress(f"Initial: {initial_count} → Final: {len(global_known)} entities")
    log_progress(f"Total gain: +{len(global_known) - initial_count} entities")
    log_progress(f"Collection: {collection}")
    log_progress(f"Results: {RESULTS_DIR / 'final_3x_results.json'}")
    log_progress("="*60)

if __name__ == "__main__":
    main()
