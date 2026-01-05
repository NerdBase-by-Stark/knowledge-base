#!/usr/bin/env python3
"""
OVERNIGHT MULTI-PASS RAG INGESTION
6 passes with diminishing returns tracking

Each pass re-processes chunks and attempts to extract MORE entities,
tracking what's new vs what was already found.
"""
import json
import time
import hashlib
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Set, Optional
from pathlib import Path
from datetime import datetime
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import ollama
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct
from rich.console import Console
from rich.progress import Progress, BarColumn, TextColumn, TimeElapsedColumn, SpinnerColumn
from rich.panel import Panel
from rich.table import Table

console = Console()

# Configuration
DOCS_DIR = Path.home() / "ai/knowledge-base/scrape-jobs/qsys-lua/markdown"
RESULTS_DIR = Path.home() / "ai/knowledge-base/multipass_results"
RESULTS_DIR.mkdir(exist_ok=True)
PROGRESS_FILE = Path.home() / "multipass_progress.txt"

MODEL = "qwen2.5:32b"
NUM_PASSES = 6
CHUNK_SIZE = 512
CHUNK_OVERLAP = 75

def log_progress(msg: str):
    """Write progress to file for monitoring"""
    timestamp = datetime.now().strftime("%H:%M:%S")
    with open(PROGRESS_FILE, "a") as f:
        f.write(f"[{timestamp}] {msg}\n")
    print(f"[{timestamp}] {msg}", flush=True)

# Prompts - each pass uses slightly different prompts to find more
ENTITY_PROMPTS = [
    # Pass 1: Standard extraction
    """Extract technical entities from this Q-SYS/Lua documentation. Return JSON array only.

Entity types: PRODUCT, TECHNOLOGY, CONCEPT, PARAMETER, FUNCTION, CLASS, EVENT, PROPERTY

Text:
---
{chunk}
---

Return JSON like: [{{"name": "Entity", "type": "TYPE"}}]
Extract up to 10 entities. JSON only:""",

    # Pass 2: Focus on functions and methods
    """Extract ALL function names, method names, and API calls from this text. Return JSON array.

Text:
---
{chunk}
---

Focus on: function names, method calls, callbacks, event handlers, API endpoints.
Return JSON like: [{{"name": "FunctionName", "type": "FUNCTION"}}]
JSON only:""",

    # Pass 3: Focus on parameters and properties
    """Extract ALL parameters, properties, configuration options, and settings from this text.

Text:
---
{chunk}
---

Focus on: function parameters, object properties, config options, default values.
Return JSON like: [{{"name": "paramName", "type": "PARAMETER"}}]
JSON only:""",

    # Pass 4: Focus on concepts and patterns
    """Extract programming concepts, design patterns, and technical terminology from this text.

Text:
---
{chunk}
---

Focus on: programming patterns, concepts, terminology, best practices.
Return JSON like: [{{"name": "Concept", "type": "CONCEPT"}}]
JSON only:""",

    # Pass 5: Focus on relationships and connections
    """Extract entities that RELATE to other entities - things that connect, inherit, or depend on each other.

Text:
---
{chunk}
---

Focus on: parent classes, dependencies, related components, protocols.
Return JSON like: [{{"name": "Entity", "type": "RELATIONSHIP"}}]
JSON only:""",

    # Pass 6: Catch-all for anything missed
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
class PassStats:
    pass_number: int
    new_entities: int
    total_entities: int
    unique_entity_names: int
    time_seconds: float
    chunks_processed: int

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
    pass_entities: Dict[int, List[Dict]] = field(default_factory=dict)  # entities by pass

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
    console.print(f"[cyan]Found {len(md_files)} documents[/]")

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

    console.print(f"[green]Total chunks: {len(all_chunks)}[/]")
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
            # Normalize
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

def run_pass(chunks: List[ChunkData], pass_num: int, known_entities: Set[str]) -> PassStats:
    """Run a single extraction pass"""
    focus_areas = ["Standard", "Functions", "Parameters", "Concepts", "Relationships", "Catch-all"]
    log_progress(f"═══ PASS {pass_num}/{NUM_PASSES} ({focus_areas[pass_num-1]}) ═══")
    console.print(f"\n[bold cyan]═══ PASS {pass_num}/{NUM_PASSES} ═══[/]")

    prompt_template = ENTITY_PROMPTS[pass_num - 1]
    start_time = time.time()
    new_entities = 0
    pass_total = 0

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        TimeElapsedColumn(),
        console=console
    ) as prog:
        task = prog.add_task(f"[cyan]Pass {pass_num}[/]", total=len(chunks))

        for idx, chunk in enumerate(chunks):
            # First pass also does context/summary/questions
            if pass_num == 1:
                resp = call_llm(CONTEXT_PROMPT.format(doc_title=chunk.doc_title, chunk=chunk.content), 150)
                if not resp.startswith("ERROR"):
                    chunk.context = resp

                resp = call_llm(SUMMARY_PROMPT.format(chunk=chunk.content), 100)
                if not resp.startswith("ERROR"):
                    chunk.summary = resp

                resp = call_llm(QA_PROMPT.format(chunk=chunk.content), 200)
                if not resp.startswith("ERROR"):
                    chunk.questions = parse_questions(resp)

            # Entity extraction for this pass
            resp = call_llm(prompt_template.format(chunk=chunk.content), 400)
            if not resp.startswith("ERROR"):
                entities = parse_entities(resp)
                chunk.pass_entities[pass_num] = entities

                for ent in entities:
                    ent_key = f"{ent['name'].lower()}:{ent['type']}"
                    pass_total += 1
                    if ent_key not in known_entities:
                        known_entities.add(ent_key)
                        new_entities += 1
                        # Add to main entity list
                        if ent not in chunk.entities:
                            chunk.entities.append(ent)

            prog.advance(task)

            # Log progress every 10 chunks
            if (idx + 1) % 10 == 0 or idx == len(chunks) - 1:
                log_progress(f"  Pass {pass_num}: {idx+1}/{len(chunks)} chunks, {new_entities} new entities")

    elapsed = time.time() - start_time
    log_progress(f"  Pass {pass_num} COMPLETE: {new_entities} new, {pass_total} total, {elapsed:.1f}s")

    stats = PassStats(
        pass_number=pass_num,
        new_entities=new_entities,
        total_entities=pass_total,
        unique_entity_names=len(known_entities),
        time_seconds=elapsed,
        chunks_processed=len(chunks)
    )

    console.print(f"  [green]New entities:[/] {new_entities}")
    console.print(f"  [yellow]Pass found:[/] {pass_total}")
    console.print(f"  [blue]Total unique:[/] {len(known_entities)}")
    console.print(f"  [dim]Time:[/] {elapsed:.1f}s")

    return stats

def store_to_qdrant(chunks: List[ChunkData], collection: str):
    """Store to Qdrant"""
    from FlagEmbedding import BGEM3FlagModel

    console.print("\n[cyan]Loading embedding model...[/]")
    embedder = BGEM3FlagModel("BAAI/bge-m3", use_fp16=False)

    console.print("[cyan]Connecting to Qdrant...[/]")
    client = QdrantClient(host="localhost", port=6333)

    try:
        client.delete_collection(collection)
    except:
        pass

    client.create_collection(
        collection_name=collection,
        vectors_config=VectorParams(size=1024, distance=Distance.COSINE)
    )

    console.print(f"[cyan]Storing {len(chunks)} chunks...[/]")

    batch_size = 10
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        console=console
    ) as prog:
        task = prog.add_task("[cyan]Embedding & storing[/]", total=len(chunks))

        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i+batch_size]

            contents = []
            for c in batch:
                ctx_content = f"{c.context}\n\n{c.content}" if c.context else c.content
                contents.append(ctx_content)

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
            prog.advance(task, len(batch))

    console.print(f"[green]Stored {len(chunks)} chunks to '{collection}'[/]")

def link_graph(chunks: List[ChunkData]):
    """Link entities in Neo4j"""
    from neo4j import GraphDatabase

    console.print("\n[cyan]Linking entities in Neo4j...[/]")

    driver = GraphDatabase.driver("bolt://localhost:7687", auth=("neo4j", "agentmemory123"))

    entity_counts = {}
    for chunk in chunks:
        for ent in chunk.entities:
            key = (ent["name"], ent["type"])
            entity_counts[key] = entity_counts.get(key, 0) + 1

    with driver.session() as session:
        # Create entities
        for (name, etype), count in entity_counts.items():
            session.run("""
                MERGE (e:Entity {name: $name})
                SET e.type = $type, e.count = $count
            """, name=name, type=etype, count=count)

        # Create chunk nodes and relationships
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
    console.print(f"[green]Linked {len(entity_counts)} unique entities[/]")

def main():
    # Clear progress file
    PROGRESS_FILE.write_text("")
    log_progress("═══════════════════════════════════════════")
    log_progress("OVERNIGHT MULTI-PASS RAG INGESTION")
    log_progress(f"Model: {MODEL}")
    log_progress(f"Passes: {NUM_PASSES}")
    log_progress("═══════════════════════════════════════════")

    console.print(Panel.fit(
        f"[bold]OVERNIGHT MULTI-PASS RAG INGESTION[/]\n"
        f"Model: {MODEL}\n"
        f"Passes: {NUM_PASSES}",
        title="Starting"
    ))

    # Load documents
    chunks = load_all_docs()
    log_progress(f"Loaded {len(chunks)} chunks from {len(set(c.doc_name for c in chunks))} documents")

    # Track all passes
    all_stats: List[PassStats] = []
    known_entities: Set[str] = set()

    # Run each pass
    for pass_num in range(1, NUM_PASSES + 1):
        stats = run_pass(chunks, pass_num, known_entities)
        all_stats.append(stats)

        # Save checkpoint after each pass
        checkpoint = {
            "pass": pass_num,
            "stats": [asdict(s) for s in all_stats],
            "total_unique": len(known_entities),
            "timestamp": datetime.now().isoformat()
        }
        with open(RESULTS_DIR / "checkpoint.json", "w") as f:
            json.dump(checkpoint, f, indent=2)

    # Display results table
    console.print("\n")
    table = Table(title="MULTI-PASS EXTRACTION RESULTS")
    table.add_column("Pass", style="cyan", justify="right")
    table.add_column("Focus", style="dim")
    table.add_column("New Entities", style="green", justify="right")
    table.add_column("Pass Total", style="yellow", justify="right")
    table.add_column("Cumulative", style="bold", justify="right")
    table.add_column("Time", style="dim", justify="right")

    focus_areas = ["Standard", "Functions", "Parameters", "Concepts", "Relationships", "Catch-all"]
    cumulative = 0
    for i, stats in enumerate(all_stats):
        cumulative += stats.new_entities
        table.add_row(
            str(stats.pass_number),
            focus_areas[i],
            str(stats.new_entities),
            str(stats.total_entities),
            str(cumulative),
            f"{stats.time_seconds:.0f}s"
        )

    console.print(table)

    # Show diminishing returns
    console.print("\n[bold]Diminishing Returns Analysis:[/]")
    if len(all_stats) > 1:
        for i in range(1, len(all_stats)):
            prev = all_stats[i-1].new_entities
            curr = all_stats[i].new_entities
            if prev > 0:
                pct = ((prev - curr) / prev) * 100
                console.print(f"  Pass {i} → {i+1}: {prev} → {curr} ([red]-{pct:.1f}%[/])")

    # Store to Qdrant
    collection = f"qsys-lua-multipass-{datetime.now().strftime('%Y%m%d')}"
    store_to_qdrant(chunks, collection)

    # Link graph
    link_graph(chunks)

    # Save final results
    total_entities = sum(len(c.entities) for c in chunks)
    results = {
        "model": MODEL,
        "passes": NUM_PASSES,
        "collection": collection,
        "total_chunks": len(chunks),
        "total_entities": total_entities,
        "unique_entities": len(known_entities),
        "pass_stats": [asdict(s) for s in all_stats],
        "completed": datetime.now().isoformat()
    }

    results_file = RESULTS_DIR / "final_results.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)

    # Final summary
    console.print(Panel.fit(
        f"[bold green]MULTI-PASS INGESTION COMPLETE[/]\n\n"
        f"Model: {MODEL}\n"
        f"Passes: {NUM_PASSES}\n"
        f"Chunks: {len(chunks)}\n"
        f"Total Entities: {total_entities}\n"
        f"Unique Entities: {len(known_entities)}\n"
        f"Collection: {collection}\n\n"
        f"Results: {results_file}",
        title="Success"
    ))

if __name__ == "__main__":
    main()
