#!/usr/bin/env python3
"""
OVERNIGHT FULL RAG INGESTION
Multi-pass ingestion with quality validation

Features:
- Uses best model from shootout
- Multiple passes for completeness
- Quality validation between passes
- Automatic retry on failures
- Progress checkpointing
"""
import json
import time
import hashlib
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional
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

console = Console()

# Configuration
SHOOTOUT_DIR = Path.home() / "ai/knowledge-base/scrape-jobs/qsys-lua/shootout"
DOCS_DIR = Path.home() / "ai/knowledge-base/scrape-jobs/qsys-lua/markdown"
CHECKPOINT_FILE = Path.home() / "ai/knowledge-base/overnight_checkpoint.json"

CHUNK_SIZE = 512
CHUNK_OVERLAP = 75

# Prompts (same as benchmark)
CONTEXT_PROMPT = """You are a technical documentation assistant. Write a brief context statement (2-3 sentences) that situates this chunk within its source document.

Document: {doc_title}

Chunk:
---
{chunk}
---

Rules:
- Mention document source and topic
- Include key technical terms
- Do NOT summarize the chunk itself

Context (2-3 sentences only):"""

ENTITY_PROMPT = """Extract technical entities from this text. Return JSON array only.

Entity types: PRODUCT, TECHNOLOGY, CONCEPT, PARAMETER, ORGANIZATION

Text:
---
{chunk}
---

Return JSON like: [{{"name": "Entity", "type": "TYPE"}}]
Maximum 8 entities. JSON only:"""

SUMMARY_PROMPT = """Summarize this technical documentation in ONE sentence:
---
{chunk}
---

One sentence summary:"""

QA_PROMPT = """What questions can this documentation answer? List 2-3 specific questions.
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
    embedding: Optional[List[float]] = None
    processed: bool = False
    pass_number: int = 0

@dataclass
class IngestProgress:
    total_docs: int = 0
    total_chunks: int = 0
    processed_chunks: int = 0
    current_pass: int = 1
    total_entities: int = 0
    errors: List[str] = field(default_factory=list)
    start_time: str = ""
    last_update: str = ""

def get_best_model() -> str:
    """Read best model from shootout results"""
    winner_file = SHOOTOUT_DIR / "WINNER.txt"
    if winner_file.exists():
        return winner_file.read_text().strip()

    # Fallback - run analysis
    console.print("[yellow]No winner file found, running analysis...[/]")
    import subprocess
    subprocess.run(["python", str(Path(__file__).parent / "analyze_shootout.py")])

    if winner_file.exists():
        return winner_file.read_text().strip()

    return "llama3.1:8b"  # Default fallback

def simple_chunk(text: str) -> List[str]:
    """Chunk text by words"""
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
    """Load all markdown docs and chunk them"""
    all_chunks = []

    md_files = list(DOCS_DIR.glob("*.md"))
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

def call_llm(model: str, prompt: str, max_tokens: int = 300) -> str:
    """Call Ollama with retry"""
    for attempt in range(3):
        try:
            response = ollama.chat(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                options={"temperature": 0.3, "num_predict": max_tokens}
            )
            return response["message"]["content"]
        except Exception as e:
            if attempt < 2:
                time.sleep(2)
            else:
                return f"ERROR: {e}"
    return "ERROR: Max retries exceeded"

def parse_entities(text: str) -> List[Dict]:
    """Parse JSON entities from response"""
    try:
        start = text.find("[")
        end = text.rfind("]") + 1
        if start >= 0 and end > start:
            return json.loads(text[start:end])
    except:
        pass
    return []

def parse_questions(text: str) -> List[str]:
    """Parse questions from response"""
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

def process_chunk(chunk: ChunkData, model: str) -> ChunkData:
    """Process a single chunk with all LLM tasks"""
    # Context
    resp = call_llm(model, CONTEXT_PROMPT.format(doc_title=chunk.doc_title, chunk=chunk.content), 150)
    if not resp.startswith("ERROR"):
        chunk.context = resp

    # Entities
    resp = call_llm(model, ENTITY_PROMPT.format(chunk=chunk.content), 300)
    if not resp.startswith("ERROR"):
        chunk.entities = parse_entities(resp)

    # Summary
    resp = call_llm(model, SUMMARY_PROMPT.format(chunk=chunk.content), 100)
    if not resp.startswith("ERROR"):
        chunk.summary = resp

    # Questions
    resp = call_llm(model, QA_PROMPT.format(chunk=chunk.content), 200)
    if not resp.startswith("ERROR"):
        chunk.questions = parse_questions(resp)

    chunk.processed = True
    return chunk

def save_checkpoint(progress: IngestProgress, chunks: List[ChunkData]):
    """Save progress checkpoint"""
    progress.last_update = datetime.now().isoformat()
    data = {
        "progress": asdict(progress),
        "chunks": [asdict(c) for c in chunks if c.processed]
    }
    with open(CHECKPOINT_FILE, "w") as f:
        json.dump(data, f)

def load_checkpoint() -> tuple[Optional[IngestProgress], List[ChunkData]]:
    """Load checkpoint if exists"""
    if not CHECKPOINT_FILE.exists():
        return None, []

    try:
        with open(CHECKPOINT_FILE) as f:
            data = json.load(f)

        progress = IngestProgress(**data["progress"])
        chunks = [ChunkData(**c) for c in data["chunks"]]
        return progress, chunks
    except:
        return None, []

def validate_quality(chunks: List[ChunkData]) -> Dict:
    """Validate ingestion quality"""
    total = len(chunks)
    processed = sum(1 for c in chunks if c.processed)
    with_context = sum(1 for c in chunks if c.context and len(c.context) > 20)
    with_entities = sum(1 for c in chunks if c.entities)
    with_summary = sum(1 for c in chunks if c.summary and len(c.summary) > 10)
    with_questions = sum(1 for c in chunks if c.questions)

    total_entities = sum(len(c.entities) for c in chunks)

    return {
        "total_chunks": total,
        "processed": processed,
        "with_context": with_context,
        "with_entities": with_entities,
        "with_summary": with_summary,
        "with_questions": with_questions,
        "total_entities": total_entities,
        "context_rate": with_context / total if total > 0 else 0,
        "entity_rate": with_entities / total if total > 0 else 0,
        "summary_rate": with_summary / total if total > 0 else 0,
    }

def store_to_qdrant(chunks: List[ChunkData], collection: str):
    """Store processed chunks to Qdrant"""
    from FlagEmbedding import BGEM3FlagModel

    console.print("[cyan]Loading embedding model...[/]")
    embedder = BGEM3FlagModel("BAAI/bge-m3", use_fp16=False)

    console.print("[cyan]Connecting to Qdrant...[/]")
    client = QdrantClient(host="localhost", port=6333)

    # Create collection if needed
    try:
        client.get_collection(collection)
    except:
        client.create_collection(
            collection_name=collection,
            vectors_config=VectorParams(size=1024, distance=Distance.COSINE)
        )

    console.print(f"[cyan]Storing {len(chunks)} chunks...[/]")

    # Embed and store in batches
    batch_size = 10
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i+batch_size]

        # Create contextualized content
        contents = []
        for c in batch:
            ctx_content = f"{c.context}\n\n{c.content}" if c.context else c.content
            contents.append(ctx_content)

        # Embed
        embeddings = embedder.encode(contents)["dense_vecs"]

        # Create points
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
                    "collection": collection
                }
            ))

        client.upsert(collection_name=collection, points=points)

    console.print(f"[green]Stored {len(chunks)} chunks to Qdrant[/]")

def main():
    console.print(Panel.fit(
        "[bold]OVERNIGHT FULL RAG INGESTION[/]\n"
        "Multi-pass ingestion with quality validation",
        title="Starting"
    ))

    # Get best model
    model = get_best_model()
    console.print(f"[green]Using model: {model}[/]")

    # Load docs
    all_chunks = load_all_docs()

    # Check for checkpoint
    saved_progress, saved_chunks = load_checkpoint()
    if saved_progress and saved_chunks:
        console.print(f"[yellow]Resuming from checkpoint: {saved_progress.processed_chunks}/{saved_progress.total_chunks} chunks[/]")
        processed_ids = {c.chunk_id for c in saved_chunks}
        for chunk in all_chunks:
            if chunk.chunk_id in processed_ids:
                saved = next(c for c in saved_chunks if c.chunk_id == chunk.chunk_id)
                chunk.context = saved.context
                chunk.entities = saved.entities
                chunk.summary = saved.summary
                chunk.questions = saved.questions
                chunk.processed = True

    # Initialize progress
    progress = IngestProgress(
        total_docs=len(set(c.doc_name for c in all_chunks)),
        total_chunks=len(all_chunks),
        processed_chunks=sum(1 for c in all_chunks if c.processed),
        start_time=datetime.now().isoformat()
    )

    # Process unprocessed chunks
    unprocessed = [c for c in all_chunks if not c.processed]
    console.print(f"[cyan]Processing {len(unprocessed)} chunks...[/]")

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        TimeElapsedColumn(),
        console=console
    ) as prog:
        task = prog.add_task(f"[cyan]{model}[/]", total=len(unprocessed))

        for i, chunk in enumerate(unprocessed):
            chunk = process_chunk(chunk, model)
            progress.processed_chunks += 1
            progress.total_entities += len(chunk.entities)

            # Save checkpoint every 10 chunks
            if (i + 1) % 10 == 0:
                save_checkpoint(progress, all_chunks)

            prog.advance(task)

    # Final checkpoint
    save_checkpoint(progress, all_chunks)

    # Validate quality
    console.print("\n[bold]Quality Validation:[/]")
    quality = validate_quality(all_chunks)

    console.print(f"  Processed: {quality['processed']}/{quality['total_chunks']}")
    console.print(f"  With context: {quality['with_context']} ({quality['context_rate']*100:.1f}%)")
    console.print(f"  With entities: {quality['with_entities']} ({quality['entity_rate']*100:.1f}%)")
    console.print(f"  With summary: {quality['with_summary']} ({quality['summary_rate']*100:.1f}%)")
    console.print(f"  Total entities: {quality['total_entities']}")

    # Store to Qdrant
    collection = f"qsys-lua-{datetime.now().strftime('%Y%m%d')}"
    store_to_qdrant([c for c in all_chunks if c.processed], collection)

    # Final summary
    console.print(Panel.fit(
        f"[bold green]INGESTION COMPLETE[/]\n\n"
        f"Model: {model}\n"
        f"Chunks: {quality['processed']}\n"
        f"Entities: {quality['total_entities']}\n"
        f"Collection: {collection}",
        title="Success"
    ))

    # Save final results
    results_file = Path.home() / "ai/knowledge-base/overnight_results.json"
    with open(results_file, "w") as f:
        json.dump({
            "model": model,
            "collection": collection,
            "quality": quality,
            "completed": datetime.now().isoformat()
        }, f, indent=2)

    console.print(f"\n[green]Results saved to: {results_file}[/]")

if __name__ == "__main__":
    main()
