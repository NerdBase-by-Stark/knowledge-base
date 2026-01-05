#!/usr/bin/env python3
"""Run Qwen3 benchmark separately"""
import json
import time
import hashlib
from dataclasses import dataclass, field, asdict
from typing import List, Dict
from pathlib import Path

import ollama
from rich.console import Console
from rich.progress import Progress, BarColumn, TextColumn, TimeElapsedColumn

console = Console()

MODEL = "Qwen3:latest"
TEST_DOCS = ["3_-_The_Language.md", "HttpClient.md", "TcpSocket.md", "Uci.md", "State_Trigger.md"]
DOCS_DIR = Path.home() / "ai/knowledge-base/scrape-jobs/qsys-lua/markdown"
OUTPUT_DIR = Path.home() / "ai/knowledge-base/scrape-jobs/qsys-lua/shootout"
CHUNK_SIZE = 512
CHUNK_OVERLAP = 75

# Same prompts as main benchmark
# NOTE: Adding /no_think to prompts to disable Qwen3's thinking mode
CONTEXT_PROMPT = """/no_think
You are a technical documentation assistant. Write a brief context statement (2-3 sentences) that situates this chunk within its source document.

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
class ChunkResult:
    chunk_id: str
    chunk_text: str
    context: str = ""
    entities: List[Dict] = field(default_factory=list)
    summary: str = ""
    questions: List[str] = field(default_factory=list)
    context_time_ms: float = 0
    entity_time_ms: float = 0
    summary_time_ms: float = 0
    qa_time_ms: float = 0
    errors: List[str] = field(default_factory=list)

def simple_chunk(text: str, chunk_size: int = 512, overlap: int = 75) -> List[str]:
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        if chunk.strip():
            chunks.append(chunk)
        start = end - overlap
    return chunks

def call_llm(prompt: str, max_tokens: int = 300) -> tuple[str, float]:
    start = time.time()
    try:
        response = ollama.chat(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": 0.3, "num_predict": max_tokens}
        )
        result = response["message"]["content"]
    except Exception as e:
        result = f"ERROR: {e}"
    return result, (time.time() - start) * 1000

def parse_entities(text: str) -> List[Dict]:
    try:
        start = text.find("[")
        end = text.rfind("]") + 1
        if start >= 0 and end > start:
            return json.loads(text[start:end])
    except:
        pass
    return []

def parse_questions(text: str) -> List[str]:
    lines = text.strip().split("\n")
    questions = []
    for line in lines:
        line = line.strip()
        if line and not line.startswith("#"):
            if line[0].isdigit() and len(line) > 1 and line[1] in ".)":
                line = line[2:].strip()
            if line:
                questions.append(line)
    return questions[:3]

def main():
    console.print(f"[bold cyan]Running benchmark for {MODEL}[/]")

    # Load docs
    docs = {}
    for doc_name in TEST_DOCS:
        path = DOCS_DIR / doc_name
        if path.exists():
            text = path.read_text()
            chunks = simple_chunk(text, CHUNK_SIZE, CHUNK_OVERLAP)
            docs[doc_name] = {"title": doc_name.replace(".md", "").replace("_", " "), "chunks": chunks}
            console.print(f"  {doc_name}: {len(chunks)} chunks")

    all_chunks = []
    for doc_name, doc_data in docs.items():
        for i, chunk in enumerate(doc_data["chunks"]):
            chunk_id = hashlib.md5(f"{doc_name}:{i}".encode()).hexdigest()[:8]
            all_chunks.append({"id": chunk_id, "title": doc_data["title"], "text": chunk})

    console.print(f"[green]Total: {len(all_chunks)} chunks, {len(all_chunks)*4} LLM calls[/]")

    results = []
    total_entities = 0
    error_count = 0
    start_time = time.time()

    with Progress(TextColumn(f"[cyan]{MODEL}[/]"), BarColumn(), TextColumn("{task.completed}/{task.total}"), TimeElapsedColumn()) as progress:
        task = progress.add_task("Processing", total=len(all_chunks))

        for chunk_data in all_chunks:
            cr = ChunkResult(chunk_id=chunk_data["id"], chunk_text=chunk_data["text"][:200])

            # Context
            resp, lat = call_llm(CONTEXT_PROMPT.format(doc_title=chunk_data["title"], chunk=chunk_data["text"]), 150)
            cr.context, cr.context_time_ms = resp, lat
            if "ERROR" in resp: cr.errors.append("context")

            # Entities
            resp, lat = call_llm(ENTITY_PROMPT.format(chunk=chunk_data["text"]), 300)
            cr.entities, cr.entity_time_ms = parse_entities(resp), lat
            total_entities += len(cr.entities)
            if "ERROR" in resp: cr.errors.append("entity")

            # Summary
            resp, lat = call_llm(SUMMARY_PROMPT.format(chunk=chunk_data["text"]), 100)
            cr.summary, cr.summary_time_ms = resp, lat
            if "ERROR" in resp: cr.errors.append("summary")

            # QA
            resp, lat = call_llm(QA_PROMPT.format(chunk=chunk_data["text"]), 200)
            cr.questions, cr.qa_time_ms = parse_questions(resp), lat
            if "ERROR" in resp: cr.errors.append("qa")

            results.append(cr)
            if cr.errors: error_count += 1
            progress.advance(task)

    total_time = time.time() - start_time
    avg_time = (total_time * 1000) / len(all_chunks)

    console.print(f"\n[bold green]Completed![/]")
    console.print(f"  Time: {total_time:.1f}s")
    console.print(f"  Avg/chunk: {avg_time:.0f}ms")
    console.print(f"  Entities: {total_entities}")
    console.print(f"  Errors: {error_count}")

    # Save results
    OUTPUT_DIR.mkdir(exist_ok=True)
    output_file = OUTPUT_DIR / "Qwen3-latest.json"
    data = {
        "model": MODEL,
        "total_chunks": len(all_chunks),
        "total_time_s": total_time,
        "avg_chunk_time_ms": avg_time,
        "total_entities": total_entities,
        "error_count": error_count,
        "chunks": [asdict(c) for c in results]
    }
    with open(output_file, "w") as f:
        json.dump(data, f, indent=2)
    console.print(f"[green]Saved: {output_file}[/]")

if __name__ == "__main__":
    main()
