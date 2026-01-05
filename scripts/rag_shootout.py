#!/usr/bin/env python3
"""
RAG Creation Shootout: Fair comparison of LLMs for knowledge base ingestion

Tests each model on identical:
- 5 dense documents (~196 chunks)
- Same prompts (context, entities, summary, QA)
- Same chunking parameters

Metrics:
- Speed (time per chunk, total time)
- Output quality (structure, completeness)
- Entity extraction count
- Error rate
"""
import json
import time
import hashlib
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import ollama
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, BarColumn, TextColumn, TimeElapsedColumn

console = Console()

# =============================================================================
# CONFIG
# =============================================================================

MODELS = [
    "qwen2.5:7b",
    "mistral:latest",
    "llama3.1:8b",
    "qwen2.5:32b",
    "deepseek-coder:33b",
]

TEST_DOCS = [
    "3_-_The_Language.md",
    "HttpClient.md",
    "TcpSocket.md",
    "Uci.md",
    "State_Trigger.md",
]

DOCS_DIR = Path.home() / "ai/knowledge-base/scrape-jobs/qsys-lua/markdown"
OUTPUT_DIR = Path.home() / "ai/knowledge-base/scrape-jobs/qsys-lua/shootout"

CHUNK_SIZE = 512  # tokens approx
CHUNK_OVERLAP = 75

# =============================================================================
# PROMPTS (identical for all models)
# =============================================================================

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

# =============================================================================
# DATA CLASSES
# =============================================================================

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

@dataclass
class ModelResult:
    model: str
    total_chunks: int = 0
    total_time_s: float = 0
    avg_chunk_time_ms: float = 0
    total_entities: int = 0
    error_count: int = 0
    chunks: List[ChunkResult] = field(default_factory=list)

# =============================================================================
# CHUNKING
# =============================================================================

def simple_chunk(text: str, chunk_size: int = 512, overlap: int = 75) -> List[str]:
    """Simple word-based chunking"""
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

def load_docs() -> Dict[str, Dict]:
    """Load test documents and chunk them"""
    docs = {}

    for doc_name in TEST_DOCS:
        path = DOCS_DIR / doc_name
        if not path.exists():
            console.print(f"[red]Missing: {doc_name}[/]")
            continue

        text = path.read_text()
        chunks = simple_chunk(text, CHUNK_SIZE, CHUNK_OVERLAP)

        docs[doc_name] = {
            "title": doc_name.replace(".md", "").replace("_", " "),
            "text": text,
            "chunks": chunks
        }
        console.print(f"  {doc_name}: {len(chunks)} chunks")

    return docs

# =============================================================================
# LLM CALLS
# =============================================================================

def call_llm(model: str, prompt: str, max_tokens: int = 300) -> tuple[str, float]:
    """Call Ollama model, return (response, latency_ms)"""
    start = time.time()
    try:
        response = ollama.chat(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": 0.3, "num_predict": max_tokens}
        )
        result = response["message"]["content"]
    except Exception as e:
        result = f"ERROR: {e}"

    latency = (time.time() - start) * 1000
    return result, latency

def parse_entities(text: str) -> List[Dict]:
    """Try to parse JSON entities from response"""
    try:
        # Find JSON array in response
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
            # Remove numbering
            if line[0].isdigit() and (line[1] == "." or line[1] == ")"):
                line = line[2:].strip()
            if line:
                questions.append(line)
    return questions[:3]

# =============================================================================
# BENCHMARK
# =============================================================================

def benchmark_model(model: str, docs: Dict[str, Dict]) -> ModelResult:
    """Run full benchmark on one model"""
    result = ModelResult(model=model)
    all_chunks = []

    # Collect all chunks with doc context
    for doc_name, doc_data in docs.items():
        for i, chunk in enumerate(doc_data["chunks"]):
            chunk_id = hashlib.md5(f"{doc_name}:{i}".encode()).hexdigest()[:8]
            all_chunks.append({
                "id": chunk_id,
                "doc": doc_name,
                "title": doc_data["title"],
                "text": chunk,
                "index": i
            })

    result.total_chunks = len(all_chunks)
    start_time = time.time()

    with Progress(
        TextColumn(f"[cyan]{model}[/]"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        TimeElapsedColumn(),
        console=console
    ) as progress:
        task = progress.add_task("Processing", total=len(all_chunks))

        for chunk_data in all_chunks:
            chunk_result = ChunkResult(
                chunk_id=chunk_data["id"],
                chunk_text=chunk_data["text"][:200] + "..."
            )

            # 1. Context generation
            prompt = CONTEXT_PROMPT.format(
                doc_title=chunk_data["title"],
                chunk=chunk_data["text"]
            )
            resp, latency = call_llm(model, prompt, 150)
            chunk_result.context = resp
            chunk_result.context_time_ms = latency
            if "ERROR" in resp:
                chunk_result.errors.append("context")

            # 2. Entity extraction
            prompt = ENTITY_PROMPT.format(chunk=chunk_data["text"])
            resp, latency = call_llm(model, prompt, 300)
            chunk_result.entities = parse_entities(resp)
            chunk_result.entity_time_ms = latency
            result.total_entities += len(chunk_result.entities)
            if not chunk_result.entities and "ERROR" not in resp:
                # Didn't parse but not an error - still count
                pass
            if "ERROR" in resp:
                chunk_result.errors.append("entity")

            # 3. Summary
            prompt = SUMMARY_PROMPT.format(chunk=chunk_data["text"])
            resp, latency = call_llm(model, prompt, 100)
            chunk_result.summary = resp
            chunk_result.summary_time_ms = latency
            if "ERROR" in resp:
                chunk_result.errors.append("summary")

            # 4. QA extraction
            prompt = QA_PROMPT.format(chunk=chunk_data["text"])
            resp, latency = call_llm(model, prompt, 200)
            chunk_result.questions = parse_questions(resp)
            chunk_result.qa_time_ms = latency
            if "ERROR" in resp:
                chunk_result.errors.append("qa")

            result.chunks.append(chunk_result)
            if chunk_result.errors:
                result.error_count += 1

            progress.advance(task)

    result.total_time_s = time.time() - start_time
    result.avg_chunk_time_ms = (result.total_time_s * 1000) / result.total_chunks

    return result

def save_results(results: List[ModelResult]):
    """Save all results to JSON"""
    OUTPUT_DIR.mkdir(exist_ok=True)

    for r in results:
        output_file = OUTPUT_DIR / f"{r.model.replace(':', '-')}.json"
        data = {
            "model": r.model,
            "total_chunks": r.total_chunks,
            "total_time_s": r.total_time_s,
            "avg_chunk_time_ms": r.avg_chunk_time_ms,
            "total_entities": r.total_entities,
            "error_count": r.error_count,
            "chunks": [asdict(c) for c in r.chunks]
        }
        with open(output_file, "w") as f:
            json.dump(data, f, indent=2)
        console.print(f"[green]Saved:[/] {output_file}")

def save_for_claude(docs: Dict[str, Dict]):
    """Save prompts for Claude comparison"""
    OUTPUT_DIR.mkdir(exist_ok=True)
    output_file = OUTPUT_DIR / "claude_prompts.json"

    prompts = []
    for doc_name, doc_data in docs.items():
        for i, chunk in enumerate(doc_data["chunks"][:3]):  # First 3 chunks per doc
            prompts.append({
                "doc": doc_name,
                "chunk_index": i,
                "chunk": chunk,
                "context_prompt": CONTEXT_PROMPT.format(
                    doc_title=doc_data["title"],
                    chunk=chunk
                ),
                "entity_prompt": ENTITY_PROMPT.format(chunk=chunk),
                "summary_prompt": SUMMARY_PROMPT.format(chunk=chunk),
                "qa_prompt": QA_PROMPT.format(chunk=chunk)
            })

    with open(output_file, "w") as f:
        json.dump(prompts, f, indent=2)

    console.print(f"\n[yellow]Claude prompts saved:[/] {output_file}")
    console.print(f"[dim]Contains {len(prompts)} sample chunks for Claude comparison[/]")

def analyze_quality(results: List[ModelResult]) -> Dict:
    """Analyze what each model missed"""
    analysis = {}

    for r in results:
        model_analysis = {
            "empty_contexts": 0,
            "empty_entities": 0,
            "empty_summaries": 0,
            "empty_questions": 0,
            "json_parse_fails": 0,
            "avg_entity_count": 0,
            "avg_question_count": 0,
            "avg_context_length": 0,
            "avg_summary_length": 0,
        }

        entity_counts = []
        question_counts = []
        context_lengths = []
        summary_lengths = []

        for c in r.chunks:
            if not c.context or len(c.context) < 20:
                model_analysis["empty_contexts"] += 1
            else:
                context_lengths.append(len(c.context))

            if not c.entities:
                model_analysis["empty_entities"] += 1
            else:
                entity_counts.append(len(c.entities))

            if not c.summary or len(c.summary) < 10:
                model_analysis["empty_summaries"] += 1
            else:
                summary_lengths.append(len(c.summary))

            if not c.questions:
                model_analysis["empty_questions"] += 1
            else:
                question_counts.append(len(c.questions))

        if entity_counts:
            model_analysis["avg_entity_count"] = sum(entity_counts) / len(entity_counts)
        if question_counts:
            model_analysis["avg_question_count"] = sum(question_counts) / len(question_counts)
        if context_lengths:
            model_analysis["avg_context_length"] = sum(context_lengths) / len(context_lengths)
        if summary_lengths:
            model_analysis["avg_summary_length"] = sum(summary_lengths) / len(summary_lengths)

        analysis[r.model] = model_analysis

    return analysis

def save_checkpoint(results: List[ModelResult], docs: Dict):
    """Save checkpoint file for recovery"""
    OUTPUT_DIR.mkdir(exist_ok=True)
    checkpoint_file = OUTPUT_DIR / "CHECKPOINT.md"

    lines = [
        "# RAG Shootout Checkpoint",
        f"\n**Last Updated**: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        f"\n## Test Configuration",
        f"- Documents: {', '.join(TEST_DOCS)}",
        f"- Total chunks: {sum(len(d['chunks']) for d in docs.values())}",
        f"- Models tested: {len(results)}",
        "\n## Completed Models\n"
    ]

    for r in results:
        lines.append(f"### {r.model}")
        lines.append(f"- Time: {r.total_time_s:.1f}s")
        lines.append(f"- Avg/chunk: {r.avg_chunk_time_ms:.0f}ms")
        lines.append(f"- Entities found: {r.total_entities}")
        lines.append(f"- Errors: {r.error_count}")
        lines.append("")

    lines.append("\n## To Resume\n")
    lines.append("```bash")
    lines.append("cd ~/ai/knowledge-base")
    lines.append("python scripts/rag_shootout.py")
    lines.append("```")

    with open(checkpoint_file, "w") as f:
        f.write("\n".join(lines))

def print_summary(results: List[ModelResult]):
    """Print comparison table"""
    console.print("\n")

    # Main results table
    table = Table(title="RAG CREATION SHOOTOUT RESULTS")
    table.add_column("Model", style="cyan")
    table.add_column("Time", justify="right")
    table.add_column("Avg/Chunk", justify="right")
    table.add_column("Entities", justify="right")
    table.add_column("Errors", justify="right")
    table.add_column("Speed", justify="right")
    table.add_column("Quality", justify="right")
    table.add_column("TOTAL", justify="right", style="bold")

    # Calculate scores
    max_entities = max(r.total_entities for r in results) or 1
    min_time = min(r.avg_chunk_time_ms for r in results) or 1

    scored_results = []
    for r in results:
        speed_score = (min_time / r.avg_chunk_time_ms) * 40
        entity_score = (r.total_entities / max_entities) * 40
        reliability_score = ((r.total_chunks - r.error_count) / r.total_chunks) * 20
        total_score = speed_score + entity_score + reliability_score
        scored_results.append((r, speed_score, entity_score + reliability_score, total_score))

    # Sort by total score
    scored_results.sort(key=lambda x: x[3], reverse=True)

    for r, speed, quality, total in scored_results:
        table.add_row(
            r.model,
            f"{r.total_time_s:.1f}s",
            f"{r.avg_chunk_time_ms:.0f}ms",
            str(r.total_entities),
            str(r.error_count),
            f"{speed:.1f}",
            f"{quality:.1f}",
            f"{total:.1f}"
        )

    console.print(table)

    # Quality analysis table
    console.print("\n")
    analysis = analyze_quality(results)

    qual_table = Table(title="QUALITY ANALYSIS (What Each Model Missed)")
    qual_table.add_column("Model", style="cyan")
    qual_table.add_column("Empty Context", justify="right")
    qual_table.add_column("No Entities", justify="right")
    qual_table.add_column("Empty Summary", justify="right")
    qual_table.add_column("No Questions", justify="right")
    qual_table.add_column("Avg Entities", justify="right")

    for model, a in analysis.items():
        qual_table.add_row(
            model,
            str(a["empty_contexts"]),
            str(a["empty_entities"]),
            str(a["empty_summaries"]),
            str(a["empty_questions"]),
            f"{a['avg_entity_count']:.1f}"
        )

    console.print(qual_table)

    # Winner
    winner = scored_results[0][0]
    console.print(f"\n[bold green]🏆 WINNER: {winner.model} (Score: {scored_results[0][3]:.1f})[/]")

    # Save analysis
    analysis_file = OUTPUT_DIR / "quality_analysis.json"
    with open(analysis_file, "w") as f:
        json.dump(analysis, f, indent=2)
    console.print(f"[dim]Quality analysis saved to: {analysis_file}[/]")

def main():
    console.print(Panel.fit(
        "[bold]RAG CREATION SHOOTOUT[/]\n"
        f"Models: {len(MODELS)}\n"
        f"Docs: {len(TEST_DOCS)}\n"
        "Tasks: Context, Entities, Summary, QA",
        title="Benchmark"
    ))

    # Load docs
    console.print("\n[bold]Loading documents...[/]")
    docs = load_docs()

    total_chunks = sum(len(d["chunks"]) for d in docs.values())
    console.print(f"[green]Total chunks to process: {total_chunks}[/]")
    console.print(f"[dim]Each model will make {total_chunks * 4} LLM calls[/]")

    # Save Claude prompts first
    save_for_claude(docs)

    # Run benchmarks
    results = []
    for model in MODELS:
        console.print(f"\n[bold yellow]Testing: {model}[/]")
        result = benchmark_model(model, docs)
        results.append(result)
        console.print(f"  Completed in {result.total_time_s:.1f}s, {result.total_entities} entities, {result.error_count} errors")

        # Save checkpoint after each model (in case of interruption)
        save_checkpoint(results, docs)
        save_results(results)
        console.print(f"  [dim]Checkpoint saved[/]")

    # Final summary
    print_summary(results)

    console.print("\n[dim]Run Claude comparison separately using claude_prompts.json[/]")

if __name__ == "__main__":
    main()
