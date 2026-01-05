#!/usr/bin/env python3
"""
Enhanced Knowledge Base Ingestion Pipeline v2

Optimized for local LLMs (qwen2.5:32b) with:
- Better prompt engineering
- Structured output handling
- Progress tracking
- Error recovery
"""
import hashlib
import json
import re
import time
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import typer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
from rich.table import Table
from rich.panel import Panel

from . import prompts

console = Console()
app = typer.Typer(help="Knowledge Base Ingestion CLI v2")


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class IngestConfig:
    """Configuration for ingestion pipeline."""
    collection: str = "default"
    llm_model: str = "qwen2.5:32b"
    chunk_size: int = 512
    chunk_overlap: int = 75  # ~15%

    # Feature flags
    enable_contextual: bool = True
    enable_entities: bool = True
    enable_summary: bool = True
    enable_qa: bool = True
    enable_keywords: bool = True

    # LLM settings
    temperature: float = 0.3
    max_tokens: int = 200

    # Processing
    batch_size: int = 10
    save_progress: bool = True


# =============================================================================
# Data Models
# =============================================================================

@dataclass
class ChunkMetadata:
    """Rich metadata for a chunk."""
    title: str = ""
    doc_type: str = ""
    keywords: List[str] = field(default_factory=list)
    summary: Optional[str] = None
    questions: List[str] = field(default_factory=list)
    entities: List[Dict[str, str]] = field(default_factory=list)
    context: Optional[str] = None
    token_count: int = 0


@dataclass
class ProcessedChunk:
    """A fully processed chunk ready for storage."""
    content: str
    contextualized_content: str
    chunk_index: int
    content_hash: str
    metadata: ChunkMetadata
    embedding: Optional[List[float]] = None


@dataclass
class IngestResult:
    """Result of ingesting a document."""
    document_id: str
    title: str
    status: str  # 'success', 'skipped', 'error'
    chunks_created: int = 0
    entities_found: int = 0
    processing_time: float = 0.0
    error: Optional[str] = None


# =============================================================================
# LLM Client
# =============================================================================

class OllamaClient:
    """Wrapper for Ollama with optimized settings for qwen2.5:32b."""

    def __init__(self, model: str = "qwen2.5:32b"):
        self.model = model
        self._client = None
        self._available = None

    @property
    def client(self):
        if self._client is None:
            try:
                import ollama
                self._client = ollama.Client()
                # Warm up
                self._client.list()
                self._available = True
            except Exception as e:
                console.print(f"[yellow]Ollama not available: {e}[/yellow]")
                self._available = False
        return self._client

    @property
    def available(self) -> bool:
        if self._available is None:
            _ = self.client
        return self._available

    def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.3,
        max_tokens: int = 200
    ) -> Optional[str]:
        """Send chat request to Ollama."""
        if not self.available:
            return None

        try:
            response = self.client.chat(
                model=self.model,
                messages=messages,
                options={
                    "temperature": temperature,
                    "num_predict": max_tokens,
                    "top_p": 0.9,
                    "repeat_penalty": 1.1
                }
            )
            return response["message"]["content"].strip()
        except Exception as e:
            console.print(f"[red]LLM error: {e}[/red]")
            return None


# =============================================================================
# Enrichment Functions
# =============================================================================

def extract_context(
    llm: OllamaClient,
    chunk: str,
    doc_title: str,
    doc_type: str,
    doc_preview: str,
    config: IngestConfig
) -> str:
    """Generate contextual retrieval context for a chunk."""
    if not config.enable_contextual or not llm.available:
        return f"From {doc_title} ({doc_type} documentation)."

    messages = prompts.format_context_prompt(
        doc_title=doc_title,
        doc_type=doc_type,
        doc_preview=doc_preview,
        chunk=chunk
    )

    result = llm.chat(messages, config.temperature, config.max_tokens)
    if result:
        # Clean up response
        result = result.strip()
        # Remove any "Context:" prefix the model might add
        if result.lower().startswith("context:"):
            result = result[8:].strip()
        return result

    return f"From {doc_title} ({doc_type} documentation)."


def extract_entities(
    llm: OllamaClient,
    chunk: str,
    config: IngestConfig
) -> List[Dict[str, str]]:
    """Extract entities from a chunk."""
    if not config.enable_entities:
        return []

    # Try LLM first
    if llm.available:
        messages = prompts.format_entity_prompt(chunk)
        result = llm.chat(messages, config.temperature, 300)

        if result:
            try:
                # Find JSON array in response
                match = re.search(r'\[.*\]', result, re.DOTALL)
                if match:
                    entities = json.loads(match.group())
                    # Validate structure
                    valid = []
                    for e in entities:
                        if isinstance(e, dict) and "name" in e and "type" in e:
                            valid.append({"name": str(e["name"]), "type": str(e["type"])})
                    return valid[:8]
            except json.JSONDecodeError:
                pass

    # Fallback: rule-based extraction
    return extract_entities_rules(chunk)


def extract_entities_rules(text: str) -> List[Dict[str, str]]:
    """Rule-based entity extraction as fallback."""
    entities = []

    # Q-SYS specific patterns
    qsys_products = re.findall(r'\b(Q-SYS\s+\w+|Core\s+\d+\w*|NV-\d+\w*)', text, re.IGNORECASE)
    for p in set(qsys_products):
        entities.append({"name": p, "type": "PRODUCT"})

    # Technical terms (CamelCase)
    camel = re.findall(r'\b([A-Z][a-z]+(?:[A-Z][a-z]+)+)\b', text)
    for c in set(camel):
        if len(c) > 4:
            entities.append({"name": c, "type": "CONCEPT"})

    # Protocols/standards
    protocols = re.findall(r'\b(Dante|AES67|AVB|CobraNet|SNMP|TCP|UDP|HTTP|HTTPS)\b', text, re.IGNORECASE)
    for p in set(protocols):
        entities.append({"name": p.upper(), "type": "TECHNOLOGY"})

    return entities[:8]


def extract_summary(
    llm: OllamaClient,
    chunk: str,
    config: IngestConfig
) -> Optional[str]:
    """Generate a one-sentence summary."""
    if not config.enable_summary or not llm.available:
        return None

    messages = prompts.format_summary_prompt(chunk)
    result = llm.chat(messages, config.temperature, 100)

    if result:
        # Clean up - take first sentence only
        sentences = result.split('.')
        if sentences:
            return sentences[0].strip() + "."

    return None


def extract_questions(
    llm: OllamaClient,
    chunk: str,
    config: IngestConfig
) -> List[str]:
    """Extract questions this chunk can answer."""
    if not config.enable_qa or not llm.available:
        return []

    messages = prompts.format_qa_prompt(chunk)
    result = llm.chat(messages, config.temperature, 200)

    if result:
        # Parse questions from response
        questions = []
        for line in result.split('\n'):
            line = line.strip()
            # Remove numbering
            line = re.sub(r'^[\d\.\-\)\s]+', '', line)
            if line and '?' in line:
                questions.append(line)
        return questions[:3]

    return []


def extract_keywords_llm(
    llm: OllamaClient,
    chunk: str,
    config: IngestConfig
) -> List[str]:
    """Extract keywords using LLM."""
    if not config.enable_keywords or not llm.available:
        return extract_keywords_rules(chunk)

    messages = prompts.format_keyword_prompt(chunk)
    result = llm.chat(messages, 0.2, 100)

    if result:
        # Parse comma-separated keywords
        keywords = [k.strip() for k in result.split(',')]
        return [k for k in keywords if k and len(k) > 2][:10]

    return extract_keywords_rules(chunk)


def extract_keywords_rules(text: str) -> List[str]:
    """Rule-based keyword extraction."""
    # Find capitalized terms
    words = re.findall(r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b', text)
    freq = {}
    for w in words:
        if len(w) > 3:
            freq[w] = freq.get(w, 0) + 1

    sorted_words = sorted(freq.items(), key=lambda x: x[1], reverse=True)
    return [w[0] for w in sorted_words[:10]]


# =============================================================================
# Main Pipeline
# =============================================================================

class IngestionPipeline:
    """Full ingestion pipeline with LLM enrichment."""

    def __init__(self, config: IngestConfig):
        self.config = config
        self.llm = OllamaClient(config.llm_model)
        self._kb = None
        self._embedding_service = None

        # Stats
        self.stats = {
            "files_processed": 0,
            "chunks_created": 0,
            "entities_found": 0,
            "errors": 0,
            "llm_calls": 0
        }

    @property
    def kb(self):
        """Lazy load knowledge base."""
        if self._kb is None:
            from .knowledge_base import KnowledgeBase
            self._kb = KnowledgeBase()
            self._kb.initialize(recreate=False)
        return self._kb

    @property
    def embedding_service(self):
        """Lazy load embedding service."""
        if self._embedding_service is None:
            from .embeddings import get_embedding_service
            self._embedding_service = get_embedding_service()
        return self._embedding_service

    def chunk_text(self, text: str) -> List[Tuple[str, int]]:
        """Split text into overlapping chunks."""
        words = text.split()

        if len(words) <= self.config.chunk_size:
            return [(text, 0)]

        chunks = []
        start = 0
        idx = 0

        while start < len(words):
            end = start + self.config.chunk_size
            chunk = " ".join(words[start:end])
            chunks.append((chunk, idx))
            start = end - self.config.chunk_overlap
            idx += 1

        return chunks

    def process_chunk(
        self,
        chunk_text: str,
        chunk_index: int,
        doc_title: str,
        doc_type: str,
        doc_preview: str
    ) -> ProcessedChunk:
        """Process a single chunk with all enrichments."""

        # 1. Generate context
        context = extract_context(
            self.llm, chunk_text, doc_title, doc_type, doc_preview, self.config
        )
        self.stats["llm_calls"] += 1

        # 2. Extract entities
        entities = extract_entities(self.llm, chunk_text, self.config)
        if self.config.enable_entities and self.llm.available:
            self.stats["llm_calls"] += 1

        # 3. Generate summary
        summary = extract_summary(self.llm, chunk_text, self.config)
        if self.config.enable_summary and self.llm.available:
            self.stats["llm_calls"] += 1

        # 4. Extract questions
        questions = extract_questions(self.llm, chunk_text, self.config)
        if self.config.enable_qa and self.llm.available:
            self.stats["llm_calls"] += 1

        # 5. Extract keywords
        keywords = extract_keywords_llm(self.llm, chunk_text, self.config)
        if self.config.enable_keywords and self.llm.available:
            self.stats["llm_calls"] += 1

        # Build contextualized content
        contextualized = f"<context>\n{context}\n</context>\n\n{chunk_text}"

        # Create metadata
        metadata = ChunkMetadata(
            title=doc_title,
            doc_type=doc_type,
            keywords=keywords,
            summary=summary,
            questions=questions,
            entities=entities,
            context=context,
            token_count=len(chunk_text.split())
        )

        return ProcessedChunk(
            content=chunk_text,
            contextualized_content=contextualized,
            chunk_index=chunk_index,
            content_hash=hashlib.sha256(chunk_text.encode()).hexdigest()[:16],
            metadata=metadata
        )

    def ingest_file(self, file_path: Path) -> IngestResult:
        """Ingest a single file with full enrichment."""
        start_time = time.time()

        try:
            # Read file
            if file_path.suffix.lower() in ['.md', '.txt']:
                content = file_path.read_text(encoding='utf-8')
                doc_type = "markdown"
            else:
                from .document_processor import DocumentProcessor
                processor = DocumentProcessor()
                doc = processor.process_file(file_path)
                content = doc.content
                doc_type = doc.doc_type

            doc_title = file_path.stem
            doc_preview = content[:2000]

            # Chunk the document
            raw_chunks = self.chunk_text(content)

            # Process each chunk
            processed_chunks = []
            all_entities = []

            for chunk_text, chunk_idx in raw_chunks:
                chunk = self.process_chunk(
                    chunk_text, chunk_idx, doc_title, doc_type, doc_preview
                )
                processed_chunks.append(chunk)
                all_entities.extend(chunk.metadata.entities)

            # Generate embeddings for contextualized content
            texts = [c.contextualized_content for c in processed_chunks]
            embeddings = self.embedding_service.encode_documents(texts)

            for chunk, emb in zip(processed_chunks, embeddings):
                chunk.embedding = emb.tolist()

            # Store in vector database
            import uuid
            document_id = str(uuid.uuid4())
            self._store_chunks(processed_chunks, document_id, doc_title, doc_type)

            # Store entities in graph
            self._store_entities(all_entities, document_id)

            self.stats["files_processed"] += 1
            self.stats["chunks_created"] += len(processed_chunks)
            # Safely count unique entity names
            entity_name_set = set()
            for e in all_entities:
                if isinstance(e, dict) and "name" in e:
                    entity_name_set.add(str(e["name"]))
                elif isinstance(e, str):
                    entity_name_set.add(e)
            self.stats["entities_found"] += len(entity_name_set)

            return IngestResult(
                document_id=document_id,
                title=doc_title,
                status="success",
                chunks_created=len(processed_chunks),
                entities_found=len(all_entities),
                processing_time=time.time() - start_time
            )

        except Exception as e:
            self.stats["errors"] += 1
            return IngestResult(
                document_id="",
                title=file_path.stem,
                status="error",
                error=str(e),
                processing_time=time.time() - start_time
            )

    def _store_chunks(
        self,
        chunks: List[ProcessedChunk],
        document_id: str,
        title: str,
        doc_type: str
    ):
        """Store processed chunks in Qdrant."""
        from qdrant_client.http.models import PointStruct
        import uuid

        points = []
        for chunk in chunks:
            # Safely extract entity names
            entity_names = []
            for e in chunk.metadata.entities:
                if isinstance(e, dict) and "name" in e:
                    entity_names.append(str(e["name"]))
                elif isinstance(e, str):
                    entity_names.append(e)

            points.append(PointStruct(
                id=str(uuid.uuid4()),
                vector=chunk.embedding,
                payload={
                    "document_id": document_id,
                    "chunk_index": chunk.chunk_index,
                    "content": chunk.content,
                    "contextualized_content": chunk.contextualized_content,
                    "content_hash": chunk.content_hash,
                    "title": title,
                    "doc_type": doc_type,
                    "collection": self.config.collection,
                    "context": chunk.metadata.context,
                    "keywords": chunk.metadata.keywords or [],
                    "summary": chunk.metadata.summary,
                    "questions": chunk.metadata.questions or [],
                    "entities": entity_names
                }
            ))

        # Upsert
        self.kb.vector_store.client.upsert(
            collection_name=self.kb.collection_name,
            points=points
        )

    def _store_entities(self, entities: List[Dict[str, str]], document_id: str):
        """Store entities in Neo4j."""
        seen = set()
        for entity in entities:
            # Safely extract entity info
            if not isinstance(entity, dict):
                continue
            name = entity.get("name")
            entity_type = entity.get("type", "UNKNOWN")
            if not name:
                continue
            name = str(name)
            if name.lower() not in seen:
                seen.add(name.lower())
                try:
                    self.kb.add_entity(
                        name=name,
                        entity_type=str(entity_type)
                    )
                except Exception:
                    pass  # Ignore duplicate errors

    def ingest_directory(
        self,
        directory: Path,
        extensions: List[str] = None
    ) -> List[IngestResult]:
        """Ingest all files in a directory with progress display."""
        extensions = extensions or [".md"]
        extensions = [e if e.startswith(".") else f".{e}" for e in extensions]

        files = []
        for ext in extensions:
            files.extend(sorted(directory.rglob(f"*{ext}")))

        console.print(Panel(
            f"[bold]Ingesting {len(files)} files[/bold]\n"
            f"Collection: {self.config.collection}\n"
            f"Model: {self.config.llm_model}\n"
            f"Features: context={self.config.enable_contextual}, "
            f"entities={self.config.enable_entities}, "
            f"summary={self.config.enable_summary}",
            title="Ingestion Pipeline v2"
        ))

        results = []

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("{task.completed}/{task.total}"),
            TimeElapsedColumn(),
            console=console
        ) as progress:
            task = progress.add_task("Processing...", total=len(files))

            for file_path in files:
                progress.update(task, description=f"[cyan]{file_path.name}[/cyan]")

                result = self.ingest_file(file_path)
                results.append(result)

                if result.status == "success":
                    progress.console.print(
                        f"  [green]✓[/green] {result.title}: "
                        f"{result.chunks_created} chunks, "
                        f"{result.entities_found} entities, "
                        f"{result.processing_time:.1f}s"
                    )
                else:
                    progress.console.print(
                        f"  [red]✗[/red] {result.title}: {result.error}"
                    )

                progress.advance(task)

        # Print summary
        self._print_summary(results)

        return results

    def _print_summary(self, results: List[IngestResult]):
        """Print final summary."""
        table = Table(title="Ingestion Summary")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")

        success = sum(1 for r in results if r.status == "success")
        errors = sum(1 for r in results if r.status == "error")
        total_time = sum(r.processing_time for r in results)

        table.add_row("Files Processed", str(success))
        table.add_row("Errors", str(errors))
        table.add_row("Total Chunks", str(self.stats["chunks_created"]))
        table.add_row("Unique Entities", str(self.stats["entities_found"]))
        table.add_row("LLM Calls", str(self.stats["llm_calls"]))
        table.add_row("Total Time", f"{total_time:.1f}s")
        table.add_row("Avg Time/File", f"{total_time/len(results):.1f}s" if results else "N/A")

        console.print(table)


# =============================================================================
# CLI
# =============================================================================

@app.command()
def ingest(
    path: str = typer.Argument(..., help="File or directory to ingest"),
    collection: str = typer.Option("qsys-full", "-c", "--collection"),
    model: str = typer.Option("qwen2.5:32b", "-m", "--model"),
    no_context: bool = typer.Option(False, "--no-context"),
    no_entities: bool = typer.Option(False, "--no-entities"),
    no_summary: bool = typer.Option(False, "--no-summary"),
    no_qa: bool = typer.Option(False, "--no-qa"),
    extensions: str = typer.Option("md", "-e", "--ext")
):
    """Ingest documents with full LLM enrichment."""
    config = IngestConfig(
        collection=collection,
        llm_model=model,
        enable_contextual=not no_context,
        enable_entities=not no_entities,
        enable_summary=not no_summary,
        enable_qa=not no_qa
    )

    pipeline = IngestionPipeline(config)

    path = Path(path)

    if path.is_file():
        result = pipeline.ingest_file(path)
        if result.status == "success":
            console.print(f"[green]Success: {result.chunks_created} chunks created[/green]")
        else:
            console.print(f"[red]Error: {result.error}[/red]")
    else:
        ext_list = [e.strip() for e in extensions.split(",")]
        pipeline.ingest_directory(path, ext_list)


@app.command()
def test_llm(
    model: str = typer.Option("qwen2.5:32b", "-m", "--model")
):
    """Test LLM connectivity and response quality."""
    llm = OllamaClient(model)

    if not llm.available:
        console.print("[red]LLM not available[/red]")
        return

    console.print(f"[green]Connected to {model}[/green]")

    # Test context generation
    test_chunk = "The Matrix Mixer supports up to 64 inputs and 64 outputs."
    messages = prompts.format_context_prompt(
        doc_title="Matrix_Mixer",
        doc_type="Q-SYS Designer",
        doc_preview="Q-SYS Designer documentation about audio components...",
        chunk=test_chunk
    )

    console.print("\n[bold]Testing context generation:[/bold]")
    start = time.time()
    result = llm.chat(messages)
    console.print(f"Time: {time.time()-start:.1f}s")
    console.print(f"Result: {result}")


if __name__ == "__main__":
    app()
