#!/usr/bin/env python3
"""
Enhanced Knowledge Base Ingestion Pipeline

Features:
- Contextual Retrieval (Anthropic's approach)
- Smart chunking with semantic awareness
- Metadata extraction (title, keywords, summary, QA)
- Entity extraction for knowledge graph
- Quality validation and deduplication
- Parallel processing support
"""
import hashlib
import json
import re
from pathlib import Path
from typing import List, Dict, Any, Optional, Iterator, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import typer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskID
from rich.table import Table

console = Console()
app = typer.Typer(help="Knowledge Base Ingestion CLI")


# ============================================================================
# Data Models
# ============================================================================

@dataclass
class ChunkMetadata:
    """Rich metadata for a chunk."""
    title: Optional[str] = None
    section: Optional[str] = None
    keywords: List[str] = field(default_factory=list)
    summary: Optional[str] = None
    questions_answered: List[str] = field(default_factory=list)
    entities: List[Dict[str, str]] = field(default_factory=list)
    context_prefix: Optional[str] = None
    token_count: int = 0
    quality_score: float = 1.0


@dataclass
class EnhancedChunk:
    """A chunk with full metadata and context."""
    content: str
    contextualized_content: str  # With context prefix
    chunk_index: int
    document_id: str
    metadata: ChunkMetadata
    embedding: Optional[List[float]] = None
    content_hash: str = ""

    def __post_init__(self):
        if not self.content_hash:
            self.content_hash = hashlib.sha256(self.content.encode()).hexdigest()[:16]


@dataclass
class IngestResult:
    """Result of ingesting a document."""
    document_id: str
    title: str
    chunk_count: int
    entity_count: int
    status: str  # 'success', 'duplicate', 'error'
    error_message: Optional[str] = None
    processing_time: float = 0.0


# ============================================================================
# Contextual Retrieval
# ============================================================================

class ContextualRetrieval:
    """
    Implements Anthropic's Contextual Retrieval approach.
    Prepends document context to each chunk before embedding.

    Reference: https://www.anthropic.com/news/contextual-retrieval
    """

    CONTEXT_PROMPT = """<document>
{document_content}
</document>

Here is the chunk we want to situate within the document:
<chunk>
{chunk_content}
</chunk>

Please provide a short, concise context (2-3 sentences) that situates this chunk within the overall document.
The context should help a reader understand what this chunk is about without reading the full document.
Focus on: what document this is from, what section/topic it covers, and key entities mentioned.
Respond with ONLY the context, no preamble."""

    def __init__(self, llm_model: str = "llama3.1:8b", use_ollama: bool = True):
        self.llm_model = llm_model
        self.use_ollama = use_ollama
        self._client = None

    @property
    def client(self):
        """Lazy load Ollama client."""
        if self._client is None and self.use_ollama:
            try:
                import ollama
                self._client = ollama.Client()
            except ImportError:
                console.print("[yellow]Ollama not available, skipping contextual retrieval[/yellow]")
        return self._client

    def generate_context(
        self,
        chunk_content: str,
        document_content: str,
        document_title: str
    ) -> str:
        """Generate context for a chunk using LLM."""
        if not self.client:
            # Fallback: simple context based on document title
            return f"This chunk is from the document '{document_title}'."

        # Truncate document if too long (keep first 4000 chars for context)
        doc_preview = document_content[:4000] if len(document_content) > 4000 else document_content

        prompt = self.CONTEXT_PROMPT.format(
            document_content=doc_preview,
            chunk_content=chunk_content
        )

        try:
            response = self.client.chat(
                model=self.llm_model,
                messages=[{"role": "user", "content": prompt}],
                options={"temperature": 0.3, "num_predict": 150}
            )
            return response["message"]["content"].strip()
        except Exception as e:
            console.print(f"[yellow]Context generation failed: {e}[/yellow]")
            return f"This chunk is from the document '{document_title}'."

    def contextualize_chunk(
        self,
        chunk_content: str,
        context: str
    ) -> str:
        """Prepend context to chunk content."""
        return f"<context>\n{context}\n</context>\n\n{chunk_content}"


# ============================================================================
# Metadata Extraction
# ============================================================================

class MetadataExtractor:
    """
    Extract rich metadata from chunks.

    Based on LlamaIndex patterns:
    https://docs.llamaindex.ai/en/stable/module_guides/indexing/metadata_extraction/
    """

    def __init__(self, llm_model: str = "llama3.1:8b", use_ollama: bool = True):
        self.llm_model = llm_model
        self.use_ollama = use_ollama
        self._client = None

    @property
    def client(self):
        if self._client is None and self.use_ollama:
            try:
                import ollama
                self._client = ollama.Client()
            except ImportError:
                pass
        return self._client

    def extract_keywords(self, text: str, max_keywords: int = 10) -> List[str]:
        """Extract keywords using simple TF approach or LLM."""
        # Simple keyword extraction without LLM
        words = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', text)
        word_freq = {}
        for word in words:
            word_lower = word.lower()
            if len(word_lower) > 3:
                word_freq[word] = word_freq.get(word, 0) + 1

        # Sort by frequency
        sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
        return [w[0] for w in sorted_words[:max_keywords]]

    def extract_title(self, text: str, document_title: str) -> str:
        """Extract or infer section title."""
        # Look for markdown headers
        header_match = re.search(r'^#{1,3}\s+(.+)$', text, re.MULTILINE)
        if header_match:
            return header_match.group(1).strip()
        return document_title

    def generate_summary(self, text: str) -> Optional[str]:
        """Generate a brief summary using LLM."""
        if not self.client or len(text) < 100:
            return None

        try:
            response = self.client.chat(
                model=self.llm_model,
                messages=[{
                    "role": "user",
                    "content": f"Summarize this text in 1-2 sentences:\n\n{text[:2000]}"
                }],
                options={"temperature": 0.3, "num_predict": 100}
            )
            return response["message"]["content"].strip()
        except Exception:
            return None

    def generate_questions(self, text: str, max_questions: int = 3) -> List[str]:
        """Generate questions this chunk can answer."""
        if not self.client or len(text) < 100:
            return []

        try:
            response = self.client.chat(
                model=self.llm_model,
                messages=[{
                    "role": "user",
                    "content": f"What {max_questions} questions does this text answer? List only the questions, one per line:\n\n{text[:2000]}"
                }],
                options={"temperature": 0.5, "num_predict": 200}
            )
            questions = response["message"]["content"].strip().split("\n")
            return [q.strip().lstrip("0123456789.-) ") for q in questions if q.strip()][:max_questions]
        except Exception:
            return []


# ============================================================================
# Entity Extraction
# ============================================================================

class EntityExtractor:
    """
    Extract entities for knowledge graph.

    Reference: https://neo4j.com/labs/genai-ecosystem/llamaindex/
    """

    ENTITY_TYPES = [
        "PERSON", "ORGANIZATION", "PRODUCT", "TECHNOLOGY",
        "CONCEPT", "LOCATION", "EVENT", "DOCUMENT"
    ]

    def __init__(self, llm_model: str = "llama3.1:8b", use_ollama: bool = True):
        self.llm_model = llm_model
        self.use_ollama = use_ollama
        self._client = None

    @property
    def client(self):
        if self._client is None and self.use_ollama:
            try:
                import ollama
                self._client = ollama.Client()
            except ImportError:
                pass
        return self._client

    def extract_entities_rule_based(self, text: str) -> List[Dict[str, str]]:
        """Extract entities using regex patterns."""
        entities = []

        # Capitalized phrases (likely proper nouns)
        proper_nouns = re.findall(r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b', text)
        for noun in set(proper_nouns):
            if len(noun) > 2:
                entities.append({"name": noun, "type": "CONCEPT"})

        # Technical terms (CamelCase or ALLCAPS)
        tech_terms = re.findall(r'\b([A-Z][a-z]+[A-Z][a-z]+\w*|[A-Z]{2,})\b', text)
        for term in set(tech_terms):
            if len(term) > 2:
                entities.append({"name": term, "type": "TECHNOLOGY"})

        return entities[:20]  # Limit entities per chunk

    def extract_entities_llm(self, text: str) -> List[Dict[str, str]]:
        """Extract entities using LLM."""
        if not self.client:
            return self.extract_entities_rule_based(text)

        prompt = f"""Extract key entities from this text. For each entity, provide the name and type.
Types: {', '.join(self.ENTITY_TYPES)}

Text: {text[:2000]}

Respond in JSON format: [{{"name": "entity name", "type": "TYPE"}}, ...]
Only include important, specific entities. Limit to 10 entities."""

        try:
            response = self.client.chat(
                model=self.llm_model,
                messages=[{"role": "user", "content": prompt}],
                options={"temperature": 0.3, "num_predict": 500}
            )

            # Parse JSON from response
            content = response["message"]["content"]
            # Find JSON array in response
            match = re.search(r'\[.*\]', content, re.DOTALL)
            if match:
                entities = json.loads(match.group())
                return entities[:10]
        except Exception as e:
            console.print(f"[yellow]Entity extraction failed: {e}[/yellow]")

        return self.extract_entities_rule_based(text)


# ============================================================================
# Quality Validation
# ============================================================================

class QualityValidator:
    """Validate ingestion quality."""

    def __init__(self):
        self.seen_hashes = set()

    def is_duplicate(self, content_hash: str) -> bool:
        """Check if content is a duplicate."""
        if content_hash in self.seen_hashes:
            return True
        self.seen_hashes.add(content_hash)
        return False

    def compute_quality_score(self, chunk: EnhancedChunk) -> float:
        """Compute quality score for a chunk (0-1)."""
        score = 1.0

        # Penalize very short chunks
        if chunk.metadata.token_count < 50:
            score *= 0.7

        # Penalize chunks without meaningful content
        if len(chunk.content.strip()) < 100:
            score *= 0.8

        # Boost chunks with rich metadata
        if chunk.metadata.keywords:
            score *= 1.1
        if chunk.metadata.entities:
            score *= 1.1
        if chunk.metadata.summary:
            score *= 1.05

        return min(score, 1.0)

    def validate_batch(self, chunks: List[EnhancedChunk]) -> Dict[str, Any]:
        """Validate a batch of chunks."""
        total = len(chunks)
        duplicates = sum(1 for c in chunks if self.is_duplicate(c.content_hash))
        low_quality = sum(1 for c in chunks if c.metadata.quality_score < 0.7)

        return {
            "total_chunks": total,
            "duplicates": duplicates,
            "low_quality": low_quality,
            "unique_entities": len(set(
                e["name"] for c in chunks for e in c.metadata.entities
            )),
            "avg_quality": sum(c.metadata.quality_score for c in chunks) / total if total else 0
        }


# ============================================================================
# Enhanced Ingestion Pipeline
# ============================================================================

class EnhancedIngestionPipeline:
    """
    Complete ingestion pipeline with all features.
    """

    def __init__(
        self,
        collection: str = "default",
        mode: str = "standard",  # quick, standard, full
        chunk_size: int = 512,
        chunk_overlap: int = 50,
        use_contextual: bool = False,
        extract_entities: bool = True,
        extract_metadata: bool = True,
        llm_model: str = "llama3.1:8b"
    ):
        self.collection = collection
        self.mode = mode
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.use_contextual = use_contextual or mode == "full"
        self.extract_entities = extract_entities and mode != "quick"
        self.extract_metadata = extract_metadata and mode != "quick"
        self.llm_model = llm_model

        # Initialize components
        self.contextual = ContextualRetrieval(llm_model) if self.use_contextual else None
        self.metadata_extractor = MetadataExtractor(llm_model) if self.extract_metadata else None
        self.entity_extractor = EntityExtractor(llm_model) if self.extract_entities else None
        self.validator = QualityValidator()

        # Lazy load knowledge base
        self._kb = None

    @property
    def kb(self):
        """Lazy load knowledge base."""
        if self._kb is None:
            from .knowledge_base import KnowledgeBase
            self._kb = KnowledgeBase(collection_name="knowledge_base")
            self._kb.initialize(recreate=False)
        return self._kb

    def _chunk_text(self, text: str) -> List[Tuple[str, int]]:
        """Split text into chunks with overlap."""
        words = text.split()
        chunks = []

        if len(words) <= self.chunk_size:
            return [(text, 0)]

        start = 0
        chunk_index = 0

        while start < len(words):
            end = start + self.chunk_size
            chunk_words = words[start:end]
            chunk_text = " ".join(chunk_words)
            chunks.append((chunk_text, chunk_index))
            start = end - self.chunk_overlap
            chunk_index += 1

        return chunks

    def _process_chunk(
        self,
        chunk_text: str,
        chunk_index: int,
        document_id: str,
        document_title: str,
        full_document: str
    ) -> EnhancedChunk:
        """Process a single chunk with all enhancements."""
        metadata = ChunkMetadata(
            title=document_title,
            token_count=len(chunk_text.split())
        )

        # Extract metadata
        if self.metadata_extractor:
            metadata.keywords = self.metadata_extractor.extract_keywords(chunk_text)
            metadata.title = self.metadata_extractor.extract_title(chunk_text, document_title)

            if self.mode == "full":
                metadata.summary = self.metadata_extractor.generate_summary(chunk_text)
                metadata.questions_answered = self.metadata_extractor.generate_questions(chunk_text)

        # Extract entities
        if self.entity_extractor:
            if self.mode == "full":
                metadata.entities = self.entity_extractor.extract_entities_llm(chunk_text)
            else:
                metadata.entities = self.entity_extractor.extract_entities_rule_based(chunk_text)

        # Generate context
        contextualized = chunk_text
        if self.contextual:
            context = self.contextual.generate_context(
                chunk_text, full_document, document_title
            )
            metadata.context_prefix = context
            contextualized = self.contextual.contextualize_chunk(chunk_text, context)

        chunk = EnhancedChunk(
            content=chunk_text,
            contextualized_content=contextualized,
            chunk_index=chunk_index,
            document_id=document_id,
            metadata=metadata
        )

        # Compute quality score
        chunk.metadata.quality_score = self.validator.compute_quality_score(chunk)

        return chunk

    def ingest_file(self, file_path: Path) -> IngestResult:
        """Ingest a single file."""
        import time
        start_time = time.time()

        try:
            # Read and process document
            if file_path.suffix.lower() == ".md":
                content = file_path.read_text(encoding="utf-8")
                doc_type = "markdown"
            else:
                # Use Docling for other formats
                from .document_processor import DocumentProcessor
                processor = DocumentProcessor()
                doc = processor.process_file(file_path)
                content = doc.content
                doc_type = doc.doc_type

            # Check for duplicates at document level
            doc_hash = hashlib.sha256(content.encode()).hexdigest()
            if self.validator.is_duplicate(doc_hash):
                return IngestResult(
                    document_id="",
                    title=file_path.stem,
                    chunk_count=0,
                    entity_count=0,
                    status="duplicate"
                )

            # Generate document ID
            import uuid
            document_id = str(uuid.uuid4())
            document_title = file_path.stem

            # Chunk the document
            raw_chunks = self._chunk_text(content)

            # Process chunks
            enhanced_chunks = []
            all_entities = []

            for chunk_text, chunk_index in raw_chunks:
                chunk = self._process_chunk(
                    chunk_text=chunk_text,
                    chunk_index=chunk_index,
                    document_id=document_id,
                    document_title=document_title,
                    full_document=content
                )
                enhanced_chunks.append(chunk)
                all_entities.extend(chunk.metadata.entities)

            # Generate embeddings
            from .embeddings import get_embedding_service
            embedding_service = get_embedding_service()

            # Use contextualized content for embeddings if available
            texts_to_embed = [c.contextualized_content for c in enhanced_chunks]
            embeddings = embedding_service.encode_documents(texts_to_embed)

            for chunk, embedding in zip(enhanced_chunks, embeddings):
                chunk.embedding = embedding.tolist()

            # Store in vector database
            self._store_chunks(enhanced_chunks, document_id, document_title, doc_type)

            # Store entities in graph
            unique_entities = self._dedupe_entities(all_entities)
            self._store_entities(unique_entities, document_id, enhanced_chunks)

            processing_time = time.time() - start_time

            return IngestResult(
                document_id=document_id,
                title=document_title,
                chunk_count=len(enhanced_chunks),
                entity_count=len(unique_entities),
                status="success",
                processing_time=processing_time
            )

        except Exception as e:
            return IngestResult(
                document_id="",
                title=file_path.stem,
                chunk_count=0,
                entity_count=0,
                status="error",
                error_message=str(e)
            )

    def _store_chunks(
        self,
        chunks: List[EnhancedChunk],
        document_id: str,
        title: str,
        doc_type: str
    ):
        """Store chunks in Qdrant."""
        from qdrant_client.http.models import PointStruct

        import uuid as uuid_lib

        points = []
        for chunk in chunks:
            # Generate proper UUID for Qdrant
            chunk_uuid = str(uuid_lib.uuid4())
            points.append(PointStruct(
                id=chunk_uuid,
                vector=chunk.embedding,
                payload={
                    "document_id": document_id,
                    "chunk_index": chunk.chunk_index,
                    "content": chunk.content,
                    "contextualized_content": chunk.contextualized_content,
                    "content_hash": chunk.content_hash,  # For deduplication
                    "title": title,
                    "doc_type": doc_type,
                    "collection": self.collection,
                    "keywords": chunk.metadata.keywords,
                    "entities": [e["name"] for e in chunk.metadata.entities],
                    "summary": chunk.metadata.summary,
                    "questions": chunk.metadata.questions_answered,
                    "quality_score": chunk.metadata.quality_score
                }
            ))

        # Upsert in batches
        batch_size = 100
        for i in range(0, len(points), batch_size):
            batch = points[i:i + batch_size]
            self.kb.vector_store.client.upsert(
                collection_name=self.kb.collection_name,
                points=batch
            )

    def _dedupe_entities(self, entities: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """Deduplicate entities by name."""
        seen = {}
        for e in entities:
            name = e["name"].lower()
            if name not in seen:
                seen[name] = e
        return list(seen.values())

    def _store_entities(
        self,
        entities: List[Dict[str, str]],
        document_id: str,
        chunks: List[EnhancedChunk]
    ):
        """Store entities in Neo4j."""
        for entity in entities:
            self.kb.add_entity(
                name=entity["name"],
                entity_type=entity["type"]
            )

        # Link entities to document
        for chunk in chunks:
            for entity in chunk.metadata.entities:
                self.kb.graph_store.link_entity_to_chunk(
                    entity_name=entity["name"],
                    document_id=document_id,
                    chunk_index=chunk.chunk_index
                )

    def ingest_directory(
        self,
        directory: Path,
        extensions: List[str] = None,
        workers: int = 1
    ) -> List[IngestResult]:
        """Ingest all files in a directory."""
        extensions = extensions or [".md", ".pdf", ".docx", ".txt", ".html"]
        extensions = [e if e.startswith(".") else f".{e}" for e in extensions]

        files = []
        for ext in extensions:
            files.extend(directory.rglob(f"*{ext}"))

        console.print(f"[bold]Found {len(files)} files to ingest[/bold]")

        results = []

        if workers > 1:
            with ThreadPoolExecutor(max_workers=workers) as executor:
                futures = {executor.submit(self.ingest_file, f): f for f in files}

                with Progress(
                    SpinnerColumn(),
                    TextColumn("[progress.description]{task.description}"),
                    BarColumn(),
                    console=console
                ) as progress:
                    task = progress.add_task("Ingesting...", total=len(files))

                    for future in as_completed(futures):
                        result = future.result()
                        results.append(result)
                        progress.advance(task)
        else:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                console=console
            ) as progress:
                task = progress.add_task("Ingesting...", total=len(files))

                for file_path in files:
                    result = self.ingest_file(file_path)
                    results.append(result)
                    progress.advance(task)

        # Print summary
        self._print_summary(results)

        return results

    def _print_summary(self, results: List[IngestResult]):
        """Print ingestion summary."""
        table = Table(title="Ingestion Summary")
        table.add_column("Status", style="cyan")
        table.add_column("Count", style="magenta")

        success = sum(1 for r in results if r.status == "success")
        duplicates = sum(1 for r in results if r.status == "duplicate")
        errors = sum(1 for r in results if r.status == "error")
        total_chunks = sum(r.chunk_count for r in results)
        total_entities = sum(r.entity_count for r in results)

        table.add_row("Success", str(success))
        table.add_row("Duplicates", str(duplicates))
        table.add_row("Errors", str(errors))
        table.add_row("Total Chunks", str(total_chunks))
        table.add_row("Total Entities", str(total_entities))

        console.print(table)

        if errors > 0:
            console.print("\n[red]Errors:[/red]")
            for r in results:
                if r.status == "error":
                    console.print(f"  - {r.title}: {r.error_message}")


# ============================================================================
# CLI Commands
# ============================================================================

@app.command()
def file(
    path: str = typer.Argument(..., help="Path to file"),
    collection: str = typer.Option("default", "-c", "--collection"),
    mode: str = typer.Option("standard", "-m", "--mode", help="quick, standard, or full"),
    contextual: bool = typer.Option(False, "--contextual", help="Enable contextual retrieval"),
    extract_entities: bool = typer.Option(True, "--entities/--no-entities")
):
    """Ingest a single file."""
    pipeline = EnhancedIngestionPipeline(
        collection=collection,
        mode=mode,
        use_contextual=contextual,
        extract_entities=extract_entities
    )

    result = pipeline.ingest_file(Path(path))

    if result.status == "success":
        console.print(f"[green]Ingested: {result.title}[/green]")
        console.print(f"  Chunks: {result.chunk_count}, Entities: {result.entity_count}")
        console.print(f"  Time: {result.processing_time:.2f}s")
    elif result.status == "duplicate":
        console.print(f"[yellow]Duplicate: {result.title}[/yellow]")
    else:
        console.print(f"[red]Error: {result.title} - {result.error_message}[/red]")


@app.command()
def directory(
    path: str = typer.Argument(..., help="Path to directory"),
    collection: str = typer.Option("default", "-c", "--collection"),
    mode: str = typer.Option("standard", "-m", "--mode"),
    workers: int = typer.Option(1, "-w", "--workers"),
    extensions: str = typer.Option("md,pdf,docx,txt", "-e", "--ext"),
    contextual: bool = typer.Option(False, "--contextual"),
    extract_entities: bool = typer.Option(True, "--entities/--no-entities")
):
    """Ingest all files in a directory."""
    pipeline = EnhancedIngestionPipeline(
        collection=collection,
        mode=mode,
        use_contextual=contextual,
        extract_entities=extract_entities
    )

    ext_list = [e.strip() for e in extensions.split(",")]

    pipeline.ingest_directory(
        directory=Path(path),
        extensions=ext_list,
        workers=workers
    )


@app.command()
def validate(
    collection: str = typer.Option("default", "-c", "--collection")
):
    """Validate ingestion quality for a collection."""
    from .knowledge_base import KnowledgeBase

    kb = KnowledgeBase()
    kb.initialize(recreate=False)

    stats = kb.get_stats()
    kb.print_stats()


if __name__ == "__main__":
    app()
