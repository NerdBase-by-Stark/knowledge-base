"""Document processing pipeline using Docling."""
import hashlib
from pathlib import Path
from typing import List, Dict, Any, Optional, Iterator, Union
from dataclasses import dataclass, field
from datetime import datetime
import json

from docling.document_converter import DocumentConverter
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.backend.pypdfium2_backend import PyPdfiumDocumentBackend
from docling_core.types.doc import DoclingDocument

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn

console = Console()


@dataclass
class ProcessedChunk:
    """A processed chunk of content."""
    content: str
    chunk_index: int
    metadata: Dict[str, Any] = field(default_factory=dict)
    token_count: Optional[int] = None


@dataclass
class ProcessedDocument:
    """A fully processed document."""
    title: str
    content: str
    content_hash: str
    doc_type: str
    source_path: Optional[str] = None
    source_url: Optional[str] = None
    chunks: List[ProcessedChunk] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    tables: List[Dict[str, Any]] = field(default_factory=list)
    images: List[Dict[str, Any]] = field(default_factory=list)
    processed_at: datetime = field(default_factory=datetime.now)


class DocumentProcessor:
    """Process documents using Docling for high-quality extraction."""

    SUPPORTED_FORMATS = {
        ".pdf": InputFormat.PDF,
        ".docx": InputFormat.DOCX,
        ".pptx": InputFormat.PPTX,
        ".xlsx": InputFormat.XLSX,
        ".html": InputFormat.HTML,
        ".htm": InputFormat.HTML,
        ".md": InputFormat.MD,
        ".txt": InputFormat.MD,  # Treat plain text as markdown
        ".png": InputFormat.IMAGE,
        ".jpg": InputFormat.IMAGE,
        ".jpeg": InputFormat.IMAGE,
        ".tiff": InputFormat.IMAGE,
    }

    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 50,
        extract_tables: bool = True,
        extract_images: bool = True,
        ocr_enabled: bool = True
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.extract_tables = extract_tables
        self.extract_images = extract_images
        self.ocr_enabled = ocr_enabled
        self._converter = None

    @property
    def converter(self) -> DocumentConverter:
        """Lazy load the document converter."""
        if self._converter is None:
            console.print("[bold blue]Initializing Docling converter...[/bold blue]")

            # Configure PDF pipeline
            pipeline_options = PdfPipelineOptions()
            pipeline_options.do_ocr = self.ocr_enabled
            pipeline_options.do_table_structure = self.extract_tables

            self._converter = DocumentConverter(
                allowed_formats=list(InputFormat),
                format_options={
                    InputFormat.PDF: {"pipeline_options": pipeline_options}
                }
            )
            console.print("[bold green]Docling converter ready[/bold green]")

        return self._converter

    def _compute_hash(self, content: str) -> str:
        """Compute SHA256 hash of content."""
        return hashlib.sha256(content.encode()).hexdigest()

    def _chunk_text(self, text: str) -> List[ProcessedChunk]:
        """Split text into overlapping chunks."""
        chunks = []
        words = text.split()

        if len(words) <= self.chunk_size:
            chunks.append(ProcessedChunk(
                content=text,
                chunk_index=0,
                token_count=len(words)
            ))
            return chunks

        start = 0
        chunk_index = 0

        while start < len(words):
            end = start + self.chunk_size
            chunk_words = words[start:end]
            chunk_text = " ".join(chunk_words)

            chunks.append(ProcessedChunk(
                content=chunk_text,
                chunk_index=chunk_index,
                token_count=len(chunk_words)
            ))

            start = end - self.chunk_overlap
            chunk_index += 1

        return chunks

    def _extract_tables(self, doc: DoclingDocument) -> List[Dict[str, Any]]:
        """Extract tables from document."""
        tables = []
        try:
            for table in doc.tables:
                table_data = {
                    "content": table.export_to_markdown() if hasattr(table, 'export_to_markdown') else str(table),
                    "metadata": {}
                }
                tables.append(table_data)
        except Exception as e:
            console.print(f"[yellow]Warning: Could not extract tables: {e}[/yellow]")
        return tables

    def process_file(self, file_path: Union[str, Path]) -> ProcessedDocument:
        """Process a single file."""
        file_path = Path(file_path)

        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        suffix = file_path.suffix.lower()
        if suffix not in self.SUPPORTED_FORMATS:
            raise ValueError(f"Unsupported format: {suffix}")

        console.print(f"[bold]Processing: {file_path.name}[/bold]")

        # Convert document
        result = self.converter.convert(str(file_path))
        doc = result.document

        # Extract content
        content = doc.export_to_markdown()

        # Extract tables if enabled
        tables = []
        if self.extract_tables:
            tables = self._extract_tables(doc)

        # Create chunks
        chunks = self._chunk_text(content)

        # Add table content to chunks if present
        for i, table in enumerate(tables):
            table_chunk = ProcessedChunk(
                content=f"[TABLE {i+1}]\n{table['content']}",
                chunk_index=len(chunks),
                metadata={"type": "table", "table_index": i}
            )
            chunks.append(table_chunk)

        return ProcessedDocument(
            title=file_path.stem,
            content=content,
            content_hash=self._compute_hash(content),
            doc_type=suffix[1:],  # Remove the dot
            source_path=str(file_path.absolute()),
            chunks=chunks,
            tables=tables,
            metadata={
                "filename": file_path.name,
                "size_bytes": file_path.stat().st_size,
                "chunk_count": len(chunks)
            }
        )

    def process_text(
        self,
        text: str,
        title: str = "Untitled",
        source_url: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> ProcessedDocument:
        """Process raw text content."""
        chunks = self._chunk_text(text)

        return ProcessedDocument(
            title=title,
            content=text,
            content_hash=self._compute_hash(text),
            doc_type="text",
            source_url=source_url,
            chunks=chunks,
            metadata=metadata or {}
        )

    def process_directory(
        self,
        directory: Union[str, Path],
        recursive: bool = True,
        extensions: Optional[List[str]] = None
    ) -> Iterator[ProcessedDocument]:
        """Process all documents in a directory."""
        directory = Path(directory)

        if not directory.is_dir():
            raise NotADirectoryError(f"Not a directory: {directory}")

        # Get all files
        pattern = "**/*" if recursive else "*"
        files = list(directory.glob(pattern))

        # Filter by extension if specified
        if extensions:
            extensions = [e.lower() if e.startswith(".") else f".{e.lower()}" for e in extensions]
            files = [f for f in files if f.suffix.lower() in extensions]
        else:
            files = [f for f in files if f.suffix.lower() in self.SUPPORTED_FORMATS]

        console.print(f"[bold]Found {len(files)} documents to process[/bold]")

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            console=console
        ) as progress:
            task = progress.add_task("Processing documents...", total=len(files))

            for file_path in files:
                try:
                    doc = self.process_file(file_path)
                    yield doc
                except Exception as e:
                    console.print(f"[red]Error processing {file_path.name}: {e}[/red]")
                finally:
                    progress.advance(task)

    def process_markdown_directory(
        self,
        directory: Union[str, Path],
        recursive: bool = True
    ) -> Iterator[ProcessedDocument]:
        """Optimized processing for markdown files (no Docling needed)."""
        directory = Path(directory)

        if not directory.is_dir():
            raise NotADirectoryError(f"Not a directory: {directory}")

        pattern = "**/*.md" if recursive else "*.md"
        files = list(directory.glob(pattern))

        console.print(f"[bold]Found {len(files)} markdown files[/bold]")

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            console=console
        ) as progress:
            task = progress.add_task("Processing markdown...", total=len(files))

            for file_path in files:
                try:
                    content = file_path.read_text(encoding="utf-8")
                    chunks = self._chunk_text(content)

                    yield ProcessedDocument(
                        title=file_path.stem,
                        content=content,
                        content_hash=self._compute_hash(content),
                        doc_type="markdown",
                        source_path=str(file_path.absolute()),
                        chunks=chunks,
                        metadata={
                            "filename": file_path.name,
                            "size_bytes": file_path.stat().st_size,
                            "chunk_count": len(chunks)
                        }
                    )
                except Exception as e:
                    console.print(f"[red]Error processing {file_path.name}: {e}[/red]")
                finally:
                    progress.advance(task)
