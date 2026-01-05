"""Unified Knowledge Base - combines vector search, graph, and hybrid retrieval."""
from typing import List, Dict, Any, Optional, Union
from pathlib import Path
from dataclasses import dataclass

from rich.console import Console
from rich.table import Table

from .config import settings
from .document_processor import DocumentProcessor, ProcessedDocument
from .vector_store import VectorStore, SearchResult
from .graph_store import GraphStore, Entity, Relationship
from .embeddings import get_embedding_service

console = Console()


@dataclass
class HybridSearchResult:
    """Result from hybrid search combining vector and graph."""
    content: str
    score: float
    source: str  # 'vector', 'graph', or 'hybrid'
    document_title: str
    document_id: str
    chunk_index: Optional[int] = None
    related_entities: List[str] = None
    metadata: Dict[str, Any] = None


class KnowledgeBase:
    """
    Unified Knowledge Base system combining:
    - Vector search (Qdrant) for semantic similarity
    - Knowledge graph (Neo4j) for relationships
    - Hybrid retrieval for best results
    """

    def __init__(
        self,
        collection_name: str = "knowledge_base",
        chunk_size: int = 512,
        chunk_overlap: int = 50
    ):
        self.collection_name = collection_name

        # Initialize components
        self.processor = DocumentProcessor(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        self.vector_store = VectorStore(collection_name=collection_name)
        self.graph_store = GraphStore()

        self._initialized = False

    def initialize(self, recreate: bool = False):
        """Initialize the knowledge base (create collections, schema, etc.)."""
        console.print("[bold blue]Initializing Knowledge Base...[/bold blue]")

        # Create vector collection
        self.vector_store.create_collection(recreate=recreate)

        # Setup graph schema
        self.graph_store.setup_schema()

        # Warm up embedding model
        _ = get_embedding_service().encode("warmup")

        self._initialized = True
        console.print("[bold green]Knowledge Base initialized successfully![/bold green]")

    def ingest_file(
        self,
        file_path: Union[str, Path],
        collection: str = "default",
        extract_entities: bool = False
    ) -> str:
        """Ingest a single file into the knowledge base."""
        # Process document
        doc = self.processor.process_file(file_path)

        # Add to vector store
        doc_id = self.vector_store.add_document(doc, collection=collection)

        # Add to graph
        self.graph_store.add_document_to_graph(doc, doc_id, collection=collection)

        return doc_id

    def ingest_directory(
        self,
        directory: Union[str, Path],
        collection: str = "default",
        recursive: bool = True,
        extensions: Optional[List[str]] = None
    ) -> List[str]:
        """Ingest all documents in a directory."""
        document_ids = []

        for doc in self.processor.process_directory(
            directory,
            recursive=recursive,
            extensions=extensions
        ):
            doc_id = self.vector_store.add_document(doc, collection=collection)
            self.graph_store.add_document_to_graph(doc, doc_id, collection=collection)
            document_ids.append(doc_id)

        console.print(f"[bold green]Ingested {len(document_ids)} documents[/bold green]")
        return document_ids

    def ingest_markdown_directory(
        self,
        directory: Union[str, Path],
        collection: str = "default",
        recursive: bool = True
    ) -> List[str]:
        """Optimized ingestion for markdown files."""
        document_ids = []

        for doc in self.processor.process_markdown_directory(
            directory,
            recursive=recursive
        ):
            doc_id = self.vector_store.add_document(doc, collection=collection)
            self.graph_store.add_document_to_graph(doc, doc_id, collection=collection)
            document_ids.append(doc_id)

        console.print(f"[bold green]Ingested {len(document_ids)} markdown files[/bold green]")
        return document_ids

    def ingest_text(
        self,
        text: str,
        title: str = "Untitled",
        collection: str = "default",
        source_url: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Ingest raw text content."""
        doc = self.processor.process_text(
            text=text,
            title=title,
            source_url=source_url,
            metadata=metadata
        )

        doc_id = self.vector_store.add_document(doc, collection=collection)
        self.graph_store.add_document_to_graph(doc, doc_id, collection=collection)

        return doc_id

    def search(
        self,
        query: str,
        limit: int = 10,
        collection: Optional[str] = None,
        mode: str = "vector"  # 'vector', 'graph', or 'hybrid'
    ) -> List[HybridSearchResult]:
        """
        Search the knowledge base.

        Args:
            query: Search query
            limit: Maximum number of results
            collection: Filter by collection name
            mode: Search mode ('vector', 'graph', or 'hybrid')
        """
        results = []

        if mode in ("vector", "hybrid"):
            vector_results = self.vector_store.search(
                query=query,
                limit=limit,
                collection=collection
            )

            for r in vector_results:
                results.append(HybridSearchResult(
                    content=r.content,
                    score=r.score,
                    source="vector",
                    document_title=r.metadata.get("title", "Unknown"),
                    document_id=r.document_id,
                    chunk_index=r.metadata.get("chunk_index"),
                    metadata=r.metadata
                ))

        if mode in ("graph", "hybrid"):
            # Search entities in graph
            entity_results = self.graph_store.search_entities(query, limit=limit)

            for entity in entity_results:
                # Find documents mentioning this entity
                docs = self.graph_store.find_documents_for_entity(
                    entity["name"],
                    limit=5
                )
                for doc in docs:
                    results.append(HybridSearchResult(
                        content=f"Entity: {entity['name']} ({entity['type']})",
                        score=0.5,  # Default score for graph results
                        source="graph",
                        document_title=doc.get("title", "Unknown"),
                        document_id=doc.get("document_id", ""),
                        related_entities=[entity["name"]],
                        metadata={"mention_count": doc.get("mention_count", 0)}
                    ))

        # Sort by score and deduplicate
        results.sort(key=lambda x: x.score, reverse=True)

        # Remove duplicates by document_id
        seen = set()
        unique_results = []
        for r in results:
            if r.document_id not in seen:
                seen.add(r.document_id)
                unique_results.append(r)

        return unique_results[:limit]

    def add_entity(
        self,
        name: str,
        entity_type: str,
        properties: Optional[Dict[str, Any]] = None
    ) -> str:
        """Add an entity to the knowledge graph."""
        entity = Entity(
            name=name,
            entity_type=entity_type,
            properties=properties or {}
        )
        return self.graph_store.add_entity(entity)

    def add_relationship(
        self,
        source: str,
        target: str,
        relationship_type: str,
        properties: Optional[Dict[str, Any]] = None
    ):
        """Add a relationship between entities."""
        relationship = Relationship(
            source=source,
            target=target,
            relationship_type=relationship_type,
            properties=properties or {}
        )
        self.graph_store.add_relationship(relationship)

    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the knowledge base."""
        vector_info = self.vector_store.get_collection_info()
        graph_stats = self.graph_store.get_graph_stats()

        return {
            "vector_store": vector_info,
            "graph_store": graph_stats,
            "collection_name": self.collection_name
        }

    def print_stats(self):
        """Print statistics as a formatted table."""
        stats = self.get_stats()

        table = Table(title="Knowledge Base Statistics")
        table.add_column("Component", style="cyan")
        table.add_column("Metric", style="magenta")
        table.add_column("Value", style="green")

        # Vector store stats
        vs = stats["vector_store"]
        table.add_row("Vector Store", "Collection", vs.get("name", "N/A"))
        table.add_row("", "Vectors", str(vs.get("vectors_count", 0)))
        table.add_row("", "Status", vs.get("status", "N/A"))

        # Graph store stats
        gs = stats["graph_store"]
        table.add_row("Graph Store", "Documents", str(gs.get("documents", 0)))
        table.add_row("", "Chunks", str(gs.get("chunks", 0)))
        table.add_row("", "Entities", str(gs.get("entities", 0)))
        table.add_row("", "Relationships", str(gs.get("relationships", 0)))

        console.print(table)

    def delete_document(self, document_id: str):
        """Delete a document from both stores."""
        self.vector_store.delete_document(document_id)
        # Note: Graph deletion would need additional implementation
        console.print(f"[yellow]Deleted document {document_id}[/yellow]")

    def get_entity_context(
        self,
        entity_name: str,
        depth: int = 2
    ) -> Dict[str, Any]:
        """Get rich context about an entity including related entities and documents."""
        # Get related entities
        related = self.graph_store.find_related_entities(
            entity_name,
            max_depth=depth
        )

        # Get documents
        documents = self.graph_store.find_documents_for_entity(entity_name)

        # Get graph visualization data
        graph_data = self.graph_store.get_entity_graph(entity_name, depth=depth)

        return {
            "entity": entity_name,
            "related_entities": related,
            "documents": documents,
            "graph": graph_data
        }
