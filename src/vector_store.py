"""Vector store implementation using Qdrant."""
import uuid
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

from qdrant_client import QdrantClient
from qdrant_client.http import models as qdrant_models
from qdrant_client.http.models import (
    Distance,
    VectorParams,
    PointStruct,
    Filter,
    FieldCondition,
    MatchValue,
    SearchParams,
    HnswConfigDiff,
    OptimizersConfigDiff
)

from rich.console import Console
from .config import settings
from .embeddings import get_embedding_service
from .document_processor import ProcessedDocument, ProcessedChunk

console = Console()


@dataclass
class SearchResult:
    """Search result from vector store."""
    chunk_id: str
    document_id: str
    content: str
    score: float
    metadata: Dict[str, Any]


class VectorStore:
    """Vector store using Qdrant for similarity search."""

    def __init__(
        self,
        collection_name: str = "knowledge_base",
        dimension: int = 1024,
        distance: str = "cosine"
    ):
        self.collection_name = collection_name
        self.dimension = dimension
        self.distance = Distance.COSINE if distance == "cosine" else Distance.DOT

        self._client = None
        self._embedding_service = None

    @property
    def client(self) -> QdrantClient:
        """Lazy load Qdrant client."""
        if self._client is None:
            self._client = QdrantClient(
                host=settings.qdrant_host,
                port=settings.qdrant_port,
                prefer_grpc=True
            )
        return self._client

    @property
    def embedding_service(self):
        """Get embedding service."""
        if self._embedding_service is None:
            self._embedding_service = get_embedding_service()
        return self._embedding_service

    def create_collection(self, recreate: bool = False) -> bool:
        """Create the collection if it doesn't exist."""
        collections = self.client.get_collections().collections
        exists = any(c.name == self.collection_name for c in collections)

        if exists and not recreate:
            console.print(f"[yellow]Collection '{self.collection_name}' already exists[/yellow]")
            return False

        if exists and recreate:
            self.client.delete_collection(self.collection_name)
            console.print(f"[yellow]Deleted existing collection '{self.collection_name}'[/yellow]")

        # Create collection with optimized settings
        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(
                size=self.dimension,
                distance=self.distance,
                on_disk=False  # Keep in memory for speed
            ),
            hnsw_config=HnswConfigDiff(
                m=16,  # Number of edges per node
                ef_construct=100,  # Construction time accuracy
                full_scan_threshold=10000
            ),
            optimizers_config=OptimizersConfigDiff(
                indexing_threshold=20000,
                memmap_threshold=50000
            )
        )

        # Create payload indexes for filtering
        self.client.create_payload_index(
            collection_name=self.collection_name,
            field_name="document_id",
            field_schema=qdrant_models.PayloadSchemaType.KEYWORD
        )
        self.client.create_payload_index(
            collection_name=self.collection_name,
            field_name="doc_type",
            field_schema=qdrant_models.PayloadSchemaType.KEYWORD
        )
        self.client.create_payload_index(
            collection_name=self.collection_name,
            field_name="collection",
            field_schema=qdrant_models.PayloadSchemaType.KEYWORD
        )

        console.print(f"[green]Created collection '{self.collection_name}'[/green]")
        return True

    def add_document(
        self,
        document: ProcessedDocument,
        collection: str = "default"
    ) -> str:
        """Add a processed document to the vector store."""
        document_id = str(uuid.uuid4())

        # Encode all chunks
        chunk_texts = [chunk.content for chunk in document.chunks]
        embeddings = self.embedding_service.encode_documents(chunk_texts)

        # Create points
        points = []
        for i, (chunk, embedding) in enumerate(zip(document.chunks, embeddings)):
            chunk_id = str(uuid.uuid4())
            points.append(PointStruct(
                id=chunk_id,
                vector=embedding.tolist(),
                payload={
                    "document_id": document_id,
                    "chunk_index": chunk.chunk_index,
                    "content": chunk.content,
                    "title": document.title,
                    "doc_type": document.doc_type,
                    "source_path": document.source_path,
                    "source_url": document.source_url,
                    "collection": collection,
                    "metadata": {**document.metadata, **chunk.metadata}
                }
            ))

        # Upsert in batches
        batch_size = 100
        for i in range(0, len(points), batch_size):
            batch = points[i:i + batch_size]
            self.client.upsert(
                collection_name=self.collection_name,
                points=batch
            )

        console.print(
            f"[green]Added document '{document.title}' with {len(points)} chunks[/green]"
        )
        return document_id

    def add_documents(
        self,
        documents: List[ProcessedDocument],
        collection: str = "default"
    ) -> List[str]:
        """Add multiple documents to the vector store."""
        document_ids = []
        for doc in documents:
            doc_id = self.add_document(doc, collection)
            document_ids.append(doc_id)
        return document_ids

    def search(
        self,
        query: str,
        limit: int = 10,
        score_threshold: float = 0.0,
        collection: Optional[str] = None,
        doc_type: Optional[str] = None,
        document_id: Optional[str] = None
    ) -> List[SearchResult]:
        """Search for similar content."""
        # Encode query
        query_embedding = self.embedding_service.encode_query(query)

        # Build filter
        filter_conditions = []
        if collection:
            filter_conditions.append(
                FieldCondition(
                    key="collection",
                    match=MatchValue(value=collection)
                )
            )
        if doc_type:
            filter_conditions.append(
                FieldCondition(
                    key="doc_type",
                    match=MatchValue(value=doc_type)
                )
            )
        if document_id:
            filter_conditions.append(
                FieldCondition(
                    key="document_id",
                    match=MatchValue(value=document_id)
                )
            )

        query_filter = Filter(must=filter_conditions) if filter_conditions else None

        # Search using query_points (qdrant-client 1.16+)
        results = self.client.query_points(
            collection_name=self.collection_name,
            query=query_embedding.tolist(),
            query_filter=query_filter,
            limit=limit,
            score_threshold=score_threshold,
            search_params=SearchParams(
                hnsw_ef=128,  # Higher = more accurate but slower
                exact=False
            ),
            with_payload=True
        )

        return [
            SearchResult(
                chunk_id=str(r.id),
                document_id=r.payload.get("document_id", ""),
                content=r.payload.get("content", ""),
                score=r.score,
                metadata={
                    "title": r.payload.get("title"),
                    "doc_type": r.payload.get("doc_type"),
                    "source_path": r.payload.get("source_path"),
                    "source_url": r.payload.get("source_url"),
                    "chunk_index": r.payload.get("chunk_index"),
                    **r.payload.get("metadata", {})
                }
            )
            for r in results.points
        ]

    def get_collection_info(self) -> Dict[str, Any]:
        """Get information about the collection."""
        try:
            info = self.client.get_collection(self.collection_name)
            return {
                "name": self.collection_name,
                "vectors_count": info.vectors_count,
                "points_count": info.points_count,
                "status": info.status.value,
                "config": {
                    "vector_size": info.config.params.vectors.size,
                    "distance": info.config.params.vectors.distance.value
                }
            }
        except Exception as e:
            return {"error": str(e)}

    def delete_document(self, document_id: str) -> bool:
        """Delete all chunks for a document."""
        self.client.delete(
            collection_name=self.collection_name,
            points_selector=qdrant_models.FilterSelector(
                filter=Filter(
                    must=[
                        FieldCondition(
                            key="document_id",
                            match=MatchValue(value=document_id)
                        )
                    ]
                )
            )
        )
        console.print(f"[yellow]Deleted document {document_id}[/yellow]")
        return True

    def list_collections(self) -> List[str]:
        """List all unique collection names in the store."""
        # This scrolls through to find unique collection values
        result = self.client.scroll(
            collection_name=self.collection_name,
            limit=1000,
            with_payload=["collection"]
        )

        collections = set()
        for point in result[0]:
            if point.payload and "collection" in point.payload:
                collections.add(point.payload["collection"])

        return list(collections)
