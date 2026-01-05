"""FastAPI REST API for the Knowledge Base."""
from typing import List, Optional, Dict, Any
from pathlib import Path
import uuid

from fastapi import FastAPI, HTTPException, UploadFile, File, Form, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import tempfile
import shutil

from .knowledge_base import KnowledgeBase
from .config import settings

# Initialize FastAPI app
app = FastAPI(
    title="Knowledge Base API",
    description="Production-ready knowledge base with vector search and knowledge graph",
    version="0.1.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global knowledge base instance
kb: Optional[KnowledgeBase] = None


# Request/Response models
class SearchRequest(BaseModel):
    query: str
    limit: int = 10
    collection: Optional[str] = None
    mode: str = "vector"  # 'vector', 'graph', 'hybrid'


class SearchResult(BaseModel):
    content: str
    score: float
    source: str
    document_title: str
    document_id: str
    chunk_index: Optional[int] = None
    metadata: Optional[Dict[str, Any]] = None


class IngestTextRequest(BaseModel):
    text: str
    title: str = "Untitled"
    collection: str = "default"
    source_url: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class IngestDirectoryRequest(BaseModel):
    directory: str
    collection: str = "default"
    recursive: bool = True
    extensions: Optional[List[str]] = None


class EntityRequest(BaseModel):
    name: str
    entity_type: str
    properties: Optional[Dict[str, Any]] = None


class RelationshipRequest(BaseModel):
    source: str
    target: str
    relationship_type: str
    properties: Optional[Dict[str, Any]] = None


class StatsResponse(BaseModel):
    vector_store: Dict[str, Any]
    graph_store: Dict[str, Any]
    collection_name: str


# Startup event
@app.on_event("startup")
async def startup_event():
    global kb
    kb = KnowledgeBase()
    kb.initialize(recreate=False)


# Health check
@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "knowledge-base"}


# Search endpoints
@app.post("/search", response_model=List[SearchResult])
async def search(request: SearchRequest):
    """Search the knowledge base."""
    if kb is None:
        raise HTTPException(status_code=503, detail="Knowledge base not initialized")

    results = kb.search(
        query=request.query,
        limit=request.limit,
        collection=request.collection,
        mode=request.mode
    )

    return [
        SearchResult(
            content=r.content,
            score=r.score,
            source=r.source,
            document_title=r.document_title,
            document_id=r.document_id,
            chunk_index=r.chunk_index,
            metadata=r.metadata
        )
        for r in results
    ]


# Ingestion endpoints
@app.post("/ingest/text")
async def ingest_text(request: IngestTextRequest):
    """Ingest raw text content."""
    if kb is None:
        raise HTTPException(status_code=503, detail="Knowledge base not initialized")

    doc_id = kb.ingest_text(
        text=request.text,
        title=request.title,
        collection=request.collection,
        source_url=request.source_url,
        metadata=request.metadata
    )

    return {"document_id": doc_id, "status": "ingested"}


@app.post("/ingest/file")
async def ingest_file(
    file: UploadFile = File(...),
    collection: str = Form(default="default")
):
    """Ingest a file upload."""
    if kb is None:
        raise HTTPException(status_code=503, detail="Knowledge base not initialized")

    # Save to temp file
    suffix = Path(file.filename).suffix
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    try:
        doc_id = kb.ingest_file(tmp_path, collection=collection)
        return {"document_id": doc_id, "filename": file.filename, "status": "ingested"}
    finally:
        Path(tmp_path).unlink(missing_ok=True)


@app.post("/ingest/directory")
async def ingest_directory(
    request: IngestDirectoryRequest,
    background_tasks: BackgroundTasks
):
    """Ingest all documents in a directory (runs in background)."""
    if kb is None:
        raise HTTPException(status_code=503, detail="Knowledge base not initialized")

    directory = Path(request.directory)
    if not directory.is_dir():
        raise HTTPException(status_code=400, detail=f"Not a directory: {request.directory}")

    # Run ingestion in background
    task_id = str(uuid.uuid4())

    def ingest_task():
        kb.ingest_directory(
            directory=request.directory,
            collection=request.collection,
            recursive=request.recursive,
            extensions=request.extensions
        )

    background_tasks.add_task(ingest_task)

    return {
        "task_id": task_id,
        "status": "started",
        "directory": request.directory,
        "collection": request.collection
    }


# Entity endpoints
@app.post("/entities")
async def add_entity(request: EntityRequest):
    """Add an entity to the knowledge graph."""
    if kb is None:
        raise HTTPException(status_code=503, detail="Knowledge base not initialized")

    entity_name = kb.add_entity(
        name=request.name,
        entity_type=request.entity_type,
        properties=request.properties
    )

    return {"entity": entity_name, "status": "created"}


@app.post("/relationships")
async def add_relationship(request: RelationshipRequest):
    """Add a relationship between entities."""
    if kb is None:
        raise HTTPException(status_code=503, detail="Knowledge base not initialized")

    kb.add_relationship(
        source=request.source,
        target=request.target,
        relationship_type=request.relationship_type,
        properties=request.properties
    )

    return {"status": "created"}


@app.get("/entities/{entity_name}/context")
async def get_entity_context(entity_name: str, depth: int = 2):
    """Get rich context about an entity."""
    if kb is None:
        raise HTTPException(status_code=503, detail="Knowledge base not initialized")

    context = kb.get_entity_context(entity_name, depth=depth)
    return context


# Stats endpoints
@app.get("/stats", response_model=StatsResponse)
async def get_stats():
    """Get knowledge base statistics."""
    if kb is None:
        raise HTTPException(status_code=503, detail="Knowledge base not initialized")

    return kb.get_stats()


@app.get("/collections")
async def list_collections():
    """List all collections."""
    if kb is None:
        raise HTTPException(status_code=503, detail="Knowledge base not initialized")

    collections = kb.vector_store.list_collections()
    return {"collections": collections}


# Document management
@app.delete("/documents/{document_id}")
async def delete_document(document_id: str):
    """Delete a document from the knowledge base."""
    if kb is None:
        raise HTTPException(status_code=503, detail="Knowledge base not initialized")

    kb.delete_document(document_id)
    return {"document_id": document_id, "status": "deleted"}


def run_server():
    """Run the API server."""
    import uvicorn
    uvicorn.run(
        "src.api:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=True
    )


if __name__ == "__main__":
    run_server()
