#!/usr/bin/env python3
"""
Post-process: Link Qdrant vectors to Neo4j knowledge graph.

Reads vector metadata from Qdrant and creates:
- Document nodes
- Chunk nodes (linked to documents)
- Entity-to-Chunk MENTIONS relationships
"""
import sys
sys.path.insert(0, '/home/spark-bitch/ai/knowledge-base')

from collections import defaultdict
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
from qdrant_client import QdrantClient
from neo4j import GraphDatabase

console = Console()

# Config
QDRANT_HOST = "localhost"
QDRANT_PORT = 6333
COLLECTION = "knowledge_base"

NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "agentmemory123"


def get_all_vectors():
    """Fetch all vectors from Qdrant with metadata."""
    client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)

    # Get collection info
    info = client.get_collection(COLLECTION)
    total = info.points_count
    console.print(f"[cyan]Found {total} vectors in Qdrant[/cyan]")

    # Scroll through all points
    all_points = []
    offset = None

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        console=console
    ) as progress:
        task = progress.add_task("Fetching vectors...", total=total)

        while True:
            results, offset = client.scroll(
                collection_name=COLLECTION,
                limit=100,
                offset=offset,
                with_payload=True,
                with_vectors=False  # Don't need vectors, just metadata
            )

            all_points.extend(results)
            progress.update(task, completed=len(all_points))

            if offset is None:
                break

    return all_points


def link_to_graph(points):
    """Create graph structure from vector metadata."""
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

    # Group chunks by document
    docs = defaultdict(list)
    for point in points:
        payload = point.payload
        doc_id = payload.get("document_id", "unknown")
        docs[doc_id].append(payload)

    console.print(f"[cyan]Found {len(docs)} unique documents[/cyan]")

    stats = {
        "documents": 0,
        "chunks": 0,
        "entity_links": 0,
        "entities_created": 0
    }

    with driver.session() as session:
        # Setup schema
        session.run("""
            CREATE CONSTRAINT doc_id IF NOT EXISTS
            FOR (d:Document) REQUIRE d.document_id IS UNIQUE
        """)
        session.run("""
            CREATE INDEX chunk_idx IF NOT EXISTS
            FOR (c:Chunk) ON (c.document_id, c.chunk_index)
        """)

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("{task.completed}/{task.total}"),
            console=console
        ) as progress:
            task = progress.add_task("Linking to graph...", total=len(docs))

            for doc_id, chunks in docs.items():
                # Get doc metadata from first chunk
                first = chunks[0]
                title = first.get("title", "Unknown")
                doc_type = first.get("doc_type", "markdown")
                collection = first.get("collection", "default")

                # Create document node
                session.run("""
                    MERGE (d:Document {document_id: $doc_id})
                    SET d.title = $title,
                        d.doc_type = $doc_type,
                        d.collection = $collection,
                        d.chunk_count = $chunk_count
                """, {
                    "doc_id": doc_id,
                    "title": title,
                    "doc_type": doc_type,
                    "collection": collection,
                    "chunk_count": len(chunks)
                })
                stats["documents"] += 1

                # Create chunk nodes and link entities
                for chunk in chunks:
                    chunk_idx = chunk.get("chunk_index", 0)
                    content_preview = chunk.get("content", "")[:200]
                    entities = chunk.get("entities", [])
                    keywords = chunk.get("keywords", [])

                    # Create chunk node linked to document
                    session.run("""
                        MATCH (d:Document {document_id: $doc_id})
                        MERGE (c:Chunk {document_id: $doc_id, chunk_index: $chunk_idx})
                        SET c.content_preview = $content,
                            c.keywords = $keywords
                        MERGE (d)-[:HAS_CHUNK]->(c)
                    """, {
                        "doc_id": doc_id,
                        "chunk_idx": chunk_idx,
                        "content": content_preview,
                        "keywords": keywords
                    })
                    stats["chunks"] += 1

                    # Link entities to chunk
                    for entity_name in entities:
                        if entity_name and len(entity_name) > 1:
                            # Ensure entity exists and link
                            session.run("""
                                MERGE (e:Entity {name: $name})
                                WITH e
                                MATCH (c:Chunk {document_id: $doc_id, chunk_index: $chunk_idx})
                                MERGE (c)-[:MENTIONS]->(e)
                            """, {
                                "name": entity_name,
                                "doc_id": doc_id,
                                "chunk_idx": chunk_idx
                            })
                            stats["entity_links"] += 1

                progress.advance(task)

    driver.close()
    return stats


def main():
    console.print("[bold]Post-Processing: Link Vectors to Knowledge Graph[/bold]\n")

    # Step 1: Get all vectors
    points = get_all_vectors()

    # Step 2: Link to graph
    stats = link_to_graph(points)

    # Summary
    console.print("\n[bold green]Complete![/bold green]")
    console.print(f"  Documents created: {stats['documents']}")
    console.print(f"  Chunks created: {stats['chunks']}")
    console.print(f"  Entity links created: {stats['entity_links']}")


if __name__ == "__main__":
    main()
