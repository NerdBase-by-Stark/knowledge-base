#!/usr/bin/env python3
"""
Store extracted KB data to ChromaDB and Neo4j.
Run after v3 ingestion completes.
"""

import json
import argparse
from pathlib import Path

def store_chromadb(input_dir: Path, collection_name: str):
    """Store chunks to ChromaDB."""
    import chromadb

    chunks_file = input_dir / "chunks_for_chromadb.json"
    if not chunks_file.exists():
        print(f"ERROR: {chunks_file} not found")
        return

    chunks = json.load(open(chunks_file))
    print(f"Loading {len(chunks)} chunks to ChromaDB...")

    client = chromadb.PersistentClient(path=str(input_dir.parent / "chromadb"))
    collection = client.get_or_create_collection(collection_name)

    for chunk in chunks:
        entity_names = [e["name"] for e in chunk.get("entities", [])]
        collection.upsert(
            ids=[chunk["chunk_id"]],
            documents=[chunk["content"]],
            metadatas=[{
                "doc_name": chunk.get("doc_name", ""),
                "entities": ",".join(entity_names[:20]),
                "headers": ",".join(chunk.get("headers", [])[:5]),
            }]
        )

    print(f"ChromaDB: Stored {len(chunks)} chunks in collection '{collection_name}'")


def store_neo4j(input_dir: Path):
    """Store graph to Neo4j."""
    from neo4j import GraphDatabase

    graph_file = input_dir / "graph_for_neo4j.json"
    if not graph_file.exists():
        print(f"ERROR: {graph_file} not found")
        return

    data = json.load(open(graph_file))
    nodes = data.get("nodes", [])
    edges = data.get("edges", [])

    print(f"Loading {len(nodes)} nodes and {len(edges)} edges to Neo4j...")

    driver = GraphDatabase.driver(
        "bolt://localhost:7687",
        auth=("neo4j", "agentmemory123")
    )

    with driver.session() as session:
        # Clear existing KB entities (optional)
        # session.run("MATCH (e:KBEntity) DETACH DELETE e")

        # Create nodes
        for node in nodes:
            session.run("""
                MERGE (e:KBEntity {name: $name})
                SET e.type = $type,
                    e.confidence = $confidence
            """, name=node["name"], type=node["type"],
                 confidence=node.get("confidence", 1.0))

        # Create edges
        for edge in edges:
            session.run("""
                MATCH (a:KBEntity {name: $source})
                MATCH (b:KBEntity {name: $target})
                MERGE (a)-[r:KB_RELATION {type: $type}]->(b)
                SET r.confidence = $confidence
            """, source=edge["source"], target=edge["target"],
                 type=edge["type"], confidence=edge.get("confidence", 0.7))

    driver.close()
    print(f"Neo4j: Stored {len(nodes)} entities and {len(edges)} relationships")


def main():
    parser = argparse.ArgumentParser(description="Store KB to databases")
    parser.add_argument("--input", required=True, help="Input directory with JSON files")
    parser.add_argument("--collection", default="qsys-lua-v3", help="ChromaDB collection")
    parser.add_argument("--skip-chromadb", action="store_true")
    parser.add_argument("--skip-neo4j", action="store_true")

    args = parser.parse_args()
    input_dir = Path(args.input)

    if not args.skip_chromadb:
        store_chromadb(input_dir, args.collection)

    if not args.skip_neo4j:
        store_neo4j(input_dir)

    print("\nDone! Access your KB:")
    print(f"  ChromaDB: ~/ai/knowledge-base/chromadb/{args.collection}")
    print(f"  Neo4j: http://localhost:7474 (neo4j/agentmemory123)")


if __name__ == "__main__":
    main()
