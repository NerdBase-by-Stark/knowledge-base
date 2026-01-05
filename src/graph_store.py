"""Knowledge graph store using Neo4j."""
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import re

from neo4j import GraphDatabase
from rich.console import Console

from .config import settings
from .document_processor import ProcessedDocument

console = Console()


@dataclass
class Entity:
    """A knowledge graph entity."""
    name: str
    entity_type: str
    properties: Dict[str, Any] = None

    def __post_init__(self):
        if self.properties is None:
            self.properties = {}


@dataclass
class Relationship:
    """A relationship between entities."""
    source: str
    target: str
    relationship_type: str
    properties: Dict[str, Any] = None

    def __post_init__(self):
        if self.properties is None:
            self.properties = {}


class GraphStore:
    """Knowledge graph store using Neo4j."""

    def __init__(self):
        self._driver = None

    @property
    def driver(self):
        """Lazy load Neo4j driver."""
        if self._driver is None:
            self._driver = GraphDatabase.driver(
                settings.neo4j_uri,
                auth=(settings.neo4j_user, settings.neo4j_password)
            )
            # Verify connection
            self._driver.verify_connectivity()
            console.print("[green]Connected to Neo4j[/green]")
        return self._driver

    def close(self):
        """Close the driver."""
        if self._driver:
            self._driver.close()
            self._driver = None

    def setup_schema(self):
        """Create indexes and constraints."""
        with self.driver.session() as session:
            # Create constraints for unique entities
            session.run("""
                CREATE CONSTRAINT entity_name IF NOT EXISTS
                FOR (e:Entity) REQUIRE e.name IS UNIQUE
            """)

            # Create constraint for documents
            session.run("""
                CREATE CONSTRAINT document_id IF NOT EXISTS
                FOR (d:Document) REQUIRE d.document_id IS UNIQUE
            """)

            # Create indexes for common queries
            session.run("""
                CREATE INDEX entity_type IF NOT EXISTS
                FOR (e:Entity) ON (e.type)
            """)

            session.run("""
                CREATE INDEX chunk_document IF NOT EXISTS
                FOR (c:Chunk) ON (c.document_id)
            """)

        console.print("[green]Graph schema setup complete[/green]")

    def add_document_to_graph(
        self,
        document: ProcessedDocument,
        document_id: str,
        collection: str = "default"
    ):
        """Add a document and its chunks to the graph."""
        with self.driver.session() as session:
            # Create document node
            session.run("""
                MERGE (d:Document {document_id: $doc_id})
                SET d.title = $title,
                    d.doc_type = $doc_type,
                    d.source_path = $source_path,
                    d.source_url = $source_url,
                    d.collection = $collection,
                    d.content_hash = $content_hash
            """, {
                "doc_id": document_id,
                "title": document.title,
                "doc_type": document.doc_type,
                "source_path": document.source_path,
                "source_url": document.source_url,
                "collection": collection,
                "content_hash": document.content_hash
            })

            # Create chunk nodes and link to document
            for chunk in document.chunks:
                session.run("""
                    MATCH (d:Document {document_id: $doc_id})
                    CREATE (c:Chunk {
                        document_id: $doc_id,
                        chunk_index: $chunk_index,
                        content: $content
                    })
                    CREATE (d)-[:HAS_CHUNK]->(c)
                """, {
                    "doc_id": document_id,
                    "chunk_index": chunk.chunk_index,
                    "content": chunk.content[:500]  # Store preview only
                })

        console.print(f"[green]Added document '{document.title}' to graph[/green]")

    def add_entity(self, entity: Entity) -> str:
        """Add an entity to the graph."""
        with self.driver.session() as session:
            result = session.run("""
                MERGE (e:Entity {name: $name})
                SET e.type = $type,
                    e += $properties
                RETURN e.name as name
            """, {
                "name": entity.name,
                "type": entity.entity_type,
                "properties": entity.properties
            })
            record = result.single()
            return record["name"] if record else None

    def add_relationship(self, relationship: Relationship):
        """Add a relationship between entities."""
        # Sanitize relationship type for Cypher
        rel_type = re.sub(r'[^a-zA-Z0-9_]', '_', relationship.relationship_type.upper())

        with self.driver.session() as session:
            session.run(f"""
                MATCH (s:Entity {{name: $source}})
                MATCH (t:Entity {{name: $target}})
                MERGE (s)-[r:{rel_type}]->(t)
                SET r += $properties
            """, {
                "source": relationship.source,
                "target": relationship.target,
                "properties": relationship.properties
            })

    def link_entity_to_chunk(
        self,
        entity_name: str,
        document_id: str,
        chunk_index: int
    ):
        """Link an entity to a specific chunk."""
        with self.driver.session() as session:
            session.run("""
                MATCH (e:Entity {name: $entity_name})
                MATCH (c:Chunk {document_id: $doc_id, chunk_index: $chunk_index})
                MERGE (c)-[:MENTIONS]->(e)
            """, {
                "entity_name": entity_name,
                "doc_id": document_id,
                "chunk_index": chunk_index
            })

    def find_related_entities(
        self,
        entity_name: str,
        max_depth: int = 2,
        limit: int = 50
    ) -> List[Dict[str, Any]]:
        """Find entities related to a given entity."""
        with self.driver.session() as session:
            result = session.run("""
                MATCH (start:Entity {name: $name})
                MATCH path = (start)-[*1..$depth]-(related:Entity)
                WHERE start <> related
                WITH DISTINCT related, length(path) as distance
                RETURN related.name as name, related.type as type, distance
                ORDER BY distance, name
                LIMIT $limit
            """, {
                "name": entity_name,
                "depth": max_depth,
                "limit": limit
            })

            return [dict(record) for record in result]

    def find_documents_for_entity(
        self,
        entity_name: str,
        limit: int = 20
    ) -> List[Dict[str, Any]]:
        """Find documents that mention an entity."""
        with self.driver.session() as session:
            result = session.run("""
                MATCH (e:Entity {name: $name})<-[:MENTIONS]-(c:Chunk)<-[:HAS_CHUNK]-(d:Document)
                RETURN DISTINCT d.document_id as document_id,
                       d.title as title,
                       d.doc_type as doc_type,
                       count(c) as mention_count
                ORDER BY mention_count DESC
                LIMIT $limit
            """, {
                "name": entity_name,
                "limit": limit
            })

            return [dict(record) for record in result]

    def get_graph_stats(self) -> Dict[str, Any]:
        """Get statistics about the knowledge graph."""
        with self.driver.session() as session:
            result = session.run("""
                MATCH (d:Document) WITH count(d) as docs
                MATCH (c:Chunk) WITH docs, count(c) as chunks
                MATCH (e:Entity) WITH docs, chunks, count(e) as entities
                MATCH ()-[r]->() WITH docs, chunks, entities, count(r) as relationships
                RETURN docs, chunks, entities, relationships
            """)
            record = result.single()

            if record:
                return {
                    "documents": record["docs"],
                    "chunks": record["chunks"],
                    "entities": record["entities"],
                    "relationships": record["relationships"]
                }
            return {}

    def search_entities(
        self,
        query: str,
        entity_type: Optional[str] = None,
        limit: int = 20
    ) -> List[Dict[str, Any]]:
        """Search for entities by name."""
        with self.driver.session() as session:
            if entity_type:
                result = session.run("""
                    MATCH (e:Entity)
                    WHERE e.type = $type AND e.name CONTAINS $query
                    RETURN e.name as name, e.type as type, e as entity
                    LIMIT $limit
                """, {"query": query, "type": entity_type, "limit": limit})
            else:
                result = session.run("""
                    MATCH (e:Entity)
                    WHERE e.name CONTAINS $query
                    RETURN e.name as name, e.type as type, e as entity
                    LIMIT $limit
                """, {"query": query, "limit": limit})

            return [dict(record) for record in result]

    def get_entity_graph(
        self,
        entity_name: str,
        depth: int = 1
    ) -> Dict[str, Any]:
        """Get a subgraph around an entity for visualization."""
        with self.driver.session() as session:
            result = session.run("""
                MATCH (start:Entity {name: $name})
                OPTIONAL MATCH path = (start)-[r*1..$depth]-(related)
                WHERE related:Entity OR related:Document
                WITH start, relationships(path) as rels, nodes(path) as nodes
                RETURN start, collect(DISTINCT nodes) as all_nodes,
                       collect(DISTINCT rels) as all_relationships
            """, {"name": entity_name, "depth": depth})

            record = result.single()
            if not record:
                return {"nodes": [], "edges": []}

            # Process nodes and edges for visualization
            nodes = []
            edges = []
            seen_nodes = set()

            start_node = record["start"]
            if start_node:
                nodes.append({
                    "id": start_node["name"],
                    "label": start_node["name"],
                    "type": start_node.get("type", "Entity")
                })
                seen_nodes.add(start_node["name"])

            for node_list in record["all_nodes"]:
                if node_list:
                    for node in node_list:
                        if node and hasattr(node, "get"):
                            node_id = node.get("name") or node.get("document_id")
                            if node_id and node_id not in seen_nodes:
                                nodes.append({
                                    "id": node_id,
                                    "label": node.get("name") or node.get("title", node_id),
                                    "type": node.get("type", "Unknown")
                                })
                                seen_nodes.add(node_id)

            for rel_list in record["all_relationships"]:
                if rel_list:
                    for rel in rel_list:
                        if rel:
                            edges.append({
                                "source": rel.start_node.get("name") or rel.start_node.get("document_id"),
                                "target": rel.end_node.get("name") or rel.end_node.get("document_id"),
                                "type": rel.type
                            })

            return {"nodes": nodes, "edges": edges}
