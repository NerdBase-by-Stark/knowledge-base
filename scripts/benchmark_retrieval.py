#!/usr/bin/env python3
"""
RAG vs Graph Benchmark with Claude vs Ollama Answer Generation
"""
import json
import time
from dataclasses import dataclass, field
from typing import List, Dict, Optional
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue
from neo4j import GraphDatabase
import ollama
from FlagEmbedding import BGEM3FlagModel

@dataclass
class RetrievalResult:
    method: str
    query: str
    latency_ms: float
    chunks: List[Dict]
    
@dataclass 
class AnswerResult:
    llm: str
    answer: str
    latency_ms: float

@dataclass
class BenchmarkResult:
    query: str
    retrieval: RetrievalResult
    claude_answer: Optional[AnswerResult] = None
    ollama_answer: Optional[AnswerResult] = None

# Test questions
QUESTIONS = [
    # Factual
    "What parameters does Timer.CallAfter accept?",
    "What is the default port for TcpSocket connections in Q-SYS?",
    # Conceptual
    "How do EventHandlers work in Q-SYS Lua?",
    "What's the difference between TcpSocket and UdpSocket in Q-SYS?",
    # Multi-hop
    "What network protocols can Q-SYS Lua scripts use for external communication?",
    "List the string manipulation functions available in Q-SYS Lua",
    # Code example
    "Show me an example of using HttpClient to make a request",
]

class Benchmark:
    def __init__(self):
        print("Initializing benchmark...")
        self.qdrant = QdrantClient(host="localhost", port=6333)
        self.neo4j = GraphDatabase.driver(
            "bolt://localhost:7687", 
            auth=("neo4j", "agentmemory123")
        )
        print("Loading embedding model...")
        self.embedder = BGEM3FlagModel("BAAI/bge-m3", use_fp16=False)
        self.ollama_model = "qwen2.5:32b"
        self.results: List[BenchmarkResult] = []
        
    def embed_query(self, query: str) -> List[float]:
        """Embed a query using BGE-M3"""
        result = self.embedder.encode([query])
        return result["dense_vecs"][0].tolist()
    
    def rag_retrieve(self, query: str, top_k: int = 5) -> RetrievalResult:
        """Pure vector search"""
        start = time.time()
        
        query_vec = self.embed_query(query)
        
        results = self.qdrant.query_points(
            collection_name="knowledge_base",
            query=query_vec,
            limit=top_k,
            query_filter=Filter(
                must=[FieldCondition(key="collection", match=MatchValue(value="qsys-lua"))]
            )
        )
        
        chunks = []
        for r in results.points:
            chunks.append({
                "content": r.payload.get("content", ""),
                "title": r.payload.get("title", ""),
                "score": r.score
            })
        
        latency = (time.time() - start) * 1000
        return RetrievalResult("RAG", query, latency, chunks)
    
    def graph_retrieve(self, query: str, top_k: int = 5) -> RetrievalResult:
        """Entity-based graph traversal"""
        start = time.time()
        
        # Extract potential entities from query (simple keyword approach)
        keywords = [w for w in query.split() if len(w) > 3 and w[0].isupper()]
        if not keywords:
            keywords = [w for w in query.split() if len(w) > 4][:3]
        
        chunks = []
        with self.neo4j.session() as session:
            for keyword in keywords[:3]:
                result = session.run("""
                    MATCH (e:Entity)
                    WHERE toLower(e.name) CONTAINS toLower($keyword)
                    MATCH (c:Chunk)-[:MENTIONS]->(e)
                    RETURN c.content as content, c.title as title, e.name as entity
                    LIMIT $limit
                """, keyword=keyword, limit=top_k)
                
                for record in result:
                    chunks.append({
                        "content": record["content"] or "",
                        "title": record["title"] or "",
                        "entity": record["entity"],
                        "score": 1.0
                    })
        
        # Deduplicate by content
        seen = set()
        unique_chunks = []
        for c in chunks:
            if c["content"] not in seen:
                seen.add(c["content"])
                unique_chunks.append(c)
        
        latency = (time.time() - start) * 1000
        return RetrievalResult("Graph", query, latency, unique_chunks[:top_k])
    
    def hybrid_retrieve(self, query: str, top_k: int = 5) -> RetrievalResult:
        """Combined RAG + Graph"""
        start = time.time()
        
        rag_results = self.rag_retrieve(query, top_k)
        graph_results = self.graph_retrieve(query, top_k)
        
        # Merge and deduplicate
        all_chunks = {}
        for c in rag_results.chunks:
            key = c["content"][:100]
            all_chunks[key] = c
            all_chunks[key]["source"] = "RAG"
            
        for c in graph_results.chunks:
            key = c["content"][:100]
            if key not in all_chunks:
                all_chunks[key] = c
                all_chunks[key]["source"] = "Graph"
            else:
                all_chunks[key]["source"] = "Both"
        
        chunks = list(all_chunks.values())[:top_k]
        latency = (time.time() - start) * 1000
        return RetrievalResult("Hybrid", query, latency, chunks)
    
    def format_context(self, chunks: List[Dict]) -> str:
        """Format chunks into context string"""
        context_parts = []
        for i, c in enumerate(chunks, 1):
            context_parts.append(f"[{i}] {c.get('title', 'Unknown')}:\n{c['content']}\n")
        return "\n".join(context_parts)
    
    def ask_ollama(self, query: str, context: str) -> AnswerResult:
        """Get answer from Ollama"""
        start = time.time()
        
        prompt = f"""Based on the following Q-SYS Lua documentation context, answer the question accurately and concisely.

Context:
{context}

Question: {query}

Answer:"""
        
        try:
            response = ollama.chat(
                model=self.ollama_model,
                messages=[{"role": "user", "content": prompt}],
                options={"temperature": 0.3, "num_predict": 500}
            )
            answer = response["message"]["content"]
        except Exception as e:
            answer = f"Error: {e}"
        
        latency = (time.time() - start) * 1000
        return AnswerResult("Ollama", answer, latency)
    
    def run_single(self, query: str, method: str = "RAG") -> BenchmarkResult:
        """Run benchmark for a single query"""
        print(f"\n{'='*60}")
        print(f"Query: {query}")
        print(f"Method: {method}")
        
        # Retrieve
        if method == "RAG":
            retrieval = self.rag_retrieve(query)
        elif method == "Graph":
            retrieval = self.graph_retrieve(query)
        else:
            retrieval = self.hybrid_retrieve(query)
        
        print(f"Retrieved {len(retrieval.chunks)} chunks in {retrieval.latency_ms:.1f}ms")
        
        if not retrieval.chunks:
            print("No chunks retrieved!")
            return BenchmarkResult(query, retrieval)
        
        context = self.format_context(retrieval.chunks)
        
        # Get Ollama answer
        print("Getting Ollama answer...")
        ollama_result = self.ask_ollama(query, context)
        print(f"Ollama: {ollama_result.latency_ms:.1f}ms")
        
        result = BenchmarkResult(
            query=query,
            retrieval=retrieval,
            ollama_answer=ollama_result
        )
        
        return result
    
    def run_all(self):
        """Run full benchmark"""
        print("\n" + "="*60)
        print("BENCHMARK: RAG vs Graph vs Hybrid + Ollama Answers")
        print("="*60)
        
        all_results = []
        
        for method in ["RAG", "Graph", "Hybrid"]:
            print(f"\n\n{'#'*60}")
            print(f"# {method} RETRIEVAL")
            print(f"{'#'*60}")
            
            for q in QUESTIONS:
                result = self.run_single(q, method)
                all_results.append(result)
                self.results.append(result)
        
        return all_results
    
    def print_summary(self):
        """Print benchmark summary"""
        print("\n\n" + "="*60)
        print("BENCHMARK SUMMARY")
        print("="*60)
        
        # Group by method
        by_method = {}
        for r in self.results:
            method = r.retrieval.method
            if method not in by_method:
                by_method[method] = []
            by_method[method].append(r)
        
        print("\n### Retrieval Latency (avg ms)")
        for method, results in by_method.items():
            avg_latency = sum(r.retrieval.latency_ms for r in results) / len(results)
            avg_chunks = sum(len(r.retrieval.chunks) for r in results) / len(results)
            print(f"  {method}: {avg_latency:.1f}ms, {avg_chunks:.1f} chunks avg")
        
        print("\n### Ollama Answer Latency (avg ms)")
        for method, results in by_method.items():
            latencies = [r.ollama_answer.latency_ms for r in results if r.ollama_answer]
            if latencies:
                avg = sum(latencies) / len(latencies)
                print(f"  {method}: {avg:.1f}ms")
        
        # Save results
        output_file = Path(__file__).parent.parent / "scrape-jobs/qsys-lua/benchmark_results.json"
        with open(output_file, "w") as f:
            results_data = []
            for r in self.results:
                results_data.append({
                    "query": r.query,
                    "method": r.retrieval.method,
                    "retrieval_latency_ms": r.retrieval.latency_ms,
                    "chunks_count": len(r.retrieval.chunks),
                    "chunks": r.retrieval.chunks[:2],  # First 2 for review
                    "ollama_latency_ms": r.ollama_answer.latency_ms if r.ollama_answer else None,
                    "ollama_answer": r.ollama_answer.answer if r.ollama_answer else None,
                })
            json.dump(results_data, f, indent=2)
        print(f"\nResults saved to: {output_file}")

if __name__ == "__main__":
    benchmark = Benchmark()
    benchmark.run_all()
    benchmark.print_summary()
