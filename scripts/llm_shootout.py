#!/usr/bin/env python3
"""
LLM Shootout: Compare answer quality across all local models
Uses same RAG context for fair comparison
"""
import json
import time
from dataclasses import dataclass
from typing import List, Dict
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue
import ollama
from FlagEmbedding import BGEM3FlagModel
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn

console = Console()

# Models to test
MODELS = [
    "qwen2.5:7b",
    "mistral:latest",
    "llama3.1:8b",
    "qwen2.5:32b",
    "deepseek-coder:33b",
]

# Test questions (Q-SYS Lua focused)
QUESTIONS = [
    "What parameters does Timer.CallAfter accept?",
    "How do EventHandlers work in Q-SYS Lua?",
    "What's the difference between TcpSocket and UdpSocket in Q-SYS?",
    "Show me an example of using HttpClient to make a POST request",
    "What network protocols can Q-SYS Lua scripts use?",
]

@dataclass
class Answer:
    model: str
    question: str
    answer: str
    latency_ms: float
    context_used: str

class LLMShootout:
    def __init__(self):
        console.print("[bold blue]Initializing LLM Shootout...[/]")
        self.qdrant = QdrantClient(host="localhost", port=6333)

        console.print("Loading BGE-M3 embeddings...")
        self.embedder = BGEM3FlagModel("BAAI/bge-m3", use_fp16=False)

        self.results: List[Answer] = []
        self.contexts: Dict[str, str] = {}  # Cache contexts per question

    def embed_query(self, query: str) -> List[float]:
        result = self.embedder.encode([query])
        return result["dense_vecs"][0].tolist()

    def retrieve_context(self, query: str, top_k: int = 5) -> str:
        """Get RAG context for a question"""
        if query in self.contexts:
            return self.contexts[query]

        query_vec = self.embed_query(query)

        results = self.qdrant.query_points(
            collection_name="knowledge_base",
            query=query_vec,
            limit=top_k,
        )

        context_parts = []
        for i, r in enumerate(results.points, 1):
            title = r.payload.get("title", "Unknown")
            content = r.payload.get("content", "")
            context_parts.append(f"[{i}] {title}:\n{content}\n")

        context = "\n".join(context_parts)
        self.contexts[query] = context
        return context

    def ask_model(self, model: str, question: str, context: str) -> Answer:
        """Get answer from a specific model"""
        prompt = f"""Based on the following Q-SYS Lua documentation, answer the question accurately and concisely.

Context:
{context}

Question: {question}

Answer:"""

        start = time.time()
        try:
            response = ollama.chat(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                options={"temperature": 0.3, "num_predict": 500}
            )
            answer = response["message"]["content"]
        except Exception as e:
            answer = f"ERROR: {e}"

        latency = (time.time() - start) * 1000

        return Answer(
            model=model,
            question=question,
            answer=answer,
            latency_ms=latency,
            context_used=context[:500] + "..."
        )

    def run_question(self, question: str):
        """Run all models on a single question"""
        console.print(f"\n[bold cyan]Question:[/] {question}")

        # Get context once
        context = self.retrieve_context(question)
        console.print(f"[dim]Retrieved {len(context)} chars of context[/]")

        for model in MODELS:
            with console.status(f"[yellow]Asking {model}...[/]"):
                result = self.ask_model(model, question, context)
                self.results.append(result)

            console.print(f"  [green]✓[/] {model}: {result.latency_ms:.0f}ms")

    def run_all(self):
        """Run full shootout"""
        console.print(Panel.fit(
            "[bold]LLM SHOOTOUT[/]\n"
            f"Models: {len(MODELS)}\n"
            f"Questions: {len(QUESTIONS)}",
            title="Benchmark"
        ))

        for q in QUESTIONS:
            self.run_question(q)

        self.print_summary()
        self.save_results()
        self.save_for_claude()

    def print_summary(self):
        """Print latency summary"""
        console.print("\n")

        # Latency table
        table = Table(title="Average Latency by Model")
        table.add_column("Model", style="cyan")
        table.add_column("Avg Latency", justify="right")
        table.add_column("Min", justify="right")
        table.add_column("Max", justify="right")

        for model in MODELS:
            model_results = [r for r in self.results if r.model == model]
            latencies = [r.latency_ms for r in model_results]
            avg = sum(latencies) / len(latencies)
            table.add_row(
                model,
                f"{avg:.0f}ms",
                f"{min(latencies):.0f}ms",
                f"{max(latencies):.0f}ms"
            )

        console.print(table)

    def save_results(self):
        """Save full results to JSON"""
        output = Path(__file__).parent.parent / "scrape-jobs/qsys-lua/llm_shootout_results.json"

        data = []
        for r in self.results:
            data.append({
                "model": r.model,
                "question": r.question,
                "answer": r.answer,
                "latency_ms": r.latency_ms,
            })

        with open(output, "w") as f:
            json.dump(data, f, indent=2)

        console.print(f"\n[green]Results saved to:[/] {output}")

    def save_for_claude(self):
        """Save context + questions for Claude comparison"""
        output = Path(__file__).parent.parent / "scrape-jobs/qsys-lua/claude_comparison.md"

        lines = ["# Claude Comparison\n"]
        lines.append("Answer these questions using ONLY the provided context.\n")

        for q in QUESTIONS:
            context = self.contexts.get(q, "")
            lines.append(f"\n## Question: {q}\n")
            lines.append(f"### Context:\n```\n{context}\n```\n")
            lines.append("### Your Answer:\n\n---\n")

        with open(output, "w") as f:
            f.write("\n".join(lines))

        console.print(f"[green]Claude comparison file:[/] {output}")

if __name__ == "__main__":
    shootout = LLMShootout()
    shootout.run_all()
