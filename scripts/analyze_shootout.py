#!/usr/bin/env python3
"""
Analyze RAG Shootout Results
Compares all models and produces a detailed report
"""
import json
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Optional
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

console = Console()

SHOOTOUT_DIR = Path.home() / "ai/knowledge-base/scrape-jobs/qsys-lua/shootout"

@dataclass
class ModelScore:
    model: str
    total_time_s: float
    avg_chunk_time_ms: float
    total_entities: int
    error_count: int
    total_chunks: int

    # Quality metrics
    empty_contexts: int = 0
    empty_entities: int = 0
    empty_summaries: int = 0
    empty_questions: int = 0
    avg_entity_count: float = 0
    avg_context_len: float = 0
    avg_summary_len: float = 0

    # Computed scores
    speed_score: float = 0
    quality_score: float = 0
    reliability_score: float = 0
    total_score: float = 0

def load_results() -> List[ModelScore]:
    """Load all JSON result files"""
    results = []

    for json_file in SHOOTOUT_DIR.glob("*.json"):
        if json_file.name in ["claude_prompts.json"]:
            continue

        try:
            with open(json_file) as f:
                data = json.load(f)

            # Handle Claude's different format
            if "claude" in json_file.name.lower():
                score = ModelScore(
                    model=data.get("model", "claude-opus"),
                    total_time_s=0,  # Claude doesn't track time the same way
                    avg_chunk_time_ms=0,
                    total_entities=sum(len(c.get("entities", [])) for c in data.get("chunks", [])),
                    error_count=0,
                    total_chunks=data.get("total_chunks", len(data.get("chunks", [])))
                )
            else:
                score = ModelScore(
                    model=data["model"],
                    total_time_s=data.get("total_time_s", 0),
                    avg_chunk_time_ms=data.get("avg_chunk_time_ms", 0),
                    total_entities=data.get("total_entities", 0),
                    error_count=data.get("error_count", 0),
                    total_chunks=data.get("total_chunks", 0)
                )

            # Analyze quality from chunks
            chunks = data.get("chunks", [])
            entity_counts = []
            context_lens = []
            summary_lens = []

            for c in chunks:
                # Context
                ctx = c.get("context", "")
                if not ctx or len(ctx) < 20:
                    score.empty_contexts += 1
                else:
                    context_lens.append(len(ctx))

                # Entities
                ents = c.get("entities", [])
                if not ents:
                    score.empty_entities += 1
                else:
                    entity_counts.append(len(ents))

                # Summary
                summ = c.get("summary", "")
                if not summ or len(summ) < 10:
                    score.empty_summaries += 1
                else:
                    summary_lens.append(len(summ))

                # Questions
                qs = c.get("questions", [])
                if not qs:
                    score.empty_questions += 1

            if entity_counts:
                score.avg_entity_count = sum(entity_counts) / len(entity_counts)
            if context_lens:
                score.avg_context_len = sum(context_lens) / len(context_lens)
            if summary_lens:
                score.avg_summary_len = sum(summary_lens) / len(summary_lens)

            results.append(score)

        except Exception as e:
            console.print(f"[red]Error loading {json_file}: {e}[/]")

    return results

def calculate_scores(results: List[ModelScore]) -> List[ModelScore]:
    """Calculate normalized scores for comparison"""
    if not results:
        return results

    # Filter out Claude for speed comparison (different measurement)
    timed_results = [r for r in results if r.avg_chunk_time_ms > 0]

    if timed_results:
        min_time = min(r.avg_chunk_time_ms for r in timed_results)
        max_entities = max(r.total_entities for r in results) or 1
    else:
        min_time = 1
        max_entities = max(r.total_entities for r in results) or 1

    for r in results:
        # Speed score (0-40 points) - lower time = higher score
        if r.avg_chunk_time_ms > 0:
            r.speed_score = (min_time / r.avg_chunk_time_ms) * 40
        else:
            r.speed_score = 40  # Claude gets max speed score (API is fast)

        # Quality score (0-40 points) - more entities = higher score
        r.quality_score = (r.total_entities / max_entities) * 40

        # Reliability score (0-20 points) - fewer errors/empties = higher score
        if r.total_chunks > 0:
            empty_rate = (r.empty_contexts + r.empty_entities + r.empty_summaries) / (r.total_chunks * 3)
            r.reliability_score = (1 - empty_rate) * 20
        else:
            r.reliability_score = 20

        r.total_score = r.speed_score + r.quality_score + r.reliability_score

    return sorted(results, key=lambda x: x.total_score, reverse=True)

def print_report(results: List[ModelScore]):
    """Print detailed comparison report"""
    console.print(Panel.fit("[bold]RAG SHOOTOUT ANALYSIS[/]", title="Results"))

    # Main comparison table
    table = Table(title="Overall Rankings")
    table.add_column("Rank", style="bold")
    table.add_column("Model", style="cyan")
    table.add_column("Time", justify="right")
    table.add_column("Avg/Chunk", justify="right")
    table.add_column("Entities", justify="right")
    table.add_column("Speed", justify="right")
    table.add_column("Quality", justify="right")
    table.add_column("Reliable", justify="right")
    table.add_column("TOTAL", justify="right", style="bold green")

    for i, r in enumerate(results, 1):
        time_str = f"{r.total_time_s:.1f}s" if r.total_time_s > 0 else "N/A"
        avg_str = f"{r.avg_chunk_time_ms:.0f}ms" if r.avg_chunk_time_ms > 0 else "API"

        rank_style = "bold green" if i == 1 else ("yellow" if i == 2 else "")
        table.add_row(
            f"#{i}",
            r.model,
            time_str,
            avg_str,
            str(r.total_entities),
            f"{r.speed_score:.1f}",
            f"{r.quality_score:.1f}",
            f"{r.reliability_score:.1f}",
            f"{r.total_score:.1f}",
            style=rank_style
        )

    console.print(table)

    # Quality breakdown table
    console.print()
    qual_table = Table(title="Quality Breakdown (What Each Model Missed)")
    qual_table.add_column("Model", style="cyan")
    qual_table.add_column("Empty Context", justify="right")
    qual_table.add_column("No Entities", justify="right")
    qual_table.add_column("Empty Summary", justify="right")
    qual_table.add_column("No Questions", justify="right")
    qual_table.add_column("Avg Entities/Chunk", justify="right")

    for r in results:
        qual_table.add_row(
            r.model,
            str(r.empty_contexts),
            str(r.empty_entities),
            str(r.empty_summaries),
            str(r.empty_questions),
            f"{r.avg_entity_count:.1f}"
        )

    console.print(qual_table)

    # Speed vs Quality analysis
    console.print()
    console.print("[bold]Speed vs Quality Analysis:[/]")

    timed = [r for r in results if r.avg_chunk_time_ms > 0]
    if timed:
        fastest = min(timed, key=lambda x: x.avg_chunk_time_ms)
        most_entities = max(results, key=lambda x: x.total_entities)
        best_balance = max(timed, key=lambda x: x.total_score)

        console.print(f"  Fastest: [green]{fastest.model}[/] ({fastest.avg_chunk_time_ms:.0f}ms/chunk)")
        console.print(f"  Most Entities: [green]{most_entities.model}[/] ({most_entities.total_entities} total)")
        console.print(f"  Best Balance: [green]{best_balance.model}[/] (Score: {best_balance.total_score:.1f})")

    # Winner
    winner = results[0] if results else None
    if winner:
        console.print()
        console.print(Panel.fit(
            f"[bold green]🏆 WINNER: {winner.model}[/]\n"
            f"Score: {winner.total_score:.1f}/100\n"
            f"Entities: {winner.total_entities}\n"
            f"Speed: {winner.avg_chunk_time_ms:.0f}ms/chunk" if winner.avg_chunk_time_ms > 0 else f"Entities: {winner.total_entities}",
            title="Champion"
        ))

    return winner

def save_report(results: List[ModelScore], winner: Optional[ModelScore]):
    """Save report to markdown file"""
    report_file = SHOOTOUT_DIR / "RESULTS.md"

    lines = [
        "# RAG Shootout Results",
        "",
        f"**Date**: {__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M')}",
        "",
        "## Rankings",
        "",
        "| Rank | Model | Time | Entities | Score |",
        "|------|-------|------|----------|-------|",
    ]

    for i, r in enumerate(results, 1):
        time_str = f"{r.total_time_s:.1f}s" if r.total_time_s > 0 else "API"
        lines.append(f"| {i} | {r.model} | {time_str} | {r.total_entities} | {r.total_score:.1f} |")

    lines.extend([
        "",
        "## Quality Analysis",
        "",
        "| Model | Empty Context | No Entities | Empty Summary | Avg Entities |",
        "|-------|---------------|-------------|---------------|--------------|",
    ])

    for r in results:
        lines.append(f"| {r.model} | {r.empty_contexts} | {r.empty_entities} | {r.empty_summaries} | {r.avg_entity_count:.1f} |")

    if winner:
        lines.extend([
            "",
            f"## Winner: {winner.model}",
            "",
            f"- **Total Score**: {winner.total_score:.1f}/100",
            f"- **Entities Found**: {winner.total_entities}",
            f"- **Avg Time/Chunk**: {winner.avg_chunk_time_ms:.0f}ms" if winner.avg_chunk_time_ms > 0 else "",
            "",
            "### Recommendation",
            "",
            f"Use **{winner.model}** for full RAG ingestion.",
        ])

    with open(report_file, "w") as f:
        f.write("\n".join(lines))

    console.print(f"\n[green]Report saved to:[/] {report_file}")

    # Also save winner to a simple file for scripts to read
    winner_file = SHOOTOUT_DIR / "WINNER.txt"
    if winner:
        with open(winner_file, "w") as f:
            f.write(winner.model)
        console.print(f"[green]Winner saved to:[/] {winner_file}")

def main():
    console.print("[bold]Loading benchmark results...[/]")
    results = load_results()

    if not results:
        console.print("[red]No results found! Run benchmarks first.[/]")
        return

    console.print(f"[green]Found {len(results)} model results[/]")

    results = calculate_scores(results)
    winner = print_report(results)
    save_report(results, winner)

if __name__ == "__main__":
    main()
