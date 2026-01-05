#!/usr/bin/env python3
"""CLI for the Knowledge Base system."""
import typer
from pathlib import Path
from typing import Optional, List
from rich.console import Console
from rich.table import Table

app = typer.Typer(help="Knowledge Base CLI - Manage your knowledge bases")
console = Console()


@app.command()
def init(recreate: bool = typer.Option(False, "--recreate", help="Recreate collections")):
    """Initialize the knowledge base."""
    from src.knowledge_base import KnowledgeBase

    kb = KnowledgeBase()
    kb.initialize(recreate=recreate)
    console.print("[bold green]Knowledge base initialized![/bold green]")


@app.command()
def ingest(
    path: str = typer.Argument(..., help="File or directory to ingest"),
    collection: str = typer.Option("default", "--collection", "-c", help="Collection name"),
    recursive: bool = typer.Option(True, "--recursive", "-r", help="Recursive directory scan"),
    extensions: Optional[str] = typer.Option(None, "--ext", "-e", help="File extensions (comma-separated)")
):
    """Ingest documents into the knowledge base."""
    from src.knowledge_base import KnowledgeBase

    kb = KnowledgeBase()
    kb.initialize(recreate=False)

    path = Path(path)
    ext_list = extensions.split(",") if extensions else None

    if path.is_file():
        doc_id = kb.ingest_file(path, collection=collection)
        console.print(f"[green]Ingested file: {path.name} -> {doc_id}[/green]")
    elif path.is_dir():
        if path.suffix == ".md" or (ext_list and ext_list == ["md"]):
            doc_ids = kb.ingest_markdown_directory(path, collection=collection, recursive=recursive)
        else:
            doc_ids = kb.ingest_directory(
                path, collection=collection, recursive=recursive, extensions=ext_list
            )
        console.print(f"[green]Ingested {len(doc_ids)} documents[/green]")
    else:
        console.print(f"[red]Path not found: {path}[/red]")
        raise typer.Exit(1)


@app.command()
def search(
    query: str = typer.Argument(..., help="Search query"),
    limit: int = typer.Option(10, "--limit", "-n", help="Number of results"),
    collection: Optional[str] = typer.Option(None, "--collection", "-c", help="Filter by collection"),
    mode: str = typer.Option("vector", "--mode", "-m", help="Search mode: vector, graph, hybrid")
):
    """Search the knowledge base."""
    from src.knowledge_base import KnowledgeBase

    kb = KnowledgeBase()
    kb.initialize(recreate=False)

    results = kb.search(query=query, limit=limit, collection=collection, mode=mode)

    if not results:
        console.print("[yellow]No results found[/yellow]")
        return

    table = Table(title=f"Search Results for: {query}")
    table.add_column("Score", style="cyan", width=8)
    table.add_column("Title", style="green", width=30)
    table.add_column("Content", style="white", width=60)
    table.add_column("Source", style="magenta", width=10)

    for r in results:
        content = r.content[:100] + "..." if len(r.content) > 100 else r.content
        table.add_row(
            f"{r.score:.3f}",
            r.document_title[:28] + ".." if len(r.document_title) > 30 else r.document_title,
            content,
            r.source
        )

    console.print(table)


@app.command()
def stats():
    """Show knowledge base statistics."""
    from src.knowledge_base import KnowledgeBase

    kb = KnowledgeBase()
    kb.initialize(recreate=False)
    kb.print_stats()


@app.command()
def serve(
    api: bool = typer.Option(True, "--api/--no-api", help="Run API server"),
    dashboard: bool = typer.Option(False, "--dashboard", "-d", help="Run dashboard"),
    api_port: int = typer.Option(8000, "--api-port", help="API port"),
    dashboard_port: int = typer.Option(8501, "--dashboard-port", help="Dashboard port")
):
    """Start the API server and/or dashboard."""
    import subprocess
    import sys

    processes = []

    if api:
        console.print(f"[blue]Starting API on http://localhost:{api_port}[/blue]")
        p = subprocess.Popen([
            sys.executable, "-m", "uvicorn", "src.api:app",
            "--host", "0.0.0.0", "--port", str(api_port)
        ])
        processes.append(p)

    if dashboard:
        console.print(f"[blue]Starting Dashboard on http://localhost:{dashboard_port}[/blue]")
        p = subprocess.Popen([
            sys.executable, "-m", "streamlit", "run", "src/dashboard.py",
            "--server.port", str(dashboard_port),
            "--server.address", "0.0.0.0"
        ])
        processes.append(p)

    if processes:
        console.print("[green]Services started. Press Ctrl+C to stop.[/green]")
        try:
            for p in processes:
                p.wait()
        except KeyboardInterrupt:
            console.print("\n[yellow]Shutting down...[/yellow]")
            for p in processes:
                p.terminate()


@app.command()
def collections():
    """List all collections."""
    from src.knowledge_base import KnowledgeBase

    kb = KnowledgeBase()
    kb.initialize(recreate=False)

    colls = kb.vector_store.list_collections()

    if colls:
        console.print("[bold]Collections:[/bold]")
        for c in colls:
            console.print(f"  - {c}")
    else:
        console.print("[yellow]No collections found[/yellow]")


if __name__ == "__main__":
    app()
