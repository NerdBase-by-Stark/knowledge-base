#!/usr/bin/env python3
"""
Batch ingestion script for parallel processing with subagents.
"""
import sys
import json
from pathlib import Path
from typing import List

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ingest import EnhancedIngestionPipeline, IngestResult


def ingest_batch(
    files: List[str],
    collection: str = "qsys",
    mode: str = "quick"
) -> dict:
    """Ingest a batch of files and return summary."""
    pipeline = EnhancedIngestionPipeline(
        collection=collection,
        mode=mode,
        use_contextual=False,
        extract_entities=False  # Disable for speed
    )

    results = []
    for file_path in files:
        try:
            result = pipeline.ingest_file(Path(file_path))
            results.append({
                "file": file_path,
                "status": result.status,
                "chunks": result.chunk_count,
                "time": result.processing_time
            })
        except Exception as e:
            results.append({
                "file": file_path,
                "status": "error",
                "error": str(e)
            })

    success = sum(1 for r in results if r.get("status") == "success")
    total_chunks = sum(r.get("chunks", 0) for r in results)

    return {
        "total_files": len(files),
        "success": success,
        "failed": len(files) - success,
        "total_chunks": total_chunks,
        "results": results
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--files", type=str, help="Comma-separated file paths or JSON file")
    parser.add_argument("--collection", type=str, default="qsys")
    parser.add_argument("--mode", type=str, default="quick")
    args = parser.parse_args()

    if args.files.endswith(".json"):
        with open(args.files) as f:
            files = json.load(f)
    else:
        files = args.files.split(",")

    result = ingest_batch(files, args.collection, args.mode)
    print(json.dumps(result, indent=2))
