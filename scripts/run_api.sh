#!/bin/bash
# Run the Knowledge Base API server

cd "$(dirname "$0")/.."

echo "Starting Knowledge Base API on http://localhost:8000"
echo "API Docs: http://localhost:8000/docs"
echo ""

python -m uvicorn src.api:app --host 0.0.0.0 --port 8000 --reload
