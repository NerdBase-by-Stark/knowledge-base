#!/bin/bash
# Run the Knowledge Base Dashboard

cd "$(dirname "$0")/.."

echo "Starting Knowledge Base Dashboard on http://localhost:8501"
echo ""

streamlit run src/dashboard.py --server.port 8501 --server.address 0.0.0.0
