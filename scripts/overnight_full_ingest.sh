#!/bin/bash
#
# OVERNIGHT FULL RAG INGESTION
# Runs the best model multiple passes over all 60 Lua docs
#
# Usage: nohup ./overnight_full_ingest.sh > ~/overnight_ingest.log 2>&1 &
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
KB_DIR="$HOME/ai/knowledge-base"
DOCS_DIR="$KB_DIR/scrape-jobs/qsys-lua/markdown"
SHOOTOUT_DIR="$KB_DIR/scrape-jobs/qsys-lua/shootout"

# Read winner from shootout results
if [ -f "$SHOOTOUT_DIR/WINNER.txt" ]; then
    BEST_MODEL=$(cat "$SHOOTOUT_DIR/WINNER.txt")
else
    echo "ERROR: No winner file found. Run analyze_shootout.py first!"
    echo "Defaulting to llama3.1:8b"
    BEST_MODEL="llama3.1:8b"
fi

echo "========================================"
echo "OVERNIGHT FULL RAG INGESTION"
echo "========================================"
echo "Start time: $(date)"
echo "Best model: $BEST_MODEL"
echo "Docs dir: $DOCS_DIR"
echo ""

# Count docs
DOC_COUNT=$(find "$DOCS_DIR" -name "*.md" | wc -l)
echo "Documents to process: $DOC_COUNT"
echo ""

# Create new collection for full ingest
COLLECTION="qsys-lua-full-$(date +%Y%m%d)"
echo "Collection: $COLLECTION"
echo ""

# Run ingestion using v2 pipeline
cd "$KB_DIR"

echo "========================================"
echo "PASS 1: Full ingestion with LLM enrichment"
echo "========================================"
echo "Started: $(date)"

python -m src.ingest_v2 ingest "$DOCS_DIR" \
    --collection "$COLLECTION" \
    --model "$BEST_MODEL" \
    --ext md \
    2>&1 | tee -a ~/overnight_pass1.log

echo ""
echo "Pass 1 completed: $(date)"
echo ""

# Check results
echo "========================================"
echo "PASS 1 RESULTS"
echo "========================================"
curl -s "http://localhost:6333/collections/knowledge_base" | python3 -c "
import sys,json
d=json.load(sys.stdin)
print(f'Total vectors: {d[\"result\"][\"points_count\"]}')"

# Run graph linking
echo ""
echo "========================================"
echo "LINKING GRAPH"
echo "========================================"
python scripts/link_graph.py 2>&1 | tee -a ~/overnight_graph.log

echo ""
echo "Graph linking completed: $(date)"

# Validation pass - spot check random chunks
echo ""
echo "========================================"
echo "VALIDATION: Testing retrieval"
echo "========================================"

TEST_QUERIES=(
    "Timer.CallAfter parameters"
    "TcpSocket connection example"
    "HttpClient POST request"
    "UCI control scripting"
    "Lua string manipulation"
)

for query in "${TEST_QUERIES[@]}"; do
    echo ""
    echo "Query: $query"
    "$KB_DIR/scripts/kb-search.sh" "$query" "$COLLECTION" 2 2>/dev/null | head -20
done

echo ""
echo "========================================"
echo "OVERNIGHT INGESTION COMPLETE"
echo "========================================"
echo "Finished: $(date)"
echo "Collection: $COLLECTION"
echo "Model used: $BEST_MODEL"
echo ""
echo "Next steps:"
echo "1. Review ~/overnight_*.log files"
echo "2. Test search quality: ~/ai/knowledge-base/scripts/kb-search.sh \"query\" $COLLECTION"
echo "3. Check Neo4j graph: http://localhost:7474"
