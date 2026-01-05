#!/bin/bash
#===============================================================================
# Q-SYS FULL EXTRACTION - Multi-Model Voting Pipeline
# Source: Firecrawl extraction from help.qsys.com (all pages)
# Output: ~/ai/knowledge-base/qsys-full-extract/
#===============================================================================

BASE="$HOME/ai/knowledge-base"
SCRIPT="$BASE/scripts/kb_ingest_v3.py"
SOURCE="/home/spark-bitch/gemini-cli/qsys_extraction_plan"
OUTPUT="$BASE/qsys-full-extract"
COLLECTION="qsys-full"
LOG="$OUTPUT/run_$(date +%Y%m%d_%H%M%S).log"

mkdir -p "$OUTPUT"

# Count docs
DOC_COUNT=$(find "$SOURCE" -name "*.md" | wc -l)

echo "=============================================="
echo "Q-SYS FULL EXTRACTION"
echo "=============================================="
echo "Source: $SOURCE"
echo "Docs: $DOC_COUNT markdown files"
echo "Output: $OUTPUT"
echo "Collection: $COLLECTION"
echo "Log: $LOG"
echo ""
echo "Pipeline features:"
echo "  - 3-model voting (LLAMA + QWEN + Mistral)"
echo "  - Source text validation"
echo "  - Rule-based type inference"
echo "  - Relationship extraction"
echo "  - Structure-aware extraction"
echo ""
echo "Estimated time: ~$(( DOC_COUNT / 10 )) hours"
echo "=============================================="
echo ""

read -p "Start extraction? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Aborted."
    exit 1
fi

# Run pipeline
echo "Starting pipeline..."
nohup python3 "$SCRIPT" \
    --source "$SOURCE" \
    --output "$OUTPUT" \
    --collection "$COLLECTION" \
    --timeout 180 \
    > "$LOG" 2>&1 &

PID=$!
echo $PID > "$OUTPUT/pipeline.pid"

echo ""
echo "Started with PID $PID"
echo ""
echo "=== MONITORING COMMANDS ==="
echo "Progress:  tail -f $OUTPUT/pipeline.log"
echo "Full log:  tail -f $LOG"
echo "Status:    ps -p $PID"
echo "Kill:      kill $PID"
echo ""
echo "=== OUTPUT FILES (when done) ==="
echo "Entities:      $OUTPUT/final.json"
echo "For ChromaDB:  $OUTPUT/chunks_for_chromadb.json"
echo "For Neo4j:     $OUTPUT/graph_for_neo4j.json"
echo ""
echo "=== AFTER COMPLETION ==="
echo "Run storage:   python3 $BASE/scripts/store_to_dbs.py --input $OUTPUT --collection $COLLECTION"
