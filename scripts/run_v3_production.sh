#!/bin/bash
#===============================================================================
# V3 PRODUCTION RUNNER
# Run this without Claude - fully autonomous
#===============================================================================

BASE="$HOME/ai/knowledge-base"
SCRIPT="$BASE/scripts/kb_ingest_v3.py"
SOURCE="$BASE/scrape-jobs/qsys-lua/markdown"  # All 60 docs
OUTPUT="$BASE/v3-production-output"
LOG="$OUTPUT/run_$(date +%Y%m%d_%H%M%S).log"

mkdir -p "$OUTPUT"

echo "=============================================="
echo "V3 PRODUCTION INGESTION"
echo "=============================================="
echo "Source: $SOURCE ($(ls $SOURCE/*.md | wc -l) docs)"
echo "Output: $OUTPUT"
echo "Log: $LOG"
echo ""
echo "Estimated time: 3-5 hours for 60 docs"
echo "=============================================="
echo ""

# Run
nohup python3 "$SCRIPT" \
    --source "$SOURCE" \
    --output "$OUTPUT" \
    --collection "qsys-lua-v3-production" \
    --timeout 180 \
    > "$LOG" 2>&1 &

PID=$!
echo $PID > "$OUTPUT/pipeline.pid"

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
echo "Run storage:   python3 $BASE/scripts/store_to_dbs.py --input $OUTPUT"
