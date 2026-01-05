#!/bin/bash
#===============================================================================
# Q-SYS FULL EXTRACTION - Auto-run (no prompts)
# Run this when firecrawl extraction is complete
#===============================================================================

BASE="$HOME/ai/knowledge-base"
SCRIPT="$BASE/scripts/kb_ingest_v3.py"
SOURCE="/home/spark-bitch/gemini-cli/qsys_extraction_plan"
OUTPUT="$BASE/qsys-full-extract"
COLLECTION="qsys-full"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG="$OUTPUT/run_${TIMESTAMP}.log"

mkdir -p "$OUTPUT"

DOC_COUNT=$(find "$SOURCE" -name "*.md" 2>/dev/null | wc -l)

echo "[$(date)] Starting Q-SYS full extraction"
echo "  Source: $SOURCE"
echo "  Docs: $DOC_COUNT"
echo "  Output: $OUTPUT"
echo "  Log: $LOG"
echo "  Resume: $1"

# Run pipeline
python3 "$SCRIPT" \
    --source "$SOURCE" \
    --output "$OUTPUT" \
    --collection "$COLLECTION" \
    --timeout 180 \
    $1 \
    2>&1 | tee "$LOG"

EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then
    echo ""
    echo "[$(date)] Extraction complete. Running storage..."

    python3 "$BASE/scripts/store_to_dbs.py" \
        --input "$OUTPUT" \
        --collection "$COLLECTION"

    echo ""
    echo "[$(date)] All done!"
    echo "  ChromaDB collection: $COLLECTION"
    echo "  Neo4j: http://100.117.79.7:7474"
else
    echo ""
    echo "[$(date)] Extraction failed with exit code $EXIT_CODE"
    echo "Check log: $LOG"
fi
