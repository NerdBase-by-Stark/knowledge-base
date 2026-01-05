#!/bin/bash
# Quick progress checker for ongoing ingestion

LOG="$HOME/ai/knowledge-base/scrape-jobs/qsys-lua/ingest.log"

echo "=== Ingestion Progress ==="
if pgrep -f "ingest_v2" > /dev/null; then
    echo "Status: RUNNING"
else
    echo "Status: COMPLETED (or stopped)"
fi

echo ""
echo "=== Recent Log ==="
tail -20 "$LOG" 2>/dev/null || echo "No log file found"

echo ""
echo "=== Stats ==="
grep -E "(✓|✗|chunks|entities)" "$LOG" 2>/dev/null | tail -10
