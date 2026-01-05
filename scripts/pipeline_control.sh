#!/bin/bash
#===============================================================================
# PIPELINE CONTROL & MONITORING
#===============================================================================

BASE_DIR="$HOME/ai/knowledge-base"
LOG_DIR="$BASE_DIR/logs"
PIPELINE_SCRIPT="$BASE_DIR/scripts/production_pipeline.sh"
PID_FILE="$LOG_DIR/pipeline.pid"

usage() {
    cat << EOF
Usage: $0 <command>

Commands:
  start       Start the full pipeline in background
  start-llama Start only LLAMA phase
  start-qwen  Start only QWEN phase
  start-merge Start only merge+cleanup phase
  start-store Start only storage phase
  status      Show pipeline status
  progress    Show extraction progress
  logs        Tail the latest log
  kill        Kill the running pipeline
  resume      Resume from last checkpoint

Monitor commands:
  watch       Watch progress in real-time (Ctrl+C to exit)

EOF
}

start_pipeline() {
    local phase="${1:-all}"

    if [ -f "$PID_FILE" ]; then
        pid=$(cat "$PID_FILE")
        if ps -p "$pid" > /dev/null 2>&1; then
            echo "Pipeline already running (PID $pid)"
            return 1
        fi
    fi

    echo "Starting pipeline (phase: $phase)..."
    mkdir -p "$LOG_DIR"

    nohup "$PIPELINE_SCRIPT" "$phase" > "$LOG_DIR/pipeline_$(date +%Y%m%d_%H%M%S).out" 2>&1 &
    echo $! > "$PID_FILE"

    echo "Started with PID $(cat $PID_FILE)"
    echo "Monitor: $0 watch"
    echo "Logs: $0 logs"
}

status() {
    echo "=== PIPELINE STATUS ==="

    if [ -f "$PID_FILE" ]; then
        pid=$(cat "$PID_FILE")
        if ps -p "$pid" > /dev/null 2>&1; then
            echo "Status: RUNNING (PID $pid)"
            echo "Runtime: $(ps -o etime= -p $pid)"
        else
            echo "Status: STOPPED (stale PID file)"
        fi
    else
        echo "Status: NOT RUNNING"
    fi

    echo ""
    echo "=== CHECKPOINTS ==="
    for dir in prod-llama prod-qwen prod-merged; do
        if [ -d "$BASE_DIR/$dir" ]; then
            latest=$(ls -t "$BASE_DIR/$dir"/checkpoint_*.json 2>/dev/null | head -1)
            if [ -n "$latest" ]; then
                count=$(python3 -c "import json; print(len(json.load(open('$latest')).get('known_entities',[])))" 2>/dev/null || echo "?")
                echo "  $dir: $count entities ($(basename $latest))"
            fi
        fi
    done

    echo ""
    echo "=== LATEST LOGS ==="
    ls -lt "$LOG_DIR"/*.log 2>/dev/null | head -5
}

progress() {
    echo "=== EXTRACTION PROGRESS ==="

    for dir in prod-llama prod-qwen; do
        log="$BASE_DIR/$dir/kb_ingest_progress.log"
        if [ -f "$log" ]; then
            echo ""
            echo "--- $dir ---"
            tail -20 "$log"
        fi
    done
}

watch_progress() {
    echo "Watching progress (Ctrl+C to exit)..."
    while true; do
        clear
        date
        echo ""
        progress
        sleep 30
    done
}

tail_logs() {
    latest=$(ls -t "$LOG_DIR"/*.log 2>/dev/null | head -1)
    if [ -n "$latest" ]; then
        echo "Tailing: $latest"
        tail -f "$latest"
    else
        echo "No logs found"
    fi
}

kill_pipeline() {
    if [ -f "$PID_FILE" ]; then
        pid=$(cat "$PID_FILE")
        echo "Killing pipeline (PID $pid)..."
        pkill -P "$pid" 2>/dev/null
        kill "$pid" 2>/dev/null
        rm -f "$PID_FILE"

        # Also kill any python ingestion processes
        pkill -f "kb_ingest_robust_v2.py" 2>/dev/null

        echo "Killed"
    else
        echo "No running pipeline found"
        # Kill any orphaned processes
        pkill -f "kb_ingest_robust_v2.py" 2>/dev/null && echo "Killed orphaned processes"
    fi
}

case "${1:-}" in
    start) start_pipeline all ;;
    start-llama) start_pipeline llama ;;
    start-qwen) start_pipeline qwen ;;
    start-merge) start_pipeline merge ;;
    start-store) start_pipeline storage ;;
    status) status ;;
    progress) progress ;;
    logs) tail_logs ;;
    watch) watch_progress ;;
    kill) kill_pipeline ;;
    *) usage ;;
esac
