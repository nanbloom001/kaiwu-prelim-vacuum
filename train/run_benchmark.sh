#!/usr/bin/env bash
#
# run_benchmark.sh — Run benchmark evaluation against a model checkpoint
#
# Usage:
#   ./run_benchmark.sh                          # eval default checkpoint
#   ./run_benchmark.sh path/to/checkpoint.pkl   # eval specific checkpoint
#   RESTART=1 ./run_benchmark.sh                # auto-restart training after eval
#
set -euo pipefail
cd "$(dirname "$0")"

COMPOSE_FILE=".docker-compose.yaml"
PROFILE="distributed"
CHECKPOINT="${1:-}"
RESTART="${RESTART:-}"

# Load .env
set -a; source .env 2>/dev/null || true; set +a

echo "========================================="
echo "  Benchmark Evaluation Tool"
echo "========================================="

# 1. Stop training
echo "[1/6] Stopping training..."
docker compose -f "$COMPOSE_FILE" --profile "$PROFILE" down 2>/dev/null || true

# 2. Set benchmark mode
export KAIWU_BENCHMARK_MODE=1
if [ -n "$CHECKPOINT" ]; then
    export KAIWU_BENCHMARK_CHECKPOINT="$CHECKPOINT"
    echo "[2/6] Checkpoint: $CHECKPOINT"
else
    echo "[2/6] Checkpoint: default (conf.py RESUME_CHECKPOINT)"
fi

# 3. Clean old benchmark marker
docker exec kaiwu-train-aisrv-1 rm -f /workspace/code/.benchmark_done 2>/dev/null || true

# 4. Start stack in benchmark mode
echo "[3/6] Starting evaluation stack..."
docker compose -f "$COMPOSE_FILE" --profile "$PROFILE" up -d 2>&1 | tail -3
sleep 3
BM_MODE=$(docker exec kaiwu-train-aisrv-1 printenv KAIWU_BENCHMARK_MODE 2>/dev/null || true)
if [ "$BM_MODE" != "1" ]; then
    echo "[WARN] KAIWU_BENCHMARK_MODE not propagated to container (got: '$BM_MODE')"
    echo "       Falling back to --force-recreate..."
    docker compose -f "$COMPOSE_FILE" --profile "$PROFILE" up -d --force-recreate 2>&1 | tail -3
fi
echo "        (40 episodes on 10 maps x 4 rounds, ~5-10 min)"

# 5. Wait for benchmark to complete
echo "[4/6] Running benchmark..."
MAX_WAIT=900  # 15 minutes max (40 episodes can take a while)
ELAPSED=0
while [ $ELAPSED -lt $MAX_WAIT ]; do
    if docker exec kaiwu-train-aisrv-1 test -f /workspace/code/.benchmark_done 2>/dev/null; then
        echo ""
        echo "[BENCHMARK] Benchmark complete."
        docker exec kaiwu-train-aisrv-1 cat /workspace/code/.benchmark_done 2>/dev/null | python3 -c "
import json, sys
d = json.load(sys.stdin)
o = d.get('overall', {})
print(
    f\"  WR={o.get('win_rate', 0):.0%}  CS={o.get('avg_clean_score', 0):.0f}  \"
    f\"({o.get('win_episode_count', '?')}/{o.get('episode_count', '?')})\"
)
" 2>/dev/null || true
        break
    fi

    # Show progress
    PROGRESS=$(docker exec kaiwu-train-aisrv-1 cat /data/projects/robot_vacuum/log/aisrv.log 2>/dev/null \
        | grep -c "\[BENCHMARK\].*START" || true)
    printf "\r  %d episodes started, waiting... (%ds)" "$PROGRESS" "$ELAPSED"

    sleep 5
    ELAPSED=$((ELAPSED + 5))
done

if [ $ELAPSED -ge $MAX_WAIT ]; then
    echo ""
    echo "[WARN] Timeout after ${MAX_WAIT}s. Check: docker logs kaiwu-train-aisrv-1"
fi

# 6. Copy results and logs
echo "[5/6] Copying results and logs..."
mkdir -p eval_logs
docker cp kaiwu-train-aisrv-1:/workspace/code/eval_results.json ./eval_results.json 2>/dev/null || true
docker cp kaiwu-train-aisrv-1:/workspace/code/eval_logs/. ./eval_logs/ 2>/dev/null || true
docker exec kaiwu-train-aisrv-1 rm -f /workspace/code/.benchmark_done 2>/dev/null || true

# 7. Show results
echo "[6/6] Results:"
if [ -f eval_results.json ]; then
    python3 compare_benchmarks.py latest 2>/dev/null || echo "(run 'python3 compare_benchmarks.py latest' to view)"
else
    echo "[WARN] No eval_results.json found"
fi

# Show log location
LATEST_LOG=$(ls -td eval_logs/*/ 2>/dev/null | head -1)
if [ -n "$LATEST_LOG" ]; then
    echo ""
    echo "Detailed logs: ${LATEST_LOG}"
    echo "  - benchmark.log     : step-level progress"
    echo "  - manifest.json     : scenario configs used"
    echo "  - result.json       : full results"
    echo "  - episodes/*.jsonl  : per-step details (battery/dirt/action/mode)"
fi

# Stop benchmark stack
echo ""
docker compose -f "$COMPOSE_FILE" --profile "$PROFILE" down 2>/dev/null || true

# Optionally restart training
if [ "$RESTART" = "1" ]; then
    echo "Restarting training..."
    unset KAIWU_BENCHMARK_MODE
    unset KAIWU_BENCHMARK_CHECKPOINT
    docker compose -f "$COMPOSE_FILE" --profile "$PROFILE" up -d 2>&1 | tail -3
    echo "Training restarted."
else
    echo "Stack stopped. To restart training:"
    echo "  cd train && docker compose -f .docker-compose.yaml --profile distributed up -d"
fi
