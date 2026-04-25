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
POLICY_MODE="${KAIWU_BENCHMARK_POLICY_MODE:-eval}"
EXTERNAL_BENCHMARK_ROUNDS_JSON="${KAIWU_BENCHMARK_ROUNDS_JSON:-}"
EXTERNAL_BENCHMARK_MAPS="${KAIWU_BENCHMARK_MAPS:-}"
EXTERNAL_BENCHMARK_POLICY_MODE="${KAIWU_BENCHMARK_POLICY_MODE:-}"
EXTERNAL_BENCHMARK_MAX_WAIT="${KAIWU_BENCHMARK_MAX_WAIT:-}"

if [[ "${CHECKPOINT}" == "--policy-mode" ]]; then
    POLICY_MODE="${2:-eval}"
    CHECKPOINT="${3:-}"
elif [[ "${2:-}" == "--policy-mode" ]]; then
    POLICY_MODE="${3:-eval}"
fi
if [[ "${POLICY_MODE}" != "train" && "${POLICY_MODE}" != "eval" ]]; then
    echo "policy-mode must be 'train' or 'eval'" >&2
    exit 1
fi

# Load .env
set -a; source .env 2>/dev/null || true; set +a
if [[ -n "${EXTERNAL_BENCHMARK_ROUNDS_JSON}" ]]; then
    export KAIWU_BENCHMARK_ROUNDS_JSON="${EXTERNAL_BENCHMARK_ROUNDS_JSON}"
fi
if [[ -n "${EXTERNAL_BENCHMARK_MAPS}" ]]; then
    export KAIWU_BENCHMARK_MAPS="${EXTERNAL_BENCHMARK_MAPS}"
fi
if [[ -n "${EXTERNAL_BENCHMARK_POLICY_MODE}" ]]; then
    export KAIWU_BENCHMARK_POLICY_MODE="${EXTERNAL_BENCHMARK_POLICY_MODE}"
fi
if [[ -n "${EXTERNAL_BENCHMARK_MAX_WAIT}" ]]; then
    export KAIWU_BENCHMARK_MAX_WAIT="${EXTERNAL_BENCHMARK_MAX_WAIT}"
fi

PROJECT_CODE="${KAIWU_PROJECT_CODE:-robot_vacuum}"
ALGORITHM="${KAIWU_ALGORITHM:-ppo}"
HOST_CODE_DIR="${KAIWU_CODE_FILE:-$(pwd)/../code}"
HOST_BENCHMARK_DONE="${HOST_CODE_DIR%/}/.benchmark_done"
HOST_STOP_DIRS=(
    "/data/ckpt/${PROJECT_CODE}_${ALGORITHM}"
    "${HOST_CODE_DIR%/}/ckpt/${PROJECT_CODE}_${ALGORITHM}"
)

echo "========================================="
echo "  Benchmark Evaluation Tool"
echo "========================================="

# 1. Stop training
echo "[1/6] Stopping training..."
docker compose -f "$COMPOSE_FILE" --profile "$PROFILE" down 2>/dev/null || true

# 2. Set benchmark mode
export KAIWU_BENCHMARK_MODE=1
export KAIWU_BENCHMARK_POLICY_MODE="${POLICY_MODE}"
if [ -n "$CHECKPOINT" ]; then
    export KAIWU_BENCHMARK_CHECKPOINT="$CHECKPOINT"
    echo "[2/6] Checkpoint: $CHECKPOINT"
else
    echo "[2/6] Checkpoint: default (conf.py RESUME_CHECKPOINT)"
fi
echo "        policy_mode: ${POLICY_MODE}"

# 3. Clean stale host-side benchmark / stop markers
echo "[3/6] Clearing stale benchmark markers..."
rm -f "$HOST_BENCHMARK_DONE"
if [ -e "$HOST_BENCHMARK_DONE" ]; then
    echo "[ERROR] Failed to clear host benchmark marker: $HOST_BENCHMARK_DONE" >&2
    exit 1
fi
for STOP_DIR in "${HOST_STOP_DIRS[@]}"; do
    if [ -d "$STOP_DIR" ]; then
        rm -f "$STOP_DIR/process_stop.done" "$STOP_DIR/process_stop.meta.json"
    fi
done
docker exec kaiwu-train-aisrv-1 rm -f /workspace/code/.benchmark_done 2>/dev/null || true

clear_container_stop_markers() {
    local stop_dir="/data/ckpt/${PROJECT_CODE}_${ALGORITHM}"
    docker exec kaiwu-train-learner-1 rm -f "${stop_dir}/process_stop.done" "${stop_dir}/process_stop.meta.json" 2>/dev/null || true
    docker exec kaiwu-train-aisrv-1 rm -f "${stop_dir}/process_stop.done" "${stop_dir}/process_stop.meta.json" 2>/dev/null || true
}

BENCHMARK_SUPPORT_SERVICES=(pushgateway backup_model gamecore learner)

start_benchmark_stack() {
    docker compose -f "$COMPOSE_FILE" --profile "$PROFILE" up -d "${BENCHMARK_SUPPORT_SERVICES[@]}" 2>&1 | tail -3
    docker compose -f "$COMPOSE_FILE" --profile "$PROFILE" up -d --no-deps aisrv 2>&1 | tail -3
}

recreate_benchmark_stack() {
    docker compose -f "$COMPOSE_FILE" --profile "$PROFILE" up -d --force-recreate "${BENCHMARK_SUPPORT_SERVICES[@]}" 2>&1 | tail -3
    docker compose -f "$COMPOSE_FILE" --profile "$PROFILE" up -d --force-recreate --no-deps aisrv 2>&1 | tail -3
}

# 4. Start stack in benchmark mode
echo "[4/6] Starting evaluation stack..."
BENCHMARK_EXISTING_SESSIONS=$(python3 -c '
from pathlib import Path
import re

base = Path("./eval_logs")
if base.exists():
    sessions = sorted(
        p.name for p in base.iterdir() if p.is_dir() and re.fullmatch(r"\d{8}-\d{6}", p.name)
    )
    if sessions:
        print("\n".join(sessions))
')
start_benchmark_stack
sleep 3
clear_container_stop_markers
BM_MODE=$(docker exec kaiwu-train-aisrv-1 printenv KAIWU_BENCHMARK_MODE 2>/dev/null || true)
if [ "$BM_MODE" != "1" ]; then
    echo "[WARN] KAIWU_BENCHMARK_MODE not propagated to container (got: '$BM_MODE')"
    echo "       Falling back to --force-recreate..."
    recreate_benchmark_stack
    sleep 3
    clear_container_stop_markers
fi
if [ -n "$CHECKPOINT" ]; then
    CONTAINER_CHECKPOINT=$(docker exec kaiwu-train-aisrv-1 printenv KAIWU_BENCHMARK_CHECKPOINT 2>/dev/null || true)
    if [ "$CONTAINER_CHECKPOINT" != "$CHECKPOINT" ]; then
        echo "[WARN] KAIWU_BENCHMARK_CHECKPOINT not propagated to container (expected: '$CHECKPOINT', got: '$CONTAINER_CHECKPOINT')"
        echo "       Falling back to --force-recreate..."
        recreate_benchmark_stack
        sleep 3
        clear_container_stop_markers
        CONTAINER_CHECKPOINT=$(docker exec kaiwu-train-aisrv-1 printenv KAIWU_BENCHMARK_CHECKPOINT 2>/dev/null || true)
        if [ "$CONTAINER_CHECKPOINT" != "$CHECKPOINT" ]; then
            echo "[ERROR] KAIWU_BENCHMARK_CHECKPOINT propagation failed after retry (expected: '$CHECKPOINT', got: '$CONTAINER_CHECKPOINT')" >&2
            exit 1
        fi
    fi
    echo "        container_checkpoint: ${CONTAINER_CHECKPOINT}"
fi
if ! docker exec kaiwu-train-aisrv-1 test ! -e /workspace/code/.benchmark_done 2>/dev/null; then
    echo "[ERROR] Benchmark marker still present after stack startup: /workspace/code/.benchmark_done" >&2
    exit 1
fi
echo "        (episode count follows KAIWU_BENCHMARK_ROUNDS_JSON / selected wrapper profile)"

benchmark_state_host_json() {
    BENCHMARK_EXISTING_SESSIONS="$BENCHMARK_EXISTING_SESSIONS" BENCHMARK_STATE_BASE="./eval_logs" python3 -c '
from pathlib import Path
import json
import os
import re

state = {
    "session_id": "",
    "started_count": 0,
    "total_episodes": 0,
    "manifest_exists": False,
    "benchmark_log_exists": False,
    "result_exists": False,
    "done_exists": False,
    "last_line": "",
}

existing_sessions = {line.strip() for line in os.getenv("BENCHMARK_EXISTING_SESSIONS", "").splitlines() if line.strip()}
base = Path(os.getenv("BENCHMARK_STATE_BASE", "./eval_logs"))
if base.exists():
    sessions = [
        p for p in base.iterdir()
        if p.is_dir() and re.fullmatch(r"\d{8}-\d{6}", p.name) and p.name not in existing_sessions
    ]
    if sessions:
        latest = max(sessions, key=lambda p: p.name)
        manifest_path = latest / "manifest.json"
        benchmark_log_path = latest / "benchmark.log"
        result_path = latest / "result.json"

        state["session_id"] = latest.name
        state["manifest_exists"] = manifest_path.exists()
        state["benchmark_log_exists"] = benchmark_log_path.exists()
        state["result_exists"] = result_path.exists()

        if manifest_path.exists():
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                state["total_episodes"] = int(manifest.get("total_episodes", 0) or 0)
            except Exception:
                pass

        if benchmark_log_path.exists():
            episode_re = re.compile(r"\[(\d+)/(\d+)\]")
            last_line = ""
            started_count = 0
            with benchmark_log_path.open("r", encoding="utf-8", errors="ignore") as handle:
                for raw_line in handle:
                    line = raw_line.strip()
                    if line:
                        last_line = line
                    match = episode_re.search(line)
                    if match:
                        started_count = max(started_count, int(match.group(1)))
                        if state["total_episodes"] <= 0:
                            state["total_episodes"] = int(match.group(2))
            state["started_count"] = started_count
            state["last_line"] = last_line[-400:]

print(json.dumps(state, ensure_ascii=False))
'
}

benchmark_state_container_json() {
    docker exec -e BENCHMARK_EXISTING_SESSIONS="$BENCHMARK_EXISTING_SESSIONS" -e BENCHMARK_STATE_BASE="/workspace/code/eval_logs" kaiwu-train-aisrv-1 python3 -c '
from pathlib import Path
import json
import os
import re

state = {
    "session_id": "",
    "started_count": 0,
    "total_episodes": 0,
    "manifest_exists": False,
    "benchmark_log_exists": False,
    "result_exists": False,
    "done_exists": Path("/workspace/code/.benchmark_done").exists(),
    "last_line": "",
}

existing_sessions = {line.strip() for line in os.getenv("BENCHMARK_EXISTING_SESSIONS", "").splitlines() if line.strip()}
base = Path(os.getenv("BENCHMARK_STATE_BASE", "/workspace/code/eval_logs"))
if base.exists():
    sessions = [
        p for p in base.iterdir()
        if p.is_dir() and re.fullmatch(r"\d{8}-\d{6}", p.name) and p.name not in existing_sessions
    ]
    if sessions:
        latest = max(sessions, key=lambda p: p.name)
        manifest_path = latest / "manifest.json"
        benchmark_log_path = latest / "benchmark.log"
        result_path = latest / "result.json"

        state["session_id"] = latest.name
        state["manifest_exists"] = manifest_path.exists()
        state["benchmark_log_exists"] = benchmark_log_path.exists()
        state["result_exists"] = result_path.exists()

        if manifest_path.exists():
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                state["total_episodes"] = int(manifest.get("total_episodes", 0) or 0)
            except Exception:
                pass

        if benchmark_log_path.exists():
            episode_re = re.compile(r"\[(\d+)/(\d+)\]")
            last_line = ""
            started_count = 0
            with benchmark_log_path.open("r", encoding="utf-8", errors="ignore") as handle:
                for raw_line in handle:
                    line = raw_line.strip()
                    if line:
                        last_line = line
                    match = episode_re.search(line)
                    if match:
                        started_count = max(started_count, int(match.group(1)))
                        if state["total_episodes"] <= 0:
                            state["total_episodes"] = int(match.group(2))
            state["started_count"] = started_count
            state["last_line"] = last_line[-400:]

print(json.dumps(state, ensure_ascii=False))
' 2>/dev/null || true
}

benchmark_state_json() {
    {
        benchmark_state_host_json
        benchmark_state_container_json
    } | python3 -c '
import json
import sys

states = []
for line in sys.stdin:
    raw = line.strip()
    if not raw:
        continue
    try:
        states.append(json.loads(raw))
    except Exception:
        pass

def score(state):
    return (
        1 if state.get("session_id") else 0,
        state.get("session_id", ""),
        1 if state.get("result_exists") else 0,
        1 if state.get("benchmark_log_exists") else 0,
        int(state.get("started_count", 0) or 0),
    )

print(json.dumps(max(states, key=score) if states else {}, ensure_ascii=False))
'
}

benchmark_state_fields() {
    python3 -c '
import json
import sys

raw = sys.stdin.read().strip()
state = json.loads(raw) if raw else {}
values = [
    state.get("session_id", ""),
    str(int(state.get("started_count", 0) or 0)),
    str(int(state.get("total_episodes", 0) or 0)),
    "1" if state.get("manifest_exists") else "0",
    "1" if state.get("benchmark_log_exists") else "0",
    "1" if state.get("result_exists") else "0",
    "1" if state.get("done_exists") else "0",
    (state.get("last_line") or "").replace("\t", " "),
]
print("\n".join(values))
'
}

print_benchmark_summary() {
    docker exec kaiwu-train-aisrv-1 python3 -c '
from pathlib import Path
import json
import re

payload = None
done_marker = Path("/workspace/code/.benchmark_done")
if done_marker.exists():
    try:
        payload = json.loads(done_marker.read_text(encoding="utf-8"))
    except Exception:
        payload = None

if payload is None:
    base = Path("/workspace/code/eval_logs")
    if base.exists():
        sessions = [p for p in base.iterdir() if p.is_dir() and re.fullmatch(r"\d{8}-\d{6}", p.name)]
        if sessions:
            latest = max(sessions, key=lambda p: p.name)
            result_path = latest / "result.json"
            if result_path.exists():
                try:
                    payload = json.loads(result_path.read_text(encoding="utf-8"))
                except Exception:
                    payload = None

if payload is None:
    raise SystemExit(1)

overall = payload.get("overall", payload.get("overall", {}))
print(
    f"  WR={overall.get('win_rate', 0):.0%}  CS={overall.get('avg_clean_score', 0):.0f}  "
    f"({overall.get('win_episode_count', '?')}/{overall.get('episode_count', '?')})"
)
' 2>/dev/null || true
}

# 5. Wait for benchmark to complete
echo "[5/6] Running benchmark..."
MAX_WAIT="${KAIWU_BENCHMARK_MAX_WAIT:-900}"  # default 15 minutes; override for slow target runs
ELAPSED=0
LAST_PROGRESS_LINE=""
LAST_SESSION_ID=""
LAST_STARTED=0
LAST_TOTAL=0
LAST_MANIFEST=0
LAST_LOG=0
LAST_RESULT=0
LAST_DONE=0
while [ $ELAPSED -lt $MAX_WAIT ]; do
    STATE_JSON=$(benchmark_state_json)
    mapfile -t STATE_VALUES < <(printf "%s" "$STATE_JSON" | benchmark_state_fields 2>/dev/null || true)
    SESSION_ID="${STATE_VALUES[0]:-}"
    STARTED_COUNT="${STATE_VALUES[1]:-0}"
    TOTAL_EPISODES="${STATE_VALUES[2]:-0}"
    HAS_MANIFEST="${STATE_VALUES[3]:-0}"
    HAS_LOG="${STATE_VALUES[4]:-0}"
    HAS_RESULT="${STATE_VALUES[5]:-0}"
    HAS_DONE="${STATE_VALUES[6]:-0}"
    LAST_LINE="${STATE_VALUES[7]:-}"

    LAST_SESSION_ID="${SESSION_ID:-}"
    LAST_STARTED=${STARTED_COUNT:-0}
    LAST_TOTAL=${TOTAL_EPISODES:-0}
    LAST_MANIFEST=${HAS_MANIFEST:-0}
    LAST_LOG=${HAS_LOG:-0}
    LAST_RESULT=${HAS_RESULT:-0}
    LAST_DONE=${HAS_DONE:-0}
    LAST_PROGRESS_LINE="${LAST_LINE:-}"

    if [ "${HAS_DONE:-0}" = "1" ] || [ "${HAS_RESULT:-0}" = "1" ]; then
        echo ""
        echo "[BENCHMARK] Benchmark complete."
        print_benchmark_summary
        break
    fi

    if [ -n "${SESSION_ID:-}" ] && [ "${HAS_LOG:-0}" = "1" ]; then
        if [ "${TOTAL_EPISODES:-0}" -gt 0 ]; then
            printf "\r  session %s progress %d/%d from benchmark.log (%ds)" "$SESSION_ID" "$STARTED_COUNT" "$TOTAL_EPISODES" "$ELAPSED"
        else
            printf "\r  session %s progress %d episodes from benchmark.log (%ds)" "$SESSION_ID" "$STARTED_COUNT" "$ELAPSED"
        fi
    elif [ -n "${SESSION_ID:-}" ] && [ "${HAS_MANIFEST:-0}" = "1" ]; then
        printf "\r  session %s created; waiting for benchmark.log (%ds)" "$SESSION_ID" "$ELAPSED"
    else
        printf "\r  waiting for benchmark session to initialize... (%ds)" "$ELAPSED"
    fi

    sleep 5
    ELAPSED=$((ELAPSED + 5))
done

if [ $ELAPSED -ge $MAX_WAIT ]; then
    echo ""
    echo "[WARN] Timeout after ${MAX_WAIT}s. Latest benchmark evidence:" 
    if [ -n "$LAST_SESSION_ID" ]; then
        echo "       session=${LAST_SESSION_ID} manifest=${LAST_MANIFEST} benchmark_log=${LAST_LOG} result=${LAST_RESULT} done=${LAST_DONE} progress=${LAST_STARTED}/${LAST_TOTAL}"
        if [ -n "$LAST_PROGRESS_LINE" ]; then
            echo "       last_log: $LAST_PROGRESS_LINE"
        fi
    else
        echo "       no benchmark session directory detected under /workspace/code/eval_logs"
    fi
    echo "       Check: docker logs kaiwu-train-aisrv-1"
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
