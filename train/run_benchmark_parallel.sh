#!/usr/bin/env bash
#
# run_benchmark_parallel.sh — Run parallel benchmark evaluation with a
# benchmark-only compose override and isolated runtime directories.
#
# Note: Kaiwu framework startup scripts still resolve peer services via the
# default "kaiwu-train-*" names, so this entrypoint intentionally reuses the
# default compose project name and must not run concurrently with normal
# training.
#
set -euo pipefail
cd "$(dirname "$0")"

BASE_COMPOSE=".docker-compose.yaml"
BENCHMARK_COMPOSE=".docker-compose.benchmark.yaml"
CHECKPOINT=""
WORKERS=4
ENVS_PER_WORKER=1
MAX_WAIT=1800
ALLOW_CONCURRENT="${ALLOW_CONCURRENT:-0}"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --workers)
            WORKERS="$2"
            shift 2
            ;;
        --envs-per-worker)
            ENVS_PER_WORKER="$2"
            shift 2
            ;;
        --max-wait)
            MAX_WAIT="$2"
            shift 2
            ;;
        -*)
            echo "Unknown option: $1" >&2
            exit 1
            ;;
        *)
            if [[ -z "$CHECKPOINT" ]]; then
                CHECKPOINT="$1"
            else
                echo "Unexpected extra argument: $1" >&2
                exit 1
            fi
            shift
            ;;
    esac
done

if ! [[ "$WORKERS" =~ ^[0-9]+$ ]] || [[ "$WORKERS" -lt 1 ]]; then
    echo "workers must be a positive integer" >&2
    exit 1
fi
if ! [[ "$ENVS_PER_WORKER" =~ ^[0-9]+$ ]] || [[ "$ENVS_PER_WORKER" -lt 1 ]]; then
    echo "envs-per-worker must be a positive integer" >&2
    exit 1
fi

set -a
source .env 2>/dev/null || true
set +a

if [[ "$ALLOW_CONCURRENT" != "1" ]]; then
    TRAINING_CONTAINERS=$(docker ps --format '{{.Names}}' | grep '^kaiwu-train-' || true)
    if [[ -n "$TRAINING_CONTAINERS" ]]; then
        echo "[WARN] Detected running training stack containers:"
        echo "$TRAINING_CONTAINERS" | sed 's/^/  - /'
        echo "Parallel benchmark currently reuses the framework default project name and cannot safely run beside training."
        echo "Set ALLOW_CONCURRENT=1 to bypass this safety check."
        exit 1
    fi
fi

SESSION_ID="$(date +%Y%m%d-%H%M%S)"
COMPOSE_PROJECT_NAME="${COMPOSE_PROJECT_NAME:-kaiwu-train}"
BENCHMARK_ROOT="$(pwd)"
BENCHMARK_LOG_ROOT="${BENCHMARK_ROOT}/benchmark_logs"
BENCHMARK_BACKUP_ROOT="${BENCHMARK_ROOT}/benchmark_backup_model"
BENCHMARK_ARCHIVE_ROOT="${BENCHMARK_ROOT}/benchmark_archive"
HOST_SESSION_DIR="${BENCHMARK_ROOT}/eval_parallel_logs/${SESSION_ID}"
HOST_RESULTS_FILE="${BENCHMARK_ROOT}/eval_parallel_results.json"
CONTAINER_SESSION_DIR="/workspace/train/benchmark_runtime/${SESSION_ID}"
CONTAINER_RESULTS_FILE="/workspace/train/eval_parallel_results.json"
GAMECORES=$((WORKERS * ENVS_PER_WORKER))

GPU1_WORKERS=$(((WORKERS + 1) / 2))
GPU2_WORKERS=$((WORKERS - GPU1_WORKERS))
STACK_STARTED=0
CLEANUP_DONE=0

compose_cmd() {
    docker compose -f "${BASE_COMPOSE}" -f "${BENCHMARK_COMPOSE}" "$@"
}

cleanup_stack() {
    if [[ "${CLEANUP_DONE}" = "1" ]]; then
        return 0
    fi
    CLEANUP_DONE=1

    if [[ "${STACK_STARTED}" = "1" ]]; then
        compose_cmd down --remove-orphans --timeout 10 >/dev/null 2>&1 || true
    fi

    # Compose occasionally leaves stale replicas around when worker counts change
    # between runs. Fall back to project-label cleanup so benchmark exits cleanly.
    mapfile -t project_container_ids < <(
        docker ps -aq --filter "label=com.docker.compose.project=${COMPOSE_PROJECT_NAME}" 2>/dev/null || true
    )
    if [[ "${#project_container_ids[@]}" -gt 0 ]]; then
        docker rm -f "${project_container_ids[@]}" >/dev/null 2>&1 || true
    fi

    docker network rm "${COMPOSE_PROJECT_NAME}_default" >/dev/null 2>&1 || true
}

trap cleanup_stack EXIT

mkdir -p \
    "${BENCHMARK_LOG_ROOT}/framework_ckpt" \
    "${BENCHMARK_BACKUP_ROOT}" \
    "${BENCHMARK_ARCHIVE_ROOT}" \
    "${BENCHMARK_ROOT}/eval_parallel_logs"

export COMPOSE_PROJECT_NAME
export KAIWU_TRAIN_LOG="${BENCHMARK_LOG_ROOT}"
export KAIWU_BACKUP_MODEL="${BENCHMARK_BACKUP_ROOT}"
export KAIWU_ARCHIVE_DIR="${BENCHMARK_ARCHIVE_ROOT}"
export KAIWU_AISRV_NUM="${WORKERS}"
export KAIWU_GAMECORE_NUM="${GAMECORES}"
export KAIWU_PARALLEL_ENV_PER_AISRV="${ENVS_PER_WORKER}"
export KAIWU_AISRV_GPU1_NUM="${GPU1_WORKERS}"
export KAIWU_AISRV_GPU2_NUM="${GPU2_WORKERS}"
export KAIWU_AISRV_GPU3_NUM=0
export KAIWU_BENCHMARK_PARALLEL_MODE=1
export KAIWU_BENCHMARK_SESSION_ID="${SESSION_ID}"
export KAIWU_BENCHMARK_WORKER_COUNT="${WORKERS}"
export KAIWU_BENCHMARK_ENVS_PER_WORKER="${ENVS_PER_WORKER}"
export KAIWU_BENCHMARK_SCHEDULER=dynamic
export KAIWU_BENCHMARK_RUNTIME_DIR="${CONTAINER_SESSION_DIR}"
export KAIWU_BENCHMARK_RESULTS_FILE="${CONTAINER_RESULTS_FILE}"
if [[ -n "${CHECKPOINT}" ]]; then
    export KAIWU_BENCHMARK_CHECKPOINT="${CHECKPOINT}"
fi

AISRV_CONTAINER="${COMPOSE_PROJECT_NAME}-aisrv-1"

echo "========================================="
echo "  Parallel Benchmark Evaluation Tool"
echo "========================================="
echo "[1/6] Session: ${SESSION_ID}"
echo "[2/6] Workers: ${WORKERS}  envs/worker(requested): ${ENVS_PER_WORKER}  gamecores: ${GAMECORES}"
echo "        compose_project: ${COMPOSE_PROJECT_NAME}"
echo "        parallel_env_per_aisrv: ${KAIWU_PARALLEL_ENV_PER_AISRV}"
if [[ -n "${CHECKPOINT}" ]]; then
    echo "[3/6] Checkpoint: ${CHECKPOINT}"
else
    echo "[3/6] Checkpoint: default (conf.py RESUME_CHECKPOINT)"
fi

compose_cmd down --remove-orphans >/dev/null 2>&1 || true

echo "[4/6] Starting benchmark stack..."
compose_cmd up -d \
    pushgateway backup_model gamecore learner aisrv 2>&1 | tail -5
STACK_STARTED=1

sleep 5
BM_MODE=$(docker exec "${AISRV_CONTAINER}" printenv KAIWU_BENCHMARK_PARALLEL_MODE 2>/dev/null || true)
if [[ "${BM_MODE}" != "1" ]]; then
    echo "[WARN] KAIWU_BENCHMARK_PARALLEL_MODE not propagated to ${AISRV_CONTAINER} (got: '${BM_MODE}')"
    echo "       Recreating benchmark stack..."
    compose_cmd up -d --force-recreate \
        pushgateway backup_model gamecore learner aisrv 2>&1 | tail -5
fi

echo "[5/6] Running benchmark..."
ELAPSED=0
while [[ "${ELAPSED}" -lt "${MAX_WAIT}" ]]; do
    if docker exec "${AISRV_CONTAINER}" test -f "${CONTAINER_SESSION_DIR}/done.json" 2>/dev/null; then
        echo ""
        echo "[PBENCH] Benchmark complete."
        docker exec "${AISRV_CONTAINER}" cat "${CONTAINER_SESSION_DIR}/done.json" 2>/dev/null | python3 -c "
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

    COMPLETED=$(docker exec "${AISRV_CONTAINER}" sh -lc "find '${CONTAINER_SESSION_DIR}/tasks/completed' -maxdepth 1 -name '*.json' 2>/dev/null | wc -l" 2>/dev/null | tr -d ' ' || true)
    printf "\r  completed=%s/40 waiting... (%ss)" "${COMPLETED:-0}" "${ELAPSED}"
    sleep 5
    ELAPSED=$((ELAPSED + 5))
done

if [[ "${ELAPSED}" -ge "${MAX_WAIT}" ]]; then
    echo ""
    echo "[WARN] Timeout after ${MAX_WAIT}s. Check: docker logs ${AISRV_CONTAINER}"
fi

echo "[6/6] Copying results..."
mkdir -p "${HOST_SESSION_DIR}"
docker cp "${AISRV_CONTAINER}:${CONTAINER_SESSION_DIR}/." "${HOST_SESSION_DIR}/" 2>/dev/null || true
docker cp "${AISRV_CONTAINER}:${CONTAINER_RESULTS_FILE}" "${HOST_RESULTS_FILE}" 2>/dev/null || true

echo ""
echo "Detailed logs: ${HOST_SESSION_DIR}"
echo "Results file: ${HOST_RESULTS_FILE}"
echo ""
echo "Stopping benchmark stack..."
cleanup_stack
