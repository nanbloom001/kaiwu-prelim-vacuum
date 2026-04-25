#!/usr/bin/env bash
#
# run_target_benchmark_900.sh — Freeze and launch the benchmark-900 target profile.
#
# This wrapper only selects a frozen profile and exports the benchmark env vars
# already consumed by run_benchmark.sh / run_benchmark_parallel.sh. It must not
# alter scoring, simulator, map, or aggregation code.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

DRY_RUN=0
PROFILE="target"
POLICY_MODE="eval"
RUNNER="serial"
CHECKPOINT=""

usage() {
    cat <<'USAGE'
Usage: bash train/run_target_benchmark_900.sh [options] [checkpoint]

Options:
  --dry-run              Print planned JSON/manifest and do not start Docker.
  --profile target|dev   Select 30-episode target or 10-episode dev profile.
  --policy-mode eval     Policy mode passed through to benchmark runner.
  --runner serial|parallel
                         Runner metadata/path; serial is canonical.
  -h, --help             Show this help.
USAGE
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run)
            DRY_RUN=1
            shift
            ;;
        --profile)
            PROFILE="${2:-}"
            shift 2
            ;;
        --policy-mode)
            POLICY_MODE="${2:-}"
            shift 2
            ;;
        --runner)
            RUNNER="${2:-}"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        --*)
            echo "Unknown option: $1" >&2
            usage >&2
            exit 1
            ;;
        *)
            if [[ -z "${CHECKPOINT}" ]]; then
                CHECKPOINT="$1"
            else
                echo "Unexpected extra argument: $1" >&2
                exit 1
            fi
            shift
            ;;
    esac
done

case "${PROFILE}" in
    target)
        PROFILE_PATH="${SCRIPT_DIR}/benchmark_profiles/target_3c4r_1000_150_30.json"
        MANIFEST_PATH="${REPO_ROOT}/.sisyphus/evidence/benchmark-900/task-1-dry-run-manifest.json"
        ;;
    dev)
        PROFILE_PATH="${SCRIPT_DIR}/benchmark_profiles/dev_3c4r_1000_150_10.json"
        MANIFEST_PATH="${REPO_ROOT}/.sisyphus/evidence/benchmark-900/task-1-dev-dry-run-manifest.json"
        ;;
    *)
        echo "profile must be 'target' or 'dev'" >&2
        exit 1
        ;;
esac

if [[ "${POLICY_MODE}" != "eval" ]]; then
    echo "policy-mode must be 'eval' for frozen benchmark-900 runs" >&2
    exit 1
fi

if [[ "${RUNNER}" != "serial" && "${RUNNER}" != "parallel" ]]; then
    echo "runner must be 'serial' or 'parallel'" >&2
    exit 1
fi

if [[ ! -f "${PROFILE_PATH}" ]]; then
    echo "Missing benchmark profile: ${PROFILE_PATH}" >&2
    exit 1
fi

MANIFEST_JSON=$(REPO_ROOT="${REPO_ROOT}" PROFILE_PATH="${PROFILE_PATH}" CHECKPOINT="${CHECKPOINT}" \
    POLICY_MODE="${POLICY_MODE}" RUNNER="${RUNNER}" DRY_RUN="${DRY_RUN}" \
    MANIFEST_PATH="${MANIFEST_PATH}" python3 - <<'PY'
import hashlib
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_commit(repo_root: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=str(repo_root), text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return "unknown"


repo_root = Path(os.environ["REPO_ROOT"])
profile_path = Path(os.environ["PROFILE_PATH"])
checkpoint = os.environ.get("CHECKPOINT", "")
checkpoint_path = Path(checkpoint) if checkpoint else None
if checkpoint_path is not None and not checkpoint_path.is_absolute():
    checkpoint_path = repo_root / checkpoint_path

profile = json.loads(profile_path.read_text(encoding="utf-8"))
maps = profile["maps"]
rounds = profile["rounds"]
planned_episodes = []
idx = 0
for round_def in rounds:
    for map_id in maps:
        idx += 1
        planned_episodes.append(
            {
                "index": idx,
                "round_name": round_def["name"],
                "map": map_id,
                "charger_count": round_def["charger_count"],
                "robot_count": round_def["robot_count"],
                "max_step": round_def["max_step"],
                "battery_max": round_def["battery_max"],
            }
        )

now = datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
runner = os.environ["RUNNER"]
result_path = "train/eval_results.json" if runner == "serial" else "train/eval_parallel_results.json"
manifest = {
    "schema_version": 1,
    "generated_by": "train/run_target_benchmark_900.sh",
    "profile_name": profile["profile_name"],
    "profile_path": str(profile_path.relative_to(repo_root)),
    "profile_sha256": sha256_file(profile_path),
    "git_commit": git_commit(repo_root),
    "checkpoint_path": checkpoint,
    "checkpoint_sha256": sha256_file(checkpoint_path) if checkpoint_path is not None and checkpoint_path.is_file() else None,
    "policy_mode": os.environ["POLICY_MODE"],
    "runner": runner,
    "dry_run": os.environ["DRY_RUN"] == "1",
    "start_time": now,
    "end_time": now if os.environ["DRY_RUN"] == "1" else None,
    "result_path": result_path,
    "env": profile["env"],
    "maps": maps,
    "rounds": rounds,
    "planned_episode_count": len(planned_episodes),
    "planned_episodes": planned_episodes,
}

manifest_path = Path(os.environ["MANIFEST_PATH"])
manifest_path.parent.mkdir(parents=True, exist_ok=True)
manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(json.dumps(manifest, indent=2, sort_keys=True))
PY
)

echo "${MANIFEST_JSON}"

if [[ "${DRY_RUN}" == "1" ]]; then
    echo "[DRY-RUN] Manifest written: ${MANIFEST_PATH}"
    echo "[DRY-RUN] Docker was not started."
    exit 0
fi

ROUNDS_JSON=$(PROFILE_PATH="${PROFILE_PATH}" python3 - <<'PY'
import json
import os
from pathlib import Path

profile = json.loads(Path(os.environ["PROFILE_PATH"]).read_text(encoding="utf-8"))
print(json.dumps(profile["rounds"], separators=(",", ":")))
PY
)
MAPS_CSV=$(PROFILE_PATH="${PROFILE_PATH}" python3 - <<'PY'
import json
import os
from pathlib import Path

profile = json.loads(Path(os.environ["PROFILE_PATH"]).read_text(encoding="utf-8"))
print(",".join(str(item) for item in profile["maps"]))
PY
)

export KAIWU_BENCHMARK_ROUNDS_JSON="${ROUNDS_JSON}"
export KAIWU_BENCHMARK_MAPS="${MAPS_CSV}"
export KAIWU_BENCHMARK_POLICY_MODE="${POLICY_MODE}"

cd "${SCRIPT_DIR}"
if [[ "${RUNNER}" == "serial" ]]; then
    exec bash run_benchmark.sh --policy-mode "${POLICY_MODE}" "${CHECKPOINT}"
fi

if [[ -n "${CHECKPOINT}" ]]; then
    exec bash run_benchmark_parallel.sh "${CHECKPOINT}" --policy-mode "${POLICY_MODE}"
fi
exec bash run_benchmark_parallel.sh --policy-mode "${POLICY_MODE}"
