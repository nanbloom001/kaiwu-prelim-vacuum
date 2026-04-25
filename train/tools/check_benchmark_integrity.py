#!/usr/bin/env python3
"""Check benchmark result integrity and anti-cheat guardrails.

The checker is intentionally data-only: it reads finished benchmark artifacts and
git metadata, but never starts Docker, imports simulator code, or runs benchmark
runners.  The default profile matches the canonical serial target:
maps 1-10, three rounds per map, 30 episodes, 3 chargers, 4 robots, 1000 steps,
and 150 battery.  The parallel-40 operational profile remains available only
through explicit CLI arguments and is not the canonical success shape.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MAPS = tuple(range(1, 11))
DEFAULT_EPISODES = 30
DEFAULT_ROUNDS_PER_MAP = 3
DEFAULT_CHARGER_COUNT = 3
DEFAULT_ROBOT_COUNT = 4
DEFAULT_MAX_STEP = 1000
DEFAULT_BATTERY_MAX = 150
OPERATIONAL_PARALLEL_EPISODES = 40
OPERATIONAL_PARALLEL_ROUNDS_PER_MAP = 4

ROUND_PROFILE_RE = re.compile(
    r"(?P<charger_count>\d+)\s+chargers\s*/\s*"
    r"(?P<robot_count>\d+)\s+robots\s*/\s*"
    r"(?P<max_step>\d+)\s+steps\s*/\s*"
    r"(?P<battery_max>\d+)\s+battery"
)

FORBIDDEN_PATH_PATTERNS = (
    (re.compile(r"(^|/)(gamecore|simulator|simulation|maps?|map_data|scoring)(/|$)", re.I), "simulator/gamecore/map/scoring path changed"),
    (re.compile(r"(^|/)tencentarena-docs/.*(map|score|scoring|simulator|gamecore)", re.I), "official simulator/map/scoring docs changed"),
)

TARGET_PROFILE_RE = re.compile(r"(^|/)train/benchmark_profiles/target_.*\.json$")

SCORING_GUARD_PATHS = (
    "code/agent_ppo/eval/benchmark.py",
    "code/agent_ppo/eval/benchmark_parallel.py",
    "train/compare_benchmarks.py",
    "train/tools/summarize_benchmark_failures.py",
)

FORBIDDEN_DIFF_LINE_PATTERNS = (
    (re.compile(r"^[-+].*avg_clean_score", re.I), "benchmark average clean-score aggregation changed"),
    (re.compile(r"^[-+].*clean_score", re.I), "benchmark clean-score handling changed"),
    (re.compile(r"^[-+].*win_rate", re.I), "benchmark win-rate aggregation changed"),
    (re.compile(r"^[-+].*completed_rate", re.I), "benchmark completed-rate aggregation changed"),
    (re.compile(r"^[-+].*broad_win_rate", re.I), "benchmark broad-win aggregation changed"),
)


class IntegrityError(ValueError):
    """Raised when benchmark integrity checks fail."""


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise IntegrityError(f"missing result file: {path}") from exc
    except json.JSONDecodeError as exc:
        raise IntegrityError(f"invalid JSON in {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise IntegrityError(f"result must be a JSON object: {path}")
    if isinstance(payload.get("benchmarks"), list):
        rows = [row for row in payload["benchmarks"] if isinstance(row, dict)]
        if not rows:
            raise IntegrityError(f"benchmark collection has no object rows: {path}")
        payload = rows[-1]
    return payload


def _parse_maps(value: str) -> list[int]:
    try:
        maps = [int(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"maps must be comma-separated integers: {value}") from exc
    if not maps:
        raise argparse.ArgumentTypeError("maps must not be empty")
    return maps


def _expected_env(args: argparse.Namespace) -> dict[str, int]:
    return {
        "charger_count": args.charger_count,
        "robot_count": args.robot_count,
        "max_step": args.max_step,
        "battery_max": args.battery_max,
    }


def _round_env(value: Any) -> dict[str, int] | None:
    if not isinstance(value, str):
        return None
    match = ROUND_PROFILE_RE.search(value)
    if match is None:
        return None
    return {key: int(raw) for key, raw in match.groupdict().items()}


def _result_payload(path: Path) -> dict[str, Any]:
    return _read_json(path)


def _episode_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    episodes = payload.get("episodes")
    if not isinstance(episodes, list):
        return []
    return [row for row in episodes if isinstance(row, dict)]


def validate_result(path: Path, args: argparse.Namespace) -> tuple[list[str], dict[str, Any]]:
    payload = _result_payload(path)
    errors: list[str] = []
    expected_maps = list(args.maps)
    expected_env = _expected_env(args)

    rounds = payload.get("rounds")
    if not isinstance(rounds, dict) or not rounds:
        errors.append(f"rounds: expected non-empty object, got {rounds!r}")
        rounds = {}
    if len(rounds) != args.rounds_per_map:
        errors.append(f"round_count: expected {args.rounds_per_map}, got {len(rounds)}")
    for name, description in sorted(rounds.items(), key=lambda item: str(item[0])):
        got_env = _round_env(description)
        if got_env != expected_env:
            errors.append(f"rounds.{name}: expected env {expected_env!r}, got {got_env!r} from {description!r}")

    episodes = _episode_rows(payload)
    if len(episodes) != args.episodes:
        errors.append(f"episode_count: expected {args.episodes}, got {len(episodes)}")

    counts: Counter[int] = Counter()
    unexpected_map_values: list[Any] = []
    round_names: set[str] = set()
    for index, row in enumerate(episodes, start=1):
        map_id = row.get("map_id", row.get("map"))
        if isinstance(map_id, int):
            counts[map_id] += 1
        else:
            unexpected_map_values.append({"episode": index, "map": map_id})
        round_name = row.get("round", row.get("round_name"))
        if isinstance(round_name, str):
            round_names.add(round_name)

    for map_id in expected_maps:
        got = counts.get(map_id, 0)
        if got != args.rounds_per_map:
            errors.append(f"rounds_per_map[{map_id}]: expected {args.rounds_per_map}, got {got}")
    extra_maps = sorted(map_id for map_id in counts if map_id not in expected_maps)
    if extra_maps:
        errors.append(f"unexpected maps: expected only {expected_maps!r}, got extra {extra_maps!r}")
    if unexpected_map_values:
        errors.append(f"episodes with missing/non-integer map_id: {unexpected_map_values!r}")
    if episodes and len(round_names) != args.rounds_per_map:
        errors.append(f"episode round names: expected {args.rounds_per_map}, got {len(round_names)}")

    overall = payload.get("overall")
    if isinstance(overall, dict) and overall.get("episode_count") is not None and int(overall["episode_count"]) != args.episodes:
        errors.append(f"overall.episode_count: expected {args.episodes}, got {overall['episode_count']!r}")

    profile = {
        "maps": expected_maps,
        "rounds_per_map": args.rounds_per_map,
        "episodes": args.episodes,
        "env": expected_env,
        "policy_mode": payload.get("policy_mode"),
        "observed_episode_count": len(episodes),
        "observed_map_counts": {str(key): counts[key] for key in sorted(counts)},
        "observed_round_count": len(rounds),
    }
    return errors, profile


def _run_git(repo_root: Path, args: list[str]) -> str:
    proc = subprocess.run(
        ["git", *args],
        cwd=repo_root,
        text=True,
        capture_output=True,
        check=False,
    )
    if proc.returncode != 0:
        raise IntegrityError(f"git {' '.join(args)} failed: {proc.stderr.strip() or proc.stdout.strip()}")
    return proc.stdout


def _changed_paths(repo_root: Path, git_base: str) -> list[str]:
    changed = set(_run_git(repo_root, ["diff", "--name-only", git_base]).splitlines())
    status = _run_git(repo_root, ["status", "--porcelain", "--untracked-files=all"])
    for line in status.splitlines():
        if not line:
            continue
        path = line[3:]
        if " -> " in path:
            path = path.split(" -> ", 1)[1]
        if path:
            changed.add(path)
    return sorted(changed)


def _diff_by_file(repo_root: Path, git_base: str, path: str) -> list[str]:
    if not (repo_root / path).exists():
        return []
    output = _run_git(repo_root, ["diff", "--", path])
    if output:
        return output.splitlines()
    if path in _run_git(repo_root, ["ls-files", "--others", "--exclude-standard", "--", path]).splitlines():
        try:
            return [f"+{line}" for line in (repo_root / path).read_text(encoding="utf-8", errors="replace").splitlines()]
        except OSError:
            return []
    return _run_git(repo_root, ["diff", git_base, "--", path]).splitlines()


def _target_profile_shape(path: str) -> tuple[int, int, str]:
    if path.endswith("target_3c4r_1000_150_40.json"):
        return OPERATIONAL_PARALLEL_EPISODES, OPERATIONAL_PARALLEL_ROUNDS_PER_MAP, "operational parallel target"
    return DEFAULT_EPISODES, DEFAULT_ROUNDS_PER_MAP, "canonical serial target"


def _check_target_profile(repo_root: Path, path: str) -> list[str]:
    profile_path = repo_root / path
    if not profile_path.exists():
        return [f"{path}: frozen target benchmark profile removed"]
    try:
        profile = json.loads(profile_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return [f"{path}: invalid frozen target benchmark profile JSON: {exc}"]
    if not isinstance(profile, dict):
        return [f"{path}: frozen target benchmark profile must be a JSON object"]

    expected_env = {
        "charger_count": DEFAULT_CHARGER_COUNT,
        "robot_count": DEFAULT_ROBOT_COUNT,
        "max_step": DEFAULT_MAX_STEP,
        "battery_max": DEFAULT_BATTERY_MAX,
    }
    expected_episodes, expected_rounds_per_map, profile_kind = _target_profile_shape(path)
    errors: list[str] = []
    if profile.get("env") != expected_env:
        errors.append(f"{path}: frozen target env drift: expected {expected_env!r}, got {profile.get('env')!r}")
    if profile.get("maps") != list(DEFAULT_MAPS):
        errors.append(f"{path}: frozen target maps drift: expected {list(DEFAULT_MAPS)!r}, got {profile.get('maps')!r}")
    if profile.get("planned_episode_count") != expected_episodes:
        errors.append(
            f"{path}: {profile_kind} episode drift: expected {expected_episodes}, "
            f"got {profile.get('planned_episode_count')!r}"
        )

    rounds = profile.get("rounds")
    if not isinstance(rounds, list) or len(rounds) != expected_rounds_per_map:
        errors.append(f"{path}: {profile_kind} round drift: expected {expected_rounds_per_map} rounds, got {rounds!r}")
        rounds = []
    for index, row in enumerate(rounds, start=1):
        if not isinstance(row, dict):
            errors.append(f"{path}: rounds[{index}] must be an object, got {row!r}")
            continue
        got_env = {key: row.get(key) for key in expected_env}
        if got_env != expected_env:
            errors.append(f"{path}: rounds[{index}] env drift: expected {expected_env!r}, got {got_env!r}")
    return errors


def check_git_changes(repo_root: Path, git_base: str) -> tuple[list[str], list[str]]:
    changed_paths = _changed_paths(repo_root, git_base)
    violations: list[str] = []
    for path in changed_paths:
        normalized = path.replace("\\", "/")
        for pattern, reason in FORBIDDEN_PATH_PATTERNS:
            if pattern.search(normalized):
                violations.append(f"{normalized}: {reason}")
                break
        if TARGET_PROFILE_RE.search(normalized):
            violations.extend(_check_target_profile(repo_root, normalized))
        if normalized in SCORING_GUARD_PATHS:
            for line in _diff_by_file(repo_root, git_base, normalized):
                if not line.startswith(("+", "-")) or line.startswith(("+++", "---")):
                    continue
                for pattern, reason in FORBIDDEN_DIFF_LINE_PATTERNS:
                    if pattern.search(line):
                        violations.append(f"{normalized}: {reason}: {line[:180]}")
                        break
    return violations, changed_paths


def build_report(result_path: Path, args: argparse.Namespace) -> dict[str, Any]:
    result_errors, profile = validate_result(result_path, args)
    git_errors, changed_paths = check_git_changes(REPO_ROOT, args.git_base)
    errors = result_errors + git_errors
    return {
        "status": "integrity_passed" if not errors else "integrity_failed",
        "integrity_passed": not errors,
        "result": str(result_path),
        "git_base": args.git_base,
        "profile": profile,
        "changed_paths_checked": changed_paths,
        "errors": errors,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--git-base", default="HEAD", help="base revision used to inspect changed paths")
    parser.add_argument("--result", type=Path, required=True, help="benchmark result JSON to validate")
    parser.add_argument("--episodes", type=int, default=DEFAULT_EPISODES)
    parser.add_argument("--rounds-per-map", type=int, default=DEFAULT_ROUNDS_PER_MAP)
    parser.add_argument("--maps", type=_parse_maps, default=list(DEFAULT_MAPS))
    parser.add_argument("--charger-count", type=int, default=DEFAULT_CHARGER_COUNT)
    parser.add_argument("--robot-count", type=int, default=DEFAULT_ROBOT_COUNT)
    parser.add_argument("--max-step", type=int, default=DEFAULT_MAX_STEP)
    parser.add_argument("--battery-max", type=int, default=DEFAULT_BATTERY_MAX)
    parser.add_argument(
        "--operational-parallel-40",
        action="store_true",
        help="validate the explicit operational/noncanonical parallel shape: maps 1-10 x4 = 40 episodes",
    )
    parser.add_argument("--output", type=Path, help="optional JSON report output path")
    args = parser.parse_args(argv)

    if args.operational_parallel_40:
        args.episodes = OPERATIONAL_PARALLEL_EPISODES
        args.rounds_per_map = OPERATIONAL_PARALLEL_ROUNDS_PER_MAP

    report = build_report(args.result, args)
    text = json.dumps(report, indent=2, sort_keys=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0 if report["integrity_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
