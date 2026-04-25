#!/usr/bin/env python3
"""Validate frozen benchmark-900 dry-run manifests.

The validator intentionally checks only wrapper/profile metadata. It does not
import benchmark runtime code and does not inspect scores, simulator behavior, or
aggregation outputs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any


def _load_json(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise SystemExit(f"manifest not found: {path}") from None
    except json.JSONDecodeError as exc:
        raise SystemExit(f"invalid JSON in {path}: {exc}") from None
    if not isinstance(data, dict):
        raise SystemExit(f"manifest must be a JSON object: {path}")
    return data


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _repo_root(start: Path) -> Path:
    current = start.resolve()
    if current.is_file():
        current = current.parent
    for candidate in (current, *current.parents):
        if (candidate / ".git").exists():
            return candidate
    return Path.cwd()


def _parse_maps(value: str) -> list[int]:
    try:
        maps = [int(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"maps must be comma-separated integers: {value}") from exc
    if not maps:
        raise argparse.ArgumentTypeError("maps must not be empty")
    return maps


def _expect(errors: list[str], label: str, got: Any, expected: Any) -> None:
    if got != expected:
        errors.append(f"{label}: expected {expected!r}, got {got!r}")


def _round_env(round_def: dict[str, Any]) -> dict[str, Any]:
    return {
        "charger_count": round_def.get("charger_count"),
        "robot_count": round_def.get("robot_count"),
        "max_step": round_def.get("max_step"),
        "battery_max": round_def.get("battery_max"),
    }


def validate_manifest(manifest_path: Path, args: argparse.Namespace) -> list[str]:
    manifest = _load_json(manifest_path)
    errors: list[str] = []

    expected_env = {
        "charger_count": args.charger_count,
        "robot_count": args.robot_count,
        "max_step": args.max_step,
        "battery_max": args.battery_max,
    }
    expected_maps = args.maps

    _expect(errors, "maps", manifest.get("maps"), expected_maps)
    _expect(errors, "planned_episode_count", manifest.get("planned_episode_count"), args.episodes)
    _expect(errors, "env", manifest.get("env"), expected_env)

    rounds = manifest.get("rounds")
    if not isinstance(rounds, list) or not rounds:
        errors.append(f"rounds: expected non-empty list, got {rounds!r}")
        rounds = []

    _expect(errors, "round_count", len(rounds), args.rounds_per_map)
    for index, round_def in enumerate(rounds, start=1):
        if not isinstance(round_def, dict):
            errors.append(f"rounds[{index}]: expected object, got {round_def!r}")
            continue
        _expect(errors, f"rounds[{index}].env", _round_env(round_def), expected_env)

    planned = manifest.get("planned_episodes")
    if not isinstance(planned, list):
        errors.append(f"planned_episodes: expected list, got {planned!r}")
        planned = []
    _expect(errors, "len(planned_episodes)", len(planned), args.episodes)

    expected_total = len(expected_maps) * args.rounds_per_map
    _expect(errors, "len(maps) * rounds_per_map", expected_total, args.episodes)

    map_counts: Counter[int] = Counter()
    round_names: set[str] = set()
    for index, episode in enumerate(planned, start=1):
        if not isinstance(episode, dict):
            errors.append(f"planned_episodes[{index}]: expected object, got {episode!r}")
            continue
        map_id = episode.get("map")
        if isinstance(map_id, int):
            map_counts[map_id] += 1
        else:
            errors.append(f"planned_episodes[{index}].map: expected int, got {map_id!r}")
        round_name = episode.get("round_name")
        if isinstance(round_name, str):
            round_names.add(round_name)
        episode_env = {
            "charger_count": episode.get("charger_count"),
            "robot_count": episode.get("robot_count"),
            "max_step": episode.get("max_step"),
            "battery_max": episode.get("battery_max"),
        }
        _expect(errors, f"planned_episodes[{index}].env", episode_env, expected_env)

    for map_id in expected_maps:
        _expect(errors, f"rounds_per_map[{map_id}]", map_counts.get(map_id, 0), args.rounds_per_map)
    unexpected_maps = sorted(map_id for map_id in map_counts if map_id not in expected_maps)
    if unexpected_maps:
        errors.append(f"unexpected maps: expected only {expected_maps!r}, got extra {unexpected_maps!r}")
    _expect(errors, "planned round names", len(round_names), args.rounds_per_map)

    profile_path_value = manifest.get("profile_path")
    if profile_path_value:
        repo_root = _repo_root(manifest_path)
        profile_path = repo_root / str(profile_path_value)
        if not profile_path.is_file():
            errors.append(f"profile_path: expected existing file, got {str(profile_path)!r}")
        else:
            profile = _load_json(profile_path)
            _expect(errors, "profile.maps", profile.get("maps"), expected_maps)
            _expect(errors, "profile.env", profile.get("env"), expected_env)
            _expect(errors, "profile.planned_episode_count", profile.get("planned_episode_count"), args.episodes)
            profile_rounds = profile.get("rounds")
            if isinstance(profile_rounds, list):
                _expect(errors, "profile.round_count", len(profile_rounds), args.rounds_per_map)
                for index, round_def in enumerate(profile_rounds, start=1):
                    if isinstance(round_def, dict):
                        _expect(errors, f"profile.rounds[{index}].env", _round_env(round_def), expected_env)
            else:
                errors.append(f"profile.rounds: expected list, got {profile_rounds!r}")
            expected_hash = manifest.get("profile_sha256")
            if expected_hash is not None:
                _expect(errors, "profile_sha256", expected_hash, _sha256_file(profile_path))

    return errors


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--episodes", type=int, required=True)
    parser.add_argument("--rounds-per-map", type=int, required=True)
    parser.add_argument("--maps", type=_parse_maps, required=True)
    parser.add_argument("--charger-count", type=int, required=True)
    parser.add_argument("--robot-count", type=int, required=True)
    parser.add_argument("--max-step", type=int, required=True)
    parser.add_argument("--battery-max", type=int, required=True)
    args = parser.parse_args(argv)

    errors = validate_manifest(args.manifest, args)
    if errors:
        print("target benchmark manifest validation failed:", file=sys.stderr)
        for error in errors:
            print(f"- {error}", file=sys.stderr)
        return 1
    print(f"target benchmark manifest validation passed: {args.manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
