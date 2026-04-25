#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

import argparse
import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

REPO_ROOT = Path(__file__).resolve().parents[2]
TOOLS_DIR = REPO_ROOT / "train" / "tools"


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {name} from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


tool = _load_module("check_benchmark_integrity", TOOLS_DIR / "check_benchmark_integrity.py")


def _args(**overrides):
    values = {
        "maps": list(range(1, 11)),
        "episodes": 30,
        "rounds_per_map": 3,
        "charger_count": 3,
        "robot_count": 4,
        "max_step": 1000,
        "battery_max": 150,
        "git_base": "HEAD",
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def _result(*, battery_max=150, episodes=30, rounds_per_map=3):
    rounds = {
        f"target_round_{index}": f"3 chargers / 4 robots / 1000 steps / {battery_max} battery"
        for index in range(1, rounds_per_map + 1)
    }
    rows = []
    episode_id = 1
    for map_id in range(1, 11):
        for round_index in range(1, rounds_per_map + 1):
            if len(rows) >= episodes:
                break
            rows.append(
                {
                    "episode_id": episode_id,
                    "map_id": map_id,
                    "round": f"target_round_{round_index}",
                    "clean_score": 100.0 + map_id,
                    "result": "win",
                }
            )
            episode_id += 1
    return {
        "schema_version": 4,
        "timestamp": "fixture",
        "policy_mode": "eval",
        "rounds": rounds,
        "episodes": rows,
        "overall": {"episode_count": len(rows), "avg_clean_score": 105.5},
    }


class CheckBenchmarkIntegrityTests(unittest.TestCase):
    def test_valid_canonical_serial_30_result_passes_with_allowed_changes(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "result.json"
            path.write_text(json.dumps(_result()), encoding="utf-8")

            with mock.patch.object(tool, "check_git_changes", return_value=([], ["README.md"])):
                errors, profile = tool.validate_result(path, _args())

        self.assertEqual(errors, [])
        self.assertEqual(profile["observed_episode_count"], 30)
        self.assertEqual(profile["observed_map_counts"], {str(i): 3 for i in range(1, 11)})

    def test_operational_parallel_40_result_requires_explicit_shape(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "result.json"
            path.write_text(json.dumps(_result(episodes=40, rounds_per_map=4)), encoding="utf-8")

            default_errors, _default_profile = tool.validate_result(path, _args())
            explicit_errors, explicit_profile = tool.validate_result(
                path,
                _args(episodes=40, rounds_per_map=4),
            )

        self.assertTrue(any("episode_count: expected 30" in error for error in default_errors), default_errors)
        self.assertEqual(explicit_errors, [])
        self.assertEqual(explicit_profile["observed_episode_count"], 40)
        self.assertEqual(explicit_profile["observed_map_counts"], {str(i): 4 for i in range(1, 11)})

    def test_battery_drift_fails_with_expected_reason(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "result.json"
            path.write_text(json.dumps(_result(battery_max=200)), encoding="utf-8")

            errors, _profile = tool.validate_result(path, _args())

        self.assertTrue(any("battery_max" in error and "200" in error for error in errors), errors)

    def test_forbidden_simulator_path_is_reported(self):
        with mock.patch.object(tool, "_changed_paths", return_value=["code/gamecore/maps/map_1.json"]):
            violations, changed = tool.check_git_changes(REPO_ROOT, "HEAD")

        self.assertEqual(changed, ["code/gamecore/maps/map_1.json"])
        self.assertTrue(any("simulator/gamecore/map/scoring path changed" in item for item in violations), violations)

    def test_forbidden_scoring_drift_is_reported(self):
        with mock.patch.object(tool, "_changed_paths", return_value=["code/agent_ppo/eval/benchmark.py"]), mock.patch.object(
            tool,
            "_diff_by_file",
            return_value=["+    overall['avg_clean_score'] = inflated_score"],
        ):
            violations, _changed = tool.check_git_changes(REPO_ROOT, "HEAD")

        self.assertTrue(any("avg_clean_score" in item for item in violations), violations)

    def test_cli_exits_nonzero_for_battery_drift_fixture(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "result.json"
            path.write_text(json.dumps(_result(battery_max=200)), encoding="utf-8")
            proc = subprocess.run(
                [sys.executable, str(TOOLS_DIR / "check_benchmark_integrity.py"), "--git-base", "HEAD", "--result", str(path)],
                text=True,
                capture_output=True,
                check=False,
            )

        self.assertNotEqual(proc.returncode, 0)
        self.assertIn("integrity_failed", proc.stdout)
        self.assertIn("battery_max", proc.stdout)


if __name__ == "__main__":
    unittest.main()
