#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
TOOLS_DIR = REPO_ROOT / "train" / "tools"
sys.path.insert(0, str(TOOLS_DIR))


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {name} from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


tool = _load_module("compare_serial_parallel_target", TOOLS_DIR / "compare_serial_parallel_target.py")


def _result(avg_1=100.0, avg_2=200.0, *, rounds=None, episode_count=10):
    return {
        "timestamp": "fixture",
        "checkpoint": "model.pkl",
        "policy_mode": "eval",
        "rounds": rounds
        or {
            "target_round_1": "3 chargers / 4 robots / 1000 steps / 150 battery",
            "target_round_2": "3 chargers / 4 robots / 1000 steps / 150 battery",
        },
        "per_round": {
            "target_round_1": {
                "avg_clean_score": avg_1,
                "episode_count": episode_count,
                "win_episode_count": 5,
            },
            "target_round_2": {
                "avg_clean_score": avg_2,
                "episode_count": episode_count,
                "win_episode_count": 6,
            },
        },
        "overall": {"avg_clean_score": (avg_1 + avg_2) / 2.0},
    }


class CompareSerialParallelTargetTests(unittest.TestCase):
    def test_avg_delta_within_tolerance_exits_zero(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            serial = root / "serial.json"
            parallel = root / "parallel.json"
            serial.write_text(json.dumps(_result(100.0, 200.0)), encoding="utf-8")
            parallel.write_text(json.dumps(_result(110.0, 205.0)), encoding="utf-8")

            report = tool.compare_results(serial, parallel, max_avg_delta=25)

        self.assertEqual(report["status"], "accepted")
        self.assertEqual(report["canonical_runner_decision"], "serial-only canonical")
        self.assertEqual(report["parallel_equivalence_decision"], "parallel operationally equivalent")
        self.assertIn("serial-only canonical remains the success authority", report["message"])
        self.assertTrue(report["profile_compatible"])
        self.assertAlmostEqual(report["average_delta"], 7.5)

    def test_avg_delta_above_tolerance_rejects_parallel(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            serial = root / "serial.json"
            parallel = root / "parallel.json"
            serial.write_text(json.dumps(_result(100.0, 200.0)), encoding="utf-8")
            parallel.write_text(json.dumps(_result(150.0, 260.0)), encoding="utf-8")

            proc = subprocess.run(
                [
                    sys.executable,
                    str(TOOLS_DIR / "compare_serial_parallel_target.py"),
                    str(serial),
                    str(parallel),
                    "--max-avg-delta",
                    "25",
                ],
                text=True,
                capture_output=True,
                check=False,
            )

        self.assertNotEqual(proc.returncode, 0)
        self.assertIn("parallel not canonical", proc.stdout + proc.stderr)

    def test_profile_or_shape_mismatch_rejects_parallel(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            serial = root / "serial.json"
            parallel = root / "parallel.json"
            serial.write_text(json.dumps(_result(100.0, 200.0)), encoding="utf-8")
            parallel.write_text(
                json.dumps(
                    _result(
                        100.0,
                        200.0,
                        rounds={
                            "target_round_1": "3 chargers / 4 robots / 1000 steps / 150 battery",
                            "target_round_2": "4 chargers / 4 robots / 1000 steps / 200 battery",
                        },
                        episode_count=20,
                    )
                ),
                encoding="utf-8",
            )

            report = tool.compare_results(serial, parallel, max_avg_delta=25)

        self.assertEqual(report["status"], "rejected")
        self.assertFalse(report["profile_compatible"])
        self.assertFalse(report["compatibility_checks"]["same_round_profile"])
        self.assertFalse(report["compatibility_checks"]["same_episode_count"])
        self.assertIn("parallel not canonical", report["message"])

    def test_missing_serial_result_blocks_canonical_parallel(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            serial = root / "missing-serial.json"
            parallel = root / "parallel.json"
            parallel.write_text(json.dumps(_result(100.0, 200.0)), encoding="utf-8")

            report = tool.compare_results(serial, parallel, max_avg_delta=25)

        self.assertEqual(report["status"], "blocked_serial_comparison_unavailable")
        self.assertEqual(report["canonical_runner_decision"], "serial-only canonical")
        self.assertIn("serial comparison unavailable", report["message"])


if __name__ == "__main__":
    unittest.main()
