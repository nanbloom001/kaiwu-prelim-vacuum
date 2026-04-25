#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

import importlib.util
import json
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


tool = _load_module("summarize_benchmark_failures", TOOLS_DIR / "summarize_benchmark_failures.py")


class SummarizeBenchmarkFailureTests(unittest.TestCase):
    def test_summary_collection_selects_latest_benchmark_and_marks_unavailable_jsonl(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            result_path = Path(tmpdir) / "eval_results.json"
            result_path.write_text(
                json.dumps(
                    {
                        "benchmarks": [
                            {"timestamp": "old", "episodes": []},
                            {
                                "timestamp": "new",
                                "checkpoint": "ckpt.pkl",
                                "rounds": {"round_1": "4 chargers / 3 robots / 1000 steps / 200 battery"},
                                "episodes": [
                                    {
                                        "round": "round_1",
                                        "map_id": 1,
                                        "result": "battery",
                                        "clean_score": 100,
                                        "steps": 200,
                                        "charge_count": 0,
                                        "remaining_charge": 0,
                                    }
                                ],
                            },
                        ]
                    }
                ),
                encoding="utf-8",
            )
            summary = tool.summarize_result(result_path)

        self.assertEqual(summary["benchmark_metadata"]["timestamp"], "new")
        self.assertEqual(summary["episode_count"], 1)
        episode = summary["episodes"][0]
        self.assertEqual(episode["failure_bucket"], "battery depletion")
        self.assertTrue(episode["zero_charge_battery_fail"])
        self.assertEqual(episode["battery_min"], tool.UNAVAILABLE)
        self.assertIn("battery depletion", summary["failure_buckets"])
        self.assertIn("per_map", summary)
        self.assertEqual(summary["per_map"], summary["per_map_table"])
        self.assertEqual(
            summary["per_map"],
            [
                {
                    "map": 1,
                    "episode_count": 1,
                    "avg_clean_score": 100.0,
                    "avg_clean_per_step": 0.5,
                    "completed_count": 0,
                    "failure_buckets": {"battery depletion": 1},
                }
            ],
        )

    def test_episode_jsonl_enriches_stuck_and_battery_metrics(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            result_path = root / "result.json"
            episode_dir = root / "episodes"
            episode_dir.mkdir()
            result_path.write_text(
                json.dumps(
                    {
                        "timestamp": "session",
                        "rounds": {"round_1": "4 chargers / 3 robots / 1000 steps / 200 battery"},
                        "episodes": [
                            {
                                "round": "round_1",
                                "map_id": 2,
                                "result": "completed",
                                "clean_score": 20,
                                "steps": 100,
                                "charge_count": 1,
                                "remaining_charge": 5,
                                "step_log": "episodes/round_1_map2.jsonl",
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            (episode_dir / "round_1_map2.jsonl").write_text(
                "\n".join(
                    json.dumps(row)
                    for row in [
                        {"step": 1, "battery": 50, "zero_progress_streak": 0, "position_repeat_16": 0.0},
                        {"step": 2, "battery": 4, "zero_progress_streak": 11, "position_repeat_16": 0.75, "nearest_charger_dist": 1},
                    ]
                ),
                encoding="utf-8",
            )
            summary = tool.summarize_result(result_path, episode_dir=episode_dir)

        episode = summary["episodes"][0]
        self.assertEqual(episode["battery_min"], 4)
        self.assertEqual(episode["time_on_charger"], 1)
        self.assertEqual(episode["failure_bucket"], "collision/stuck loop")
        self.assertTrue(summary["next_recommended_levers"])
        self.assertEqual(summary["per_map"], summary["per_map_table"])
        self.assertEqual(summary["per_map"][0]["map"], 2)
        self.assertEqual(summary["per_map"][0]["episode_count"], 1)
        self.assertEqual(summary["per_map"][0]["failure_buckets"], {"collision/stuck loop": 1})


if __name__ == "__main__":
    unittest.main()
