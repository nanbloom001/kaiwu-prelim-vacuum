#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

import json
import sys
import tempfile
import time
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agent_ppo.eval.benchmark_parallel import (
    _claim_next_task,
    _complete_task,
    _count_json_files,
    _ensure_runtime_layout,
    _read_json,
    build_parallel_tasks,
    determine_effective_slot_count,
    recover_stale_claims,
)


def _write_json(path: Path, payload):
    path.write_text(json.dumps(payload), encoding="utf-8")


class BenchmarkParallelTests(unittest.TestCase):
    def test_build_parallel_tasks_matches_round_map_cartesian_product(self):
        tasks = build_parallel_tasks()
        self.assertEqual(len(tasks), 40)
        self.assertEqual(tasks[0]["task_id"], "0001-round_1-map1")
        self.assertEqual(tasks[-1]["task_id"], "0040-round_4-map10")

    def test_determine_effective_slot_count_clamps_to_available_handles(self):
        envs = [object(), object()]
        agents = [object()]
        self.assertEqual(determine_effective_slot_count(4, envs, agents), 1)
        self.assertEqual(determine_effective_slot_count(2, envs, envs), 2)

    def test_claim_and_complete_task_moves_file_to_completed(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            runtime_dir = Path(tmpdir)
            _ensure_runtime_layout(runtime_dir)

            task = {
                "task_id": "0001-round_1-map1",
                "idx": 1,
                "total": 40,
                "round_name": "round_1",
                "map_id": 1,
                "round_def": {"name": "round_1"},
                "requeue_count": 0,
            }
            _write_json(runtime_dir / "tasks" / "pending" / "0001-round_1-map1.json", task)

            claimed = _claim_next_task(runtime_dir, "1", "aisrv-1-slot-1")
            self.assertIsNotNone(claimed)
            self.assertFalse((runtime_dir / "tasks" / "pending" / "0001-round_1-map1.json").exists())

            _complete_task(
                runtime_dir,
                "1",
                claimed,
                {"round": "round_1", "map_id": 1, "result": "completed", "clean_score": 800, "steps": 100},
            )

            self.assertEqual(_count_json_files(runtime_dir / "tasks" / "completed"), 1)
            completed = _read_json(runtime_dir / "tasks" / "completed" / "0001-round_1-map1.json")
            self.assertEqual(completed["episode_result"]["result"], "completed")

    def test_recover_stale_claims_requeues_when_worker_heartbeat_expires(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            runtime_dir = Path(tmpdir)
            _ensure_runtime_layout(runtime_dir)

            task = {
                "task_id": "0002-round_1-map2",
                "idx": 2,
                "total": 40,
                "round_name": "round_1",
                "map_id": 2,
                "round_def": {"name": "round_1"},
                "requeue_count": 0,
            }
            claimed_dir = runtime_dir / "tasks" / "claimed" / "2"
            claimed_dir.mkdir(parents=True, exist_ok=True)
            _write_json(claimed_dir / "0002-round_1-map2.json", task)
            _write_json(
                runtime_dir / "workers" / "2.json",
                {"worker_id": "2", "updated_at": time.time() - 120},
            )

            requeued = recover_stale_claims(runtime_dir, worker_timeout_seconds=30)
            self.assertEqual(requeued, ["0002-round_1-map2"])
            pending = _read_json(runtime_dir / "tasks" / "pending" / "0002-round_1-map2.json")
            self.assertEqual(pending["requeue_count"], 1)


if __name__ == "__main__":
    unittest.main()
