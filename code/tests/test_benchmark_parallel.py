#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# pyright: reportImplicitRelativeImport=false

import json
import os
import sys
import tempfile
import time
import types
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agent_ppo.eval.benchmark_parallel import (
    _claim_next_task,
    _complete_task,
    _count_json_files,
    _ensure_runtime_layout,
    _finalize_parallel_benchmark,
    _initialize_session,
    _is_parallel_benchmark_coordinator,
    _logical_worker_id,
    _read_json,
    _resolve_benchmark_aisrv_worker_id,
    build_parallel_tasks,
    determine_effective_slot_count,
    recover_stale_claims,
)


def _write_json(path: Path, payload):
    path.write_text(json.dumps(payload), encoding="utf-8")


class BenchmarkParallelTests(unittest.TestCase):
    def test_operational_parallel_target_profile_is_four_fixed_rounds_on_official_maps(self):
        profile_path = (
            Path(__file__).resolve().parents[2]
            / "train"
            / "benchmark_profiles"
            / "target_3c4r_1000_150_40.json"
        )
        profile = json.loads(profile_path.read_text(encoding="utf-8"))
        expected_env = {
            "charger_count": 3,
            "robot_count": 4,
            "max_step": 1000,
            "battery_max": 150,
        }

        self.assertEqual(profile["profile_name"], "target_3c4r_1000_150_40")
        self.assertEqual(profile["maps"], list(range(1, 11)))
        self.assertEqual(profile["env"], expected_env)
        self.assertEqual(profile["planned_episode_count"], 40)
        self.assertEqual(len(profile["rounds"]), 4)
        for round_def in profile["rounds"]:
            self.assertEqual(
                {
                    "charger_count": round_def["charger_count"],
                    "robot_count": round_def["robot_count"],
                    "max_step": round_def["max_step"],
                    "battery_max": round_def["battery_max"],
                },
                expected_env,
            )

    def test_target_benchmark_wrapper_keeps_serial30_canonical_and_parallel40_explicit(self):
        script_path = Path(__file__).resolve().parents[2] / "train" / "run_target_benchmark_900.sh"
        script_text = script_path.read_text(encoding="utf-8")

        self.assertIn("target_3c4r_1000_150_30.json", script_text)
        self.assertIn("target-parallel", script_text)
        self.assertIn("target_3c4r_1000_150_40.json", script_text)
        self.assertIn("operational/noncanonical", script_text)
        self.assertRegex(script_text, r"WORKERS=\"?4\"?")
        self.assertRegex(script_text, r"ENVS_PER_WORKER=\"?10\"?")
        self.assertIn("--workers", script_text)
        self.assertIn('"workers": int(os.environ["WORKERS"])', script_text)
        self.assertIn('"envs_per_worker": int(os.environ["ENVS_PER_WORKER"])', script_text)
        self.assertIn('exec bash run_benchmark_parallel.sh "${CHECKPOINT}"', script_text)

    def test_parallel_benchmark_script_defaults_to_4x10_and_40_gamecores(self):
        script_path = Path(__file__).resolve().parents[2] / "train" / "run_benchmark_parallel.sh"
        script_text = script_path.read_text(encoding="utf-8")

        self.assertRegex(script_text, r"(?m)^WORKERS=4$")
        self.assertRegex(script_text, r"(?m)^ENVS_PER_WORKER=10$")
        self.assertIn("GAMECORES=$((WORKERS * ENVS_PER_WORKER))", script_text)
        self.assertIn('export KAIWU_GAMECORE_NUM="${GAMECORES}"', script_text)
        self.assertIn('export KAIWU_PARALLEL_ENV_PER_AISRV="${ENVS_PER_WORKER}"', script_text)
        self.assertIn('assert_container_env "KAIWU_GAMECORE_NUM" "${GAMECORES}"', script_text)
        self.assertIn(
            'assert_container_env "KAIWU_PARALLEL_ENV_PER_AISRV" "${ENVS_PER_WORKER}"',
            script_text,
        )
        self.assertIn(
            'assert_toml_key "aisrv_connect_to_kaiwu_env_count" "${ENVS_PER_WORKER}"',
            script_text,
        )

    def test_compose_patches_start_train_client_before_aisrv_start(self):
        compose_path = Path(__file__).resolve().parents[2] / "train" / ".docker-compose.yaml"
        compose_text = compose_path.read_text(encoding="utf-8")
        anchor = "sh tools/change_alloc_process_count.sh kaiwu_env $${parallel_env_per_aisrv}"
        inserted = (
            "change_config_in_file aisrv_connect_to_kaiwu_env_count "
            "$${parallel_env_per_aisrv} $${kaiwudrl_configure_file} int"
        )
        deployment_anchor = (
            "change_config_in_file deployment_platforms client "
            "$$kaiwudrl_configure_file str"
        )
        startup = "bash -c '/root/tools/start_train_client.sh aisrv"

        self.assertIn(anchor, compose_text)
        self.assertIn(inserted, compose_text)
        self.assertIn(deployment_anchor, compose_text)
        self.assertIn("deployment_pos = text.find(deployment_anchor, inserted_pos)", compose_text)
        self.assertLess(compose_text.index(inserted), compose_text.index(startup))

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

    def test_logical_worker_id_maps_aisrv_and_process_index(self):
        self.assertEqual(_logical_worker_id(1, 0, 10), 1)
        self.assertEqual(_logical_worker_id(1, 9, 10), 10)
        self.assertEqual(_logical_worker_id(2, 0, 10), 11)
        self.assertEqual(_logical_worker_id(4, 9, 10), 40)

    def test_parallel_coordinator_is_only_first_aisrv_first_helper_process(self):
        self.assertTrue(_is_parallel_benchmark_coordinator(1, 0))
        self.assertFalse(_is_parallel_benchmark_coordinator(1, 1))
        self.assertFalse(_is_parallel_benchmark_coordinator(2, 0))

    def test_resolve_aisrv_worker_id_preserves_env_index_fallback(self):
        with mock.patch.dict(os.environ, {"KAIWU_AISRV_INDEX": "3"}, clear=False):
            self.assertEqual(_resolve_benchmark_aisrv_worker_id(4), 3)

        with mock.patch.dict(os.environ, {"KAIWU_AISRV_INDEX": ""}, clear=False):
            self.assertGreaterEqual(_resolve_benchmark_aisrv_worker_id(4), 1)

    def test_initialize_session_records_parallel_identity_accounting(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            runtime_dir = Path(tmpdir)
            benchmark_api = types.SimpleNamespace(
                SCHEMA_VERSION=3,
                _benchmark_policy_mode=lambda: "eval",
                _get_git_commit=lambda: "test-commit",
                _configured_rounds=lambda: [{"name": "round_1", "desc": "test"}],
                _configured_maps=lambda: [1],
            )
            with mock.patch("agent_ppo.eval.benchmark_parallel._benchmark_api", return_value=benchmark_api):
                _initialize_session(
                    runtime_dir=runtime_dir,
                    session_id="session-test",
                    checkpoint="checkpoint.pkl",
                    aisrv_worker_count=4,
                    configured_envs_per_aisrv=10,
                    logical_worker_count=40,
                    logical_worker_id=1,
                    aisrv_worker_id=1,
                    process_index=0,
                    effective_envs_per_aisrv=10,
                    effective_slots_per_process=1,
                    visible_env_handles_per_process=1,
                    visible_agent_handles_per_process=1,
                    scheduler="dynamic",
                    base_env_conf={},
                )

            execution = _read_json(runtime_dir / "manifest.json")["execution"]
            self.assertEqual(execution["worker_count"], 4)
            self.assertEqual(execution["aisrv_worker_count"], 4)
            self.assertEqual(execution["configured_envs_per_aisrv"], 10)
            self.assertEqual(execution["logical_worker_count"], 40)
            self.assertEqual(execution["logical_worker_id"], 1)
            self.assertEqual(execution["aisrv_worker_id"], 1)
            self.assertEqual(execution["process_index"], 0)
            self.assertEqual(execution["effective_envs_per_worker"], 10)
            self.assertEqual(execution["effective_envs_per_aisrv"], 10)
            self.assertEqual(execution["effective_env_processes_per_worker"], 10)
            self.assertEqual(execution["available_env_handles"], 1)
            self.assertEqual(execution["visible_env_handles_per_process"], 1)
            self.assertEqual(execution["effective_slots_per_process"], 1)

    def test_finalize_session_records_parallel_identity_accounting(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            runtime_dir = Path(tmpdir)
            _ensure_runtime_layout(runtime_dir)
            _write_json(
                runtime_dir / "manifest.json",
                {
                    "created_at": time.time(),
                    "policy_mode": "eval",
                    "git_commit": "test-commit",
                    "rounds": [{"name": "round_1", "desc": "test"}],
                    "execution": {
                        "available_env_handles": 1,
                        "available_agent_handles": 1,
                        "visible_env_handles_per_process": 1,
                        "visible_agent_handles_per_process": 1,
                    },
                },
            )
            _write_json(
                runtime_dir / "tasks" / "completed" / "0001-round_1-map1.json",
                {
                    "episode_result": {
                        "round": "round_1",
                        "map_id": 1,
                        "result": "completed",
                        "clean_score": 800,
                        "steps": 100,
                    }
                },
            )
            benchmark_api = types.SimpleNamespace(
                SCHEMA_VERSION=3,
                ROUNDS=[{"name": "round_1", "desc": "test"}],
                _benchmark_policy_mode=lambda: "eval",
                _get_git_commit=lambda: "test-commit",
                _aggregate_results=lambda episodes: {
                    "per_round": {},
                    "overall": {
                        "win_rate": 1.0,
                        "avg_clean_score": 800,
                        "win_episode_count": 1,
                        "episode_count": len(episodes),
                    },
                },
                _build_ai_summary=lambda snapshot: {"overall": snapshot["overall"]},
                _save_results=lambda path, snapshot: _write_json(Path(path), snapshot),
            )

            with mock.patch("agent_ppo.eval.benchmark_parallel._benchmark_api", return_value=benchmark_api):
                _finalize_parallel_benchmark(
                    runtime_dir=runtime_dir,
                    session_id="session-test",
                    checkpoint="checkpoint.pkl",
                    aisrv_worker_count=4,
                    configured_envs_per_aisrv=10,
                    logical_worker_count=40,
                    logical_worker_id=1,
                    aisrv_worker_id=1,
                    process_index=0,
                    effective_envs_per_aisrv=10,
                    effective_slots_per_process=1,
                    scheduler="dynamic",
                    total_episodes=1,
                    results_file=runtime_dir / "results.json",
                    logger=types.SimpleNamespace(info=lambda *args, **kwargs: None),
                )

            execution = _read_json(runtime_dir / "result.json")["execution"]
            self.assertEqual(execution["effective_envs_per_worker"], 10)
            self.assertEqual(execution["effective_envs_per_aisrv"], 10)
            self.assertEqual(execution["effective_env_processes_per_worker"], 10)
            self.assertEqual(execution["effective_slots_per_process"], 1)
            self.assertEqual(execution["available_env_handles"], 1)
            self.assertEqual(execution["visible_env_handles_per_process"], 1)

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
