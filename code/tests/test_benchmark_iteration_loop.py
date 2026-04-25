#!/usr/bin/env python3
"""Tests for the benchmark-900 iteration loop controller."""

from __future__ import annotations

import argparse
import importlib.util
import json
from types import ModuleType
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, cast
import unittest


def _load_loop_module() -> ModuleType:
    script_path = Path(__file__).resolve().parents[2] / "train" / "benchmark_iteration_loop.py"
    spec = importlib.util.spec_from_file_location("benchmark_iteration_loop", script_path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _baseline_fixture() -> dict[str, Any]:
    return {
        "benchmark_metadata": {
            "checkpoint": "/workspace/code/runtime_state/preload_cache/model.ckpt-734599.pkl",
            "manifest": {"timestamp": "20260425-155300"},
            "overall": {"avg_clean_score": 386.4, "broad_win_rate": 0.7},
        },
        "failure_buckets": {"battery depletion": 7, "collision/stuck loop": 33},
        "per_map": [{"map": 1}, {"map": 2}],
    }


def _audit_fixture() -> dict[str, Any]:
    return {
        "opportunity_ranking": [
            {"id": "observe-a", "intervention_class": "P0_observe_only", "title": "Observe A"},
            {"id": "observe-b", "intervention_class": "P1_information_additive", "title": "Observe B"},
            {"id": "eval-c", "intervention_class": "P2_eval_only_safety", "title": "Eval C"},
            {"id": "risk-d", "intervention_class": "R1_small_threshold", "title": "Risk D"},
        ]
    }


class BenchmarkIterationLoopTests(unittest.TestCase):
    def test_dry_run_creates_two_planned_iterations_and_resume_evidence(self):
        module = _load_loop_module()
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            baseline_path = root / "baseline.json"
            audit_path = root / "audit.json"
            state_dir = root / "state"
            evidence_dir = root / "evidence"
            baseline_path.write_text(json.dumps(_baseline_fixture()), encoding="utf-8")
            audit_path.write_text(json.dumps(_audit_fixture()), encoding="utf-8")
            setattr(module, "STATE_DIR", state_dir)
            setattr(module, "EVIDENCE_DIR", evidence_dir)

            run_controller = cast(Any, getattr(module, "run_controller"))
            first = run_controller(
                argparse.Namespace(
                    baseline_summary=baseline_path,
                    dry_run=True,
                    max_iterations=2,
                    resume=False,
                    state_dir=state_dir,
                    stop_on_success=False,
                    wave0_audit=audit_path,
                )
            )
            second = run_controller(
                argparse.Namespace(
                    baseline_summary=baseline_path,
                    dry_run=True,
                    max_iterations=0,
                    resume=True,
                    state_dir=state_dir,
                    stop_on_success=False,
                    wave0_audit=audit_path,
                )
            )

            dry_run_payload = cast(
                dict[str, Any],
                json.loads((evidence_dir / "task-5-loop-dry-run.json").read_text(encoding="utf-8")),
            )
            resume_text = (evidence_dir / "task-5-loop-resume.txt").read_text(encoding="utf-8")
            iteration_evidence_exists = (evidence_dir / "iteration-0001.json").exists()

        self.assertEqual(first["iteration_count"], 2)
        self.assertEqual(second["iteration_count"], 2)
        self.assertEqual(dry_run_payload["iteration_count"], 2)
        self.assertTrue(iteration_evidence_exists)
        self.assertIn("next pending: iteration-0001", resume_text)
        self.assertEqual(dry_run_payload["iterations"][0]["state"], "planned")
        self.assertIn("--profile dev", dry_run_payload["iterations"][0]["benchmark_commands"]["dev_slice"])
        self.assertIn("--runner parallel", dry_run_payload["iterations"][0]["benchmark_commands"]["full"])

    def test_risk_classes_require_prior_p_classes_terminal(self):
        module = _load_loop_module()
        state = {"candidate_statuses": [{"id": "observe-a", "status": "passed"}], "iterations": []}
        candidate = {"modification_class": "R1_small_threshold"}
        validate_modification_order = cast(Any, getattr(module, "validate_modification_order"))

        with self.assertRaisesRegex(ValueError, "blocked until all P0/P1/P2"):
            validate_modification_order(candidate, state, _audit_fixture())

        terminal_state = {
            "candidate_statuses": [
                {"id": "observe-a", "status": "passed"},
                {"id": "observe-b", "status": "failed"},
                {"id": "eval-c", "status": "not_applicable"},
            ],
            "iterations": [],
        }
        validate_modification_order(candidate, terminal_state, _audit_fixture())

    def test_numeric_only_guard_requires_one_bucket_and_one_fixed_observation_diff(self):
        module = _load_loop_module()
        validate_numeric_guard = cast(Any, getattr(module, "validate_numeric_guard"))
        with self.assertRaisesRegex(ValueError, "numeric-only"):
            validate_numeric_guard(
                {
                    "numeric_only": True,
                    "failure_buckets": ["battery depletion", "collision/stuck loop"],
                    "fixed_observation_diffs": ["reward_diff"],
                }
            )

        validate_numeric_guard(
            {
                "numeric_only": True,
                "failure_buckets": ["battery depletion"],
                "fixed_observation_diffs": ["reward_diff"],
            }
        )

    def test_sweep_guard_allows_one_lever_with_two_values(self):
        module = _load_loop_module()
        validate_sweep_guard = cast(Any, getattr(module, "validate_sweep_guard"))
        with self.assertRaisesRegex(ValueError, "no more than two"):
            validate_sweep_guard({"sweep": {"reward_scale": [0.1, 0.2, 0.3]}})
        with self.assertRaisesRegex(ValueError, "exactly one lever"):
            validate_sweep_guard({"sweep": {"a": [1], "b": [2]}})

        validate_sweep_guard({"sweep": {"reward_scale": [0.1, 0.2]}})


if __name__ == "__main__":
    unittest.main()
